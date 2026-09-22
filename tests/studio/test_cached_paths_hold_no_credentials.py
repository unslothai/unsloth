# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""No job may persist a directory that it has also configured a credential to live in.

On 2026-09-21 the Rust project disclosed that `cargo miri` wrote the ENTIRE process
environment to a file under `target/` (`cargo-miri/src/util.rs:40-42`), that CI cached
`target/` with `actions/cache` or `Swatinem/rust-cache`, and that GitHub lets
`pull_request` runs restore caches written on the default branch. A secret that had only
ever existed in a privileged run therefore became readable by anyone who could open a
pull request, and the exposure was invisible in the workflow YAML, because the leak
happened inside a tool nobody had reason to distrust.

The class needs four things at once: a secret in the job's environment, something that
serialises it to disk, that path being persisted by a cache save or an artifact upload,
and the result being readable from a pull request. This repository satisfies one, three
and four in 24 jobs. What has kept it safe is the second link, and only the second link.

That is worth stating precisely, because it is the part a future commit can undo without
looking dangerous. There is no `miri` here. Secrets are step-scoped everywhere: a scan
for a workflow- or job-level `env:` whose value is a `${{ secrets.* }}` expression
returns nothing across all 57 workflows and 9 composite actions, no secret is ever
written to `$GITHUB_ENV`, there is no `set -x` anywhere, and the only two `env` /
`printenv` uses discard values through `cut -d= -f1`. Every upload names a narrow path
rather than a tree.

The one structural hole left is the shape this module pins. Eight workflows point
`HF_HOME` at `${{ github.workspace }}/hf-cache` and then cache `hf-cache` itself
(`studio-api-smoke.yml:114,154`, `studio-ui-smoke.yml:182,239`,
`studio-mac-ui-smoke.yml:152,199`, `studio-windows-api-smoke.yml:103,129`,
`studio-windows-ui-smoke.yml:113,152`, `studio-windows-inference-smoke.yml:411,1522`,
`studio-inference-smoke.yml:269`, `local-agent-guides-ci.yml:783,823`), under a
cross-OS key that carries no ref and is saved from main. `huggingface_hub` persists
credentials INSIDE `HF_HOME` -- `$HF_HOME/token`, and `stored_tokens` in newer versions.
So a single added `hf auth login` in one of those jobs would write a real token into a
cache every pull request can restore, and nothing would flag it: the token is withheld on
pull requests, which protects the PR run's own environment and does nothing about a value
baked into main's cache. Cache scoping means the leak direction is main-writes /
PR-reads, which is the opposite of the direction the rest of the tooling models.

Nothing in the existing machinery can see this. `scripts/lint_workflow_triggers.py`
compares cache keys as byte-identical strings and only models PR-poisons-publish; it
never parses `path:`, and it globs only `.github/workflows`, so the keys that live in
`.github/actions/*/action.yml` are outside it. `tests/studio/test_cache_budget_discipline.py`
reasons only about which ref a save lands on. No test related a cached path to the
secrets a job holds, which is why this file exists.

One measured result recorded here because it is load-bearing elsewhere and expires.
`release-desktop.yml` was the only job in the organisation where a secret-bearing run
wrote a cache pull requests can read. Cargo serialises the VALUE of any
`cargo:rerun-if-env-changed` variable into `target/<profile>/.fingerprint/*.json`, and
rust-cache preserves dependency artifacts, so the question was whether any crate in the
Tauri graph names a signing variable. On 2026-09-22 all 673 locked dependencies were
audited from source: 671 crates.io tarballs, plus `unsloth-studio` itself (a three-line
`build.rs` calling `tauri_build::build()`) and `fix-path-env`, a git dependency with no
`build.rs` at all. Twenty build scripts declare `rerun-if-env-changed`, naming 27
distinct variables, none of them a secret this CI holds. The two runtime-constructed
cases both resolve safely: `ring`'s wrapper takes a `&'static str` and its only literals
are `CARGO_MANIFEST_DIR` and `OUT_DIR`, and `aws-lc-rs` emits its interpolated form
through `eprintln!`, i.e. on stderr, where cargo never reads it as a directive. So the
finding was a risky configuration rather than a live leak, no rotation was warranted, and
`release-desktop.yml` is now `save-if: false` anyway -- because that clean result is a
property of 673 third-party build scripts, not of this repository, and every `cargo
update` decides it again.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
ACTIONS = REPO / ".github" / "actions"

# Environment variables that designate a directory a tool will write credentials into.
# The value is the credential-bearing file or subdirectory, for the message only; the
# test cares that the variable's directory is inside something persisted.
CREDENTIAL_HOMES = {
    # huggingface_hub writes $HF_HOME/token and, since 0.25, $HF_HOME/stored_tokens.
    "HF_HOME": "token, stored_tokens",
    # And HF_TOKEN_PATH names the token file directly, overriding the location above.
    "HF_TOKEN_PATH": "the token file itself",
    # npm and cargo both keep registry credentials in their config roots.
    "NPM_CONFIG_USERCONFIG": ".npmrc auth tokens",
    "CARGO_HOME": "credentials.toml",
    "DOCKER_CONFIG": "config.json auth entries",
    "AWS_SHARED_CREDENTIALS_FILE": "aws credentials",
    "GOOGLE_APPLICATION_CREDENTIALS": "service account json",
}

# HUGGINGFACE_HUB_CACHE and TRANSFORMERS_CACHE are deliberately NOT here. Both select a
# MODEL cache, not a credential home: verified against huggingface_hub 1.32.0, where
# HUGGINGFACE_HUB_CACHE defaults to `$HF_HOME/hub` and the token is read from
# HF_TOKEN_PATH, default `$HF_HOME/token`, computed independently of it. So pointing
# HUGGINGFACE_HUB_CACHE at a directory and caching that directory persists blobs and no
# token. Listing them rejected the arrangement this module recommends everywhere else,
# which is the kind of false failure that gets a security guard switched off.

# Where those same tools keep credentials when nothing overrides them. Caching one of
# these is the identical hazard reached WITHOUT setting any variable, which is the version
# that reads as harmless: there is no HF_HOME in the file to notice.
# unsloth-zoo's gemma4-audio-probe.yml cached ~/.cache/huggingface until 2026-09-22 for
# exactly that reason, and the explicit-variable check below would have passed it.
DEFAULT_CREDENTIAL_HOMES = {
    "~/.cache/huggingface": "HF_HOME default; token, stored_tokens",
    "~/.huggingface": "legacy HF_HOME default; token",
    "~/.cargo": "CARGO_HOME default; credentials.toml",
    "~/.docker": "DOCKER_CONFIG default; config.json auth entries",
    "~/.npmrc": "npm auth tokens",
    "~/.aws": "aws credentials",
    "~/.config/gh": "gh CLI oauth token",
}

# Anything that makes a tool persist a credential into one of the directories above.
# Matched against the shell body of every step, so a login added anywhere in a job that
# caches its own credential home fails the guard.
# Which variable overrides each default. A default is only where a tool looks when
# nothing points it elsewhere, so a job that sets the variable writes its credentials
# there instead and persisting the default location holds none. Flagging it anyway was a
# false failure on a correct configuration.
DEFAULT_OWNERS = {
    "~/.cache/huggingface": "HF_HOME",
    "~/.huggingface": "HF_HOME",
    "~/.cargo": "CARGO_HOME",
    "~/.docker": "DOCKER_CONFIG",
    "~/.npmrc": "NPM_CONFIG_USERCONFIG",
    "~/.aws": "AWS_SHARED_CREDENTIALS_FILE",
    "~/.config/gh": None,
}

# Which credential homes each shell login actually writes into. Matching every pattern
# against every variable reported a job that caches `HF_HOME` and runs `docker login` as
# leaking the Hugging Face token, which it plainly does not: docker writes
# $DOCKER_CONFIG/config.json. The same false failure applied to npm, cargo, aws and gcloud.
#
# `None` means "any credential home", used for the Hugging Face patterns because those
# write to whichever of HF_HOME or HF_TOKEN_PATH is in force.
LOGIN_PATTERN_HOMES = {
    r"\bnpm\s+login\b": ("NPM_CONFIG_USERCONFIG",),
    r"\bcargo\s+login\b": ("CARGO_HOME",),
    r"\bdocker\s+login\b": ("DOCKER_CONFIG",),
    r"\bgcloud\s+auth\s+(?:application-default\s+)?login\b": (
        "GOOGLE_APPLICATION_CREDENTIALS",
    ),
    r"\baws\s+configure\b": ("AWS_SHARED_CREDENTIALS_FILE",),
}

LOGIN_PATTERNS = (
    r"\bhf\s+auth\s+login\b",
    r"\bhuggingface-cli\s+login\b",
    r"\bhf\s+login\b",
    r"huggingface_hub[.\s]*\.?\s*login\s*\(",
    r"\bfrom\s+huggingface_hub\s+import\s+[^\n]*\blogin\b",
    r"\bHfFolder\b[^\n]*\bsave_token\b",
    r"\bsave_token\s*\(",
    r"add_to_git_credential\s*=\s*True",
    r"\bnpm\s+login\b",
    r"\bcargo\s+login\b",
    r"\bdocker\s+login\b",
    r"\bgcloud\s+auth\s+(?:application-default\s+)?login\b",
    r"\baws\s+configure\b",
)

_CACHE_SAVE = ("actions/cache/save", "actions/cache@")
# Actions that write a credential into a tool's credential home. Same hazard as a `run:`
# login, with no shell body for a pattern to match.
#
# Each maps to the credential-home variables it ACTUALLY writes, and the pairing is the
# whole point. Keying only on the action fired on eight live workflows where
# `actions/setup-node` sat in a job whose cached directory was named by HF_HOME, a
# variable setup-node has nothing to do with. A guard that reports a Hugging Face leak
# because a job installs Node teaches people to switch it off.
#
# `condition` names a `with:` input that must be present before the action writes
# anything: setup-node only creates an .npmrc token when `registry-url` is set, and
# otherwise just installs a runtime.
LOGIN_ACTIONS = {
    "docker/login-action": {
        "vars": ("DOCKER_CONFIG",),
        "why": "registry auth into $DOCKER_CONFIG/config.json",
    },
    "aws-actions/configure-aws-credentials": {
        "vars": ("AWS_SHARED_CREDENTIALS_FILE",),
        "why": "aws credentials",
    },
    "google-github-actions/auth": {
        "vars": ("GOOGLE_APPLICATION_CREDENTIALS",),
        "why": "an application default credentials file",
    },
    "actions/setup-node": {
        "vars": ("NPM_CONFIG_USERCONFIG",),
        "why": "an .npmrc auth token",
        "condition": "registry-url",
    },
}

_PERSIST = _CACHE_SAVE + ("actions/upload-artifact",)


def _on(doc):
    """The `on:` mapping, which PyYAML parses as the boolean True."""
    return doc.get(True) if True in doc else doc.get("on")


def _docs():
    """Every workflow and composite action, parsed, with its path."""
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        try:
            doc = yaml.safe_load(path.read_text(encoding = "utf-8"))
        except yaml.YAMLError:
            continue
        if isinstance(doc, dict):
            yield path, doc
    for path in sorted(ACTIONS.rglob("action.y*ml")):
        try:
            doc = yaml.safe_load(path.read_text(encoding = "utf-8"))
        except yaml.YAMLError:
            continue
        if isinstance(doc, dict):
            yield path, doc


def _jobs(doc):
    jobs = doc.get("jobs")
    if isinstance(jobs, dict):
        for jid, job in jobs.items():
            if isinstance(job, dict):
                yield jid, job
    # A composite action has one implicit job: its `runs.steps`.
    runs = doc.get("runs")
    if isinstance(runs, dict) and isinstance(runs.get("steps"), list):
        yield "runs", {"steps": runs["steps"]}


def _steps(job):
    steps = job.get("steps")
    return [s for s in steps if isinstance(s, dict)] if isinstance(steps, list) else []


def _env_of(job, doc):
    """Env visible to the job: workflow-level overlaid with job-level."""
    env = {}
    for source in (doc.get("env"), job.get("env")):
        if isinstance(source, dict):
            env.update({str(k): str(v) for k, v in source.items()})
    return env


def _step_envs(job):
    """Every step-level `env:` block in the job, one dict each.

    Setting the credential home on the step that runs the tool is the natural way to
    write it, and reading only workflow- and job-level `env:` missed that spelling
    entirely: a job whose cached directory was named by the very step writing into it
    passed this guard. Each is merged over the job's env, which is the precedence
    Actions applies.
    """
    out = []
    for step in _steps(job):
        source = step.get("env")
        if isinstance(source, dict):
            out.append({str(k): str(v) for k, v in source.items()})
    return out


def _expand(value: str, env: dict, inputs: dict | None = None) -> str:
    """Substitute `${{ env.X }}`, and `${{ inputs.X }}`, before comparing paths.

    `_normalise` deletes expressions wholesale, so `path: ${{ env.HF_HOME }}` reduced to
    the empty string and `_inside` refuses an empty operand. A job that cached exactly
    its own credential home, spelled through the variable rather than repeated
    literally, was therefore silently exempt from the rule aimed at it.

    `inputs` matters for the same reason one level down. A composite whose cache step
    says `path: ${{ inputs.path }}` names its persisted directory through an input, so
    without the caller's `with:` block the path resolved to empty and the composite
    appeared to persist nothing at all.
    """
    value = re.sub(
        r"\$\{\{\s*env\.([A-Za-z_]\w*)\s*\}\}",
        lambda m: env.get(m.group(1), ""),
        value,
    )
    if inputs:
        value = re.sub(
            r"\$\{\{\s*inputs\.([A-Za-z_][\w-]*)\s*\}\}",
            lambda m: str(inputs.get(m.group(1), "")),
            value,
        )
    return value

def _persisted_paths(job):
    """Every path this job writes to a cache or an artifact."""
    out = []
    for step in _steps(job):
        uses = str(step.get("uses") or "").casefold()
        if not any(marker.casefold() in uses for marker in _PERSIST):
            continue
        with_ = step.get("with")
        if not isinstance(with_, dict):
            continue
        raw = with_.get("path")
        if raw is None:
            continue
        for line in str(raw).splitlines():
            line = line.strip()
            if line and not line.startswith("!"):
                out.append((line, step))
    return out




def _reusable_jobs(job, env = None, inputs = None):
    """(steps, env, inputs) for a job that delegates to a local reusable workflow.

    `jobs.<id>.uses: ./.github/workflows/x.yml` has no `steps:` of its own, so the caller
    scanned as an empty job and the called workflow was scanned separately with no access
    to the caller's `with:` values. A reusable job that sets `HF_HOME: ${{ inputs.path }}`,
    logs in and caches `${{ inputs.path }}` was therefore accepted when the caller passed
    `path: hf-cache`: both expressions normalised away and neither document held enough to
    see the combination. The same split that hid composite logins, one level up.
    """
    ref = str(job.get("uses") or "").strip().strip("'\"")
    if not ref.startswith("./"):
        return []
    target = REPO / ref[2:]
    if not target.is_file():
        return []
    try:
        doc = yaml.safe_load(target.read_text(encoding = "utf-8"))
    except yaml.YAMLError:
        return []
    if not isinstance(doc, dict):
        return []
    passed = {}
    on = doc.get(True) if True in doc else doc.get("on")
    call = on.get("workflow_call") if isinstance(on, dict) else None
    declared = call.get("inputs") if isinstance(call, dict) else None
    if isinstance(declared, dict):
        for name, spec in declared.items():
            if isinstance(spec, dict) and spec.get("default") is not None:
                passed[str(name)] = str(spec["default"])
    with_ = job.get("with")
    if isinstance(with_, dict):
        # Resolved against the CALLER's env and inputs, exactly as a composite's
        # forwarded `with:` is. Copying verbatim meant reusable workflow A handing
        # `path: ${{ inputs.path }}` to B gave B a self-referential value: the path and
        # the credential home both normalised away and the login plus cache was accepted
        # even though the outer caller supplied a concrete directory.
        passed.update({
            str(k): _expand(str(v), env or {}, inputs or {}) for k, v in with_.items()
        })
    out = []
    for _jid, inner in _jobs(doc):
        out.append((inner, _env_of(inner, doc), passed))
    return out

def _flat_steps(job, inherited = None, inputs = None, stack = None):
    """(step, inherited env, caller inputs) for this job and every local composite it uses.

    A composite's steps execute INSIDE the calling job, and the invoking step's `env:`
    applies while they run, so that environment has to travel with them. Appending the
    inner step dictionaries alone discarded it: a composite invoked with
    `env: {HF_HOME: hf-cache}` whose inner step logs in showed no credential home at all,
    and the login was accepted.

    The caller's `with:` block travels too, because a composite may name its own cached
    directory through an input (`path: ${{ inputs.path }}`), which resolves to nothing
    without it.

    `stack` is the recursion path, NOT a set of everything already visited. A shared
    visited set suppressed every invocation after the first, and each invocation runs
    with its own environment: a login composite called once with `HF_HOME` outside the
    cache and once with it inside was only ever inspected in its safe form. Verified
    before fixing -- that job produced no offender at all. Cycle detection still needs
    the path, so an action that (transitively) uses itself is not expanded forever.
    """
    inherited = {} if inherited is None else inherited
    inputs = {} if inputs is None else inputs
    stack = () if stack is None else stack
    out = []
    # A job may delegate wholesale to a local reusable workflow instead of listing steps.
    for inner, inner_env, passed in _reusable_jobs(job, inherited, inputs):
        out.extend(_flat_steps(inner, {**inherited, **inner_env}, passed, stack))
    for step in _steps(job):
        own = step.get("env")
        env = {**inherited, **({str(k): str(v) for k, v in own.items()}
                               if isinstance(own, dict) else {})}
        out.append((step, inherited, inputs))
        uses = str(step.get("uses") or "").strip().strip("'\"")
        if not uses.startswith("./"):
            continue
        base = REPO / uses[2:]
        for cand in (base, base / "action.yml", base / "action.yaml"):
            if not cand.is_file():
                continue
            if cand in stack:
                break                      # a cycle, not a sibling invocation
            try:
                doc = yaml.safe_load(cand.read_text(encoding = "utf-8"))
            except yaml.YAMLError:
                break
            if not isinstance(doc, dict):
                break
            runs = doc.get("runs")
            if isinstance(runs, dict) and isinstance(runs.get("steps"), list):
                with_ = step.get("with")
                passed = {}
                # A declared default is what Actions applies when the caller omits the
                # input, so a composite invoked bare still names its real path.
                declared = doc.get("inputs")
                if isinstance(declared, dict):
                    for field, spec in declared.items():
                        if isinstance(spec, dict) and spec.get("default") is not None:
                            passed[str(field)] = str(spec["default"])
                if isinstance(with_, dict):
                    for field, value in with_.items():
                        # Resolved against the CALLER's env and inputs, so a forwarded
                        # `${{ inputs.path }}` arrives as the value it stands for rather
                        # than as the same expression one level down.
                        passed[str(field)] = _expand(str(value), env, inputs)
                out.extend(_flat_steps(
                    {"steps": runs["steps"]}, env, passed, stack + (cand,),
                ))
            break
    return out


def _local_action_steps(job):
    """The composite-provided steps only, for tests that assert flattening happened."""
    own = {id(s) for s in _steps(job)}
    return [step for step, _env, _in in _flat_steps(job) if id(step) not in own]


def _persisted_with_env(job, doc):
    """(path, env) for every path this job persists, with the env of the DECLARING step.

    Actions resolves a `path: ${{ env.CACHE_DIR }}` against the environment of the step
    that performs the save, so expanding it with some other step's environment answers a
    different question. Carrying the pair keeps the two apart, and it also brings in
    persistence that happens inside a local composite, which the caller-only scan could
    not see at all: a composite saving the job's credential home while the workflow logs
    in was a combination neither half observed.
    """
    job_env = _env_of(job, doc)
    out = []
    # The job's own env seeds the walk. Without it a root-level call passing
    # `path: ${{ env.CACHE_DIR }}` resolved against inherited and step-level values only,
    # `_expand` produced an empty string, and the composite appeared to persist nothing.
    for step, inherited, inputs in _flat_steps(job, job_env):
        uses = str(step.get("uses") or "").casefold()
        if not any(marker.casefold() in uses for marker in _PERSIST):
            continue
        with_ = step.get("with")
        if not isinstance(with_, dict) or with_.get("path") is None:
            continue
        own = step.get("env")
        env = {**job_env, **inherited, **({str(k): str(v) for k, v in own.items()}
                                         if isinstance(own, dict) else {})}
        for line in str(with_["path"]).splitlines():
            line = line.strip()
            if line and not line.startswith("!"):
                out.append((_expand(line, env, inputs), step))
    return out


def _login_offenders(doc, job):
    """Steps that log in while THEIR OWN credential home sits inside a persisted path.

    Evaluated per step rather than per job, because the job-wide version rejected a safe
    arrangement: with one step setting a cached `HF_HOME` and a later step setting a
    different, uncached `HF_HOME` and logging in, combining every home in the job with
    every login in the job flagged the second step even though its token cannot reach the
    cache. The effective environment of the step that actually runs the login is what
    decides, which is also the only thing the runtime cares about.

    A login can be an ACTION as well as a shell command, and skipping every step without
    a `run:` body missed that class entirely: `docker/login-action` writes registry
    credentials into `$DOCKER_CONFIG/config.json` with no shell for a pattern to match.
    An action only counts against the variables it really writes, per LOGIN_ACTIONS.
    """
    job_env = _env_of(job, doc)
    persisted = [path for path, _step in _persisted_with_env(job, doc)]
    if not persisted:
        return []
    offenders = []
    for step, inherited, inputs in _flat_steps(job, job_env):
        body = str(step.get("run") or "")
        uses = str(step.get("uses") or "").strip().strip("'\"")
        action = uses.split("@")[0]
        # GitHub treats owner/repo case-insensitively, so `Docker/login-action` runs the
        # same credential-writing action as `docker/login-action`. A case-sensitive
        # lookup let a capital letter skip the step entirely, since it has no `run:` body.
        spec = LOGIN_ACTIONS.get(action.casefold())
        if spec is not None:
            with_ = step.get("with")
            needed = spec.get("condition")
            if needed and not (isinstance(with_, dict) and with_.get(needed) is not None):
                spec = None
        if not body and spec is None:
            continue
        own = step.get("env")
        env = {**job_env, **inherited, **({str(k): str(v) for k, v in own.items()}
                                         if isinstance(own, dict) else {})}
        homes = {v: _expand(str(env[v]), env, inputs) for v in CREDENTIAL_HOMES if v in env}
        if not homes:
            continue
        for var, home in sorted(homes.items()):
            hit = next((p for p in persisted if _inside(home, p)), None)
            if hit is None:
                continue
            if spec is not None and var in spec["vars"]:
                offenders.append(
                    f"{step.get('name') or uses}: {var}={home} is inside cached "
                    f"{hit!r}, and this step uses {action} ({spec['why']})"
                )
                break
            matched = next(
                (
                    p
                    for p in LOGIN_PATTERNS
                    if body
                    and re.search(p, body, re.IGNORECASE)
                    and var in (LOGIN_PATTERN_HOMES.get(p) or (var,))
                ),
                None,
            )
            if matched is not None:
                offenders.append(
                    f"{step.get('name') or uses or 'run'}: {var}={home} is inside "
                    f"cached {hit!r}, and this step matches /{matched}/"
                )
                break
    return offenders

def _raw_persisted(job, doc = None):
    """Expanded persisted paths, including those declared inside local composites."""
    return [path for path, _step in _persisted_with_env(job, doc or {})]

def _normalise(path: str) -> str:
    """Strip expressions and separators so a path and an env value can be compared."""
    path = re.sub(r"\$\{\{[^}]*\}\}", "", path)
    path = path.replace("\\", "/").strip().strip("'\"")
    path = re.sub(r"^\$(HOME|\{HOME\})/", "~/", path)
    while path.startswith("./"):
        path = path[2:]
    return path.strip("/")



def _deglob(path: str) -> str:
    """The fixed directory a glob pattern lives under.

    `hf-cache/**` and `hf-cache/*.bin` both persist things inside `hf-cache`, so for a
    containment question the fixed leading part is what matters. Anything from the first
    wildcard segment onward is dropped.
    """
    parts = []
    for segment in path.replace("\\", "/").split("/"):
        if any(ch in segment for ch in "*?["):
            break
        parts.append(segment)
    return "/".join(parts) if parts else path

def _inside(inner: str, outer: str) -> bool:
    """Is `inner` the same directory as `outer`, or below it?

    A glob suffix is trimmed off `outer` first. `path: hf-cache/**` uploads everything
    beneath `hf-cache`, including a token written there, but compared literally
    `_inside("hf-cache", "hf-cache/**")` is false and the whole upload looked unrelated to
    the credential home it contains.
    """
    inner, outer = _normalise(inner), _normalise(_deglob(outer))
    inner, outer = inner.strip("/"), outer.strip("/")
    if not inner or not outer:
        return False
    return inner == outer or inner.startswith(outer + "/")


def _offending_jobs():
    """(label, var, path, home_value) for every job caching its own credential home."""
    for path, doc in _docs():
        for jid, job in _jobs(doc):
            job_env = _env_of(job, doc)
            for scope in [job_env] + _step_envs(job):
                env = {**job_env, **scope}
                homes = {v: env[v] for v in CREDENTIAL_HOMES if v in env}
                if not homes:
                    continue
                for persisted, _step in _persisted_with_env(job, doc):
                    for var, home in homes.items():
                        if _inside(_expand(home, env), persisted):
                            yield f"{path.name}:{jid}", var, persisted, home


def test_the_scan_finds_the_jobs_it_claims_to():
    """A scan that matched nothing would pass every check below on an empty set."""
    found = {label for label, _, _, _ in _offending_jobs()}
    assert len(found) >= 5, (
        f"only found {len(found)} jobs that persist their own credential home; the scan is "
        f"wrong. This is expected to be non-empty: pointing HF_HOME at a cached directory "
        f"is deliberate here, and the point of this module is that a login must never be "
        f"added to one of those jobs, not that the arrangement is forbidden."
    )
    names = {label.split(":")[0] for label in found}
    for expected in ("studio-api-smoke.yml", "studio-windows-ui-smoke.yml"):
        assert expected in names, f"{expected} caches its HF_HOME but the scan missed it"


def test_the_inside_predicate_reads_the_path():
    """The guard is only as good as this predicate, so the predicate is tested too."""
    cases = [
        # (home, persisted, inside)
        ("hf-cache", "hf-cache", True),
        ("${{ github.workspace }}/hf-cache", "hf-cache", True),
        ("hf-cache/hub", "hf-cache", True),  # credential home below the cached dir
        ("./hf-cache", "hf-cache", True),
        ("hf-cache-vision", "hf-cache", False),  # prefix, not a child
        ("other", "hf-cache", False),
        ("hf-cache", "hf-cache/hub", False),  # cached dir below the home, not covered
        ("", "hf-cache", False),
        ("hf-cache", "", False),
        (r"${{ github.workspace }}\hf-cache", "hf-cache", True),  # Windows separators
    ]
    for home, persisted, expected in cases:
        assert _inside(home, persisted) is expected, f"_inside({home!r}, {persisted!r})"


@pytest.mark.parametrize(
    "label,var,persisted,home",
    sorted(_offending_jobs()),
    ids = lambda v: str(v).replace("/", "_") if isinstance(v, str) else str(v),
)
def test_a_job_that_caches_its_credential_home_performs_no_login(label, var, persisted, home):
    name, jid = label.split(":", 1)
    path = WORKFLOWS / name
    if not path.exists():
        candidates = list(ACTIONS.rglob(name))
        path = candidates[0] if candidates else path
    doc = yaml.safe_load(path.read_text(encoding = "utf-8"))
    job = dict(_jobs(doc)).get(jid) or {}

    # Per step, with that step's own environment, and including the steps of any local
    # composite this job uses. See _login_offenders for why the job-wide version was both
    # too broad (a later step with an uncached home) and too narrow (a login inside an
    # action).
    offenders = _login_offenders(doc, job)

    assert not offenders, (
        f"{label} sets {var}={home}, which is inside the persisted path {persisted!r}, and a "
        f"step in it logs in:\n  " + "\n  ".join(offenders) + "\n\n"
        f"{var} is where the tool keeps its credentials ({CREDENTIAL_HOMES[var]}), so a login "
        f"writes a real token into that directory, and the directory is then saved to a cache "
        f"or uploaded as an artifact. GitHub lets every pull request restore caches written on "
        f"the default branch, so the token would be readable by anyone who can open one. "
        f"Withholding the secret on pull requests does not help: it protects the PR run's own "
        f"environment, not a value already baked into main's cache.\n\n"
        f"Read the token from the environment instead of logging in, which is what this repo "
        f"does today ({var} is set for the download path and no step authenticates), or point "
        f"{var} somewhere outside the persisted path."
    )


def test_expanding_an_env_reference_finds_the_path_the_expression_names():
    """`path: ${{ env.HF_HOME }}` has to resolve, or the rule cannot see its own case.

    `_normalise` deletes expressions, so an unexpanded reference reduces to the empty
    string and `_inside` refuses an empty operand. A job caching exactly its own
    credential home, spelled through the variable instead of repeated literally, was
    therefore exempt from the rule written for it.
    """
    env = {"HF_HOME": "${{ github.workspace }}/hf-cache"}
    assert _expand("${{ env.HF_HOME }}", env) == "${{ github.workspace }}/hf-cache"
    assert _inside(_expand("${{ env.HF_HOME }}", env), "hf-cache") is True
    # An undefined name expands to empty, which is what Actions itself does.
    assert _expand("${{ env.NOT_SET }}", env) == ""
    # Nothing else is touched, so `github.workspace` is still normalised away later.
    assert _expand("hf-cache", env) == "hf-cache"


def test_a_subdirectory_of_a_credential_home_is_not_a_finding():
    """Caching `~/.cache/huggingface/hub` is the RECOMMENDED arrangement, not a hazard.

    huggingface_hub keeps the token at `~/.cache/huggingface/token`, a SIBLING of `hub`,
    so a cache of the `hub` subdirectory holds the model blobs and no credential. An
    earlier version of the check compared containment in both directions and rejected
    it, which is the kind of false failure that gets a security guard switched off.
    """
    assert _inside("~/.cache/huggingface/hub", "~/.cache/huggingface") is True
    assert _inside("~/.cache/huggingface", "~/.cache/huggingface/hub") is False
    home = "~/.cache/huggingface"
    flagged = [p for p in ("~/.cache/huggingface/hub", "~/.cache", home) if _inside(home, p)]
    assert flagged == ["~/.cache", home], (
        "only a persisted path CONTAINING the credential home is a finding; the hub "
        f"subdirectory must not be, and the set flagged was {flagged}"
    )


def test_a_credential_home_set_on_a_step_is_seen():
    """Step-level `env:` is where a credential home is most naturally written.

    Reading only workflow- and job-level `env:` missed it, so a job whose cached
    directory was named by the very step writing into it passed this guard.
    """
    job = {"steps": [
        {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
        {"run": "python probe.py", "env": {"HF_HOME": "hf-cache"}},
    ]}
    scopes = _step_envs(job)
    assert {"HF_HOME": "hf-cache"} in scopes, f"step env not collected: {scopes}"
    assert _env_of(job, {}) == {}, "the job itself sets nothing, which is the point"
    assert any(
        _inside(scope["HF_HOME"], persisted)
        for scope in scopes if "HF_HOME" in scope
        for persisted, _step in _persisted_paths(job)
    ), "the step-scoped credential home is inside the cached path and must be a finding"


def test_no_job_persists_a_default_credential_home():
    """The spelling that needs no variable set, and so reads as harmless.

    A job can reach the same hazard by caching the location a tool uses when nothing
    overrides it. There is then no `HF_HOME` in the workflow to notice, which is why the
    check above cannot see it. unsloth-zoo's gemma4-audio-probe.yml cached
    `~/.cache/huggingface` until 2026-09-22 and was invisible to the sibling guard for
    precisely this reason; it now points HF_HOME at a workspace directory and caches that,
    the way the model caches in this repository do.

    Nothing here does it today. The test exists so the next model cache is written the
    same way.

    Only the credential home being INSIDE the persisted path counts. Comparing both
    directions was wrong and rejected the recommended arrangement: caching
    `~/.cache/huggingface/hub` persists the model blobs while the token stays a SIBLING
    at `~/.cache/huggingface/token`, outside the cache entirely.
    """
    offenders = []
    for path, doc in _docs():
        for jid, job in _jobs(doc):
            env = _env_of(job, doc)
            overridden = {v for v in CREDENTIAL_HOMES if v in env}
            for persisted, _step in _persisted_with_env(job, doc):
                for default, creds in DEFAULT_CREDENTIAL_HOMES.items():
                    # A default is only where the tool looks when nothing overrides it.
                    # A job setting `CARGO_HOME: /tmp/cargo` writes credentials there, so
                    # persisting `~/.cargo` holds none, and flagging it was a false
                    # failure on a correct configuration.
                    if DEFAULT_OWNERS.get(default) in overridden:
                        continue
                    if _inside(default, persisted):
                        offenders.append(
                            f"{path.name}:{jid}: caches {persisted!r}, a default "
                            f"credential home ({creds})"
                        )
    assert not offenders, (
        "these jobs cache or upload a tool's default credential home:\n  "
        + "\n  ".join(sorted(set(offenders)))
        + "\n\n"
        "Anything that logs in writes a token there, and the directory is then saved to a "
        "cache every pull request can restore. Point the tool at a directory the workflow "
        "owns and cache that instead, as the smoke workflows here do with "
        "`HF_HOME: ${{ github.workspace }}/hf-cache` and `path: hf-cache`."
    )


def test_model_cache_variables_are_not_treated_as_credential_homes():
    """HUGGINGFACE_HUB_CACHE and TRANSFORMERS_CACHE select a MODEL cache, not a token.

    Listing them rejected the arrangement this module recommends everywhere else: point a
    variable at a directory you own and cache that. Checked against the installed
    huggingface_hub rather than asserted from memory, because the whole claim is about
    what that library does with these names.
    """
    for var in ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        assert var not in CREDENTIAL_HOMES, (
            f"{var} names a model cache, not a credential home. The token is read from "
            f"HF_TOKEN_PATH (default $HF_HOME/token), computed independently of it, so "
            f"caching a directory {var} points at persists blobs and no token."
        )
    assert "HF_TOKEN_PATH" in CREDENTIAL_HOMES, (
        "HF_TOKEN_PATH names the token file directly and overrides HF_HOME, so it is the "
        "variable that actually has to be tracked"
    )

    hub = pytest.importorskip("huggingface_hub.constants")
    home = str(getattr(hub, "HF_HOME", ""))
    token = str(getattr(hub, "HF_TOKEN_PATH", ""))
    cache = str(getattr(hub, "HUGGINGFACE_HUB_CACHE", ""))
    assert token.startswith(home) and token.endswith("token"), (
        f"expected the token under HF_HOME; got HF_HOME={home!r} HF_TOKEN_PATH={token!r}"
    )
    assert not _inside(token, cache), (
        f"the token is supposed to sit OUTSIDE the hub cache, which is the reason "
        f"HUGGINGFACE_HUB_CACHE is not a credential home; got {token!r} inside {cache!r}"
    )


def test_a_login_is_judged_against_that_step_s_own_environment():
    """The job-wide version was both too broad and, for the safe case, simply wrong.

    Combining every credential home in a job with every login in the same job flagged a
    step that sets its own uncached `HF_HOME` and logs in there, even though its token
    cannot reach the cache. The effective environment of the step running the login is
    what decides, which is also all the runtime cares about.
    """
    safe = {"steps": [
        {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
        {"name": "warm the cache", "run": "python download.py", "env": {"HF_HOME": "hf-cache"}},
        {"name": "log in elsewhere", "run": "hf auth login --token x",
         "env": {"HF_HOME": "/tmp/scratch-home"}},
    ]}
    assert _login_offenders({}, safe) == [], (
        "the login step points HF_HOME at an uncached directory, so its token cannot "
        "enter the cache and it must not be a finding"
    )

    # And the same shape with the login step's own home inside the cache must fire.
    unsafe = {"steps": [
        {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
        {"name": "log in", "run": "hf auth login --token x", "env": {"HF_HOME": "hf-cache"}},
    ]}
    offenders = _login_offenders({}, unsafe)
    assert offenders, "a login whose own HF_HOME is inside the cached path must be a finding"
    assert "hf-cache" in offenders[0] and "log in" in offenders[0], offenders

    # A job-level home with a login anywhere in the job is still caught, which is the
    # original rule and must not have been lost in making the above per-step.
    job_level = {
        "env": {"HF_HOME": "hf-cache"},
        "steps": [
            {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
            {"name": "log in", "run": "huggingface-cli login"},
        ],
    }
    assert _login_offenders({}, job_level), (
        "a job-level credential home inside the cached path, with a login in any step, "
        "is the case this module was written for and must still fire"
    )


def test_a_login_inside_a_local_composite_action_is_seen(tmp_path, monkeypatch):
    """A composite's run steps execute in the calling job, so its logins are the job's.

    Scanning only the workflow's own `run` bodies let a job persist its credential home,
    delegate the login to `uses: ./.github/actions/whatever`, and pass: the workflow
    contains no login, and the action is scanned separately where neither the caller's
    environment nor its persisted paths are visible. Neither half sees the combination.
    """
    import sys

    module = sys.modules[__name__]
    action = tmp_path / ".github" / "actions" / "hf-login"
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: hf login\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - shell: bash\n"
        "      run: hf auth login --token \"$HF_TOKEN\"\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    job = {
        "env": {"HF_HOME": "hf-cache"},
        "steps": [
            {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
            {"uses": "./.github/actions/hf-login"},
        ],
    }
    assert _local_action_steps(job), "the composite's steps were not flattened into the job"
    offenders = _login_offenders({}, job)
    assert offenders, (
        "the login happens inside the composite, and it writes into the cached HF_HOME "
        "just the same, so it must be a finding"
    )
    assert "hf auth login" in offenders[0] or "login" in offenders[0], offenders


def test_a_composite_invoked_with_a_credential_home_carries_that_env(tmp_path, monkeypatch):
    """The invoking step's `env:` applies while the composite runs, so it must travel.

    Appending the inner step dictionaries alone discarded it: a composite invoked with
    `env: {HF_HOME: hf-cache}` whose inner step logs in showed no credential home at all,
    neither job-level nor inner-step, and the login was accepted even though the job saves
    `hf-cache`.
    """
    import sys

    module = sys.modules[__name__]
    action = tmp_path / ".github" / "actions" / "hf-login"
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: hf login\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - shell: bash\n"
        "      run: hf auth login --token \"$HF_TOKEN\"\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    job = {"steps": [
        {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
        {"uses": "./.github/actions/hf-login", "env": {"HF_HOME": "hf-cache"}},
    ]}
    offenders = _login_offenders({}, job)
    assert offenders, (
        "the credential home is set on the step that invokes the composite, and the "
        "composite's login writes into it, so this must be a finding"
    )


def test_persistence_inside_a_local_composite_is_seen(tmp_path, monkeypatch):
    """A composite can hold the `actions/cache/save`, and the caller-only scan saw none.

    `_raw_persisted` examined the workflow's own steps and returned empty, so the login
    scan never ran at all. Scanning the composite separately does not help either,
    because that document has neither the caller's environment nor its login step, so
    neither half observed the combination.
    """
    import sys

    module = sys.modules[__name__]
    action = tmp_path / ".github" / "actions" / "save-cache"
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: save cache\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache/save@v4\n"
        "      with:\n"
        "        path: hf-cache\n"
        "        key: k\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    job = {
        "env": {"HF_HOME": "hf-cache"},
        "steps": [
            {"name": "log in", "run": "hf auth login --token x"},
            {"uses": "./.github/actions/save-cache"},
        ],
    }
    paths = [p for p, _s in _persisted_with_env(job, {})]
    assert "hf-cache" in paths, f"the composite's cache save was not collected: {paths}"
    assert _login_offenders({}, job), (
        "the composite saves the job's credential home and the workflow logs in, which "
        "is the combination this module exists to refuse"
    )


def test_a_cached_path_is_expanded_with_its_own_step_s_environment():
    """Actions resolves `path:` against the environment of the step doing the save.

    Expanding it with some other step's environment answers a different question: with
    `env: {CACHE_DIR: creds}` on the cache step and a login step setting a credential
    home under `creds`, the path resolved to empty against the login step's environment
    and the combination was missed.
    """
    job = {"steps": [
        {"name": "save", "uses": "actions/cache/save@v4",
         "with": {"path": "${{ env.CACHE_DIR }}", "key": "k"},
         "env": {"CACHE_DIR": "creds"}},
        {"name": "log in", "run": "hf auth login --token x", "env": {"HF_HOME": "creds/hf"}},
    ]}
    paths = [p for p, _s in _persisted_with_env(job, {})]
    assert paths == ["creds"], (
        f"the path had to resolve against the saving step's own CACHE_DIR; got {paths}"
    )
    assert _login_offenders({}, job), (
        "HF_HOME is creds/hf, inside the cached creds, and that step logs in"
    )


def test_every_invocation_of_a_composite_is_flattened(tmp_path, monkeypatch):
    """A shared visited set suppressed every invocation after the first.

    Each invocation runs with its own environment, so a login composite called once with
    `HF_HOME` outside the cache and once with it inside was only ever inspected in its
    safe form. Verified before fixing: that job produced no offender at all. Cycle
    detection needs the recursion PATH, not a set of everything already seen.
    """
    import sys

    module = sys.modules[__name__]
    action = tmp_path / ".github" / "actions" / "hf-login"
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: hf login\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - shell: bash\n"
        "      run: hf auth login --token x\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    job = {"steps": [
        {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
        {"uses": "./.github/actions/hf-login", "env": {"HF_HOME": "/tmp/outside"}},
        {"uses": "./.github/actions/hf-login", "env": {"HF_HOME": "hf-cache"}},
    ]}
    offenders = _login_offenders({}, job)
    assert offenders, (
        "the SECOND invocation points HF_HOME inside the cached path, and suppressing it "
        "as already-visited is what hid the dangerous one behind the safe one"
    )
    # Two invocations of the one-step composite, so both inner steps are present.
    assert len(_local_action_steps(job)) == 2, _local_action_steps(job)


def test_a_composite_cached_path_given_by_input_is_resolved(tmp_path, monkeypatch):
    """A composite may name its cached directory through an input.

    `path: ${{ inputs.path }}` resolved to the empty string without the caller's `with:`
    block, and `_inside` refuses an empty operand, so the composite appeared to persist
    nothing and the login beside it was accepted.
    """
    import sys

    module = sys.modules[__name__]
    action = tmp_path / ".github" / "actions" / "save-in"
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: save in\n"
        "inputs:\n"
        "  path:\n"
        "    required: true\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache/save@v4\n"
        "      with:\n"
        "        path: ${{ inputs.path }}\n"
        "        key: k\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    job = {
        "env": {"HF_HOME": "hf-cache"},
        "steps": [
            {"name": "log in", "run": "hf auth login --token x"},
            {"uses": "./.github/actions/save-in", "with": {"path": "hf-cache"}},
        ],
    }
    paths = [p for p, _s in _persisted_with_env(job, {})]
    assert paths == ["hf-cache"], f"the input-backed path did not resolve; got {paths}"
    assert _login_offenders({}, job), (
        "the composite persists the job's credential home and the workflow logs in"
    )


def test_no_job_that_persists_anything_performs_a_login():
    """The whole invariant, swept over every job, independent of label discovery.

    The parametrised test above is instantiated from the credential-home scan, and that
    scan reads each job's DIRECT steps. So a composite whose inner login step declares
    its own `HF_HOME` produced no label, the parametrisation never covered that job, and
    `_login_offenders` -- which would have caught it -- was simply never called. Scanning
    the action on its own does not help either, because that document has neither the
    caller's persisted paths nor its environment.

    This sweep depends on nothing but the job list, so a gap in discovery cannot hide a
    finding again. It is the backstop; the parametrised test stays for its per-job
    reporting.
    """
    offenders = []
    for path, doc in _docs():
        for jid, job in _jobs(doc):
            for item in _login_offenders(doc, job):
                offenders.append(f"{path.name}:{jid}: {item}")
    assert not offenders, (
        "these steps log in while their own credential home sits inside a path the job "
        "persists:\n  " + "\n  ".join(sorted(offenders)) + "\n\n"
        "A login writes a real token into that directory, and the directory is then "
        "saved to a cache or uploaded. GitHub lets every pull request restore caches "
        "written on the default branch, so the token becomes readable by anyone who can "
        "open one. Read the token from the environment instead of logging in, or point "
        "the credential home outside the persisted path."
    )


def test_a_login_performed_by_an_action_is_seen():
    """A login can be an action, with no shell body for a pattern to match.

    `docker/login-action` writes registry credentials into `$DOCKER_CONFIG/config.json`,
    so a job pointing DOCKER_CONFIG at a cached directory leaks exactly as a
    `docker login` command would. Skipping every step without a `run:` missed the class.
    """
    job = {
        "env": {"DOCKER_CONFIG": "docker-cache"},
        "steps": [
            {"uses": "docker/login-action@v3", "with": {"username": "u", "password": "p"}},
            {"uses": "actions/cache/save@v4", "with": {"path": "docker-cache", "key": "k"}},
        ],
    }
    offenders = _login_offenders({}, job)
    assert offenders, "an action-performed login into a cached credential home must fire"
    assert "docker/login-action" in offenders[0], offenders

    safe = {
        "env": {"DOCKER_CONFIG": "/tmp/docker"},
        "steps": [
            {"uses": "docker/login-action@v3"},
            {"uses": "actions/cache/save@v4", "with": {"path": "docker-cache", "key": "k"}},
        ],
    }
    assert _login_offenders({}, safe) == [], "the credential home is outside the cache"


def test_a_login_action_only_counts_against_what_it_writes():
    """The pairing is the point, and getting it wrong is how this rule cries wolf.

    Keying only on the action name fired on eight live workflows where
    `actions/setup-node` sat in a job whose cached directory was named by HF_HOME, a
    variable setup-node has nothing to do with. And setup-node writes an .npmrc token
    only when `registry-url` is set; otherwise it installs a runtime and touches no
    credential at all.
    """
    unrelated = {
        "env": {"HF_HOME": "hf-cache"},
        "steps": [
            {"uses": "actions/setup-node@v4", "with": {"node-version": "20"}},
            {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
        ],
    }
    assert _login_offenders({}, unrelated) == [], (
        "setup-node does not write HF_HOME, so caching an HF_HOME directory beside it "
        "is not a finding"
    )

    no_registry = {
        "env": {"NPM_CONFIG_USERCONFIG": "npm-cache/.npmrc"},
        "steps": [
            {"uses": "actions/setup-node@v4", "with": {"node-version": "20"}},
            {"uses": "actions/cache/save@v4", "with": {"path": "npm-cache", "key": "k"}},
        ],
    }
    assert _login_offenders({}, no_registry) == [], (
        "without `registry-url` setup-node writes no token"
    )

    with_registry = {
        "env": {"NPM_CONFIG_USERCONFIG": "npm-cache/.npmrc"},
        "steps": [
            {"uses": "actions/setup-node@v4",
             "with": {"registry-url": "https://registry.npmjs.org"}},
            {"uses": "actions/cache/save@v4", "with": {"path": "npm-cache", "key": "k"}},
        ],
    }
    assert _login_offenders({}, with_registry), (
        "with `registry-url` it writes an .npmrc token into the cached config path"
    )


def test_an_input_forwarded_between_composites_is_resolved(tmp_path, monkeypatch):
    """One composite handing `${{ inputs.path }}` to another kept the outer expression.

    The inner step then expanded it back to itself and `_normalise` erased it, so a
    wrapper receiving `path: hf-cache` could save that directory while the job logged
    into `HF_HOME=hf-cache` and the guard reported nothing.
    """
    import sys

    module = sys.modules[__name__]
    inner = tmp_path / ".github" / "actions" / "save-in"
    outer = tmp_path / ".github" / "actions" / "wrap"
    inner.mkdir(parents = True)
    outer.mkdir(parents = True)
    (inner / "action.yml").write_text(
        "name: save in\ninputs:\n  path:\n    required: true\nruns:\n"
        "  using: composite\n  steps:\n    - uses: actions/cache/save@v4\n"
        "      with:\n        path: ${{ inputs.path }}\n        key: k\n"
    )
    (outer / "action.yml").write_text(
        "name: wrap\ninputs:\n  path:\n    required: true\nruns:\n"
        "  using: composite\n  steps:\n    - uses: ./.github/actions/save-in\n"
        "      with:\n        path: ${{ inputs.path }}\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    job = {
        "env": {"HF_HOME": "hf-cache"},
        "steps": [
            {"name": "log in", "run": "hf auth login --token x"},
            {"uses": "./.github/actions/wrap", "with": {"path": "hf-cache"}},
        ],
    }
    paths = [p for p, _s in _persisted_with_env(job, {})]
    assert paths == ["hf-cache"], f"the forwarded input did not resolve; got {paths}"
    assert _login_offenders({}, job), "the nested composite saves the credential home"


def test_a_composite_input_default_is_applied(tmp_path, monkeypatch):
    """Actions applies a declared default when the caller omits the `with:` entirely.

    A composite declaring `path` with default `hf-cache` and caching
    `${{ inputs.path }}` was invoked bare, the empty input map left the expression
    unresolved, `_normalise` erased it, and the composite looked as though it persisted
    nothing at all.
    """
    import sys

    module = sys.modules[__name__]
    action = tmp_path / ".github" / "actions" / "save-default"
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: save default\ninputs:\n  path:\n    default: hf-cache\nruns:\n"
        "  using: composite\n  steps:\n    - uses: actions/cache/save@v4\n"
        "      with:\n        path: ${{ inputs.path }}\n        key: k\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    job = {
        "env": {"HF_HOME": "hf-cache"},
        "steps": [
            {"name": "log in", "run": "hf auth login --token x"},
            {"uses": "./.github/actions/save-default"},
        ],
    }
    paths = [p for p, _s in _persisted_with_env(job, {})]
    assert paths == ["hf-cache"], f"the declared default was not applied; got {paths}"
    assert _login_offenders({}, job), (
        "the composite persists the credential home via its default, and the job logs in"
    )


def test_a_glob_path_still_contains_its_directory():
    """`path: hf-cache/**` uploads whatever is beneath hf-cache, token included."""
    assert _deglob("hf-cache/**") == "hf-cache"
    assert _deglob("hf-cache/*.bin") == "hf-cache"
    assert _deglob("hf-cache") == "hf-cache"
    assert _inside("hf-cache", "hf-cache/**") is True
    assert _inside("hf-cache", "hf-cache/*.bin") is True
    assert _inside("other", "hf-cache/**") is False


def test_a_login_action_is_matched_case_insensitively():
    """GitHub treats owner/repo case-insensitively; a capital letter is not an escape."""
    job = {"env": {"DOCKER_CONFIG": "docker-cache"}, "steps": [
        {"uses": "Docker/Login-Action@v3"},
        {"uses": "actions/cache/save@v4", "with": {"path": "docker-cache", "key": "k"}},
    ]}
    assert _login_offenders({}, job), (
        "`Docker/Login-Action` runs the same credential-writing action, and the step has "
        "no `run:` body, so a case-sensitive lookup skipped it entirely"
    )


def test_a_cached_path_resolves_against_job_level_env():
    """The job's own env has to seed the walk, or a root-level call resolves to nothing."""
    job = {"env": {"CACHE_DIR": "hf-cache", "HF_HOME": "hf-cache"}, "steps": [
        {"name": "log in", "run": "hf auth login --token x"},
        {"uses": "actions/cache/save@v4",
         "with": {"path": "${{ env.CACHE_DIR }}", "key": "k"}},
    ]}
    assert [p for p, _s in _persisted_with_env(job, {})] == ["hf-cache"]
    assert _login_offenders({}, job)


def test_an_overridden_default_home_is_not_a_finding():
    """A default is only where a tool looks when nothing points it elsewhere.

    A job setting `CARGO_HOME: /tmp/cargo` writes credentials there, so persisting
    `~/.cargo` holds none of them. Flagging it anyway was a false failure on a correct
    configuration, and every variable in CREDENTIAL_HOMES had the same problem.
    """
    for default, owner in DEFAULT_OWNERS.items():
        if owner is None:
            continue
        assert owner in CREDENTIAL_HOMES, (
            f"{owner} overrides {default} but is not tracked as a credential home"
        )
    assert DEFAULT_OWNERS["~/.cargo"] == "CARGO_HOME"
    assert DEFAULT_OWNERS["~/.cache/huggingface"] == "HF_HOME"


def test_a_reusable_workflow_job_is_flattened(tmp_path, monkeypatch):
    """A job may delegate wholesale to a local reusable workflow instead of listing steps.

    The caller then scanned as an empty job and the called workflow was scanned
    separately with no access to the caller's `with:` values, so a reusable job that sets
    `HF_HOME: ${{ inputs.path }}`, logs in and caches `${{ inputs.path }}` was accepted.
    The same split that hid composite logins, one level up.
    """
    import sys

    module = sys.modules[__name__]
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "shared.yml").write_text(
        "name: shared\n"
        "on:\n  workflow_call:\n    inputs:\n      path:\n        type: string\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    env:\n      HF_HOME: ${{ inputs.path }}\n"
        "    steps:\n      - run: hf auth login --token x\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: ${{ inputs.path }}\n          key: k\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    caller = {"uses": "./.github/workflows/shared.yml", "with": {"path": "hf-cache"}}
    assert [p for p, _s in _persisted_with_env(caller, {})] == ["hf-cache"]
    assert _login_offenders({}, caller), (
        "the reusable job logs in and caches the same input-named directory"
    )


def test_a_reusable_workflow_resolves_the_inputs_it_was_handed(tmp_path, monkeypatch):
    """A reusable workflow forwarding `${{ inputs.path }}` means the CALLER's value.

    Copying a call site's `with:` verbatim made the forwarded expression
    self-referential one level down: workflow A hands B `path: ${{ inputs.path }}`, B
    resolves it against its own empty inputs, and both the credential home and the cached
    path normalise to the empty string. The containment test then compares nothing with
    nothing, the login plus cache is accepted, and the concrete directory the outer
    caller actually supplied never enters the comparison. This is the same forwarding bug
    already fixed for composites, one layer up, and it is a bypass rather than a false
    failure.
    """
    import sys

    module = sys.modules[__name__]
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "inner.yml").write_text(
        "name: inner\n"
        "on:\n  workflow_call:\n    inputs:\n      path:\n        type: string\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    env:\n      HF_HOME: ${{ inputs.path }}\n"
        "    steps:\n      - run: hf auth login --token x\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: ${{ inputs.path }}\n          key: k\n"
    )
    (wf / "outer.yml").write_text(
        "name: outer\n"
        "on:\n  workflow_call:\n    inputs:\n      path:\n        type: string\n"
        "jobs:\n  forward:\n    uses: ./.github/workflows/inner.yml\n"
        "    with:\n      path: ${{ inputs.path }}\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    caller = {"uses": "./.github/workflows/outer.yml", "with": {"path": "hf-cache"}}
    assert [p for p, _s in _persisted_with_env(caller, {})] == ["hf-cache"], (
        "the outer caller's concrete directory has to survive two hops of forwarding"
    )
    assert _login_offenders({}, caller), (
        "the innermost job logs in and caches the directory the outermost caller named"
    )


def test_a_shell_login_only_counts_against_what_that_command_writes():
    """`docker login` writes $DOCKER_CONFIG. It does not write the Hugging Face token.

    Every login pattern was tested against every credential home in the job, so a job
    that legitimately caches `HF_HOME` and separately runs `docker login` -- with
    `DOCKER_CONFIG` nowhere near the cache -- was reported as leaking a Hugging Face
    token into the cache. The same false pairing applied to npm, cargo, aws and gcloud,
    and the report named a credential the command never touches, which is worse than
    silence: it sends the reader looking for a leak that is not there.
    """
    safe = {
        "env": {"HF_HOME": "hf-cache", "DOCKER_CONFIG": "/tmp/docker"},
        "steps": [
            {"run": "echo y | docker login -u u --password-stdin"},
            {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
        ],
    }
    assert _login_offenders({}, safe) == [], (
        "docker writes $DOCKER_CONFIG/config.json, which is outside the cached "
        f"directory: {_login_offenders({}, safe)}"
    )

    unsafe = dict(safe, steps = [
        {"run": "hf auth login --token x"},
        {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
    ])
    assert _login_offenders({}, unsafe), (
        "the Hugging Face login does write the cached HF_HOME, and still has to be caught"
    )

    docker_cached = {
        "env": {"DOCKER_CONFIG": "docker-cache"},
        "steps": [
            {"run": "echo y | docker login -u u --password-stdin"},
            {
                "uses": "actions/cache/save@v4",
                "with": {"path": "docker-cache", "key": "k"},
            },
        ],
    }
    assert _login_offenders({}, docker_cached), (
        "narrowing the pairing must not stop docker being caught against its own home"
    )
