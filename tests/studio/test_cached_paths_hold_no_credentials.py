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
# What each home actually holds, for deciding whether a restrictive glob could capture
# it. An empty tuple means the variable names the credential FILE itself rather than a
# directory, so the configured path is what a pattern has to match.
#
# Derived per variable rather than from one global list of filenames. The global list
# was a guess that omitted `credentials.toml`, which `CREDENTIAL_HOMES` three lines up
# already documents, so `cargo-home/*.toml` was accepted while it persisted exactly the
# file cargo writes. A canned list cannot cover a file-valued home at all, since those
# take whatever basename the job chooses.
CREDENTIAL_FILES = {
    "HF_HOME": ("token", "stored_tokens"),
    "HF_TOKEN_PATH": (),
    "NPM_CONFIG_USERCONFIG": (),
    "CARGO_HOME": ("credentials", "credentials.toml"),
    "DOCKER_CONFIG": ("config.json",),
    "AWS_SHARED_CREDENTIALS_FILE": (),
    "GOOGLE_APPLICATION_CREDENTIALS": (),
}


# The files each DEFAULT home holds, so a workflow persisting one exactly is caught.
# The configured-home branch already tested `<home>/<file>`; defaults had no variable to
# take filenames from, so uploading `~/.cargo/credentials.toml` outright passed both
# guards -- the default directory is not inside the persisted file, which is the only
# question that was being asked.
DEFAULT_HOME_FILES = {
    "~/.cache/huggingface": ("token", "stored_tokens"),
    "~/.huggingface": ("token",),
    "~/.cargo": ("credentials", "credentials.toml"),
    "~/.docker": ("config.json",),
    "~/.npmrc": (),
    "~/.aws": ("credentials",),
    "~/.config/gh": ("hosts.yml",),
}

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
# Any ONE of these variables being set moves the credential out of the default
# location. Two variables reach the Hugging Face default: HF_HOME relocates the whole
# directory, and HF_TOKEN_PATH relocates the token and `stored_tokens` on their own, so
# a job setting only HF_TOKEN_PATH writes nothing of interest to `~/.cache/huggingface`
# and persisting it was reported anyway. A single owner per default could not express
# that.
DEFAULT_OWNERS = {
    "~/.cache/huggingface": ("HF_HOME", "HF_TOKEN_PATH"),
    "~/.huggingface": ("HF_HOME", "HF_TOKEN_PATH"),
    "~/.cargo": ("CARGO_HOME",),
    "~/.docker": ("DOCKER_CONFIG",),
    "~/.npmrc": ("NPM_CONFIG_USERCONFIG",),
    "~/.aws": ("AWS_SHARED_CREDENTIALS_FILE",),
    "~/.config/gh": (),
}

# Which credential homes each shell login actually writes into. Matching every pattern
# against every variable reported a job that caches `HF_HOME` and runs `docker login` as
# leaking the Hugging Face token, which it plainly does not: docker writes
# $DOCKER_CONFIG/config.json. The same false failure applied to npm, cargo, aws and gcloud.
#
# `None` means "any credential home", used for the Hugging Face patterns because those
# write to whichever of HF_HOME or HF_TOKEN_PATH is in force.
# Both, because `huggingface_hub` computes the token path from HF_TOKEN_PATH when it is
# set and from HF_HOME when it is not, independently of each other.
_HF_HOMES = ("HF_HOME", "HF_TOKEN_PATH")

LOGIN_PATTERN_HOMES = {
    # `hf auth login` writes whichever of these is in force, and nothing else. Leaving
    # them out of this map meant the "any home" fallback below matched them against every
    # variable in the job, so a job caching CARGO_HOME while running an unrelated Hugging
    # Face login failed, naming a Cargo directory the token never goes near. That is the
    # same false pairing the map was added to remove, left in place for the patterns the
    # map exists for.
    r"\bhf\s+auth\s+login\b": _HF_HOMES,
    r"\bhuggingface-cli\s+login\b": _HF_HOMES,
    r"\bhf\s+login\b": _HF_HOMES,
    r"huggingface_hub[.\s]*\.?\s*login\s*\(": _HF_HOMES,
    r"\bfrom\s+huggingface_hub\s+import\s+[^\n]*\blogin\b": _HF_HOMES,
    r"\bHfFolder\b[^\n]*\bsave_token\b": _HF_HOMES,
    r"\bsave_token\s*\(": _HF_HOMES,
    r"add_to_git_credential\s*=\s*True": _HF_HOMES,
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
    # To a fixed point, because a variable's value may name another variable. With
    # `CACHE_ROOT: hf-cache` at workflow level, `HF_HOME: ${{ env.CACHE_ROOT }}` on the
    # job and `path: ${{ env.HF_HOME }}` on the cache step, one pass left
    # `${{ env.CACHE_ROOT }}` standing, `_normalise` erased it, and the job cached its
    # own credential home with no offender reported. Actions resolves the whole chain.
    #
    # The iteration is bounded rather than trusting the chain to terminate: two variables
    # that name each other would otherwise substitute forever. A value still holding an
    # expression after the last pass is left as it is and handled downstream, which is
    # the same outcome as an unknown variable.
    for _ in range(8):
        before = value
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
        if value == before:
            break
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


def _units(job, doc, env = None, inputs = None, depth = 0):
    """The independent runners this job's work lands on, as (job, env, inputs).

    A job is one runner, so a login in it and a cache save in it share a filesystem and
    the pairing is meaningful. A job that delegates with `uses: ./.github/workflows/x.yml`
    is NOT one runner: every job inside that workflow gets its own, exactly as if they
    had been written out separately.

    Flattening all of them into the calling job pooled their paths, so a login in one
    inner job was reported against a cache saved by a different inner job that merely
    used the same pathname. Two jobs writing `hf-cache` write two different directories
    on two different machines, and a finding that says otherwise is describing something
    that cannot happen.

    The split recurses, because a reusable workflow may delegate in turn, and `depth`
    stops a workflow that (transitively) calls itself from expanding forever.
    """
    env = _env_of(job, doc) if env is None else env
    inputs = {} if inputs is None else inputs
    if depth > 8:
        return [(job, env, inputs)]
    inner = _reusable_jobs(job, env, inputs)
    if not inner:
        return [(job, env, inputs)]
    out = []
    for inner_job, inner_env, passed in inner:
        # The called workflow's OWN env, not the caller's. GitHub does not propagate a
        # calling workflow's `env:` into a reusable workflow -- only `with:` and
        # `secrets:` cross that boundary -- so merging them reported a caller-level
        # `HF_HOME: hf-cache` against a called job that saves an unrelated `hf-cache`
        # and writes its token to the default home instead. The inputs still travel,
        # because those are what the caller really passes.
        out.extend(_units(inner_job, doc, inner_env, passed, depth + 1))
    return out



def _ref_candidates(uses: str):
    """Every source-tree path a `./...` action reference could name.

    A job that checks this repository out into a subdirectory writes
    `./unsloth/.github/actions/x`, which is the same action through a layout that only
    exists on the runner. Probing the reference as written found nothing, so the
    composite was never flattened: a job could persist its credential home and delegate
    the login to such a composite with nothing seeing it. `notebooks-ci.yml` and
    `version-compat-ci.yml` both use this form.
    """
    ref = uses.strip()
    if ref.startswith("./"):
        ref = ref[2:]
    ref = ref.rstrip("/")
    bases = [REPO / ref]
    parts = ref.split("/")
    if ".github" in parts[1:]:
        bases.append(REPO / "/".join(parts[parts.index(".github"):]))
    for base in bases:
        for cand in (base, base / "action.yml", base / "action.yaml"):
            yield cand


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

    for step in _steps(job):
        own = step.get("env")
        env = {**inherited, **({str(k): str(v) for k, v in own.items()}
                               if isinstance(own, dict) else {})}
        out.append((step, inherited, inputs))
        uses = str(step.get("uses") or "").strip().strip("'\"")
        if not uses.startswith("./"):
            continue
        for cand in _ref_candidates(uses):
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
    out = []
    for unit, job_env, unit_inputs in _units(job, doc):
        out.extend(_persisted_in_unit(unit, job_env, unit_inputs))
    return out


def _persisted_in_unit(job, job_env, unit_inputs):
    """The paths one runner persists. See `_units` for why that boundary matters."""
    out = []
    # The job's own env seeds the walk. Without it a root-level call passing
    # `path: ${{ env.CACHE_DIR }}` resolved against inherited and step-level values only,
    # `_expand` produced an empty string, and the composite appeared to persist nothing.
    for step, inherited, inputs in _flat_steps(job, job_env, unit_inputs):
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
    offenders = []
    # Per runner, and a login is only ever paired with what its OWN runner persists.
    for unit, unit_env, unit_inputs in _units(job, doc):
        offenders.extend(_login_offenders_in_unit(unit, unit_env, unit_inputs))
    return offenders


def _login_offenders_in_unit(job, job_env, unit_inputs):
    persisted = [path for path, _s in _persisted_in_unit(job, job_env, unit_inputs)]
    if not persisted:
        return []
    offenders = []
    for step, inherited, inputs in _flat_steps(job, job_env, unit_inputs):
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
        # The two Hugging Face variables are alternatives with a precedence, not two
        # places a login writes. `huggingface_hub` takes HF_TOKEN_PATH when it is set and
        # falls back to `$HF_HOME/token` only when it is not, so a job with a cached
        # HF_HOME and HF_TOKEN_PATH pointing outside it writes the token outside it. The
        # report named HF_HOME anyway, which is a false failure that also misdescribes
        # where the credential lives.
        if "HF_TOKEN_PATH" in homes:
            homes.pop("HF_HOME", None)
        if not homes:
            continue
        for var, home in sorted(homes.items()):
            hit = next(
                (p for p in persisted if _inside(home, p, CREDENTIAL_FILES.get(var, ()))),
                None,
            )
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
                    and var in LOGIN_PATTERN_HOMES.get(p, (var,))
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




def _glob_regex(pattern: str):
    """A glob compiled with `/` respected: `**` crosses separators, `*` does not."""
    out = []
    i = 0
    pattern = pattern.replace("\\", "/")
    while i < len(pattern):
        ch = pattern[i]
        if pattern.startswith("**/", i):
            # Zero directories included. `hf-cache/**/token` matches
            # `hf-cache/token`, and translating `**` to `.*` before escaping the
            # following slash demanded a separator that need not be there, so a
            # pattern that does persist the token read as though it did not.
            out.append("(?:[^/]+/)*")
            i += 3
            continue
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
            continue
        if ch == "[":
            # A bracket expression, compiled rather than escaped. `_inside` already
            # counted `[` as making the path a glob, so escaping it here meant
            # `hf-cache/[t]oken` was treated as a glob that matches the literal text
            # `[t]oken` -- it matched nothing, and an upload that does include the token
            # was permitted.
            close = pattern.find("]", i + 1)
            if close == -1:
                out.append(re.escape(ch))
                i += 1
                continue
            body = pattern[i + 1 : close]
            negate = body.startswith("!") or body.startswith("^")
            if negate:
                body = body[1:]
            out.append("[" + ("^" if negate else "") + body.replace("\\", "\\\\") + "]")
            i = close + 1
            continue
        if ch == "*":
            out.append("[^/]*")
        elif ch == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(ch))
        i += 1
    return re.compile("^" + "".join(out) + "$")


def _glob_captures(pattern: str, home: str, files = None) -> bool:
    """Could this persistence pattern include the credential this home holds?

    Asked of the REAL path, not of a list of likely filenames. `path: hf-cache/**` takes
    the token with everything else; `path: hf-cache/*.bin` takes the weight files and
    cannot contain it. Dropping the wildcard segment treated the two alike.

    `files` are the names the credential takes beneath a directory-valued home. A home
    with none is itself the credential file, which is the only workable rule for
    AWS_SHARED_CREDENTIALS_FILE and friends: those take whatever basename the job gives
    them, so no list of filenames can anticipate them and the configured value is the
    only thing that can be tested.
    """
    home = _normalise(home).rstrip("/")
    if files is None:
        # The caller did not say which variable this is, so the filename is unknown and
        # any pattern reaching into the home could match it. Falling back to the old
        # containment keeps an unannotated call site conservative: forgetting the
        # argument must not turn a finding off, which an empty tuple would have done.
        outer = _normalise(_deglob(pattern)).strip("/")
        return bool(outer) and (home == outer or home.startswith(outer + "/"))
    rx = _glob_regex(_normalise(pattern))
    # The home itself, for a variable that names the credential FILE, and each known
    # filename beneath it for one that names a directory.
    candidates = [home] + [home + "/" + f for f in files]
    return any(rx.match(c) for c in candidates)


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

def _inside(inner: str, outer: str, files = None) -> bool:
    """Is `inner` the same directory as `outer`, or below it?

    A glob suffix is trimmed off `outer` first. `path: hf-cache/**` uploads everything
    beneath `hf-cache`, including a token written there, but compared literally
    `_inside("hf-cache", "hf-cache/**")` is false and the whole upload looked unrelated to
    the credential home it contains.
    """
    if any(ch in outer for ch in "*?["):
        # Decided by matching the credential's own path against the pattern, which is
        # both stricter and more honest than widening the pattern to its fixed prefix.
        return _glob_captures(outer, inner, files)
    inner, outer = _normalise(inner), _normalise(_deglob(outer))
    inner, outer = inner.strip("/"), outer.strip("/")
    if not inner or not outer:
        return False
    if inner == outer or inner.startswith(outer + "/"):
        return True
    # The persisted path may name the credential FILE outright rather than a directory
    # holding it. `path: hf-cache/token` is not the home and does not contain it, so a
    # containment test answered no while the upload carried the token itself. The
    # filenames are the same ones a glob is tested against.
    return any(outer == inner.rstrip("/") + "/" + f for f in (files or ()))



def _default_home_hits(persisted: str, overridden: set) -> list:
    """Which default credential homes this persisted path reaches.

    Factored out so the rule can be exercised directly. It was previously inline, and
    the test that meant to check it asserted on the module's own source text -- which
    counted the assertion itself and passed whatever the rule did.
    """
    hits = []
    for default, creds in DEFAULT_CREDENTIAL_HOMES.items():
        # A default is only where the tool looks when nothing overrides it. A job
        # setting `CARGO_HOME: /tmp/cargo` writes credentials there, so persisting
        # `~/.cargo` holds none, and flagging it was a false failure.
        if any(v in overridden for v in DEFAULT_OWNERS.get(default, ())):
            continue
        # With the filenames that home holds, so persisting the credential FILE outright
        # counts as well as persisting the directory.
        if _inside(default, persisted, DEFAULT_HOME_FILES.get(default, ())):
            hits.append((default, creds))
    return hits


def _offending_jobs():
    """(label, var, path, home_value) for every job caching its own credential home.

    Resolved through `_units`, exactly as the login scan is. Reading the caller's own
    job and step environments only meant a job that delegates to a reusable workflow
    contributed nothing: the caller declares no credential home, and the called workflow
    read on its own cannot resolve `HF_HOME: ${{ inputs.path }}` because the value lives
    at the call site. So the parametrized guard below was never instantiated for that
    shape, and `_login_offenders` -- which does detect it -- was never asked. A check
    that works when called directly and is never called is not a check.
    """
    for path, doc in _docs():
        for jid, job in _jobs(doc):
            for unit, unit_env, unit_inputs in _units(job, doc):
                # Through `_flat_steps`, so an env declared INSIDE a composite counts.
                # `_step_envs` sees the calling job's own steps only, and a composite
                # whose inner login step declares its own `HF_HOME` therefore produced
                # no label at all -- so the parametrized guard was never handed the job,
                # though `_login_offenders` detects it. Same shape as the reusable
                # workflow gap, one level further in.
                scopes = [unit_env]
                for step, inherited, _si in _flat_steps(unit, unit_env, unit_inputs):
                    own = step.get("env")
                    scopes.append({
                        **inherited,
                        **({str(k): str(v) for k, v in own.items()}
                           if isinstance(own, dict) else {}),
                    })
                for scope in scopes:
                    env = {**unit_env, **scope}
                    homes = {v: env[v] for v in CREDENTIAL_HOMES if v in env}
                    if "HF_TOKEN_PATH" in homes:
                        homes.pop("HF_HOME", None)
                    if not homes:
                        continue
                    for persisted, _s in _persisted_in_unit(unit, unit_env, unit_inputs):
                        for var, home in homes.items():
                            if _inside(_expand(home, env, unit_inputs), persisted,
                                       CREDENTIAL_FILES.get(var, ())):
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
          # Per resolved unit, because the override and the persistence must come from
          # the SAME runner. `_persisted_with_env` traverses into a called reusable
          # workflow while `overridden` was read from the caller, and caller env does
          # not cross that boundary -- so a caller-level `CARGO_HOME=/tmp/cargo`
          # exempted a callee that caches the real `~/.cargo`, which the caller's
          # variable does nothing to move.
          for unit, unit_env, unit_inputs in _units(job, doc):
            env = unit_env
            overridden = {v for v in CREDENTIAL_HOMES if v in env}
            for persisted, _step in _persisted_in_unit(unit, unit_env, unit_inputs):
                for _default, creds in _default_home_hits(persisted, overridden):
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
    """`path: hf-cache/**` uploads whatever is beneath hf-cache, token included.

    A RESTRICTIVE glob does not. `hf-cache/*.bin` uploads the weight files and nothing
    else, and the token is not one of them, so reporting a login against it was a false
    failure. This assertion originally required the opposite, which recorded the
    over-broad behaviour as if it were the intent; dropping the wildcard segment is
    right for a recursive pattern and wrong for a narrow one.
    """
    assert _deglob("hf-cache/**") == "hf-cache"
    assert _deglob("hf-cache/*.bin") == "hf-cache"
    assert _deglob("hf-cache") == "hf-cache"
    assert _inside("hf-cache", "hf-cache/**") is True
    hf = CREDENTIAL_FILES["HF_HOME"]
    assert _inside("hf-cache", "hf-cache/*.bin", hf) is False
    assert _inside("other", "hf-cache/**", hf) is False
    # A bare `*` matches any NAME, `token` among them, so it still counts.
    assert _inside("hf-cache", "hf-cache/*", hf) is True
    # And a narrow pattern that matches the file THAT home really holds counts too:
    # docker writes `config.json`, cargo writes `credentials.toml`.
    assert _inside("docker-cache", "docker-cache/*.json",
                   CREDENTIAL_FILES["DOCKER_CONFIG"]) is True
    assert _inside("cargo-home", "cargo-home/*.toml",
                   CREDENTIAL_FILES["CARGO_HOME"]) is True
    # A caller that does not say which variable it means gets the conservative answer,
    # because an unknown filename could be anything.
    assert _inside("hf-cache", "hf-cache/*.bin") is True


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
    for default, owners in DEFAULT_OWNERS.items():
        for owner in owners:
            assert owner in CREDENTIAL_HOMES, (
                f"{owner} overrides {default} but is not tracked as a credential home"
                )
    assert DEFAULT_OWNERS["~/.cargo"] == ("CARGO_HOME",)
    # Two variables reach the Hugging Face default. HF_TOKEN_PATH moves the token and
    # `stored_tokens` on its own, so a job setting only that one leaves nothing of
    # interest in `~/.cache/huggingface`, and a single owner could not say so.
    assert DEFAULT_OWNERS["~/.cache/huggingface"] == ("HF_HOME", "HF_TOKEN_PATH")


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


def test_a_hugging_face_login_is_not_judged_against_an_unrelated_home():
    """`hf auth login` writes a Hugging Face home. It does not write CARGO_HOME.

    Mapping only the non-Hugging-Face patterns left the "any home" fallback covering the
    Hugging Face ones, so they matched against every credential variable in the job. A
    job caching `CARGO_HOME` while logging into Hugging Face was reported as leaking its
    token into the Cargo cache -- the same false pairing the map was introduced to
    remove, still in force for the patterns it was written for.
    """
    safe = {
        "env": {"CARGO_HOME": "cargo-cache", "HF_HOME": "/tmp/hf"},
        "steps": [
            {"run": "hf auth login --token x"},
            {
                "uses": "actions/cache/save@v4",
                "with": {"path": "cargo-cache", "key": "k"},
            },
        ],
    }
    assert _login_offenders({}, safe) == [], (
        f"the token goes to /tmp/hf, nowhere near the cached Cargo home: "
        f"{_login_offenders({}, safe)}"
    )

    unsafe = {
        "env": {"CARGO_HOME": "cargo-cache", "HF_HOME": "cargo-cache/hf"},
        "steps": [
            {"run": "hf auth login --token x"},
            {
                "uses": "actions/cache/save@v4",
                "with": {"path": "cargo-cache", "key": "k"},
            },
        ],
    }
    assert _login_offenders({}, unsafe), (
        "with HF_HOME actually inside the cached path this is still a finding"
    )


def test_two_jobs_of_a_reusable_workflow_are_not_one_runner(tmp_path, monkeypatch):
    """Each job runs on its own machine, so `hf-cache` in two jobs is two directories.

    Flattening every job of a called workflow into the calling job pooled their
    persisted paths, so a login in one job was reported against a cache saved by a
    different job that merely used the same pathname. That finding describes something
    that cannot happen, and the cure for a guard like that is usually an exemption
    entry -- which then hides the real case too.
    """
    import sys

    module = sys.modules[__name__]
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "split.yml").write_text(
        "name: split\n"
        "on:\n  workflow_call:\n"
        "jobs:\n"
        # logs in, persists nothing
        "  login:\n    runs-on: ubuntu-latest\n"
        "    env:\n      HF_HOME: hf-cache\n"
        "    steps:\n      - run: hf auth login --token x\n"
        # persists the same NAME, on a different runner, and never logs in
        "  save:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: hf-cache\n          key: k\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    caller = {"uses": "./.github/workflows/split.yml"}
    assert _login_offenders({}, caller) == [], (
        f"the login and the save are on different runners: "
        f"{_login_offenders({}, caller)}"
    )

    # Both in ONE job is a real finding, so the split did not cost the check its teeth.
    (wf / "split.yml").write_text(
        "name: split\n"
        "on:\n  workflow_call:\n"
        "jobs:\n"
        "  both:\n    runs-on: ubuntu-latest\n"
        "    env:\n      HF_HOME: hf-cache\n"
        "    steps:\n      - run: hf auth login --token x\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: hf-cache\n          key: k\n"
    )
    assert _login_offenders({}, caller), (
        "one job logging in and caching its own credential home is still caught"
    )


def test_a_chain_of_environment_references_is_resolved():
    """A variable's value may name another variable, and Actions resolves the chain.

    One substitution pass left the inner expression standing, `_normalise` erased it,
    and a job that cached its own credential home through two hops produced no offender
    at all. Spelling the same thing in two steps instead of one was a complete bypass.
    """
    env = {"CACHE_ROOT": "hf-cache", "HF_HOME": "${{ env.CACHE_ROOT }}"}
    assert _expand("${{ env.HF_HOME }}", env) == "hf-cache"

    job = {
        "env": {"CACHE_ROOT": "hf-cache", "HF_HOME": "${{ env.CACHE_ROOT }}"},
        "steps": [
            {"run": "hf auth login --token x"},
            {
                "uses": "actions/cache/save@v4",
                "with": {"path": "${{ env.HF_HOME }}", "key": "k"},
            },
        ],
    }
    assert _login_offenders({}, job), (
        "the cached path and the credential home are the same directory, reached "
        "through two references"
    )


def test_a_reference_cycle_does_not_hang_the_expansion():
    """Two variables naming each other must terminate, not spin.

    The fixed point is bounded for this reason. A value still holding an expression
    after the last pass is treated exactly like an unknown variable, which is the
    existing behaviour rather than a new one.
    """
    env = {"A": "${{ env.B }}", "B": "${{ env.A }}"}
    out = _expand("${{ env.A }}", env)
    assert "${{" in out, f"an unresolvable cycle stays an expression, got {out!r}"


def test_a_restrictive_glob_does_not_capture_a_token():
    """`hf-cache/*.bin` uploads weight files. The token is not one of them.

    Widening every glob to its containing directory was right for `**` and wrong here,
    and the difference is whether the pattern can match a credential filename at all.
    An unfamiliar name still counts as capturable, so the narrowing only ever accepts a
    pattern that demonstrably excludes every credential file this check knows about.
    """
    hf = CREDENTIAL_FILES["HF_HOME"]
    assert _glob_captures("hf-cache/**", "hf-cache", hf) is True
    assert _glob_captures("hf-cache/*", "hf-cache", hf) is True
    assert _glob_captures("hf-cache/*.bin", "hf-cache", hf) is False
    # docker writes config.json, so a `*.json` pattern does capture a credential.
    assert _glob_captures(
        "docker-cache/*.json", "docker-cache", CREDENTIAL_FILES["DOCKER_CONFIG"]
    ) is True
    # cargo writes credentials.toml, which a canned filename list had omitted.
    assert _glob_captures(
        "cargo-home/*.toml", "cargo-home", CREDENTIAL_FILES["CARGO_HOME"]
    ) is True
    # A file-valued home is matched as the path it names, whatever the basename. No
    # list of likely filenames can cover these, which is why the real value is tested.
    assert _glob_captures("creds/*.ini", "creds/my-profile.ini", ()) is True
    assert _glob_captures("creds/*.ini", "elsewhere/my-profile.ini", ()) is False
    # A pattern reaching only into a SUBDIRECTORY cannot hold a token at the root.
    assert _glob_captures("hf-cache/models/**", "hf-cache", hf) is False

    safe = {
        "env": {"HF_HOME": "hf-cache"},
        "steps": [
            {"run": "hf auth login --token x"},
            {
                "uses": "actions/upload-artifact@v4",
                "with": {"path": "hf-cache/*.bin", "name": "w"},
            },
        ],
    }
    assert _login_offenders({}, safe) == [], (
        f"the upload cannot contain the token: {_login_offenders({}, safe)}"
    )


def test_hf_token_path_overrides_the_home_when_both_are_set():
    """They are alternatives with a precedence, not two places a login writes.

    `huggingface_hub` uses HF_TOKEN_PATH when it is set and `$HF_HOME/token` only when
    it is not, so a cached HF_HOME with HF_TOKEN_PATH pointing outside it holds no
    token. Matching every Hugging Face pattern against both variables reported that
    arrangement and named the wrong file while doing it.
    """
    safe = {
        "env": {"HF_HOME": "hf-cache", "HF_TOKEN_PATH": "/tmp/token"},
        "steps": [
            {"run": "hf auth login --token x"},
            {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
        ],
    }
    assert _login_offenders({}, safe) == [], (
        f"the token is written to /tmp/token: {_login_offenders({}, safe)}"
    )

    unsafe = {
        "env": {"HF_HOME": "/tmp/hf", "HF_TOKEN_PATH": "hf-cache/token"},
        "steps": [
            {"run": "hf auth login --token x"},
            {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
        ],
    }
    assert _login_offenders({}, unsafe), (
        "HF_TOKEN_PATH inside the cache is the finding, and it is the variable that "
        "decides"
    )


def test_a_reusable_workflow_does_not_inherit_the_callers_env(tmp_path, monkeypatch):
    """`env:` does not cross the reusable-workflow boundary. `with:` does.

    Merging the caller's environment into every inner job meant a caller-level
    `HF_HOME: hf-cache` was attributed to a called workflow that saves an unrelated
    `hf-cache` and writes its token to the default home instead.
    """
    import sys

    module = sys.modules[__name__]
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "inner.yml").write_text(
        "name: inner\n"
        "on:\n  workflow_call:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n      - run: hf auth login --token x\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: hf-cache\n          key: k\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    caller = {"env": {"HF_HOME": "hf-cache"}, "uses": "./.github/workflows/inner.yml"}
    assert _login_offenders({}, caller) == [], (
        f"the caller's HF_HOME never reaches the called workflow: "
        f"{_login_offenders({}, caller)}"
    )

    # Passed as an INPUT, it does reach it, and that is still a finding.
    (wf / "inner.yml").write_text(
        "name: inner\n"
        "on:\n  workflow_call:\n    inputs:\n      home:\n        type: string\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    env:\n      HF_HOME: ${{ inputs.home }}\n"
        "    steps:\n      - run: hf auth login --token x\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: hf-cache\n          key: k\n"
    )
    passed = {
        "uses": "./.github/workflows/inner.yml",
        "with": {"home": "hf-cache"},
    }
    assert _login_offenders({}, passed), "an input does cross the boundary"


def test_the_discovery_scan_reaches_a_reusable_workflow_unit(tmp_path, monkeypatch):
    """The scan that DRIVES the parametrized guard has to resolve what the guard does.

    `_login_offenders` detects a called workflow that sets `HF_HOME` from an input,
    logs in and caches that input. The discovery generator read only the caller's own
    job and step environments, so it yielded nothing for that shape: the caller declares
    no credential home, and the called workflow read alone cannot resolve the input
    because its value lives at the call site. The parametrized case was therefore never
    created, and a check that is never called is not a check.
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
    (wf / "caller.yml").write_text(
        "name: caller\n"
        "on:\n  push:\n"
        "jobs:\n  go:\n    uses: ./.github/workflows/inner.yml\n"
        "    with:\n      path: hf-cache\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "WORKFLOWS", wf)
    monkeypatch.setattr(module, "ACTIONS", tmp_path / ".github" / "actions")

    found = [f for f in _offending_jobs() if "caller.yml" in str(f[0])]
    assert found, (
        "the discovery scan has to reach the delegating job, or the guard below is "
        "never instantiated for it"
    )


def test_a_restrictive_glob_is_matched_against_the_real_credential_path():
    """The filename comes from the home being tested, not from a canned list.

    A global list of likely filenames omitted `credentials.toml`, which
    `CREDENTIAL_HOMES` already documents for CARGO_HOME, so `cargo-home/*.toml` was
    accepted while persisting exactly the file cargo writes. And a file-valued home such
    as AWS_SHARED_CREDENTIALS_FILE takes whatever basename the job chooses, which no
    list can anticipate -- the configured value is the only thing that can be tested.
    """
    cargo = CREDENTIAL_FILES["CARGO_HOME"]
    assert _inside("cargo-home", "cargo-home/*.toml", cargo) is True

    leaking = {
        "env": {"CARGO_HOME": "cargo-home"},
        "steps": [
            {"run": "cargo login $TOKEN"},
            {
                "uses": "actions/cache/save@v4",
                "with": {"path": "cargo-home/*.toml", "key": "k"},
            },
        ],
    }
    assert _login_offenders({}, leaking), (
        "cargo writes cargo-home/credentials.toml, which this pattern persists"
    )

    named = {
        "env": {"AWS_SHARED_CREDENTIALS_FILE": "creds/my-profile.ini"},
        "steps": [
            {"run": "aws configure set aws_access_key_id x"},
            {
                "uses": "actions/upload-artifact@v4",
                "with": {"path": "creds/*.ini", "name": "c"},
            },
        ],
    }
    assert _login_offenders({}, named), (
        "the configured path matches the pattern, whatever its basename"
    )


def test_hf_token_path_alone_moves_the_token_out_of_the_default_home():
    """Two variables reach the Hugging Face default, and either one relocates the token.

    A job setting only HF_TOKEN_PATH writes the token and `stored_tokens` beside it, so
    `~/.cache/huggingface` holds none and persisting it is safe. Tracking a single owner
    per default could not say that, and the default-home rule reported it anyway.
    """
    assert DEFAULT_OWNERS["~/.cache/huggingface"] == ("HF_HOME", "HF_TOKEN_PATH")
    assert DEFAULT_OWNERS["~/.huggingface"] == ("HF_HOME", "HF_TOKEN_PATH")
    # Every named owner is a variable this module actually tracks.
    for default, owners in DEFAULT_OWNERS.items():
        for owner in owners:
            assert owner in CREDENTIAL_HOMES, f"{owner} for {default}"


def test_a_globstar_matches_zero_directories():
    """`hf-cache/**/token` includes `hf-cache/token`. GitHub's globstar may match none.

    Translating `**` to `.*` and then escaping the slash after it demanded a separator
    that need not be there, so a pattern which really does persist the token read as
    though it did not, and a job could log in and upload it with neither guard firing.
    """
    hf = CREDENTIAL_FILES["HF_HOME"]
    assert _inside("hf-cache", "hf-cache/**/token", hf) is True
    assert _inside("hf-cache", "hf-cache/**/*", hf) is True
    assert _inside("hf-cache", "hf-cache/**", hf) is True
    # Depth beyond zero still matches.
    assert _glob_captures("hf-cache/**/token", "hf-cache", hf) is True
    # And the narrowing is not lost: a restrictive tail still excludes the token.
    assert _inside("hf-cache", "hf-cache/**/*.bin", hf) is False

    cargo = CREDENTIAL_FILES["CARGO_HOME"]
    leaking = {
        "env": {"CARGO_HOME": "cargo-home"},
        "steps": [
            {"run": "cargo login $TOKEN"},
            {
                "uses": "actions/upload-artifact@v4",
                "with": {"path": "cargo-home/**/credentials.toml", "name": "c"},
            },
        ],
    }
    assert _inside("cargo-home", "cargo-home/**/credentials.toml", cargo) is True
    assert _login_offenders({}, leaking), (
        "the upload includes cargo-home/credentials.toml at depth zero"
    )


def test_the_discovery_scan_sees_an_env_declared_inside_a_composite(tmp_path, monkeypatch):
    """A credential home set by a composite's own step has to activate the check.

    The discovery generator read the calling job's direct step environments, so a
    composite whose inner login step declares `HF_HOME` produced no label and the
    parametrized guard was never handed the job -- though `_login_offenders` detects it.
    The same shape as the reusable-workflow gap, one level further in.
    """
    import sys

    module = sys.modules[__name__]
    wf = tmp_path / ".github" / "workflows"
    act = tmp_path / ".github" / "actions" / "inner-login"
    wf.mkdir(parents = True)
    act.mkdir(parents = True)
    (act / "action.yml").write_text(
        "name: inner login\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - run: hf auth login --token x\n"
        "      shell: bash\n"
        "      env:\n        HF_HOME: hf-cache\n"
    )
    (wf / "caller.yml").write_text(
        "name: caller\n"
        "on:\n  push:\n"
        "jobs:\n  go:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/inner-login\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: hf-cache\n          key: k\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "WORKFLOWS", wf)
    monkeypatch.setattr(module, "ACTIONS", tmp_path / ".github" / "actions")

    found = [f for f in _offending_jobs() if "caller.yml" in str(f[0])]
    assert found, (
        "the env is declared inside the composite, and the scan has to reach it or the "
        "guard is never instantiated for this job"
    )


def test_a_character_class_is_compiled_not_escaped():
    """`hf-cache/[t]oken` includes `hf-cache/token`.

    `_inside` already counted `[` as making the path a glob, and the compiler then
    escaped it as a literal bracket, so the pattern matched nothing at all and an upload
    that does carry the token was permitted.
    """
    hf = CREDENTIAL_FILES["HF_HOME"]
    assert _inside("hf-cache", "hf-cache/[t]oken", hf) is True
    assert _inside("hf-cache", "hf-cache/[a-z]oken", hf) is True
    # A negated class that excludes the token really does exclude it.
    assert _inside("hf-cache", "hf-cache/[!t]oken", hf) is False

    leaking = {
        "env": {"HF_HOME": "hf-cache"},
        "steps": [
            {"run": "hf auth login --token x"},
            {
                "uses": "actions/upload-artifact@v4",
                "with": {"path": "hf-cache/[t]oken", "name": "a"},
            },
        ],
    }
    assert _login_offenders({}, leaking), "the upload pattern includes the token"


def test_a_persisted_path_that_names_the_credential_file_is_caught():
    """A job may persist the credential itself rather than the directory holding it.

    `path: hf-cache/token` is not the home and does not contain it, so a containment
    test answered no while the upload carried the token outright. The filenames tested
    are the ones a glob is already checked against.
    """
    hf = CREDENTIAL_FILES["HF_HOME"]
    assert _inside("hf-cache", "hf-cache/token", hf) is True
    assert _inside("hf-cache", "hf-cache/stored_tokens", hf) is True
    assert _inside("hf-cache", "hf-cache/weights.bin", hf) is False

    leaking = {
        "env": {"HF_HOME": "hf-cache"},
        "steps": [
            {"run": "hf auth login --token x"},
            {
                "uses": "actions/upload-artifact@v4",
                "with": {"path": "hf-cache/token", "name": "a"},
            },
        ],
    }
    assert _login_offenders({}, leaking), "the token itself is the uploaded path"


def test_a_default_home_override_must_come_from_the_same_runner(tmp_path, monkeypatch):
    """Caller `env:` does not reach a called workflow, so it overrides nothing there.

    A caller-level `CARGO_HOME=/tmp/cargo` was read as exempting a called job that
    caches the real `~/.cargo`, because the persistence traversal crossed the boundary
    while the override was still read from the caller. The variable moves nothing in the
    callee.
    """
    import sys

    module = sys.modules[__name__]
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "inner.yml").write_text(
        "name: inner\n"
        "on:\n  workflow_call:\n    inputs:\n      path:\n        type: string\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: ${{ inputs.path }}\n          key: k\n"
    )
    (wf / "caller.yml").write_text(
        "name: caller\n"
        "on:\n  push:\n"
        "jobs:\n  go:\n"
        "    env:\n      CARGO_HOME: /tmp/cargo\n"
        "    uses: ./.github/workflows/inner.yml\n"
        "    with:\n      path: ~/.cargo\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "WORKFLOWS", wf)
    monkeypatch.setattr(module, "ACTIONS", tmp_path / ".github" / "actions")

    units = _units({"uses": "./.github/workflows/inner.yml", "with": {"path": "~/.cargo"},
                    "env": {"CARGO_HOME": "/tmp/cargo"}}, {})
    assert units, "the delegation resolves to at least one unit"
    for _unit, unit_env, _inputs in units:
        assert "CARGO_HOME" not in unit_env, (
            "the caller's variable must not appear in the called job's environment, or "
            "it will be read as overriding a default it cannot move"
        )


def test_a_composite_used_from_a_checkout_subdirectory_is_flattened(tmp_path, monkeypatch):
    """`./unsloth/.github/actions/x` names the same composite, one layout later.

    Probing the reference as written found nothing in the source tree, so the steps were
    never flattened and a job could persist its credential home while delegating the
    login to such a composite with nothing seeing it.
    """
    import sys

    module = sys.modules[__name__]
    wf = tmp_path / ".github" / "workflows"
    act = tmp_path / ".github" / "actions" / "hidden-login"
    wf.mkdir(parents = True)
    act.mkdir(parents = True)
    (act / "action.yml").write_text(
        "name: hidden login\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - run: hf auth login --token x\n      shell: bash\n"
    )
    monkeypatch.setattr(module, "REPO", tmp_path)

    job = {
        "env": {"HF_HOME": "hf-cache"},
        "steps": [
            {"uses": "./unsloth/.github/actions/hidden-login"},
            {"uses": "actions/cache/save@v4", "with": {"path": "hf-cache", "key": "k"}},
        ],
    }
    assert _login_offenders({}, job), (
        "the prefixed reference names the same composite, which logs in"
    )


def test_a_default_credential_file_persisted_exactly_is_caught():
    """`path: ~/.cargo/credentials.toml` is the credential, not a directory holding it.

    The configured-home branch was taught to match `<home>/<file>`; defaults had no
    variable to take filenames from, so persisting the exact default file asked only
    whether the default DIRECTORY was inside the persisted FILE, which is never true.
    """
    assert DEFAULT_HOME_FILES["~/.cargo"] == ("credentials", "credentials.toml")
    assert _inside(
        "~/.cargo", "~/.cargo/credentials.toml", DEFAULT_HOME_FILES["~/.cargo"]
    ) is True
    assert _inside(
        "~/.docker", "~/.docker/config.json", DEFAULT_HOME_FILES["~/.docker"]
    ) is True
    assert _inside(
        "~/.cache/huggingface",
        "~/.cache/huggingface/token",
        DEFAULT_HOME_FILES["~/.cache/huggingface"],
    ) is True
    # Something else under the same home is still not a credential.
    assert _inside(
        "~/.cargo", "~/.cargo/registry", DEFAULT_HOME_FILES["~/.cargo"]
    ) is False
    # The filenames are load-bearing: without them the same question answers no, which
    # is exactly what the default-home rule was asking before they were supplied.
    assert _inside("~/.cargo", "~/.cargo/credentials.toml") is False
    # Every default this module reports has its filenames recorded.
    for default in DEFAULT_CREDENTIAL_HOMES:
        assert default in DEFAULT_HOME_FILES, default
    # And the rule really consults them. Asserted by calling it, not by reading this
    # file's own source -- the previous version counted its own assertion text and
    # passed however the rule behaved.
    assert _default_home_hits("~/.cargo/credentials.toml", set()), (
        "persisting the exact default credential file has to be a finding"
    )
    assert _default_home_hits("~/.docker/config.json", set())
    assert not _default_home_hits("~/.cargo/registry", set())
    # An override still exempts it, so the narrowing from earlier rounds is intact.
    assert not _default_home_hits("~/.cargo/credentials.toml", {"CARGO_HOME"})
