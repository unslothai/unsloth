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
    # Both are consulted before HF_HOME by older releases still pinned in some lanes.
    "HUGGINGFACE_HUB_CACHE": "token",
    "TRANSFORMERS_CACHE": "token",
    # npm and cargo both keep registry credentials in their config roots.
    "NPM_CONFIG_USERCONFIG": ".npmrc auth tokens",
    "CARGO_HOME": "credentials.toml",
    "DOCKER_CONFIG": "config.json auth entries",
    "AWS_SHARED_CREDENTIALS_FILE": "aws credentials",
    "GOOGLE_APPLICATION_CREDENTIALS": "service account json",
}

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


def _expand(value: str, env: dict) -> str:
    """Substitute `${{ env.X }}` from the job's environment before comparing paths.

    `_normalise` deletes expressions wholesale, so `path: ${{ env.HF_HOME }}` reduced to
    the empty string and `_inside` refuses an empty operand. A job that cached exactly
    its own credential home, spelled through the variable rather than repeated
    literally, was therefore silently exempt from the rule aimed at it.
    """
    return re.sub(
        r"\$\{\{\s*env\.([A-Za-z_]\w*)\s*\}\}",
        lambda m: env.get(m.group(1), ""),
        value,
    )


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


def _normalise(path: str) -> str:
    """Strip expressions and separators so a path and an env value can be compared."""
    path = re.sub(r"\$\{\{[^}]*\}\}", "", path)
    path = path.replace("\\", "/").strip().strip("'\"")
    path = re.sub(r"^\$(HOME|\{HOME\})/", "~/", path)
    while path.startswith("./"):
        path = path[2:]
    return path.strip("/")


def _inside(inner: str, outer: str) -> bool:
    """Is `inner` the same directory as `outer`, or below it?"""
    inner, outer = _normalise(inner), _normalise(outer)
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
                for persisted, _step in _persisted_paths(job):
                    persisted = _expand(persisted, env)
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

    offenders = []
    for step in _steps(job):
        body = str(step.get("run") or "")
        if not body:
            continue
        for pattern in LOGIN_PATTERNS:
            if re.search(pattern, body, re.IGNORECASE):
                offenders.append(f"{step.get('name') or step.get('uses')}: matched /{pattern}/")

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
            for persisted, _step in _persisted_paths(job):
                persisted = _expand(persisted, env)
                for default, creds in DEFAULT_CREDENTIAL_HOMES.items():
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
