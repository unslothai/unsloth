# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cache budget is shared 50 GiB LRU: PR-ref saves evict main's copies, so only main may save."""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"

# Jobs whose pip cache earns its place (torch/transformers-class installs). They use the
# pip-cache-restore / pip-cache-save pair, not setup-python's `cache: 'pip'`, which saves
# ungated on PR refs and evicts main's shared entry.
PIP_CACHE_JOBS = {
    ("consolidated-tests-ci.yml", "consolidated"),
    # Same install as `consolidated` but needs its own key: saves are gated on
    # `cache-hit != 'true'`, so a shared key lets only one job ever save.
    ("consolidated-tests-ci.yml", "consolidated-zoo"),
    ("consolidated-tests-ci.yml", "llama-cpp-smoke"),
    ("mlx-ci.yml", "dispatch"),
    ("notebooks-ci.yml", "api-introspect"),
    # Eight matrix legs install the same 709-line Colab freeze; cron and dispatch only.
    ("notebooks-ci.yml", "smoke-install"),
    ("studio-backend-ci.yml", "pytest"),
    ("studio-backend-ci.yml", "repo-cpu-tests"),
    ("studio-export-capability-ci.yml", "capability"),
    ("version-compat-ci.yml", "zoo-imports-under-spoof"),
    ("version-compat-ci.yml", "grpo-fake-run"),
}

# Partial saves on purpose: ccache checksums entries, so a truncated cache only costs misses.
PARTIAL_SAVE_JOBS = {("prebuilt-cuda-wheels.yml", "warm")}

HEAVY = re.compile(
    r"torch|transformers|trl|peft|vllm|bitsandbytes|sentence-transformers|diffusers"
    r"|accelerate|datasets|requirements/"
)


def _workflows():
    for f in sorted(WORKFLOWS.glob("*.yml")):
        try:
            doc = yaml.safe_load(f.read_text(encoding = "utf-8"))
        except yaml.YAMLError as exc:  # a broken workflow is another test's problem
            pytest.fail(f"{f.name} does not parse: {exc}")
        if isinstance(doc, dict) and isinstance(doc.get("jobs"), dict):
            yield f.name, doc


def _jobs():
    for name, doc in _workflows():
        for jid, job in doc["jobs"].items():
            if isinstance(job, dict):
                yield name, jid, job


def _split_top(expr: str, op: str) -> list[str]:
    """``expr`` split on its TOP-LEVEL ``op``, ignoring occurrences in parens or quotes."""
    parts, depth, quote, buf, i = [], 0, "", [], 0
    while i < len(expr):
        ch = expr[i]
        if quote:
            if ch == quote:
                quote = ""
            buf.append(ch)
        elif ch in "'\"":
            quote = ch
            buf.append(ch)
        elif ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif depth == 0 and expr[i : i + 2] == op:
            parts.append("".join(buf))
            buf = []
            i += 2
            continue
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def _balanced(expr: str) -> bool:
    """Whether parentheses in ``expr`` are balanced outside quotes."""
    depth, quote = 0, ""
    for ch in expr:
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


# A positive equality against main in either quote style; `!=` must not match.
_MAIN_ONLY = re.compile(r"github\.ref\s*==\s*['\"]refs/heads/main['\"]")
_LEAF_MAIN = re.compile(
    r"github\.ref\s*==\s*['\"]refs/heads/main['\"]|['\"]refs/heads/main['\"]\s*==\s*github\.ref"
)
# Staging repos (scripts/pr_review/staging_ci.py) open PRs against `upstream-main` and never
# push their own main, so their dependency caches are seeded by a run dispatched on that branch.
# The equality counts as restricted ONLY next to an AND-ed guard that excludes this repository:
# a branch ref is never a PR ref, but here upstream-main has no business saving anything.
_LEAF_SEED = re.compile(
    r"github\.ref\s*==\s*['\"]refs/heads/upstream-main['\"]"
    r"|['\"]refs/heads/upstream-main['\"]\s*==\s*github\.ref"
)
_NOT_THIS_REPO = re.compile(r"github\.repository\s*!=\s*['\"]unslothai/unsloth['\"]")


def _restricted_to_main(expr: str) -> bool:
    """An OR restricts only if every branch does, an AND if any does; negated expressions are refused."""
    if not expr.strip():
        return False

    def strip(part: str) -> str:
        part = part.strip()
        while part.startswith("(") and part.endswith(")") and _balanced(part[1:-1]):
            part = part[1:-1].strip()
        return part

    def restricted(part: str, seed_ok: bool = False) -> bool:
        part = strip(part)
        ors = _split_top(part, "||")
        if len(ors) > 1:
            return all(restricted(p) for p in ors)
        ands = _split_top(part, "&&")
        if len(ands) > 1:
            # The staging seed branch is accepted only when a SIBLING of this same AND
            # excludes this repository; the guard cannot come from an outer level.
            guarded = any(_NOT_THIS_REPO.fullmatch(strip(p)) for p in ands)
            return any(restricted(p, seed_ok = guarded) for p in ands)
        if re.search(r"!(?!=)", part):
            return False
        if seed_ok and _LEAF_SEED.fullmatch(part):
            return True
        # The leaf must BE the equality, not contain it. Searching inside accepted
        # every wrapper that quotes it and inverts it: `(...) == false`, `... != true`,
        # and `startsWith(..., 'false')`, which is true off main because GitHub casts
        # the inner boolean to a string. A whitelist ends the class.
        return bool(_LEAF_MAIN.fullmatch(part))

    return restricted(expr)


@pytest.mark.parametrize(
    "expr,restricted",
    [
        ("always() && github.ref == 'refs/heads/main'", True),
        ('always() && github.ref == "refs/heads/main"', True),
        ("github.ref == 'refs/heads/main' && steps.x.outcome == 'success'", True),
        ("github.ref != 'refs/heads/main'", False),
        ("github.ref == 'refs/heads/main' || github.event_name == 'pull_request'", False),
        ("(github.ref == 'refs/heads/main' && always()) || github.event_name == 'push'", False),
        (
            "(github.ref == 'refs/heads/main' && always()) || "
            "(github.ref == 'refs/heads/main' && failure())",
            True,
        ),
        # A `||` inside a string is not a split.
        ("github.ref == 'refs/heads/main' && contains(x, 'a||b')", True),
        ("", False),
        # A `||` inside parens is still a `||`.
        (
            "always() && (github.ref == 'refs/heads/main' "
            "|| github.event_name == 'pull_request')",
            False,
        ),
        ("!(github.ref == 'refs/heads/main')", False),
        ("(github.ref == 'refs/heads/main') && always()", True),
        ("(github.ref == 'refs/heads/main') == false", False),
        ("github.ref == 'refs/heads/main' != true", False),
        # String functions cast the inner boolean, so this is true only off main.
        ("startsWith(github.ref == 'refs/heads/main', 'false')", False),
        ("contains(github.ref == 'refs/heads/main', 'false')", False),
        ("'refs/heads/main' == github.ref", True),
        ("always() && (github.ref == 'refs/heads/main' && !cancelled())", True),
        # The staging seed: upstream-main, and only outside this repository.
        (
            "always() && (github.ref == 'refs/heads/main' || (github.ref == 'refs/heads/upstream-main'"
            " && github.repository != 'unslothai/unsloth')) && x != 'true'",
            True,
        ),
        ("github.ref == 'refs/heads/upstream-main'", False),
        ("github.ref == 'refs/heads/upstream-main' && always()", False),
        # The repository guard must be an AND sibling of the seed equality itself.
        (
            "github.repository != 'unslothai/unsloth' && (github.ref == 'refs/heads/upstream-main'"
            " || github.event_name == 'pull_request')",
            False,
        ),
        (
            "github.ref == 'refs/heads/upstream-main' && github.repository == 'unslothai/unsloth'",
            False,
        ),
        (
            "github.ref == 'refs/heads/upstream-main' || github.repository != 'unslothai/unsloth'",
            False,
        ),
    ],
)
def test_the_main_only_expression_check_reads_the_expression(expr, restricted):
    """The guard below is only as good as this predicate, so the predicate is tested too."""
    assert _restricted_to_main(expr) is restricted, expr


def _composite_actions():
    """Yields every composite action's steps too, so rules still apply once logic is moved into one."""
    for f in sorted((REPO / ".github" / "actions").rglob("action.yml")):
        doc = yaml.safe_load(f.read_text(encoding = "utf-8"))
        if isinstance(doc, dict):
            yield f.parent.name, ((doc.get("runs") or {}).get("steps") or [])


#: `workflow_call` runs on the caller's ref, so a PR caller saves on the PR ref.
_PR_REACHABLE_TRIGGERS = frozenset({"pull_request", "pull_request_target", "workflow_call"})

# Triggers that never run on a PR ref unprompted; unlisted events such as merge_group stay indicted.
_REF_SAFE_TRIGGERS = frozenset({"schedule", "repository_dispatch", "workflow_dispatch"})


def _triggers(doc: dict) -> dict:
    """Normalises the on: block to a dict; YAML 1.1 parses a bare on: key as True, so both are checked."""
    raw = doc.get("on", doc.get(True))
    if isinstance(raw, str):
        return {raw: None}
    if isinstance(raw, list):
        return {event: None for event in raw}
    return raw if isinstance(raw, dict) else {}


def _pull_request_reachable(doc: dict) -> bool:
    """Reachable unless on: proves otherwise; an unfiltered push or unknown trigger counts as reachable."""
    triggers = _triggers(doc)
    if not triggers:
        # No parseable `on:` says nothing, so it is not exempt.
        return True
    for event, spec in triggers.items():
        if event in _PR_REACHABLE_TRIGGERS:
            return True
        if event == "push":
            branches = (spec or {}).get("branches") if isinstance(spec, dict) else None
            if not branches or [b for b in branches if b != "main"]:
                return True
            continue
        if event not in _REF_SAFE_TRIGGERS:
            return True
    return False


@pytest.mark.parametrize(
    ("on_block", "reachable"),
    [
        ({"workflow_dispatch": None}, False),
        ({"schedule": [{"cron": "0 0 * * *"}]}, False),
        ({"push": {"branches": ["main"]}}, False),
        ({"workflow_dispatch": None, "push": {"branches": ["main"]}}, False),
        ({"pull_request": None}, True),
        ({"pull_request_target": {"types": ["opened"]}}, True),
        ({"workflow_call": None}, True),
        # No `branches` filter means every branch, where PRs here come from.
        ({"push": None}, True),
        ({"push": {"branches": ["main", "release/**"]}}, True),
        ({"push": {"tags": ["v*"]}}, True),
        ({"workflow_dispatch": None, "pull_request": {"paths": ["x"]}}, True),
        ({"repository_dispatch": {"types": ["x"]}}, False),
        # The merge queue runs on refs/heads/gh-readonly-queue/..., not main.
        ({"merge_group": None}, True),
        ({"release": {"types": ["published"]}}, True),
        ({}, True),
    ],
)
def test_the_pull_request_reachability_check_reads_the_trigger_block(on_block, reachable):
    """The reachability predicate is tested directly, since a bug in it silently disables the rule."""
    assert _pull_request_reachable({"on": on_block}) is reachable, on_block
    # YAML 1.1 turns a bare `on:` key into True.
    assert _pull_request_reachable({True: on_block}) is reachable, on_block


def test_no_workflow_saves_a_cache_on_a_pull_request_ref():
    """Saves on a PR ref are flagged by trigger, not by filename; composite actions are always flagged."""
    offenders = []
    for name, steps in _composite_actions():
        for step in steps:
            uses = _uses(step)
            if "actions/cache" not in uses or "/restore@" in uses:
                continue
            if "refs/heads/main" not in str(step.get("if", "")):
                offenders.append(f"action {name}: {step.get('name') or step.get('uses')}")
    for name, doc in _workflows():
        if not _pull_request_reachable(doc):
            continue
        for jid, job in doc["jobs"].items():
            if not isinstance(job, dict):
                continue
            for step in job.get("steps") or []:
                uses = _uses(step)
                # setup-python's `cache:` saves via an ungated post-step on whatever ref the job ran on.
                if "setup-python" in uses and (step.get("with") or {}).get("cache"):
                    offenders.append(f"{name}:{jid}: setup-python implicit post-step save")
                    continue
                if "actions/cache" not in uses:
                    continue
                saves = "/restore@" not in uses  # read-write and /save@ both write
                if not saves:
                    continue
                if not _restricted_to_main(str(step.get("if", ""))):
                    offenders.append(f"{name}:{jid}: {step.get('name') or step.get('uses')}")
    assert not offenders, (
        "these steps save a cache on whatever ref they run on, so every PR writes its own "
        "copy and evicts the copy on main that all PRs share:\n  " + "\n  ".join(offenders)
    )


def test_no_job_uses_setup_pythons_built_in_pip_cache():
    """setup-python cache: 'pip' saves on any ref ungated; use pip-cache-restore and pip-cache-save."""
    offenders = [
        f"{name}:{jid}"
        for name, jid, job in _jobs()
        for step in job.get("steps") or []
        if "setup-python" in _uses(step) and (step.get("with") or {}).get("cache")
    ]
    assert not offenders, (
        f"these jobs use setup-python's built-in cache, which saves on every ref with no "
        f"way to gate it: {offenders}. Swap to the pip-cache-restore / pip-cache-save pair."
    )


def _pip_cache_users():
    """Every job that touches either half of the pip-cache action pair, discovered."""
    return {
        (name, jid)
        for name, jid, job in _jobs()
        for step in job.get("steps") or []
        if "pip-cache-restore" in str(step.get("uses", ""))
        or "pip-cache-save" in str(step.get("uses", ""))
    }


def test_only_the_allowlisted_jobs_use_the_pip_cache_actions():
    """Checks real pip cache usage against PIP_CACHE_JOBS, so a new job cannot escape the other checks."""
    extra = _pip_cache_users() - PIP_CACHE_JOBS
    assert not extra, (
        f"these jobs use the pip cache without being listed in PIP_CACHE_JOBS: "
        f"{sorted(extra)}. Every entry competes for the shared 50 GiB budget, so a job "
        f"earns one by installing a torch/transformers-class dependency set where the "
        f"download dominates. Add it to the allowlist with that justification, or drop the "
        f"cache."
    )


def test_every_pip_cache_user_actually_installs_something_heavy():
    """Each PIP_CACHE_JOBS entry must still install something heavy; a thin job should not keep a cache."""
    thin = []
    for name, jid in sorted(_pip_cache_users()):
        job = dict(_workflows())[name]["jobs"][jid]
        body = "\n".join(
            str(step.get("run", "")) + str(step.get("with", "")) for step in job.get("steps") or []
        )
        if not HEAVY.search(body):
            thin.append(f"{name}:{jid}")
    assert not thin, (
        f"these jobs hold a pip cache but no longer install anything that justifies it: " f"{thin}"
    )


def _pip_cache_steps(name, jid):
    """(restore step, save step) for a job, either of which may be None."""
    job = dict(_workflows())[name]["jobs"][jid]
    restore = save = None
    for step in job.get("steps") or []:
        uses = str(step.get("uses", ""))
        if "pip-cache-restore" in uses:
            restore = step
        elif "pip-cache-save" in uses:
            save = step
    return restore, save


@pytest.mark.parametrize("name,jid", sorted(PIP_CACHE_JOBS))
def test_every_pip_cache_scopes_its_key_to_what_it_installs(name, jid):
    """Scope pip cache keys to the files each job installs from, or any requirements edit orphans them."""
    restore, _ = _pip_cache_steps(name, jid)
    assert restore is not None, f"{name}:{jid} no longer restores a pip cache"
    files = [
        l.strip()
        for l in str((restore.get("with") or {}).get("key-files") or "").splitlines()
        if l.strip()
    ]
    assert files, f"{name}:{jid} passes no key-files, so the key describes nothing"


@pytest.mark.parametrize("name,jid", sorted(PIP_CACHE_JOBS))
def test_every_restored_pip_cache_is_also_saved_and_wired_to_its_restore(name, jid):
    """A restore without a save fills nothing; a save reading the wrong restore outputs writes nothing."""
    restore, save = _pip_cache_steps(name, jid)
    assert restore is not None and save is not None, (
        f"{name}:{jid} has restore={restore is not None}, save={save is not None}; the "
        f"pair has to stay together or the cache is never populated"
    )
    ident = restore.get("id")
    assert ident, f"{name}:{jid}'s restore step has no id, so the save cannot read its outputs"
    with_ = save.get("with") or {}
    for field in ("dir", "key", "cache-hit"):
        assert f"steps.{ident}.outputs.{field}" in str(
            with_.get(field, "")
        ), f"{name}:{jid}'s save does not take {field} from steps.{ident}.outputs"


def test_the_pip_cache_save_action_is_gated_on_the_default_branch():
    """The one place the gate lives, now that nine call sites share it."""
    doc = yaml.safe_load(
        (REPO / ".github" / "actions" / "pip-cache-save" / "action.yml").read_text(encoding = "utf-8")
    )
    steps = (doc.get("runs") or {}).get("steps") or []
    saves = [s for s in steps if "actions/cache" in _uses(s)]
    assert saves, "pip-cache-save no longer saves anything"
    for s in saves:
        cond = str(s.get("if", ""))
        assert "refs/heads/main" in cond, (
            "the pip cache save is no longer gated on the default branch, so all nine call "
            "sites went back to writing PR-scoped entries at once"
        )


def test_the_pip_cache_key_carries_the_interpreter_minor_not_its_patch():
    """Key on the interpreter minor, not patch: each image bump would duplicate the whole cache."""
    body = (REPO / ".github" / "actions" / "pip-cache-restore" / "action.yml").read_text(
        encoding = "utf-8"
    )
    assert 'print("%d.%d" % sys.version_info[:2])' in body, (
        "the pip cache key no longer derives the interpreter version as a minor. If it "
        "went back to sys.version_info[:3], every runner-image patch bump silently "
        "doubles the largest family in the cache."
    )
    assert "sys.version_info[:3]" not in body, (
        "the pip cache key is back to the full patch version, which duplicated 22.06 "
        "GiB across two otherwise identical entries the last time it was measured"
    )


def _workflows_by_name():
    return dict(_workflows())


@pytest.mark.parametrize("name,jid", sorted(PIP_CACHE_JOBS))
def test_every_allowed_pip_cache_job_still_exists_and_still_earns_it(name, jid):
    """The list must not outlive the jobs, or it silently permits nothing."""
    doc = dict(_workflows()).get(name)
    assert doc is not None, f"{name} no longer exists; drop it from PIP_CACHE_JOBS"
    job = doc["jobs"].get(jid)
    assert job is not None, f"{name} no longer has job {jid}; drop it from PIP_CACHE_JOBS"
    body = "\n".join(str(s.get("run", "")) for s in job.get("steps") or [])
    assert HEAVY.search(body), (
        f"{name}:{jid} is allowed a pip cache but no longer installs anything heavy; it "
        f"should give the budget back"
    )


def test_the_cold_install_lanes_never_restore_a_cache():
    """Cold-install lanes must not restore a cache: a warm run proves nothing about a cold install."""
    cold = [
        "clean-machine-install-ci.yml",
        "desktop-app-clean-machine-ci.yml",
        "interrupted-install-ci.yml",
    ]
    offenders = []
    for name, jid, job in _jobs():
        if name not in cold:
            continue
        for step in job.get("steps") or []:
            uses = _uses(step)
            if "actions/cache" in uses:
                offenders.append(f"{name}:{jid}: {step.get('name') or step.get('uses')}")
            if "setup-python" in uses and (step.get("with") or {}).get("cache"):
                offenders.append(f"{name}:{jid}: setup-python cache on a cold-install lane")
    assert not offenders, "a cold-install lane must not be warmed by a cache:\n  " + "\n  ".join(
        offenders
    )


def test_every_setup_python_step_still_pins_an_interpreter():
    """Each setup-python step must pin python-version, or the job silently runs the image's Python."""
    offenders = [
        f"{name}:{jid}"
        for name, jid, job in _jobs()
        for step in job.get("steps") or []
        if "setup-python" in _uses(step) and not (step.get("with") or {}).get("python-version")
    ]
    assert not offenders, f"setup-python without an explicit python-version: {offenders}"


def test_a_cache_save_of_downloaded_artifacts_waits_for_the_download_to_succeed():
    """Saves of downloaded artifacts must gate on the download's outcome, since always() stores partials."""
    offenders = []
    for name, jid, job in _jobs():
        if (name, jid) in PARTIAL_SAVE_JOBS:
            continue
        steps = job.get("steps") or []
        producers = {
            s.get("id")
            for s in steps
            if s.get("id") and re.search(r"install|download|build|prime", str(s.get("run", "")))
        }
        for step in steps:
            uses = _uses(step)
            if "actions/cache" not in uses or "/restore@" in uses:
                continue
            cond = str(step.get("if", ""))
            if "always()" not in cond:
                continue
            if not any(f"steps.{pid}.outcome" in cond for pid in producers if pid):
                offenders.append(f"{name}:{jid}: {step.get('name') or step.get('uses')}")
    assert not offenders, (
        "these cache saves run under always() without checking that the step which "
        "produced the payload succeeded, so a partial download can be stored under an "
        "immutable key and served to every later run:\n  " + "\n  ".join(offenders)
    )


def test_every_cache_key_path_resolves_where_the_job_checked_out():
    """Cache-key globs are resolved under the job's own checkout; a miss hashes empty and collapses keys."""
    offenders = []
    for name, jid, job in _jobs():
        steps = job.get("steps") or []
        # A path under another checked-out repository cannot be resolved here, so it is skipped. A
        # checkout with no `repository:` is this repo.
        own_prefixes, foreign_prefixes = [], []
        for s in steps:
            if "actions/checkout" not in _uses(s):
                continue
            with_ = s.get("with") or {}
            prefix = str(with_.get("path") or "").strip("/")
            repo = str(with_.get("repository") or "")
            (foreign_prefixes if repo and not repo.endswith("/unsloth") else own_prefixes).append(
                prefix
            )
        if not own_prefixes:
            own_prefixes = [""]

        for step in steps:
            with_ = step.get("with") or {}
            paths = with_.get("cache-dependency-path") or (
                with_.get("key-files") if "pip-cache-restore" in str(step.get("uses", "")) else None
            )
            for line in str(paths or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                if any(p and line.startswith(p + "/") for p in foreign_prefixes):
                    continue
                # Prefix must match a checkout of this repo and the rest must resolve to an existing file.
                relative = None
                for prefix in sorted(own_prefixes, key = len, reverse = True):
                    if not prefix:
                        relative = line
                        break
                    if line.startswith(prefix + "/"):
                        relative = line[len(prefix) + 1 :]
                        break
                if relative is None:
                    offenders.append(
                        f"{name}:{jid}: {line!r} is workspace-root-relative, but this job "
                        f"checks the repo out under {own_prefixes}"
                    )
                    continue
                if not list(REPO.glob(relative)):
                    offenders.append(
                        f"{name}:{jid}: {line!r} matches no file in the repo "
                        f"(resolved to {relative!r})"
                    )
    assert not offenders, (
        "these cache key paths are workspace-root-relative in a job that checks the repo "
        "out into a subdirectory, so they match no file:\n  " + "\n  ".join(offenders)
    )


def test_no_setup_python_step_declares_a_cache_path_without_a_cache():
    """A leftover cache-dependency-path is inert but reads as a cache that is not in force; remove it."""
    offenders = [
        f"{name}:{jid}"
        for name, jid, job in _jobs()
        for step in job.get("steps") or []
        if "setup-python" in _uses(step)
        and (step.get("with") or {}).get("cache-dependency-path")
        and not (step.get("with") or {}).get("cache")
    ]
    assert not offenders, (
        f"these steps declare cache-dependency-path but no cache, so the key is never "
        f"used and the config only misleads: {offenders}"
    )


def test_local_action_references_use_the_nested_checkout_path():
    """Local action paths resolve from GITHUB_WORKSPACE, so a nested checkout must prefix them."""
    offenders = []
    for name, jid, job in _jobs():
        steps = job.get("steps") or []
        checkout_dirs = [
            str((s.get("with") or {}).get("path")).rstrip("/")
            for s in steps
            if "actions/checkout" in _uses(s) and (s.get("with") or {}).get("path")
        ]
        if not checkout_dirs:
            continue
        for step in steps:
            uses = str(step.get("uses", ""))
            if uses.startswith("./") and not any(uses.startswith(f"./{d}/") for d in checkout_dirs):
                offenders.append(f"{name}:{jid}: {uses} (checkouts: {checkout_dirs})")
    assert not offenders, (
        "these local action references are workspace-root-relative in a job that checks "
        "the repo out into a subdirectory, so the runner cannot find the action:\n  "
        + "\n  ".join(offenders)
    )


# Playwright browser caches (~470 MB per engine). test_ui_shard_engines.py checks key tokens
# against shards; this catches two jobs storing identical engines under different keys.
# Derived from the workflows, so a new consumer with a novel token fails here.

_PW_EXPR = re.compile(r"\$\{\{\s*([A-Za-z_][A-Za-z0-9_.]*)\s*\}\}")
# `install-deps` cannot match: `install` must be followed by whitespace.
_PW_INSTALL = re.compile(r"playwright\s+install\s+([^\n|&;]+)")

PW_ENGINES = ("chromium", "firefox", "webkit")


def _uses(step):
    """Casefolded, since GitHub resolves owner/repo case-insensitively; Actions/Cache is actions/cache."""
    return str(step.get("uses", "")).casefold()


def _matrix_rows(job) -> list[dict]:
    """Expands base lists, then applies exclude and include in GitHub's order, as Actions does."""
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    if not isinstance(matrix, dict):
        return [{}]
    include = [r for r in (matrix.get("include") or []) if isinstance(r, dict)]
    exclude = [r for r in (matrix.get("exclude") or []) if isinstance(r, dict)]
    base_keys = [
        k for k, v in matrix.items() if k not in ("include", "exclude") and isinstance(v, list)
    ]

    combos = [{}]
    for k in base_keys:
        combos = [{**c, k: v} for c in combos for v in matrix[k]]

    # Exclude is processed before include, so include can add a combination back.
    def excluded(combo):
        return any(all(str(combo.get(k)) == str(v) for k, v in ex.items()) for ex in exclude)

    combos = [c for c in combos if not excluded(c)]

    rows, matched = [], set()
    for combo in combos:
        row = dict(combo)
        for i, inc in enumerate(include):
            # Mergeable when it overwrites nothing original; keys outside the base matrix are additions.
            if all(str(inc[k]) == str(combo[k]) for k in set(inc) & set(combo)):
                row.update(inc)
                matched.add(i)
        rows.append(row)
    rows += [inc for i, inc in enumerate(include) if i not in matched]
    if not rows:
        return [{}]
    return [{f"matrix.{k}": str(v) for k, v in row.items()} for row in rows]


def _resolve(text: str, row: dict) -> str:
    """Substitutes matrix.* only; runner.os and step outputs stay literal, so keys compare as strings."""
    return _PW_EXPR.sub(lambda m: row.get(m.group(1), m.group(0)), text)


_PRIMARY_KEY = re.compile(r"\$\{\{\s*steps\.([A-Za-z0-9_-]+)\.outputs\.cache-primary-key\s*\}\}")


def _forwarded_key(key: str, steps: list, row: dict) -> str:
    """Maps a forwarded cache-primary-key output to its restore key, so a save matches its restore."""
    m = _PRIMARY_KEY.fullmatch(key.strip())
    if not m:
        return key
    for step in steps:
        if step.get("id") == m.group(1):
            return _resolve(str((step.get("with") or {}).get("key", "")), row)
    return key


def _playwright_jobs():
    """(label, engines, restore_keys, save_keys) for every job that caches the engines."""
    for name, jid, job in _jobs():
        steps = job.get("steps") or []
        installs = [
            m.group(1) for step in steps for m in _PW_INSTALL.finditer(str(step.get("run", "")))
        ]
        cache_steps = [
            step
            for step in steps
            if "actions/cache" in _uses(step)
            and "ms-playwright" in str((step.get("with") or {}).get("path", ""))
        ]
        if not cache_steps:
            continue
        for row in _matrix_rows(job):
            engines = {
                word
                for spec in installs
                for word in _resolve(spec, row).split()
                if word in PW_ENGINES
            }
            restore, save = [], []
            for step in cache_steps:
                key = _resolve(str((step.get("with") or {}).get("key", "")), row)
                key = _forwarded_key(key, steps, row)
                # actions/cache folds the path into the entry version, so key and path both identify a cache.
                ident = (key, _resolve(str((step.get("with") or {}).get("path", "")), row))
                (restore if "/restore@" in str(step["uses"]) else save).append(ident)
            shard = row.get("matrix.shard") or row.get("matrix.engine_key")
            label = f"{name}:{jid}" + (f"[{shard}]" if shard else "")
            yield label, frozenset(engines), restore, save


def test_playwright_caches_key_the_same_engines_the_same_way():
    """Two Playwright cache keys naming the same engine set means a second copy of the same bytes."""
    by_engines, by_key = {}, {}
    for label, engines, restore, save in _playwright_jobs():
        assert engines, (
            f"{label} caches ~/.cache/ms-playwright but no step names an engine to "
            f"install, so this guard cannot tell what the entry holds"
        )
        for key in restore + save:
            by_engines.setdefault(engines, {}).setdefault(key, []).append(label)
            by_key.setdefault(key, {}).setdefault(engines, []).append(label)

    split = {
        " ".join(sorted(engines)): {f"{k} @ {pth}": sorted(set(v)) for (k, pth), v in keys.items()}
        for engines, keys in by_engines.items()
        if len(keys) > 1
    }
    assert not split, (
        "these engine sets are cached under more than one key, so each extra key is a "
        f"duplicate copy of the same browsers: {split}"
    )

    shared = {
        f"{key} @ {pth}": {" ".join(sorted(e)): sorted(set(v)) for e, v in engines.items()}
        for (key, pth), engines in by_key.items()
        if len(engines) > 1
    }
    assert not shared, (
        "these keys are used for more than one engine set, so a job can restore a hit "
        f"that is missing an engine it will try to launch: {shared}"
    )


def test_every_playwright_cache_saves_under_the_key_it_restored():
    """A save that drifts from its restore refills a key nothing reads, forever."""
    offenders = [
        f"{label}: restores {sorted(set(restore))}, saves {sorted(set(save))}"
        for label, _engines, restore, save in _playwright_jobs()
        if save and sorted(set(restore)) != sorted(set(save))
    ]  # identities are (key, path); a save matching on only one of the two is drift
    assert not offenders, (
        "these jobs save the Playwright engines under a key they did not restore:\n  "
        + "\n  ".join(offenders)
    )
