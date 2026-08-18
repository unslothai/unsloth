# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""No workflow may spend the shared Actions cache budget carelessly.

This repo's Actions cache budget is 50 GiB (not GitHub's 10 GB default), and GitHub evicts
least-recently-used once it is exceeded. Measured before this file existed, unslothai/unsloth
held **49.63 GiB across 258 entries -- 99.3% full**, so eviction runs at the margin and every
new entry displaces an existing one. What fills it is almost entirely redundancy:

    20.74 GiB  duplicate waste: the SAME key held on several refs (42% of the cache)
                 6.67 GiB   13 copies  setup-python ... python-3.13.15-pip-85e247d7...
                 6.50 GiB   11 copies  setup-python ... python-3.11.15-pip-85e247d7...
                 3.83 GiB   10 copies  setup-python ... python-3.12.13-pip-85e247d7...
                 0.91 GiB    3 copies  ms-playwright-Linux-1.62.0-cfw-v1
    10.70 GiB  84 entries written and never read again, 10.60 GiB of it setup-python
    24.01 GiB  198 entries on PR refs, restorable only by re-runs of that same PR
     0.00 GiB  entries unread for 7+ days -- nothing is idle, the cache is churning

Every one of those duplicated keys already has a copy on `main`, which every PR can restore
from. The PR-scoped copies are therefore redundant by construction: they buy no hit rate and
evict the copy that does.

That is a self-reinforcing loop, and the repo had already diagnosed it once for the GGUF
caches (see the save step in studio-inference-smoke.yml: "PR misses -> downloads -> saves its
own copy -> evicts main's -> next PR misses"). It reappeared through two doors this file now
closes.

Both failure modes are silent. Nothing goes red when a cache is evicted; CI just quietly
re-downloads a 4.6 GB model and everyone assumes that is what it costs.

Door 1 -- saving on a PR ref. `actions/cache` (the read-write form) saves from its post-step
on every ref. A PR-scoped entry can never be read by anyone except re-runs of that same PR,
yet it competes for the budget against main's copy, which every PR *can* read. Saves belong
on main only, via `actions/cache/restore` plus a `github.ref == 'refs/heads/main'` save.

Door 2 -- `cache: 'pip'` on a job that installs almost nothing. `actions/setup-python`
derives one pip key per interpreter from dependency files across the whole repo, so dozens of
unrelated jobs share it and race to save under it. The entries measured 666-715 MB, and the
four interpreter keys between them account for 19.44 GiB of the 20.74 GiB of duplicate waste.
A job that only pip-installs `huggingface_hub` or `pytest` was paying that for the 0-7s its
restore step took. Jobs that really do install torch/transformers keep the cache; the rest do
not.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"

# Jobs whose pip cache earns its place: they install a torch/transformers-class dependency
# set, where the download genuinely dominates. Anything not listed here must not ask for it.
# The jobs that still use setup-python's built-in cache, and therefore still save on
# every ref. Listed rather than skipped silently: the check above would otherwise report
# a clean bill while nine jobs did exactly the thing it exists to prohibit, and a tenth
# could join them unnoticed. The follow-up converts these to an explicit
# restore + main-only save pair and empties this set.
PIP_CACHE_JOBS_PENDING_CONVERSION = {
    ("consolidated-tests-ci.yml", "consolidated"),
    ("consolidated-tests-ci.yml", "llama-cpp-smoke"),
    ("mlx-ci.yml", "dispatch"),
    ("notebooks-ci.yml", "api-introspect"),
    ("studio-backend-ci.yml", "pytest"),
    ("studio-backend-ci.yml", "repo-cpu-tests"),
    ("studio-export-capability-ci.yml", "capability"),
    ("version-compat-ci.yml", "zoo-imports-under-spoof"),
    ("version-compat-ci.yml", "grpo-fake-run"),
}

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


def _or_alternatives(expr: str) -> list[str]:
    """``expr`` split on its TOP-LEVEL ``||``, ignoring ``||`` inside parens or quotes."""
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
        elif ch == "|" and depth == 0 and expr[i : i + 2] == "||":
            parts.append("".join(buf))
            buf = []
            i += 2
            continue
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


# A POSITIVE equality against main, in either quote style. `!=` must not match: an
# expression restricting a save to everything EXCEPT main is the exact inversion of the
# rule, and a substring search for "refs/heads/main" accepts it.
_MAIN_ONLY = re.compile(r"github\.ref\s*==\s*['\"]refs/heads/main['\"]")


def _restricted_to_main(expr: str) -> bool:
    """Whether ``expr`` can only be true on ``refs/heads/main``.

    Every alternative of a top-level `||` has to carry the main check, because `||` is how
    a condition GAINS refs: `github.ref == 'refs/heads/main' || github.event_name ==
    'pull_request'` mentions main and runs on every PR. Requiring the check in each
    alternative is conservative -- it rejects some conditions that happen to be safe -- and
    that is the right direction for a guard whose failure mode is a silently refilled cache.

    Structural rather than a substring test because three shapes all contain the literal
    "refs/heads/main" while permitting PR saves: a `!=` comparison, an `||` that admits
    another event, and the check appearing only inside one branch of one.
    """
    alternatives = _or_alternatives(expr)
    return bool(expr.strip()) and all(_MAIN_ONLY.search(a) for a in alternatives)


@pytest.mark.parametrize(
    "expr,restricted",
    [
        ("always() && github.ref == 'refs/heads/main'", True),
        ('always() && github.ref == "refs/heads/main"', True),
        ("github.ref == 'refs/heads/main' && steps.x.outcome == 'success'", True),
        # The three shapes a substring test accepts and should not.
        ("github.ref != 'refs/heads/main'", False),
        ("github.ref == 'refs/heads/main' || github.event_name == 'pull_request'", False),
        ("(github.ref == 'refs/heads/main' && always()) || github.event_name == 'push'", False),
        # Both alternatives restricted is still restricted.
        (
            "(github.ref == 'refs/heads/main' && always()) || "
            "(github.ref == 'refs/heads/main' && failure())",
            True,
        ),
        # A `||` inside a string or parenthesised sub-expression is not a top-level split.
        ("github.ref == 'refs/heads/main' && contains(x, 'a||b')", True),
        ("", False),
    ],
)
def test_the_main_only_expression_check_reads_the_expression(expr, restricted):
    """The guard below is only as good as this predicate, so the predicate is tested too."""
    assert _restricted_to_main(expr) is restricted, expr


def test_no_workflow_saves_a_cache_on_a_pull_request_ref():
    offenders = []
    for name, jid, job in _jobs():
        for step in job.get("steps") or []:
            uses = str(step.get("uses", ""))
            # setup-python's `cache:` is a save too, and an invisible one: the action
            # registers a post-step (`post: dist/cache-save/index.js` in its own
            # action.yml) that runs after the job on whatever ref it ran on, with no
            # condition to gate it. A scan that only looked for `actions/cache` steps read
            # as green while nine jobs wrote PR-scoped entries every run.
            if "setup-python" in uses and (step.get("with") or {}).get("cache"):
                if (name, jid) not in PIP_CACHE_JOBS_PENDING_CONVERSION:
                    offenders.append(f"{name}:{jid}: setup-python implicit post-step save")
                continue
            if "actions/cache" not in uses:
                continue
            saves = "/restore@" not in uses  # read-write and /save@ both write
            if not saves:
                continue
            if not _restricted_to_main(str(step.get("if", ""))):
                offenders.append(f"{name}:{jid}: {step.get('name') or uses}")
    assert not offenders, (
        "these steps save a cache on whatever ref they run on, so every PR writes its own "
        "copy and evicts the copy on main that all PRs share:\n  " + "\n  ".join(offenders)
    )


def test_only_jobs_that_install_heavy_dependencies_ask_for_the_pip_cache():
    asking = {
        (name, jid)
        for name, jid, job in _jobs()
        for step in job.get("steps") or []
        if "setup-python" in str(step.get("uses", ""))
        and (step.get("with") or {}).get("cache") == "pip"
    }
    extra = asking - PIP_CACHE_JOBS_PENDING_CONVERSION
    assert not extra, (
        f"these jobs ask for the shared pip cache without installing anything that justifies "
        f"a ~700MB entry: {sorted(extra)}. The setup-python pip key is one per interpreter "
        f"across the whole repo, so every extra claimant is another racer saving under it."
    )


@pytest.mark.parametrize("name,jid", sorted(PIP_CACHE_JOBS_PENDING_CONVERSION))
def test_every_allowed_pip_cache_scopes_its_key_to_what_it_installs(name, jid):
    """Without `cache-dependency-path`, setup-python hashes dependency files repo-wide.

    That is the second multiplier behind the 27.05 GiB: 16 distinct keys appeared in a
    week, because any requirements edit anywhere invalidates every interpreter's entry at
    once and orphans the old ones. Scoping the key to the files a job actually installs
    from -- or, for the jobs that pin their dependencies inline, to the workflow file that
    IS the dependency spec -- keeps an unrelated edit from costing ~700MB per interpreter.
    """
    job = _workflows_by_name()[name]["jobs"][jid]
    for step in job.get("steps") or []:
        with_ = step.get("with") or {}
        if "setup-python" not in str(step.get("uses", "")) or with_.get("cache") != "pip":
            continue
        assert with_.get("cache-dependency-path"), (
            f"{name}:{jid} caches pip without a cache-dependency-path, so its key is a hash "
            f"of dependency files across the whole repo and moves for reasons that have "
            f"nothing to do with what this job installs"
        )


def _workflows_by_name():
    return dict(_workflows())


@pytest.mark.parametrize("name,jid", sorted(PIP_CACHE_JOBS_PENDING_CONVERSION))
def test_every_allowed_pip_cache_job_still_exists_and_still_earns_it(name, jid):
    """The allowlist must not outlive the jobs, or it silently permits nothing."""
    doc = dict(_workflows()).get(name)
    assert (
        doc is not None
    ), f"{name} no longer exists; drop it from PIP_CACHE_JOBS_PENDING_CONVERSION"
    job = doc["jobs"].get(jid)
    assert (
        job is not None
    ), f"{name} no longer has job {jid}; drop it from PIP_CACHE_JOBS_PENDING_CONVERSION"
    body = "\n".join(str(s.get("run", "")) for s in job.get("steps") or [])
    assert HEAVY.search(body), (
        f"{name}:{jid} is allowed a pip cache but no longer installs anything heavy; it "
        f"should give the budget back"
    )


def test_the_cold_install_lanes_never_restore_a_cache():
    """These workflows exist to prove a cold install works. A warm one proves nothing.

    They would still pass with a cache in front of them, which is exactly why this is
    asserted rather than left to review.
    """
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
            uses = str(step.get("uses", ""))
            if "actions/cache" in uses:
                offenders.append(f"{name}:{jid}: {step.get('name') or uses}")
            if "setup-python" in uses and (step.get("with") or {}).get("cache"):
                offenders.append(f"{name}:{jid}: setup-python cache on a cold-install lane")
    assert not offenders, "a cold-install lane must not be warmed by a cache:\n  " + "\n  ".join(
        offenders
    )


def test_every_setup_python_step_still_pins_an_interpreter():
    """Guards the edit that produced this file.

    Removing `cache: 'pip'` from an inline-flow mapping (`with: { python-version: '3.12',
    cache: 'pip' }`) by deleting the line takes the interpreter pin with it, and the job then
    silently runs on whatever Python the image happens to ship.
    """
    offenders = [
        f"{name}:{jid}"
        for name, jid, job in _jobs()
        for step in job.get("steps") or []
        if "setup-python" in str(step.get("uses", ""))
        and not (step.get("with") or {}).get("python-version")
    ]
    assert not offenders, f"setup-python without an explicit python-version: {offenders}"


def test_a_cache_save_of_downloaded_artifacts_waits_for_the_download_to_succeed():
    """A partial download saved under an immutable key poisons every later run.

    `playwright install` fetches three engines from a CDN. If it fails part-way, the
    directory still exists with some of them in it, and a save gated only on `always()`
    stores that. The key is pinned to the resolved Playwright version, so it does not roll
    over: every subsequent run restores the partial tree, sees `cache-hit == 'true'`, runs
    only `install-deps`, and drives a browser that was never downloaded. The UI jobs fail
    until somebody deletes the entry by hand, and nothing in the log says cache.

    So a save step whose payload is produced by an earlier step must check that step's
    outcome. `always()` on its own is the bug, not the fix.
    """
    offenders = []
    for name, jid, job in _jobs():
        steps = job.get("steps") or []
        producers = {
            s.get("id")
            for s in steps
            if s.get("id") and re.search(r"install|download|build|prime", str(s.get("run", "")))
        }
        for step in steps:
            uses = str(step.get("uses", ""))
            if "actions/cache" not in uses or "/restore@" in uses:
                continue
            cond = str(step.get("if", ""))
            if "always()" not in cond:
                continue  # not force-run, so a failed producer already skips it
            if not any(f"steps.{pid}.outcome" in cond for pid in producers if pid):
                offenders.append(f"{name}:{jid}: {step.get('name') or uses}")
    assert not offenders, (
        "these cache saves run under always() without checking that the step which "
        "produced the payload succeeded, so a partial download can be stored under an "
        "immutable key and served to every later run:\n  " + "\n  ".join(offenders)
    )


def test_every_cache_dependency_path_resolves_where_the_job_checked_out():
    """setup-python FAILS the job when the path matches nothing; it does not skip the cache.

    "Error: No file in <workspace> matched to [...], make sure you have checked out the
    target repository" -- actions/setup-python#807 is an open request to downgrade that to
    a warning, so today it is fatal. Three jobs here check the repo out under `unsloth/`
    (notebooks-ci api-introspect, version-compat-ci zoo-imports-under-spoof and
    grpo-fake-run) because they need a second repo beside it. A key written as if the
    checkout were at the workspace root therefore does not merely miss the cache, it stops
    the job before it installs anything.

    Scoping the keys is what introduced this, which is why it is asserted rather than
    remembered: the failure is loud, but only once it reaches CI.

    Each entry is resolved against the checkout it belongs to and then globbed, rather than
    prefix-matched. A prefix check calls `unsloth/.github/workflows/typo.yml` correct
    because it starts with `unsloth/`, and a job checked out at the workspace root was
    skipped entirely, so a misspelling there was never examined at all. Both fail the job
    exactly as hard as the wrong prefix does.
    """
    offenders = []
    for name, jid, job in _jobs():
        steps = job.get("steps") or []
        # Where THIS repo lands, which is not the same question as "is there a `path:`".
        # notebooks-ci api-introspect checks out two repositories side by side, and a path
        # under the OTHER one cannot be resolved against this tree at all, so it is skipped
        # rather than reported. A checkout with no `repository:` is this repo by definition.
        own_prefixes, foreign_prefixes = [], []
        for s in steps:
            if "actions/checkout" not in str(s.get("uses", "")):
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
            if "setup-python" not in str(step.get("uses", "")):
                continue
            for line in str(with_.get("cache-dependency-path") or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                if any(p and line.startswith(p + "/") for p in foreign_prefixes):
                    continue
                # The prefix has to match a checkout of this repo, AND what remains has to
                # resolve to a file that exists. Checking only the prefix accepted
                # `unsloth/.github/workflows/typo.yml`, which fails the job just as hard.
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
        "setup-python fails a job outright when cache-dependency-path matches nothing, so "
        "each of these stops its job before it installs anything:\n  " + "\n  ".join(offenders)
    )


def test_no_setup_python_step_declares_a_cache_path_without_a_cache():
    """Dead config reads as a caching decision that is not in force.

    Removing `cache: 'pip'` and leaving `cache-dependency-path` behind is inert -- the
    action only reads the path inside its `if (cache && isCacheFeatureAvailable())` branch
    -- but the next reader sees a scoped cache key and believes the job is cached.
    """
    offenders = [
        f"{name}:{jid}"
        for name, jid, job in _jobs()
        for step in job.get("steps") or []
        if "setup-python" in str(step.get("uses", ""))
        and (step.get("with") or {}).get("cache-dependency-path")
        and not (step.get("with") or {}).get("cache")
    ]
    assert not offenders, (
        f"these steps declare cache-dependency-path but no cache, so the key is never "
        f"used and the config only misleads: {offenders}"
    )
