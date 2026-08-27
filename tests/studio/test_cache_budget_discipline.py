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
#
# These nine no longer use setup-python's built-in `cache: 'pip'`. That form is read-write
# and saves from its post-step on whatever ref the job ran on, with no knob to gate it, so
# every PR wrote a ~700MB entry only its own re-runs could ever restore while evicting the
# copy on main that all PRs share. Measured at 19.45 GiB across 40 entries, 15.49 GiB of it
# on PR refs. They use the pip-cache-restore / pip-cache-save action pair instead, which
# splits the halves so the save can be gated on the default branch.
PIP_CACHE_JOBS = {
    ("consolidated-tests-ci.yml", "consolidated"),
    ("consolidated-tests-ci.yml", "llama-cpp-smoke"),
    ("mlx-ci.yml", "dispatch"),
    ("notebooks-ci.yml", "api-introspect"),
    # Installs a 709-line Colab pip-freeze, eight matrix legs at once, and each leg
    # downloads the identical set. It is cron and dispatch only, so none of that is
    # on a pull request's critical path -- but eight ubuntu runners holding a 25
    # minute cap contend for the same pool every other job queues against.
    ("notebooks-ci.yml", "smoke-install"),
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


def _composite_actions():
    """(name, steps) for every composite action in the repo.

    Scanned because the pip cache save now lives in one. A guard that reads only workflow
    steps would have gone blind to it the moment the logic was factored out, which is the
    failure mode where a rule quietly stops applying to the thing it was written for.
    """
    for f in sorted((REPO / ".github" / "actions").rglob("action.yml")):
        doc = yaml.safe_load(f.read_text(encoding = "utf-8"))
        if isinstance(doc, dict):
            yield f.parent.name, ((doc.get("runs") or {}).get("steps") or [])


def test_no_workflow_saves_a_cache_on_a_pull_request_ref():
    offenders = []
    for name, steps in _composite_actions():
        for step in steps:
            uses = str(step.get("uses", ""))
            if "actions/cache" not in uses or "/restore@" in uses:
                continue
            if "refs/heads/main" not in str(step.get("if", "")):
                offenders.append(f"action {name}: {step.get('name') or uses}")
    for name, jid, job in _jobs():
        for step in job.get("steps") or []:
            uses = str(step.get("uses", ""))
            # setup-python's `cache:` is a save too, and an invisible one: the action
            # registers a post-step (`post: dist/cache-save/index.js` in its own
            # action.yml) that runs after the job on whatever ref it ran on, with no
            # condition to gate it. A scan that only looked for `actions/cache` steps
            # read as green while nine jobs wrote PR-scoped entries every run. Nothing
            # is exempt now that all nine are converted.
            if "setup-python" in uses and (step.get("with") or {}).get("cache"):
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


def test_no_job_uses_setup_pythons_built_in_pip_cache():
    """The built-in cache cannot be gated, so it is not used here at all any more.

    `actions/setup-python`'s `cache: 'pip'` is the read-write form: it restores in the step
    and saves from its own post-step, on whatever ref the job ran on, and exposes no
    condition to stop that. A PR-ref entry is restorable only by re-runs of that same PR, so
    it buys no hit rate while competing for the shared 50 GiB budget against main's copy,
    which every PR can read. Use pip-cache-restore plus pip-cache-save instead.
    """
    offenders = [
        f"{name}:{jid}"
        for name, jid, job in _jobs()
        for step in job.get("steps") or []
        if "setup-python" in str(step.get("uses", "")) and (step.get("with") or {}).get("cache")
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
    """The allowlist has to be enforced against what the workflows DO, not iterated over.

    Every other check in this file is parametrized over PIP_CACHE_JOBS, which means a new
    job that adds the restore/save pair is simply never visited: it gets a ~700MB entry
    with no scoping check, no wiring check and no justification, and this file stays green.

    That hole opened when the built-in `cache: 'pip'` went away. The previous guard
    discovered claimants by scanning for setup-python's `cache:` key, so replacing that
    mechanism removed the discovery along with it, leaving nine hardcoded names and nothing
    watching for a tenth.
    """
    extra = _pip_cache_users() - PIP_CACHE_JOBS
    assert not extra, (
        f"these jobs use the pip cache without being listed in PIP_CACHE_JOBS: "
        f"{sorted(extra)}. Every entry competes for the shared 50 GiB budget, so a job "
        f"earns one by installing a torch/transformers-class dependency set where the "
        f"download dominates. Add it to the allowlist with that justification, or drop the "
        f"cache."
    )


def test_every_pip_cache_user_actually_installs_something_heavy():
    """The allowlist records a judgement; this checks the judgement still matches the job.

    A job whose heavy install is later moved elsewhere keeps its cache entry, and nothing
    else in this file would notice: the name stays in the list and every parametrized check
    still passes.
    """
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
    """Without scoping, the key is a hash of dependency files repo-wide.

    That is the second multiplier behind the 19.45 GiB: 16 distinct keys appeared in a
    week, because any requirements edit anywhere invalidates every interpreter's entry at
    once and orphans the old ones. Scoping the key to the files a job actually installs
    from -- or, for the jobs that pin their dependencies inline, to the workflow file that
    IS the dependency spec -- keeps an unrelated edit from costing ~700MB per interpreter.
    """
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
    """A restore with no save fills nothing; a save reading the wrong ids saves nothing.

    Both halves are silent when wrong. The save takes the directory, the key and the
    hit flag from the restore step's outputs, so a renamed or missing id yields empty
    inputs and an entry that is never written, with a green job either way.
    """
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
    saves = [s for s in steps if "actions/cache" in str(s.get("uses", ""))]
    assert saves, "pip-cache-save no longer saves anything"
    for s in saves:
        cond = str(s.get("if", ""))
        assert "refs/heads/main" in cond, (
            "the pip cache save is no longer gated on the default branch, so all nine call "
            "sites went back to writing PR-scoped entries at once"
        )


def test_the_pip_cache_key_carries_the_interpreter_minor_not_its_patch():
    """
    A key field more specific than anything the workflows request is pure churn.

    No step in this repo pins a patch version: 53 ask for '3.12' and the one matrix
    offers '3.11' and '3.13'. So the patch is whatever the hosted image ships that
    week, and putting it in the key duplicates the ENTIRE cache each time GitHub
    bumps it. Measured 2026-08-20: two entries alike in everything but 3.12.13 vs
    3.12.14 held 10.85 and 11.21 GiB, 44% of the 50 GiB budget between them, with the
    same pairing in 10 of the 12 pip entries.

    Nothing goes red when that happens. The cache simply sits at 99% full and evicts
    entries someone else was about to read, which is the failure this whole file is
    about.
    """
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


def test_every_cache_key_path_resolves_where_the_job_checked_out():
    """A key-files glob that matches nothing collapses every job onto one key.

    Three jobs check the repo out under `unsloth/` because they need a second repo beside
    it (notebooks-ci api-introspect, version-compat-ci zoo-imports-under-spoof and
    grpo-fake-run), so a path written as if the checkout were at the workspace root
    resolves to nothing. Under setup-python's built-in cache that was fatal outright
    ("No file in ... matched to ..."); hashFiles is quieter and simply returns empty, which
    is why pip-cache-restore fails loudly on an empty hash and why this stays asserted.

    Each entry is resolved against the checkout it belongs to and then globbed, rather than
    prefix-matched. A prefix check calls `unsloth/.github/workflows/typo.yml` correct
    because it starts with `unsloth/`, and a job checked out at the workspace root was
    skipped entirely, so a misspelling there was never examined at all.
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
            paths = with_.get("cache-dependency-path") or (
                with_.get("key-files") if "pip-cache-restore" in str(step.get("uses", "")) else None
            )
            for line in str(paths or "").splitlines():
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
        "these cache key paths are workspace-root-relative in a job that checks the repo "
        "out into a subdirectory, so they match no file:\n  " + "\n  ".join(offenders)
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


def test_local_action_references_use_the_nested_checkout_path():
    """`uses: ./...` resolves from GITHUB_WORKSPACE, not from the workflow file.

    GitHub's own docs put it plainly: if the action checks the repository out to a
    different location than the workflow, the relative path for a local action has to be
    updated. Three jobs here check out under `unsloth/` because they need a second repo
    beside it, so an unprefixed `./.github/actions/...` points at a directory that does not
    exist and the step fails with "Can't find 'action.yml', 'action.yaml' or 'Dockerfile'".

    Same root cause as the cache-key path check above, one level out: the key paths were
    fixed for these jobs and the action paths were not.
    """
    offenders = []
    for name, jid, job in _jobs():
        steps = job.get("steps") or []
        checkout_dirs = [
            str((s.get("with") or {}).get("path")).rstrip("/")
            for s in steps
            if "actions/checkout" in str(s.get("uses", "")) and (s.get("with") or {}).get("path")
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
