# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""
The uv download cache must stay a download cache, and must never reach a
cold-install lane.

`Install Unsloth (--local, --no-torch)` is the single largest cost in CI: 92s
median across 39 job runs in one sample, more total time than any test, and all
of it uv re-downloading the same wheels because its cache is per-runner.

Caching that is safe *because of what is cached*. uv's cache is content-addressed
by URL and hash, so a stale entry cannot serve wrong content -- the worst it can
do is miss. That property is the whole justification, and it is exactly what a
later edit could take away by pointing the same cache config at the venv, or at
`~/.unsloth`, where an editable overlay, a moving `unsloth-zoo @ git+main` and
absolute paths in console scripts all live. These tests pin the distinction.

The second invariant is the one with teeth. `clean-machine-install-ci.yml` and
`desktop-app-clean-machine-ci.yml` exist to prove the installer works on a
machine with nothing on it; both set their own `UV_CACHE_DIR` and delete it
before running. If either ever adopted this action, the composite writes
`UV_CACHE_DIR` to `$GITHUB_ENV`, which outranks a job-level `env:` for every
later step -- so a warm cache would silently replace the cold machine those
workflows are named after, and they would still go green.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTION = REPO_ROOT / ".github" / "actions" / "install-unsloth-local" / "action.yml"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# Named, not detected: a lane whose whole point is a cold machine should have to be removed from this list deliberately,
# in a diff someone reads.
COLD_INSTALL_WORKFLOWS = (
    "clean-machine-install-ci.yml",
    "desktop-app-clean-machine-ci.yml",
    "interrupted-install-ci.yml",
    # Publishes the desktop app from a clean checkout; a restored dist would ship a
    # bundle this run never built.
    "release-desktop.yml",
)


def _own_steps(action: Path = ACTION) -> list[dict]:
    return yaml.safe_load(action.read_text(encoding = "utf-8"))["runs"]["steps"]


def _steps() -> list[dict]:
    """This action's steps, with the steps of every local action it delegates to inlined.

    The frontend dist cache moved into `.github/actions/frontend-dist-restore` and
    `-save` when the Windows jobs adopted it, because they do not go through this action
    and the key must have exactly one definition. A reader that only walked this file's
    own steps would have gone blind to that cache the moment it was factored out -- and
    silently, since `uses: ./.github/actions/frontend-dist-restore` does not contain the
    substring `actions/cache` that the path check below looks for. Every assertion in
    this file would have kept passing while guarding one cache instead of two.

    That is the same failure mode test_cache_budget_discipline.py's `_composite_actions`
    was written for, one level in: a rule quietly stops applying to the thing it was
    written for.
    """
    flat: list[dict] = []
    for step in _own_steps():
        uses = str(step.get("uses", ""))
        local = re.match(r"\./\.github/actions/([\w-]+)$", uses)
        if local:
            nested = REPO_ROOT / ".github" / "actions" / local.group(1) / "action.yml"
            assert nested.is_file(), f"{uses} does not exist, so this action cannot run"
            flat.extend(_own_steps(nested))
        else:
            flat.append(step)
    return flat


def _index_of(predicate) -> int:
    for i, step in enumerate(_steps()):
        if predicate(step):
            return i
    return -1


# What this action is allowed to cache, and the argument for each. Anything else has to be added here in a diff
# someone reads, with its own argument written down.
#
#   .uv-cache            uv's download cache. Content-addressed by URL and hash, so a stale entry cannot serve wrong
#                        content; the worst it can do is miss.
#   studio/frontend/dist the built frontend. NOT a download, so it does not get the argument above and needs its own:
#                        it is a directory of static assets with no absolute paths, no interpreter coupling and no
#                        console scripts, which is exactly what makes a venv unsafe to cache and this safe. Its key
#                        hashes the same inputs studio/setup.sh checks before rebuilding, so a hit means the build
#                        inputs are byte-identical rather than merely similar.
#                        tests/studio/test_frontend_dist_cache.py holds that agreement together and is where the
#                        reasoning lives.
CACHEABLE_PATHS = (".uv-cache", "studio/frontend/dist")


def test_the_cache_holds_downloads_and_build_output_but_never_the_venv() -> None:
    """
    A venv cache would have to reason about the editable overlay, a moving
    unsloth-zoo pin, and absolute paths in console scripts. Neither of the two
    things this action caches reasons about any of that, which is why they are safe
    at all. The forbidden list below is the invariant with teeth and applies to
    every cache step regardless of which allowed path it uses.
    """
    for step in _steps():
        if "cache" not in str(step.get("uses", "")):
            continue
        path = str((step.get("with") or {}).get("path", ""))
        assert any(allowed in path for allowed in CACHEABLE_PATHS), (
            f"cache step points at {path!r}, which is not one of the paths this action "
            f"is allowed to cache ({', '.join(CACHEABLE_PATHS)}). Add it to "
            f"CACHEABLE_PATHS with the argument for why restoring it cannot be wrong."
        )
        for forbidden in (".unsloth", "site-packages", "unsloth_studio", "venv"):
            assert forbidden not in path, (
                f"cache step points at {path!r}, which is an INSTALL, not a download "
                f"cache. A restored install can be wrong; a restored download cannot."
            )


def test_uv_cache_dir_is_set_before_the_install_runs() -> None:
    """Set afterwards it configures nothing, and the step would still look right."""
    setter = _index_of(lambda s: "UV_CACHE_DIR" in str(s.get("run", "")))
    install = _index_of(lambda s: "install.sh --local --no-torch" in str(s.get("run", "")))
    assert setter != -1, "the action no longer points UV_CACHE_DIR anywhere"
    assert install != -1, "the action no longer runs the local install"
    assert setter < install, (
        "UV_CACHE_DIR is set after the install, so the install used uv's default "
        "cache and the restored one was never read"
    )


def test_the_restore_happens_before_the_install_too() -> None:
    restore = _index_of(lambda s: "cache/restore" in str(s.get("uses", "")))
    install = _index_of(lambda s: "install.sh --local --no-torch" in str(s.get("run", "")))
    assert restore != -1 and restore < install


def test_a_near_miss_still_supplies_most_wheels() -> None:
    """
    restore-keys is what makes this worth having on a PR whose requirements moved
    by one line. It is correct here precisely because the entry is content-
    addressed; the same fallback on a venv cache would be a bug.
    """
    restore = next(s for s in _steps() if "cache/restore" in str(s.get("uses", "")))
    assert (restore.get("with") or {}).get("restore-keys"), (
        "no restore-keys, so any change to requirements or pyproject drops the "
        "cache to zero instead of to almost-full"
    )


def test_the_cache_is_saved_on_main_only() -> None:
    """
    A PR-scoped entry can only be restored by re-runs of that same PR, while every
    PR can restore from the default branch. Saving on PRs spends a budget measured
    at 99.3% full once already, and evicts main's copy -- the one everyone reads.
    """
    saves = [s for s in _steps() if "cache/save" in str(s.get("uses", ""))]
    assert saves, "the cache is never saved, so it can never be restored either"
    for step in saves:
        condition = str(step.get("if", ""))
        assert (
            "refs/heads/main" in condition
        ), f"a cache/save step is not gated on main: if: {condition!r}"


@pytest.mark.parametrize("name", COLD_INSTALL_WORKFLOWS)
def test_cold_install_lanes_never_adopt_this_action(name: str) -> None:
    """
    These prove the installer works on a machine with nothing on it. The composite
    writes UV_CACHE_DIR to $GITHUB_ENV, which outranks a job-level `env:` for every
    later step, so adopting it would hand a cold lane a warm cache and the lane
    would still report success.
    """
    path = WORKFLOWS / name
    if not path.exists():
        pytest.skip(f"{name} no longer exists")
    text = path.read_text(encoding = "utf-8")
    assert "install-unsloth-local" not in text, (
        f"{name} uses install-unsloth-local, which warms uv's cache. A cached "
        f"cold-install test proves nothing and still goes green."
    )
    # Named separately because the frontend dist cache can now be adopted WITHOUT this action -- that is the whole
    # point of splitting it out for the Windows jobs, which call install.ps1 from a hand-written step. Checking only
    # for install-unsloth-local would let a cold lane paste in the two `uses:` lines and stay green.
    assert "frontend-dist-" not in text, (
        f"{name} restores a prebuilt frontend. A cold-install lane handed a bundle built "
        f"on another machine last week is not testing a cold install."
    )


def test_the_action_is_actually_used() -> None:
    """Otherwise every assertion above guards something nothing runs."""
    users = [
        p.name
        for p in WORKFLOWS.glob("*.yml")
        if "install-unsloth-local" in p.read_text(encoding = "utf-8")
    ]
    assert len(users) >= 5, f"only {len(users)} workflows use the action: {users}"


# The Windows jobs run install.ps1 from a hand-written pwsh step and never come through
# install-unsloth-local, so the restore and the save are their own composite actions,
# which that composite delegates to and the Windows jobs call directly. What follows
# holds that shape: one key, every warm installer job restoring it, every restore paired
# with a save that reads its outputs, and the cold lanes untouched.

ACTIONS = REPO_ROOT / ".github" / "actions"
UV_RESTORE = ACTIONS / "uv-cache-restore" / "action.yml"
UV_SAVE = ACTIONS / "uv-cache-save" / "action.yml"

# Cold at JOB level, inside a workflow whose other jobs legitimately use the cache; the
# same list tests/studio/test_frontend_dist_cache.py keeps for the dist.
COLD_INSTALL_JOBS = (("studio-windows-inference-smoke.yml", "no-vs-cpu"),)

# An INVOCATION, not a mention: mlx-ci.yml explains in a comment why it does NOT run
# `install.sh --local`, and a bare substring match called that an uncached install.
_INSTALLER = re.compile(r"(?m)^\s*[^#\n]*?(?:^|[\s/&'\"])install\.(?:ps1|sh) --local")
_HELPER = re.compile(r"\.github/scripts/([A-Za-z0-9_.-]+\.sh)")


def _runs_installer(step: dict) -> bool:
    """Whether ``step`` runs the installer itself or through a helper under .github/scripts.

    studiobench-ui-parity installs each side through parity-install-side.sh, so the
    invocation is one file away from the workflow; followed one level, like the smoke
    trigger guard does for its helpers.
    """
    run = str(step.get("run", ""))
    if _INSTALLER.search(run):
        return True
    for helper in _HELPER.findall(run):
        path = REPO_ROOT / ".github" / "scripts" / helper
        if path.is_file() and _INSTALLER.search(path.read_text(encoding = "utf-8", errors = "replace")):
            return True
    return False


def _jobs():
    for f in sorted(WORKFLOWS.glob("*.yml")):
        doc = yaml.safe_load(f.read_text(encoding = "utf-8"))
        if isinstance(doc, dict) and isinstance(doc.get("jobs"), dict):
            for jid, job in doc["jobs"].items():
                if isinstance(job, dict):
                    yield f.name, jid, job


def _produces_on_main(name: str) -> bool:
    """Same rule as the dist guard: `push` to main or `schedule`, never `workflow_dispatch`."""
    doc = yaml.safe_load((WORKFLOWS / name).read_text(encoding = "utf-8"))
    on = doc.get("on", doc.get(True)) or {}
    if isinstance(on, str):
        on = {on: None}
    elif isinstance(on, list):
        on = dict.fromkeys(on)
    if "schedule" in on:
        return True
    if "push" not in on:
        return False
    push = on.get("push")
    branches = push.get("branches") if isinstance(push, dict) else None
    return branches is None or "main" in branches


def test_the_uv_cache_key_has_exactly_one_definition() -> None:
    """A second copy agrees today and drifts silently, with the cache still hitting."""
    definers = []
    for path in sorted(list(ACTIONS.rglob("action.yml")) + list(WORKFLOWS.glob("*.yml"))):
        if re.search(r"key:\s*uv-\$\{\{", path.read_text(encoding = "utf-8")):
            definers.append(str(path.relative_to(REPO_ROOT)))
    assert definers == [
        ".github/actions/uv-cache-restore/action.yml"
    ], f"the uv cache key is defined in {definers}; it must have exactly one definition"


def test_install_unsloth_local_delegates_the_uv_cache() -> None:
    uses = [str(s.get("uses", "")) for s in _own_steps()]
    assert "./.github/actions/uv-cache-restore" in uses, uses
    assert "./.github/actions/uv-cache-save" in uses, uses
    assert not any("actions/cache" in u for u in uses), (
        "install-unsloth-local carries its own cache step again; the key must stay in "
        "uv-cache-restore"
    )


def test_the_uv_actions_nest_nothing() -> None:
    """`uses: ./...` inside a composite resolves from GITHUB_WORKSPACE and takes no expressions.

    install-unsloth-local nests these two and is root-checkout-only for it. The leaf
    actions must stay leaves, so a nested-checkout job can still call them directly.
    """
    for action in (UV_RESTORE, UV_SAVE):
        nested = [
            str(s.get("uses", ""))
            for s in _own_steps(action)
            if str(s.get("uses", "")).startswith("./")
        ]
        assert not nested, f"{action.relative_to(REPO_ROOT)} nests {nested}"


def test_every_warm_installer_job_restores_the_uv_cache() -> None:
    """A job that runs the installer without the cache pays the full download every run.

    The cold lanes are the deliberate exception, named in COLD_INSTALL_WORKFLOWS and
    COLD_INSTALL_JOBS, and a new installer call site has to either restore the cache or
    be added to one of those lists in a diff someone reads.
    """
    offenders = []
    for name, jid, job in _jobs():
        if name in COLD_INSTALL_WORKFLOWS or (name, jid) in COLD_INSTALL_JOBS:
            continue
        steps = job.get("steps") or []
        runs_installer = any(_runs_installer(s) for s in steps)
        if not runs_installer:
            continue
        restores = any(
            "uv-cache-restore" in str(s.get("uses", ""))
            or "install-unsloth-local" in str(s.get("uses", ""))
            for s in steps
        )
        if not restores:
            offenders.append(f"{name}:{jid}")
    assert not offenders, (
        "these jobs run the installer with no uv download cache, so every run downloads "
        "every wheel again:\n  " + "\n  ".join(offenders)
    )


def test_every_uv_restore_is_paired_with_a_save_wired_to_it() -> None:
    """A restore with no save fills nothing; a save reading the wrong id saves nothing.

    The upload half is derived from the workflow's triggers rather than allowlisted, as
    for the dist: a consumer-only lane (no `push`, no `schedule`) passes `save: 'false'`
    and a producer must not.
    """
    offenders = []
    for name, jid, job in _jobs():
        steps = job.get("steps") or []
        restore = next((s for s in steps if "uv-cache-restore" in str(s.get("uses", ""))), None)
        save = next((s for s in steps if "uv-cache-save" in str(s.get("uses", ""))), None)
        if restore is None and save is None:
            continue
        if restore is None or save is None:
            offenders.append(f"{name}:{jid}: restore={restore is not None} save={save is not None}")
            continue
        ident = restore.get("id")
        if not ident:
            offenders.append(f"{name}:{jid}: the restore step has no id")
            continue
        with_ = save.get("with") or {}
        for field in ("cache-hit", "key"):
            if f"steps.{ident}.outputs.{field}" not in str(with_.get(field, "")):
                offenders.append(f"{name}:{jid}: save does not take {field} from steps.{ident}")
        installs = [i for i, s in enumerate(steps) if _runs_installer(s)]
        if not installs:
            offenders.append(f"{name}:{jid}: restores the uv cache but never runs an installer")
            continue
        if steps.index(restore) > min(installs):
            offenders.append(f"{name}:{jid}: the restore runs after the install")
        if steps.index(save) < max(installs):
            offenders.append(f"{name}:{jid}: the save runs before the install")
        uploads = str(with_.get("save", "true")) != "false"
        if uploads and not _produces_on_main(name):
            offenders.append(
                f"{name}:{jid}: saves, but {name} never runs on main; pass save: 'false'"
            )
        if not uploads and _produces_on_main(name):
            offenders.append(f"{name}:{jid}: passes save: 'false', but {name} runs on main")
    assert not offenders, "\n  ".join(["broken uv cache wiring:"] + offenders)


@pytest.mark.parametrize("name", COLD_INSTALL_WORKFLOWS)
def test_cold_install_lanes_never_adopt_the_uv_actions(name: str) -> None:
    path = WORKFLOWS / name
    if not path.exists():
        pytest.skip(f"{name} no longer exists")
    assert "uv-cache-" not in path.read_text(
        encoding = "utf-8"
    ), f"{name} restores a warm uv cache; a cached cold-install test proves nothing"


@pytest.mark.parametrize("name,jid", COLD_INSTALL_JOBS)
def test_cold_install_jobs_never_adopt_the_uv_actions(name: str, jid: str) -> None:
    doc = yaml.safe_load((WORKFLOWS / name).read_text(encoding = "utf-8"))
    job = (doc.get("jobs") or {}).get(jid)
    assert job is not None, f"{name} no longer has job {jid}; update COLD_INSTALL_JOBS"
    offenders = [
        str(s.get("uses", ""))
        for s in job.get("steps") or []
        if "uv-cache-" in str(s.get("uses", ""))
    ]
    assert not offenders, f"{name}:{jid} is a deliberate cold-install lane: {offenders}"


def test_the_uv_actions_are_actually_used_directly() -> None:
    """The Windows jobs are the reason the pair exists; if none calls it, it is dead code."""
    direct = [
        p.name
        for p in WORKFLOWS.glob("*.yml")
        if "uv-cache-restore" in p.read_text(encoding = "utf-8")
    ]
    assert (
        len(direct) >= 4
    ), f"only {len(direct)} workflows call uv-cache-restore directly: {direct}"
