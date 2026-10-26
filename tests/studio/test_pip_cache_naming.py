# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Each installing job needs its own pip cache key; sharing one leaves jobs unsaved and unprunable."""

import re
import pathlib

import pytest
import yaml


REPO = pathlib.Path(__file__).resolve().parents[2]
WORKFLOWS = sorted((REPO / ".github" / "workflows").glob("*.yml"))
RESTORE_SUFFIX = "actions/pip-cache-restore"
SAVE_SUFFIX = "actions/pip-cache-save"

# Jobs that restore another job's entry and never save. Only safe when the install
# is identical, so a borrower's extras cannot exist; consolidated-zoo runs the same
# core-cpu-setup on the same matrix, asserted by test_zoo_suite_parallel_isolation.py.
READ_ONLY_BORROWERS = {"consolidated-tests-ci.yml:consolidated-zoo": "consolidated"}


def _jobs(path):
    doc = yaml.safe_load(path.read_text(encoding = "utf-8")) or {}
    for name, job in (doc.get("jobs") or {}).items():
        yield name, (job.get("steps") or [])


def _restores(steps):
    # Suffix match: jobs checking out into a subdirectory say ./unsloth/.github/...
    return [s for s in steps if str(s.get("uses", "")).endswith(RESTORE_SUFFIX)]


def test_call_sites_exist():
    # A suffix that matches nothing would make every assertion below vacuous.
    found = sum(len(_restores(steps)) for p in WORKFLOWS for _, steps in _jobs(p))
    assert found >= 10, f"only {found} pip-cache-restore call sites found; this test is stale"


@pytest.mark.parametrize("path", WORKFLOWS, ids = lambda p: p.name)
def test_no_builtin_setup_python_pip_cache(path):
    for job, steps in _jobs(path):
        for step in steps:
            if not str(step.get("uses", "")).startswith("actions/setup-python@"):
                continue
            assert "cache" not in (step.get("with") or {}), (
                f"{path.name}:{job} uses setup-python's built-in cache, which saves on "
                f"whatever ref the job ran on. Use the pip-cache-restore / -save pair."
            )


@pytest.mark.parametrize("path", WORKFLOWS, ids = lambda p: p.name)
def test_every_restore_names_itself(path):
    for job, steps in _jobs(path):
        for step in _restores(steps):
            name = (step.get("with") or {}).get("name", "")
            assert re.fullmatch(r"[a-z0-9-]+", name or ""), (
                f"{path.name}:{job} restore name={name!r} must be lowercase letters, "
                f"digits and dashes; it goes into the cache key verbatim."
            )
            files = [f for f in ((step.get("with") or {}).get("key-files") or "").split() if f]
            assert files, f"{path.name}:{job} passes no key-files"


def _saves(steps):
    return [s for s in steps if str(s.get("uses", "")).endswith(SAVE_SUFFIX)]


def _key_files(steps):
    return " ".join(((_restores(steps)[0].get("with") or {}).get("key-files") or "").split())


def test_no_two_jobs_share_a_cache_name():
    seen = {}
    for path in WORKFLOWS:
        for job, steps in _jobs(path):
            for step in _restores(steps):
                name = (step.get("with") or {}).get("name") or ""
                where = f"{path.name}:{job}"
                if READ_ONLY_BORROWERS.get(where) == name:
                    continue
                assert name not in seen, (
                    f"cache name {name!r} is used by both {seen[name]} and {where}. "
                    f"A shared name is a shared key: only the first job to finish on "
                    f"main saves, the rest silently re-download their extras every run."
                )
                seen[name] = where


def test_jobs_sharing_key_files_still_have_distinct_names():
    by_files = {}
    for path in WORKFLOWS:
        for job, steps in _jobs(path):
            where = f"{path.name}:{job}"
            for step in _restores(steps):
                with_ = step.get("with") or {}
                if READ_ONLY_BORROWERS.get(where) == with_.get("name"):
                    continue
                files = " ".join((with_.get("key-files") or "").split())
                by_files.setdefault(files, []).append((where, with_.get("name")))
    for files, entries in by_files.items():
        names = [n for _, n in entries]
        assert len(names) == len(set(names)), (
            f"{len(entries)} jobs share key-files {files!r} and reuse a name among "
            f"{names}; each needs its own so each can save its own wheels"
        )


@pytest.mark.parametrize("where,owner", sorted(READ_ONLY_BORROWERS.items()))
def test_read_only_borrowers_never_save_and_their_owner_does(where, owner):
    jobs = {f"{p.name}:{j}": steps for p in WORKFLOWS for j, steps in _jobs(p)}
    assert where in jobs, f"{where} no longer exists; drop it from READ_ONLY_BORROWERS"
    restores = _restores(jobs[where])
    assert [(s.get("with") or {}).get("name") for s in restores] == [
        owner
    ], f"{where} must restore exactly one pip cache, {owner!r}'s"
    assert not _saves(jobs[where]), (
        f"{where} borrows {owner!r}'s entry, so it must not save: two writers of one key "
        f"race, and the loser's wheels are never written"
    )
    owners = [
        w
        for w, steps in jobs.items()
        if w != where
        and _saves(steps)
        and any((s.get("with") or {}).get("name") == owner for s in _restores(steps))
    ]
    assert len(owners) == 1, f"{owner!r} needs exactly one saving job, found {owners}"
    assert _key_files(jobs[where]) == _key_files(jobs[owners[0]]), (
        f"{where} keys on different files than {owners[0]}, so it would look up a key "
        f"its owner never writes"
    )


def test_key_is_versioned_and_carries_the_name():
    action = (REPO / ".github/actions/pip-cache-restore/action.yml").read_text(encoding = "utf-8")
    assert 'prefix="pip-v2-${name}-' in action, (
        "the key must stay pip-v2-<name>-... . The v2 segment is what lets "
        "cache-janitor.yml match only post-rename keys: 'Linux' is a valid name, so "
        "a legacy pip-<os>-... key is otherwise indistinguishable from a new one."
    )


def test_janitor_matches_v2_keys_only():
    janitor = (REPO / ".github/workflows/cache-janitor.yml").read_text(encoding = "utf-8")
    assert (
        "pip-v2-*|uv-*|fe-dist-*)" in janitor
    ), "janitor no longer ranks the pip/uv/fe-dist families"
    assert "\n              pip-*)" not in janitor, (
        "the janitor must not match bare pip-* : legacy keys from before `name` "
        "existed would then be ranked against unrelated jobs' keys and deleted."
    )


def test_save_runs_on_the_default_branch_only():
    action = yaml.safe_load(
        (REPO / ".github/actions/pip-cache-save/action.yml").read_text(encoding = "utf-8")
    )
    condition = " ".join(str(action["runs"]["steps"][0].get("if", "")).split())
    assert "github.ref == 'refs/heads/main'" in condition
    assert "always()" in condition
