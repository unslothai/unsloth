# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every installing job must own its pip cache key, and no two may share one.

Five jobs did share one, and neither consequence showed up as red.

pip-cache-save is gated on `cache-hit != 'true'`, so whichever job finished
first on main wrote the cache and the other four restored it EXACTLY, installed
their own extra wheels, and never saved them -- re-downloading those on every
run of main, forever.

And a shared prefix cannot be ranked. cache-janitor.yml prunes by generation
within a prefix, and five live jobs under `pip-<os>-<arch>-py<ver>-` are
indistinguishable from five generations of one, so the family could not be
pruned at all: 57 stale entries on 2026-08-26, against 41.4 of 50 GiB.

`name` fixes both, and only stays fixed if nothing silently reuses one.
"""

import re
import pathlib

import pytest
import yaml


REPO = pathlib.Path(__file__).resolve().parents[2]
WORKFLOWS = sorted((REPO / ".github" / "workflows").glob("*.yml"))
RESTORE_SUFFIX = "actions/pip-cache-restore"
SAVE_SUFFIX = "actions/pip-cache-save"


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


def test_no_two_jobs_share_a_cache_name():
    seen = {}
    for path in WORKFLOWS:
        for job, steps in _jobs(path):
            for step in _restores(steps):
                name = (step.get("with") or {}).get("name") or ""
                where = f"{path.name}:{job}"
                assert name not in seen, (
                    f"cache name {name!r} is used by both {seen[name]} and {where}. "
                    f"A shared name is a shared key: only the first job to finish on "
                    f"main saves, the rest silently re-download their extras every run."
                )
                seen[name] = where


def test_jobs_sharing_key_files_still_have_distinct_names():
    # Jobs given the same key-files are the likeliest to be given the same name, which is the exact case that produced
    # the original defect.
    by_files = {}
    for path in WORKFLOWS:
        for job, steps in _jobs(path):
            for step in _restores(steps):
                with_ = step.get("with") or {}
                files = " ".join((with_.get("key-files") or "").split())
                by_files.setdefault(files, []).append((f"{path.name}:{job}", with_.get("name")))
    for files, entries in by_files.items():
        names = [n for _, n in entries]
        assert len(names) == len(set(names)), (
            f"{len(entries)} jobs share key-files {files!r} and reuse a name among "
            f"{names}; each needs its own so each can save its own wheels"
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
