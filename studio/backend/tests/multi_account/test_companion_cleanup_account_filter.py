# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Delete previews and orphan listings read only the repos the calling account may see."""

from dataclasses import dataclass

from hub.services.models import account_access as access
from hub.services.models import cache_inventory, companion_cleanup
from utils.account_context import AccountContext, OWNER, run_as

ALICE = AccountContext("a" * 32, "alice")


@dataclass(frozen = True)
class Scan:
    repos: frozenset


@dataclass(frozen = True)
class Repo:
    repo_id: str
    repo_type: str = "model"
    revisions: tuple = ()


def _repo(repo_id):
    return Repo(repo_id)


def test_managed_previews_see_only_visible_repos(monkeypatch):
    mine, theirs = _repo("alice/base"), _repo("bob/private-base")
    scans = [Scan(repos = frozenset({mine, theirs}))]
    monkeypatch.setattr(cache_inventory, "all_hf_cache_scans", lambda: scans)
    monkeypatch.setattr(
        access,
        "filter_model_rows",
        lambda rows, **_: [row for row in rows if row.repo_id.startswith("alice/")],
    )
    assert run_as(OWNER, companion_cleanup._account_scans) is scans
    filtered = run_as(ALICE, companion_cleanup._account_scans)
    assert [scan.repos for scan in filtered] == [frozenset({mine})]
    assert set(companion_cleanup._repos_by_id(filtered)) == {"alice/base"}
