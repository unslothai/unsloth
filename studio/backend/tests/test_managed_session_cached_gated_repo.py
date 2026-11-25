# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's UI session token is not cache-authorized for an ungranted gated repo."""

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from hub.services.models import account_access  # noqa: E402
from hub.utils import hf_tokens  # noqa: E402
from utils.account_context import AccountContext, run_as  # noqa: E402

BOB = AccountContext("b" * 32, "bob")


def test_this_tree_is_under_test():
    assert Path(hf_tokens.__file__).resolve().is_relative_to(BACKEND.resolve())


@pytest.fixture
def hub(monkeypatch):
    probes = []
    grants = set()

    def reaches(
        repo,
        token,
        repo_type,
        offline = False,
    ):
        probes.append((repo, token, repo_type))
        return False  # the account's own token is refused by the Hub

    monkeypatch.setattr(hf_tokens, "_explicit_token_reaches_repo", reaches)
    monkeypatch.setattr(account_access, "model_grants", lambda: grants)
    monkeypatch.setattr(account_access, "model_allowed", lambda repo_id: True)
    monkeypatch.setattr(
        account_access,
        "repo_is_public",
        lambda repo_id, repo_type = "model", offline = False: repo_id.startswith("public/"),
    )
    return probes, grants


def _refused(token, repo, repo_type):
    return hf_tokens.cached_read_refused(
        token, repo_id = repo, repo_type = repo_type, is_cached = lambda: True
    )


@pytest.mark.parametrize("repo_type", ["model", "dataset"])
def test_managed_session_cannot_read_an_ungranted_cached_gated_repo(hub, repo_type):
    probes, _ = hub
    token = hf_tokens.hf_token_arg("hf_bob", allow_ambient_token = True)
    assert isinstance(token, hf_tokens.AmbientAuthorizedToken)
    assert run_as(BOB, _refused, token, "gated/repo", repo_type) is True
    assert probes == [("gated/repo", "hf_bob", repo_type)]


@pytest.mark.parametrize("repo", ["public/repo", "gated/granted"])
def test_managed_session_keeps_public_and_granted_cached_repos(hub, repo):
    probes, grants = hub
    grants.add(account_access._grant_key("gated/granted", "dataset"))
    token = hf_tokens.hf_token_arg("hf_bob", allow_ambient_token = True)
    assert run_as(BOB, _refused, token, repo, "dataset") is False
    assert probes == []


def test_owner_session_is_unchanged(hub):
    probes, _ = hub
    token = hf_tokens.hf_token_arg("hf_owner", allow_ambient_token = True)
    assert _refused(token, "gated/repo", "dataset") is False
    assert probes == []
