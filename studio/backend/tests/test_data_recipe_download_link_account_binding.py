# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Data Recipe export link must name the account that minted it.

The four sibling signed links in this backend all bind the tenant: preview shares and
the RAG document link carry an account id and re-bind it at redemption, and gallery
images and video use media_link_account plus run_as. This one did not, so a link minted
by a managed account was redeemed with the account ContextVar at its OWNER default.
Every recipe root is derived from that ContextVar, so the artifact_path the link carries
was validated under the minter's root and then resolved under the owner's.
"""

from __future__ import annotations

import asyncio

import pytest

from utils.account_context import (
    OWNER,
    AccountContext,
    bind_account,
    current_account,
    reset_account,
)

from routes.data_recipe import jobs

ALICE = AccountContext("a1b2c3d4e5f6a7b8", "alice", "user")

PARTS = dict(
    job_id = "job-42",
    export_format = "jsonl",
    artifact_path = "recipe_my_run",
    filename = None,
)


def _mint_as(account):
    marker = bind_account(account)
    try:
        return jobs._sign_download_link(**PARTS)
    finally:
        reset_account(marker)


def _redeem(token, account_lookup = None, monkeypatch = None):
    """Run the dependency and report the account in force inside the handler's scope."""
    if account_lookup is not None:
        import auth.storage

        monkeypatch.setattr(auth.storage, "get_account_by_id", account_lookup)

    async def drive():
        generator = jobs._authorize_dataset_download(
            request = None, token = token, **PARTS
        )
        seen = None
        async for _ in generator:
            seen = current_account()
            break
        await generator.aclose()
        return seen

    return asyncio.run(drive())


def test_a_managed_account_link_is_redeemed_as_that_account(monkeypatch):
    """The regression: this used to redeem as OWNER, under the owner's recipe root."""
    token = _mint_as(ALICE)
    seen = _redeem(token, lambda account_id: ALICE if account_id == ALICE.account_id else None,
                   monkeypatch)
    assert seen == ALICE, "the link was redeemed under a different account than it was minted by"


def test_an_owner_link_is_redeemed_as_the_owner(monkeypatch):
    token = _mint_as(OWNER)
    seen = _redeem(token, lambda account_id: None, monkeypatch)
    assert seen == OWNER


def test_a_managed_link_dies_with_its_account(monkeypatch):
    """get_account_by_id returning None means deactivated or deleted."""
    token = _mint_as(ALICE)
    with pytest.raises(Exception):
        _redeem(token, lambda account_id: None, monkeypatch)


def test_one_accounts_link_does_not_verify_for_another(monkeypatch):
    """The account id is inside the signed payload, so it cannot be swapped."""
    token = _mint_as(ALICE)
    expiry, _account_id, signature = token.split(".", 2)
    forged = f"{expiry}.{'f' * 16}.{signature}"
    assert jobs._download_link_account(forged, **PARTS) is None


def test_the_owner_token_shape_is_not_accepted_for_a_managed_payload():
    """An owner-shaped token must not open a managed account's artifact."""
    owner_token = _mint_as(OWNER)
    expiry, _empty, signature = owner_token.split(".", 2)
    forged = f"{expiry}.{ALICE.account_id}.{signature}"
    assert jobs._download_link_account(forged, **PARTS) is None


def test_a_tampered_artifact_path_still_fails_the_mac():
    token = _mint_as(OWNER)
    other = dict(PARTS, artifact_path = "recipe_someone_elses_run")
    assert jobs._download_link_account(token, **other) is None
