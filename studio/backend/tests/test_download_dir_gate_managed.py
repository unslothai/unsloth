# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's download_dir is refused, so resolving it must not create folders first."""

from hub.services.models import downloads


def test_managed_download_dir_creates_nothing(monkeypatch, tmp_path):
    target = tmp_path / "auth.db-journal"
    monkeypatch.setattr(downloads.account_access, "managed_account", lambda: True)
    assert downloads._gate_hub_cache(str(target)) is None
    assert not target.exists()


def test_owner_download_dir_still_resolves(monkeypatch, tmp_path):
    target = tmp_path / "models"
    monkeypatch.setattr(downloads.account_access, "managed_account", lambda: False)
    hub = downloads._gate_hub_cache(str(target))
    assert hub is not None and str(hub).startswith(str(target))
