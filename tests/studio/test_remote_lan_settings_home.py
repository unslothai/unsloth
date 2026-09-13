# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Remote and LAN access cards live on one Settings tab, not two."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SETTINGS_TABS = REPO / "studio/frontend/src/features/settings/tabs"
API_KEYS_TAB = SETTINGS_TABS / "api-keys-tab.tsx"
REMOTE_LAN_TAB = SETTINGS_TABS / "remote-lan-tab.tsx"
README = REPO / "README.md"

CARDS = ("<RemoteAccessSection", "<LanAccessSection")


def test_remote_and_lan_cards_mount_only_on_the_remote_lan_tab():
    # #9519: both tabs rendered the same two cards. #9389 left the API copy so
    # the old path still worked. The dedicated tab is the home now.
    api = API_KEYS_TAB.read_text(encoding = "utf-8")
    remote = REMOTE_LAN_TAB.read_text(encoding = "utf-8")
    for tag in CARDS:
        assert tag not in api, f"{tag} still mounts on the API tab"
        assert remote.count(tag) == 1, f"{tag} must mount once on Remote & LAN"


def test_api_tab_points_at_the_remote_lan_tab():
    api = API_KEYS_TAB.read_text(encoding = "utf-8")
    assert 'setActiveTab("remote-lan")' in api
    assert "settings.tabs.remoteLan" in api


def test_readme_sends_lan_access_to_the_remote_lan_tab():
    text = README.read_text(encoding = "utf-8")
    assert "Settings > Remote & LAN > LAN access" in text
    assert "Settings > API keys > LAN access" not in text
