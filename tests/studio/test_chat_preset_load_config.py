# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Contract coverage for preset load settings (#7347)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    path = ROOT / relative
    if not path.exists():
        path = ROOT / "unsloth_repo" / relative
    return path.read_text(encoding = "utf-8")


def test_preset_interface_includes_load_config():
    policy = _read("studio/frontend/src/features/chat/presets/preset-policy.ts")
    assert "loadConfig?: PresetLoadConfig" in policy


def test_preset_save_captures_load_config():
    sheet = _read("studio/frontend/src/features/chat/chat-settings-sheet.tsx")
    assert "capturePresetLoadConfig()" in sheet
    assert "applyPresetLoadConfig" in sheet


def test_preset_apply_restores_load_config():
    sheet = _read("studio/frontend/src/features/chat/chat-settings-sheet.tsx")
    assert "applyPresetLoadConfig(p.loadConfig)" in sheet


def test_persisted_preset_serializes_load_config():
    storage = _read("studio/frontend/src/features/chat/utils/chat-settings-storage.ts")
    assert "normalizePresetLoadConfig(item.loadConfig)" in storage
    api = _read("studio/frontend/src/features/chat/api/chat-settings-api.ts")
    assert "loadConfig?: Record<string, unknown>" in api


def test_capture_reads_gguf_loaded_context():
    source = _read("studio/frontend/src/features/chat/presets/preset-load-config.ts")
    assert "store.ggufContextLength" in source
    assert "effectiveContextLength" in source


def test_backend_chat_preset_accepts_load_config():
    routes = _read("studio/backend/routes/chat_history.py")
    assert "class ChatPresetLoadConfig" in routes
    assert "loadConfig: Optional[ChatPresetLoadConfig]" in routes
