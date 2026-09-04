# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Contract coverage for preset load settings (#7347)."""

import re
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
    assert "if (p.loadConfig)" in sheet
    assert "applyPresetLoadConfig(p.loadConfig)" in sheet


def test_persisted_preset_serializes_load_config():
    storage = _read("studio/frontend/src/features/chat/utils/chat-settings-storage.ts")
    assert "normalizePresetLoadConfig(item.loadConfig)" in storage
    api = _read("studio/frontend/src/features/chat/api/chat-settings-api.ts")
    assert "loadConfig?: Record<string, unknown>" in api


def test_capture_reads_gguf_loaded_context():
    source = _read("studio/frontend/src/features/chat/presets/preset-load-config.ts")
    assert "store.loadedContextLength" in source
    assert "effectiveContextLength" in source


def test_apply_skips_missing_load_config():
    source = _read("studio/frontend/src/features/chat/presets/preset-load-config.ts")
    assert "if (config == null)" in source
    assert "selectedGpuIds: store.selectedGpuIds" in source
    sheet = _read("studio/frontend/src/features/chat/chat-settings-sheet.tsx")
    assert "if (p.loadConfig)" in sheet


def test_hydration_does_not_replay_preset_load_config():
    store = _read("studio/frontend/src/features/chat/stores/chat-runtime-store.ts")
    assert "applyPresetLoadConfig(activeDefinition.loadConfig)" not in store


def test_capture_coalesces_default_load_knobs():
    source = _read("studio/frontend/src/features/chat/presets/preset-load-config.ts")
    assert "coalesceDefaultLoadKnobs" in source
    assert "DEFAULT_MAX_SEQ_LENGTH" in source


def test_backend_chat_preset_accepts_load_config():
    routes = _read("studio/backend/routes/chat_history.py")
    assert "class ChatPresetLoadConfig" in routes
    assert "loadConfig: Optional[ChatPresetLoadConfig]" in routes


def test_preset_load_config_carries_parallel_slots():
    # Captured, clamped on read, applied, and accepted by the extra="forbid"
    # backend model (a missing backend field would 422 every settings sync).
    source = _read("studio/frontend/src/features/chat/presets/preset-load-config.ts")
    assert '| "nParallel"' in source
    assert "nParallel: snapshot.nParallel ?? null" in source
    assert "nParallel: config.nParallel ?? null" in source
    assert "N_PARALLEL_MAX, Math.round(partial.nParallel)" in source
    routes = _read("studio/backend/routes/chat_history.py")
    assert (
        "nParallel: Optional[int] = Field(default = None, ge = PARALLEL_MIN, le = PARALLEL_MAX)"
        in routes
    )


def test_a_preset_records_a_self_sizing_load_s_pin_and_not_its_window():
    """A preset must reproduce the setup that ran. A window nobody pinned is reached again
    on replay, so only the pin is stored -- in the one field that means "pinned"."""
    source = _read("studio/frontend/src/features/chat/presets/preset-load-config.ts")
    body = source[source.index("export function capturePresetLoadConfig") :]
    body = body[: body.index("\n}\n")]
    # Here: capture asks the rule, off classifiers rather than constants, and bounds it.
    assert "requestableContextLength(\n    capturedContextLength(" in body, body
    assert "isServedByLlamaCpp({\n    loadedIsGguf: store.loadedIsGguf," in body, body
    assert not re.search(r"LlamaCpp\(\{[^}]*?(\w+): (?!store\.[\w.]*\1,)", body), body
    assert "isServedByMlx(isGguf, platform.deviceType, platform.chatOnlyReason)" in body, body
    # Compared as a pin, not as a field: another backend holds it in the other field.
    compare = source[source.index("function toComparablePerModelConfig(") :]
    compare = compare[: compare.index("\n}\n")]
    assert "const pin = savedContextPin(config);" in compare, compare
    assert "customContextLength: pin,\n    maxSeqLength: null," in compare, compare
    # Both bounds, in the one rule capture and storage share, else the replays disagree.
    rule = source[source.index("function requestableContextLength(") :]
    rule = rule[: rule.index("\n}\n")]
    assert "Math.min(MAX_SEQ_LENGTH_MAX, Math.max(CONTEXT_LENGTH_MIN," in rule, rule
    assert source.count("requestableContextLength(") == 3, source
    assert "loadedContextLength: store.loadedContextLength," in body, body
    assert "controlPin: snapshot.customContextLength," in body, body
    # A self-sized window is never captured: bounded, it replays a wider one narrower.
    assert re.search(r"capturedContextLength\(\{\n\s*isGguf,\n\s*controlPin:", body), body
    assert "maxSeqLength: isMlx ? null : normalizeMaxSeqLength(" in body, body
