# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import ast
import subprocess
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_SAVE_PY = _BACKEND.parents[1] / "unsloth" / "save.py"


@pytest.fixture
def lce():
    # Imported per test: a collection-time import of utils.transformers_version reorders the
    # utils.hardware import chain other modules rely on.
    import utils.llm_compressor_export as module
    return module


@pytest.mark.parametrize(
    "shadow_valid, shadow_disabled, offline, workspace_ok, autoinstall_off, ready, kind, blocked",
    [
        (True, False, False, False, False, True, None, False),
        (False, False, False, False, False, False, "shadow", False),
        # export.py prefers the shadow over a workspace copy, so that still needs consent.
        (False, False, False, True, False, False, "shadow", False),
        (False, True, False, True, False, True, None, False),
        (False, False, True, True, False, True, None, False),
        (False, True, False, False, False, False, "workspace", False),
        (False, True, False, False, True, False, None, True),
        (True, True, False, False, False, False, "workspace", False),
        (True, True, False, True, False, True, None, False),
    ],
)
def test_probe_routing(
    lce,
    monkeypatch,
    shadow_valid,
    shadow_disabled,
    offline,
    workspace_ok,
    autoinstall_off,
    ready,
    kind,
    blocked,
):
    monkeypatch.setattr(lce, "_llmcompressor_shadow_is_valid", lambda: shadow_valid)
    monkeypatch.setattr(lce, "_llmcompressor_main_disabled", lambda: shadow_disabled)
    monkeypatch.setattr(lce, "_env_offline", lambda: offline)
    monkeypatch.setattr(lce, "_workspace_llmcompressor_ok", lambda spec: workspace_ok)
    monkeypatch.setenv(
        "UNSLOTH_DISABLE_LLM_COMPRESSOR_AUTOINSTALL", "1" if autoinstall_off else "0"
    )
    probe = lce.probe_llm_compressor_for_compressed_export()
    assert probe["ready"] is ready
    assert probe["consent_kind"] == kind
    assert probe["needs_consent"] is (kind is not None)
    assert bool(probe["blocked_reason"]) is blocked


def test_spec_matches_save_py(lce):
    tree = ast.parse(_SAVE_PY.read_text(encoding = "utf-8"))
    expected = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "_LLM_COMPRESSOR_SPEC" for t in n.targets)
    )
    assert lce._LLM_COMPRESSOR_SPEC == expected
    assert expected in lce.probe_llm_compressor_for_compressed_export()["workspace_install_command"]


def test_probe_does_not_import_unsloth_or_llmcompressor():
    code = (
        "import sys\n"
        "from utils.llm_compressor_export import probe_llm_compressor_for_compressed_export as p\n"
        "p()\n"
        "bad = [m for m in ('unsloth', 'llmcompressor', 'torch') if m in sys.modules]\n"
        "print(bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd = _BACKEND, capture_output = True, text = True)
    assert r.returncode == 0, r.stdout + r.stderr
