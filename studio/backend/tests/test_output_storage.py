# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path

import pytest

from utils.paths.output_storage import resolve_configured_outputs_root
from utils.paths.storage_roots import outputs_root


def test_outputs_root_preserves_default_when_unset_or_empty(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.delenv("UNSLOTH_OUTPUTS_DIR", raising = False)
    assert outputs_root() == tmp_path / "studio" / "outputs"
    monkeypatch.setenv("UNSLOTH_OUTPUTS_DIR", "   ")
    assert outputs_root() == tmp_path / "studio" / "outputs"


def test_outputs_root_reads_environment_at_call_time(monkeypatch, tmp_path):
    configured = tmp_path / "persistent" / "Outputs"
    monkeypatch.delenv("UNSLOTH_OUTPUTS_DIR", raising = False)
    assert outputs_root() != configured
    monkeypatch.setenv("UNSLOTH_OUTPUTS_DIR", str(configured))
    assert outputs_root() == configured


@pytest.mark.parametrize("value", ["relative", "../escape", "/tmp/../escape"])
def test_outputs_root_rejects_unsafe_values(monkeypatch, value):
    monkeypatch.setenv("UNSLOTH_OUTPUTS_DIR", value)
    with pytest.raises(ValueError):
        resolve_configured_outputs_root(default = Path("/default"))


def test_outputs_root_rejects_null_bytes(monkeypatch):
    from utils.paths import output_storage
    monkeypatch.setattr(output_storage.os, "environ", {"UNSLOTH_OUTPUTS_DIR": "bad\x00path"})
    with pytest.raises(ValueError, match = "null"):
        resolve_configured_outputs_root(default = Path("/default"))


def test_output_child_symlink_escape_is_rejected(monkeypatch, tmp_path):
    root = tmp_path / "outputs"
    outside = tmp_path / "models"
    root.mkdir()
    outside.mkdir()
    (root / "escaped").symlink_to(outside, target_is_directory = True)
    monkeypatch.setenv("UNSLOTH_OUTPUTS_DIR", str(root))
    from utils.paths.storage_roots import resolve_output_dir
    with pytest.raises(ValueError, match = "escapes root"):
        resolve_output_dir("escaped/model")
