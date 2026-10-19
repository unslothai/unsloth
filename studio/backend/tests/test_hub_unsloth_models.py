# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Hub's On Device list takes Studio's own fine-tunes and exports as source "unsloth"."""

import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from hub.services.models import local_inventory  # noqa: E402
from hub.utils.host_paths import HOST_PATH_LOCAL_SOURCES  # noqa: E402


def _roots(tmp_path, monkeypatch):
    from utils.paths import storage_roots

    outputs, exports = tmp_path / "outputs", tmp_path / "exports"
    outputs.mkdir()
    exports.mkdir()
    monkeypatch.setattr(storage_roots, "outputs_root", lambda: outputs)
    monkeypatch.setattr(storage_roots, "exports_root", lambda: exports)
    import utils.models.model_config as model_config

    monkeypatch.setattr(model_config, "within_account", lambda _path: True)
    return outputs, exports


def test_fine_tunes_and_exports_are_listed_as_unsloth(tmp_path, monkeypatch):
    outputs, exports = _roots(tmp_path, monkeypatch)
    run = outputs / "my-run"
    run.mkdir()
    (run / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "unsloth/Qwen3-0.6B", "peft_type": "LORA"})
    )
    (run / "adapter_model.safetensors").write_bytes(b"x" * 16)
    gguf = exports / "my-run-finetune-gguf"
    gguf.mkdir()
    (gguf / "model-Q4_K_M.gguf").write_bytes(b"GGUF" + b"\0" * 64)

    rows = local_inventory._scan_unsloth_models()

    assert {row.source for row in rows} == {"unsloth"}
    by_path = {Path(row.path).resolve(): row for row in rows}
    adapter = by_path[run.resolve()]
    assert adapter.model_format == "adapter"
    assert adapter.base_model == "unsloth/Qwen3-0.6B"
    assert any(Path(row.path).resolve().is_relative_to(gguf.resolve()) for row in rows)


def test_nothing_trained_lists_nothing(tmp_path, monkeypatch):
    _roots(tmp_path, monkeypatch)
    assert local_inventory._scan_unsloth_models() == []


def test_unsloth_rows_have_their_host_paths_redacted():
    assert "unsloth" in HOST_PATH_LOCAL_SOURCES


def test_a_gguf_export_with_several_quants_offers_all_of_them(tmp_path, monkeypatch):
    _, exports = _roots(tmp_path, monkeypatch)
    export = exports / "my-run-finetune-gguf"
    export.mkdir()
    (export / "my-run-Q4_K_M.gguf").write_bytes(b"GGUF" + b"\0" * 64)
    (export / "my-run-Q8_0.gguf").write_bytes(b"GGUF" + b"\0" * 128)

    rows = local_inventory._scan_unsloth_models()

    # The folder, so the card offers every quant, not just the largest file.
    [row] = rows
    assert Path(row.path).resolve() == export.resolve()
    assert row.capabilities.requires_variant
