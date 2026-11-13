# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama-server flags for a drafter trained with the fine-tune: an EAGLE-3 sidecar paired by the export's
drafter.json, and a trained MTP head that lifts Auto's sub-3B / MLA defaults. GGUFs without that manifest
keep today's routing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_llama_cpp_mtp_detection import (  # noqa: F401 (stubs are installed on import)
    LlamaCppBackend,
    _canonicalize_spec_mode,
    _flags_dict,
)

_SUB_3B = "unsloth/Qwen3.5-0.8B-MTP-GGUF"
_PLAIN = "unsloth/Qwen3-4B-Instruct-GGUF"


def _backend(monkeypatch, *, eagle3 = True):
    caps = {
        "found": True,
        "mtp_token": "draft-mtp",
        "supports_mtp": True,
        "supports_dspark": True,
        "supports_dflash": True,
        "supports_eagle3": eagle3,
        "mtp_probe_inconclusive": False,
        "ngram_mod_flavor": "new",
        "supports_ngram_mod": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }
    monkeypatch.setattr(
        LlamaCppBackend, "probe_server_capabilities", classmethod(lambda cls, binary = None: caps)
    )
    backend = LlamaCppBackend()
    backend._nextn_predict_layers = None
    return backend


def _flags(
    backend,
    mode,
    model,
    path,
    gpus = True,
):
    return backend._build_speculative_flags(
        speculative_type = mode,
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = model,
        model_path = path,
        gpus = gpus,
        binary = "/fake/llama-server",
    )


def _export(
    tmp_path,
    info,
    name = "model.Q4_K_M.gguf",
):
    gguf = tmp_path / name
    gguf.write_bytes(b"GGUF")
    (tmp_path / "drafter.json").write_text(json.dumps(info))
    return str(gguf)


def test_eagle3_mode_canonicalizes():
    assert _canonicalize_spec_mode("eagle3") == "eagle3"
    assert _canonicalize_spec_mode("draft-eagle3") == "eagle3"


def test_auto_uses_paired_eagle3_sidecar(monkeypatch, tmp_path):
    (tmp_path / "eagle3-model-F16.gguf").write_bytes(b"GGUF")
    path = _export(
        tmp_path,
        {"strategy": "eagle3", "files": ["eagle3-model-F16.gguf"], "ggufs": ["model.Q4_K_M.gguf"]},
    )
    flags = _flags_dict(_flags(_backend(monkeypatch), "auto", _PLAIN, path))
    assert flags["--spec-type"] == "draft-eagle3"
    assert flags["--model-draft"] == str(tmp_path / "eagle3-model-F16.gguf")
    assert flags["--spec-draft-n-max"] == "3"


def test_eagle3_without_binary_support_falls_back(monkeypatch, tmp_path):
    (tmp_path / "eagle3-model-F16.gguf").write_bytes(b"GGUF")
    path = _export(tmp_path, {"strategy": "eagle3", "files": ["eagle3-model-F16.gguf"]})
    auto = _flags(_backend(monkeypatch, eagle3 = False), "auto", _PLAIN, path)
    assert "draft-eagle3" not in auto
    forced = _flags(_backend(monkeypatch, eagle3 = False), "eagle3", _PLAIN, path)
    assert forced == ["--spec-default"]


def test_forced_eagle3_without_sidecar_falls_back(monkeypatch, tmp_path):
    backend = _backend(monkeypatch)
    assert _flags(backend, "eagle3", _PLAIN, str(tmp_path / "x.gguf")) == ["--spec-default"]
    assert backend._spec_fallback_reason == "drafter_not_found"


def test_manifest_naming_another_gguf_is_ignored(monkeypatch, tmp_path):
    (tmp_path / "eagle3-model-F16.gguf").write_bytes(b"GGUF")
    _export(
        tmp_path,
        {"strategy": "eagle3", "files": ["eagle3-model-F16.gguf"], "ggufs": ["other.gguf"]},
    )
    path = str(tmp_path / "model.Q4_K_M.gguf")
    assert "draft-eagle3" not in _flags(_backend(monkeypatch), "auto", _PLAIN, path)


@pytest.mark.parametrize("gpus", [True, False])
def test_trained_mtp_head_lifts_the_sub_3b_floor(monkeypatch, tmp_path, gpus):
    stock = _flags_dict(_flags(_backend(monkeypatch), "auto", _SUB_3B, None, gpus))
    assert stock.get("--spec-type") == "ngram-mod"  # unchanged without a manifest
    path = _export(
        tmp_path,
        {"strategy": "mtp", "mtp_trained": True, "files": [], "ggufs": ["model.Q4_K_M.gguf"]},
    )
    trained = _flags_dict(_flags(_backend(monkeypatch), "auto", _SUB_3B, path, gpus))
    assert "draft-mtp" in trained["--spec-type"]


def test_trained_mtp_head_lifts_the_mla_default_off(monkeypatch, tmp_path):
    backend = _backend(monkeypatch)
    backend._nextn_predict_layers = 1
    backend._kv_lora_rank = 512
    monkeypatch.delenv("UNSLOTH_MLA_MTP_ENABLED", raising = False)
    model = "unsloth/GLM-4.7-Flash-GGUF"
    stock = _flags_dict(_flags(backend, "auto", model, None))
    assert stock.get("--spec-type") != "draft-mtp"
    path = _export(tmp_path, {"strategy": "mtp", "mtp_trained": True, "files": []})
    trained = _flags_dict(_flags(backend, "auto", model, path))
    assert trained["--spec-type"] == "draft-mtp"


def test_no_manifest_keeps_routing(monkeypatch, tmp_path):
    path = str(tmp_path / "model.Q4_K_M.gguf")
    for mode in ("auto", "mtp", "ngram", "off", "dspark"):
        a = _flags(_backend(monkeypatch), mode, _SUB_3B, None)
        b = _flags(_backend(monkeypatch), mode, _SUB_3B, path)
        assert a == b, mode


def test_exported_draft_steps_set_the_default_depth(monkeypatch, tmp_path):
    (tmp_path / "eagle3-model-F16.gguf").write_bytes(b"GGUF")
    path = _export(
        tmp_path,
        {"strategy": "eagle3", "files": ["eagle3-model-F16.gguf"], "draft_steps": 5},
    )
    assert (
        _flags_dict(_flags(_backend(monkeypatch), "auto", _PLAIN, path))["--spec-draft-n-max"]
        == "5"
    )
    override = _backend(monkeypatch)._build_speculative_flags(
        speculative_type = "auto",
        spec_draft_n_max = 2,
        extra_args = None,
        model_identifier = _PLAIN,
        model_path = path,
        gpus = True,
        binary = "/fake/llama-server",
    )
    assert _flags_dict(override)["--spec-draft-n-max"] == "2"
    mtp = tmp_path / "mtp"
    mtp.mkdir()
    path = _export(mtp, {"strategy": "mtp", "mtp_trained": True, "files": [], "draft_steps": 4})
    assert (
        _flags_dict(_flags(_backend(monkeypatch), "auto", _SUB_3B, path))["--spec-draft-n-max"]
        == "4"
    )
