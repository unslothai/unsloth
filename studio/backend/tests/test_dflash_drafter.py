# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""DFlash companion, launch, and reload contracts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from hub.utils.gguf import is_mtp_drafter_path
from utils.models.model_config import (
    _is_mtp_drafter,
    detect_dflash_file,
    detect_gguf_model,
    detect_mtp_file,
)
from core.inference.llama_cpp import (
    LlamaCppBackend,
    _extra_args_requests_dflash,
    _is_companion_gguf_path,
    _should_download_dflash,
)
from models.inference import LoadRequest


DFLASH_DRAFTER_CASES = [
    ("dflash-Qwen3-4B.gguf", True),
    ("models/dflash-qwen3.6-27b.gguf", True),
    ("Qwen3-4B-DFlash.gguf", True),
    ("Qwen3-4B-DFlash-q8_0.gguf", True),
    ("qwen3-4b-dflash-Q4_K_M.gguf", True),
    ("mydflashmodel.gguf", False),
    ("dflash-readme.txt", False),
    ("Qwen3-4B-Q4_K_M.gguf", False),
]


@pytest.fixture
def dflash_pair(tmp_path):
    weight = tmp_path / "Qwen3-4B-Q4_K_M.gguf"
    drafter = tmp_path / "dflash-Qwen3-4B.gguf"
    weight.touch()
    drafter.touch()
    return weight, drafter


@pytest.mark.parametrize("path,expected", DFLASH_DRAFTER_CASES)
def test_dflash_predicate_and_mirrors_agree(path, expected):
    assert is_mtp_drafter_path(path) is expected
    assert _is_mtp_drafter(path) is expected
    assert _is_companion_gguf_path(path) is expected


def test_detect_dflash_file_finds_root_sibling(tmp_path):
    (tmp_path / "Qwen3-4B-Q4_K_M.gguf").touch()
    drafter = tmp_path / "dflash-Qwen3-4B.gguf"
    drafter.touch()
    found = detect_dflash_file(str(tmp_path / "Qwen3-4B-Q4_K_M.gguf"))
    assert found == str(drafter.resolve())


def test_detect_dflash_file_skips_foreign_drafter(tmp_path):
    (tmp_path / "qwen3-8b-Q4_K_M.gguf").touch()
    (tmp_path / "dflash-gemma-4-12b-it.gguf").touch()
    assert detect_dflash_file(str(tmp_path / "qwen3-8b-Q4_K_M.gguf")) is None


def test_detect_dflash_file_search_root(tmp_path):
    sub = tmp_path / "Q4_K_M"
    sub.mkdir()
    (sub / "Qwen3-4B-Q4_K_M.gguf").touch()
    drafter = tmp_path / "dflash-Qwen3-4B.gguf"
    drafter.touch()
    found = detect_dflash_file(str(sub / "Qwen3-4B-Q4_K_M.gguf"), search_root = str(tmp_path))
    assert found == str(drafter.resolve())


def test_detect_dflash_and_mtp_are_disjoint(tmp_path):
    (tmp_path / "Qwen3-4B-Q4_K_M.gguf").touch()
    (tmp_path / "dflash-Qwen3-4B.gguf").touch()
    (tmp_path / "mtp-Qwen3-4B.gguf").touch()
    weight = str(tmp_path / "Qwen3-4B-Q4_K_M.gguf")
    assert detect_dflash_file(weight).endswith("dflash-Qwen3-4B.gguf")
    assert detect_mtp_file(weight).endswith("mtp-Qwen3-4B.gguf")


def test_detect_dflash_file_finds_infix_dflash(tmp_path):
    (tmp_path / "Qwen3-4B-Q4_K_M.gguf").touch()
    drafter = tmp_path / "Qwen3-4B-DFlash-q8_0.gguf"
    drafter.touch()
    found = detect_dflash_file(str(tmp_path / "Qwen3-4B-Q4_K_M.gguf"))
    assert found == str(drafter.resolve())


def test_detect_gguf_model_rejects_infix_dflash_drafter(tmp_path):
    drafter = tmp_path / "Qwen3-4B-DFlash.gguf"
    drafter.touch()
    assert detect_gguf_model(str(drafter)) is None


def test_dflash_pairs_weight_by_model_name():
    from utils.models.model_config import _dflash_pairs_weight

    assert _dflash_pairs_weight("Qwen3-8B-DFlash-q8_0.gguf", "Qwen3-8B-Q4_K_M.gguf") is True
    assert _dflash_pairs_weight("dflash-Qwen3-8B.gguf", "Qwen3-8B-Q4_K_M.gguf") is True
    assert _dflash_pairs_weight("dflash-Qwen3-8B-q8_0.gguf", "Qwen3-8B-Q4_K_M.gguf") is True
    assert _dflash_pairs_weight("Qwen3-4B-DFlash-q8_0.gguf", "Qwen3-8B-Q4_K_M.gguf") is False
    assert _dflash_pairs_weight("dflash-Qwen.gguf", "Qwen3-8B-Q4_K_M.gguf") is False
    assert _dflash_pairs_weight("OtherModel-DFlash-q8_0.gguf", "Q8_0.gguf") is False
    assert _dflash_pairs_weight("Qwen3-8B-DFlash.gguf", None) is True
    assert _dflash_pairs_weight("Qwen3-8B-Q4_K_M.gguf", None) is False


def test_detect_dflash_file_prefers_quantized_over_bf16(tmp_path):
    (tmp_path / "Qwen3-4B-Q4_K_M.gguf").touch()
    bf16 = tmp_path / "Qwen3-4B-DFlash-bf16.gguf"
    bf16.write_bytes(b"\0" * 4096)
    q8 = tmp_path / "Qwen3-4B-DFlash-q8_0.gguf"
    q8.write_bytes(b"\0" * 512)
    found = detect_dflash_file(str(tmp_path / "Qwen3-4B-Q4_K_M.gguf"))
    assert found == str(q8.resolve())


_CAPS_WITH_DFLASH = {
    "dflash_token": "draft-dflash",
    "spec_draft_n_max_flag": "--spec-draft-n-max",
}
_CAPS_WITHOUT_DFLASH = {
    "dflash_token": None,
    "spec_draft_n_max_flag": "--spec-draft-n-max",
}


def _emit(monkeypatch, caps, **kwargs):
    backend = LlamaCppBackend()
    monkeypatch.setattr(
        LlamaCppBackend, "probe_server_capabilities", lambda self, binary = None: caps
    )
    base = dict(
        speculative_type = "auto",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = "unsloth/Qwen3-4B-GGUF",
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
    )
    base.update(kwargs)
    return backend, backend._build_speculative_flags(**base)


def test_emit_dflash_auto_engages_with_default_n_max(monkeypatch):
    backend, flags = _emit(monkeypatch, _CAPS_WITH_DFLASH, dflash_draft_path = "/d/dflash.gguf")
    assert flags == [
        "--model-draft",
        "/d/dflash.gguf",
        "--spec-type",
        "draft-dflash",
        "--spec-draft-n-max",
        "4",
    ]
    assert backend._speculative_type == "draft-dflash"


def test_emit_dflash_honors_n_max_override(monkeypatch):
    _, flags = _emit(
        monkeypatch, _CAPS_WITH_DFLASH, dflash_draft_path = "/d/dflash.gguf", spec_draft_n_max = 6
    )
    assert "--spec-draft-n-max" in flags
    assert flags[flags.index("--spec-draft-n-max") + 1] == "6"


def test_emit_dflash_falls_back_when_binary_lacks_token(monkeypatch):
    backend, flags = _emit(monkeypatch, _CAPS_WITHOUT_DFLASH, dflash_draft_path = "/d/dflash.gguf")
    assert "draft-dflash" not in flags
    assert "--spec-default" in flags
    assert backend._speculative_type == "default"
    assert backend._spec_fallback_reason == "binary_no_dflash"


def test_dflash_does_not_engage_in_forced_mtp_mode(monkeypatch):
    _, flags = _emit(
        monkeypatch,
        _CAPS_WITH_DFLASH,
        speculative_type = "mtp",
        dflash_draft_path = "/d/dflash.gguf",
    )
    assert "draft-dflash" not in flags


def test_dflash_binary_missing_falls_through_to_mtp(monkeypatch):
    caps = {
        "mtp_token": "mtp",
        "supports_mtp": True,
        "dflash_token": None,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }
    backend = LlamaCppBackend()
    backend._nextn_predict_layers = 1  # embedded MTP head (Qwen-style)
    monkeypatch.setattr(
        LlamaCppBackend, "probe_server_capabilities", lambda self, binary = None: caps
    )
    flags = backend._build_speculative_flags(
        speculative_type = "auto",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = "unsloth/Qwen3-4B-GGUF",
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
        dflash_draft_path = "/d/dflash.gguf",
    )
    assert "draft-dflash" not in flags
    assert backend._speculative_type != "default"
    assert backend._spec_fallback_reason != "binary_no_dflash"
    assert backend.dflash_retry_needed is False


def test_dflash_does_not_engage_for_vision_loads(monkeypatch):
    _, flags = _emit(
        monkeypatch,
        _CAPS_WITH_DFLASH,
        dflash_draft_path = "/d/dflash.gguf",
        is_vision = True,
    )
    assert "draft-dflash" not in flags
    assert "--model-draft" not in flags


class _FakeProcess:
    def terminate(self):
        pass

    def wait(self, timeout = None):
        return 0

    def kill(self):
        pass

    def poll(self):
        return 0


def _dflash_backend(gguf_path, dflash_path):
    backend = LlamaCppBackend()
    backend._process = _FakeProcess()
    backend._healthy = True
    backend._model_identifier = "/local/Qwen3-4B-Q4_K_M.gguf"
    backend._hf_variant = None
    backend._requested_n_ctx = 8192
    backend._cache_type_kv = None
    backend._speculative_type = "draft-dflash"
    backend._requested_spec_mode = "auto"
    backend._chat_template_override = None
    backend._is_vision = False
    backend._extra_args = None
    backend._extra_args_source = None
    backend._gguf_path = gguf_path
    backend._dflash_draft_path = dflash_path
    return backend


def _in_target(backend, gguf_path, dflash_draft_path):
    return backend._already_in_target_state(
        gguf_path = gguf_path,
        dflash_draft_path = dflash_draft_path,
        model_identifier = backend._model_identifier,
        hf_variant = None,
        n_ctx = 8192,
        cache_type_kv = None,
        speculative_type = "auto",
        chat_template_override = None,
        extra_args = None,
        is_vision = False,
    )


def test_already_in_target_state_dedups_same_dflash_drafter(dflash_pair):
    weight, drafter = dflash_pair
    backend = _dflash_backend(str(weight), str(drafter))
    assert _in_target(backend, str(weight), str(drafter)) is True


def test_already_in_target_state_retries_fallback_after_binary_change(
    monkeypatch, dflash_pair, tmp_path
):
    weight, drafter = dflash_pair
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"old")
    backend = _dflash_backend(str(weight), str(drafter))
    backend._spec_fallback_reason = "binary_no_dflash"
    monkeypatch.setattr(backend, "_find_llama_server_binary", lambda: str(binary))
    backend._remember_dflash_fallback_inputs(str(binary), str(drafter))

    assert _in_target(backend, str(weight), str(drafter)) is True
    binary.write_bytes(b"new-binary")
    assert _in_target(backend, str(weight), str(drafter)) is False


def test_already_in_target_state_reloads_when_dflash_drafter_appears(dflash_pair):
    weight, drafter = dflash_pair
    backend = _dflash_backend(str(weight), None)
    assert _in_target(backend, str(weight), str(drafter)) is False


def test_already_in_target_state_reloads_when_dflash_drafter_removed(tmp_path):
    weight = tmp_path / "Qwen3-4B-Q4_K_M.gguf"
    weight.touch()
    backend = _dflash_backend(str(weight), "/old/dflash-Qwen3-4B.gguf")
    assert _in_target(backend, str(weight), None) is False


def test_hf_backend_reloads_when_cached_dflash_appears(dflash_pair):
    weight, _ = dflash_pair
    backend = _dflash_backend(str(weight), None)
    backend._hf_repo = "unsloth/Qwen3-4B-GGUF"
    assert _in_target(backend, None, None) is False


def test_hf_vision_backend_ignores_cached_dflash(dflash_pair):
    weight, _ = dflash_pair
    backend = _dflash_backend(str(weight), None)
    backend._hf_repo = "unsloth/Qwen3-VL-4B-GGUF"
    backend._is_vision = True
    assert _in_target(backend, None, None) is True


def test_extra_args_requests_dflash():
    assert _extra_args_requests_dflash(["--spec-type", "draft-dflash"]) is True
    assert _extra_args_requests_dflash(["--spec-type=draft-dflash"]) is True
    assert _extra_args_requests_dflash(["--spec-type", "draft-mtp"]) is False
    assert _extra_args_requests_dflash(["--model-draft", "/d/dflash.gguf"]) is False
    assert _extra_args_requests_dflash(None) is False
    assert _extra_args_requests_dflash([], env = {"LLAMA_ARG_SPEC_TYPE": "draft-dflash"}) is True
    assert (
        _extra_args_requests_dflash(
            ["--spec-type", "draft-mtp"], env = {"LLAMA_ARG_SPEC_TYPE": "draft-dflash"}
        )
        is False
    )


def test_text_only_vlm_downloads_dflash():
    assert _should_download_dflash("auto", ["--no-mmproj"], is_vision = True, mmproj_path = None)
    assert not _should_download_dflash(
        "auto", None, is_vision = True, mmproj_path = "/models/mmproj.gguf"
    )


def test_transient_dflash_download_failure_is_retryable(monkeypatch):
    import core.inference.llama_cpp as llama_cpp
    import huggingface_hub

    backend = LlamaCppBackend()
    monkeypatch.setattr(
        huggingface_hub,
        "list_repo_files",
        lambda *args, **kwargs: ["Qwen3-4B-DFlash-q8_0.gguf"],
    )

    def fail_download(*args, **kwargs):
        raise ConnectionError("temporary")

    monkeypatch.setattr(llama_cpp, "hf_hub_download_with_xet_fallback", fail_download)
    assert backend._download_dflash(hf_repo = "org/repo", weight_name = "Qwen3-4B-Q4_K_M.gguf") is None
    assert backend.dflash_retry_needed is True


def _load_inference_routes_module():
    route_path = Path(_BACKEND_DIR) / "routes" / "inference.py"
    spec = importlib.util.spec_from_file_location("dflash_drafter_inference_routes", route_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope = "module")
def routes():
    return _load_inference_routes_module()


def _route_dedup_backend(gguf_path, *, hf_repo):
    backend = LlamaCppBackend()
    backend._process = _FakeProcess()
    backend._healthy = True
    backend._model_identifier = "unsloth/Qwen3-4B-GGUF"
    backend._hf_variant = "Q4_K_M"
    backend._requested_n_ctx = 0
    backend._cache_type_kv = None
    backend._tensor_parallel = False
    backend._layer_preserves_tensor_intent = False
    backend._requested_spec_mode = "auto"
    backend._spec_fallback_reason = None
    backend._chat_template_override = None
    backend._extra_args = None
    backend._gguf_path = gguf_path
    backend._hf_repo = hf_repo
    backend._mtp_draft_path = None
    backend._dflash_draft_path = None
    return backend


def test_hf_load_dedups_when_dflash_drafter_matches(routes, dflash_pair):
    weight, drafter = dflash_pair
    req = LoadRequest(model_path = "unsloth/Qwen3-4B-GGUF", gguf_variant = "Q4_K_M")
    backend = _route_dedup_backend(str(weight), hf_repo = "unsloth/Qwen3-4B-GGUF")
    backend._dflash_draft_path = str(drafter.resolve())
    assert routes._request_matches_loaded_settings(req, backend) is True


def test_hf_load_reloads_when_dflash_appears_unresolved(routes, dflash_pair):
    weight, _ = dflash_pair
    req = LoadRequest(model_path = "unsloth/Qwen3-4B-GGUF", gguf_variant = "Q4_K_M")
    backend = _route_dedup_backend(str(weight), hf_repo = "unsloth/Qwen3-4B-GGUF")
    assert routes._request_matches_loaded_settings(req, backend) is False


def test_hf_load_retries_transient_dflash_download(routes, tmp_path):
    weight = tmp_path / "Qwen3-4B-Q4_K_M.gguf"
    weight.touch()
    req = LoadRequest(model_path = "unsloth/Qwen3-4B-GGUF", gguf_variant = "Q4_K_M")
    backend = _route_dedup_backend(str(weight), hf_repo = "unsloth/Qwen3-4B-GGUF")
    backend._dflash_retry_needed = True
    assert routes._request_matches_loaded_settings(req, backend) is False


def test_vision_load_with_dflash_sibling_does_not_thrash(routes, tmp_path):
    weight = tmp_path / "Qwen3-VL-4B-Q4_K_M.gguf"
    weight.touch()
    (tmp_path / "dflash-Qwen3-VL-4B.gguf").touch()

    req = LoadRequest(model_path = str(weight))
    backend = _route_dedup_backend(str(weight), hf_repo = None)
    backend._is_vision = True
    assert routes._request_matches_loaded_settings(req, backend) is True


def test_local_load_still_reloads_when_dflash_sibling_appears(routes, dflash_pair):
    weight, _ = dflash_pair
    req = LoadRequest(model_path = str(weight))
    backend = _route_dedup_backend(str(weight), hf_repo = None)
    assert routes._request_matches_loaded_settings(req, backend) is False


def test_remote_companion_bytes_counts_preferred_dflash(routes, monkeypatch):
    import huggingface_hub

    siblings = [
        SimpleNamespace(rfilename = "Qwen3-4B-Q4_K_M.gguf", size = 4_000),
        SimpleNamespace(rfilename = "mtp-Qwen3-4B.gguf", size = 100),
        SimpleNamespace(rfilename = "Qwen3-4B-DFlash-bf16.gguf", size = 1_200),
        SimpleNamespace(rfilename = "Qwen3-4B-DFlash-q8_0.gguf", size = 575),
    ]
    monkeypatch.setattr(
        huggingface_hub, "model_info", lambda *a, **k: SimpleNamespace(siblings = siblings)
    )
    total = routes._remote_gguf_companion_bytes(
        "org/repo",
        hf_token = None,
        include_mmproj = True,
        weight_name = "Qwen3-4B-Q4_K_M.gguf",
    )
    assert total == 675
    auto_total = routes._remote_gguf_companion_bytes(
        "org/repo",
        hf_token = None,
        include_mmproj = True,
        dflash_precedes_mtp = True,
        weight_name = "Qwen3-4B-Q4_K_M.gguf",
    )
    assert auto_total == 575


def test_remote_companion_bytes_skips_foreign_dflash(routes, monkeypatch):
    import huggingface_hub

    siblings = [SimpleNamespace(rfilename = "Gemma-4B-DFlash-q8_0.gguf", size = 575)]
    monkeypatch.setattr(
        huggingface_hub, "model_info", lambda *a, **k: SimpleNamespace(siblings = siblings)
    )
    total = routes._remote_gguf_companion_bytes(
        "org/repo",
        hf_token = None,
        include_mmproj = False,
        weight_name = "Qwen3-4B-Q4_K_M.gguf",
    )
    assert total == 0


def test_dflash_reload_dedup_finds_root_sibling(routes, tmp_path):
    sub = tmp_path / "Q4_K_M"
    sub.mkdir()
    weight = sub / "Qwen3-4B-Q4_K_M.gguf"
    weight.touch()
    drafter = tmp_path / "dflash-Qwen3-4B.gguf"
    drafter.touch()

    backend = _route_dedup_backend(str(weight), hf_repo = None)
    backend._dflash_draft_path = str(drafter.resolve())
    req = LoadRequest(model_path = str(weight))
    assert routes._request_matches_loaded_settings(req, backend) is True


def test_dflash_failure_retries_after_drafter_change(routes, dflash_pair):
    weight, drafter = dflash_pair
    backend = _route_dedup_backend(str(weight), hf_repo = None)
    backend._dflash_draft_path = str(drafter.resolve())
    backend._spec_fallback_reason = "dflash_drafter_incompatible"
    backend._remember_dflash_fallback_inputs(None, str(drafter))
    req = LoadRequest(model_path = str(weight))
    assert routes._request_matches_loaded_settings(req, backend) is True

    drafter.write_bytes(b"replacement")
    assert routes._request_matches_loaded_settings(req, backend) is False


def test_extra_args_dflash_counts_as_separate_draft():
    from core.inference.llama_cpp import _extra_args_requests_separate_draft

    assert _extra_args_requests_separate_draft(["--spec-type", "draft-dflash"]) is True
    assert _extra_args_requests_separate_draft(["--spec-type", "draft-simple"]) is True
    assert _extra_args_requests_separate_draft(["--spec-type", "draft-mtp"]) is False
    assert _extra_args_requests_separate_draft(None) is False


def test_dflash_nmax_change_forces_reload(routes, dflash_pair):
    weight, drafter = dflash_pair
    backend = _route_dedup_backend(str(weight), hf_repo = None)
    backend._speculative_type = "draft-dflash"
    backend._spec_draft_n_max = 4
    backend._dflash_draft_path = str(drafter)

    changed = LoadRequest(model_path = str(weight), spec_draft_n_max = 8)
    assert routes._request_matches_loaded_settings(changed, backend) is False
    same = LoadRequest(model_path = str(weight), spec_draft_n_max = 4)
    assert routes._request_matches_loaded_settings(same, backend) is True


def test_auto_mtp_nmax_change_also_reloads(routes, tmp_path):
    weight = tmp_path / "gemma-4-12b-it-Q4_K_M.gguf"
    weight.touch()
    backend = _route_dedup_backend(str(weight), hf_repo = None)
    backend._speculative_type = "draft-mtp"
    backend._spec_draft_n_max = 2

    changed = LoadRequest(model_path = str(weight), spec_draft_n_max = 6)
    assert routes._request_matches_loaded_settings(changed, backend) is False
