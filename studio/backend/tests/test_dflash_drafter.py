# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Separate-file DFlash drafter contracts.

DFlash is a standalone block-diffusion draft GGUF paired to a target,
mechanically the Gemma-style separate-drafter case. Its name carries ``dflash``
as a delimited token anywhere (``dflash-<model>.gguf`` and the
``<model>-DFlash[-<quant>].gguf`` form llama.cpp's converter documents). Pins:
the companion predicate + its two layering mirrors, sibling detection /
foreign-drafter rejection, the ``draft-dflash`` flag emission (including the
no-binary fallback and that it does not fire in a forced ``mtp`` mode), and the
reload-dedup bounce when the drafter appears / changes / disappears or its
draft depth changes.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
)


# ── Companion predicate + layering mirrors ───────────────────────────

DFLASH_DRAFTER_CASES = [
    # unsloth's dflash- prefix form.
    ("dflash-Qwen3-4B.gguf", True),
    ("models/dflash-qwen3.6-27b.gguf", True),
    # llama.cpp's documented converter output + community/HF naming: dflash as a
    # delimited token in the middle or at the end of the name.
    ("Qwen3-4B-DFlash.gguf", True),
    ("Qwen3-4B-DFlash-q8_0.gguf", True),
    ("qwen3-4b-dflash-Q4_K_M.gguf", True),
    # "dflash" embedded in a word (no delimiter) is not a drafter.
    ("mydflashmodel.gguf", False),
    # Not a gguf, and a real quant with no dflash token.
    ("dflash-readme.txt", False),
    ("Qwen3-4B-Q4_K_M.gguf", False),
]


@pytest.mark.parametrize("path,expected", DFLASH_DRAFTER_CASES)
def test_dflash_predicate_and_mirrors_agree(path, expected):
    # The three mirrors must change in lockstep; a DFlash drafter is a companion
    # everywhere mmproj / mtp- are.
    assert is_mtp_drafter_path(path) is expected
    assert _is_mtp_drafter(path) is expected
    assert _is_companion_gguf_path(path) is expected


# ── Sibling detection ────────────────────────────────────────────────


def test_detect_dflash_file_finds_root_sibling(tmp_path):
    (tmp_path / "Qwen3-4B-Q4_K_M.gguf").touch()
    drafter = tmp_path / "dflash-Qwen3-4B.gguf"
    drafter.touch()
    found = detect_dflash_file(str(tmp_path / "Qwen3-4B-Q4_K_M.gguf"))
    assert found == str(drafter.resolve())


def test_detect_dflash_file_none_without_sibling(tmp_path):
    (tmp_path / "Qwen3-4B-Q4_K_M.gguf").touch()
    assert detect_dflash_file(str(tmp_path / "Qwen3-4B-Q4_K_M.gguf")) is None


def test_detect_dflash_file_skips_foreign_drafter(tmp_path):
    # A drafter whose stem does not prefix the weight must not attach
    # (fail-safe: no DFlash, never a mismatched draft).
    (tmp_path / "qwen3-8b-Q4_K_M.gguf").touch()
    (tmp_path / "dflash-gemma-4-12b-it.gguf").touch()
    assert detect_dflash_file(str(tmp_path / "qwen3-8b-Q4_K_M.gguf")) is None


def test_detect_dflash_file_pairs_by_weight_name(tmp_path):
    # The base-model drafter pairs with a quant of that base.
    (tmp_path / "gemma-4-31B-it-Q4_K_M.gguf").touch()
    drafter = tmp_path / "dflash-gemma-4-31B-it.gguf"
    drafter.touch()
    found = detect_dflash_file(str(tmp_path / "gemma-4-31B-it-Q4_K_M.gguf"))
    assert found == str(drafter.resolve())


def test_detect_dflash_file_search_root(tmp_path):
    # Weight in a quant subdir, drafter at the snapshot root.
    sub = tmp_path / "Q4_K_M"
    sub.mkdir()
    (sub / "Qwen3-4B-Q4_K_M.gguf").touch()
    drafter = tmp_path / "dflash-Qwen3-4B.gguf"
    drafter.touch()
    found = detect_dflash_file(str(sub / "Qwen3-4B-Q4_K_M.gguf"), search_root = str(tmp_path))
    assert found == str(drafter.resolve())


def test_detect_gguf_model_rejects_dflash_drafter(tmp_path):
    # A dflash-*.gguf is a companion, never the selectable main model.
    drafter = tmp_path / "dflash-Qwen3-4B.gguf"
    drafter.touch()
    assert detect_gguf_model(str(drafter)) is None


def test_detect_dflash_and_mtp_are_disjoint(tmp_path):
    # The two prefixes never cross-detect.
    (tmp_path / "Qwen3-4B-Q4_K_M.gguf").touch()
    (tmp_path / "dflash-Qwen3-4B.gguf").touch()
    (tmp_path / "mtp-Qwen3-4B.gguf").touch()
    weight = str(tmp_path / "Qwen3-4B-Q4_K_M.gguf")
    assert detect_dflash_file(weight).endswith("dflash-Qwen3-4B.gguf")
    assert detect_mtp_file(weight).endswith("mtp-Qwen3-4B.gguf")


@pytest.mark.parametrize(
    "drafter_name",
    [
        # llama.cpp's documented converter output (docs/speculative.md).
        "Qwen3-4B-DFlash.gguf",
        # community / HF releases append a quant tag.
        "Qwen3-4B-DFlash-q8_0.gguf",
    ],
)
def test_detect_dflash_file_finds_infix_dflash(tmp_path, drafter_name):
    # The `<model>-DFlash[-<quant>].gguf` form pairs with a quant of that model,
    # so a drafter following llama.cpp's own naming is actually engaged.
    (tmp_path / "Qwen3-4B-Q4_K_M.gguf").touch()
    drafter = tmp_path / drafter_name
    drafter.touch()
    found = detect_dflash_file(str(tmp_path / "Qwen3-4B-Q4_K_M.gguf"))
    assert found == str(drafter.resolve())


def test_detect_gguf_model_rejects_infix_dflash_drafter(tmp_path):
    # A `<model>-DFlash.gguf` must not be offered as the selectable main model.
    drafter = tmp_path / "Qwen3-4B-DFlash.gguf"
    drafter.touch()
    assert detect_gguf_model(str(drafter)) is None


def test_dflash_pairs_weight_by_model_name():
    # The shared pairing predicate: a drafter pairs only with a weight its model
    # side prefixes, so a multi-model repo can't attach a foreign drafter (used
    # by both detect_dflash_file and the -hf drafter pick).
    from utils.models.model_config import _dflash_pairs_weight

    assert _dflash_pairs_weight("Qwen3-8B-DFlash-q8_0.gguf", "Qwen3-8B-Q4_K_M.gguf") is True
    assert _dflash_pairs_weight("dflash-Qwen3-8B.gguf", "Qwen3-8B-Q4_K_M.gguf") is True
    # A 4B drafter must not attach to an 8B weight (multi-model repo).
    assert _dflash_pairs_weight("Qwen3-4B-DFlash-q8_0.gguf", "Qwen3-8B-Q4_K_M.gguf") is False
    # A partial model token must end at a delimiter, not inside another token.
    assert _dflash_pairs_weight("dflash-Qwen.gguf", "Qwen3-8B-Q4_K_M.gguf") is False
    # The quant suffix side never pairs a quant-only weight name.
    assert _dflash_pairs_weight("OtherModel-DFlash-q8_0.gguf", "Q8_0.gguf") is False
    # Unknown weight (None) pairs with any dflash drafter; a non-dflash file never.
    assert _dflash_pairs_weight("Qwen3-8B-DFlash.gguf", None) is True
    assert _dflash_pairs_weight("Qwen3-8B-Q4_K_M.gguf", None) is False


def test_detect_dflash_file_ignores_quant_only_weight_name(tmp_path):
    # A native weight named after its quant (Q8_0.gguf) must not attach a foreign
    # drafter via the drafter's quant suffix (OtherModel-DFlash-q8_0 -> q8_0).
    (tmp_path / "Q8_0.gguf").touch()
    (tmp_path / "OtherModel-DFlash-q8_0.gguf").touch()
    assert detect_dflash_file(str(tmp_path / "Q8_0.gguf")) is None


def test_detect_dflash_file_prefers_quantized_over_bf16(tmp_path):
    # The documented flow produces `<model>-DFlash-bf16.gguf` then quantizes to
    # `-q8_0.gguf`, leaving both beside the model. Pick the smaller (quantized)
    # so Studio doesn't launch the oversized full-precision drafter.
    (tmp_path / "Qwen3-4B-Q4_K_M.gguf").touch()
    bf16 = tmp_path / "Qwen3-4B-DFlash-bf16.gguf"
    bf16.write_bytes(b"\0" * 4096)
    q8 = tmp_path / "Qwen3-4B-DFlash-q8_0.gguf"
    q8.write_bytes(b"\0" * 512)
    found = detect_dflash_file(str(tmp_path / "Qwen3-4B-Q4_K_M.gguf"))
    assert found == str(q8.resolve())


# ── Flag emission ────────────────────────────────────────────────────

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
    # A drafter present but the user forced "mtp": DFlash must not hijack it.
    _, flags = _emit(
        monkeypatch,
        _CAPS_WITH_DFLASH,
        speculative_type = "mtp",
        dflash_draft_path = "/d/dflash.gguf",
    )
    assert "draft-dflash" not in flags


def test_dflash_binary_missing_falls_through_to_mtp(monkeypatch):
    # A model with an embedded MTP head AND a dflash sibling, on a binary that
    # lacks draft-dflash, must fall back to its MTP head instead of regressing
    # to no speculative decoding just because the dflash file is present.
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


def test_dflash_does_not_engage_for_vision_loads(monkeypatch):
    # DFlash multimodal drafting is unsupported upstream: a dflash sibling beside
    # a VLM must not emit --model-draft (it would abort alongside --mmproj).
    _, flags = _emit(
        monkeypatch,
        _CAPS_WITH_DFLASH,
        dflash_draft_path = "/d/dflash.gguf",
        is_vision = True,
    )
    assert "draft-dflash" not in flags
    assert "--model-draft" not in flags


# ── Reload-dedup bounce ──────────────────────────────────────────────


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


def test_already_in_target_state_dedups_same_dflash_drafter(tmp_path):
    weight = tmp_path / "Qwen3-4B-Q4_K_M.gguf"
    weight.touch()
    drafter = tmp_path / "dflash-Qwen3-4B.gguf"
    drafter.touch()
    backend = _dflash_backend(str(weight), str(drafter))
    assert _in_target(backend, str(weight), str(drafter)) is True


@pytest.mark.parametrize(
    "reason",
    ["binary_no_dflash", "dflash_drafter_incompatible", "dflash_runtime_error"],
)
def test_already_in_target_state_bypasses_after_dflash_fallback(tmp_path, reason):
    # Backend parity with the route guard: a prior DFlash fallback must not
    # dedupe, so a direct/CLI reload retries after an update or a fixed drafter.
    weight = tmp_path / "Qwen3-4B-Q4_K_M.gguf"
    weight.touch()
    drafter = tmp_path / "dflash-Qwen3-4B.gguf"
    drafter.touch()
    backend = _dflash_backend(str(weight), str(drafter))
    backend._spec_fallback_reason = reason
    assert _in_target(backend, str(weight), str(drafter)) is False


def test_already_in_target_state_reloads_when_dflash_drafter_appears(tmp_path):
    # Loaded without a drafter; a dflash sibling now resolves -> must reload.
    weight = tmp_path / "Qwen3-4B-Q4_K_M.gguf"
    weight.touch()
    drafter = tmp_path / "dflash-Qwen3-4B.gguf"
    drafter.touch()
    backend = _dflash_backend(str(weight), None)
    assert _in_target(backend, str(weight), str(drafter)) is False


def test_already_in_target_state_reloads_when_dflash_drafter_removed(tmp_path):
    weight = tmp_path / "Qwen3-4B-Q4_K_M.gguf"
    weight.touch()
    backend = _dflash_backend(str(weight), "/old/dflash-Qwen3-4B.gguf")
    assert _in_target(backend, str(weight), None) is False


# ── User-supplied --spec-type draft-dflash in extras ─────────────────


def test_extra_args_requests_dflash():
    # Auto-detect emits draft-dflash in spec_flags; a manually supplied drafter
    # only shows up here, so the crash-recovery retry must key off it too (like
    # MTP's _extra_args_requests_mtp), else a user drafter that aborts the
    # server fails the whole load instead of retrying without speculation.
    assert _extra_args_requests_dflash(["--spec-type", "draft-dflash"]) is True
    assert _extra_args_requests_dflash(["--spec-type=draft-dflash"]) is True
    assert _extra_args_requests_dflash(["--spec-type", "draft-mtp"]) is False
    assert _extra_args_requests_dflash(["--model-draft", "/d/dflash.gguf"]) is False
    assert _extra_args_requests_dflash(None) is False
    # A CLI flag wins over the env, matching llama.cpp / _effective_spec_type.
    assert _extra_args_requests_dflash([], env = {"LLAMA_ARG_SPEC_TYPE": "draft-dflash"}) is True
    assert (
        _extra_args_requests_dflash(
            ["--spec-type", "draft-mtp"], env = {"LLAMA_ARG_SPEC_TYPE": "draft-dflash"}
        )
        is False
    )


# ── Route reload-dedup: HF must not thrash on a snapshot dflash sibling ───


def _load_inference_routes_module():
    """Load routes/inference.py directly, bypassing routes/__init__.py (which
    imports every router, dragging in unrelated deps)."""
    route_path = Path(_BACKEND_DIR) / "routes" / "inference.py"
    spec = importlib.util.spec_from_file_location("dflash_drafter_inference_routes", route_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_hf_load_dedups_when_dflash_drafter_matches(tmp_path):
    # HF loads now resolve the DFlash drafter (_download_dflash), so the dedup
    # runs for HF too: when the stored drafter matches the snapshot sibling, a
    # duplicate /load dedups (no thrash).
    routes = _load_inference_routes_module()
    from models.inference import LoadRequest

    weight = tmp_path / "Qwen3-4B-Q4_K_M.gguf"
    weight.touch()
    drafter = tmp_path / "dflash-Qwen3-4B.gguf"
    drafter.touch()

    req = LoadRequest(model_path = "unsloth/Qwen3-4B-GGUF", gguf_variant = "Q4_K_M")
    backend = _route_dedup_backend(str(weight), hf_repo = "unsloth/Qwen3-4B-GGUF")
    backend._dflash_draft_path = str(drafter.resolve())
    assert routes._request_matches_loaded_settings(req, backend) is True


def test_hf_load_reloads_when_dflash_appears_unresolved(tmp_path):
    # A DFlash file present in the HF snapshot but not yet the stored drafter
    # must force a reload so the newly available drafter engages, rather than
    # sticking on already_loaded.
    routes = _load_inference_routes_module()
    from models.inference import LoadRequest

    weight = tmp_path / "Qwen3-4B-Q4_K_M.gguf"
    weight.touch()
    (tmp_path / "dflash-Qwen3-4B.gguf").touch()

    req = LoadRequest(model_path = "unsloth/Qwen3-4B-GGUF", gguf_variant = "Q4_K_M")
    backend = _route_dedup_backend(str(weight), hf_repo = "unsloth/Qwen3-4B-GGUF")
    assert routes._request_matches_loaded_settings(req, backend) is False


def test_vision_load_with_dflash_sibling_does_not_thrash(tmp_path):
    # A VLM suppresses DFlash, so its stored path stays None; a detected sibling
    # must not be compared against it (that would reload every /load).
    routes = _load_inference_routes_module()
    from models.inference import LoadRequest

    weight = tmp_path / "Qwen3-VL-4B-Q4_K_M.gguf"
    weight.touch()
    (tmp_path / "dflash-Qwen3-VL-4B.gguf").touch()

    req = LoadRequest(model_path = str(weight))
    backend = _route_dedup_backend(str(weight), hf_repo = None)
    backend._is_vision = True
    assert routes._request_matches_loaded_settings(req, backend) is True


def test_local_load_still_reloads_when_dflash_sibling_appears(tmp_path):
    # The HF guard must not weaken the local path: a dflash sibling that
    # resolves next to a local weight with no stored drafter still reloads.
    routes = _load_inference_routes_module()
    from models.inference import LoadRequest

    weight = tmp_path / "Qwen3-4B-Q4_K_M.gguf"
    weight.touch()
    (tmp_path / "dflash-Qwen3-4B.gguf").touch()

    req = LoadRequest(model_path = str(weight))
    backend = _route_dedup_backend(str(weight), hf_repo = None)
    assert routes._request_matches_loaded_settings(req, backend) is False


def test_remote_companion_bytes_counts_preferred_dflash(monkeypatch):
    # The training VRAM guard must count the DFlash drafter the -hf load will
    # fetch (the quantized one), not the oversized bf16 and not every build.
    from types import SimpleNamespace

    import huggingface_hub

    routes = _load_inference_routes_module()
    siblings = [
        SimpleNamespace(rfilename = "Qwen3-4B-Q4_K_M.gguf", size = 4_000),
        SimpleNamespace(rfilename = "Qwen3-4B-DFlash-bf16.gguf", size = 1_200),
        SimpleNamespace(rfilename = "Qwen3-4B-DFlash-q8_0.gguf", size = 575),
    ]
    monkeypatch.setattr(
        huggingface_hub, "model_info", lambda *a, **k: SimpleNamespace(siblings = siblings)
    )
    total = routes._remote_gguf_companion_bytes("org/repo", hf_token = None, include_mmproj = True)
    assert total == 575


def test_dflash_reload_dedup_finds_root_sibling(tmp_path):
    # Weight in a quant subdir, drafter at the snapshot root: the reload dedup
    # scans the companion root (like initial detection), so a duplicate /load
    # dedups instead of reloading every time.
    routes = _load_inference_routes_module()
    from models.inference import LoadRequest

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


@pytest.mark.parametrize(
    "reason",
    ["binary_no_dflash", "dflash_drafter_incompatible", "dflash_runtime_error"],
)
def test_dflash_failure_forces_reload_to_retry(tmp_path, reason):
    # After any DFlash fallback (old binary, fork-format drafter, runtime crash),
    # a same-settings reload must retry so a newer binary or a replaced drafter
    # engages, instead of deduping to the fallback server.
    routes = _load_inference_routes_module()
    from models.inference import LoadRequest

    weight = tmp_path / "Qwen3-4B-Q4_K_M.gguf"
    weight.touch()
    drafter = tmp_path / "dflash-Qwen3-4B.gguf"
    drafter.touch()

    backend = _route_dedup_backend(str(weight), hf_repo = None)
    backend._dflash_draft_path = str(drafter.resolve())
    backend._spec_fallback_reason = reason
    req = LoadRequest(model_path = str(weight))
    assert routes._request_matches_loaded_settings(req, backend) is False


def test_extra_args_dflash_counts_as_separate_draft():
    # A user-supplied --spec-type draft-dflash must be treated as a separate
    # draft model so the VRAM budget reserves for it (like draft-simple/eagle3).
    from core.inference.llama_cpp import _extra_args_requests_separate_draft

    assert _extra_args_requests_separate_draft(["--spec-type", "draft-dflash"]) is True
    assert _extra_args_requests_separate_draft(["--spec-type", "draft-simple"]) is True
    assert _extra_args_requests_separate_draft(["--spec-type", "draft-mtp"]) is False
    assert _extra_args_requests_separate_draft(None) is False


def test_dflash_nmax_change_forces_reload(tmp_path):
    # DFlash engages only in Auto (backend_mode stays "auto"), so the route must
    # key the n-max compare off the resolved draft-dflash spec, else a changed
    # --spec-draft-n-max is deduped and the backend n-max guard never runs.
    routes = _load_inference_routes_module()
    from models.inference import LoadRequest

    weight = tmp_path / "Qwen3-4B-Q4_K_M.gguf"
    weight.touch()
    drafter = tmp_path / "dflash-Qwen3-4B.gguf"
    drafter.touch()

    backend = _route_dedup_backend(str(weight), hf_repo = None)
    backend._speculative_type = "draft-dflash"
    backend._spec_draft_n_max = 4
    backend._dflash_draft_path = str(drafter)

    changed = LoadRequest(model_path = str(weight), spec_draft_n_max = 8)
    assert routes._request_matches_loaded_settings(changed, backend) is False
    same = LoadRequest(model_path = str(weight), spec_draft_n_max = 4)
    assert routes._request_matches_loaded_settings(same, backend) is True


def test_auto_mtp_nmax_change_also_reloads(tmp_path):
    # Parity: the route n-max compare mirrors the backend guard, which covers
    # draft-mtp too. An Auto MTP load stays backend_mode "auto", so a changed
    # n-max must not dedup just because it isn't a forced mtp / mtp+ngram mode.
    routes = _load_inference_routes_module()
    from models.inference import LoadRequest

    weight = tmp_path / "gemma-4-12b-it-Q4_K_M.gguf"
    weight.touch()
    backend = _route_dedup_backend(str(weight), hf_repo = None)
    backend._speculative_type = "draft-mtp"
    backend._spec_draft_n_max = 2

    changed = LoadRequest(model_path = str(weight), spec_draft_n_max = 6)
    assert routes._request_matches_loaded_settings(changed, backend) is False
