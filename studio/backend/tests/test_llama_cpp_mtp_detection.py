# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the MTP auto-detection path (llama.cpp #22673).

Pins three contracts: name-based detector, user-override detector, and
the _already_in_target_state mirror that prevents needless reloads.
"""

from __future__ import annotations

import struct
import sys
import types as _types
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

_httpx_stub = _types.ModuleType("httpx")
for _exc in (
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "CloseError",
):
    setattr(_httpx_stub, _exc, type(_exc, (Exception,), {}))
_httpx_stub.Timeout = type("T", (), {"__init__": lambda s, *a, **k: None})
_httpx_stub.Client = type(
    "C",
    (),
    {
        "__init__": lambda s, **kw: None,
        "__enter__": lambda s: s,
        "__exit__": lambda s, *a: None,
    },
)
sys.modules.setdefault("httpx", _httpx_stub)

import pytest

from core.inference.llama_cpp import (
    LlamaCppBackend,
    _extra_args_set_spec_type,
    _is_mtp_model_name,
)


# Synthetic GGUF helper (mirrors test_gguf_metadata.py).

_GGUF_MAGIC = 0x46554747
_VTYPE_STRING = 8
_VTYPE_UINT32 = 4


def _enc_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _enc_kv_string(key: str, value: str) -> bytes:
    return _enc_string(key) + struct.pack("<I", _VTYPE_STRING) + _enc_string(value)


def _enc_kv_uint32(key: str, value: int) -> bytes:
    return (
        _enc_string(key) + struct.pack("<I", _VTYPE_UINT32) + struct.pack("<I", value)
    )


def _write_minimal_gguf(
    path: Path,
    *,
    arch: str,
    nextn: int | None,
    extra_uint32: dict[str, int] | None = None,
) -> Path:
    """Header-only GGUF with arch + optional nextn_predict_layers."""
    extra_uint32 = dict(extra_uint32 or {})
    body = _enc_kv_string("general.architecture", arch)
    kv_count = 1
    if nextn is not None:
        body += _enc_kv_uint32(f"{arch}.nextn_predict_layers", nextn)
        kv_count += 1
    for k, v in extra_uint32.items():
        body += _enc_kv_uint32(k, v)
        kv_count += 1
    header = struct.pack("<IIQQ", _GGUF_MAGIC, 3, 0, kv_count)
    path.write_bytes(header + body)
    return path


# _is_mtp_model_name helper.


@pytest.mark.parametrize(
    "identifier",
    [
        "unsloth/Qwen3.6-27B-MTP-GGUF",
        "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
        "unsloth/qwen3.6-27b-mtp-gguf",
        "unsloth/Qwen3.6-27B-Mtp-GGUF",
        "unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL",
    ],
)
def test_is_mtp_model_name_detects_marker_in_identifier(identifier):
    assert _is_mtp_model_name(identifier) is True


@pytest.mark.parametrize(
    "identifier",
    [
        "unsloth/Qwen3-27B-GGUF",
        "unsloth/Llama-3.1-8B-Instruct-GGUF",
        "google/gemma-3-4b-it",
        # mtp inside an org name should not match.
        "mtp-research/foo",
        "MTPower/bar",
    ],
)
def test_is_mtp_model_name_does_not_overmatch(identifier):
    assert _is_mtp_model_name(identifier) is False


def test_is_mtp_model_name_handles_none():
    assert _is_mtp_model_name(None) is False
    assert _is_mtp_model_name(None, None) is False
    assert _is_mtp_model_name("", "") is False


def test_is_mtp_model_name_detects_marker_in_filename(tmp_path):
    gguf = tmp_path / "Qwen3.6-27B-MTP-Q4_K_M.gguf"
    gguf.write_bytes(b"")
    assert _is_mtp_model_name("local-model", str(gguf)) is True


def test_is_mtp_model_name_filename_case_insensitive(tmp_path):
    gguf = tmp_path / "qwen3.6-35b-a3b-mtp-q4_k_m.gguf"
    gguf.write_bytes(b"")
    assert _is_mtp_model_name(None, str(gguf)) is True


def test_is_mtp_model_name_ignores_non_mtp_filename(tmp_path):
    gguf = tmp_path / "Qwen3.6-27B-Q4_K_M.gguf"
    gguf.write_bytes(b"")
    assert _is_mtp_model_name("local-model", str(gguf)) is False


# _already_in_target_state MTP promotion.


class _FakeProcess:
    """Minimal stand-in so is_loaded returns True."""

    def terminate(self):
        pass

    def wait(self, timeout = None):
        return 0

    def kill(self):
        pass

    def poll(self):
        return 0


def _mtp_backend(**overrides):
    """MTP-named GGUF backend that's already running with draft-mtp."""
    backend = LlamaCppBackend()
    backend._process = _FakeProcess()
    backend._healthy = True
    backend._model_identifier = "unsloth/Qwen3.6-27B-MTP-GGUF"
    backend._hf_variant = "Q4_K_M"
    backend._requested_n_ctx = 8192
    backend._cache_type_kv = None
    backend._speculative_type = "draft-mtp"
    backend._chat_template_override = None
    backend._is_vision = False
    backend._extra_args = None
    backend._extra_args_source = None
    backend._gguf_path = None
    for key, value in overrides.items():
        setattr(backend, key, value)
    return backend


def test_already_in_target_state_matches_when_request_omits_spec_for_mtp_model():
    # Duplicate /load with no spec must match a running draft-mtp backend.
    backend = _mtp_backend()
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = "unsloth/Qwen3.6-27B-MTP-GGUF",
            hf_variant = "Q4_K_M",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = None,
            chat_template_override = None,
            extra_args = None,
            is_vision = False,
        )
        is True
    )


def test_already_in_target_state_matches_when_request_uses_default_for_mtp_model():
    backend = _mtp_backend()
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = "unsloth/Qwen3.6-27B-MTP-GGUF",
            hf_variant = "Q4_K_M",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = "default",
            chat_template_override = None,
            extra_args = None,
            is_vision = False,
        )
        is True
    )


def test_already_in_target_state_non_mtp_model_unaffected():
    # Promotion is gated on the name; non-MTP must still mismatch req=None.
    backend = _mtp_backend(_model_identifier = "unsloth/Qwen3.6-27B-GGUF")
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = "unsloth/Qwen3.6-27B-GGUF",
            hf_variant = "Q4_K_M",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = None,
            chat_template_override = None,
            extra_args = None,
            is_vision = False,
        )
        is False
    )


def test_already_in_target_state_explicit_off_still_mismatches_mtp_backend():
    backend = _mtp_backend()
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = "unsloth/Qwen3.6-27B-MTP-GGUF",
            hf_variant = "Q4_K_M",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = "off",
            chat_template_override = None,
            extra_args = None,
            is_vision = False,
        )
        is False
    )


# User override via extra_args (unsloth run / unsloth studio run).


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--spec-type", "none"],
        ["--spec-type", "ngram-mod"],
        ["--spec-type", "draft-mtp"],
        ["--spec-type=none"],
        ["--top-k", "20", "--spec-type", "ngram-simple", "--seed", "42"],
        ["--spec-default"],
    ],
)
def test_extra_args_set_spec_type_detects_user_override(extra_args):
    assert _extra_args_set_spec_type(extra_args) is True


@pytest.mark.parametrize(
    "extra_args",
    [
        None,
        [],
        # Scalar tuning knobs compose safely with auto-emitted --spec-type.
        ["--spec-draft-n-max", "4"],
        ["--spec-ngram-mod-n-match", "32"],
        ["--draft-max", "32"],
        ["--top-k", "20", "--seed", "42"],
    ],
)
def test_extra_args_set_spec_type_passes_on_non_spec_type_args(extra_args):
    assert _extra_args_set_spec_type(extra_args) is False


def test_already_in_target_state_user_spec_type_override_matches_clean_backend():
    # User --spec-type none suppressed auto-MTP; repeat /load must not re-promote.
    backend = _mtp_backend(
        _speculative_type = None,
        _extra_args = ["--spec-type", "none"],
    )
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = "unsloth/Qwen3.6-27B-MTP-GGUF",
            hf_variant = "Q4_K_M",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = None,
            chat_template_override = None,
            extra_args = ["--spec-type", "none"],
            is_vision = False,
        )
        is True
    )


def test_already_in_target_state_local_file_mtp_match(tmp_path):
    # Local-file load: -MTP marker comes from the filename.
    gguf = tmp_path / "Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf"
    gguf.write_bytes(b"")
    backend = _mtp_backend(
        _model_identifier = "local-qwen-mtp",
        _gguf_path = str(gguf),
        _hf_variant = None,
    )
    assert (
        backend._already_in_target_state(
            gguf_path = str(gguf),
            model_identifier = "local-qwen-mtp",
            hf_variant = None,
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = None,
            chat_template_override = None,
            extra_args = None,
            is_vision = False,
        )
        is True
    )


# GGUF-metadata-based detection (nextn_predict_layers).


@pytest.mark.parametrize(
    "arch, nextn",
    [
        # Verified against real Unsloth MTP GGUFs (qwen35 / qwen35moe).
        ("qwen35", 1),
        ("qwen35moe", 1),
        # Future-proofing: any arch + n>0 should match.
        ("qwen3moe", 2),
        ("hypothetical_future_arch", 4),
    ],
)
def test_read_gguf_metadata_captures_nextn_predict_layers(tmp_path, arch, nextn):
    gguf = _write_minimal_gguf(
        tmp_path / "model.gguf",
        arch = arch,
        nextn = nextn,
        extra_uint32 = {f"{arch}.block_count": 4},
    )
    backend = LlamaCppBackend()
    backend._read_gguf_metadata(str(gguf))
    assert backend._nextn_predict_layers == nextn


def test_read_gguf_metadata_leaves_nextn_unset_for_non_mtp_arch(tmp_path):
    gguf = _write_minimal_gguf(
        tmp_path / "model.gguf",
        arch = "qwen3",
        nextn = None,
        extra_uint32 = {"qwen3.block_count": 4},
    )
    backend = LlamaCppBackend()
    backend._read_gguf_metadata(str(gguf))
    assert backend._nextn_predict_layers is None


def test_read_gguf_metadata_zero_nextn_is_falsy(tmp_path):
    # bool(0) is False, so the spec block short-circuits.
    gguf = _write_minimal_gguf(
        tmp_path / "model.gguf",
        arch = "qwen35",
        nextn = 0,
        extra_uint32 = {"qwen35.block_count": 4},
    )
    backend = LlamaCppBackend()
    backend._read_gguf_metadata(str(gguf))
    assert backend._nextn_predict_layers == 0
    assert bool(backend._nextn_predict_layers) is False


def test_unload_resets_nextn_predict_layers():
    # MTP state from a previous load must not bleed into the next load.
    backend = LlamaCppBackend()
    backend._nextn_predict_layers = 1
    backend.unload_model()
    assert backend._nextn_predict_layers is None
