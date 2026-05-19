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
    _backfill_usage_from_timings,
    _build_ngram_mod_flags,
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


def test_already_in_target_state_vision_mtp_match():
    # llama.cpp #22673: MTP is compatible with mmproj. A vision MTP load
    # with auto/default spec must match a backend already running draft-mtp.
    backend = _mtp_backend(_is_vision = True)
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
            is_vision = True,
        )
        is True
    )


def test_already_in_target_state_vision_mtp_default_matches():
    backend = _mtp_backend(_is_vision = True)
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
            is_vision = True,
        )
        is True
    )


def test_already_in_target_state_vision_non_mtp_unaffected():
    # Vision non-MTP repo (no -MTP marker) must still mismatch req=None
    # against a backend running draft-mtp.
    backend = _mtp_backend(
        _model_identifier = "unsloth/Qwen3-VL-4B-Instruct-GGUF",
        _is_vision = True,
    )
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = "unsloth/Qwen3-VL-4B-Instruct-GGUF",
            hf_variant = "Q4_K_M",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = None,
            chat_template_override = None,
            extra_args = None,
            is_vision = True,
        )
        is False
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


# llama-server capability probe.


def _make_fake_llama_server(path: Path, help_text: str) -> Path:
    """Bash stub that prints `help_text` on --help."""
    path.write_text("#!/usr/bin/env bash\n" f"cat <<'EOF'\n{help_text}\nEOF\n")
    path.chmod(0o755)
    return path


_NEEDS_BASH = pytest.mark.skipif(
    sys.platform == "win32",
    reason = "fake llama-server is a bash stub; Windows has no direct executor",
)


def _clear_caps_cache():
    LlamaCppBackend._capability_cache.clear()


@_NEEDS_BASH
def test_probe_server_capabilities_detects_draft_mtp(tmp_path):
    # Original naming from llama.cpp #22673.
    fake = _make_fake_llama_server(
        tmp_path / "llama-server",
        "--spec-type none,draft-simple,draft-eagle3,draft-mtp,"
        "ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache",
    )
    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps["found"] is True
    assert caps["mtp_token"] == "draft-mtp"
    assert caps["supports_mtp"] is True


@_NEEDS_BASH
def test_probe_server_capabilities_detects_renamed_mtp(tmp_path):
    # Renamed upstream: draft-mtp -> mtp.
    fake = _make_fake_llama_server(
        tmp_path / "llama-server",
        "--spec-type [none|mtp|ngram-cache|ngram-simple|ngram-map-k|"
        "ngram-map-k4v|ngram-mod]",
    )
    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps["mtp_token"] == "mtp"
    assert caps["supports_mtp"] is True


@_NEEDS_BASH
def test_probe_server_capabilities_reports_outdated_binary(tmp_path):
    # Pre-MTP llama.cpp: only ngram variants.
    fake = _make_fake_llama_server(
        tmp_path / "llama-server",
        "--spec-type none,ngram-simple,ngram-mod",
    )
    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps["found"] is True
    assert caps["mtp_token"] is None
    assert caps["supports_mtp"] is False


def test_probe_server_capabilities_handles_missing_binary():
    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities("/no/such/llama-server")
    assert caps["found"] is False
    assert caps["supports_mtp"] is False


# ngram-mod flag flavor detection (new vs legacy llama-server).

# Help-text fixtures mirror the actual `llama-server --help` block
# layout (flag on its own line; description indented underneath).
_POST_RENAME_HELP = """\
--spec-draft-n-max N                    number of tokens to draft for speculative decoding (default: 16)
                                        (env: LLAMA_ARG_SPEC_DRAFT_N_MAX)
--spec-draft-n-min N                    minimum number of draft tokens to use for speculative decoding (default: 0)
                                        (env: LLAMA_ARG_SPEC_DRAFT_N_MIN)
--spec-draft-p-min, --draft-p-min P     minimum speculative decoding probability (greedy) (default: 0.75)
                                        (env: LLAMA_ARG_SPEC_DRAFT_P_MIN)
--spec-ngram-mod-n-min N                minimum number of ngram tokens (default: 48)
--spec-ngram-mod-n-max N                maximum number of ngram tokens (default: 64)
--spec-ngram-mod-n-match N              ngram-mod lookup length (default: 24)
--spec-type none,draft-simple,draft-mtp,ngram-mod                                        comma-separated list of types of speculative decoding to use
                                        (env: LLAMA_ARG_SPEC_TYPE)
--draft, --draft-n, --draft-max N       the argument has been removed. use --spec-draft-n-max or --spec-ngram-mod-n-max
                                        (env: LLAMA_ARG_DRAFT_MAX)
--draft-min, --draft-n-min N            the argument has been removed. use --spec-draft-n-min or --spec-ngram-mod-n-min
                                        (env: LLAMA_ARG_DRAFT_MIN)
--spec-ngram-size-n N                   the argument has been removed. use the respective --spec-ngram-*-size-n or --spec-ngram-mod-n-match
"""

_LEGACY_HELP = """\
--draft, --draft-n, --draft-max N       number of tokens to draft for speculative decoding (default: 8)
                                        (env: LLAMA_ARG_DRAFT_MAX)
--draft-min, --draft-n-min N            minimum number of draft tokens to use for speculative decoding (default: 0)
                                        (env: LLAMA_ARG_DRAFT_MIN)
--spec-ngram-size-n N                   ngram lookup length (default: 24)
--spec-type none,ngram-mod,ngram-simple                                        comma-separated list of types of speculative decoding to use
"""


@_NEEDS_BASH
def test_probe_detects_post_rename_ngram_mod_flavor(tmp_path):
    fake = _make_fake_llama_server(tmp_path / "llama-server", _POST_RENAME_HELP)
    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps["found"] is True
    assert caps["ngram_mod_flavor"] == "new"
    assert caps["supports_ngram_mod"] is True
    assert caps["spec_draft_n_max_flag"] == "--spec-draft-n-max"


@_NEEDS_BASH
def test_probe_detects_legacy_ngram_mod_flavor(tmp_path):
    fake = _make_fake_llama_server(tmp_path / "llama-server", _LEGACY_HELP)
    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps["found"] is True
    assert caps["ngram_mod_flavor"] == "legacy"
    assert caps["supports_ngram_mod"] is True
    assert caps["spec_draft_n_max_flag"] == "--draft-max"


@_NEEDS_BASH
def test_probe_ignores_removal_stub_descriptions(tmp_path):
    # Post-rename binary: legacy flags are present but with
    # "argument has been removed" descriptions; must not be detected
    # as legacy.
    fake = _make_fake_llama_server(tmp_path / "llama-server", _POST_RENAME_HELP)
    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps["ngram_mod_flavor"] == "new"


@_NEEDS_BASH
def test_probe_no_ngram_mod_on_minimal_binary(tmp_path):
    # Pre-anything: neither set present.
    fake = _make_fake_llama_server(
        tmp_path / "llama-server",
        "--spec-type none\n--threads N\n",
    )
    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps["ngram_mod_flavor"] is None
    assert caps["supports_ngram_mod"] is False


def test_build_ngram_mod_flags_new():
    flags = _build_ngram_mod_flags({"ngram_mod_flavor": "new"})
    assert flags == [
        "--spec-ngram-mod-n-match",
        "24",
        "--spec-ngram-mod-n-min",
        "48",
        "--spec-ngram-mod-n-max",
        "64",
    ]


def test_build_ngram_mod_flags_legacy():
    flags = _build_ngram_mod_flags({"ngram_mod_flavor": "legacy"})
    assert flags == [
        "--spec-ngram-size-n",
        "24",
        "--draft-min",
        "48",
        "--draft-max",
        "64",
    ]


def test_build_ngram_mod_flags_empty_when_unsupported():
    assert _build_ngram_mod_flags({"ngram_mod_flavor": None}) == []
    assert _build_ngram_mod_flags(None) == []
    assert _build_ngram_mod_flags({}) == []


def test_build_ngram_mod_flags_respects_custom_values():
    flags = _build_ngram_mod_flags(
        {"ngram_mod_flavor": "new"}, n_match = 16, n_min = 24, n_max = 32
    )
    assert flags == [
        "--spec-ngram-mod-n-match",
        "16",
        "--spec-ngram-mod-n-min",
        "24",
        "--spec-ngram-mod-n-max",
        "32",
    ]


@_NEEDS_BASH
def test_probe_server_capabilities_caches_by_mtime(tmp_path):
    # Same (path, mtime) -> cache hit. Bumped mtime -> re-probe.
    fake = _make_fake_llama_server(
        tmp_path / "llama-server",
        "--spec-type none,ngram-mod",
    )
    _clear_caps_cache()
    caps1 = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps1["supports_mtp"] is False

    import os
    import time

    _make_fake_llama_server(
        fake,
        "--spec-type none,draft-mtp,ngram-mod",
    )
    new_mtime = int(time.time()) + 2
    os.utime(fake, (new_mtime, new_mtime))
    caps2 = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps2["mtp_token"] == "draft-mtp"
    assert caps2["supports_mtp"] is True


# spec_draft_n_max plumbing (first-class --spec-draft-n-max override).


def test_already_in_target_state_matches_when_draft_n_max_unset():
    # None on the request means "platform default"; matches any backend.
    backend = _mtp_backend(_spec_draft_n_max = None)
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = "unsloth/Qwen3.6-27B-MTP-GGUF",
            hf_variant = "Q4_K_M",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = None,
            spec_draft_n_max = None,
            chat_template_override = None,
            extra_args = None,
            is_vision = False,
        )
        is True
    )


def test_already_in_target_state_matches_when_draft_n_max_equals_backend():
    backend = _mtp_backend(_spec_draft_n_max = 4)
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = "unsloth/Qwen3.6-27B-MTP-GGUF",
            hf_variant = "Q4_K_M",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = None,
            spec_draft_n_max = 4,
            chat_template_override = None,
            extra_args = None,
            is_vision = False,
        )
        is True
    )


def test_already_in_target_state_mismatches_when_draft_n_max_differs():
    backend = _mtp_backend(_spec_draft_n_max = 4)
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = "unsloth/Qwen3.6-27B-MTP-GGUF",
            hf_variant = "Q4_K_M",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = None,
            spec_draft_n_max = 8,
            chat_template_override = None,
            extra_args = None,
            is_vision = False,
        )
        is False
    )


def test_already_in_target_state_draft_n_max_ignored_when_not_mtp():
    # ngram-mod backend; spec_draft_n_max is MTP-only and must not force
    # a reload against a non-MTP active spec.
    backend = _mtp_backend(_speculative_type = "ngram-mod", _spec_draft_n_max = None)
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = "unsloth/Qwen3.6-27B-MTP-GGUF",
            hf_variant = "Q4_K_M",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = "ngram-mod",
            spec_draft_n_max = 8,
            chat_template_override = None,
            extra_args = None,
            is_vision = False,
        )
        is True
    )


# Sub-3B MTP gate -- tiny dense models regress with the MTP draft
# head, so load_model falls back to ngram-mod (when the binary supports
# it) instead of draft-mtp. The reload-skip mirror must follow the
# same fallback so a sub-3B reload-with-default does not bounce a
# correctly-configured ngram-mod / off backend.


def _patch_probe(monkeypatch, ngram_supported):
    """Force probe_server_capabilities to a deterministic result so
    tests don't depend on whatever llama-server happens to be on PATH."""
    fake = {
        "found": True,
        "mtp_token": "draft-mtp",
        "supports_mtp": True,
        "ngram_mod_flavor": "new" if ngram_supported else None,
        "supports_ngram_mod": bool(ngram_supported),
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary=None: fake),
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_find_llama_server_binary",
        classmethod(lambda cls: "/fake/llama-server"),
    )


def test_already_in_target_state_sub_3b_falls_back_to_ngram_mod_when_supported(monkeypatch):
    # 0.8B MTP request -- load_model would have promoted to ngram-mod
    # (no MTP head); reload check must match a ngram-mod backend.
    _patch_probe(monkeypatch, ngram_supported=True)
    backend = _mtp_backend(
        _model_identifier = "unsloth/Qwen3.5-0.8B-MTP-GGUF",
        _speculative_type = "ngram-mod",
        _spec_draft_n_max = None,
    )
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = "unsloth/Qwen3.5-0.8B-MTP-GGUF",
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


def test_already_in_target_state_sub_3b_falls_back_to_off_when_no_ngram(monkeypatch):
    # 0.8B + binary lacks ngram-mod -> fall back to off.
    _patch_probe(monkeypatch, ngram_supported=False)
    backend = _mtp_backend(
        _model_identifier = "unsloth/Qwen3.5-0.8B-MTP-GGUF",
        _speculative_type = None,
        _spec_draft_n_max = None,
    )
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = "unsloth/Qwen3.5-0.8B-MTP-GGUF",
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


def test_already_in_target_state_4b_mtp_request_promotes_as_before(monkeypatch):
    # 4B is above the 3B threshold -> auto-promote still applies.
    _patch_probe(monkeypatch, ngram_supported=True)
    backend = _mtp_backend(
        _model_identifier = "unsloth/Qwen3.5-4B-MTP-GGUF",
        _speculative_type = "draft-mtp",
        _spec_draft_n_max = None,
    )
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = "unsloth/Qwen3.5-4B-MTP-GGUF",
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


def test_already_in_target_state_2b_falls_back_to_ngram_below_threshold(monkeypatch):
    # 2.0B is below the 3B threshold -> ngram-mod fallback, not
    # draft-mtp. Clean-bench shows 2B regresses with draft-mtp.
    _patch_probe(monkeypatch, ngram_supported=True)
    backend = _mtp_backend(
        _model_identifier = "unsloth/Qwen3.5-2B-MTP-GGUF",
        _speculative_type = "ngram-mod",
        _spec_draft_n_max = None,
    )
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = "unsloth/Qwen3.5-2B-MTP-GGUF",
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


# usage backfill from timings (Studio UI t/s widget fix).


def test_backfill_usage_from_timings_fills_when_completion_tokens_zero():
    out = _backfill_usage_from_timings(
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        {"prompt_n": 42, "predicted_n": 128, "predicted_per_second": 100.0},
    )
    assert out["completion_tokens"] == 128
    assert out["prompt_tokens"] == 42
    assert out["total_tokens"] == 170


def test_backfill_usage_from_timings_fills_when_usage_missing():
    out = _backfill_usage_from_timings(
        None,
        {"prompt_n": 42, "predicted_n": 128, "predicted_per_second": 100.0},
    )
    assert out["completion_tokens"] == 128
    assert out["prompt_tokens"] == 42
    assert out["total_tokens"] == 170


def test_backfill_usage_from_timings_preserves_real_usage():
    # Non-zero completion_tokens means llama-server reported correctly;
    # do not overwrite.
    real = {"prompt_tokens": 50, "completion_tokens": 200, "total_tokens": 250}
    out = _backfill_usage_from_timings(real, {"predicted_n": 999, "prompt_n": 999})
    assert out is real
    assert out["completion_tokens"] == 200


def test_backfill_usage_from_timings_passthrough_when_timings_empty():
    assert _backfill_usage_from_timings(None, None) is None
    assert _backfill_usage_from_timings(None, {}) is None
    usage = {"completion_tokens": 0}
    # No timings.predicted_n -> nothing to fill, return as-is.
    assert _backfill_usage_from_timings(usage, {"prompt_ms": 5.0}) is usage
