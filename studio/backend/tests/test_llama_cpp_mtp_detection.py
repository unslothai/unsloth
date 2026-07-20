# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the MTP auto-detection path (llama.cpp #22673).

Pins three contracts: name-based detector, user-override detector, and
the _already_in_target_state mirror that prevents needless reloads.
"""

from __future__ import annotations

import inspect
import os
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
    _GPU_OFFLOAD_OVERRIDE_FLAGS,
    _THREAD_OVERRIDE_FLAGS,
    _backfill_usage_from_timings,
    _build_ngram_mod_flags,
    _canonicalize_spec_mode,
    _extra_args_set_any_flag,
    _extra_args_set_spec_type,
    _is_mtp_model_name,
    _mla_mtp_auto_enabled,
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
    return _enc_string(key) + struct.pack("<I", _VTYPE_UINT32) + struct.pack("<I", value)


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
    # Fixture simulates Auto having auto-promoted to draft-mtp. Tests
    # override _requested_spec_mode for a forced mode or the
    # user---spec-type-extra-args path.
    backend._requested_spec_mode = "auto"
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


def test_already_in_target_state_auto_request_matches_auto_backend_for_non_mtp_model():
    # In the requested-mode round-trip model, Auto-vs-Auto matches regardless
    # of model name. The resolved emission (--spec-default vs draft-mtp) is
    # handled by the load path and reflected in _speculative_type; the
    # short-circuit only cares whether the *intent* changed.
    backend = _mtp_backend(
        _model_identifier = "unsloth/Qwen3.6-27B-GGUF",
        _speculative_type = "default",
    )
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
        is True
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


@pytest.mark.parametrize(
    "extra_args",
    [
        ["-ngl", "12"],
        ["--gpu-layers", "12"],
        ["--n-gpu-layers=12"],
        ["-fit", "off"],
        ["--fit=off"],
    ],
)
def test_extra_args_detect_gpu_offload_overrides(extra_args):
    assert _extra_args_set_any_flag(extra_args, _GPU_OFFLOAD_OVERRIDE_FLAGS) is True


@pytest.mark.parametrize("extra_args", [["-t", "8"], ["--threads=8"]])
def test_extra_args_detect_thread_overrides(extra_args):
    assert _extra_args_set_any_flag(extra_args, _THREAD_OVERRIDE_FLAGS) is True


def test_windows_full_offload_flags_use_current_llama_server_args():
    src = inspect.getsource(LlamaCppBackend.load_model)
    stale_checkpoint_flag = "--checkpoint-" + "every-n-tokens"
    assert '"--cache-ram"' in src
    assert '"--ctx-checkpoints"' in src
    assert '"--no-cache-prompt"' in src
    assert stale_checkpoint_flag not in src


def test_load_model_sets_threads_once():
    src = inspect.getsource(LlamaCppBackend.load_model)
    assert src.count('cmd.extend(["--threads", str(') == 1


def test_llama_cpp_annotations_stay_python39_safe():
    src = inspect.getsource(LlamaCppBackend.generate_chat_completion)
    helper_src = inspect.getsource(_extra_args_set_any_flag)
    assert "Generator[str | dict" not in src
    assert "set[str] | frozenset[str]" not in helper_src


def test_already_in_target_state_user_spec_type_override_matches_clean_backend():
    # User --spec-type none suppressed auto-MTP; repeat /load must not re-promote.
    backend = _mtp_backend(
        _speculative_type = None,
        _requested_spec_mode = None,
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


def test_already_in_target_state_vision_off_matches_vision_backend():
    # Vision loads drop speculative decoding at the route level (req -> "off").
    # _already_in_target_state compares canonical requested modes; a vision
    # backend with _requested_spec_mode="off" matches req "off" or None+vision.
    backend = _mtp_backend(
        _model_identifier = "unsloth/Qwen3-VL-4B-Instruct-GGUF",
        _is_vision = True,
        _speculative_type = None,
        _requested_spec_mode = "off",
    )
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = "unsloth/Qwen3-VL-4B-Instruct-GGUF",
            hf_variant = "Q4_K_M",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = "off",
            chat_template_override = None,
            extra_args = None,
            is_vision = True,
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


# llama-server capability probe.


def _make_fake_llama_server(path: Path, help_text: str) -> Path:
    """Bash stub that prints `help_text` on --help."""
    path.write_text(f"#!/usr/bin/env bash\ncat <<'EOF'\n{help_text}\nEOF\n")
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
def test_probe_server_capabilities_uses_binary_library_env(tmp_path, monkeypatch):
    fake = _make_fake_llama_server(
        tmp_path / "llama-server",
        "--spec-type none,mtp,ngram-simple\n",
    )
    captured = {}

    monkeypatch.setattr(
        "core.inference.llama_cpp.child_env_without_native_path_secret",
        lambda: {"LD_LIBRARY_PATH": "/already-there"},
    )

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return _types.SimpleNamespace(stdout = "--spec-type none,mtp,ngram-simple\n", stderr = "")

    monkeypatch.setattr("core.inference.llama_cpp.subprocess.run", fake_run)

    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))

    assert caps["found"] is True
    assert caps["supports_mtp"] is True
    assert captured["cmd"] == [str(fake), "--help"]
    assert captured["env"] is not None
    ld_dirs = captured["env"]["LD_LIBRARY_PATH"].split(os.pathsep)
    assert str(fake.parent) in ld_dirs
    assert "/already-there" in ld_dirs


@_NEEDS_BASH
def test_probe_server_capabilities_detects_renamed_mtp(tmp_path):
    # Renamed upstream: draft-mtp -> mtp.
    fake = _make_fake_llama_server(
        tmp_path / "llama-server",
        "--spec-type [none|mtp|ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod]",
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
    assert caps["supports_cache_ram"] is False
    assert caps["supports_ctx_checkpoints"] is False
    assert caps["supports_no_cache_prompt"] is False


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

_CACHE_FLAGS_HELP = """\
--cache-ram N                           store prompt cache in RAM (default: 0)
--ctx-checkpoints N                     number of context checkpoints (default: 0)
--no-cache-prompt                       do not reuse prompt cache
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
    # Post-rename binary: legacy flags present but with "argument has been
    # removed" descriptions; must not be detected as legacy.
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


@_NEEDS_BASH
def test_probe_detects_windows_cache_flags(tmp_path):
    fake = _make_fake_llama_server(tmp_path / "llama-server", _CACHE_FLAGS_HELP)
    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps["supports_cache_ram"] is True
    assert caps["supports_ctx_checkpoints"] is True
    assert caps["supports_no_cache_prompt"] is True


@_NEEDS_BASH
def test_probe_reports_windows_cache_flags_absent_for_older_binary(tmp_path):
    fake = _make_fake_llama_server(tmp_path / "llama-server", "--threads N\n")
    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps["supports_cache_ram"] is False
    assert caps["supports_ctx_checkpoints"] is False
    assert caps["supports_no_cache_prompt"] is False


@_NEEDS_BASH
def test_probe_detects_slot_save_path(tmp_path):
    fake = _make_fake_llama_server(
        tmp_path / "llama-server",
        "--slot-save-path PATH  path to save slot kv cache\n--threads N\n",
    )
    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps["supports_slot_save"] is True


@_NEEDS_BASH
def test_probe_reports_slot_save_absent_for_older_binary(tmp_path):
    fake = _make_fake_llama_server(tmp_path / "llama-server", "--threads N\n")
    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps["supports_slot_save"] is False


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
    assert flags == ["--spec-ngram-size-n", "24", "--draft-min", "48", "--draft-max", "64"]


def test_build_ngram_mod_flags_empty_when_unsupported():
    assert _build_ngram_mod_flags({"ngram_mod_flavor": None}) == []
    assert _build_ngram_mod_flags(None) == []
    assert _build_ngram_mod_flags({}) == []


def test_build_ngram_mod_flags_respects_custom_values():
    flags = _build_ngram_mod_flags({"ngram_mod_flavor": "new"}, n_match = 16, n_min = 24, n_max = 32)
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
    backend = _mtp_backend(
        _speculative_type = "ngram-mod",
        _requested_spec_mode = "ngram",
        _spec_draft_n_max = None,
    )
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


# Sub-3B MTP gate -- tiny dense models regress with the MTP draft head, so
# load_model falls back to ngram-mod (when the binary supports it) instead of
# draft-mtp. The reload-skip mirror must follow the same fallback so a sub-3B
# reload-with-default doesn't bounce a correctly-configured ngram-mod/off backend.


def _patch_probe(monkeypatch, ngram_supported):
    """Force probe_server_capabilities to a deterministic result so tests
    don't depend on whatever llama-server is on PATH."""
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
        classmethod(lambda cls, binary = None: fake),
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_find_llama_server_binary",
        classmethod(lambda cls: "/fake/llama-server"),
    )


def test_already_in_target_state_sub_3b_falls_back_to_ngram_mod_when_supported(monkeypatch):
    # 0.8B MTP request -- load_model would have promoted to ngram-mod (no MTP
    # head); reload check must match a ngram-mod backend.
    _patch_probe(monkeypatch, ngram_supported = True)
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
    _patch_probe(monkeypatch, ngram_supported = False)
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
    _patch_probe(monkeypatch, ngram_supported = True)
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
    # 2.0B is below the 3B threshold -> ngram-mod fallback, not draft-mtp.
    # Clean-bench shows 2B regresses with draft-mtp.
    _patch_probe(monkeypatch, ngram_supported = True)
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


# usage backfill from timings (Unsloth UI t/s widget fix).


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


# ── _canonicalize_spec_mode (pure) ─────────────────────────────────


@pytest.mark.parametrize(
    "value, expected",
    [
        # New canonical values pass through unchanged.
        ("auto", "auto"),
        ("mtp", "mtp"),
        ("ngram", "ngram"),
        ("mtp+ngram", "mtp+ngram"),
        ("off", "off"),
        ("ngram-simple", "ngram-simple"),
        # Legacy wire values map onto the new vocabulary.
        ("default", "auto"),
        ("draft-mtp", "mtp"),
        ("ngram-mod", "ngram"),
        # Comma-chained legacy values (e.g. from persisted state) collapse
        # to the right canonical mode.
        ("ngram-mod,draft-mtp", "mtp+ngram"),
        ("draft-mtp,ngram-mod", "mtp+ngram"),
        ("draft-mtp,mtp", "mtp"),
        ("ngram-mod,ngram", "ngram"),
        # Case and whitespace are ignored.
        ("  AUTO  ", "auto"),
        ("MTP", "mtp"),
        ("MTP+Ngram", "mtp+ngram"),
        # None / empty / whitespace pass through as None.
        (None, None),
        ("", None),
        ("   ", None),
        # Non-string inputs collapse to None.
        (42, None),
        (True, None),
        # Unknown strings fall back to "auto" (safe default).
        ("bogus", "auto"),
    ],
)
def test_canonicalize_spec_mode(value, expected):
    assert _canonicalize_spec_mode(value) == expected


# ── _build_speculative_flags resolver matrix ──────────────────────


def _resolver_backend(
    monkeypatch,
    *,
    ngram_supported = True,
    mtp_token = "draft-mtp",
):
    """Backend with a deterministic probe so the resolver is hermetic."""
    fake = {
        "found": True,
        "mtp_token": mtp_token,
        "supports_mtp": bool(mtp_token),
        "ngram_mod_flavor": "new" if ngram_supported else None,
        "supports_ngram_mod": bool(ngram_supported),
        "spec_draft_n_max_flag": "--spec-draft-n-max",
    }
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: fake),
    )
    backend = LlamaCppBackend()
    backend._nextn_predict_layers = None
    return backend


def _flags_dict(flags):
    """Parse the spec-flag list into a {flag: value} dict; collapses repeated
    flags by keeping the last (only --spec-type can repeat, and never does
    in our resolver)."""
    out = {}
    i = 0
    while i < len(flags):
        token = flags[i]
        if i + 1 < len(flags) and not flags[i + 1].startswith("--"):
            out[token] = flags[i + 1]
            i += 2
        else:
            out[token] = True
            i += 1
    return out


_MTP_MODEL = "unsloth/Qwen3.6-27B-MTP-GGUF"
_NON_MTP_MODEL = "unsloth/Qwen3-7B-Instruct-GGUF"
_SUB_3B_MTP_MODEL = "unsloth/Qwen3.5-0.8B-MTP-GGUF"


@pytest.mark.parametrize(
    "requested, gpus, model, expect_spec_type, expect_n_max, expect_ngram_knobs",
    [
        # ── auto + MTP model + 3B+: GPU = mtp only, CPU = chain ──
        ("auto", True, _MTP_MODEL, "draft-mtp", "2", False),
        ("auto", False, _MTP_MODEL, "ngram-mod,draft-mtp", "3", True),
        # ── auto + non-MTP: emit --spec-default ──
        ("auto", True, _NON_MTP_MODEL, None, None, False),
        ("auto", False, _NON_MTP_MODEL, None, None, False),
        # ── auto + sub-3B MTP: fallback to ngram-mod ──
        ("auto", True, _SUB_3B_MTP_MODEL, "ngram-mod", None, True),
        ("auto", False, _SUB_3B_MTP_MODEL, "ngram-mod", None, True),
        # ── mtp forced: MTP-only on BOTH platforms ──
        ("mtp", True, _MTP_MODEL, "draft-mtp", "2", False),
        ("mtp", False, _MTP_MODEL, "draft-mtp", "3", False),
        # ── mtp forced on sub-3B: engage anyway ──
        ("mtp", True, _SUB_3B_MTP_MODEL, "draft-mtp", "2", False),
        # ── mtp forced on non-MTP: default back (no head/drafter) ──
        ("mtp", True, _NON_MTP_MODEL, None, None, False),
        # ── ngram forced: ngram-mod alone on BOTH platforms ──
        ("ngram", True, _MTP_MODEL, "ngram-mod", None, True),
        ("ngram", False, _MTP_MODEL, "ngram-mod", None, True),
        ("ngram", True, _NON_MTP_MODEL, "ngram-mod", None, True),
        # ── mtp+ngram forced: chain on BOTH platforms ──
        ("mtp+ngram", True, _MTP_MODEL, "ngram-mod,draft-mtp", "2", True),
        ("mtp+ngram", False, _MTP_MODEL, "ngram-mod,draft-mtp", "3", True),
        ("mtp+ngram", True, _SUB_3B_MTP_MODEL, "ngram-mod,draft-mtp", "2", True),
        # ── mtp+ngram forced on non-MTP: keep ngram, drop draft-mtp ──
        ("mtp+ngram", True, _NON_MTP_MODEL, "ngram-mod", None, True),
        # ── off: nothing emitted ──
        ("off", True, _MTP_MODEL, None, None, False),
        ("off", False, _MTP_MODEL, None, None, False),
        # ── legacy values round-trip to the canonical emission ──
        ("default", True, _MTP_MODEL, "draft-mtp", "2", False),
        ("draft-mtp", True, _MTP_MODEL, "draft-mtp", "2", False),
        ("ngram-mod", True, _MTP_MODEL, "ngram-mod", None, True),
        ("ngram-mod,draft-mtp", False, _MTP_MODEL, "ngram-mod,draft-mtp", "3", True),
        # ── ngram-simple: pass through ──
        ("ngram-simple", True, _MTP_MODEL, "ngram-simple", None, False),
    ],
)
def test_build_speculative_flags_matrix(
    monkeypatch, requested, gpus, model, expect_spec_type, expect_n_max, expect_ngram_knobs
):
    backend = _resolver_backend(monkeypatch)
    flags = backend._build_speculative_flags(
        speculative_type = requested,
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = model,
        model_path = None,
        gpus = gpus,
        binary = "/fake/llama-server",
    )
    parsed = _flags_dict(flags)
    if expect_spec_type is None:
        assert "--spec-type" not in parsed
    else:
        assert parsed.get("--spec-type") == expect_spec_type
    if expect_n_max is None:
        assert "--spec-draft-n-max" not in parsed
    else:
        assert parsed.get("--spec-draft-n-max") == expect_n_max
    if expect_ngram_knobs:
        assert "--spec-ngram-mod-n-match" in parsed
        assert "--spec-ngram-mod-n-min" in parsed
        assert "--spec-ngram-mod-n-max" in parsed
    else:
        assert "--spec-ngram-mod-n-match" not in parsed


def test_build_speculative_flags_user_extra_args_owns_spec_type(monkeypatch):
    # User --spec-type in extra_args bypasses the dropdown entirely.
    backend = _resolver_backend(monkeypatch)
    flags = backend._build_speculative_flags(
        speculative_type = "mtp",  # would normally force MTP
        spec_draft_n_max = None,
        extra_args = ["--spec-type", "ngram-mod"],
        model_identifier = _MTP_MODEL,
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
    )
    # Resolver emits nothing -- the user's extra_args carries the --spec-type,
    # and the resolver records requested_spec_mode = None.
    assert flags == []
    assert backend.requested_spec_mode is None
    assert backend.speculative_type is None


@pytest.mark.parametrize("mode", ["auto", "mtp", "ngram", "mtp+ngram", "off"])
def test_build_speculative_flags_round_trips_requested_mode(monkeypatch, mode):
    # The status round-trip is the contract that lets the UI dropdown
    # restore its picked value after reload / refresh.
    backend = _resolver_backend(monkeypatch)
    backend._build_speculative_flags(
        speculative_type = mode,
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = _MTP_MODEL,
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
    )
    assert backend.requested_spec_mode == mode


def test_build_speculative_flags_user_draft_n_max_override(monkeypatch):
    backend = _resolver_backend(monkeypatch)
    flags = backend._build_speculative_flags(
        speculative_type = "mtp",
        spec_draft_n_max = 5,
        extra_args = None,
        model_identifier = _MTP_MODEL,
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
    )
    parsed = _flags_dict(flags)
    assert parsed.get("--spec-draft-n-max") == "5"
    assert backend.spec_draft_n_max == 5


def test_build_speculative_flags_mtp_token_missing_emits_spec_default(monkeypatch):
    # Outdated llama-server with no MTP support: forced MTP must degrade (warned)
    # and emit --spec-default so an inherited LLAMA_ARG_SPEC_TYPE=draft-mtp (CLI
    # wins over env) can't make the child attempt MTP the gate budgeted off.
    backend = _resolver_backend(monkeypatch, mtp_token = None)
    flags = backend._build_speculative_flags(
        speculative_type = "mtp",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = _MTP_MODEL,
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
    )
    assert "--spec-type" not in flags
    assert "--spec-default" in flags
    # Degraded to non-speculative; the user's choice is still reflected.
    assert backend.speculative_type == "default"
    assert backend.requested_spec_mode == "mtp"
    assert backend.spec_fallback_reason == "binary_no_mtp"


def test_forced_mtp_on_non_mtp_model_defaults_back(monkeypatch):
    # Forcing MTP on a model with no head/drafter must NOT emit draft-mtp:
    # llama-server aborts on it ("failed to measure MTP context memory")
    # rather than no-op'ing. Default back to --spec-default instead.
    backend = _resolver_backend(monkeypatch)
    flags = backend._build_speculative_flags(
        speculative_type = "mtp",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = _NON_MTP_MODEL,
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
    )
    assert "--spec-type" not in flags
    assert "--spec-default" in flags
    assert backend.speculative_type == "default"
    assert backend.requested_spec_mode == "mtp"


def test_forced_mtp_ngram_on_non_mtp_model_keeps_ngram(monkeypatch):
    # mtp+ngram on a non-MTP model drops the doomed draft-mtp chain but keeps
    # the ngram half, which needs no head.
    backend = _resolver_backend(monkeypatch)
    flags = backend._build_speculative_flags(
        speculative_type = "mtp+ngram",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = _NON_MTP_MODEL,
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
    )
    parsed = _flags_dict(flags)
    assert parsed.get("--spec-type") == "ngram-mod"
    assert backend.speculative_type == "ngram-mod"
    assert backend.requested_spec_mode == "mtp+ngram"


# ── Auto drops embedded MTP for MLA models (GLM-5.2 et al.) ───────────
#
# llama.cpp's MLA/DSA MTP path runs ~2x slower than no speculation (GLM-5.2
# bench), so Auto downgrades it to ngram-mod (or spec-off). The clean
# metadata separator from non-MLA MTP (Qwen, kept on draft-mtp) is
# self._kv_lora_rank. Forced mtp / mtp+ngram and separate drafters (Gemma)
# stay on draft-mtp; UNSLOTH_MLA_MTP_ENABLED=1 re-enables Auto promotion.

# GLM-5.2's repo name has no "MTP" marker, so its MTP signal is metadata-only
# (nextn_predict_layers) -- exactly the embedded-MLA case we gate.
_GLM_MLA_MODEL = "unsloth/GLM-5.2-GGUF"


def _mla_resolver_backend(
    monkeypatch,
    *,
    ngram_supported = True,
    kv_lora_rank = 512,
    nextn = 1,
):
    """Resolver backend posing as an embedded-MTP MLA model (kv_lora_rank set)."""
    backend = _resolver_backend(monkeypatch, ngram_supported = ngram_supported)
    backend._nextn_predict_layers = nextn
    backend._kv_lora_rank = kv_lora_rank
    return backend


@pytest.mark.parametrize("gpus", [True, False])
def test_auto_mla_embedded_mtp_falls_back_to_ngram(monkeypatch, gpus):
    # Auto + MLA embedded MTP + ngram supported -> ngram-mod on BOTH platforms
    # (the CPU chain ngram-mod,draft-mtp is dropped: no draft-mtp for MLA).
    backend = _mla_resolver_backend(monkeypatch)
    flags = backend._build_speculative_flags(
        speculative_type = "auto",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = _GLM_MLA_MODEL,
        model_path = None,
        gpus = gpus,
        binary = "/fake/llama-server",
    )
    parsed = _flags_dict(flags)
    assert parsed.get("--spec-type") == "ngram-mod"
    assert "--spec-draft-n-max" not in parsed
    assert "--spec-ngram-mod-n-match" in parsed
    assert backend.speculative_type == "ngram-mod"
    assert backend.requested_spec_mode == "auto"
    assert backend.spec_fallback_reason == "mla_mtp_disabled"
    assert backend.spec_draft_n_max is None


def test_auto_mla_embedded_mtp_no_ngram_disables_spec(monkeypatch):
    # Auto + MLA embedded MTP + no ngram-mod support -> emit nothing (spec-off),
    # mirroring the sub-3B no-ngram path. Still flagged as a policy downgrade.
    backend = _mla_resolver_backend(monkeypatch, ngram_supported = False)
    flags = backend._build_speculative_flags(
        speculative_type = "auto",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = _GLM_MLA_MODEL,
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
    )
    assert "--spec-type" not in flags
    assert backend.speculative_type is None
    assert backend.requested_spec_mode == "auto"
    assert backend.spec_fallback_reason == "mla_mtp_disabled"


def test_auto_non_mla_embedded_mtp_keeps_draft_mtp(monkeypatch):
    # Auto + embedded MTP + NON-MLA (kv_lora_rank None, e.g. Qwen) -> unchanged:
    # still draft-mtp at the platform default. No policy downgrade.
    backend = _mla_resolver_backend(monkeypatch, kv_lora_rank = None)
    flags = backend._build_speculative_flags(
        speculative_type = "auto",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = _MTP_MODEL,
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
    )
    parsed = _flags_dict(flags)
    assert parsed.get("--spec-type") == "draft-mtp"
    assert parsed.get("--spec-draft-n-max") == "2"
    assert backend.speculative_type == "draft-mtp"
    assert backend.spec_fallback_reason is None


def test_auto_mla_separate_drafter_keeps_mtp(monkeypatch):
    # Auto + MLA + a separate drafter (mtp_draft_path) -> the drafter exemption
    # wins over the MLA gate: still draft-mtp (Gemma-style external drafter is
    # not the slow embedded MLA/DSA path).
    backend = _mla_resolver_backend(monkeypatch)
    flags = backend._build_speculative_flags(
        speculative_type = "auto",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = _GLM_MLA_MODEL,
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
        mtp_draft_path = "/fake/mtp-draft.gguf",
    )
    parsed = _flags_dict(flags)
    assert parsed.get("--spec-type") == "draft-mtp"
    assert backend.speculative_type == "draft-mtp"
    assert backend.spec_fallback_reason is None


def test_auto_non_mtp_mla_model_unaffected(monkeypatch):
    # Auto + MLA but NO embedded MTP head (kv_lora_rank set, nextn None, e.g.
    # GLM-4.7-Flash) -> non-MTP default; no accidental ngram drop.
    backend = _mla_resolver_backend(monkeypatch, nextn = None)
    flags = backend._build_speculative_flags(
        speculative_type = "auto",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = "unsloth/GLM-4.7-Flash-GGUF",
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
    )
    assert "--spec-default" in flags
    assert "ngram-mod" not in flags
    assert backend.speculative_type == "default"
    assert backend.spec_fallback_reason is None


@pytest.mark.parametrize(
    "mode, expect_spec_type, expect_n_max",
    [
        ("mtp", "draft-mtp", "2"),
        ("mtp+ngram", "ngram-mod,draft-mtp", "2"),
    ],
)
def test_forced_mtp_on_mla_still_engages(monkeypatch, mode, expect_spec_type, expect_n_max):
    # Explicit override engages the deliberately-slower MTP route on MLA models,
    # regardless of the Auto gate. No policy downgrade reason.
    backend = _mla_resolver_backend(monkeypatch)
    flags = backend._build_speculative_flags(
        speculative_type = mode,
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = _GLM_MLA_MODEL,
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
    )
    parsed = _flags_dict(flags)
    assert parsed.get("--spec-type") == expect_spec_type
    assert parsed.get("--spec-draft-n-max") == expect_n_max
    assert backend.speculative_type == "draft-mtp"
    assert backend.requested_spec_mode == mode
    assert backend.spec_fallback_reason is None


def test_env_flag_reenables_auto_mla_mtp(monkeypatch):
    # UNSLOTH_MLA_MTP_ENABLED=1 -> Auto promotes MLA embedded MTP to draft-mtp
    # again (the forward hook for when llama.cpp optimizes the path).
    monkeypatch.setenv("UNSLOTH_MLA_MTP_ENABLED", "1")
    backend = _mla_resolver_backend(monkeypatch)
    flags = backend._build_speculative_flags(
        speculative_type = "auto",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = _GLM_MLA_MODEL,
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
    )
    parsed = _flags_dict(flags)
    assert parsed.get("--spec-type") == "draft-mtp"
    assert backend.speculative_type == "draft-mtp"
    assert backend.spec_fallback_reason is None


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "On"])
def test_mla_mtp_auto_enabled_truthy_values(monkeypatch, value):
    monkeypatch.setenv("UNSLOTH_MLA_MTP_ENABLED", value)
    assert _mla_mtp_auto_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "  ", "bogus"])
def test_mla_mtp_auto_disabled_default_and_falsy(monkeypatch, value):
    monkeypatch.setenv("UNSLOTH_MLA_MTP_ENABLED", value)
    assert _mla_mtp_auto_enabled() is False


def test_mla_mtp_auto_disabled_when_unset(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MLA_MTP_ENABLED", raising = False)
    assert _mla_mtp_auto_enabled() is False


def test_read_gguf_metadata_captures_kv_lora_rank(tmp_path):
    # GLM-5.2-style header: MLA (kv_lora_rank) + embedded MTP (nextn) populate
    # both fields, so the Auto gate sees an MLA embedded-MTP model.
    gguf = _write_minimal_gguf(
        tmp_path / "model.gguf",
        arch = "glm-dsa",
        nextn = 1,
        extra_uint32 = {
            "glm-dsa.block_count": 4,
            "glm-dsa.attention.kv_lora_rank": 512,
        },
    )
    backend = LlamaCppBackend()
    backend._read_gguf_metadata(str(gguf))
    assert backend._nextn_predict_layers == 1
    assert backend._kv_lora_rank == 512


def test_read_gguf_metadata_qwen_mtp_has_no_kv_lora_rank(tmp_path):
    # Qwen MTP header: embedded MTP but non-MLA, so kv_lora_rank stays None and
    # Auto keeps it on draft-mtp.
    gguf = _write_minimal_gguf(
        tmp_path / "model.gguf",
        arch = "qwen35moe",
        nextn = 1,
        extra_uint32 = {"qwen35moe.block_count": 4},
    )
    backend = LlamaCppBackend()
    backend._read_gguf_metadata(str(gguf))
    assert backend._nextn_predict_layers == 1
    assert backend._kv_lora_rank is None


def test_reload_skip_auto_mla_ngram_is_idempotent():
    # A GLM model resolved to ngram-mod under Auto must not churn: a duplicate
    # Auto /load at the same settings is already-satisfied.
    backend = _mtp_backend(
        _model_identifier = _GLM_MLA_MODEL,
        _speculative_type = "ngram-mod",
        _requested_spec_mode = "auto",
    )
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = _GLM_MLA_MODEL,
            hf_variant = "Q4_K_M",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = "auto",
            chat_template_override = None,
            extra_args = None,
            is_vision = False,
        )
        is True
    )


def test_reload_forced_mtp_bounces_auto_mla():
    # Overriding Auto (ngram-mod) with a forced mtp request must reload (to the
    # slower draft-mtp route), not dedup against the running ngram-mod server.
    backend = _mtp_backend(
        _model_identifier = _GLM_MLA_MODEL,
        _speculative_type = "ngram-mod",
        _requested_spec_mode = "auto",
    )
    assert (
        backend._already_in_target_state(
            gguf_path = None,
            model_identifier = _GLM_MLA_MODEL,
            hf_variant = "Q4_K_M",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = "mtp",
            chat_template_override = None,
            extra_args = None,
            is_vision = False,
        )
        is False
    )


# ── Full named-repo resolver matrix (the shipping Unsloth families) ─────
#
# Locks auto / off / forced-mtp routing for every Qwen3.5 (MTP + plain) and
# gemma-4 (regular + QAT) GGUF repo, including the giant MoEs that stay
# resolver-only (122B-A10B / 397B-A17B). Expectations are derived from the
# same signals load_model uses -- _extract_model_size_b (active>effective>
# total, so E2B->2, A3B->3, A10B->10, A17B->17), _is_mtp_model_name, and the
# separate-drafter flag -- so each row mirrors what the loader emits on a
# B200 (GPU default, n=2). gemma carries no -MTP marker; its MTP comes from
# the root mtp-*.gguf drafter, modelled here by passing mtp_draft_path.
#
# auto_spec: "draft-mtp" = head/drafter engaged (>=3B MTP, or any size with a
# separate drafter); "ngram-mod" = embedded sub-3B drop (zero-VRAM); None =
# non-MTP -> llama-server --spec-default.

_GEMMA_DRAFTER = "/snap/mtp-gemma-4-it.gguf"  # stand-in separate drafter

_REAL_REPO_MATRIX = [
    # repo, drafter, auto_spec, auto_ngram_knobs
    ("unsloth/Qwen3.5-0.8B-MTP-GGUF", None, "ngram-mod", True),
    ("unsloth/Qwen3.5-2B-MTP-GGUF", None, "ngram-mod", True),
    ("unsloth/Qwen3.5-4B-MTP-GGUF", None, "draft-mtp", False),
    ("unsloth/Qwen3.5-9B-MTP-GGUF", None, "draft-mtp", False),
    ("unsloth/Qwen3.5-27B-MTP-GGUF", None, "draft-mtp", False),
    ("unsloth/Qwen3.5-35B-A3B-MTP-GGUF", None, "draft-mtp", False),
    ("unsloth/Qwen3.5-122B-A10B-MTP-GGUF", None, "draft-mtp", False),
    ("unsloth/Qwen3.5-397B-A17B-MTP-GGUF", None, "draft-mtp", False),
    ("unsloth/Qwen3.5-0.8B-GGUF", None, None, False),
    ("unsloth/Qwen3.5-2B-GGUF", None, None, False),
    ("unsloth/Qwen3.5-4B-GGUF", None, None, False),
    ("unsloth/Qwen3.5-9B-GGUF", None, None, False),
    # E2B is 2B but ships a separate drafter -> exempt from the sub-3B drop.
    ("unsloth/gemma-4-E2B-it-GGUF", _GEMMA_DRAFTER, "draft-mtp", False),
    ("unsloth/gemma-4-E4B-it-GGUF", _GEMMA_DRAFTER, "draft-mtp", False),
    ("unsloth/gemma-4-12b-it-GGUF", _GEMMA_DRAFTER, "draft-mtp", False),
    ("unsloth/gemma-4-26B-A4B-it-GGUF", _GEMMA_DRAFTER, "draft-mtp", False),
    ("unsloth/gemma-4-31B-it-GGUF", _GEMMA_DRAFTER, "draft-mtp", False),
    ("unsloth/gemma-4-E2B-it-qat-GGUF", _GEMMA_DRAFTER, "draft-mtp", False),
    ("unsloth/gemma-4-E4B-it-qat-GGUF", _GEMMA_DRAFTER, "draft-mtp", False),
    ("unsloth/gemma-4-12b-it-qat-GGUF", _GEMMA_DRAFTER, "draft-mtp", False),
    ("unsloth/gemma-4-26B-A4B-it-qat-GGUF", _GEMMA_DRAFTER, "draft-mtp", False),
    ("unsloth/gemma-4-31B-it-qat-GGUF", _GEMMA_DRAFTER, "draft-mtp", False),
]


def _resolve_real(monkeypatch, repo, drafter, mode):
    backend = _resolver_backend(monkeypatch)
    if "qwen" in repo.lower() and "-mtp" in repo.lower():
        backend._nextn_predict_layers = 1
    flags = backend._build_speculative_flags(
        speculative_type = mode,
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = repo,
        model_path = None,
        gpus = True,  # B200 default
        binary = "/fake/llama-server",
        mtp_draft_path = drafter,
    )
    return backend, flags, _flags_dict(flags)


@pytest.mark.parametrize(
    "repo, drafter, auto_spec, auto_ngram_knobs",
    _REAL_REPO_MATRIX,
    ids = [r[0].split("/")[-1] for r in _REAL_REPO_MATRIX],
)
def test_real_repo_auto_routing(monkeypatch, repo, drafter, auto_spec, auto_ngram_knobs):
    # Auto is the default mode the dropdown ships with.
    backend, flags, parsed = _resolve_real(monkeypatch, repo, drafter, "auto")
    if auto_spec is None:
        # Non-MTP: no draft-mtp, hand off to llama-server's own default.
        assert "--spec-type" not in parsed
        assert "--spec-default" in flags
        assert backend.speculative_type == "default"
    elif auto_spec == "draft-mtp":
        assert parsed.get("--spec-type") == "draft-mtp"
        assert parsed.get("--spec-draft-n-max") == "2"
        assert backend.speculative_type == "draft-mtp"
        # gemma ships a separate drafter; Qwen bakes the head into the GGUF.
        assert (
            (parsed.get("--model-draft") == drafter) if drafter else ("--model-draft" not in parsed)
        )
    else:  # ngram-mod (sub-3B MTP drop)
        assert parsed.get("--spec-type") == "ngram-mod"
        assert "--model-draft" not in parsed  # draft head dropped
        assert backend.speculative_type == "ngram-mod"
    if auto_ngram_knobs:
        assert "--spec-ngram-mod-n-match" in parsed
    assert backend.requested_spec_mode == "auto"


@pytest.mark.parametrize(
    "repo, drafter",
    [(r[0], r[1]) for r in _REAL_REPO_MATRIX],
    ids = [r[0].split("/")[-1] for r in _REAL_REPO_MATRIX],
)
def test_real_repo_off_emits_nothing(monkeypatch, repo, drafter):
    # Off must suppress speculative decoding for every family.
    backend, flags, _ = _resolve_real(monkeypatch, repo, drafter, "off")
    assert flags == []
    assert backend.speculative_type is None
    assert backend.requested_spec_mode == "off"


@pytest.mark.parametrize(
    "repo, drafter",
    [(r[0], r[1]) for r in _REAL_REPO_MATRIX],
    ids = [r[0].split("/")[-1] for r in _REAL_REPO_MATRIX],
)
def test_real_repo_forced_mtp_never_aborts(monkeypatch, repo, drafter):
    # Forcing MTP on the dropdown: real MTP models (name marker or separate
    # drafter) engage draft-mtp even below 3B; non-MTP models default back to
    # --spec-default instead of emitting a draft-mtp llama-server will abort on.
    backend, flags, parsed = _resolve_real(monkeypatch, repo, drafter, "mtp")
    is_real_mtp = _is_mtp_model_name(repo) or bool(drafter)
    if is_real_mtp:
        assert parsed.get("--spec-type") == "draft-mtp"
        assert backend.speculative_type == "draft-mtp"
        assert (
            (parsed.get("--model-draft") == drafter) if drafter else ("--model-draft" not in parsed)
        )
    else:
        assert "--spec-type" not in parsed
        assert "--spec-default" in flags
        assert backend.speculative_type == "default"
    assert backend.requested_spec_mode == "mtp"


# ── Sub-3B separate-drafter exemption (Gemma) ─────────────────────────
#
# The sub-3B MTP drop is an embedded-head cost (Qwen). A separate drafter
# (Gemma's root mtp-*.gguf) is a cheap standalone model that wins below 3B
# (B200 Q4_K_XL: gemma-4-E2B draft-mtp n=2 = 1.21x vs OFF), so it is exempt.


def test_sub3b_gemma_separate_drafter_engages_mtp(monkeypatch):
    backend = _resolver_backend(monkeypatch)
    flags = backend._build_speculative_flags(
        speculative_type = "auto",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = "unsloth/gemma-4-E2B-it-GGUF",  # 2B
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
        mtp_draft_path = "/snap/mtp-gemma-4-E2B-it.gguf",  # separate drafter
    )
    parsed = _flags_dict(flags)
    assert parsed.get("--spec-type") == "draft-mtp"
    assert parsed.get("--model-draft") == "/snap/mtp-gemma-4-E2B-it.gguf"
    assert "--spec-ngram-mod-n-match" not in parsed
    assert backend.speculative_type == "draft-mtp"


def test_sub3b_qwen_embedded_head_still_drops_to_ngram(monkeypatch):
    backend = _resolver_backend(monkeypatch)
    flags = backend._build_speculative_flags(
        speculative_type = "auto",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = "unsloth/Qwen3.5-2B-MTP-GGUF",  # 2B, embedded head
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
        mtp_draft_path = None,  # no separate drafter
    )
    parsed = _flags_dict(flags)
    assert parsed.get("--spec-type") == "ngram-mod"
    assert "--model-draft" not in parsed
    assert backend.speculative_type == "ngram-mod"


def test_auto_mode_drops_mtp_exempts_separate_drafter():
    from core.inference.llama_cpp import _auto_mode_drops_mtp

    assert _auto_mode_drops_mtp("auto", 2.0) is True
    assert _auto_mode_drops_mtp("auto", 2.0, has_separate_drafter = True) is False
    assert _auto_mode_drops_mtp("auto", 4.0) is False
    assert _auto_mode_drops_mtp("mtp", 2.0) is False  # forced engages regardless


# ── spec_fallback_reason (drives the "update llama.cpp" UI hint) ───────


def test_spec_fallback_reason_set_when_binary_lacks_mtp(monkeypatch):
    # Outdated llama-server with no mtp token: a forced MTP request can't emit
    # draft-mtp, so record the reason for the UI update affordance.
    backend = _resolver_backend(monkeypatch, mtp_token = None)
    backend._build_speculative_flags(
        speculative_type = "mtp",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = _MTP_MODEL,
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
    )
    assert backend.spec_fallback_reason == "binary_no_mtp"


def test_spec_fallback_reason_none_when_mtp_engages(monkeypatch):
    backend = _resolver_backend(monkeypatch)
    backend._build_speculative_flags(
        speculative_type = "auto",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = _MTP_MODEL,
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
    )
    assert backend.speculative_type == "draft-mtp"
    assert backend.spec_fallback_reason is None


def test_spec_fallback_reason_reset_on_off(monkeypatch):
    # A subsequent off load must clear a stale reason.
    backend = _resolver_backend(monkeypatch, mtp_token = None)
    backend._build_speculative_flags(
        speculative_type = "mtp",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = _MTP_MODEL,
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
    )
    assert backend.spec_fallback_reason == "binary_no_mtp"
    backend._build_speculative_flags(
        speculative_type = "off",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = _MTP_MODEL,
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
    )
    assert backend.spec_fallback_reason is None


def test_is_gemma_mtp_family():
    from core.inference.llama_cpp import _is_gemma_mtp_family

    assert _is_gemma_mtp_family("unsloth/gemma-4-E4B-it-GGUF") is True
    assert _is_gemma_mtp_family("unsloth/gemma-4-12b-it-GGUF") is True
    # gemma-3n ships no separate drafter, so it is not a drafter family.
    assert _is_gemma_mtp_family("unsloth/gemma-3n-E2B-it-GGUF") is False
    assert _is_gemma_mtp_family("unsloth/Qwen3.5-35B-A3B-MTP-GGUF") is False
    assert _is_gemma_mtp_family("unsloth/llama-3-8b") is False


def test_gemma_3n_without_drafter_is_not_mtp(monkeypatch):
    # gemma-3n ships no drafter; it must take the normal non-MTP path, not
    # drafter_not_found (which would make every reload retry a missing drafter).
    backend = _resolver_backend(monkeypatch)
    backend._build_speculative_flags(
        speculative_type = "auto",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = "unsloth/gemma-3n-E4B-it-GGUF",
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
        mtp_draft_path = None,
    )
    assert backend.spec_fallback_reason is None


def test_spec_fallback_reason_drafter_not_found(monkeypatch):
    # Drafterless Gemma should fall back to ngram-mod + drafter_not_found.
    backend = _resolver_backend(monkeypatch)
    flags = backend._build_speculative_flags(
        speculative_type = "auto",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = "unsloth/gemma-4-E4B-it-GGUF",
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
        mtp_draft_path = None,  # Drafter download failed
    )
    parsed = _flags_dict(flags)
    assert parsed.get("--spec-type") == "ngram-mod"
    assert backend.speculative_type == "ngram-mod"
    assert backend.spec_fallback_reason == "drafter_not_found"


def test_is_gemma_mtp_name_none_safe():
    # model_identifier=None (local load) must not raise; recognise via filename.
    from core.inference.llama_cpp import _is_gemma_mtp_family, _is_gemma_mtp_name

    assert _is_gemma_mtp_family(None) is False
    assert _is_gemma_mtp_name(None, "/models/gemma-4-E4B-it-Q4_K_M.gguf") is True
    assert _is_gemma_mtp_name("unsloth/Qwen3.5-4B-MTP-GGUF", None) is False


@pytest.mark.parametrize("mode", ["mtp", "mtp+ngram"])
def test_forced_mtp_gemma_without_drafter_falls_back(monkeypatch, mode):
    # Forced MTP on a drafterless Gemma must fall back, not emit draft-mtp.
    backend = _resolver_backend(monkeypatch)
    flags = backend._build_speculative_flags(
        speculative_type = mode,
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = "unsloth/gemma-4-E4B-it-GGUF",
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
        mtp_draft_path = None,
    )
    parsed = _flags_dict(flags)
    assert parsed.get("--spec-type") == "ngram-mod"
    assert "--model-draft" not in parsed
    assert backend.spec_fallback_reason == "drafter_not_found"


def test_local_gemma_gguf_without_identifier_falls_back(monkeypatch):
    # Local Gemma GGUF (family only in filename) must not crash; falls back.
    backend = _resolver_backend(monkeypatch)
    flags = backend._build_speculative_flags(
        speculative_type = "auto",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = None,
        model_path = "/models/gemma-4-E4B-it-Q4_K_M.gguf",
        gpus = True,
        binary = "/fake/llama-server",
        mtp_draft_path = None,
    )
    parsed = _flags_dict(flags)
    assert parsed.get("--spec-type") == "ngram-mod"
    assert backend.spec_fallback_reason == "drafter_not_found"


def _drafter_not_found_kwargs():
    return dict(
        model_identifier = "unsloth/gemma-4-E4B-it-GGUF",
        hf_variant = "Q4_K_M",
        n_ctx = 8192,
        cache_type_kv = None,
        speculative_type = "auto",
        chat_template_override = None,
        extra_args = None,
        is_vision = False,
        gguf_path = None,  # HF load: drafter resolves inside load_model
    )


def test_already_in_target_state_retries_after_hf_drafter_not_found():
    # Recoverable drafter_not_found must not dedupe; reload re-attempts download.
    backend = _mtp_backend(
        _model_identifier = "unsloth/gemma-4-E4B-it-GGUF",
        _speculative_type = "ngram-mod",
        _spec_fallback_reason = "drafter_not_found",
        _mtp_draft_path = None,
        _gguf_path = None,
    )
    assert backend._already_in_target_state(**_drafter_not_found_kwargs()) is False
    # Sanity: with no fallback reason the same request still dedupes (matches).
    ok = _mtp_backend(_model_identifier = "unsloth/gemma-4-E4B-it-GGUF", _gguf_path = None)
    assert ok._already_in_target_state(**_drafter_not_found_kwargs()) is True
