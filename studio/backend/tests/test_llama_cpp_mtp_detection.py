# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the MTP auto-detection path (llama.cpp #22673).

Pins three contracts: name-based detector, user-override detector, and
the _already_in_target_state mirror that prevents needless reloads.
"""

from __future__ import annotations

import ast
import inspect
import os
import struct
import sys
import types as _types
from importlib.util import find_spec as _find_spec
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

# Real httpx wins when installed: setdefault only checks whether httpx is already
# imported, not whether it exists, so the stub used to shadow the real module for the
# rest of the session, and it is a subset (no Response) that blew routes/inference up in
# combined runs. find_spec rather than an import, so installing the stub stays this
# module's only side effect on sys.modules. sys.modules is tested first because find_spec
# raises ValueError on a module already there without a spec (every bare-ModuleType httpx
# stub in this tree), which would be a collection error rather than one failed test.
if "httpx" not in sys.modules and _find_spec("httpx") is None:
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
    _httpx_stub.Response = type("Response", (), {})
    _httpx_stub.Client = type(
        "C",
        (),
        {
            "__init__": lambda s, **kw: None,
            "__enter__": lambda s: s,
            "__exit__": lambda s, *a: None,
        },
    )
    sys.modules["httpx"] = _httpx_stub

import pytest

from core.inference.llama_cpp import (
    GgufLoadIntent,
    LlamaCppBackend,
    _GPU_OFFLOAD_OVERRIDE_FLAGS,
    _THREAD_OVERRIDE_FLAGS,
    _backfill_usage_from_timings,
    _build_ngram_mod_flags,
    _canonicalize_spec_mode,
    _extra_args_set_any_flag,
    _extra_args_set_spec_type,
    _is_mtp_model_name,
    _kv_unified_from_args,
    _mla_mtp_auto_enabled,
    _swa_full_from_args_or_env,
)


def _matches(backend: LlamaCppBackend, **kwargs) -> bool:
    return backend.adopt_load_intent_if_matched(GgufLoadIntent(**kwargs))


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


@pytest.mark.parametrize("flag", ["--swa-full", "--swa_full"])
def test_swa_full_detects_llama_cpp_long_flag_spellings(flag):
    assert _swa_full_from_args_or_env([flag], {}) is True


@pytest.mark.parametrize("value", ["on", "enabled", "true", "1"])
def test_swa_full_detects_llama_cpp_env_truth_values(value):
    assert _swa_full_from_args_or_env([], {"LLAMA_ARG_SWA_FULL": value}) is True


@pytest.mark.parametrize("value", ["", "off", "yes", "TRUE", " true ", "0"])
def test_swa_full_rejects_values_llama_cpp_treats_as_false(value):
    assert _swa_full_from_args_or_env([], {"LLAMA_ARG_SWA_FULL": value}) is False


def test_swa_full_cli_wins_when_env_is_false():
    assert _swa_full_from_args_or_env(["--swa-full"], {"LLAMA_ARG_SWA_FULL": "0"}) is True


@pytest.mark.parametrize("flag", ["--kv-unified", "--kv_unified", "-kvu"])
def test_kv_unified_detects_enable_aliases(flag):
    assert _kv_unified_from_args([flag]) is True


@pytest.mark.parametrize("flag", ["--no-kv-unified", "--no_kv_unified", "-no-kvu"])
def test_kv_unified_detects_disable_aliases(flag):
    assert _kv_unified_from_args(["--kv-unified", flag]) is False


def test_kv_unified_uses_environment_before_cli():
    assert _kv_unified_from_args([], env = {"LLAMA_ARG_KV_UNIFIED": "true"}) is True
    assert _kv_unified_from_args([], default = True, env = {"LLAMA_ARG_KV_UNIFIED": "false"}) is True
    assert _kv_unified_from_args(["--kv-unified"], env = {"LLAMA_ARG_KV_UNIFIED": "false"}) is True


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
        _matches(
            backend,
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
        _matches(
            backend,
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
        _matches(
            backend,
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


def test_forced_dspark_without_a_sidecar_stops_reloading():
    """drafter_not_found covers both "the fetch failed" and "this repo publishes
    none". The second is the permanent state of every repo but one, so retrying
    it relaunched an identical server on every Apply."""
    backend = _mtp_backend(
        _model_identifier = "unsloth/Qwen3-7B-GGUF",
        _speculative_type = "default",
        _requested_spec_mode = "dspark",
        _spec_fallback_reason = "drafter_not_found",
        _spec_drafter_kind = "dspark",
        _dspark_sidecar_absent = True,
    )
    assert (
        _matches(
            backend,
            gguf_path = None,
            model_identifier = "unsloth/Qwen3-7B-GGUF",
            hf_variant = "Q4_K_M",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = "dspark",
            chat_template_override = None,
            extra_args = None,
            is_vision = False,
        )
        is True
    )


def test_forced_dspark_retries_when_the_fetch_failed_rather_than_the_repo():
    """The other half: the repo does publish a sidecar and the download failed, so
    the next Apply should reload and re-run _download_dspark."""
    backend = _mtp_backend(
        _model_identifier = "unsloth/DeepSeek-V4-Flash-0731-GGUF",
        _speculative_type = "default",
        _requested_spec_mode = "dspark",
        _spec_fallback_reason = "drafter_not_found",
        _spec_drafter_kind = "dspark",
        _dspark_sidecar_absent = False,
    )
    assert (
        _matches(
            backend,
            gguf_path = None,
            model_identifier = "unsloth/DeepSeek-V4-Flash-0731-GGUF",
            hf_variant = "Q4_K_M",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = "dspark",
            chat_template_override = None,
            extra_args = None,
            is_vision = False,
        )
        is False
    )


def test_mtp_without_a_drafter_still_reloads_to_retry_the_fetch():
    """Negative control for the above: the MTP retry must survive."""
    backend = _mtp_backend(
        _model_identifier = "unsloth/gemma-4-12b-it-GGUF",
        _speculative_type = "default",
        _requested_spec_mode = "mtp",
        _spec_fallback_reason = "drafter_not_found",
        _spec_drafter_kind = "mtp",
    )
    assert (
        _matches(
            backend,
            gguf_path = None,
            model_identifier = "unsloth/gemma-4-12b-it-GGUF",
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


def test_auto_resolved_dspark_reuses_its_server_instead_of_reloading(tmp_path):
    """Auto keeps arriving as "auto" while the launch stored the DSpark sidecar in
    the drafter field. Comparing the MTP field (None for these repos) against it
    would tear down and relaunch an identical server on every Apply."""
    sidecar = tmp_path / "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    backend = _mtp_backend(
        _model_identifier = "unsloth/DeepSeek-V4-Flash-0731-GGUF",
        _speculative_type = "draft-dspark",
        _mtp_draft_path = str(sidecar),
    )
    assert (
        _matches(
            backend,
            gguf_path = None,
            model_identifier = "unsloth/DeepSeek-V4-Flash-0731-GGUF",
            hf_variant = "Q4_K_M",
            n_ctx = 8192,
            cache_type_kv = None,
            speculative_type = None,
            chat_template_override = None,
            extra_args = None,
            is_vision = False,
            dspark_draft_path = str(sidecar),
            compare_mtp_draft = True,
        )
        is True
    )


def test_already_in_target_state_explicit_off_still_mismatches_mtp_backend():
    backend = _mtp_backend()
    assert (
        _matches(
            backend,
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
    # Prompt caching stays on (in-VRAM prefix reuse); #5692 only needed the host-RAM
    # checkpoints (--cache-ram / --ctx-checkpoints) disabled, not prompt reuse.
    assert '"--no-cache-prompt"' not in src
    assert stale_checkpoint_flag not in src


# Backend-wide guard: Unsloth must never inject --no-cache-prompt into a llama-server
# command. It disables in-VRAM prompt-prefix reuse, re-prefilling every repeated prompt
# (#5692 only needed --cache-ram / --ctx-checkpoints off; #7260 dropped the stray flag).
# Detecting it (_is_real) or honouring a user-supplied one (_prompt_cache_off) is fine.
_NO_CACHE_PROMPT_FLAG = "--no-cache-prompt"
_LIST_MUTATORS = frozenset({"append", "extend", "insert"})


def _has_flag_literal(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Constant) and n.value == _NO_CACHE_PROMPT_FLAG for n in ast.walk(node)
    )


def _no_cache_prompt_injections(source: str, filename: str) -> list[tuple[str, int]]:
    """(file, lineno) for each spot adding --no-cache-prompt to a list."""
    hits: list[tuple[str, int]] = []
    for node in ast.walk(ast.parse(source, filename = filename)):
        # cmd.append/extend/insert(... flag ...) or cmd += [... flag ...]
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _LIST_MUTATORS
            and any(_has_flag_literal(a) for a in node.args)
        ) or (
            isinstance(node, ast.AugAssign)
            and isinstance(node.op, ast.Add)
            and _has_flag_literal(node.value)
        ):
            hits.append((filename, node.lineno))
    return hits


def test_unsloth_never_injects_no_cache_prompt_into_any_command():
    root = Path(_BACKEND_DIR)
    files = [p for p in root.rglob("*.py") if "tests" not in p.relative_to(root).parts]
    violations: list[tuple[str, int]] = []
    for path in files:
        try:
            violations += _no_cache_prompt_injections(path.read_text(encoding = "utf-8"), str(path))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
    assert files, "no backend source files were scanned"
    assert violations == [], (
        "Unsloth must never add --no-cache-prompt to a llama-server command "
        "(it disables prompt-prefix reuse); detecting or honouring a user-supplied "
        f"one is fine. Offending sites: {violations}"
    )


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
        _matches(
            backend,
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
        _matches(
            backend,
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
        _matches(
            backend,
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
        _matches(
            backend,
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
        _matches(
            backend,
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


# One fixed wall-clock second, so two revisions of a file differ only below the
# resolution a whole-second mtime can see.
_FIXED_MTIME_SECOND = 1_700_000_000


def _pin_mtime(path: Path, *, nanos: int) -> Path:
    """Pin a file's mtime `nanos` into one fixed second, so a later rewrite of it
    differs only below the resolution a whole-second mtime can see."""
    stamp = _FIXED_MTIME_SECOND * 1_000_000_000 + nanos
    os.utime(path, ns = (stamp, stamp))
    return path


_NEEDS_BASH = pytest.mark.skipif(
    sys.platform == "win32",
    reason = "fake llama-server is a bash stub; Windows has no direct executor",
)


def _clear_caps_cache():
    with LlamaCppBackend._capability_cache_lock:
        LlamaCppBackend._capability_cache.clear()
        LlamaCppBackend._capability_retry_after.clear()


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
def test_probe_server_capabilities_detects_dspark(tmp_path):
    fake = _make_fake_llama_server(
        tmp_path / "llama-server",
        "--spec-type none,draft-mtp,draft-dspark,ngram-mod",
    )
    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps["supports_dspark"] is True


_DFLASH_SPEC_HELP = "--spec-type none,draft-mtp,draft-dflash,ngram-mod"
# Padded to the same length as the DFlash one, so the two builds are the same size on
# disk and only the sub-second mtime tells them apart.
_PRE_DFLASH_SPEC_HELP = "--spec-type none,draft-mtp,ngram-mod".ljust(len(_DFLASH_SPEC_HELP))


@_NEEDS_BASH
def test_probe_server_capabilities_rereads_a_binary_replaced_in_the_same_second(tmp_path):
    """`unsloth studio update` overwrites llama-server in place. Keyed on whole
    seconds, an update landing in the second the old build was probed in kept the key
    identical and the new build was answered with the old one's capabilities -- and
    "the user just installed the missing capability" is exactly the moment the cache
    has to notice."""
    binary = _make_fake_llama_server(tmp_path / "llama-server", _PRE_DFLASH_SPEC_HELP)
    _pin_mtime(binary, nanos = 100_000)
    _clear_caps_cache()
    assert LlamaCppBackend.probe_server_capabilities(str(binary))["supports_dflash"] is False

    before = binary.stat().st_size
    _make_fake_llama_server(binary, _DFLASH_SPEC_HELP)
    _pin_mtime(binary, nanos = 900_000)
    assert binary.stat().st_size == before
    assert LlamaCppBackend.probe_server_capabilities(str(binary))["supports_dflash"] is True


@_NEEDS_BASH
def test_probe_server_capabilities_gates_known_broken_dspark_prebuilt(tmp_path, monkeypatch):
    fake = _make_fake_llama_server(
        tmp_path / "llama-server",
        "--spec-type none,draft-mtp,draft-dspark,ngram-mod",
    )
    monkeypatch.setattr(
        "utils.llama_cpp_freshness.read_install_marker",
        lambda _binary: {"release_tag": "b10265-mix-89aa77b"},
    )
    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps["supports_dspark"] is False
    assert caps["supports_mtp"] is True


def test_dspark_release_gate_covers_the_whole_broken_window():
    """Every prebuilt between the llama.cpp#26531 reshape regression and its #26577 fix
    aborts on a DSpark load, not only the b10265 one that was first seen."""
    broken = LlamaCppBackend._dspark_release_is_broken
    assert broken("b10259-mix-abc1234") and broken("b10268-mix-abc1234")
    assert not broken("b10258-mix-abc1234")
    assert not broken("b10269-mix-abc1234")
    assert not broken(None) and not broken("")


@_NEEDS_BASH
def test_probe_server_capabilities_uses_binary_library_env(tmp_path, monkeypatch):
    fake = _make_fake_llama_server(
        tmp_path / "llama-server",
        "--spec-type none,mtp,ngram-simple\n",
    )
    captured = {}

    monkeypatch.setattr(
        "core.inference.llama_cpp.child_env_without_native_path_secret",
        lambda: {
            "LD_LIBRARY_PATH": "/already-there",
            "DYLD_LIBRARY_PATH": "/already-inherited",
            "LLAMA_ARG_DEVICE": "MTL0",
            "LLAMA_ARG_OVERRIDE_TENSOR": ".*=MTL0",
        },
    )
    monkeypatch.setattr("core.inference.llama_cpp.sys.platform", "darwin")

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return _types.SimpleNamespace(
            stdout = "--spec-type none,mtp,ngram-simple\n", stderr = "", returncode = 0
        )

    monkeypatch.setattr("core.inference.llama_cpp.subprocess.run", fake_run)

    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))

    assert caps["found"] is True
    assert caps["supports_mtp"] is True
    assert captured["cmd"] == [str(fake), "--help"]
    assert captured["env"] is not None
    assert captured["env"]["GGML_METAL_DEVICES"] == "0"
    assert not any(name.startswith("LLAMA_ARG_") for name in captured["env"])
    # macOS: the probe must go on DYLD_LIBRARY_PATH. dyld ignores
    # LD_LIBRARY_PATH, so putting the dir there left the probe with no search
    # path (#8566), and an inherited LD_LIBRARY_PATH is left as the user set it.
    dyld_dirs = captured["env"]["DYLD_LIBRARY_PATH"].split(os.pathsep)
    assert str(fake.parent) in dyld_dirs
    assert "/already-inherited" in dyld_dirs
    assert captured["env"]["LD_LIBRARY_PATH"] == "/already-there"


@_NEEDS_BASH
def test_probe_server_capabilities_does_not_disable_devices_off_macos(tmp_path, monkeypatch):
    fake = _make_fake_llama_server(
        tmp_path / "llama-server",
        "--spec-type none,mtp,ngram-simple\n",
    )
    captured = {}
    monkeypatch.setattr("core.inference.llama_cpp.child_env_without_native_path_secret", dict)
    monkeypatch.setattr("core.inference.llama_cpp.sys.platform", "linux")

    def fake_run(_cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return _types.SimpleNamespace(
            stdout = "--spec-type none,mtp,ngram-simple\n", stderr = "", returncode = 0
        )

    monkeypatch.setattr("core.inference.llama_cpp.subprocess.run", fake_run)
    _clear_caps_cache()
    LlamaCppBackend.probe_server_capabilities(str(fake))

    assert "GGML_METAL_DEVICES" not in captured["env"]


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
    assert caps["mtp_probe_inconclusive"] is False


@_NEEDS_BASH
def test_probe_server_capabilities_reads_mtp_from_multiline_help(tmp_path):
    # Enum on the indented line: first-line-only probing falsely reported
    # "lacks MTP" (#7302).
    fake = _make_fake_llama_server(
        tmp_path / "llama-server",
        "--spec-type TYPE\n"
        "                                        speculative decoding type\n"
        "                                        (none,draft-simple,draft-mtp,ngram-mod)\n",
    )
    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps["mtp_token"] == "draft-mtp"
    assert caps["supports_mtp"] is True
    assert caps["mtp_probe_inconclusive"] is False


@_NEEDS_BASH
def test_probe_server_capabilities_empty_help_fails_open(tmp_path):
    # --help prints nothing: must not claim the prebuilt lacks MTP (#7302).
    fake = tmp_path / "llama-server"
    fake.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake.chmod(0o755)
    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps["found"] is True
    assert caps["mtp_token"] is None
    assert caps["supports_mtp"] is False
    assert caps["mtp_probe_inconclusive"] is True


@_NEEDS_BASH
def test_probe_server_capabilities_no_spec_type_is_definitive(tmp_path):
    # Nonempty --help without --spec-type: pre-spec binary, not inconclusive.
    fake = _make_fake_llama_server(
        tmp_path / "llama-server",
        "--gpu-layers N\n  GPU layers to offload\n",
    )
    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps["found"] is True
    assert caps["mtp_token"] is None
    assert caps["supports_mtp"] is False
    assert caps["mtp_probe_inconclusive"] is False


@_NEEDS_BASH
def test_probe_server_capabilities_failed_help_with_output_is_inconclusive(tmp_path):
    fake = tmp_path / "llama-server"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "--help" ]; then\n'
        "  echo 'illegal instruction'\n"
        "  exit 1\n"
        "fi\n"
    )
    fake.chmod(0o755)
    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps["found"] is True
    assert caps["supports_mtp"] is False
    assert caps["mtp_probe_inconclusive"] is True


@_NEEDS_BASH
def test_probe_server_capabilities_crash_on_help_fails_open(tmp_path):
    fake = tmp_path / "llama-server"
    fake.write_text("#!/usr/bin/env bash\nkill -SEGV $$\n")
    fake.chmod(0o755)
    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert caps["found"] is True
    assert caps["mtp_token"] is None
    assert caps["supports_mtp"] is False
    assert caps["mtp_probe_inconclusive"] is True


def test_mtp_token_from_spec_help_prefers_draft_mtp():
    assert (
        LlamaCppBackend._mtp_token_from_spec_help("--spec-type none,draft-mtp,mtp,ngram-mod")
        == "draft-mtp"
    )
    assert LlamaCppBackend._mtp_token_from_spec_help("--spec-type [none|mtp|ngram-cache]") == "mtp"
    assert LlamaCppBackend._mtp_token_from_spec_help("--spec-type none,ngram-mod") is None
    # No incidental substring matches.
    assert LlamaCppBackend._mtp_token_from_spec_help("prompt cache") is None


def test_probe_server_capabilities_handles_missing_binary():
    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities("/no/such/llama-server")
    assert caps["found"] is False
    assert caps["supports_mtp"] is False
    assert caps["mtp_probe_inconclusive"] is True
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


def _draft_n_max_matches(
    backend,
    requested,
    *,
    speculative_type = None,
):
    return _matches(
        backend,
        gguf_path = None,
        model_identifier = "unsloth/Qwen3.6-27B-MTP-GGUF",
        hf_variant = "Q4_K_M",
        n_ctx = 8192,
        cache_type_kv = None,
        speculative_type = speculative_type,
        spec_draft_n_max = requested,
        chat_template_override = None,
        extra_args = None,
        is_vision = False,
    )


def test_already_in_target_state_matches_when_draft_n_max_unset():
    # None on the request means "platform default"; matches any backend.
    assert _draft_n_max_matches(_mtp_backend(_spec_draft_n_max = None), None)


def test_already_in_target_state_matches_when_draft_n_max_equals_backend():
    assert _draft_n_max_matches(_mtp_backend(_spec_draft_n_max = 4), 4)


def test_mtp_draft_n_max_mismatch_survives_active_runtime_state():
    assert not _draft_n_max_matches(_mtp_backend(_spec_draft_n_max = 4), 8, speculative_type = "mtp")


@pytest.mark.parametrize(
    ("saved_draft_n_max", "expected_match"),
    [(8, True), (4, False), (None, False)],
)
def test_mtp_draft_n_max_compares_saved_runtime_fallback_intent(saved_draft_n_max, expected_match):
    backend = _mtp_backend(
        _requested_spec_mode = "mtp",
        _speculative_type = None,
        _spec_draft_n_max = None,
        _spec_fallback_reason = "runtime_error",
        _last_load_intent = GgufLoadIntent(
            model_identifier = "unsloth/Qwen3.6-27B-MTP-GGUF",
            spec_draft_n_max = saved_draft_n_max,
        ),
    )
    assert _draft_n_max_matches(backend, 8, speculative_type = "mtp") is expected_match


def test_mtp_draft_n_max_ignored_when_binary_lacks_mtp():
    backend = _mtp_backend(
        _requested_spec_mode = "mtp",
        _speculative_type = "default",
        _spec_draft_n_max = None,
        _spec_fallback_reason = "binary_no_mtp",
    )
    assert _draft_n_max_matches(backend, 8, speculative_type = "mtp")


def test_already_in_target_state_draft_n_max_ignored_when_not_mtp():
    # ngram-mod backend; spec_draft_n_max is MTP-only and must not force
    # a reload against a non-MTP active spec.
    backend = _mtp_backend(
        _speculative_type = "ngram-mod",
        _requested_spec_mode = "ngram",
        _spec_draft_n_max = None,
    )
    assert _draft_n_max_matches(backend, 8, speculative_type = "ngram-mod")


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
        _matches(
            backend,
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
        _matches(
            backend,
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
        _matches(
            backend,
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
        _matches(
            backend,
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
        ("dspark", "dspark"),
        ("ngram", "ngram"),
        ("mtp+ngram", "mtp+ngram"),
        ("off", "off"),
        ("ngram-simple", "ngram-simple"),
        # Legacy wire values map onto the new vocabulary.
        ("default", "auto"),
        ("draft-mtp", "mtp"),
        ("draft-dspark", "dspark"),
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
    mtp_probe_inconclusive = False,
):
    """Backend with a deterministic probe so the resolver is hermetic."""
    fake = {
        "found": True,
        "mtp_token": mtp_token,
        "supports_mtp": bool(mtp_token),
        "supports_dspark": True,
        "mtp_probe_inconclusive": mtp_probe_inconclusive,
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


def test_build_speculative_flags_dspark_requires_sidecar_and_leaves_fit_to_placement(
    monkeypatch, tmp_path
):
    backend = _resolver_backend(monkeypatch)
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    flags = backend._build_speculative_flags(
        speculative_type = "dspark",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = "unsloth/DeepSeek-V4-Flash-0731-GGUF",
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
        dspark_draft_path = str(sidecar),
    )
    parsed = _flags_dict(flags)
    assert "--fit" not in parsed
    assert parsed["--model-draft"] == str(sidecar)
    assert parsed["--spec-type"] == "draft-dspark"
    assert parsed["--spec-draft-n-max"] == "3"


def test_build_speculative_flags_dspark_missing_sidecar_falls_back(monkeypatch):
    backend = _resolver_backend(monkeypatch)
    flags = backend._build_speculative_flags(
        speculative_type = "dspark",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = "unsloth/DeepSeek-V4-Flash-0731-GGUF",
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
    )
    assert flags == ["--spec-default"]
    assert backend.spec_fallback_reason == "drafter_not_found"


def test_build_speculative_flags_dspark_blames_the_binary_not_the_missing_sidecar(monkeypatch):
    """The sidecar fetch is gated on the same supports_dspark answer, so on a
    binary that cannot run DSpark the sidecar is legitimately absent. Reporting
    drafter_not_found would tell the user to place a file that is not the problem,
    and would reload the server on every Apply via the drafter_not_found dedup."""
    backend = _resolver_backend(monkeypatch)
    caps = LlamaCppBackend.probe_server_capabilities()
    caps["supports_dspark"] = False
    flags = backend._build_speculative_flags(
        speculative_type = "dspark",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = "unsloth/DeepSeek-V4-Flash-0731-GGUF",
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
        dspark_draft_path = None,
    )
    assert flags == ["--spec-default"]
    assert backend.spec_fallback_reason == "binary_no_mtp"


# ── Auto defaults to DSpark, and only where a sidecar actually exists ──
#
# The three shapes are the real published repos, checked against the Hub:
#   unsloth/DeepSeek-V4-Flash-0731-GGUF  -> dspark-*.gguf at the root and in dspark/
#   unsloth/gemma-4-12b-it-GGUF          -> MTP/mtp-gemma-4-12b-it-*.gguf, no dspark
#   unsloth/Qwen3.5-4B-MTP-GGUF          -> head baked into the main GGUF, neither file


def test_auto_defaults_to_dspark_when_a_sidecar_is_available(monkeypatch, tmp_path):
    """DSpark beats every other Auto outcome for this architecture, and without it
    these models fall through to --spec-default, i.e. no drafter at all."""
    backend = _resolver_backend(monkeypatch)
    sidecar = tmp_path / "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    flags = backend._build_speculative_flags(
        speculative_type = "auto",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = "unsloth/DeepSeek-V4-Flash-0731-GGUF",
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
        dspark_draft_path = str(sidecar),
    )
    parsed = _flags_dict(flags)
    assert parsed["--spec-type"] == "draft-dspark"
    assert parsed["--model-draft"] == str(sidecar)
    assert parsed["--spec-draft-n-max"] == "3"
    assert backend.spec_fallback_reason is None
    # The request stays "auto": rewriting it would make the reuse check compare
    # "dspark" against a caller that keeps sending "auto" and reload every Apply.
    assert backend.requested_spec_mode == "auto"


def test_auto_keeps_mtp_for_a_separate_drafter_model(monkeypatch, tmp_path):
    """gemma-4-12b-it ships MTP/mtp-*.gguf and no dspark sidecar, so the DSpark
    branch must not intercept it."""
    backend = _resolver_backend(monkeypatch)
    drafter = tmp_path / "mtp-gemma-4-12b-it-Q8_0.gguf"
    drafter.write_bytes(b"draft")
    flags = backend._build_speculative_flags(
        speculative_type = "auto",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = "unsloth/gemma-4-12b-it-GGUF",
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
        mtp_draft_path = str(drafter),
        dspark_draft_path = None,
    )
    parsed = _flags_dict(flags)
    assert parsed["--spec-type"] == "draft-mtp"
    assert parsed["--model-draft"] == str(drafter)
    assert backend.spec_fallback_reason is None


def test_auto_keeps_embedded_mtp(monkeypatch):
    """Qwen3.5-4B-MTP bakes the head into the main GGUF: no separate file of
    either kind, so Auto must still emit MTP."""
    backend = _resolver_backend(monkeypatch)
    backend._nextn_predict_layers = 1
    flags = backend._build_speculative_flags(
        speculative_type = "auto",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = "unsloth/Qwen3.5-4B-MTP-GGUF",
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
        dspark_draft_path = None,
    )
    parsed = _flags_dict(flags)
    assert parsed["--spec-type"] == "draft-mtp"
    assert backend.spec_fallback_reason is None


def test_auto_does_not_promote_dspark_on_a_binary_that_cannot_run_it(monkeypatch, tmp_path):
    """_download_dspark still reports a cached sidecar an incapable binary cannot
    launch. Promoting there would turn Auto's fallback into no speculation at all,
    so the promotion is capability-gated and this model keeps its ngram-mod path."""
    backend = _resolver_backend(monkeypatch)
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(
            lambda cls, binary = None: {
                "found": True,
                "mtp_token": "draft-mtp",
                "supports_mtp": True,
                "supports_dspark": False,
                "mtp_probe_inconclusive": False,
                "ngram_mod_flavor": "new",
                "supports_ngram_mod": True,
                "spec_draft_n_max_flag": "--spec-draft-n-max",
            }
        ),
    )
    sidecar = tmp_path / "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    flags = backend._build_speculative_flags(
        speculative_type = "auto",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = "unsloth/DeepSeek-V4-Flash-0731-GGUF",
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
        dspark_draft_path = str(sidecar),
    )
    assert "draft-dspark" not in _flags_dict(flags).get("--spec-type", "")
    assert backend.speculative_type != "draft-dspark"


def test_build_speculative_flags_dspark_engages_under_auto_fit(monkeypatch, tmp_path):
    """--fit on is not a blocker. The sidecar borrows the target's token_embd and
    output, so llama.cpp's fit step cannot build a standalone draft context to
    measure it and skips the reserve; the load itself still gets the drafter."""
    backend = _resolver_backend(monkeypatch)
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    flags = backend._build_speculative_flags(
        speculative_type = "dspark",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = "unsloth/DeepSeek-V4-Flash-0731-GGUF",
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
        dspark_draft_path = str(sidecar),
        dspark_fit_sized = False,
    )
    assert flags[flags.index("--spec-type") + 1] == "draft-dspark"
    assert flags[flags.index("--model-draft") + 1] == str(sidecar)
    assert backend.speculative_type == "draft-dspark"
    assert backend.spec_fallback_reason is None


@pytest.mark.parametrize("mode", ["auto", "mtp", "dspark", "ngram", "mtp+ngram", "off"])
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
        _matches(
            backend,
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
        _matches(
            backend,
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


def test_spec_fallback_reason_none_when_mtp_probe_inconclusive(monkeypatch):
    backend = _resolver_backend(
        monkeypatch,
        mtp_token = None,
        mtp_probe_inconclusive = True,
    )
    backend._build_speculative_flags(
        speculative_type = "mtp",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = _MTP_MODEL,
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
    )
    assert backend.spec_fallback_reason is None


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
    assert _matches(backend, **_drafter_not_found_kwargs()) is False
    # Sanity: with no fallback reason the same request still dedupes (matches).
    ok = _mtp_backend(_model_identifier = "unsloth/gemma-4-E4B-it-GGUF", _gguf_path = None)
    assert _matches(ok, **_drafter_not_found_kwargs()) is True


# ── A binary that has since gained the drafter ───────────────────────
#
# Standing down on speculative decoding because llama-server cannot run it tells the
# user to run `unsloth studio update`. The update changes nothing about the request, so
# the comparators see the same intent and skip the reload: the one load the update
# exists to fix is the one that never happens again.


def _binary_fallback_kwargs():
    """An Auto request for the model the fallen-back server is already running."""
    return dict(
        model_identifier = "unsloth/Muse-Glimmer-30B-GGUF",
        hf_variant = "Q4_K_M",
        n_ctx = 8192,
        cache_type_kv = None,
        speculative_type = "auto",
        chat_template_override = None,
        extra_args = None,
        is_vision = False,
        gguf_path = None,
    )


def _stood_down_backend(**overrides):
    """Live server that dropped its drafter because the binary could not run it."""
    state = dict(
        _model_identifier = "unsloth/Muse-Glimmer-30B-GGUF",
        _speculative_type = "default",
        _spec_fallback_reason = "binary_no_mtp",
        _gguf_path = None,
    )
    state.update(overrides)
    return _mtp_backend(**state)


def _fake_caps(monkeypatch, **capabilities):
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: dict(capabilities)),
    )


@pytest.mark.parametrize(
    ("kind", "capability"),
    [("dflash", "supports_dflash"), ("dspark", "supports_dspark"), ("mtp", "supports_mtp")],
)
def test_already_in_target_state_reloads_once_the_binary_can_run_the_drafter(
    monkeypatch, kind, capability
):
    _fake_caps(monkeypatch, **{capability: True})
    backend = _stood_down_backend(_spec_drafter_kind = kind)
    assert _matches(backend, **_binary_fallback_kwargs()) is False


@pytest.mark.parametrize("kind", ["dflash", "dspark", "mtp"])
def test_already_in_target_state_keeps_deduping_while_the_binary_still_cannot(monkeypatch, kind):
    """The half that stops this becoming a reload loop: nothing has changed, so the
    healthy drafterless server has to be left alone."""
    _fake_caps(
        monkeypatch,
        supports_dflash = False,
        supports_dspark = False,
        supports_mtp = False,
    )
    backend = _stood_down_backend(_spec_drafter_kind = kind)
    assert _matches(backend, **_binary_fallback_kwargs()) is True


def test_already_in_target_state_asks_about_the_drafter_that_actually_stood_down(monkeypatch):
    """Every kind records the same "binary_no_mtp", so a check keyed on the reason
    alone would read a DSpark stand-down as answered by any build carrying MTP -- and
    tear down a healthy server on every Apply for a capability it never gained."""
    _fake_caps(monkeypatch, supports_mtp = True, supports_dspark = False, supports_dflash = False)
    backend = _stood_down_backend(_spec_drafter_kind = "dspark")
    assert _matches(backend, **_binary_fallback_kwargs()) is True


def test_already_in_target_state_never_reprobes_a_binary_nothing_has_touched(tmp_path, monkeypatch):
    """The steady state, and the reason it has to be cheap: this runs on every Apply,
    while the probe behind it spawns `llama-server --help` on a cold cache -- and the
    cache is cold exactly when the binary was just replaced."""
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"unchanged build")
    monkeypatch.setattr(
        LlamaCppBackend,
        "_find_llama_server_binary",
        staticmethod(lambda **_kwargs: str(binary)),
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: pytest.fail("an untouched binary was reprobed")),
    )
    backend = _stood_down_backend(_spec_drafter_kind = "dflash")
    backend._launch_binary_revision = LlamaCppBackend._binary_revision(str(binary))

    assert _matches(backend, **_binary_fallback_kwargs()) is True


def test_already_in_target_state_sits_out_an_install_still_in_flight(tmp_path, monkeypatch):
    """An update is not atomic, and mid-install the binary is unreadable. Reading that
    as "a different build is installed" would tear the server down for a file that is
    not there yet, and the reload would kill the process before finding that out."""
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"old build")
    monkeypatch.setattr(
        LlamaCppBackend,
        "_find_llama_server_binary",
        staticmethod(lambda **_kwargs: str(binary)),
    )
    backend = _stood_down_backend(
        _spec_fallback_reason = "binary_outdated",
        _spec_drafter_kind = "mtp",
    )
    backend._launch_binary_revision = LlamaCppBackend._binary_revision(str(binary))

    binary.unlink()
    assert LlamaCppBackend._binary_revision(str(binary)) == ()
    assert _matches(backend, **_binary_fallback_kwargs()) is True


def test_already_in_target_state_reloads_when_the_crashed_binary_was_replaced(
    tmp_path, monkeypatch
):
    """A binary_outdated stand-down comes from a launch that died on an architecture
    the build did not know, and no --help flag advertises those, so this one has to
    compare the file itself."""
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"old build")
    _pin_mtime(binary, nanos = 100_000)
    monkeypatch.setattr(
        LlamaCppBackend,
        "_find_llama_server_binary",
        staticmethod(lambda **_kwargs: str(binary)),
    )
    backend = _stood_down_backend(
        _spec_fallback_reason = "binary_outdated",
        _spec_drafter_kind = "mtp",
    )
    backend._launch_binary_revision = LlamaCppBackend._binary_revision(str(binary))
    assert _matches(backend, **_binary_fallback_kwargs()) is True

    # Same path, same size, same second: an update landing right after the crash.
    binary.write_bytes(b"new build")
    _pin_mtime(binary, nanos = 900_000)
    assert _matches(backend, **_binary_fallback_kwargs()) is False


def test_diffusion_load_clears_the_previous_models_spec_fallback():
    """The diffusion early-return skips _build_speculative_flags, which is what clears
    the stand-down on every other load, and only /unload clears it otherwise. Both
    retry rules in the dedupe read it, so an MTP model's verdict left behind by a
    switch to DiffusionGemma relaunches the diffusion server on every Apply -- forever,
    since the relaunch takes this same path and leaves the verdict exactly as it was."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    diffusion = src.find("if self._is_diffusion:")
    assert diffusion != -1
    start = src.find("return self._start_diffusion_server", diffusion)
    assert start != -1
    assert "self._spec_fallback_reason = None" in src[diffusion:start]
    assert "self._spec_drafter_kind = None" in src[diffusion:start]
    # And the DFlash retry flag: discovery runs before the metadata read that
    # classifies this as diffusion, so a transient sidecar failure can set it for a
    # server that will never carry a drafter, and the dedupe reads it too.
    assert "self._dflash_retry_needed = False" in src[diffusion:start]


def test_already_in_target_state_settles_a_dflash_listing_that_never_answered():
    """A permanent listing error (gated repo, offline) records no answer, so
    _dflash_sidecar_absent stays False. The drafter_not_found arm read that as "worth
    another go" and relaunched a healthy drafter-free server on every Apply. DFlash
    asks through _dflash_retry_needed instead, which a permanent error never sets."""
    backend = _mtp_backend(
        _model_identifier = "unsloth/Muse-Glimmer-30B-GGUF",
        _speculative_type = "default",
        _gguf_path = None,
        _spec_fallback_reason = "drafter_not_found",
        _spec_drafter_kind = "dflash",
        _dflash_sidecar_absent = False,
        _dflash_retry_needed = False,
    )
    assert _matches(backend, **_binary_fallback_kwargs()) is True


def test_already_in_target_state_reloads_after_a_dflash_fetch_that_dropped():
    """Under Auto a lost sidecar leaves no fallback reason at all -- the promotion
    never ran -- so the flag is the only thing that can ask for one more attempt."""
    backend = _mtp_backend(
        _model_identifier = "unsloth/Muse-Glimmer-30B-GGUF",
        _speculative_type = "default",
        _gguf_path = None,
    )
    assert _matches(backend, **_binary_fallback_kwargs()) is True

    backend._dflash_retry_needed = True
    assert _matches(backend, **_binary_fallback_kwargs()) is False


_MODERN_DRAFT_NGL_HELP = """usage: llama-server [options]

--spec-draft-ngl N                      layers to offload for the draft model
--parallel N                            number of parallel sequences
"""

_LEGACY_DRAFT_NGL_HELP = """usage: llama-server [options]

-ngld, --gpu-layers-draft N             layers to offload for the draft model
--parallel N                            number of parallel sequences
"""


@_NEEDS_BASH
def test_probe_reports_the_draft_ngl_alias_the_build_actually_has(tmp_path):
    """--spec-draft-ngl only exists from llama.cpp b8955, so a plain "yes" for either
    alias would make the paravirtual drafter pin emit a name an older build does not
    know and the server would refuse to start."""
    modern = _make_fake_llama_server(tmp_path / "modern", _MODERN_DRAFT_NGL_HELP)
    _clear_caps_cache()
    assert LlamaCppBackend.probe_server_capabilities(str(modern))["spec_draft_ngl_flag"] == (
        "--spec-draft-ngl"
    )

    legacy = _make_fake_llama_server(tmp_path / "legacy", _LEGACY_DRAFT_NGL_HELP)
    _clear_caps_cache()
    assert LlamaCppBackend.probe_server_capabilities(str(legacy))["spec_draft_ngl_flag"] == (
        "--gpu-layers-draft"
    )


@_NEEDS_BASH
def test_probe_reports_no_draft_ngl_flag_when_the_build_has_neither(tmp_path):
    """The negative: without it the pin must be skipped, not guessed at."""
    neither = _make_fake_llama_server(tmp_path / "neither", "usage: llama-server\n\n--parallel N\n")
    _clear_caps_cache()
    assert LlamaCppBackend.probe_server_capabilities(str(neither))["spec_draft_ngl_flag"] is None


@_NEEDS_BASH
def test_inconclusive_probe_retries_after_a_bounded_cache_window(tmp_path, monkeypatch):
    """A transient timeout may not be pinned for the whole process, while a
    persistent failure may not make every capability caller wait again (#8317)."""
    import subprocess as _subprocess

    fake = _make_fake_llama_server(
        tmp_path / "llama-server",
        "--spec-type none,draft-mtp,ngram-mod",
    )
    _clear_caps_cache()
    now = [100.0]
    calls = []

    def _run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) <= 2:
            raise _subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 10))
        return _types.SimpleNamespace(
            stdout = "--spec-type none,draft-mtp,ngram-mod\n",
            stderr = "",
            returncode = 0,
        )

    monkeypatch.setattr("core.inference.llama_cpp.subprocess.run", _run)
    monkeypatch.setattr("core.inference.llama_cpp.time.monotonic", lambda: now[0])
    first = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert first["mtp_probe_inconclusive"] is True
    assert first["supports_mtp"] is False

    # Immediate callers reuse the inconclusive answer instead of each paying
    # the subprocess timeout again.
    second = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert second is first
    assert len(calls) == 1

    # Every failed retry receives a fresh bounded window; persistent failures
    # still do not make every caller pay the subprocess timeout.
    now[0] += LlamaCppBackend._CAPABILITY_PROBE_RETRY_SECONDS + 1
    retried = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert retried["mtp_probe_inconclusive"] is True
    assert len(calls) == 2
    assert LlamaCppBackend.probe_server_capabilities(str(fake)) is retried
    assert len(calls) == 2

    # Once a later retry succeeds, the result returns to the normal long-lived
    # cache.
    now[0] += LlamaCppBackend._CAPABILITY_PROBE_RETRY_SECONDS + 1
    recovered = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert recovered["mtp_probe_inconclusive"] is False
    assert recovered["supports_mtp"] is True
    assert len(calls) == 3
    assert LlamaCppBackend.probe_server_capabilities(str(fake)) is recovered
    assert len(calls) == 3


@_NEEDS_BASH
def test_concurrent_timeout_cannot_overwrite_a_successful_probe(tmp_path, monkeypatch):
    """A late, lower-confidence result may not replace a conclusive result."""
    import subprocess as _subprocess
    import threading as _threading

    fake = _make_fake_llama_server(
        tmp_path / "llama-server",
        "--spec-type none,draft-mtp,ngram-mod",
    )
    _clear_caps_cache()
    barrier = _threading.Barrier(2)
    assignment_lock = _threading.Lock()
    success_published = _threading.Event()
    call_ids = []
    results = []

    def _run(cmd, **kwargs):
        with assignment_lock:
            call_id = len(call_ids)
            call_ids.append(call_id)
        barrier.wait(timeout = 2)
        if call_id == 0:
            return _types.SimpleNamespace(
                stdout = "--spec-type none,draft-mtp,ngram-mod\n",
                stderr = "",
                returncode = 0,
            )
        assert success_published.wait(timeout = 2)
        raise _subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 10))

    def _probe():
        result = LlamaCppBackend.probe_server_capabilities(str(fake))
        results.append(result)
        if not result["mtp_probe_inconclusive"]:
            success_published.set()

    monkeypatch.setattr("core.inference.llama_cpp.subprocess.run", _run)
    threads = [_threading.Thread(target = _probe) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout = 3)

    assert not any(thread.is_alive() for thread in threads)
    assert len(call_ids) == 2
    assert len(results) == 2
    assert all(result["supports_mtp"] for result in results)
    cached = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert cached["supports_mtp"] is True
    assert cached["mtp_probe_inconclusive"] is False


@_NEEDS_BASH
def test_a_conclusive_probe_is_never_expired_by_the_retry_window(tmp_path, monkeypatch):
    """Only the inconclusive answer is time-bounded. A conclusive one describes the
    binary, which cannot change while its mtime holds, so applying the retry window to it
    too would re-run --help forever on a perfectly healthy install (#8317)."""
    fake = _make_fake_llama_server(
        tmp_path / "llama-server",
        "--spec-type none,draft-mtp,ngram-mod",
    )
    _clear_caps_cache()
    now = [100.0]
    calls = []

    def _run(cmd, **kwargs):
        calls.append(cmd)
        return _types.SimpleNamespace(
            stdout = "--spec-type none,draft-mtp,ngram-mod\n",
            stderr = "",
            returncode = 0,
        )

    monkeypatch.setattr("core.inference.llama_cpp.subprocess.run", _run)
    monkeypatch.setattr("core.inference.llama_cpp.time.monotonic", lambda: now[0])

    first = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert first["supports_mtp"] is True
    assert len(calls) == 1

    now[0] += LlamaCppBackend._CAPABILITY_PROBE_RETRY_SECONDS * 100
    again = LlamaCppBackend.probe_server_capabilities(str(fake))
    assert again is first
    assert len(calls) == 1, "a conclusive probe must not be re-run once cached"


@_NEEDS_BASH
def test_a_hanging_binary_is_probed_once_per_model_load(tmp_path, monkeypatch):
    """The cost half of #8317, stated as the caller sees it. One model load makes seven
    capability calls; a binary that hangs for a permanent reason must pay the --help
    timeout once across all of them, not once each."""
    import subprocess as _subprocess

    fake = _make_fake_llama_server(tmp_path / "llama-server", "--spec-type none,draft-mtp")
    _clear_caps_cache()
    now = [100.0]
    calls = []

    def _run(cmd, **kwargs):
        calls.append(cmd)
        raise _subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 10))

    monkeypatch.setattr("core.inference.llama_cpp.subprocess.run", _run)
    monkeypatch.setattr("core.inference.llama_cpp.time.monotonic", lambda: now[0])

    # probe_server_capabilities call sites reached by a single load:
    # llama_cpp.py 8003, 9443, 9636, 9706, 10117, 11038, 12786.
    for _ in range(7):
        LlamaCppBackend.probe_server_capabilities(str(fake))
    assert len(calls) == 1, f"expected one --help per load, got {len(calls)}"


@_NEEDS_BASH
def test_a_missing_binary_is_not_cached_so_it_is_seen_as_soon_as_it_lands(tmp_path):
    """The found:False early return sits above the cache and costs a stat rather than a
    subprocess, so it must stay uncached: an install finishing mid-session has to be
    picked up without a Studio restart."""
    binary = tmp_path / "llama-server"
    _clear_caps_cache()

    absent = LlamaCppBackend.probe_server_capabilities(str(binary))
    assert absent["found"] is False
    assert LlamaCppBackend._capability_cache == {}

    _make_fake_llama_server(binary, "--spec-type none,draft-mtp,ngram-mod")
    present = LlamaCppBackend.probe_server_capabilities(str(binary))
    assert present["found"] is True
    assert present["supports_mtp"] is True


def _inconclusive_fallback_backend():
    """A backend that asked for MTP, got an inconclusive probe, and launched without
    speculative decoding. _spec_fallback_reason stays None so the UI banner is suppressed."""
    return _mtp_backend(
        _speculative_type = "default",
        _spec_fallback_reason = None,
        _capability_probe_inconclusive = True,
        _gguf_path = None,
    )


def _same_settings_apply():
    """Apply pressed again with the settings the fallback load already used. The identifier
    has to be the fixture's own, or the reuse check refuses on identity and every assertion
    below passes without ever reaching the speculative branch."""
    return dict(
        model_identifier = "unsloth/Qwen3.6-27B-MTP-GGUF",
        hf_variant = "Q4_K_M",
        n_ctx = 8192,
        cache_type_kv = None,
        speculative_type = "auto",
        chat_template_override = None,
        extra_args = None,
        is_vision = False,
        gguf_path = None,
    )


def _stub_caps(monkeypatch, **caps):
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: caps),
    )


def test_apply_reloads_once_an_inconclusive_probe_starts_answering(monkeypatch):
    # The retry window is worth nothing if Apply dedupes against the fallback: nothing
    # re-probes, so MTP stays off for the life of the process, which is the symptom the
    # window exists to end (#8317).
    _stub_caps(
        monkeypatch,
        found = True,
        mtp_token = "draft-mtp",
        supports_mtp = True,
        mtp_probe_inconclusive = False,
    )
    assert _matches(_inconclusive_fallback_backend(), **_same_settings_apply()) is False


def test_apply_still_dedupes_while_the_probe_keeps_hanging(monkeypatch):
    # A binary that hangs for a permanent reason must not relaunch an identical server on
    # every Apply. Only a probe that has actually turned conclusive earns the reload.
    _stub_caps(
        monkeypatch,
        found = True,
        mtp_token = None,
        supports_mtp = False,
        mtp_probe_inconclusive = True,
    )
    assert _matches(_inconclusive_fallback_backend(), **_same_settings_apply()) is True


def test_apply_reloads_once_even_when_the_build_turns_out_to_have_no_mtp(monkeypatch):
    # Conclusive-and-negative still earns exactly one reload: the degradation has to be
    # re-derived from a real answer rather than from a probe that never returned. That
    # reload records the conclusive probe, clearing the flag, so it does not loop.
    _stub_caps(
        monkeypatch,
        found = True,
        mtp_token = None,
        supports_mtp = False,
        mtp_probe_inconclusive = False,
    )
    assert _matches(_inconclusive_fallback_backend(), **_same_settings_apply()) is False
    # Cleared flag (what the reload leaves behind) dedupes from then on.
    settled = _mtp_backend(_speculative_type = "default", _gguf_path = None)
    assert settled._capability_probe_inconclusive is False
    assert _matches(settled, **_same_settings_apply()) is True


def test_a_slot_clamp_from_an_inconclusive_probe_is_also_retried(monkeypatch):
    # An inconclusive probe reports --kv-unified absent too, so n_parallel is clamped to
    # 1 while _requested_n_parallel keeps the ASK. The two then compare equal and Apply
    # would never restore the slots once the probe recovers. No speculative decoding is
    # involved, so the spec-only version of this guard missed it entirely.
    _stub_caps(
        monkeypatch,
        found = True,
        mtp_token = "draft-mtp",
        supports_mtp = True,
        supports_kv_unified = True,
        mtp_probe_inconclusive = False,
    )
    clamped = _mtp_backend(
        _speculative_type = "default",
        _capability_probe_inconclusive = True,
        _requested_n_parallel = 4,
        _gguf_path = None,
    )
    assert _matches(clamped, n_parallel = 4, **_same_settings_apply()) is False


def test_the_probe_marker_is_committed_only_once_the_runtime_is_replaced():
    """The marker describes the RUNNING runtime, so load_model must not write it before
    the launch is committed.

    Two early exits sit between the probe and the commit and both leave the old server
    up: the Vulkan-ordinal preflight rejects an invalid GPU selection, and a diffusion
    load returns through _start_diffusion_server without using any llama-server
    capability. Writing the marker early would clear it for a runtime that is still
    degraded, or set it on a diffusion runner that would then be torn down and reloaded
    for no reason. Checked structurally because load_model is not unit-callable.
    """
    import ast
    import inspect

    from core.inference import llama_cpp as mod

    source = inspect.getsource(mod)
    tree = ast.parse(source)
    load_model = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "load_model"
    )

    writes = [
        node.lineno
        for node in ast.walk(load_model)
        for target in getattr(node, "targets", [])
        if isinstance(node, ast.Assign)
        and isinstance(target, ast.Attribute)
        and target.attr == "_capability_probe_inconclusive"
    ]
    assert len(writes) == 1, f"expected one commit-point write, found {len(writes)}"

    diffusion_returns = [
        node.lineno
        for node in ast.walk(load_model)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "_start_diffusion_server"
    ]
    assert diffusion_returns, "the diffusion early return moved; re-pin this test"
    assert writes[0] > max(diffusion_returns), (
        "the marker is written before the diffusion early return, so a diffusion runner "
        "would carry it and be reloaded once the probe recovers"
    )


def test_a_diffusion_runtime_is_never_reloaded_by_the_capability_recovery(monkeypatch):
    # A diffusion runner consumes no llama-server capability, so it cannot be degraded by
    # one. A marker left over from an earlier llama-server load must not make every
    # otherwise identical diffusion Apply tear it down and start it again.
    _stub_caps(
        monkeypatch,
        found = True,
        mtp_token = "draft-mtp",
        supports_mtp = True,
        mtp_probe_inconclusive = False,
    )
    diffusion = _mtp_backend(
        _speculative_type = "default",
        _capability_probe_inconclusive = True,
        _is_diffusion = True,
        _gguf_path = None,
    )
    assert _matches(diffusion, **_same_settings_apply()) is True
    # The same stale marker on a llama-server runtime still earns its reload.
    assert _matches(_inconclusive_fallback_backend(), **_same_settings_apply()) is False


def test_unload_clears_the_capability_marker():
    # Otherwise it outlives the runtime it describes and follows the next load in.
    backend = _mtp_backend(_capability_probe_inconclusive = True)
    backend._process = None
    backend.unload_model()
    assert backend._capability_probe_inconclusive is False


def test_a_diffusion_load_never_pays_for_the_capability_probe():
    # The probe is read at the snapshot commit, which a diffusion load returns long
    # before. Reading it beside the capability gates instead made an independent
    # diffusion launch wait out the full --help timeout this change exists to bound.
    import ast
    import inspect

    from core.inference import llama_cpp as mod

    load_model = next(
        node
        for node in ast.walk(ast.parse(inspect.getsource(mod)))
        if isinstance(node, ast.FunctionDef) and node.name == "load_model"
    )
    # In-load probes go through the accumulating _launch_caps helper, so count both it
    # and any direct call.
    probe_calls = [
        node.lineno
        for node in ast.walk(load_model)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == "probe_server_capabilities")
            or (isinstance(node.func, ast.Name) and node.func.id == "_launch_caps")
        )
    ]
    diffusion_return = max(
        node.lineno
        for node in ast.walk(load_model)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "_start_diffusion_server"
    )
    # The probes that legitimately sit above the diffusion return are the pre-existing
    # ones, and every one of them is guarded by the feature that needs it (the
    # --kv-unified clamp behind n_parallel > 1, the DSpark lookup behind its own request),
    # so a diffusion load short-circuits past them. An unconditional probe added up there
    # would make an independent diffusion launch wait out the full --help timeout.
    guarded = {
        node.lineno
        for branch in ast.walk(load_model)
        if isinstance(branch, ast.If)
        for node in ast.walk(branch)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Attribute) and node.func.attr == "probe_server_capabilities")
            or (isinstance(node.func, ast.Name) and node.func.id == "_launch_caps")
        )
    }
    helper_body = {
        node.lineno
        for fn in ast.walk(load_model)
        if isinstance(fn, ast.FunctionDef) and fn.name == "_launch_caps"
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
    }
    unconditional = [
        ln
        for ln in probe_calls
        if ln < diffusion_return and ln not in guarded and ln not in helper_body
    ]
    assert unconditional == [], (
        f"unguarded probe calls above the diffusion return: {unconditional}; a diffusion "
        "load must not pay for a capability it never consumes"
    )


def test_the_marker_comes_from_the_launch_snapshot_not_a_probe_after_startup():
    """The marker must describe the probe that BUILT the command, not one taken after the
    server is up.

    _wait_for_health allows up to 600s, and the retry window is 30s, so a large model's
    startup expires the inconclusive entry many times over. Re-probing at the commit point
    would then record False for a server that was launched without speculative decoding or
    unified KV slots, and every identical Apply would dedupe against that degraded runtime
    for good -- the original bug, reintroduced. Checked structurally because load_model is
    not unit-callable.
    """
    import ast
    import inspect

    from core.inference import llama_cpp as mod

    load_model = next(
        node
        for node in ast.walk(ast.parse(inspect.getsource(mod)))
        if isinstance(node, ast.FunctionDef) and node.name == "load_model"
    )

    commit = next(
        node
        for node in ast.walk(load_model)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "_capability_probe_inconclusive"
            for t in node.targets
        )
    )
    # The committed value must be a plain name pinned earlier, never a live probe call.
    assert isinstance(commit.value, ast.Name), (
        "the marker is computed at the commit point; it must be pinned from the snapshot "
        "that built the launch command instead"
    )

    health_waits = [
        node.lineno
        for node in ast.walk(load_model)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_wait_for_health"
    ]
    pins = [
        node.lineno
        for node in ast.walk(load_model)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == commit.value.id for t in node.targets)
    ]
    assert pins, "the pinned local vanished; re-pin this test"
    assert health_waits, "the startup health wait moved; re-pin this test"
    assert min(pins) < min(health_waits), (
        "the capability snapshot is pinned after the startup wait, so a slow load can "
        "still record a probe that is not the one the server was launched with"
    )


def test_a_later_successful_probe_cannot_erase_an_earlier_degrading_one():
    """The launch's capability decisions are spread across the whole load, and the retry
    window is 30s. The slot clamp runs before an HF download; the command is built after
    it. Sampling one probe lets a later success erase the fact that an earlier one already
    clamped the slots, so the marker has to accumulate.

    Exercised on the accumulator itself: driving load_model would need a real download.
    """
    import ast
    import inspect

    from core.inference import llama_cpp as mod

    load_model = next(
        node
        for node in ast.walk(ast.parse(inspect.getsource(mod)))
        if isinstance(node, ast.FunctionDef) and node.name == "load_model"
    )
    helper = next(
        (
            node
            for node in ast.walk(load_model)
            if isinstance(node, ast.FunctionDef) and node.name == "_launch_caps"
        ),
        None,
    )
    assert helper is not None, "the accumulating probe helper vanished; re-pin this test"

    # It must only ever latch True, never assign the raw probe result.
    assigns = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "_launch_probe_inconclusive" for t in node.targets
        )
    ]
    assert assigns, "the helper no longer records the probe state"
    for node in assigns:
        assert isinstance(node.value, ast.Constant) and node.value.value is True, (
            "the helper assigns the probe result directly; a later conclusive probe would "
            "then erase an earlier degrading one"
        )

    # And nothing outside it may overwrite the accumulator mid-load.
    outside = [
        node.lineno
        for node in ast.walk(load_model)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "_launch_probe_inconclusive" for t in node.targets
        )
        and not (helper.lineno <= node.lineno <= (helper.end_lineno or helper.lineno))
    ]
    assert (
        len(outside) == 1
    ), f"expected only the initialisation outside the helper, found {outside}"


def test_the_accumulator_latches_across_probes():
    """Behavioural check of the same property, on a stand-in with the helper's shape."""
    state = {"inconclusive": False}

    def launch_caps(caps):
        if caps.get("mtp_probe_inconclusive"):
            state["inconclusive"] = True
        return caps

    launch_caps({"mtp_probe_inconclusive": True})  # slot clamp, probe timed out
    launch_caps({"mtp_probe_inconclusive": False})  # command build, probe recovered
    assert state["inconclusive"] is True, "a later success must not erase the earlier guess"


def test_the_dspark_pre_download_gate_latches_into_the_launch_accumulator():
    """The sidecar gate shapes the launch as much as the slot clamp does: an inconclusive
    probe there skips an ~11 GB drafter, so that load ran degraded. It sits before a
    download that can outlast the 30s retry window, so if it probes on its own the later
    launch probe can come back conclusive and the load is remembered as a good one --
    every identical Apply after it then dedupes against a server with no drafter.
    """
    import ast
    import inspect

    from core.inference import llama_cpp as mod

    tree = ast.parse(inspect.getsource(mod))
    download = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_download_dspark"
    )
    direct = [
        node.lineno
        for node in ast.walk(download)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "probe_server_capabilities"
    ]
    assert direct == [], (
        f"_download_dspark probes directly at {direct}; route it through the caller's "
        "accumulator so a guess there is not forgotten"
    )

    load_model = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "load_model"
    )
    call = next(
        node
        for node in ast.walk(load_model)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_download_dspark"
    )
    passed = {
        kw.value.id
        for kw in call.keywords
        if kw.arg == "caps_probe" and isinstance(kw.value, ast.Name)
    }
    assert passed == {
        "_launch_caps"
    }, "load_model must hand _download_dspark the accumulating probe helper"


def test_the_dspark_gate_uses_the_probe_it_is_given():
    """Behavioural half: the injected probe is the one consulted, and its verdict still
    drives the skip. A default is kept so the direct callers in the tests and the CLI
    keep working unchanged.
    """
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend

    signature = inspect.signature(LlamaCppBackend._download_dspark)
    default = signature.parameters["caps_probe"].default
    assert default is None, "caps_probe must stay optional for the standalone callers"

    seen = []

    def probe(binary):
        seen.append(binary)
        return {"supports_dspark": False, "mtp_probe_inconclusive": True}

    server = LlamaCppBackend.__new__(LlamaCppBackend)
    result = LlamaCppBackend._download_dspark(
        server,
        hf_repo = "unsloth/does-not-matter",
        near_path = None,
        binary = "/nonexistent/llama-server",
        caps_probe = probe,
    )
    assert seen == ["/nonexistent/llama-server"], "the injected probe was not consulted"
    # No sidecar on disk and an incapable binary: the fetch is skipped, which is exactly
    # the degraded launch the accumulator has to remember.
    assert result is None
