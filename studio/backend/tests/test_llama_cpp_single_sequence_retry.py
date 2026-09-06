# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The single-sequence retry for architectures that refuse a unified KV cache.

Studio appends ``--kv-unified`` on its own whenever it asks for more than one
slot, to stop llama.cpp splitting ``-c`` into per-slot windows. Some
architectures need one sequence per stream and refuse to build a context that
way, so the model cannot load at all and nothing the user changes in the UI
reaches the flag that caused it. The retry reverses Studio's choice, not the
user's context.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.inference.llama_cpp import LlamaCppBackend

# Verbatim from a Strix Halo report: llama-server refusing GLM-5.3-Flash.
_REFUSAL = (
    "0.56.012.623 E llama_init_from_model: failed to initialize the context: "
    "glm5next: the pooled indexer needs one sequence per stream, so a unified "
    "KV cache is only supported with a single sequence"
)


def test_the_reported_refusal_is_recognised():
    assert LlamaCppBackend._is_kv_unified_refused(_REFUSAL)


def test_an_unrelated_failure_is_not():
    assert not LlamaCppBackend._is_kv_unified_refused(
        "error loading model: unknown model architecture: 'qwen4exp'"
    )
    assert not LlamaCppBackend._is_kv_unified_refused("")


def test_the_reported_launch_becomes_one_slot_with_no_unified_cache():
    """The argv is the one in the report, trimmed to the flags that matter."""
    cmd = [
        "llama-server",
        "-m",
        "GLM-5.3-Flash-UD-IQ4_XS-00001-of-00004.gguf",
        "--parallel",
        "4",
        "--flash-attn",
        "on",
        "--no-context-shift",
        "-c",
        "128000",
        "--gpu-layers",
        "47",
        "--fit",
        "off",
        "--kv-unified",
        "--jinja",
    ]

    out = LlamaCppBackend._with_single_sequence(cmd)

    assert out is not None
    assert "--kv-unified" not in out
    assert out[out.index("--parallel") + 1] == "1"
    # The user's context is theirs; only Studio's own choice is reversed.
    assert out[out.index("-c") + 1] == "128000"
    assert out[out.index("--gpu-layers") + 1] == "47"
    assert "--jinja" in out and "--no-context-shift" in out


def test_a_parallel_surviving_in_the_extras_tail_is_rewritten_too():
    """llama.cpp is last-wins and extras are appended after Unsloth's flags."""
    cmd = ["llama-server", "--parallel", "4", "--kv-unified", "-np", "8"]

    out = LlamaCppBackend._with_single_sequence(cmd)

    assert out == ["llama-server", "--parallel", "1", "-np", "1"]


def test_every_spelling_of_the_two_flags_is_handled():
    cmd = ["llama-server", "--parallel=4", "-kvu", "--alias", "m"]
    assert LlamaCppBackend._with_single_sequence(cmd) == [
        "llama-server",
        "--parallel",
        "1",
        "--alias",
        "m",
    ]

    # -np8 is llama.cpp's attached short form, which _flag_name peels to -np.
    cmd = ["llama-server", "-np8", "--kv-unified", "1"]
    assert LlamaCppBackend._with_single_sequence(cmd) == ["llama-server", "-np", "1"]


def test_a_command_already_running_one_sequence_has_nothing_to_retry():
    assert LlamaCppBackend._with_single_sequence(["llama-server", "-c", "8192"]) is None
    assert LlamaCppBackend._with_single_sequence(["llama-server", "--parallel", "1"]) is None


def test_a_slot_count_is_added_when_the_command_carried_none():
    out = LlamaCppBackend._with_single_sequence(["llama-server", "--kv-unified"])

    assert out == ["llama-server", "--parallel", "1"]


def test_the_inherited_environment_is_dropped_so_it_cannot_undo_the_retry():
    """llama.cpp applies its environment before parsing argv."""
    env = {"LLAMA_ARG_KV_UNIFIED": "1", "LLAMA_ARG_N_PARALLEL": "4", "PATH": "/usr/bin"}

    assert LlamaCppBackend._drop_env_single_sequence(env) is True
    assert "LLAMA_ARG_KV_UNIFIED" not in env
    assert "LLAMA_ARG_N_PARALLEL" not in env
    assert env["PATH"] == "/usr/bin"


def test_a_clean_environment_reports_nothing_dropped():
    env = {"PATH": "/usr/bin"}

    assert LlamaCppBackend._drop_env_single_sequence(env) is False
    assert env == {"PATH": "/usr/bin"}


def test_the_retry_commits_the_one_slot_geometry_it_launched():
    # The launch commits n_parallel and kv_cache_unified after the ladder, so the
    # retry has to overwrite the locals it reverses, or Studio would advertise and
    # admit the multi-slot geometry that llama-server just refused.
    import inspect

    src = inspect.getsource(LlamaCppBackend.load_model)
    start = src.index('label = "-single-seq"')
    block = src[src.rindex("cmd = _kvu_cmd", 0, start) : start]
    assert "n_parallel = 1" in block
    assert "kv_cache_unified = False" in block
