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


# What the binary actually prints: the GGML_ASSERT of ggml-org/llama.cpp#27754,
# stringified expression and all, then a backtrace and abort(). The full buffer
# is scanned, not its tail, so the backtrace cannot push the marker out of view.
_ASSERT = (
    "/build/llama.cpp/src/models/glm5next.cpp:1018: GGML_ASSERT(n_ps == 1 && "
    '"the per-cell pool view needs one sequence per stream") failed\n'
    "[New LWP 4242]\n#0  0x00007f0000000000 in abort ()"
)


def test_the_reported_refusal_is_recognised():
    assert LlamaCppBackend._is_kv_unified_refused(_REFUSAL)


def test_the_assert_the_binary_actually_prints_is_recognised():
    assert LlamaCppBackend._is_kv_unified_refused(_ASSERT)


def test_an_unrelated_failure_is_not():
    assert not LlamaCppBackend._is_kv_unified_refused(
        "error loading model: unknown model architecture: 'qwen4exp'"
    )
    assert not LlamaCppBackend._is_kv_unified_refused("")
    # Upstream's minimax-m3 warning degrades to dense attention and still loads;
    # retrying it as one slot would cost three slots for nothing.
    assert not LlamaCppBackend._is_kv_unified_refused(
        "minimax_m3: unified KV cache with n_seq_max > 1; MSA needs per-sequence "
        "streams -> running DENSE attention. Drop --kv-unified to enable MSA."
    )


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


def test_every_alias_of_the_slot_count_is_rewritten():
    """--n-parallel is in the denylist group too, and llama.cpp is last-wins."""
    out = LlamaCppBackend._with_single_sequence(
        ["llama-server", "--parallel", "4", "--kv-unified", "--n-parallel", "8"]
    )

    assert out == ["llama-server", "--parallel", "1", "--n-parallel", "1"]


def test_a_valueless_slot_flag_does_not_swallow_the_next_flag():
    """Malformed, but eating the token behind it would delete --kv-unified."""
    out = LlamaCppBackend._with_single_sequence(["llama-server", "--parallel", "--kv-unified"])

    assert out == ["llama-server", "--parallel", "1"]


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


def test_the_fit_recovery_rungs_stand_down_for_this_refusal():
    """The abort is a startup crash, so _spawn_and_wait's --fit rungs would take
    it first: a second full model load on a theory unrelated to the failure, and
    the argv this retry then reads would be the fit-rewritten one, not the one
    that refused. They are excluded the same way the tensor-capability crash is."""
    import inspect

    src = inspect.getsource(LlamaCppBackend.load_model)
    assert "_capability_crash = _tensor_capability_crash or self._is_kv_unified_refused" in src
    # Every --fit rung, and only those: the HIP rung keeps its own narrower gate.
    assert src.count("and not _capability_crash") == 3
    assert src.count("and not _tensor_capability_crash") == 1


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
