# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ggml graph-scheduler abort (GGML_ASSERT(*cur_backend_id != -1)): matcher, message, memo.

Load-path behaviour (fail-fast replay, retry on changed settings) lives in
test_cpu_only_defaults.py beside the launch harness.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402


# The real crash, faithfully reproduced from the user's llama-server log (HF
# screenshots discussion #23). The GGML_ASSERT line is followed by ~130 [New LWP]
# lines and then the gdb backtrace; the assert line scrolls out of a short tail.
_FULL_ABORT = "\n".join(
    [
        "0.09.350.752 W llama_context: n_ctx_seq (4096) < n_ctx_train (1048576)",
        "/tmp/llama.cpp/ggml/src/ggml-backend.cpp:1242: GGML_ASSERT(*cur_backend_id != -1) failed",
    ]
    + [f"[New LWP {2147067 - i}]" for i in range(130)]
    + [
        "[Thread debugging using libthread_db enabled]",
        "#1  0x... in ggml_print_backtrace () from /tmp/llama.cpp/build/bin/libggml-base.so.0",
        "#2  0x... in ggml_abort () from /tmp/llama.cpp/build/bin/libggml-base.so.0",
        "#3  0x... in ggml_backend_sched_split_graph () from /tmp/llama.cpp/build/bin/libggml-base.so.0",
        "#4  0x... in llama_context::graph_reserve(...) () from /tmp/llama.cpp/build/bin/libllama.so.0",
        "#5  0x... in llama_context::sched_reserve() () from /tmp/llama.cpp/build/bin/libllama.so.0",
    ]
)

# What a 50-line tail actually contains: the GGML_ASSERT line is gone, but the
# backtrace markers (ggml_abort, ggml_backend_sched_split_graph) remain. The matcher
# must still fire on this -- that's the realistic input at the recording site.
_ABORT_TAIL = "\n".join(_FULL_ABORT.splitlines()[-50:])

# The other ggml abort Studio already handles (#6415 split-axis): must NOT be
# misclassified as a scheduler-reserve abort.
_SPLIT_AXIS_ABORT = (
    "ggml/src/ggml-backend-meta.cpp:541: "
    "GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0) failed\n"
    "#3 ggml_backend_sched_split_graph ()"
)

_OOM_OUTPUT = "llama_model_load: error loading model: unable to allocate buffer\nkilled"
_CLEAN_OUTPUT = "main: server is listening on http://127.0.0.1:8080 - starting the main loop"


# ---- matcher ---------------------------------------------------------------


def test_matcher_fires_on_full_abort_and_short_tail():
    assert LlamaCppBackend._is_sched_reserve_abort(_FULL_ABORT)
    # The headline guarantee: the matcher survives the [New LWP] scroll.
    assert "GGML_ASSERT(*cur_backend_id != -1)".lower() not in _ABORT_TAIL.lower()
    assert LlamaCppBackend._is_sched_reserve_abort(_ABORT_TAIL)


def test_matcher_ignores_unrelated_crashes():
    assert not LlamaCppBackend._is_sched_reserve_abort(_SPLIT_AXIS_ABORT)
    assert not LlamaCppBackend._is_sched_reserve_abort(_OOM_OUTPUT)
    assert not LlamaCppBackend._is_sched_reserve_abort(_CLEAN_OUTPUT)
    assert not LlamaCppBackend._is_sched_reserve_abort("")


def test_matcher_requires_both_a_ggml_marker_and_a_scheduler_marker():
    # scheduler word without any ggml abort/assert -> not our abort.
    assert not LlamaCppBackend._is_sched_reserve_abort("graph_reserve completed in 3ms")
    # ggml abort without a scheduler marker -> some other assert, not ours.
    assert not LlamaCppBackend._is_sched_reserve_abort("ggml_abort: tensor type mismatch")


# ---- classifier ------------------------------------------------------------


def test_classifier_surfaces_actionable_message():
    msg = LlamaCppBackend._classify_llama_start_failure(
        _ABORT_TAIL,
        gguf_path = "/x/GLM-5.2-UD-Q6_K-00001-of-00014.gguf",
        model_identifier = "unsloth/GLM-5.2-GGUF",
        returncode = -6,
    )
    assert msg == LlamaCppBackend._sched_reserve_abort_message()
    # The message names the real cause, not the generic invalid-GGUF/OOM fallback.
    assert "ggml_backend_sched_split_graph" in msg
    assert "enough memory" not in msg  # i.e. not the generic fallback


def test_classifier_generic_fallback_unchanged_for_unknown_crash():
    msg = LlamaCppBackend._classify_llama_start_failure(
        "some unrelated failure",
        gguf_path = None,
        model_identifier = None,
        returncode = -6,
    )
    assert "failed to start" in msg and "enough memory" in msg


# ---- memo round-trip / invalidation ---------------------------------------


def test_memo_round_trip_and_isolation(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_text("x")
    b, model = str(binary), "unsloth/GLM-5.2-GGUF"

    LlamaCppBackend._sched_reserve_abort_keys.clear()
    assert not LlamaCppBackend._sched_reserve_aborts(b, model)
    LlamaCppBackend._record_sched_reserve_abort(b, model)
    assert LlamaCppBackend._sched_reserve_aborts(b, model)
    # A different model on the same binary is unaffected.
    assert not LlamaCppBackend._sched_reserve_aborts(b, "unsloth/Qwen3.5-4B-MTP-GGUF")
    LlamaCppBackend._sched_reserve_abort_keys.clear()


def test_memo_invalidated_by_binary_mtime_change(tmp_path):
    """A `unsloth studio update` swaps the binary -> the memo must not persist."""
    binary = tmp_path / "llama-server"
    binary.write_text("v1")
    b, model = str(binary), "unsloth/GLM-5.2-GGUF"

    LlamaCppBackend._sched_reserve_abort_keys.clear()
    LlamaCppBackend._record_sched_reserve_abort(b, model)
    assert LlamaCppBackend._sched_reserve_aborts(b, model)
    # Bump mtime to a distinct ns value (simulate a rebuilt binary).
    st = binary.stat()
    os.utime(binary, ns = (st.st_atime_ns + 10**9, st.st_mtime_ns + 10**9))
    assert not LlamaCppBackend._sched_reserve_aborts(b, model)
    LlamaCppBackend._sched_reserve_abort_keys.clear()


def test_memo_safe_with_missing_binary_or_model():
    # None binary/model -> no key -> never aborts, never raises.
    assert not LlamaCppBackend._sched_reserve_aborts(None, "m")
    assert not LlamaCppBackend._sched_reserve_aborts("/x", None)
    LlamaCppBackend._record_sched_reserve_abort(None, None)  # no-op, no raise
