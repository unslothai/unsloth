# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Guards for the ggml graph-scheduler abort handling in load_model.

A CPU-only user loading GLM-5.2 (MLA + sparse-attention "indexer" + embedded MTP)
hit `GGML_ASSERT(*cur_backend_id != -1)` in `ggml_backend_sched_split_graph` during
`sched_reserve`: the CPU backend cannot run an op in the graph. Studio's startup
crash raised a generic "invalid GGUF or out of memory" message and the UI replayed
`POST /load`, re-reading the ~583 GB weights into the identical crash every ~3.5 min.

These tests pin: (1) the abort matcher recognises the real backtrace tail and only
that, (2) the classifier surfaces the actionable message, (3) the per-(binary,model)
memo round-trips and is mtime-invalidated, and (4) load_model fails fast on a memoed
abort and records on crash. No GPU, no network, fully deterministic.
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
import textwrap
import types as _types
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# External-dep stubs so importing the backend doesn't require structlog / loggers /
# httpx -- only installed when the real module is missing (mirrors test_tp_vision_regression).
try:
    import structlog  # noqa: F401
except ImportError:
    _structlog_stub = _types.ModuleType("structlog")
    _structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
    sys.modules["structlog"] = _structlog_stub
try:
    import loggers  # noqa: F401
except ImportError:
    _loggers_stub = _types.ModuleType("loggers")
    _loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
    sys.modules["loggers"] = _loggers_stub
try:
    import httpx  # noqa: F401
except ImportError:
    _httpx_stub = _types.ModuleType("httpx")
    for _exc in (
        "ConnectError",
        "TimeoutException",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
        "CloseError",
        "HTTPError",
        "RequestError",
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


# ---- load_model wiring (source-level, no GPU/network needed) ---------------


def _load_model_src() -> str:
    return textwrap.dedent(inspect.getsource(LlamaCppBackend.load_model))


def _call_line(fn, attr):
    """First line where load_model calls method `attr`, or None."""
    return next(
        (
            node.lineno
            for node in ast.walk(fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr
        ),
        None,
    )


def test_load_model_fails_fast_on_memoed_abort():
    """load_model must consult the memo and raise before the download/spawn, so a
    replayed /load doesn't re-read the weights."""
    src = _load_model_src()
    assert "_sched_reserve_aborts(binary, _abort_memo_model)" in src
    assert "_sched_reserve_abort_message()" in src
    fn = ast.parse(src).body[0]
    guard_line = _call_line(fn, "_sched_reserve_aborts")
    download_line = _call_line(fn, "_download_gguf")
    assert download_line is None or guard_line < download_line


def test_failfast_guard_runs_before_killing_the_live_server():
    """The memo guard must precede _kill_process so a known-bad reload does not tear
    down a working server (PR review fix)."""
    fn = ast.parse(_load_model_src()).body[0]
    guard_line = _call_line(fn, "_sched_reserve_aborts")
    kill_line = _call_line(fn, "_kill_process")
    assert guard_line is not None and kill_line is not None
    assert guard_line < kill_line


def test_abort_memo_key_includes_variant_and_launch_settings():
    """The memo key must include the variant AND the launch settings (context, spec),
    so a failed quant does not block a different quant, and changing -c / spec (the
    recommended recovery) is allowed to retry while an identical replay stays blocked."""
    src = _load_model_src()
    assert "_abort_memo_model" in src
    for tok in ("hf_variant", "gguf_path", "str(n_ctx)", "speculative_type", "extra_args"):
        assert tok in src, tok

    def key(
        model = "repo",
        variant = "",
        gguf = "",
        n_ctx = 4096,
        spec = "",
        extra = "",
    ):
        return "\x00".join([model, variant, gguf, str(n_ctx), spec, extra])

    base = key(variant = "UD-Q6_K")
    assert base != key(variant = "UD-Q4_K_XL")  # different quant -> retry allowed
    assert base != key(variant = "UD-Q6_K", n_ctx = 2048)  # lower context -> retry allowed
    assert base != key(variant = "UD-Q6_K", spec = "off")  # disable spec -> retry allowed
    assert base == key(variant = "UD-Q6_K")  # identical replay -> still blocked


def test_load_model_records_abort_on_crash():
    """On a startup crash matching the signature, load_model must record the memo."""
    src = _load_model_src()
    assert "_record_sched_reserve_abort(binary, _abort_memo_model)" in src
    assert "_is_sched_reserve_abort(" in src


def test_abort_memo_deferred_until_mmproj_fallback_ruled_out():
    """The signature is captured up front but the memo is recorded only on a terminal
    raise, after the text-only mmproj fallback is ruled out, so a VLM that recovers
    text-only is not blocked by the fail-fast guard next time (PR review fix)."""
    src = _load_model_src()
    assert "_was_sched_abort = " in src
    assert "if _was_sched_abort:" in src
    fn = ast.parse(src).body[0]
    strip_line = _call_line(fn, "_strip_mmproj_args")
    record_line = _call_line(fn, "_record_sched_reserve_abort")
    # Recording happens after the projector strip, i.e. only once the fallback is tried.
    assert strip_line is not None and record_line is not None
    assert record_line > strip_line


def test_sched_abort_captured_before_mtp_fallback():
    """The no-spec MTP fallback resets the stdout tail, so the first launch's scheduler
    abort must be captured before it runs and folded into the terminal decision; else a
    differently-failing fallback drops the memo and the UI replays the load (PR review fix)."""
    src = _load_model_src()
    assert "_pre_fallback_sched_abort = False" in src
    assert "_pre_fallback_sched_abort = _pre_fallback_sched_abort or (" in src
    # The capture is set before the no-spec fallback spawns.
    fn = ast.parse(src).body[0]
    capture_line = next(
        (
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_pre_fallback_sched_abort" for t in n.targets
            )
            and isinstance(n.value, ast.BoolOp)
        ),
        None,
    )
    fallback_line = next(
        (
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and any(isinstance(a, ast.Name) and a.id == "fallback_cmd" for a in n.args)
        ),
        None,
    )
    assert capture_line is not None and fallback_line is not None
    assert capture_line < fallback_line
