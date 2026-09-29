# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ggml graph-scheduler abort (GGML_ASSERT(*cur_backend_id != -1)): matcher, message, memo."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402


# Real crash log: the GGML_ASSERT line scrolls out of a short tail behind the [New LWP] dump.
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

_ABORT_TAIL = "\n".join(_FULL_ABORT.splitlines()[-50:])

# #6415 split-axis abort: must NOT be classified as a scheduler-reserve abort.
_SPLIT_AXIS_ABORT = (
    "ggml/src/ggml-backend-meta.cpp:541: "
    "GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0) failed\n"
    "#3 ggml_backend_sched_split_graph ()"
)

# A CUDA OOM while reserving: same reserve frames, different assert.
_CUDA_OOM_ABORT = "\n".join(
    [
        "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 23810.00 MiB on device 0: "
        "cudaMalloc failed: out of memory",
        "/src/ggml/src/ggml-cuda/ggml-cuda.cu:95: GGML_ASSERT(err == cudaSuccess) failed",
        "#2  ggml_abort () from libggml-base.so",
        "#3  ggml_gallocr_reserve_n () from libggml-base.so",
        "#4  ggml_backend_sched_reserve () from libggml-base.so",
        "#5  llama_context::graph_reserve(...) () from libllama.so",
        "#6  llama_context::sched_reserve() () from libllama.so",
    ]
)
# Other aborts inside split_graph (ggml-backend.cpp) share its frame.
_CONTEXT_INIT_ABORT = (
    "ggml/src/ggml-backend.cpp:1082: ggml_backend_sched_split_graph: failed to initialize context\n"
    "#2  ggml_abort ()\n#3  ggml_backend_sched_split_graph ()"
)
_SPLITS_ALLOC_ABORT = (
    "ggml/src/ggml-backend.cpp:1352: GGML_ASSERT(sched->splits != NULL) failed\n"
    "#2  ggml_abort ()\n#3  ggml_backend_sched_split_graph ()"
)
_OOM_OUTPUT = "llama_model_load: error loading model: unable to allocate buffer\nkilled"
_CLEAN_OUTPUT = "main: server is listening on http://127.0.0.1:8080 - starting the main loop"


def test_matcher_fires_on_full_abort_and_short_tail():
    assert LlamaCppBackend._is_sched_reserve_abort(_FULL_ABORT)
    assert "GGML_ASSERT(*cur_backend_id != -1)".lower() not in _ABORT_TAIL.lower()
    assert LlamaCppBackend._is_sched_reserve_abort(_ABORT_TAIL)


def test_matcher_ignores_unrelated_crashes():
    assert not LlamaCppBackend._is_sched_reserve_abort(_SPLIT_AXIS_ABORT)
    assert not LlamaCppBackend._is_sched_reserve_abort(_OOM_OUTPUT)
    assert not LlamaCppBackend._is_sched_reserve_abort(_CUDA_OOM_ABORT)
    assert not LlamaCppBackend._is_sched_reserve_abort(_CONTEXT_INIT_ABORT)
    assert not LlamaCppBackend._is_sched_reserve_abort(_SPLITS_ALLOC_ABORT)
    assert not LlamaCppBackend._is_sched_reserve_abort(_CLEAN_OUTPUT)
    assert not LlamaCppBackend._is_sched_reserve_abort("")


def test_matcher_requires_both_a_ggml_marker_and_a_scheduler_marker():
    assert not LlamaCppBackend._is_sched_reserve_abort("graph_reserve completed in 3ms")
    assert not LlamaCppBackend._is_sched_reserve_abort("ggml_abort: tensor type mismatch")


def test_classifier_surfaces_actionable_message():
    msg = LlamaCppBackend._classify_llama_start_failure(
        _ABORT_TAIL,
        gguf_path = "/x/GLM-5.2-UD-Q6_K-00001-of-00014.gguf",
        model_identifier = "unsloth/GLM-5.2-GGUF",
        returncode = -6,
    )
    assert msg == LlamaCppBackend._sched_reserve_abort_message()
    assert "cur_backend_id" in msg
    assert "failed to start" not in msg  # not the generic fallback


def test_classifier_generic_fallback_unchanged_for_unknown_crash():
    msg = LlamaCppBackend._classify_llama_start_failure(
        "some unrelated failure",
        gguf_path = None,
        model_identifier = None,
        returncode = -6,
    )
    assert "failed to start" in msg and "enough memory" in msg


def test_memo_round_trip_and_isolation(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_text("x")
    b, model = str(binary), "unsloth/GLM-5.2-GGUF"

    LlamaCppBackend._sched_reserve_abort_keys.clear()
    assert not LlamaCppBackend._sched_reserve_aborts(b, model)
    LlamaCppBackend._record_sched_reserve_abort(b, model)
    assert LlamaCppBackend._sched_reserve_aborts(b, model)
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
    st = binary.stat()
    os.utime(binary, ns = (st.st_atime_ns + 10**9, st.st_mtime_ns + 10**9))
    assert not LlamaCppBackend._sched_reserve_aborts(b, model)
    LlamaCppBackend._sched_reserve_abort_keys.clear()


def test_memo_safe_with_missing_binary_or_model():
    assert not LlamaCppBackend._sched_reserve_aborts(None, "m")
    assert not LlamaCppBackend._sched_reserve_aborts("/x", None)
    LlamaCppBackend._record_sched_reserve_abort(None, None)  # no-op, no raise


_REAL_POPEN = subprocess.Popen
_ABORT_OUTPUT = [
    "/src/ggml/src/ggml-backend.cpp:1242: GGML_ASSERT(*cur_backend_id != -1) failed\n",
    "#3  ggml_backend_sched_split_graph ()\n",
    "#5  llama_context::sched_reserve() ()\n",
]


def _write_gguf(path: Path) -> Path:
    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    metadata = string("general.architecture") + struct.pack("<I", 8) + string("llama")
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)
    return path


class _Crashing:
    """A CPU-only backend whose every llama-server spawn aborts in the graph scheduler."""

    def __init__(self, tmp_path):
        binary = tmp_path / "llama-server"
        binary.write_text("x")
        b = LlamaCppBackend()
        b._get_gpu_memory = lambda _binary = None, **kw: []
        b._get_gpu_free_memory = lambda _binary = None, **kw: []
        b._read_gguf_metadata = lambda _path: None
        b._can_estimate_kv = lambda: False
        b._get_gguf_size_bytes = lambda _path: 1024
        b._mmproj_vram_bytes = lambda _path: 0
        b._resolve_launch_mmproj_path = lambda **kwargs: None
        b._apu_ram_shortfall_message = lambda *args, **kwargs: None
        b._amd_apu_wants_unified_memory = lambda *args, **kwargs: False
        b._find_llama_server_binary = lambda include_denied = False: str(binary)
        b._is_vulkan_backend = lambda _binary = None: False
        b._detect_audio_type_strict = lambda: None
        b._apply_detected_audio = lambda _detected: True

        def crashed(timeout, **_kw):
            # As the real wait does on exit: let the drain thread collect the tail.
            if b._stdout_thread is not None:
                b._stdout_thread.join(timeout = 2)
            return False

        b._wait_for_health = crashed
        self.backend = b
        self.gguf = _write_gguf(tmp_path / "model.gguf")
        self.spawns = 0
        self.outputs = None  # per-spawn output override, consumed in order
        self.projector_fails = False  # --mmproj spawns fail without the abort

    def load(self, **load_kwargs) -> str:
        def fake_popen(cmd, **kwargs):
            if not cmd or "--port" not in [str(c) for c in cmd]:
                return _REAL_POPEN(cmd, **kwargs)
            self.spawns += 1
            lines = self.outputs.pop(0) if self.outputs else _ABORT_OUTPUT
            if self.projector_fails and "--mmproj" in [str(c) for c in cmd]:
                lines = ["clip_init: failed to load model\n"]
            return type(
                "Process",
                (),
                {
                    "pid": 123,
                    "returncode": -6,
                    "stdout": iter(lines),
                    "poll": lambda self: -6,
                    "terminate": lambda self: None,
                    "wait": lambda self, timeout = None: -6,
                    "kill": lambda self: None,
                },
            )()

        with (
            patch.object(subprocess, "Popen", side_effect = fake_popen),
            pytest.raises(RuntimeError) as err,
        ):
            self.backend.load_model(
                GgufLoadIntent(gguf_path = str(self.gguf), model_identifier = "t", **load_kwargs)
            )
        return str(err.value)


@pytest.fixture
def crashing(tmp_path):
    LlamaCppBackend._sched_reserve_abort_keys.clear()
    yield _Crashing(tmp_path)
    LlamaCppBackend._sched_reserve_abort_keys.clear()


def test_scheduler_abort_names_the_cause(crashing):
    assert crashing.load() == LlamaCppBackend._sched_reserve_abort_message()


def test_identical_replay_fails_fast_without_spawning(crashing):
    crashing.load()
    before = crashing.spawns
    kills = []
    crashing.backend._kill_process = lambda: kills.append(1)
    assert crashing.load() == LlamaCppBackend._sched_reserve_abort_message()
    assert crashing.spawns == before
    assert not kills, "a memoed replay must not tear down the running server"


@pytest.mark.parametrize(
    "change", [{"n_ctx": 2048}, {"tensor_parallel": True}, {"force_reload": True}]
)
def test_a_changed_setting_is_allowed_to_retry(crashing, change):
    crashing.load()
    before = crashing.spawns
    crashing.load(**change)
    assert crashing.spawns > before


def test_hub_cache_hint_does_not_defeat_the_memo(crashing):
    crashing.load()
    before = crashing.spawns
    hint = (str(crashing.gguf), "unsloth/GLM-5.2-GGUF", "main", ((crashing.gguf.name, 1),))
    crashing.load(verified_gguf = hint)
    assert crashing.spawns == before


def test_forced_reload_that_does_not_abort_clears_the_memo(crashing):
    crashing.load()
    crashing.outputs = [["llama_model_load: error loading model\n"]] * 2
    crashing.load(force_reload = True)
    before = crashing.spawns
    crashing.load()
    assert crashing.spawns > before


def test_text_only_retry_abort_is_memoed(crashing, tmp_path):
    mmproj = _write_gguf(tmp_path / "mmproj.gguf")
    crashing.backend._resolve_launch_mmproj_path = lambda **kwargs: str(mmproj)
    crashing.projector_fails = True
    crashing.load(mmproj_path = str(mmproj), is_vision = True)
    before = crashing.spawns
    crashing.load(mmproj_path = str(mmproj), is_vision = True)
    assert crashing.spawns == before
