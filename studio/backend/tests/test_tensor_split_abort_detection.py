# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Detect buried tensor-split asserts without latching drafter crashes."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = str(_TESTS_DIR.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402


def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, _TESTS_DIR / file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_placement = _load("_placement_harness_split_abort", "test_llama_cpp_placement.py")

_REAL_POPEN = subprocess.Popen
_TWO_GPUS = [(0, 20_000, 24_000), (1, 20_000, 24_000)]

# gdb output pushes the assert beyond the last 50 lines.
_GDB_ABORT = (
    [
        "common_speculative_impl_draft_dflash: - n_max=3, n_min=0, p_min=0.00\n",
        "ggml/src/ggml-backend-meta.cpp:543: "
        "GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0) failed\n",
    ]
    + [f"[New LWP {400000 + i}]\n" for i in range(130)]
    + [
        "[Thread debugging using libthread_db enabled]\n",
        "#1  0x... in ggml_print_backtrace () from libggml-base.so.0\n",
        "#2  0x... in ggml_abort () from libggml-base.so.0\n",
    ]
)


@pytest.fixture
def crashing(tmp_path, monkeypatch):
    for name in ("CUDA_VISIBLE_DEVICES", "LLAMA_ARG_SPLIT_MODE", "LLAMA_ARG_SPEC_TYPE"):
        monkeypatch.delenv(name, raising = False)
    LlamaCppBackend._tensor_split_abort_keys.clear()
    backend, gguf = _placement._backend(tmp_path, vulkan = False, memory = list(_TWO_GPUS))

    def crashed(timeout, **_kw):
        if backend._stdout_thread is not None:
            backend._stdout_thread.join(timeout = 2)
        return False

    backend._wait_for_health = crashed
    spawns: list[list[str]] = []

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        spawns.append([str(c) for c in cmd])
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "returncode": -6,
                "stdout": iter(_GDB_ABORT),
                "poll": lambda self: -6,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: -6,
                "kill": lambda self: None,
            },
        )()

    def load(
        extra_args = None,
        speculative_type = "off",
        dflash_draft_path = None,
    ):
        with (
            patch.object(subprocess, "Popen", side_effect = fake_popen),
            pytest.raises(RuntimeError) as err,
        ):
            backend.load_model(
                GgufLoadIntent(
                    gguf_path = str(gguf),
                    model_identifier = "test",
                    tensor_parallel = True,
                    speculative_type = speculative_type,
                    dflash_draft_path = dflash_draft_path,
                    extra_args = tuple(extra_args) if extra_args else None,
                )
            )
        return str(err.value)

    yield backend, load, spawns, tmp_path
    LlamaCppBackend._tensor_split_abort_keys.clear()


def _latched(backend) -> bool:
    return LlamaCppBackend._tensor_split_aborts(
        backend._find_llama_server_binary(), "test", ("f16", "f16")
    )


def test_assert_behind_a_gdb_dump_raises_to_layer_split_on_the_first_spawn(crashing):
    backend, load, spawns, _tmp = crashing

    message = load()

    assert "split-axis geometry" in message
    assert len(spawns) == 1, "the --fit and flash-attn retries cannot fix a split-axis abort"
    assert "--split-mode" in spawns[0] and "tensor" in spawns[0]
    assert _latched(backend)


@pytest.mark.parametrize("source", ["auto_sidecar", "extra_args", "environment"])
def test_abort_with_a_separate_drafter_is_not_latched_for_the_model(crashing, monkeypatch, source):
    backend, load, spawns, tmp_path = crashing
    drafter = tmp_path / "dflash-test-Q4_K_M.gguf"
    drafter.write_bytes(b"GGUF")

    if source == "auto_sidecar":
        probe = backend.probe_server_capabilities
        backend.probe_server_capabilities = lambda *a, **kw: {
            **probe(*a, **kw),
            "supports_dflash": True,
        }
        message = load(speculative_type = "auto", dflash_draft_path = str(drafter))
    elif source == "extra_args":
        message = load(["--model-draft", str(drafter), "--spec-type", "draft-dflash"])
    else:
        monkeypatch.setenv("LLAMA_ARG_SPEC_DRAFT_MODEL", str(drafter))
        message = load(["--spec-type", "draft-dflash"])

    assert "split-axis geometry" in message
    assert len(spawns) == 1
    if source == "environment":
        assert "--model-draft" not in spawns[0]
    else:
        assert spawns[0][spawns[0].index("--model-draft") + 1] == str(drafter)
    assert not _latched(backend), "a drafter's abort must not keep tensor off for other loads"
