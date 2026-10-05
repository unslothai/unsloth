# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import queue
import ast
from pathlib import Path
import subprocess
import threading
import os
from types import ModuleType, SimpleNamespace
import sys

import pytest

from core.training import ddp
from core.training.dataset_bounds import world_size_from_env
from core.training.ddp import (
    _RankEvents,
    model_load_dtype_for_training,
    training_precision_flags_for_dtype,
)
from core.training.worker import _create_trainer_progress_callback


class _Events:
    def __init__(self):
        self.items = []

    def put(self, event):
        self.items.append(event)


@pytest.mark.parametrize(
    "world_size, checkpointing, family, audio, expected",
    [
        (2, True, "llama", False, False),
        (1, True, "llama", False, None),
        (2, False, "llama", False, None),
        (2, True, "qwen3_moe", False, None),
        (2, True, "llama", True, None),
        ("auto", True, "llama", False, None),
        ("", True, "llama", False, None),
        ("eight", True, "llama", False, None),
        ("auto", True, "llama", False, False),
    ],
)
def test_dense_llama_ddp_checkpointing_disables_unused_parameter_discovery(
    monkeypatch, world_size, checkpointing, family, audio, expected
):
    # Execute the real configuration block without importing GPU dependencies.
    source = Path(__file__).parents[1] / "core" / "training" / "trainer.py"
    tree = ast.parse(source.read_text())
    blocks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Assign)
            and any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "config_args"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "ddp_find_unused_parameters"
                for target in child.targets
            )
            for child in node.body
        )
    ]
    assert len(blocks) == 1
    # Clear other launchers so this block reads only this test's WORLD_SIZE.
    from core.training.dataset_bounds import WORLD_SIZE_ENV_FILES, WORLD_SIZE_ENV_VARS

    for name in WORLD_SIZE_ENV_VARS + WORLD_SIZE_ENV_FILES:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    if world_size == "auto" and expected is False:
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    config = {"gradient_checkpointing": checkpointing}
    backend = SimpleNamespace(
        is_audio = audio,
        is_vlm = False,
        is_audio_vlm = False,
        model = SimpleNamespace(config = SimpleNamespace(model_type = family)),
    )
    exec(
        compile(ast.Module(body = blocks, type_ignores = []), str(source), "exec"),
        {"world_size_from_env": world_size_from_env, "self": backend, "config_args": config},
    )
    assert config.get("ddp_find_unused_parameters") is expected


def test_embedding_worker_imports_only_used_unsloth_symbols():
    source = Path(__file__).parents[1] / "core" / "training" / "worker.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    worker = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_embedding_training"
    )
    imports = [
        name.name
        for node in ast.walk(worker)
        if isinstance(node, ast.ImportFrom) and node.module == "unsloth"
        for name in node.names
    ]
    assert imports == ["FastSentenceTransformer"]


def test_nonzero_rank_oom_is_captured_and_signals_failure_without_duplicate_event():
    events = _Events()
    failures = {}
    rank_events = _RankEvents(1, events, failures)

    rank_events.put({"type": "error", "error": "CUDA out of memory", "stack": "trace"})

    assert rank_events.error["error"] == "CUDA out of memory"
    assert failures[1]["error"] == "CUDA out of memory"
    assert events.items == []


def test_launcher_relays_one_terminal_error_with_original_rank_and_cause(monkeypatch):
    class _Manager:
        def __init__(self):
            self.rank_queue = queue.Queue()
            self.failures = {}

        def Queue(self):
            return self.rank_queue

        def Event(self):
            return threading.Event()

        def Value(self, _kind, value):
            return SimpleNamespace(value = value)

        def dict(self):
            return self.failures

        def shutdown(self):
            pass

    manager = _Manager()
    monkeypatch.setattr(ddp.mp, "Manager", lambda: manager)
    monkeypatch.setattr(ddp, "_common_cuda_dtype", lambda _gpu_ids: "fp16")

    def _failed_spawn(_fn, *, args, nprocs, join):
        assert nprocs == 3 and join is True
        failures = args[-1]
        failures[2] = {
            "type": "error",
            "error": "CUDA out of memory",
            "stack": "original traceback",
            "ts": 123.0,
        }
        raise RuntimeError("spawn observed failed rank")

    torch_module = ModuleType("torch")
    torch_mp = ModuleType("torch.multiprocessing")
    torch_mp.spawn = _failed_spawn
    torch_module.multiprocessing = torch_mp
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.multiprocessing", torch_mp)
    events = _Events()

    ddp.run_ddp_training_process(
        event_queue = events,
        stop_queue = queue.Queue(),
        config = {"resolved_gpu_ids": [0, 1, 2]},
    )

    assert events.items == [
        {
            "type": "error",
            "error": "DDP rank 2 failed: CUDA out of memory",
            "stack": "original traceback",
            "ts": 123.0,
        }
    ]


def test_rank_zero_error_is_captured_for_launcher(monkeypatch):
    events = _Events()
    failures = {}
    rank_events = _RankEvents(0, events, failures)

    rank_events.put({"type": "error", "error": "model load failed", "stack": "trace"})

    assert failures[0]["error"] == "model load failed"
    assert rank_events.error["error"] == "model load failed"
    assert events.items == []


def test_repeated_rank_error_does_not_relay_twice():
    events = _Events()
    failures = {}
    rank_events = _RankEvents(1, events, failures)
    error = {"type": "error", "error": "CUDA out of memory", "stack": "trace"}

    rank_events.put(error)
    rank_events.put(error)

    assert len(failures) == 1
    assert events.items == []


def test_nonzero_rank_does_not_publish_progress_as_global_progress():
    events = _Events()
    rank_events = _RankEvents(2, events, {})

    rank_events.put({"type": "progress", "step": 1})
    rank_events.put({"type": "status", "message": "loading model"})

    assert events.items == []


def test_rank_zero_forwards_normal_training_events():
    events = _Events()
    rank_events = _RankEvents(0, events, {})

    rank_events.put({"type": "progress", "step": 3})
    complete = {"type": "complete", "output_dir": "/tmp/model"}
    rank_events.put(complete)

    assert events.items == [{"type": "progress", "step": 3}, complete]


def test_rank_config_carries_full_ddp_world_size(monkeypatch):
    # _rank_entry intentionally rewrites the child environment. Isolate that
    # mutation so later CPU tests do not inherit a fictitious distributed run.
    monkeypatch.setattr(ddp.os, "environ", os.environ.copy())
    captured = []
    worker_module = ModuleType("core.training.worker")
    worker_module.run_training_process = lambda **kwargs: captured.append(kwargs["config"])
    monkeypatch.setitem(sys.modules, "core.training.worker", worker_module)

    ddp._rank_entry(
        1,
        {"resolved_gpu_ids": [2, 4, 7], "_ddp_common_dtype": "bf16"},
        _Events(),
        threading.Event(),
        SimpleNamespace(value = True),
        12345,
        {},
    )

    assert captured[0]["_ddp_world_size"] == 3
    assert captured[0]["resolved_gpu_ids"] == [4]
    assert ddp.os.environ["UNSLOTH_DDP_COMMON_DTYPE"] == "bf16"


@pytest.mark.parametrize(
    "gpu_ids, output, expected",
    [
        ([0, 1], "0, 7.5\n1, 8.6\n", "fp16"),  # Turing + Ampere
        ([1, 2], "1, 8.6\n2, 12.0\n", "bf16"),  # Ampere+ cards only
        ([0, 1], "0, 6.1\n1, 6.1\n", "fp16"),  # Pascal cards
        ([0, 1], "0, 8.6\n", "fp16"),  # Missing capability fails closed
    ],
)
def test_common_cuda_dtype_requires_native_bf16_on_every_selected_gpu(
    monkeypatch, gpu_ids, output, expected
):
    from utils.hardware import gpu_query

    monkeypatch.setattr(
        gpu_query,
        "run_nvidia_smi",
        lambda *args, **kwargs: SimpleNamespace(returncode = 0, stdout = output),
    )
    monkeypatch.setattr(ddp, "_nvidia_smi_child_env", lambda: {})
    assert ddp._common_cuda_dtype(gpu_ids) == expected


def test_common_cuda_dtype_falls_back_to_fp16_when_probe_fails(monkeypatch):
    from utils.hardware import gpu_query

    def fail_probe(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 10)

    monkeypatch.setattr(gpu_query, "run_nvidia_smi", fail_probe)
    monkeypatch.setattr(ddp, "_nvidia_smi_child_env", lambda: {})
    assert ddp._common_cuda_dtype([0, 1]) == "fp16"


@pytest.mark.parametrize(
    "ddp_dtype, local_bf16, expected",
    [
        ("fp16", True, {"fp16": True, "bf16": False}),
        ("fp16", False, {"fp16": True, "bf16": False}),
        ("bf16", False, {"fp16": False, "bf16": True}),
        (None, True, {"fp16": False, "bf16": True}),
        (None, False, {"fp16": True, "bf16": False}),
    ],
)
def test_trainer_precision_flags_honor_ddp_common_dtype(ddp_dtype, local_bf16, expected):
    assert training_precision_flags_for_dtype(ddp_dtype, local_bf16) == expected


@pytest.mark.parametrize(
    "ddp_dtype, is_rocm, supports_bf16, expected",
    [
        ("fp16", False, True, "fp16"),
        ("fp16", False, False, "fp16"),
        ("bf16", False, False, "bf16"),
        (None, True, False, "fp16"),
        (None, False, False, None),
    ],
)
def test_model_load_dtype_matches_ddp_training_precision(
    ddp_dtype, is_rocm, supports_bf16, expected
):
    assert model_load_dtype_for_training(ddp_dtype, is_rocm, supports_bf16) == expected


def test_progress_callback_preserves_aggregated_eval_loss_for_ddp(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    events = _Events()
    progress = SimpleNamespace(
        step = 5,
        loss = 0.4,
        learning_rate = 0.001,
        grad_norm = 1.0,
        num_tokens = 100,
        epoch = 0.5,
        eval_loss = 0.8,
        total_steps = 20,
        is_run_summary = False,
        elapsed_seconds = 2.0,
        eta_seconds = 10.0,
        session_start_step = 0,
        status_message = None,
        warnings = [],
    )

    _create_trainer_progress_callback(events)(progress)

    assert events.items[0]["eval_loss"] == 0.8
    assert events.items[0]["loss"] == 0.4


def test_progress_callback_leaves_single_gpu_eval_loss_unchanged():
    events = _Events()
    progress = SimpleNamespace(
        step = 5,
        loss = 0.4,
        learning_rate = 0.001,
        grad_norm = 1.0,
        num_tokens = 100,
        epoch = 0.5,
        eval_loss = 0.8,
        total_steps = 20,
        is_run_summary = False,
        elapsed_seconds = 2.0,
        eta_seconds = 10.0,
        session_start_step = 0,
        status_message = None,
        warnings = [],
    )

    _create_trainer_progress_callback(events)(progress)

    assert events.items[0]["eval_loss"] == 0.8
