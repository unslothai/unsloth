# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""offload_layers layer-major gradient accumulation under DDP, two gloo ranks on CPU (tests/_layer_major_ddp_worker.py):
- gradients at every optimizer step, final LoRA weights and logged losses bitwise equal to plain DDP, and within fp32
  tolerance of one process running each step's global batch; one window and one gradient reduce per step;
- a dropped reduce makes that comparison fail (the check can fail);
- decoder layers with a constant DDP-broadcast buffer (as Unsloth's per-layer rotary inv_freq) keep their windows;
  a forward that writes one makes every rank run the step the standard way;
- a rank that cannot run a window (memory check, an abort in its forward or in its decoder backward) makes every
  rank run the step the standard way, bitwise equal to plain DDP, with no hang;
- TRL's SFTTrainer, whose loss parts gather metrics across ranks: weights and logged metrics bitwise, also when one
  rank stops in its forward (no rank enters a loss part's gather alone)."""

import importlib.util
import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")


def _zoo_dir():
    explicit = os.environ.get("UNSLOTH_ZOO_DIR")
    if explicit:
        return explicit
    spec = importlib.util.find_spec("unsloth_zoo")
    if spec is None or not spec.submodule_search_locations:
        return None
    return os.path.dirname(list(spec.submodule_search_locations)[0])


def _engine_votes(zoo_dir):
    path = os.path.join(zoo_dir, "unsloth_zoo", "block_swap_layer_major.py")
    if not os.path.exists(path):
        return False
    spec = importlib.util.spec_from_file_location("_lmga_engine_probe", path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return "agree" in inspect.signature(engine.run_layer_major).parameters


@pytest.fixture(scope = "module")
def reports(tmp_path_factory):
    if sys.platform == "win32":
        pytest.skip("torchrun's rendezvous needs libuv, which Windows torch builds lack")
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed is not available")
    zoo = _zoo_dir()
    if zoo is None or not _engine_votes(zoo):
        pytest.skip("unsloth_zoo without rank votes in block_swap_layer_major")
    from tests._rendezvous_port import free_port

    tmp = tmp_path_factory.mktemp("lmga_ddp")
    worker = Path(__file__).with_name("_layer_major_ddp_worker.py")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", TMPDIR = str(tmp), OMP_NUM_THREADS = "1")
    env.pop("UNSLOTH_OFFLOAD_LAYER_MAJOR", None)
    # Earlier tests' Trainers export ACCELERATE_MIXED_PRECISION; the ranks and the fp32 reference must match.
    for key in [k for k in env if k.startswith("ACCELERATE_")]:
        env.pop(key)
    # The suite's conftest refuses unsloth_zoo's CPU path; these ranks are CPU by design (gloo).
    env.pop("UNSLOTH_DISABLE_CPU_FALLBACK", None)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            "2",
            "--master_port",
            str(free_port()),
            str(worker),
            str(tmp),
            zoo,
        ],
        env = env,
        capture_output = True,
        text = True,
        timeout = 900,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-6000:]
    found = {int(r): json.loads(js) for r, js in re.findall(r"RANK_REPORT (\d) (\{.*\})", out)}
    assert sorted(found) == [0, 1], out[-6000:]
    return found


def test_layer_major_ddp_matches_plain_ddp_bitwise_and_one_process(reports):
    for rank, r in reports.items():
        plain, lm = r["plain"], r["layer_major"]
        assert plain["windows"] == 0 and plain["reduces"] == 0, (rank, plain)
        assert lm["windows"] == 2 and lm["reduces"] == 2 and lm["aborts"] == 0, (rank, lm)
        assert lm["steps"] == 2 and lm["bitwise_vs_plain"], (rank, lm)
        assert lm["close_to_one_process"] and plain["close_to_one_process"], (rank, r)
        assert r["one_process_vs_plain_max_diff"] < 1e-5, (rank, r)


def test_a_dropped_reduce_fails_the_comparison(reports):
    for rank, r in reports.items():
        d = r["dropped_reduce"]
        assert d["windows"] == 2 and d["reduces"] == 0, (rank, d)
        assert not d["bitwise_vs_plain"] and not d["close_to_one_process"], (rank, d)
        assert d["max_diff_vs_plain"] > 1e-3, (rank, d)


@pytest.mark.parametrize(
    "arm, aborts",
    [
        ("rank1_unfit", (0, 0)),
        ("rank1_instance_wrapper", (0, 0)),
        ("rank1_forward_abort", (1, 1)),
        ("rank1_backward_oom", (1, 1)),
    ],
)
def test_a_rank_without_a_window_keeps_every_rank_standard(reports, arm, aborts):
    for rank, r in reports.items():
        a = r[arm]
        assert a["windows"] == 0 and a["reduces"] == 0, (rank, a)
        assert a["aborts"] == aborts[rank], (rank, a)
        assert a["bitwise_vs_plain"] and a["close_to_one_process"], (rank, a)
    # Rank 0 could run its window and says why it did not.
    assert reports[0][arm]["other_rank_message"], reports[0][arm]


def test_a_forward_that_writes_a_broadcast_buffer_keeps_the_step_standard(reports):
    for rank, r in reports.items():
        b = r["buffer_written"]
        assert b["windows"] == 0 and b["aborts"] == 1 and b["reduces"] == 0, (rank, b)
        # The output reads the buffer: the retry is bitwise plain DDP only if the window's writes were put back.
        assert b["bitwise_vs_its_plain"], (rank, b)


def test_hooks_on_the_ddp_wrapper_keep_the_step_standard(reports):
    """The window runs the module inside DDP: a hook on the wrapper would be skipped, so no window runs."""
    for rank, r in reports.items():
        h = r["ddp_wrapper_hook"]
        assert h["windows"] == 0 and h["reduces"] == 0 and h["wrapper_hook_message"], (rank, h)
        assert h["bitwise_vs_plain"], (rank, h)


def test_unsloth_gradient_checkpointing_at_seq_512(reports):
    """Unsloth's default (use_gradient_checkpointing = "unsloth", max_seq_length >= 512): its checkpoint shim on
    every layer. An engine that accepts the shim runs windows; one that refuses it aborts on every rank alike."""
    for rank, r in reports.items():
        g = r["unsloth_gc"]
        if g is None:
            pytest.skip("unsloth_zoo.gradient_checkpointing does not import here")
        assert g["plain_windows"] == 0 and g["bitwise"], (rank, g)
        if g["accepted"]:
            assert g["windows"] == 2 and g["reduces"] == 2 and g["aborts"] == 0, (rank, g)
        else:
            assert g["windows"] == 0 and g["reduces"] == 0 and g["aborts"] == 1, (rank, g)


def test_sft_trainer_metric_gathers_inside_windows(reports):
    for rank, r in reports.items():
        sft = r["sft"]
        if sft is None:
            pytest.skip("trl / datasets / tokenizers not installed")
        assert sft["plain_windows"] == 0 and sft["windows"] == 2, (rank, sft)
        assert sft["bitwise"] and sft["logs_equal"], (rank, sft)
        assert sft["abort_windows"] == 0 and sft["abort_aborts"] == 1 and sft["abort_bitwise"], (
            rank,
            sft,
        )
