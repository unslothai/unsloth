# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Peer venv relocation under another home, mandatory peer activation, schedule limits checked
before loading, torch <= 2.9 output merging, and Phi-3's fused projections."""

from __future__ import annotations

import copy
import importlib.util
import os
import socket
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cluster():
    return _load("spark_cluster_for_r20", "studio/spark_cluster.py")


@pytest.fixture
def pipeline():
    return _load("spark_pipeline_for_r20", "studio/spark_pipeline.py")


@pytest.mark.skipif(os.name == "nt", reason = "runs the peer's bash script")
def test_a_venv_under_the_peers_own_home_is_relocated(cluster, tmp_path) -> None:
    home = tmp_path / "peer_home"
    venv = home / ".unsloth" / "studio" / "unsloth_studio"
    (venv / "bin").mkdir(parents = True)
    old = "/home/initiator/.unsloth/studio/unsloth_studio"
    (venv / "bin" / "activate").write_text(f'VIRTUAL_ENV="{old}"\n', encoding = "utf-8")
    (venv / "bin" / "torchrun").write_text(f"#!{old}/bin/python\n", encoding = "utf-8")
    script = cluster.venv_relocate_script(old, "$HOME/.unsloth/studio/unsloth_studio")
    env = dict(os.environ, HOME = str(home))
    subprocess.run(["bash", "-s"], input = script.encode(), env = env, check = True)
    assert str(venv) in (venv / "bin" / "activate").read_text(encoding = "utf-8")
    assert (venv / "bin" / "torchrun").read_text(encoding = "utf-8").startswith(f"#!{venv}/bin/")


def test_the_peer_rank_does_not_start_without_its_venv(cluster) -> None:
    import inspect

    src = inspect.getsource(cluster)
    line = next(l for l in src.splitlines() if "then {env}; {node1};" in l)
    assert line.strip().startswith('f"if [ -f {activate} ] && . {activate}; then')


@pytest.mark.parametrize(
    "schedule, world, microbatches",
    [("gpipe", 2, 1), ("1f1b", 2, 1), ("interleaved", 2, 5), ("zerobubble", 2, 7)],
)
def test_schedule_limits_fail_before_any_load(pipeline, schedule, world, microbatches) -> None:
    with pytest.raises(RuntimeError, match = "--microbatches"):
        pipeline.torch_pp_plan(schedule, world, microbatches, 2, 8)


@pytest.mark.parametrize(
    "schedule, world, microbatches",
    [("gpipe", 2, 2), ("1f1b", 2, 4), ("interleaved", 2, 4), ("interleaved", 2, 3)],
)
def test_schedule_limits_leave_valid_counts_alone(pipeline, schedule, world, microbatches) -> None:
    assert pipeline.torch_pp_plan(schedule, world, microbatches, 2, 8)["schedule"] == schedule


def test_phi3_fused_projections_get_lora(pipeline) -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.Phi3Config(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        max_position_embeddings = 32,
        pad_token_id = 0,
    )
    targets = pipeline.lora_target_modules(transformers.Phi3ForCausalLM(config))
    assert {"qkv_proj", "o_proj", "gate_up_proj", "down_proj"} <= set(targets)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_torch_without_return_outputs_does_not_merge_the_logits(pipeline, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch.distributed.pipelining")
    import torch.distributed as dist
    from torch.distributed.pipelining import schedules

    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    original = schedules.ScheduleGPipe.step

    def step_as_in_2_9(
        self,
        *args,
        target = None,
        losses = None,
        **kwargs,
    ):
        return original(self, *args, target = target, losses = losses, **kwargs)

    monkeypatch.setattr(schedules.ScheduleGPipe, "step", step_as_in_2_9)
    config = transformers.LlamaConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        max_position_embeddings = 32,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config)
    model.config._attn_implementation = "eager"
    model.train()
    ids = torch.randint(0, 64, (4, 6))
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(_free_port()))
    dist.init_process_group("gloo", rank = 0, world_size = 1)
    try:
        plan = pipeline.torch_pp_plan("gpipe", 1, 2, 2, config.num_hidden_layers)
        schedule, _mods, step_kw, _scale = pipeline.build_torch_schedule(
            copy.deepcopy(model),
            plan,
            pipeline.plan_for_rank(plan, 0),
            microbatches = 2,
            device = "cpu",
            grad_checkpoint = False,
            log = lambda *a, **k: None,
        )
        assert "return_outputs" not in step_kw
        assert schedule.step(ids, target = ids.clone(), losses = [], **step_kw) is None
    finally:
        dist.destroy_process_group()
