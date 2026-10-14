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


def test_the_estimate_counts_every_retained_logit_chunk(cluster) -> None:
    dims = dict(hidden = 4096, layers = 32, vocab = 128256)
    base = dict(size_gib = 16.0, world = 2, batch = 32, microbatches = 32, seq = 4096, **dims)
    kept = cluster.training_memory_estimate(**base, retains_outputs = True)
    freed = cluster.training_memory_estimate(**base, retains_outputs = False)
    # One row of 4096 x 128256 bf16 logits per microbatch, all 32 alive until the step ends.
    assert kept["logits_gib"] - freed["logits_gib"] == pytest.approx(32 * 4096 * 128256 * 2 / 2**30)


def test_mla_attention_inputs_get_lora(pipeline) -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.DeepseekV3Config(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        moe_intermediate_size = 16,
        num_hidden_layers = 1,
        first_k_dense_replace = 1,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        q_lora_rank = 16,
        kv_lora_rank = 16,
        qk_rope_head_dim = 4,
        qk_nope_head_dim = 4,
        v_head_dim = 8,
        max_position_embeddings = 32,
    )
    targets = set(pipeline.lora_target_modules(transformers.DeepseekV3ForCausalLM(config)))
    assert {"q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"} <= targets


def test_a_middle_stage_does_not_read_the_tied_embedding(pipeline, tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import safetensors

    config = transformers.LlamaConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 6,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        max_position_embeddings = 32,
        tie_word_embeddings = True,
    )
    saved = str(tmp_path / "tied")
    transformers.LlamaForCausalLM(config).save_pretrained(saved, safe_serialization = True)
    read = []
    real = safetensors.safe_open

    class Recording:
        def __init__(self, *a, **k):
            self._h = real(*a, **k)

        def __enter__(self):
            self._inner = self._h.__enter__()
            return self

        def __exit__(self, *exc):
            return self._h.__exit__(*exc)

        def keys(self):
            return self._inner.keys()

        def get_tensor(self, key):
            read.append(key)
            return self._inner.get_tensor(key)

        def get_slice(self, key):
            read.append(key)
            return self._inner.get_slice(key)

    monkeypatch.setattr(safetensors, "safe_open", Recording)
    pipeline.build_stage_model(
        saved,
        rank = 1,
        world = 3,
        device = "cpu",
        shard_load = True,
        dtype = torch.float32,
        log = lambda *a, **k: None,
    )
    assert read and "model.embed_tokens.weight" not in read


def test_a_tied_weight_shared_by_two_stages_on_a_rank_is_scaled_once(pipeline, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch.distributed.pipelining")
    import torch.distributed as dist

    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    config = transformers.LlamaConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        max_position_embeddings = 32,
        tie_word_embeddings = True,
    )
    torch.manual_seed(0)
    base = transformers.LlamaForCausalLM(config)
    base.config._attn_implementation = "eager"
    ids = torch.randint(0, 64, (4, 6))
    reference = copy.deepcopy(base)
    reference.train()
    pipeline.pp_loss_fn(reference(ids).logits, ids.clone()).backward()
    want = reference.model.embed_tokens.weight.grad.clone()

    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(_free_port()))
    dist.init_process_group("gloo", rank = 0, world_size = 1)
    try:
        # A V layout on one rank: stage 0 (embedding) and stage 1 (tied head) share a Parameter.
        plan = pipeline.torch_pp_plan("zbv", 1, 2, 2, config.num_hidden_layers)
        model = copy.deepcopy(base)
        model.train()
        schedule, mods, step_kw, scale = pipeline.build_torch_schedule(
            model,
            plan,
            pipeline.plan_for_rank(plan, 0),
            microbatches = 2,
            device = "cpu",
            grad_checkpoint = False,
            log = lambda *a, **k: None,
        )
        assert len(mods) == 2
        for mod in mods:
            for p in mod.parameters():
                p.grad = None
        schedule.step(ids, target = ids.clone(), losses = [], **step_kw)
        scale()
        got = model.model.embed_tokens.weight.grad
        assert got is not None
        ratio = (got.norm() / want.norm()).item()
        assert ratio == pytest.approx(1.0, abs = 1e-3), f"tied gradient is {ratio:.3f}x the reference"
    finally:
        dist.destroy_process_group()
