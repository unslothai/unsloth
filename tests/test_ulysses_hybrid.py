# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses context parallelism for Gated DeltaNet hybrids (Qwen3.5, Qwen3-Next), on CPU gloo ranks."""

import os
import tempfile
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

import unsloth  # noqa: F401
from unsloth import context_parallel as cp
from unsloth.utils import attention_dispatch as ad
from unsloth.utils import ulysses_hybrid as uh

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3NextConfig"):
    pytest.skip("transformers without Qwen3-Next", allow_module_level = True)
KINDS = ["qwen3_5", "qwen3_next"] if hasattr(transformers, "Qwen3_5TextConfig") else ["qwen3_next"]


def _config(kind, **overrides):
    common = dict(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 3,
        layer_types = ["linear_attention", "full_attention", "linear_attention"],
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        linear_num_key_heads = 2,
        linear_num_value_heads = 4,
        linear_key_head_dim = 8,
        linear_value_head_dim = 8,
        linear_conv_kernel_dim = 4,
        attention_dropout = 0.0,
        tie_word_embeddings = False,
    )
    common.update(overrides)
    if kind == "qwen3_next":
        common.update(
            num_experts = 2,
            num_experts_per_tok = 1,
            moe_intermediate_size = 16,
            shared_expert_intermediate_size = 16,
            # Dense MLPs: the MoE path may pick CUDA kernels in a forked CPU worker.
            mlp_only_layers = list(range(common["num_hidden_layers"])),
        )
        return transformers.Qwen3NextConfig(**common), transformers.Qwen3NextForCausalLM
    return transformers.Qwen3_5TextConfig(**common), transformers.Qwen3_5ForCausalLM


def _vlm(**overrides):
    # Shipped Qwen3.5 checkpoints carry a vision_config even for text-only training.
    torch.manual_seed(0)
    text_config, _ = _config("qwen3_5", **overrides)
    config = transformers.Qwen3_5Config(
        text_config = text_config.to_dict(),
        vision_config = dict(
            depth = 1, hidden_size = 16, intermediate_size = 32, num_heads = 2, out_hidden_size = 32
        ),
        tie_word_embeddings = False,
    )
    model = transformers.Qwen3_5ForConditionalGeneration(config).train()
    model.config._attn_implementation = "sdpa"
    return model


def _model(kind, **overrides):
    torch.manual_seed(0)
    config, cls = _config(kind, **overrides)
    model = cls(config).train()
    model.config._attn_implementation = "sdpa"
    return model


def _force_torch_kernels(model):
    # The parity check is about the exchange, not the kernels: fla / Triton would touch CUDA in a
    # forked CPU worker whenever a GPU is visible. Only called inside the worker process.
    import inspect

    from unsloth.utils.packing import _hub_closure
    for module in model.modules():
        if not type(module).__name__.endswith("GatedDeltaNet"):
            continue
        globs = inspect.getmodule(type(module)).__dict__
        for name in ("causal_conv1d_fn", "torch_chunk_gated_delta_rule"):
            torch_function = _hub_closure(globs.get(name)).get("torch_function")
            if torch_function is not None:
                globs[name] = torch_function
        if "chunk_gated_delta_rule" in vars(module):
            # transformers 4.57 binds the kernels on the instance.
            module.chunk_gated_delta_rule = globs["torch_chunk_gated_delta_rule"]
            module.causal_conv1d_fn = None


def _manager(size, rank):
    manager = object.__new__(cp.ContextParallelManager)
    manager.size = size
    manager.backend = "ulysses"
    manager.mesh = types.SimpleNamespace(get_group = lambda: None, get_local_rank = lambda: rank)
    return manager


def _loss(logits, shift_labels, count):
    return (
        F.cross_entropy(
            logits.float().flatten(0, 1), shift_labels.flatten(), ignore_index = -100, reduction = "sum"
        )
        / count
    )


def _parity_worker(rank, world, kind, control):
    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    model = _vlm() if kind == "qwen3_5_vlm" else _model(kind)
    if kind == "qwen3_5_vlm":
        assert model.config.vision_config is not None
        cp._check_cp_model(model, world, "ulysses")
    _force_torch_kernels(model)
    calls = []
    exchange = uh.ulysses_gated_delta_forward

    def counted(*args, **kwargs):
        calls.append(1)
        return exchange(*args, **kwargs)

    uh.ulysses_gated_delta_forward = counted
    ids = torch.randint(0, 64, (2, 13), generator = torch.Generator().manual_seed(1))
    shift = F.pad(ids, (0, 1), value = -100)[:, 1:]
    count = shift.ne(-100).sum()
    reference_loss = _loss(model(input_ids = ids, use_cache = False).logits, shift, count)
    reference_loss.backward()
    reference = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none = True)
    assert not calls

    manager = _manager(world, rank)
    manager.attach_attention_hooks(model)
    if control:
        for module in model.modules():
            if getattr(module, "_unsloth_cp_orig_forward", None) is not None:
                module.forward = module._unsloth_cp_orig_forward
    inputs = {"input_ids": ids.clone(), "labels": ids.clone()}
    manager._prepare_inputs(inputs)
    with manager.apply(inputs):
        out = model(
            input_ids = inputs["input_ids"], position_ids = inputs["position_ids"], use_cache = False
        )
        loss = _loss(out.logits, inputs["shift_labels"], count)
        loss.backward()
    if not control:
        config_kind = "qwen3_5" if kind == "qwen3_5_vlm" else kind
        assert len(calls) == sum(
            t == "linear_attention" for t in _config(config_kind)[0].layer_types
        )
    total = loss.detach().clone()
    dist.all_reduce(total)
    torch.testing.assert_close(total, reference_loss.detach(), rtol = 1e-5, atol = 1e-6)
    for name, param in model.named_parameters():
        grad = torch.zeros_like(param) if param.grad is None else param.grad
        dist.all_reduce(grad)
        if name not in reference:
            # The vision tower gets no gradient from text-only batches, on either side.
            assert not grad.any(), name
            continue
        torch.testing.assert_close(grad, reference[name], rtol = 1e-4, atol = 1e-6, msg = name)


def _spawn(fn, world, *args):
    # Fork: a spawned child would re-import unsloth without the conftest CUDA spoof.
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("needs the fork start method")
    with tempfile.TemporaryDirectory() as d:
        mp.start_processes(
            _entry,
            args = (world, os.path.join(d, "init"), fn, args),
            nprocs = world,
            join = True,
            start_method = "fork",
        )


def _entry(rank, world, init_file, fn, args):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method = f"file://{init_file}", rank = rank, world_size = world)
    try:
        fn(rank, world, *args)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("kind", KINDS)
def test_ulysses_hybrid_gradients_match_one_rank(kind):
    _spawn(_parity_worker, 2, kind, False)


@pytest.mark.skipif(
    not hasattr(transformers, "Qwen3_5Config"), reason = "transformers without Qwen3.5"
)
def test_text_only_qwen3_5_checkpoint_with_a_vision_config_matches_one_rank():
    _spawn(_parity_worker, 2, "qwen3_5_vlm", False)


@pytest.mark.skipif(
    not hasattr(transformers, "Qwen3_5Config"), reason = "transformers without Qwen3.5"
)
def test_qwen3_5_checkpoint_with_a_vision_config_passes_the_gate():
    model = _vlm()
    assert cp._hybrid_config_blocker(model) is None
    assert cp._supports_context_parallel(model, "ulysses")
    cp._check_cp_model(model, 2, "ulysses")


def _ddp_text_only_worker(rank, world, freeze_vision):
    from torch.nn.parallel import DistributedDataParallel

    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    model = _vlm()
    _force_torch_kernels(model)
    if freeze_vision:
        model.model.visual.requires_grad_(False)
    manager = _manager(world, rank)
    manager.mesh = types.SimpleNamespace(
        get_group = lambda: dist.group.WORLD, get_local_rank = lambda: rank
    )
    manager.attach_attention_hooks(model)
    ddp = DistributedDataParallel(model, find_unused_parameters = False)
    for step in range(2):
        ids = torch.randint(0, 64, (2, 13), generator = torch.Generator().manual_seed(step))
        shift = F.pad(ids, (0, 1), value = -100)[:, 1:]
        inputs = {"input_ids": ids, "shift_labels": shift}
        manager._prepare_inputs(inputs)
        with manager.apply(inputs):
            out = ddp(
                input_ids = inputs["input_ids"], position_ids = inputs["position_ids"], use_cache = False
            )
            _loss(out.logits, inputs["shift_labels"], shift.ne(-100).sum()).backward()


@pytest.mark.skipif(
    not hasattr(transformers, "Qwen3_5Config"), reason = "transformers without Qwen3.5"
)
def test_ddp_text_only_fails_at_step_two_with_a_trainable_vision_tower():
    # The unused vision tower means DDP's reduction never finishes.
    with pytest.raises(Exception, match = "finished reduction in the prior iteration"):
        _spawn(_ddp_text_only_worker, 2, False)
    _spawn(_ddp_text_only_worker, 2, True)


def _cp_trainer(
    model,
    distributed_type,
    num_processes = 2,
    fsdp_version = None,
):
    plugin = None if fsdp_version is None else types.SimpleNamespace(fsdp_version = fsdp_version)
    accelerator = types.SimpleNamespace(
        distributed_type = types.SimpleNamespace(name = distributed_type),
        num_processes = num_processes,
        state = types.SimpleNamespace(fsdp_plugin = plugin, parallelism_config = None),
        dispatch_batches = None,
    )
    return types.SimpleNamespace(accelerator = accelerator, model = model)


@pytest.mark.skipif(
    not hasattr(transformers, "Qwen3_5Config"), reason = "transformers without Qwen3.5"
)
def test_ddp_refuses_a_trainable_vision_tower_but_not_fsdp2_frozen_or_one_process():
    pytest.importorskip("accelerate", minversion = "1.10.0")
    model = _vlm()
    trainer = _cp_trainer(model, "MULTI_CPU")
    with pytest.raises(NotImplementedError, match = "finetune_vision_layers = False"):
        cp._check_cp_trainer(trainer, 2, "ulysses", [model], allow_fsdp2 = True)
    # FSDP2 reduces unused parameters as zeros; no DDP reducer runs without several processes.
    fsdp2 = _cp_trainer(model, "FSDP", fsdp_version = 2)
    assert cp._check_cp_trainer(fsdp2, 2, "ulysses", [model], allow_fsdp2 = True)
    for single in (_cp_trainer(model, "NO"), _cp_trainer(model, "MULTI_CPU", num_processes = 1)):
        cp._check_cp_trainer(single, 2, "ulysses", [model], allow_fsdp2 = True)
    model.model.visual.requires_grad_(False)
    cp._check_cp_trainer(trainer, 2, "ulysses", [model], allow_fsdp2 = True)


@pytest.mark.parametrize(
    "key", ["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"]
)
def test_batches_with_image_or_video_inputs_are_refused(key):
    inputs = {"input_ids": torch.ones(1, 6, dtype = torch.long), key: torch.ones(1, 3)}
    with pytest.raises(ValueError, match = key):
        _manager(2, 0)._prepare_inputs(inputs)
    with pytest.raises(ValueError, match = "image / video"):
        with _manager(2, 0).apply(dict(inputs)):
            pass
    # A collator's None placeholder is a text-only batch.
    inputs[key] = None
    _manager(2, 0)._prepare_inputs(inputs)


def test_shard_local_gated_delta_is_wrong():
    # Negative control: the stock forward on a sequence shard restarts the scan at the shard seam.
    with pytest.raises(Exception, match = "AssertionError"):
        _spawn(_parity_worker, 2, KINDS[0], True)


def test_hybrid_gates():
    model = _model(KINDS[0])
    assert "ulysses" in cp._unsupported_attention(model.config, "ring")
    assert cp._unsupported_attention(model.config, "ulysses") is None
    assert cp._supports_context_parallel(model, "ulysses")
    assert not cp._supports_context_parallel(model, "ring")
    with pytest.raises(NotImplementedError, match = "context_parallel_backend = 'ulysses'"):
        cp._check_stateful_mixers(model, "ring", 2)
    cp._check_stateful_mixers(model, "ulysses", 2)
    with pytest.raises(ValueError, match = "num_k_heads"):
        cp._check_stateful_mixers(model, "ulysses", 4)

    class FakeMamba2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.A_log = torch.nn.Parameter(torch.zeros(2))
            self.dt_bias = torch.nn.Parameter(torch.zeros(2))
            self.conv1d = torch.nn.Conv1d(2, 2, 4, groups = 2)

    model.model.layers[0].mixer = FakeMamba2()
    with pytest.raises(NotImplementedError, match = "Gated DeltaNet linear attention only"):
        cp._check_stateful_mixers(model, "ulysses", 2)


def test_eager_attention_is_refused():
    model = _model(KINDS[0])
    attn = model.model.layers[1].self_attn
    with pytest.raises(NotImplementedError, match = "attn_implementation"):
        uh.install_hf_attention_wrapper(attn, "eager", cp.get_cp_manager)


def test_hybrid_packing_metadata_is_off_under_context_parallelism():
    from unsloth.utils import packing

    kwargs = {
        "input_ids": torch.ones(1, 8, dtype = torch.long),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]]),
        "packed_seq_lengths": torch.tensor([3, 5], dtype = torch.int32),
    }
    assert packing._hybrid_varlen_metadata(kwargs) is not None
    manager = _manager(2, 0)
    with manager.apply({"input_ids": torch.ones(1, 8, dtype = torch.long)}):
        # The shard sees 4 tokens: local boundaries would split documents at the wrong places.
        assert packing._hybrid_varlen_metadata(kwargs) is None


def test_ulysses_refuses_position_only_packing():
    inputs = {
        "input_ids": torch.ones(1, 6, dtype = torch.long),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2]]),
    }
    with pytest.raises(ValueError, match = "packed_seq_lengths"):
        _manager(2, 0)._prepare_inputs(inputs)
    inputs["packed_seq_lengths"] = torch.tensor([3, 3], dtype = torch.int32)
    _manager(2, 0)._prepare_inputs(inputs)


def test_ulysses_accepts_right_padding_with_mask_derived_positions():
    inputs = {
        "input_ids": torch.ones(2, 6, dtype = torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]]),
        "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 0, 0]]),
    }
    _manager(2, 0)._prepare_inputs(inputs)
    # A restart between real tokens is still packing.
    inputs["position_ids"] = torch.tensor([[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 0, 0]])
    with pytest.raises(ValueError, match = "packed_seq_lengths"):
        _manager(2, 0)._prepare_inputs(inputs)


def test_hybrid_refuses_router_aux_loss_but_not_a_vision_config():
    model = _model(KINDS[0])
    assert cp._supports_context_parallel(model, "ulysses")
    model.config.output_router_logits = True
    assert "output_router_logits" in cp._hybrid_config_blocker(model)
    assert not cp._supports_context_parallel(model, "ulysses")
    model.config.output_router_logits = False
    model.config.vision_config = {"model_type": "qwen3_5_vision"}
    assert cp._hybrid_config_blocker(model) is None
    assert cp._supports_context_parallel(model, "ulysses")
    model.config.vision_config = None
    # transformers 4.x Qwen3-Next: per-expert modules, each run only when routed on that rank.
    model.mlp_holder = torch.nn.Module()
    model.mlp_holder.experts = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
    assert "sparse MoE" in cp._hybrid_config_blocker(model)
    assert not cp._supports_context_parallel(model, "ulysses")


def test_ulysses_refuses_per_batch_router_logits_and_3d_positions():
    inputs = {"input_ids": torch.ones(1, 6, dtype = torch.long), "output_router_logits": True}
    with pytest.raises(ValueError, match = "output_router_logits"):
        _manager(2, 0)._prepare_inputs(inputs)
    inputs = {
        "input_ids": torch.ones(1, 6, dtype = torch.long),
        "position_ids": torch.arange(6).expand(4, 1, 6).contiguous(),
    }
    with pytest.raises(ValueError, match = "2D"):
        _manager(2, 0)._prepare_inputs(inputs)


def test_linear_attn_gets_packed_lengths_the_decoder_did_not_forward(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        uh, "ulysses_gated_delta_forward", lambda module, *a, group, **kw: seen.update(kw)
    )
    module = torch.nn.Identity()
    manager = types.SimpleNamespace(
        backend = "ulysses", group = None, packed_seq_lengths = torch.tensor([3, 5])
    )
    uh.wrap_gated_delta_forward(module, lambda: manager)
    module(torch.zeros(1, 4, 2))
    assert torch.equal(seen["packed_seq_lengths"], torch.tensor([3, 5]))
    # Lengths the decoder did forward win.
    module(torch.zeros(1, 4, 2), packed_seq_lengths = torch.tensor([8]))
    assert torch.equal(seen["packed_seq_lengths"], torch.tensor([8]))
