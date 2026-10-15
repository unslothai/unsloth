# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FSDP2 wrap through accelerate's real `fsdp2_prepare_model`, two gloo ranks on CPU.

What Unsloth adds after accelerate shards the model (unsloth/models/fsdp2.py):
- a layer that reads `.weight` directly (the fused LoRA kernels) sees whole, unsharded weights
  inside a per-block unit's forward, so the fused path is kept there;
- a finer wrap (a LoRA submodule as its own unit) would hand that layer a DTensor shard, so the
  fused path is declined on every rank;
- lm_head read directly by the CausalLM forward (Unsloth's fused CE) is gathered first;
- the last decoder block stays gathered after forward;
- losses and gradients match an unsharded copy of the model.
"""

from __future__ import annotations

import os
import socket
import types

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")
pytest.importorskip("torch.distributed.fsdp")
pytest.importorskip("peft")
pytest.importorskip("accelerate")

if not dist.is_available() or not dist.is_gloo_available():
    pytest.skip("gloo backend unavailable", allow_module_level = True)

N_LAYERS = 3


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spy_mlp(self, X):
    """Stands in for a fused LoRA MLP: reads every weight directly, never calls the projections."""
    log = self._unsloth_spy_log

    def proj(p, x):
        base = p.base_layer.weight
        A = p.lora_A["default"].weight
        B = p.lora_B["default"].weight
        from torch.distributed.tensor import DTensor

        for w, full in ((base, (p.out_features, p.in_features)), (A, None), (B, None)):
            log.append((isinstance(w, DTensor), tuple(w.shape), full))
        return x @ base.t() + (x @ A.t()) @ B.t() * p.scaling["default"]

    gate = proj(self.gate_proj, X)
    up = proj(self.up_proj, X)
    return proj(self.down_proj, torch.nn.functional.silu(gate) * up)


def _spy_causal_forward(self, input_ids, labels):
    """Unsloth's CausalLM forward reads lm_head.weight for the fused CE instead of calling it."""
    hidden = self.model(input_ids = input_ids, use_cache = False).last_hidden_state
    logits = hidden @ self.lm_head.weight.t()
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(), labels[:, 1:].reshape(-1)
    )


def _build(seed):
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = N_LAYERS,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        vocab_size = 96,
        tie_word_embeddings = False,
    )
    torch.manual_seed(seed)
    base = LlamaForCausalLM(cfg)
    lora = LoraConfig(
        r = 4,
        lora_alpha = 8,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout = 0.0,
        init_lora_weights = False,
    )
    model = get_peft_model(base, lora)
    log = []
    for layer in model.base_model.model.model.layers:
        layer.mlp._unsloth_spy_log = log
        layer.mlp.forward = types.MethodType(_spy_mlp, layer.mlp)
    causal = model.base_model.model
    causal.forward = types.MethodType(_spy_causal_forward, causal)
    return model, log


def _fake_accelerator(rank, mesh, plugin):
    return types.SimpleNamespace(
        state = types.SimpleNamespace(fsdp_plugin = plugin),
        torch_device_mesh = mesh,
        parallelism_config = types.SimpleNamespace(fsdp_dim_names = ("dp_shard",)),
        mixed_precision = "no",
        device = torch.device("cpu"),
        is_main_process = rank == 0,
    )


def _worker(rank, port, case, out_dir):
    os.environ.update(
        RANK = str(rank),
        LOCAL_RANK = str(rank),
        WORLD_SIZE = "2",
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
    )
    for key in (
        "UNSLOTH_FSDP2",
        "FSDP_VERSION",
        "ACCELERATE_USE_FSDP",
        "UNSLOTH_FORCE_FUSED_LORA",
        "UNSLOTH_FSDP2_FUSED_LORA",
    ):
        os.environ.pop(key, None)
    if case == "opt_out":
        os.environ["UNSLOTH_FSDP2_FUSED_LORA"] = "0"
    import json

    dist.init_process_group("gloo", rank = rank, world_size = 2)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor

        import accelerate.accelerator as accelerator_module
        from accelerate.utils import FullyShardedDataParallelPlugin
        from unsloth.models import fsdp2, llama

        llama._FUSED_LORA_MLPS = llama._FUSED_LORA_MLPS + (_spy_mlp,)
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names = ("dp_shard",))
        wrap_names = ["LlamaDecoderLayer"] + (["Linear"] if case == "finer" else [])
        plugin = FullyShardedDataParallelPlugin(
            fsdp_version = 2,
            auto_wrap_policy = "TRANSFORMER_BASED_WRAP",
            transformer_cls_names_to_wrap = wrap_names,
            reshard_after_forward = True,
            activation_checkpointing = case == "checkpointed",
        )
        result = {}
        result["agree_mixed"] = fsdp2._all_ranks_agree(rank == 0, torch.device("cpu"))

        model, log = _build(seed = 0)
        reference, _ = _build(seed = 0)
        assert fsdp2.install_fsdp2_prepare_hook()
        accelerator = _fake_accelerator(rank, mesh, plugin)
        if case == "checkpointed":
            # What Accelerator._prepare_fsdp2 does before sharding when the plugin asks for it.
            from accelerate.utils.fsdp_utils import fsdp2_apply_ac

            model = fsdp2_apply_ac(accelerator, model)
        model = accelerator_module.fsdp2_prepare_model(accelerator, model)
        result["fused_kept"] = int(getattr(model, "_unsloth_fsdp2_fused_lora", -1))
        result["blocks"] = int(getattr(model, "_unsloth_fsdp2_blocks", -1))
        result["direct_read_units"] = int(getattr(model, "_unsloth_fsdp2_direct_read_units", -1))
        layers = model.base_model.model.model.layers
        result["spy_left"] = sum("forward" in layer.mlp.__dict__ for layer in layers)

        torch.manual_seed(1)
        ids = torch.randint(0, 96, (4, 12))
        local = ids[rank * 2 : rank * 2 + 2]
        loss = model(input_ids = local, labels = local)
        result["last_block_gathered"] = not isinstance(
            layers[-1].mlp.gate_proj.base_layer.weight, DTensor
        )
        result["first_block_gathered"] = not isinstance(
            layers[0].mlp.gate_proj.base_layer.weight, DTensor
        )
        loss.backward()
        result["spy_calls"] = len(log)
        result["spy_saw_shard"] = sum(1 for is_dt, _, _ in log if is_dt)
        result["spy_full_shapes"] = all(full is None or shape == full for _, shape, full in log)

        ref_loss = reference(input_ids = ids, labels = ids)
        ref_loss.backward()
        local_loss = torch.tensor([loss.item()])
        dist.all_reduce(local_loss)
        result["loss_diff"] = abs(local_loss.item() / 2 - ref_loss.item())
        ref_params = dict(reference.named_parameters())
        worst = 0.0
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            # FSDP averages over the 2 ranks; each rank's loss is the mean over its half.
            r = ref_params[name.replace("._checkpoint_wrapped_module", "")].grad
            worst = max(worst, (g - r).abs().max().item() / (r.abs().max().item() + 1e-12))
        result["grad_rel_err"] = worst
        if rank == 0:
            with open(os.path.join(out_dir, f"{case}.json"), "w") as f:
                json.dump(result, f)
    finally:
        dist.destroy_process_group()


def _run(case, tmp_path):
    import json

    import torch.multiprocessing as mp

    mp.spawn(_worker, args = (_free_port(), case, str(tmp_path)), nprocs = 2, join = True)
    with open(tmp_path / f"{case}.json") as f:
        return json.load(f)


def test_per_block_wrap_keeps_fused_layers_on_whole_weights(tmp_path):
    r = _run("per_block", tmp_path)
    assert r["agree_mixed"] is False, "ranks must agree on the strictest answer"
    assert r["blocks"] == N_LAYERS
    assert r["fused_kept"] == N_LAYERS
    assert r["spy_left"] == N_LAYERS
    # 3 MLPs x 3 projections x (base, A, B), forward only.
    assert r["spy_calls"] == N_LAYERS * 9
    assert r["spy_saw_shard"] == 0, "a direct .weight read saw a DTensor shard inside a block"
    assert r["spy_full_shapes"]
    import accelerate.utils.fsdp_utils as fsdp_utils

    if hasattr(fsdp_utils, "_find_final_norm"):
        # accelerate >= 1.14 gives [final norm, lm_head] their own unit, which nothing would gather.
        assert r["direct_read_units"] >= 1, "lm_head's unit is not gathered before the CausalLM forward"
    else:
        # Older accelerate leaves lm_head in the root unit, which its own pre-forward gathers.
        assert r["direct_read_units"] == 0
    assert r["last_block_gathered"] is True
    assert r["first_block_gathered"] is False
    assert r["loss_diff"] < 1e-5
    assert r["grad_rel_err"] < 1e-4


def test_a_finer_wrap_declines_the_fused_layers(tmp_path):
    r = _run("finer", tmp_path)
    assert r["fused_kept"] == 0
    assert r["spy_left"] == 0, "the fused forwards must be removed on every rank"
    assert r["spy_calls"] == 0
    assert r["loss_diff"] < 1e-5
    assert r["grad_rel_err"] < 1e-4


def test_the_fused_lora_opt_out_is_honoured_after_wrapping(tmp_path):
    """A raw Accelerator loop patches the model before any FSDP plugin exists."""
    r = _run("opt_out", tmp_path)
    assert r["fused_kept"] == 0
    assert r["spy_left"] == 0
    assert r["spy_calls"] == 0
    assert r["loss_diff"] < 1e-5


def test_activation_checkpointed_blocks_keep_the_last_block_gathered(tmp_path):
    r = _run("checkpointed", tmp_path)
    assert r["blocks"] == N_LAYERS
    assert r["fused_kept"] == N_LAYERS
    assert r["last_block_gathered"] is True
    assert r["first_block_gathered"] is False
    assert r["spy_saw_shard"] == 0
    assert r["loss_diff"] < 1e-5
    assert r["grad_rel_err"] < 1e-4
