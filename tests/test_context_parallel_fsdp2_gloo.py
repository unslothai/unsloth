# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Context parallelism under FSDP2 on a (dp_shard = 2, cp = 2) mesh, four gloo ranks on CPU.

Checks the mesh (cp peers adjacent, FSDP2 over dp_shard x cp flattened), that accelerate's data loader
hands both cp peers the same batch, and that one patched training step (Ulysses split, token count
division, accelerate's real fsdp2_prepare_model) reproduces the gradients of an unsharded model on the
whole batch. The model is token-wise, so attention exchange is not exercised here (tests/test_ulysses.py).
"""

from __future__ import annotations

import os
import socket
import types

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")
pytest.importorskip("torch.distributed.fsdp")
pytest.importorskip("accelerate")

if not dist.is_available() or not dist.is_gloo_available():
    pytest.skip("gloo backend unavailable", allow_module_level = True)

WORLD = 4
CP = 2
VOCAB = 50
SEQ = 13


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Block(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.up = torch.nn.Linear(hidden, 2 * hidden)
        self.down = torch.nn.Linear(2 * hidden, hidden)

    def forward(self, x):
        return x + self.down(torch.nn.functional.silu(self.up(x)))


class TokenModel(torch.nn.Module):
    _no_split_modules = ["Block"]

    def __init__(self, hidden = 16):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, hidden)
        self.layers = torch.nn.ModuleList([Block(hidden) for _ in range(2)])
        self.head = torch.nn.Linear(hidden, VOCAB)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def _batches():
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(0, VOCAB, (WORLD, SEQ), generator = g)
    labels = ids.clone()
    labels[0, 2:9] = -100
    # Row 1: every target sits in the first half, so its second cp shard counts zero tokens.
    labels[1, SEQ // 2 :] = -100
    return ids, labels


def _worker(rank, port, out_dir):
    os.environ.update(
        RANK = str(rank),
        LOCAL_RANK = str(rank),
        WORLD_SIZE = str(WORLD),
        LOCAL_WORLD_SIZE = str(WORLD),
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        ACCELERATE_USE_CPU = "true",
    )
    import json

    dist.init_process_group("gloo", rank = rank, world_size = WORLD)
    try:
        import trl
        from torch.distributed.tensor import DTensor

        import accelerate.accelerator as accelerator_module
        from accelerate.data_loader import prepare_data_loader
        from accelerate.utils import FullyShardedDataParallelPlugin
        from unsloth import context_parallel as cp

        cp.DEVICE_TYPE_TORCH = "cpu"
        result = {}
        manager = cp.ContextParallelManager(CP, "ulysses", fsdp2 = True)
        mesh = manager.device_mesh
        result["names"] = list(mesh.mesh_dim_names)
        result["cp_group"] = manager.mesh.mesh.tolist()
        result["dp_shard_group"] = mesh["dp_shard"].mesh.tolist()
        result["fsdp_group"] = mesh["dp_shard_cp"].mesh.tolist()

        state = types.SimpleNamespace(device_mesh = mesh, parallelism_config = None)
        cp._install_fsdp2_parallelism_config(state, manager)
        pc = state.parallelism_config

        loader = prepare_data_loader(
            torch.utils.data.DataLoader(list(range(8)), batch_size = 1),
            num_processes = WORLD,
            process_index = rank,
            put_on_device = False,
            torch_device_mesh = mesh,
        )
        result["samples"] = [int(x) for batch in loader for x in batch]

        torch.manual_seed(0)
        model = TokenModel()
        reference = TokenModel()
        reference.load_state_dict(model.state_dict())
        plugin = FullyShardedDataParallelPlugin(
            fsdp_version = 2,
            auto_wrap_policy = "TRANSFORMER_BASED_WRAP",
            transformer_cls_names_to_wrap = ["Block"],
            reshard_after_forward = True,
        )
        accelerator = types.SimpleNamespace(
            state = types.SimpleNamespace(fsdp_plugin = plugin),
            torch_device_mesh = mesh,
            parallelism_config = pc,
            mixed_precision = "no",
            device = torch.device("cpu"),
            is_main_process = rank == 0,
        )
        model = accelerator_module.fsdp2_prepare_model(accelerator, model)
        result["param_mesh"] = next(model.parameters()).device_mesh.mesh.tolist()

        class Trainer:
            def training_step(self, model, inputs, num_items_in_batch = None):
                logits = model(inputs["input_ids"]).float()
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, VOCAB),
                    inputs["shift_labels"].reshape(-1),
                    ignore_index = -100,
                    reduction = "sum",
                )
                # What transformers' compute_loss does with num_items_in_batch under DDP / FSDP.
                loss = loss / num_items_in_batch * dist.get_world_size()
                loss.backward()
                return loss.detach()

        trl.SFTTrainer = Trainer
        cp.patch_sft_trainer()
        trainer = object.__new__(Trainer)
        trainer._context_parallel_manager = manager

        ids, labels = _batches()
        dp_index = rank // CP
        # transformers sums the pre-shard counts over every rank (cp peers count the same row twice).
        count = (labels[dp_index, 1:] != -100).sum().reshape(1)
        dist.all_reduce(count)
        inputs = {"input_ids": ids[dp_index : dp_index + 1], "labels": labels[dp_index : dp_index + 1]}
        trainer.training_step(model, inputs, count.clone())
        result["local_tokens"] = int(inputs["input_ids"].shape[1])

        rows = slice(0, WORLD // CP)
        logits = reference(ids[rows]).float()
        shift = torch.nn.functional.pad(labels[rows], (0, 1), value = -100)[:, 1:]
        ref_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB), shift.reshape(-1), ignore_index = -100
        )
        ref_loss.backward()
        ref = dict(reference.named_parameters())
        worst = 0.0
        for name, p in model.named_parameters():
            g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            r = ref[name].grad
            worst = max(worst, ((g - r).abs().max() / (r.abs().max() + 1e-12)).item())
        result["grad_rel_err"] = worst
        gathered = [None] * WORLD
        dist.all_gather_object(gathered, result)
        if rank == 0:
            with open(os.path.join(out_dir, "result.json"), "w") as f:
                json.dump(gathered, f)
    finally:
        dist.destroy_process_group()


def test_cp_composes_with_fsdp2_on_a_dp_shard_cp_mesh(tmp_path):
    import json

    import torch.multiprocessing as mp

    mp.spawn(_worker, args = (_free_port(), str(tmp_path)), nprocs = WORLD, join = True)
    with open(tmp_path / "result.json") as f:
        results = json.load(f)
    for rank, r in enumerate(results):
        peer = rank - rank % CP
        assert r["names"] == ["dp_shard", "cp"]
        assert r["cp_group"] == [peer, peer + 1]
        assert r["dp_shard_group"] == [rank % CP, rank % CP + CP]
        assert r["fsdp_group"] == list(range(WORLD))
        assert r["param_mesh"] == list(range(WORLD)), "FSDP2 must shard over dp_shard x cp"
        assert r["local_tokens"] == (SEQ + 1) // CP
        assert r["grad_rel_err"] < 1e-5, r["grad_rel_err"]
    # cp peers read the same rows, dp_shard ranks different ones.
    samples = [r["samples"] for r in results]
    assert samples[0] == samples[1] and samples[2] == samples[3]
    assert samples[0] != samples[2]
    assert sorted(samples[0] + samples[2]) == list(range(8))
