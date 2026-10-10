# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Context parallel for Qwen3.5 on CPU (gloo): 2 ranks must match 1 process."""

import os
import socket
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("transformers.models.qwen3_5")


def _load_parallel():
    # Load unsloth/distributed on its own: spawned ranks cannot run unsloth's
    # accelerator check on a CPU-only runner, and the package does not need it.
    import importlib.util
    import pathlib
    import sys

    name = "_unsloth_distributed_under_test"
    if name not in sys.modules:
        root = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "distributed"
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations = [str(root)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{name}.parallel")


P = _load_parallel()


def _tiny_model():
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM

    config = Qwen3_5TextConfig(
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        layer_types = ["linear_attention", "full_attention"],
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        linear_num_key_heads = 2,
        linear_num_value_heads = 4,
        linear_key_head_dim = 16,
        linear_value_head_dim = 16,
    )
    torch.manual_seed(3407)
    model = Qwen3_5ForCausalLM(config).float()
    model.config.use_cache = False
    return model


def _batch(seq):
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 128, (2, seq), generator = g)
    labels = ids.clone()
    labels[:, :3] = -100
    return ids, labels


def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def _worker(rank, world, port, seq, out):
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, init_method = f"tcp://127.0.0.1:{port}"
    )
    try:
        model = _tiny_model()
        P.apply_cp(model, dist.group.WORLD)
        ids, labels = _batch(seq)
        loss = P.sft_loss(model, ids, labels, dist.group.WORLD)
        loss.backward()
        P.sync_replicated_grads(model, dist.group.WORLD)
        if rank == 0:
            torch.save({"loss": loss.detach(), "grads": _grads(model)}, out)
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("seq", [32, 33])
def test_cp_matches_single_process(seq):
    model = _tiny_model()
    ids, labels = _batch(seq)
    ref = P.sft_loss(model, ids, labels, None)
    ref.backward()
    ref_grads = _grads(model)

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "cp.pt")
        mp.spawn(_worker, args = (2, _free_port(), seq, out), nprocs = 2, join = True)
        got = torch.load(out)

    torch.testing.assert_close(got["loss"], ref.detach(), rtol = 1e-5, atol = 1e-5)
    assert got["grads"].keys() == ref_grads.keys()
    for name, grad in ref_grads.items():
        torch.testing.assert_close(got["grads"][name], grad, rtol = 1e-4, atol = 1e-6, msg = name)


def test_packed_and_padded_masks():
    seq = 8
    pos = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    keep = P._blocked_key_mask(None, pos, 1, seq, "cpu")
    assert not bool(keep[0, 0, 4, 0]) and not bool(keep[0, 0, 7, 3])
    assert bool(keep[0, 0, 5, 4]) and not bool(keep[0, 0, 5, 6])

    pad = torch.ones(1, seq, dtype = torch.long)
    pad[0, 6:] = 0
    keep_pad = P._blocked_key_mask(pad, None, 1, seq, "cpu")
    assert not bool(keep_pad[0, 0, 3, 6]) and bool(keep_pad[0, 0, 3, 1])

    assert P._blocked_key_mask(None, torch.arange(seq).view(1, -1), 1, seq, "cpu") is None

    torch.manual_seed(0)
    q = torch.randn(1, seq, 2, 8, requires_grad = True)
    k = torch.randn(1, seq, 1, 8, requires_grad = True)
    v = torch.randn(1, seq, 1, 8, requires_grad = True)
    out = P.sdpa_flash_fn(q, k, v, None, is_causal = True, position_ids = pos)
    out[:, 4:].sum().backward()
    assert v.grad[:, :4].abs().max().item() == 0
    assert v.grad[:, 4:].abs().max().item() > 0
