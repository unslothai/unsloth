# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tensor parallel serves a large frozen, untied input embedding from host memory instead of replicating
it on every rank (Qwen3.8-27B 4-bit on 2x T4: 2.37 GiB per rank). Gloo CPU world 2 vs one process."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only

import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_oracle import tpn  # noqa: E402

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

VOCAB = 512
HIDDEN = 32


def _model(tied = False, seed = 0):
    from peft import LoraConfig, get_peft_model

    config = transformers.Qwen3Config(
        vocab_size = VOCAB,
        hidden_size = HIDDEN,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        tie_word_embeddings = tied,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = transformers.Qwen3ForCausalLM(config).float()
    torch.manual_seed(seed + 1)
    lora = LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False)
    return get_peft_model(model, lora).eval()


def _train(
    model,
    ids,
    steps = 2,
):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr = 0.5)
    losses, grads = [], None
    for step in range(steps):
        out = model(input_ids = ids, labels = ids, use_cache = False)
        out.loss.backward()
        if step == 0:
            grads = {
                n: tpn._full(p.grad, p).detach().clone()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(out.loss))
    return losses, grads


def _case_parity(
    rank,
    world,
    fault = False,
):
    from torch.distributed.device_mesh import init_device_mesh

    os.environ["UNSLOTH_TP_EMBED_ON_CPU"] = "1"
    ref, model = _model(), _model()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    tpn.apply_tensor_parallel(model, mesh["tp"])
    embed = model.get_input_embeddings()
    served = bool(getattr(embed, "_unsloth_fsdp2_cpu_lookup", False))
    q = tpn._unwrap(model).model.layers[0].self_attn.q_proj.base_layer.weight
    if fault and rank == 1:
        with torch.no_grad():
            embed.weight[:, 0] += 0.05  # one rank looks up different rows
    ids = torch.randint(0, VOCAB, (2, 12), generator = torch.Generator().manual_seed(7))
    return dict(
        served = served,
        local_q = tuple(tpn._local(q).shape),
        full_q = tuple(q.shape),
        ref = _train(ref, ids),
        tp = _train(model, ids),
    )


def _case_world_wider_than_tp(rank, world):
    from torch.distributed.device_mesh import init_device_mesh

    os.environ["UNSLOTH_TP_EMBED_ON_CPU"] = "1"
    model = _model()
    mesh = init_device_mesh("cpu", (world, 1), mesh_dim_names = ("dp", "tp"))
    return dict(pick = tpn._host_input_embedding(model, mesh["tp"], 0.15) is not None)


CASES = {"parity": _case_parity, "dp_x_tp": _case_world_wider_than_tp}


def _worker(rank, world, port, case, tmp, kwargs):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    finished = False
    try:
        res = CASES[case](rank, world, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
        finished = True
    finally:
        dist.destroy_process_group()
    if finished:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _spawn(
    case,
    world = 2,
    **kwargs,
):
    from tests._rendezvous_port import free_port

    ctx = multiprocessing.get_context("spawn")
    port = free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, case, tmp, kwargs))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(300)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"rank exit codes {codes}"
        return [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]


def _parity_errors(res):
    errors = []
    ref_losses, ref_grads = res["ref"]
    losses, grads = res["tp"]
    if losses != pytest.approx(ref_losses, rel = 1e-5, abs = 1e-5):
        errors.append(f"losses {losses} vs {ref_losses}")
    if set(grads) != set(ref_grads):
        errors.append(f"grad names differ: {sorted(set(grads) ^ set(ref_grads))[:4]}")
    for name in ref_grads.keys() & grads.keys():
        if not torch.allclose(grads[name], ref_grads[name], rtol = 1e-5, atol = 1e-5):
            errors.append(f"grad {name}")
    return errors


def test_candidate_rules(monkeypatch):
    """Frozen untied tables only; auto needs a CUDA weight holding the fraction; 0 turns it off."""
    model = _model()
    embed = model.get_input_embeddings()
    assert tpn._host_embedding_candidate(model, True, 0.15) is embed
    assert (
        tpn._host_embedding_candidate(model, None, 0.15) is None
    )  # auto: CPU weight, nothing to move
    assert tpn._host_embedding_candidate(_model(tied = True), True, 0.15) is None
    embed.weight.requires_grad_(True)
    assert tpn._host_embedding_candidate(model, True, 0.15) is None  # trainable (modules_to_save)
    embed.weight.requires_grad_(False)
    monkeypatch.setenv("UNSLOTH_TP_EMBED_ON_CPU", "0")
    assert tpn._host_input_embedding(model, None, 0.15) is None
    monkeypatch.setenv("UNSLOTH_TP_EMBED_ON_CPU", "auto")
    assert tpn._host_input_embedding(model, None, 0) is None  # TPConfig(host_input_embedding = 0)
    assert tpn.TPConfig().host_input_embedding == 0.15


def test_tp2_host_embedding_matches_one_process():
    results = _spawn("parity")
    for rank, res in enumerate(results):
        assert res["served"], rank
        assert res["local_q"][0] * 2 == res["full_q"][0]  # the decoder really is tensor parallel
        assert _parity_errors(res) == [], rank


def test_tp2_host_embedding_fault_breaks_parity():
    results = _spawn("parity", fault = True)
    assert all(res["served"] for res in results)
    assert _parity_errors(results[0]) != []  # rank 1's rows reach rank 0 through the tp all-reduces


def test_tp_narrower_than_world_keeps_the_embedding():
    assert [r["pick"] for r in _spawn("dp_x_tp")] == [False, False]


def _load_replica():
    import importlib
    return importlib.import_module("unsloth.models.tp_replica")


def test_replica_device_skips_the_host_served_embedding(monkeypatch):
    """embed_tokens is the first parameter: once served from the CPU, the replica gather and its NCCL
    collectives must still run on the device the model computes on (meta stands in for the GPU here)."""
    replica = _load_replica()
    model = _model()
    model.to("meta")
    embed = model.get_input_embeddings()
    assert next(model.parameters()) is embed.weight
    embed.to_empty(device = "cpu")
    tpn._serve_embedding_from_host(model, embed, 0)  # CPU table: the plain CPU lookup
    assert embed.weight.device.type == "cpu" and embed._unsloth_fsdp2_cpu_lookup
    assert replica._device(model).type == "meta"
    assert tpn._compute_device(model).type == "meta"
    nccl = type("Dist", (), {"get_backend": staticmethod(lambda group = None: "nccl")})
    monkeypatch.setattr(replica, "_dist", lambda: nccl)
    assert replica._comm_device(None, replica._device(model)).type == "meta"
