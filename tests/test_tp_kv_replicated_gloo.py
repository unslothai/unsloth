# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tensor parallelism with num_key_value_heads < tp (Gemma3 270M / 1B: one KV head) on a gloo CPU world vs one
process: the key / value projections are replicated, each rank computes the head its query heads read."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_env_gloo import _ENV_KEYS, _case_env, _close  # noqa: E402
from test_tp_native_gloo import (  # noqa: E402
    _batch,
    _case_parity,
    _case_sharded_load,
    _check_parity,
    _free_port,
    _quantize_layers,
    _save_checkpoint,
    _train,
    build,
    tpn,
)
from test_tp_native_oracle import LLAMA  # noqa: E402

KV1 = {"num_attention_heads": 4, "num_key_value_heads": 1}


def _case_layout(rank, world, tmp, arch, overrides):
    """The derived plan and the per-rank attention attributes (local heads, GQA groups, the KV head)."""
    from test_tp_native_oracle import ARCHS
    from torch.distributed.device_mesh import init_device_mesh

    repo, base = ARCHS[arch]
    model = build(repo, {**base, **overrides})
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    tpn.apply_tensor_parallel(model, mesh["tp"])
    region = model._unsloth_tp_plan.regions[0]
    attn = region.module
    styles = {p.name: p.style for p in region.projs}
    k = next(p.module for p in region.projs if p.name == "k_proj")
    return {
        "styles": styles,
        "kv_replicated": region.kv_replicated,
        "heads": (attn.num_heads, attn.num_key_value_heads, attn.num_key_value_groups),
        "rows": k.__dict__.get("_unsloth_tp_kv_rows"),
        "k_placements": [p.is_replicate() for p in tpn._weight(k)[0].placements],
        "saved_groups": attn.__dict__["_unsloth_tp_saved_attrs"].get("num_key_value_groups"),
    }


def _qlora(seed = 0):
    from peft import LoraConfig, get_peft_model

    sizes = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 32,
    }
    model = _quantize_layers(build(LLAMA, sizes, lora = False, seed = seed), False)
    torch.manual_seed(seed + 1)
    config = LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False)
    return get_peft_model(model, config).train()


def _case_qlora(rank, world, tmp):
    import bitsandbytes as bnb
    from torch.distributed.device_mesh import init_device_mesh

    ref, model = _qlora(), _qlora()
    k_full = model.base_model.model.model.layers[0].self_attn.k_proj.base_layer
    k_full = bnb.functional.dequantize_4bit(k_full.weight.data, k_full.weight.quant_state)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    tpn.apply_tensor_parallel(model, mesh["tp"])
    k = model.base_model.model.model.layers[0].self_attn.k_proj
    k_local = bnb.functional.dequantize_4bit(
        k.base_layer.weight.data, k.base_layer.weight.quant_state
    )
    # generate gathers a whole replica (the 4-bit k_proj regathered from the ranks' head rows).
    generated = model.generate(
        input_ids = torch.zeros(1, 2, dtype = torch.long), max_new_tokens = 2, pad_token_id = 0
    )
    ids = _batch(ref)
    return {
        "k_local_equal": torch.equal(k_local, k_full),  # one KV head: every rank holds all rows
        "generated": tuple(generated.shape),
        "ref": _train(ref, ids, 3, torch.nn.utils.clip_grad_norm_),
        "tp": _train(model, ids, 3, tpn.tp_clip_grad_norm_),
    }


CASES = {
    "parity": _case_parity,
    "layout": _case_layout,
    "qlora": _case_qlora,
    "sharded_load": _case_sharded_load,
    "env": _case_env,
}


def _worker(rank, world, port, case, tmp, env, kwargs):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    for key in _ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(env)
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(2)
    if case != "env":  # tp_env creates the process group from the environment
        dist.init_process_group(
            "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
        )
    try:
        res = CASES[case](rank, world, tmp, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn(
    case,
    world = 2,
    env = None,
    ckpt_overrides = None,
    **kwargs,
):
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        if ckpt_overrides is not None:
            kwargs["ckpt"] = _save_checkpoint("llama", os.path.join(tmp, "ckpt"), ckpt_overrides)
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, case, tmp, env or {}, kwargs))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(600)
        codes = [p.exitcode for p in procs]
        for p in procs:
            if p.is_alive():
                p.kill()
        assert codes == [0] * world, f"rank exit codes {codes}"
        results = [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]
    if "skip" in results[0]:
        pytest.skip(f"import unsloth failed on this host: {results[0]['skip']}")
    return results


def test_kv_heads_below_tp_are_replicated_per_rank():
    # tp 4 over 2 KV heads: ranks 0, 1 compute head 0 and ranks 2, 3 head 1.
    overrides = {"num_attention_heads": 8, "num_key_value_heads": 2, "intermediate_size": 64}
    results = _spawn("layout", 4, arch = "qwen3", overrides = overrides)
    for rank, res in enumerate(results):
        assert res["kv_replicated"]
        assert res["styles"]["k_proj"] == res["styles"]["v_proj"] == tpn.KV_REPLICATED
        assert res["styles"]["q_proj"] == tpn.COLWISE and res["styles"]["o_proj"] == tpn.ROWWISE
        assert res["heads"] == (2, 1, 2), res["heads"]  # 8 / 4 query heads over 1 local KV head
        head_dim = 128
        assert res["rows"] == ((rank // 2) * head_dim, head_dim), res["rows"]
        assert res["k_placements"] == [True]
        assert res["saved_groups"] == 4  # tp_replica restores the 1-GPU grouping


@pytest.mark.parametrize(
    "arch,world,overrides",
    [
        (
            "gemma3",
            2,
            KV1,
        ),  # one KV head, trainable q_norm / k_norm (k_norm on the replicated head)
        ("llama", 2, KV1),
        ("llama_bias", 2, KV1),  # k / v biases narrowed to the head
        ("llama", 4, KV1),  # 4 ranks reading one KV head
        ("qwen3", 4, {"num_attention_heads": 8, "num_key_value_heads": 2, "intermediate_size": 64}),
    ],
)
def test_kv_heads_below_tp_match_one_process(arch, world, overrides):
    _check_parity(_spawn("parity", world, arch = arch, overrides = overrides), world)


def test_kv_heads_below_tp_with_trainable_biases():
    results = _spawn("parity", 2, arch = "llama_bias", overrides = KV1, lora_kwargs = {"bias": "all"})
    _check_parity(results, 2)
    assert any("k_proj.base_layer.bias" in key for key in results[0]["saved_tp"])


def test_kv_heads_below_tp_sharded_loading_reads_kv_whole():
    with tempfile.TemporaryDirectory() as ckpt:
        _save_checkpoint("llama", ckpt, KV1)
        results = _spawn("sharded_load", 2, arch = "llama", ckpt = ckpt)
    _check_parity(results, 2)
    for res in results:
        assert res["diffs"] == [], res["diffs"][:6]
        assert res["tp"][2] == res["v1"][2] and res["tp"][3] == res["v1"][3]
        stats = res["stats"]
        assert stats["read_bytes_sliced"] * 2 == stats["sharded_numel"] * 4, stats
        assert stats["read_bytes_full"] == stats["replicated_numel"] * 4, stats


def _bnb_cpu():
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _bnb_cpu(), reason = "bitsandbytes not installed")
def test_kv_heads_below_tp_qlora_matches_one_process():
    for res in _spawn("qlora", 2):
        assert res["k_local_equal"] and res["generated"] == (1, 4)
        ref_logits, ref_grads, ref_losses, ref_norms = res["ref"]
        logits, grads, losses, norms = res["tp"]
        torch.testing.assert_close(logits, ref_logits, rtol = 1e-5, atol = 1e-5)
        assert set(grads) == set(ref_grads) and grads
        for name in ref_grads:
            torch.testing.assert_close(grads[name], ref_grads[name], rtol = 1e-4, atol = 1e-6, msg = name)
        assert losses == pytest.approx(ref_losses, rel = 1e-5, abs = 1e-6)
        assert norms == pytest.approx(ref_norms, rel = 1e-5)


def test_kv_heads_below_tp_env_trainer_matches_one_process():
    # UNSLOTH_TP_SIZE=2 with a one-KV-head checkpoint: sharded 16-bit load + Trainer, 3 steps vs one process.
    ref = _spawn("env", 1, ckpt_overrides = KV1, bs = 2)[0]
    assert ref["layout"] is None and not ref["is_tp"]
    results = _spawn("env", 2, {"UNSLOTH_TP_SIZE": "2"}, ckpt_overrides = KV1, bs = 2)
    for rank, res in enumerate(results):
        assert res["layout"] == "tp 2" and res["is_tp"] and res["sharded_load"], res
        _close(res["a"], ref["a"], what = rank)


def test_kv_heads_not_dividing_tp_are_refused():
    from test_tp_native_oracle import ARCHS

    repo, base = ARCHS["qwen3"]
    model = build(repo, {**base, "num_attention_heads": 12, "num_key_value_heads": 3})
    with pytest.raises(ValueError, match = "not a multiple of it"):
        tpn.derive_tp_plan(model, 4)
