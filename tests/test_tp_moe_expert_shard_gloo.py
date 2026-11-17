# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tensor parallel on a gloo CPU world vs one process: mixture-of-experts routed experts sharded by index over tp
(UNSLOTH_TP_MOE_SHARD, default on), expert LoRA sharded with its experts, uneven expert counts padded; the kill
switch keeps them replicated; a dropped expert reduction must break parity."""

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
from test_tp_native_gloo import _free_port, _train  # noqa: E402
from test_tp_native_oracle import QWEN3_MOE, _batch, build, tpn  # noqa: E402

MOE_LORA = dict(target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"])
EXPERT_STACKS = ("gate_up_proj", "down_proj")


_FUSED = []


def _require_fused_experts():
    """Expert sharding needs fused expert stacks called as experts(hidden_states, top_k_index, top_k_weights)
    (transformers 5.x); transformers 4.x builds one MLP per expert and keeps the replicated path."""
    if not _FUSED:
        model = build(QWEN3_MOE, _overrides(4), **MOE_LORA)
        _FUSED.append(tpn.expert_shard_problem(tpn.derive_tp_plan(model, 2).moe_blocks[0]))
    if _FUSED[0] is not None:
        pytest.skip(f"experts stay replicated with this transformers: {_FUSED[0]}")


def _randomize_lora_b(model, seed = 1234):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.copy_(torch.randn(p.shape, generator = g) * 0.02)


def _overrides(experts, aux = False):
    out = {"mlp_only_layers": [1], "num_experts": experts}
    if aux:
        out.update(output_router_logits = True, router_aux_loss_coef = 0.01)
    return out


def _case_parity(
    rank,
    world,
    tmp,
    experts,
    aux = False,
    kill_switch = False,
    fault = None,
):
    from safetensors.torch import load_file
    from torch.distributed.device_mesh import init_device_mesh

    if kill_switch:
        os.environ["UNSLOTH_TP_MOE_SHARD"] = "0"
    else:
        os.environ.pop("UNSLOTH_TP_MOE_SHARD", None)
    if fault == "reduce":  # the partial expert outputs are never summed over tp
        tpn._experts_reduce = lambda y, group: y
    elif fault == "copy":  # the experts' input / routing-weight gradients are never summed over tp
        tpn._experts_copy = lambda x, group: x
    overrides = _overrides(experts, aux)
    ref = build(QWEN3_MOE, overrides, **MOE_LORA)
    model = build(QWEN3_MOE, overrides, **MOE_LORA)
    for m in (ref, model):
        _randomize_lora_b(
            m
        )  # lora_B = 0 would leave every lora_A grad of the first step at exactly 0
    zoo_before = {k for k in sys.modules if k.startswith("unsloth_zoo")}
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    tpn.apply_tensor_parallel(model, mesh["tp"])
    ref_dir0, tp_dir0 = os.path.join(tmp, f"ref0_{rank}"), os.path.join(tmp, "tp0")
    ref.save_pretrained(ref_dir0)
    tpn.save_tp_adapter(
        model, tp_dir0
    )  # before any step: the gather / unpermute / padding trim must be exact
    plan = model._unsloth_tp_plan
    hooked = [
        any(getattr(h, "__name__", "") == "_moe_post_hook" for h in b._forward_hooks.values())
        for b in plan.moe_blocks
    ]
    local_experts = {}
    for name, p in model.named_parameters():
        if "experts" in name and name.rsplit(".", 1)[-1] in EXPERT_STACKS:
            local_experts[name] = (
                tpn._is_dtensor(p),
                tuple(p.shape),
                tuple(p.to_local().shape) if tpn._is_dtensor(p) else tuple(p.shape),
            )
    ids = _batch(ref)
    ref_out = _train(ref, ids, 3, torch.nn.utils.clip_grad_norm_)
    tp_out = _train(model, ids, 3, tpn.tp_clip_grad_norm_)
    generated = model.generate(
        input_ids = torch.zeros(1, 2, dtype = torch.long), max_new_tokens = 2, pad_token_id = 0
    )
    ref_dir, tp_dir = os.path.join(tmp, f"ref{rank}"), os.path.join(tmp, "tp")
    ref.save_pretrained(ref_dir)
    tpn.save_tp_adapter(model, tp_dir)
    return {
        "zoo_imported": sorted(
            {k for k in sys.modules if k.startswith("unsloth_zoo")} - zoo_before
        ),
        "saved0_ref": load_file(os.path.join(ref_dir0, "adapter_model.safetensors")),
        "saved0_tp": load_file(os.path.join(tp_dir0, "adapter_model.safetensors"))
        if rank == 0
        else None,
        "ref": ref_out,
        "tp": tp_out,
        "moe_blocks": len(plan.moe_blocks),
        "sharded_moe": getattr(plan, "sharded_moe", 0),
        "mean_hooked": hooked,
        "local_experts": local_experts,
        "generated": tuple(generated.shape),
        "saved_ref": load_file(os.path.join(ref_dir, "adapter_model.safetensors")),
        "saved_tp": load_file(os.path.join(tp_dir, "adapter_model.safetensors"))
        if rank == 0
        else None,
    }


CASES = {"parity": _case_parity}


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

    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    try:
        res = CASES[case](rank, world, tmp, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _spawn(
    case,
    world = 2,
    **kwargs,
):
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, case, tmp, kwargs))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(600)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"rank exit codes {codes}"
        return [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]


def _diffs(res):
    """(max |logit diff|, max |grad diff|, max |loss diff|) of this rank against its one-process reference."""
    ref_logits, ref_grads, ref_losses, _ = res["ref"]
    logits, grads, losses, _ = res["tp"]
    assert set(grads) == set(ref_grads), sorted(set(grads) ^ set(ref_grads))[:6]
    grad = max((grads[n] - ref_grads[n]).abs().max().item() for n in ref_grads)
    loss = max(abs(a - b) for a, b in zip(losses, ref_losses))
    return (logits - ref_logits).abs().max().item(), grad, loss


def _check_parity(results):
    worst = (0.0, 0.0, 0.0)
    for rank, res in enumerate(results):
        ref_logits, ref_grads, ref_losses, ref_norms = res["ref"]
        logits, grads, losses, norms = res["tp"]
        # apply_tensor_parallel must not import unsloth_zoo: its import patches transformers process-wide
        # (empty training logits on Windows / ROCm), which a plain transformers model must never get.
        assert not res[
            "zoo_imported"
        ], f"apply_tensor_parallel / training imported {res['zoo_imported'][:5]}"
        assert logits.numel() and logits.shape == ref_logits.shape, (
            tuple(logits.shape),
            tuple(ref_logits.shape),
        )
        torch.testing.assert_close(logits, ref_logits, rtol = 1e-5, atol = 1e-5)
        assert set(grads) == set(ref_grads), sorted(set(grads) ^ set(ref_grads))[:6]
        assert any("experts" in n and "lora_" in n for n in ref_grads), sorted(ref_grads)[:8]
        for name in ref_grads:  # every grad, the expert LoRA factors included
            torch.testing.assert_close(grads[name], ref_grads[name], rtol = 1e-5, atol = 1e-5, msg = name)
        assert losses == pytest.approx(ref_losses, rel = 1e-5, abs = 1e-5), (rank, losses, ref_losses)
        assert norms[0] == pytest.approx(ref_norms[0], rel = 1e-5), (rank, norms, ref_norms)
        # Steps 2-3 norms follow AdamW (eps 1e-8), which turns fp32 reduction-order noise on near-zero grads into
        # 1e-4 .. 5e-4 relative depending on the CPU; the replicated path (UNSLOTH_TP_MOE_SHARD=0) shows the same
        # gap. A missed or doubled reduction is caught by the first-step grads / norm above (1e-5).
        assert norms == pytest.approx(ref_norms, rel = 2e-3), (rank, norms, ref_norms)
        assert res["generated"] == (1, 4)
        worst = tuple(max(a, b) for a, b in zip(worst, _diffs(res)))
    # Before any step the saved adapter is the same tensors: gather, lora_B unpermute and padding trim are exact.
    saved_tp, saved_ref = results[0]["saved0_tp"], results[0]["saved0_ref"]
    assert set(saved_tp) == set(saved_ref)
    for key in saved_ref:
        assert saved_tp[key].shape == saved_ref[key].shape, key  # padding experts dropped on save
        torch.testing.assert_close(saved_tp[key], saved_ref[key], rtol = 0, atol = 0, msg = key)
    # After 3 AdamW steps only the layout is compared: AdamW's g / sqrt(v) turns fp32 reduction-order noise on
    # near-zero grads into up to ~lr changes (seen at 2-3e-4 on gfx1151 / x86 CPUs, replicated path included).
    saved_tp, saved_ref = results[0]["saved_tp"], results[0]["saved_ref"]
    assert set(saved_tp) == set(saved_ref)
    for key in saved_ref:
        assert saved_tp[key].shape == saved_ref[key].shape, key
    print(f"max |diff| logits {worst[0]:.3g} grads {worst[1]:.3g} losses {worst[2]:.3g}")
    return worst


def _check_sharded(results, experts, world):
    # Every rank holds at least 2 local experts (a 1-expert stack reads as a plain 2-D LoRA): padding fills the rest.
    per_rank = max(2, -(-experts // world))
    for res in results:
        assert res["moe_blocks"] == 1
        assert res["sharded_moe"] == res["moe_blocks"], res["sharded_moe"]
        assert not any(res["mean_hooked"]), "a sharded block still averages its output over tp"
        assert len(res["local_experts"]) == 2, res["local_experts"]
        for name, (is_dt, global_shape, local_shape) in res["local_experts"].items():
            assert is_dt, name
            assert global_shape[0] == per_rank * world and local_shape[0] == per_rank, (
                name,
                global_shape,
                local_shape,
            )


@pytest.mark.parametrize(
    "world,experts",
    [(2, 4), (2, 8), (4, 4), (4, 8), (2, 5), (4, 6), (4, 3)],
    ids = [
        "tp2_e4",
        "tp2_e8",
        "tp4_e4_min2_padding_ranks",
        "tp4_e8",
        "tp2_e5_uneven",
        "tp4_e6_uneven",
        "tp4_e3_padding_rank",
    ],
)
def test_sharded_experts_match_one_process(world, experts):
    _require_fused_experts()
    results = _spawn("parity", world, experts = experts)
    _check_sharded(results, experts, world)
    _check_parity(results)


def test_router_aux_loss_matches_one_process():
    _require_fused_experts()
    results = _spawn("parity", 2, experts = 4, aux = True)
    _check_sharded(results, 4, 2)
    _check_parity(results)


def test_kill_switch_keeps_experts_replicated():
    results = _spawn("parity", 2, experts = 4, kill_switch = True)
    for res in results:
        assert res["sharded_moe"] == 0
        assert all(res["mean_hooked"])
        assert all(not is_dt for is_dt, _, _ in res["local_experts"].values()), res["local_experts"]
    _check_parity(results)


def test_unfused_experts_stay_replicated_and_match_one_process():
    """transformers 4.x per-expert modules: never sharded (whatever the switch says), output mean path, parity."""
    if not _FUSED:
        try:
            _require_fused_experts()
        except pytest.skip.Exception:
            pass
    if _FUSED[0] is None:
        pytest.skip("fused experts (transformers 5.x): covered by the sharded cases")
    results = _spawn("parity", 2, experts = 4)
    for res in results:
        assert res["sharded_moe"] == 0 and all(res["mean_hooked"])
    _check_parity(results)


def test_dropped_forward_reduction_breaks_parity():
    _require_fused_experts()
    results = _spawn("parity", 2, experts = 4, fault = "reduce")
    _check_sharded(results, 4, 2)
    logit, _, loss = max((_diffs(res) for res in results), key = lambda d: d[0])
    assert logit > 1e-3 and loss > 1e-3, (logit, loss)


def test_dropped_backward_reduction_breaks_grad_parity():
    _require_fused_experts()
    results = _spawn("parity", 2, experts = 4, fault = "copy")
    _check_sharded(results, 4, 2)
    diffs = [_diffs(res) for res in results]
    assert max(d[0] for d in diffs) < 1e-5, diffs  # the forward is untouched
    assert max(d[1] for d in diffs) > 1e-3, diffs


class _Experts(torch.nn.Module):
    """A fused experts module; `dense` reads routing weights as [tokens, experts] (indexed by expert id)."""

    def __init__(self, experts, hidden, dense):
        super().__init__()
        self.num_experts, self.dense = experts, dense
        self.weight = torch.nn.Parameter(torch.randn(experts, hidden, hidden))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        for k in range(top_k_index.shape[1]):
            e = top_k_index[:, k]
            w = top_k_weights.gather(1, e[:, None]) if self.dense else top_k_weights[:, k : k + 1]
            out = out + w * torch.einsum("th,thg->tg", hidden_states, self.weight[e])
        return out


@pytest.mark.parametrize("dense", [False, True], ids = ["per_slot_weights", "dense_weights"])
def test_local_dispatch_sums_to_the_whole_call(dense, monkeypatch):
    """_sharded_experts on every rank in turn (collectives as identity): partial outputs and input / routing-weight
    grads sum to the whole experts call, for [tokens, k] and dense [tokens, experts] routing weights."""
    import types

    monkeypatch.setattr(tpn, "_experts_copy", lambda t, group: t)
    monkeypatch.setattr(tpn, "_experts_reduce", lambda t, group: t)
    torch.manual_seed(0)
    E, H, T, K, tp = 5, 4, 9, 2, 2
    full = _Experts(E, H, dense)
    x0 = torch.randn(T, H)
    probs = torch.randn(T, E).softmax(-1)
    index = probs.topk(K, dim = -1).indices
    w0 = probs if dense else probs.gather(1, index)
    x, w = x0.clone().requires_grad_(True), w0.clone().requires_grad_(True)
    full(x, index, w).square().sum().backward()
    per_rank = max(2, -(-E // tp))
    y_sum, dx_sum, dw_sum = 0, 0, 0
    for rank in range(tp):
        lo = min(rank * per_rank, E)
        hi = min(lo + per_rank, E)
        local = _Experts(per_rank, H, dense)
        with torch.no_grad():
            local.weight.zero_()
            local.weight[: hi - lo] = full.weight[lo:hi]
        shard = types.SimpleNamespace(
            block = None,
            leaf = local,
            group = None,
            lo = lo,
            hi = hi,
            per_rank = per_rank,
            experts = E,
            swaps = [],
            frozen = {},
        )
        xr, wr = x0.clone().requires_grad_(True), w0.clone().requires_grad_(True)
        y = tpn._sharded_experts(shard, local.forward, xr, index, wr)
        y_sum = y_sum + y.detach()
        # The loss is quadratic in the summed output: feed every rank the whole output's gradient.
        y.backward(2 * full(x0, index, w0).detach())
        dx_sum, dw_sum = dx_sum + xr.grad, dw_sum + wr.grad
    torch.testing.assert_close(y_sum, full(x0, index, w0).detach(), rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(dx_sum, x.grad, rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(dw_sum, w.grad, rtol = 1e-5, atol = 1e-5)


def test_lora_b_layout_is_resolved_per_wrapper(monkeypatch):
    # A wrapper still on PEFT's forward stays rank-major even when zoo's process mode is grouped_by_expert.
    import types

    calls = []
    zoo = types.SimpleNamespace(
        moe_lora_b_layout = lambda: "grouped_by_expert",
        moe_lora_b_layout_for_wrapper = lambda w, a: calls.append((w, a)) or "rank_major",
    )
    monkeypatch.setitem(sys.modules, "unsloth_zoo.temporary_patches.moe_utils", zoo)
    wrapper = object()
    assert tpn._lora_b_layout(wrapper, "default") == "rank_major"
    assert calls == [(wrapper, "default")]
    assert tpn._lora_b_layout() == "grouped_by_expert"
