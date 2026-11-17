# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pipeline parallelism over a mixture-of-experts model whose router auxiliary loss it does not carry (gpt-oss,
Mixtral) trains when that loss is off (output_router_logits False, gpt-oss's default, or a zero coefficient):
a gloo CPU world of 2 stages vs gradient accumulation over the same micro-batches in one process. With the loss
on it stays refused, at the split, at Trainer init (TRL >= 1.x turns it on from the architecture's coefficient)
and at every stage forward."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_pp_moe_aux_gloo import REPO, _load, _reference, batch  # noqa: E402

transformers = pytest.importorskip("transformers")
ppn = _load("pp_native")
assert os.path.realpath(ppn.__file__).startswith(os.path.realpath(REPO))  # this checkout's tree

VOCAB = 128


def build(
    family = "gpt_oss",
    aux = False,
    coef = 0.9,
    lora = False,
    experts = False,
    gc = None,
    seed = 0,
):
    common = dict(
        vocab_size = VOCAB,
        hidden_size = 64,
        num_hidden_layers = 4,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        tie_word_embeddings = False,
        output_router_logits = aux,
        router_aux_loss_coef = coef,
        attention_dropout = 0.0,
    )
    if family == "gpt_oss":
        config = transformers.GptOssConfig(
            intermediate_size = 32,
            num_local_experts = 8,
            num_experts_per_tok = 2,
            sliding_window = 4,
            layer_types = ["sliding_attention", "full_attention"] * 2,
            **common,
        )
    elif family == "mixtral":
        config = transformers.MixtralConfig(
            intermediate_size = 96, num_local_experts = 8, num_experts_per_tok = 2, **common
        )
    else:
        raise ValueError(family)
    config._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = transformers.AutoModelForCausalLM.from_config(config, dtype = torch.float32)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and p.ndim == 1:
                p.add_(torch.randn_like(p) * 0.1)
            if name.endswith(("mlp.router.weight", "mlp.gate.weight")):
                p.normal_(0.0, 0.5)  # uneven routing: an auxiliary loss would be far from zero
    if gc is not None:
        # HF gradient checkpointing (reentrant or not): with the auxiliary loss off nothing is captured across it.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs = {"use_reentrant": gc == "reentrant"}
        )
    if not lora:
        return model.train()
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed + 1)
    config = LoraConfig(
        r = 4,
        lora_alpha = 8,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        target_parameters = (
            ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"] if experts else None
        ),
        init_lora_weights = False,
        lora_dropout = 0.0,
    )
    model = get_peft_model(model, config)
    if gc is not None:
        model.enable_input_require_grads()
    return model.train()


def _group(name):
    prefix = "lora_" if "lora_" in name else ""
    if any(k in name for k in ("mlp.router.", "mlp.gate.", "block_sparse_moe.gate.")):
        return prefix + "router"
    if "mlp.experts" in name or "block_sparse_moe" in name:
        return prefix + "experts"
    if "self_attn" in name:
        return prefix + "attention"
    return prefix + "other"


def _case_parity(
    rank,
    world,
    n_mb = 2,
    schedule = "gpipe",
    steps = 2,
    **build_kw,
):
    from torch.distributed.device_mesh import init_device_mesh

    ids, labels, attention_mask = batch()
    ref = _reference(build(**build_kw), ids, labels, attention_mask, n_mb, steps)
    model = build(**build_kw)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))["pp"]
    pp = ppn.apply_pipeline_parallel(model, mesh, n_microbatches = n_mb, schedule = schedule)
    opt = torch.optim.AdamW(pp.parameters(), lr = 1e-2, foreach = False)
    losses, diffs, missing = [], {}, []
    for step in range(steps):
        loss = pp.step(ids, labels, attention_mask = attention_mask)
        if step == 0:
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                want = ref["grads"].get(name)
                if p.grad is None:
                    if want is not None and float(want.abs().max()) > 0:
                        missing.append(name)
                    continue
                d = float((p.grad - want).abs().max())
                group = _group(name)
                prev = diffs.get(group, (0.0, 0.0))
                diffs[group] = (max(prev[0], d), max(prev[1], float(want.abs().max())))
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(loss))
    # Turned on after the split (what TRL >= 1.x does in SFTTrainer.__init__): every stage forward refuses
    # (checked on each rank's own stage context: a raising stage would leave its peer waiting in the schedule).
    refused = None
    guarded = pp.ctx.uncarried_moe is not None
    if build_kw.get("coef", 0.9):
        model.config.output_router_logits = True
        try:
            ppn._refuse_uncarried_router_aux(pp.ctx, {})
        except ValueError as exc:
            refused = str(exc)
    return {
        "ref": ref["losses"],
        "ref_aux": ref["aux"],
        "losses": losses,
        "diffs": diffs,
        "missing": missing,
        "refused": refused,
        "guarded": guarded,
        "stage": (pp.start, pp.end),
    }


def _case_refusals(rank, world):
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))["pp"]
    out = {}
    for key, kw in (
        ("gpt_oss_aux_on", dict(aux = True)),
        ("mixtral_aux_on", dict(family = "mixtral", aux = True)),
    ):
        try:
            ppn.apply_pipeline_parallel(build(**kw), mesh)
            out[key] = None
        except ValueError as exc:
            out[key] = str(exc)
    return out


CASES = {"parity": _case_parity, "refusals": _case_refusals}


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
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 180)
    )
    try:
        res = CASES[case](rank, world, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _spawn(
    case,
    world = 2,
    **kwargs,
):
    try:
        ppn._check_torch_pipelining()
    except RuntimeError as exc:
        pytest.skip(str(exc))
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
            p.join(600)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"rank exit codes {codes}"
        return [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]


CELLS = {
    "gpt_oss_full_logits_off_gpipe": dict(),
    "gpt_oss_lora_logits_off_1f1b": dict(lora = True, schedule = "1f1b"),
    "gpt_oss_full_coef0_logits_on_gpipe_mb4": dict(aux = True, coef = 0.0, n_mb = 4),
    "gpt_oss_lora_logits_off_gpipe_gc": dict(lora = True, gc = "nonreentrant"),
    "gpt_oss_lora_experts_logits_off_gpipe": dict(lora = True, experts = True),
    "gpt_oss_lora_experts_logits_off_1f1b_gc_reentrant": dict(
        lora = True, experts = True, schedule = "1f1b", gc = "reentrant"
    ),
    "mixtral_full_logits_off_gpipe": dict(family = "mixtral"),
}


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_pp2_uncarried_moe_with_aux_off_matches_gradient_accumulation(cell):
    kwargs = CELLS[cell]
    results = _spawn("parity", 2, **kwargs)
    for res in results:
        assert res["losses"] == pytest.approx(res["ref"], rel = 1e-5, abs = 1e-6), res
        if not kwargs.get("aux"):
            assert res["ref_aux"] == [0.0, 0.0], res["ref_aux"]  # else computed but weighted by 0
        assert not res["missing"], res["missing"]
        for group, (diff, scale) in res["diffs"].items():
            assert diff <= 1e-6 + 1e-4 * scale, (cell, res["stage"], group, diff, scale)
        assert res["guarded"], res
        if kwargs.get("coef", 0.9):
            assert "router_aux_loss_coef = 0.0" in (res["refused"] or ""), res["refused"]
    groups = set(results[0]["diffs"]) | set(results[1]["diffs"])
    want = {"router", "experts", "attention"}
    if kwargs.get("lora"):
        want = {"lora_attention", "lora_experts"} if kwargs.get("experts") else {"lora_attention"}
    assert want <= groups, groups


def test_pp2_uncarried_moe_with_aux_on_stays_refused():
    out = _spawn("refusals", 2)[0]
    for key in ("gpt_oss_aux_on", "mixtral_aux_on"):
        assert "mixture-of-experts" in (out[key] or ""), (key, out[key])
        assert "is not proven" in out[key], (key, out[key])


def _sft_trainer_class():
    return type("SFTTrainer", (), {})


@pytest.mark.parametrize(
    "trl_coef, logits, refused",
    [
        (
            None,
            False,
            True,
        ),  # TRL >= 1.x: None = the architecture's 0.9, set on inside SFTTrainer.__init__
        (0.0, False, False),
        (0.0, True, False),  # TRL turns it off again
        (0.5, False, True),
    ],
)
def test_trainer_init_refuses_trl_turning_the_aux_loss_on(trl_coef, logits, refused):
    pp_env = _load("pp_env")
    model = build(aux = logits)
    args = types.SimpleNamespace(router_aux_loss_coef = trl_coef)
    call = lambda: pp_env._refuse_uncarried_router_aux_at_init(_sft_trainer_class(), args, model)
    if refused:
        with pytest.raises(NotImplementedError, match = "router_aux_loss_coef = 0.0"):
            call()
    else:
        call()


@pytest.mark.parametrize("logits, refused", [(False, False), (True, True)])
def test_trainer_init_plain_trainer_follows_the_config(logits, refused):
    pp_env = _load("pp_env")
    model = build(aux = logits)
    call = lambda: pp_env._refuse_uncarried_router_aux_at_init(type("Trainer", (), {}), None, model)
    if refused:
        with pytest.raises(NotImplementedError, match = "output_router_logits"):
            call()
    else:
        call()


def test_split_time_rule():
    # Flip: before, every mixture-of-experts family outside the carried set was refused, aux loss on or off.
    for kw, allowed in (
        (dict(), True),
        (dict(aux = True), False),
        (dict(aux = True, coef = 0.0), True),
        (dict(family = "mixtral"), True),
        (dict(family = "mixtral", aux = True), False),
    ):
        model = build(**kw)
        parts = ppn._find_parts(model)
        try:
            ppn._refuse_unsupported(model, parts)
            ok = True
        except ValueError as exc:
            assert "is not proven" in str(exc), str(exc)
            ok = False
        assert ok is allowed, (kw, ok)
        assert (ppn._uncarried_moe_owner(model) is not None) is True


@pytest.mark.parametrize(
    "kw, trainer, trl_coef, aux_on",
    [
        (
            dict(aux = True, coef = 0.0),
            "Trainer",
            None,
            False,
        ),  # logits on, zero coefficient: adds nothing
        (dict(aux = True), "Trainer", None, True),
        (dict(aux = True), "SFTTrainer", 0.0, False),  # TRL turns it off again
        (dict(aux = True, coef = 0.0), "SFTTrainer", None, False),
        (dict(), "SFTTrainer", None, True),  # TRL >= 1.x enables the architecture's coefficient
        (dict(), "Trainer", None, False),
    ],
)
def test_capability_query_classifies_the_effective_coefficient(kw, trainer, trl_coef, aux_on):
    from unsloth import parallel_capabilities as pc

    model = build(**kw)
    args = types.SimpleNamespace(router_aux_loss_coef = trl_coef) if trainer == "SFTTrainer" else None
    query = pc.query_from(trainer_cls = type(trainer, (), {}), model = model, args = args)
    assert query.router_aux_loss is aux_on, (kw, trainer, trl_coef, query.router_aux_loss)
