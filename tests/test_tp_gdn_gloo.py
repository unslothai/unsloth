# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Gated DeltaNet (linear attention) layers sharded by heads under tensor parallelism, gloo CPU, 2 ranks, fp32,
vs one process: Qwen3.5, Qwen3.5-MoE, Qwen3.8 (Qwen4Exp) and Qwen3-Next (in_proj_qkvz / in_proj_ba grouped per
key head), 16-bit and bitsandbytes 4-bit, LoRA on every projection. A wrong head split must break parity.

The configs are tiny shrinks of each model type's config, built offline (GQA grouping kept: 2 key heads, 4 value
heads)."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import importlib.machinery  # noqa: E402
import importlib.util  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    """unsloth/models/<name>.py without importing the unsloth package (which needs a GPU)."""
    full = f"unsloth.models.{name}"
    if full in sys.modules:
        return sys.modules[full]
    pkg = types.ModuleType("unsloth")
    pkg.__path__ = [os.path.join(REPO, "unsloth")]
    models = types.ModuleType("unsloth.models")
    models.__path__ = [os.path.join(REPO, "unsloth", "models")]
    for package in (pkg, models):
        package.__spec__ = importlib.machinery.ModuleSpec(package.__name__, None, is_package = True)
        package.__spec__.submodule_search_locations = package.__path__
    sys.modules.setdefault("unsloth", pkg)
    sys.modules.setdefault("unsloth.models", models)
    spec = importlib.util.spec_from_file_location(
        full, os.path.join(REPO, "unsloth", "models", f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


tpn = _load("tp_native")

VOCAB = 512
_BASE = dict(
    vocab_size = VOCAB,
    hidden_size = 64,
    intermediate_size = 128,
    num_attention_heads = 4,
    num_key_value_heads = 2,
    head_dim = 16,
    linear_conv_kernel_dim = 4,
    linear_num_key_heads = 2,
    linear_num_value_heads = 4,  # two value heads per key head (Qwen3.5 / 3.8 GQA-style grouping)
    linear_key_head_dim = 16,
    linear_value_head_dim = 16,
    tie_word_embeddings = False,
)
_MOE = dict(
    num_experts = 4,
    num_experts_per_tok = 2,
    moe_intermediate_size = 32,
    shared_expert_intermediate_size = 32,
)
_LAYERS = ["linear_attention", "full_attention", "linear_attention"]
# arch -> (model_type, config kwargs)
CONFIGS = {
    "qwen3_5": ("qwen3_5_text", {**_BASE, "num_hidden_layers": 3, "layer_types": _LAYERS}),
    "qwen3_5_moe": (
        "qwen3_5_moe_text",
        {**_BASE, **_MOE, "num_hidden_layers": 3, "layer_types": _LAYERS},
    ),
    # unsloth/Qwen3.8-Flash-Next's text config shrunk: hyper-connection streams, PLE n-gram table, QSA indexer.
    "qwen3_8": (
        "qwen4_exp_text",
        {
            **_BASE,
            **_MOE,
            "bos_token_id": VOCAB - 1,
            "eos_token_id": VOCAB - 1,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "full_attention_interval": 4,
            "hc_count": 4,
            "hc_lowrank": 8,
            "indexer_budget": 64,
            "indexer_compress_ratio": 4,
            "indexer_head_dim": 16,
            "indexer_kv_heads": 1,
            "indexer_n_heads": 2,
            "ple_layer_ids": [2],
            "ple_embed_dim": 64,
            "heads_per_ngram": 8,
            "ngram_size": 3,
            "ngram_vocab_size_base": 509,
            "split_ngram_parts": 4,
            "make_ngram_vocab_size_divisible_by": 8,
            "ple_conv_kernel_size": 4,
            "mtp_num_hidden_layers": 0,
            "max_position_embeddings": 512,
        },
    ),
    "qwen3_next": (
        "qwen3_next",
        {**_BASE, **_MOE, "num_hidden_layers": 3, "layer_types": _LAYERS, "decoder_sparse_step": 1},
    ),
}
# 4-bit: bitsandbytes blocks of 64 along in_features, so out_proj needs value_dim / tp = 64.
FOUR_BIT = {"linear_value_head_dim": 32}


def _available(arch):
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    return CONFIGS[arch][0] in CONFIG_MAPPING


def _bnb():
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        return False
    return True


def _quantize_linear_attention(model):
    """Every Linear inside a Gated DeltaNet -> a bitsandbytes NF4 Linear4bit (same bytes in every process)."""
    import bitsandbytes as bnb

    for name, module in list(model.named_modules()):
        if not name.endswith("linear_attn"):
            continue
        for child_name, child in list(module.named_children()):
            if type(child) is not torch.nn.Linear:
                continue
            q = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias = False,
                compute_dtype = torch.float32,
                quant_type = "nf4",
            )
            packed, state = bnb.functional.quantize_4bit(
                child.weight.data, blocksize = 64, quant_type = "nf4"
            )
            q.weight = bnb.nn.Params4bit(
                packed,
                requires_grad = False,
                quant_state = state,
                blocksize = 64,
                quant_type = "nf4",
                module = q,
                bnb_quantized = True,
            )
            setattr(module, child_name, q)
    return model


def build(
    arch,
    seed = 0,
    four_bit = False,
    overrides = None,
    lora = True,
):
    from transformers import AutoModelForCausalLM
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    model_type, kwargs = CONFIGS[arch]
    kwargs = {**kwargs, **(FOUR_BIT if four_bit else {}), **(overrides or {})}
    config = CONFIG_MAPPING[model_type](**kwargs)
    config._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config, dtype = torch.float32)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and p.ndim == 1:
                p.add_(
                    torch.randn_like(p) * 0.1
                )  # per-head gated norm weights: their placement matters
            elif name.endswith(("A_log", "dt_bias")):
                p.add_(
                    torch.randn_like(p) * 0.5
                )  # per value head: a wrong head split changes the decay
            elif name.endswith("conv1d.weight"):
                p.normal_(0, 0.5)
            elif name.endswith("gate.weight") and p.ndim == 2:
                p.normal_(0, 0.5)  # routers start at zero: spread the routing
    if four_bit:
        _quantize_linear_attention(model)
    if not lora:
        return model.train()
    from peft import LoraConfig, get_peft_model

    names = {n.rsplit(".", 1)[-1] for n, m in model.named_modules() if "Linear" in type(m).__name__}
    names.discard("lm_head")
    # Every linear but the QSA indexer (integer top-k, no gradient) and MoE routers (replicated either way).
    regex = r"^(?!.*\.indexer\.).*\.(" + "|".join(sorted(names - {"gate"})) + r")$"
    torch.manual_seed(seed + 1)
    config = LoraConfig(r = 4, lora_alpha = 8, target_modules = regex, init_lora_weights = False)
    return get_peft_model(model, config).train()


def batch(bsz = 2, seq = 12):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, VOCAB - 1, (bsz, seq), generator = g)
    labels = ids.clone()
    labels[0, :3] = -100
    return ids, labels


def _train(
    model,
    ids,
    labels,
    full = None,
    steps = 2,
):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr = 1e-2)
    losses, grads = [], None
    for step in range(steps):
        loss = model(input_ids = ids, labels = labels, use_cache = False).loss
        loss.backward()
        if step == 0:
            grads = {
                n: (full(p.grad, p) if full else p.grad).detach().clone().double()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(loss))
    return {"losses": losses, "grads": grads}


def _linear_attention_bytes(model):
    """(this rank's bytes, whole bytes) of the Gated DeltaNet base parameters (4-bit: packed codes). LoRA A of
    a column projection (B of a row projection) is replicated by design: left out."""
    local = whole = 0
    for name, p in model.named_parameters():
        if ".linear_attn." not in name or "lora_" in name:
            continue
        t = p.to_local() if tpn._is_dtensor(p) else p
        local += t.numel() * t.element_size()
        layout = getattr(p, "_unsloth_tp_layout", None)
        tp = layout[3] if layout is not None and layout[0] != tpn.KV_REPLICATED else 1
        whole += (
            (p.numel() * p.element_size())
            if tpn._is_dtensor(p)
            else t.numel() * t.element_size() * tp
        )
    return local, whole


def _sharded_projections(model):
    """{module path: local out / in features} of the Gated DeltaNet projections apply_tensor_parallel took."""
    out = {}
    for name, module in model.named_modules():
        style = module.__dict__.get("_unsloth_tp_style")
        if style is not None and ".linear_attn." in f".{name}.":
            out[name.rsplit(".", 1)[-1]] = style
    return out


_MODELING = {
    "qwen3_5": "qwen3_5",
    "qwen3_5_moe": "qwen3_5_moe",
    "qwen3_8": "qwen4_exp",
    "qwen3_next": "qwen3_next",
}


def _force_conv_module(arch):
    """Run the Gated DeltaNet convolution as self.conv1d(mixed_qkv), as transformers 4.57 - 5.16 do without
    causal_conv1d (later versions always call causal_conv1d_fn, whose torch fallback reads only conv1d.weight):
    the module's in / out channels and groups must then be the local ones. Returns the conv1d call counter."""
    import importlib

    from transformers.activations import ACT2FN

    pkg = _MODELING[arch]
    modeling = importlib.import_module(f"transformers.models.{pkg}.modeling_{pkg}")
    gdn = next(c for n, c in vars(modeling).items() if n.endswith("GatedDeltaNet"))
    current, calls = [None], [0]

    def conv_fn(
        x,
        weight = None,
        bias = None,
        activation = None,
        **kwargs,
    ):
        calls[0] += 1
        out = current[0].conv1d(x.to(current[0].conv1d.weight.dtype))[..., : x.shape[-1]]
        return (ACT2FN[activation](out) if activation else out).to(x.dtype)

    forward = gdn.forward

    def gdn_forward(self, *args, **kwargs):
        current[0] = self
        if "causal_conv1d_fn" in self.__dict__:  # transformers 4.57: None without causal_conv1d
            self.causal_conv1d_fn = None
        return forward(self, *args, **kwargs)

    gdn.forward = gdn_forward
    modeling.causal_conv1d_fn = conv_fn
    original = torch.nn.Conv1d.forward

    def conv_forward(self, x):
        calls[0] += 1
        return original(self, x)

    torch.nn.Conv1d.forward = conv_forward
    return calls


def _case_parity(
    rank,
    world,
    tmp,
    arch,
    four_bit = False,
    conv_module = False,
):
    from torch.distributed.device_mesh import init_device_mesh

    calls = _force_conv_module(arch) if conv_module else [0]
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))["tp"]
    ids, labels = batch()
    ref_model = build(arch, four_bit = four_bit)
    ref = _train(ref_model, ids, labels)
    model = build(arch, four_bit = four_bit)
    whole_conv = {
        n: p.detach().clone() for n, p in model.named_parameters() if n.endswith("conv1d.weight")
    }
    tpn.apply_tensor_parallel(model, mesh)
    plan = model._unsloth_tp_plan
    local, whole = _linear_attention_bytes(model)
    state = tpn.tp_full_state_dict(model)
    conv_equal = all(torch.equal(state[n], p) for n, p in whole_conv.items())
    tp = _train(model, ids, labels, full = tpn._full)
    # generate gathers a whole replica (tp_replica): the global head counts and conv1d channels come back.
    kw = dict(input_ids = ids[:1, :5], max_new_tokens = 4, do_sample = False, pad_token_id = 0)
    # Train mode (no dropout here): bitsandbytes repacks 4-bit weights for its CPU inference kernel in eval
    # mode, which needs out_features to be a multiple of its block (in_proj_b / in_proj_a have 4 rows).
    with torch.no_grad():
        generated = (ref_model.generate(**kw), model.generate(**kw))
    # Fault injection: rank r keeps contiguous rows of the fused [q, k, v] (Qwen3.5) / treats the per key head
    # groups of in_proj_qkvz as [q, k, v, z] blocks (Qwen3-Next): heads mismatched across q, k, v and conv1d.
    original = tpn._gdn_parts

    def wrong(module, layout):
        parts, conv = original(module, layout)
        k = module.num_k_heads * module.head_k_dim
        v = module.num_v_heads * module.head_v_dim
        if layout == "split":
            parts = {**parts, "in_proj_qkv": (2 * k + v,)}
        else:
            parts = {**parts, "in_proj_qkvz": (k, k, v, v)}
        return parts, conv

    tpn._gdn_parts = wrong
    try:
        bad = build(arch, four_bit = four_bit)
        tpn.apply_tensor_parallel(bad, mesh)
        fault = _train(bad, ids, labels, full = tpn._full, steps = 1)
    finally:
        tpn._gdn_parts = original
    return {
        "ref": ref,
        "tp": tp,
        "fault": fault,
        "regions": sorted({r.kind for r in plan.regions}),
        "linear_regions": sum(r.kind == "linear_attention" for r in plan.regions),
        "sharded": _sharded_projections(model),
        "bytes": (local, whole),
        "conv_equal": conv_equal,
        "generated": generated,
        "conv_calls": calls[0],
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

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 300)
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


def _check(results, arch, four_bit):
    expected = (
        {"in_proj_qkvz": "colwise", "in_proj_ba": "colwise", "out_proj": "rowwise"}
        if arch == "qwen3_next"
        else {
            "in_proj_qkv": "colwise",
            "in_proj_z": "colwise",
            "in_proj_b": "colwise",
            "in_proj_a": "colwise",
            "out_proj": "rowwise",
        }
    )
    for rank, res in enumerate(results):
        assert res["linear_regions"] >= 2, res["regions"]
        assert res["sharded"] == expected, res["sharded"]
        local, whole = res["bytes"]
        # Everything but the per-head gated norm weight (head_v_dim floats, replicated) is split in two.
        assert local < 0.55 * whole, (local, whole)
        # tp_full_state_dict puts the conv1d channels back in [q, k, v] order.
        assert res["conv_equal"]
        # generate on a whole replica: the global head counts and conv1d channels are back.
        assert torch.equal(*res["generated"]), res["generated"]
        ref, tp = res["ref"], res["tp"]
        assert tp["losses"] == pytest.approx(ref["losses"], abs = 1e-5, rel = 0), (
            rank,
            tp["losses"],
            ref["losses"],
        )
        assert set(tp["grads"]) == set(ref["grads"]), sorted(set(tp["grads"]) ^ set(ref["grads"]))[
            :6
        ]
        gdn = [n for n in ref["grads"] if ".linear_attn." in n and "lora_" in n]
        assert len(gdn) >= 2 * len(expected), gdn
        for name, want in ref["grads"].items():
            scale = float(want.abs().max())
            diff = float((tp["grads"][name] - want).abs().max())
            assert diff <= 1e-3 * scale + 1e-9, (rank, name, diff, scale)
        # A wrong head split runs (same local shapes) but no longer computes the model.
        assert abs(res["fault"]["losses"][0] - ref["losses"][0]) > 1e-3, (
            res["fault"]["losses"],
            ref["losses"],
        )


@pytest.mark.parametrize("arch", sorted(CONFIGS))
def test_gdn_heads_sharded_matches_one_process(arch):
    if not _available(arch):
        pytest.skip(f"transformers has no {CONFIGS[arch][0]}")
    _check(_spawn("parity", 2, arch = arch), arch, False)


@pytest.mark.skipif(not _bnb(), reason = "bitsandbytes not installed")
@pytest.mark.parametrize("arch", ["qwen3_5", "qwen3_next"])
def test_gdn_heads_sharded_4bit_matches_one_process(arch):
    if not _available(arch):
        pytest.skip(f"transformers has no {CONFIGS[arch][0]}")
    _check(_spawn("parity", 2, arch = arch, four_bit = True), arch, True)


@pytest.mark.parametrize("arch", ["qwen3_5", "qwen3_next"])
def test_gdn_heads_sharded_through_the_conv1d_module(arch):
    # causal_conv1d unavailable on transformers 4.57 - 5.16: the forward calls self.conv1d(mixed_qkv).
    if not _available(arch):
        pytest.skip(f"transformers has no {CONFIGS[arch][0]}")
    results = _spawn("parity", 2, arch = arch, conv_module = True)
    assert all(res["conv_calls"] > 0 for res in results)
    _check(results, arch, False)


@pytest.mark.parametrize("arch", sorted(CONFIGS))
def test_gdn_plan_shards_by_heads(arch):
    if not _available(arch):
        pytest.skip(f"transformers has no {CONFIGS[arch][0]}")
    model = build(arch, lora = False)
    plan = tpn.derive_tp_plan(model, 2)
    gdn = [r for r in plan.regions if r.kind == "linear_attention"]
    assert len(gdn) == sum(t == "linear_attention" for t in CONFIGS[arch][1]["layer_types"])
    region = gdn[0]
    assert (region.heads, region.kv_heads, region.head_dim) == (4, 2, 16)
    k, v = 2 * 16, 4 * 16
    parts = {p.name: (p.style, tuple(p.parts)) for p in region.projs}
    if arch == "qwen3_next":
        assert parts["in_proj_qkvz"] == ("colwise", (2 * k + 2 * v,))
        assert parts["in_proj_ba"] == ("colwise", (8,))
    else:
        assert parts["in_proj_qkv"] == ("colwise", (k, k, v))
        assert parts["in_proj_z"] == ("colwise", (v,))
        assert parts["in_proj_b"] == parts["in_proj_a"] == ("colwise", (4,))
    assert parts["out_proj"] == ("rowwise", (64,))
    assert [tuple(p) for _, _, p in region.child_params] == [(k, k, v)]  # conv1d channels
    assert region.head_params == ["A_log", "dt_bias"]
    assert [name for name, _ in region.norms] == ["norm"]


@pytest.mark.parametrize(
    "overrides, tp, match",
    [
        ({}, 4, "does not divide linear_num_key_heads = 2"),
        ({"linear_num_key_heads": 3, "linear_num_value_heads": 6}, 2, "linear_num_key_heads = 3"),
        (
            {"linear_num_key_heads": 4, "linear_num_value_heads": 6},
            2,
            "not a multiple of its 4 key heads",
        ),
    ],
)
def test_gdn_heads_not_dividing_tp_are_refused(overrides, tp, match):
    if not _available("qwen3_5"):
        pytest.skip("transformers has no qwen3_5_text")
    model = build("qwen3_5", lora = False, overrides = overrides)
    with pytest.raises(ValueError, match = match):
        tpn.derive_tp_plan(model, tp)
