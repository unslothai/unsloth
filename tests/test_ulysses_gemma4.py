# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses context parallelism for Gemma 4 text models on CPU gloo ranks.

Gemma 4 runs main's generic route (its own attention function after the exchange, with its sliding
window, scale 1.0, KV sharing, per-layer embeddings, attention_k_eq_v and MoE): fp64 against one
process, every gradient and one AdamW step. What Gemma 4 adds on top: its heterogeneous config
(transformers >= 5.15 refuses a global num_key_value_heads read; the global layers carry their own
key / value head count), so the head split is checked per layer before any collective.
"""

import copy
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
from unsloth.utils import ulysses_hybrid as uh

transformers = pytest.importorskip("transformers")
gemma4 = pytest.importorskip("transformers.models.gemma4")
LAYER_TYPES = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
SEQ = 13


def _config(
    kind,
    window = 4,
    **overrides,
):
    common = dict(
        vocab_size = 64,
        vocab_size_per_layer_input = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 4,
        num_attention_heads = 4,
        head_dim = 8,
        global_head_dim = 16,
        # A window of 4 inside 13 tokens: windows cross the shard seam.
        sliding_window = window,
        final_logit_softcapping = 30.0,
        layer_types = LAYER_TYPES,
    )
    if kind == "e2b":
        # E2B / E4B: per-layer embeddings, the last two layers reuse earlier KV, one KV head.
        common.update(
            hidden_size_per_layer_input = 8,
            num_kv_shared_layers = 2,
            num_key_value_heads = 1,
            use_double_wide_mlp = True,
        )
    elif kind == "moe":
        # 26B-A4B / 31B: global layers share K and V and have fewer KV heads; MoE; no PLE.
        common.update(
            hidden_size_per_layer_input = 0,
            num_kv_shared_layers = 0,
            num_key_value_heads = 2,
            num_global_key_value_heads = 1,
            attention_k_eq_v = True,
            enable_moe_block = True,
            num_experts = 4,
            top_k_experts = 2,
            moe_intermediate_size = 16,
        )
    elif kind == "hd512":
        # The release global head_dim (512, past flash-attn's 256) with heterogeneous KV heads.
        common.update(
            hidden_size_per_layer_input = 0,
            num_kv_shared_layers = 0,
            num_key_value_heads = 2,
            num_global_key_value_heads = 1,
            attention_k_eq_v = True,
            global_head_dim = 512,
        )
    else:
        raise ValueError(kind)
    return gemma4.Gemma4TextConfig(**{**common, **overrides})


def _multimodal_config(kind, window = 4):
    vision = gemma4.Gemma4VisionConfig(
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        head_dim = 8,
    )
    return gemma4.Gemma4Config(
        text_config = _config(kind.removeprefix("mm_"), window).to_dict(),
        vision_config = vision.to_dict(),
        audio_config = None,
    )


def _model(
    kind,
    *,
    window = 4,
    attn = "sdpa",
    dtype = torch.float64,
    checkpoint = False,
    lora = None,
):
    # Unsloth's grouped-GEMM MoE kernels need CUDA; the CPU parity check uses the torch experts.
    os.environ.setdefault("UNSLOTH_MOE_BACKEND", "native_torch")
    torch.manual_seed(0)
    if kind.startswith("mm_"):
        model = gemma4.Gemma4ForConditionalGeneration(_multimodal_config(kind, window))
    else:
        model = gemma4.Gemma4ForCausalLM(_config(kind, window))
    model = model.to(dtype).train()
    model.config._attn_implementation = attn
    model.config.use_cache = False
    for module in model.modules():
        # Grouped expert kernels may touch CUDA in a forked CPU worker.
        if hasattr(module, "set_experts_implementation"):
            module.set_experts_implementation("eager")
    if checkpoint:
        # HF gradient checkpointing: every decoder layer (and its CP attention) is recomputed in backward.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
    if lora:
        from peft import LoraConfig, get_peft_model

        extra = {}
        if lora == "experts":
            # LoRA on the routed experts too: the fused 3D expert weights.
            extra["target_parameters"] = ["experts.gate_up_proj", "experts.down_proj"]
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        model = get_peft_model(
            model, LoraConfig(r = 4, lora_alpha = 8, target_modules = targets, **extra)
        )
        g = torch.Generator().manual_seed(5)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "lora_B" in name:  # nonzero B: the adapter changes the forward
                    p.copy_(0.05 * torch.randn(p.shape, generator = g, dtype = p.dtype))
    return model


def _manager(
    size,
    group,
    local = None,
):
    manager = object.__new__(cp.ContextParallelManager)
    manager.size = size
    manager.backend = "ulysses"
    local = dist.get_rank(group) if local is None else local
    manager.mesh = types.SimpleNamespace(get_group = lambda: group, get_local_rank = lambda: local)
    return manager


def _loss(logits, shift_labels, count):
    return (
        F.cross_entropy(
            logits.to(torch.promote_types(logits.dtype, torch.float32)).flatten(0, 1),
            shift_labels.flatten(),
            ignore_index = -100,
            reduction = "sum",
        )
        / count
    )


def _batch(labels_scheme):
    ids = torch.randint(0, 64, (2, SEQ), generator = torch.Generator().manual_seed(1))
    mask = torch.ones_like(ids)
    mask[1, 6:] = 0  # row 1: 6 valid tokens, right padded
    labels = ids.masked_fill(mask == 0, -100)
    if labels_scheme == "prompt_masked":
        # Completion-only labels: the first shard(s) carry no labels at all.
        labels[:, :9] = -100
    return ids, mask, labels


def _grad_status(grad):
    return 0 if grad is None else (1 if not grad.any() else 2)


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


def _cp_groups(world, size):
    """Contiguous CP groups (rank // size = DP replica); every rank creates every group."""
    mine = None
    for start in range(0, world, size):
        ranks = list(range(start, start + size))
        group = dist.new_group(ranks)
        if dist.get_rank() in ranks:
            mine = group
    return mine


# ---------------------------------------------------------------------------------------------
# Model-level parity on main's generic route: fp64 against one process.


def _model_case(
    rank,
    world,
    case,
    control = None,
):
    size = case["cp"]
    dp = world // size
    group = _cp_groups(world, size)
    window = case.get("window", 4)
    model = _model(
        case["kind"],
        window = window,
        attn = case.get("attn", "sdpa"),
        checkpoint = case.get("checkpoint", False),
        lora = case.get("lora"),
    )
    reference = copy.deepcopy(model)
    ids, mask, labels = _batch(case.get("labels", "pad"))
    shift = F.pad(labels, (0, 1), value = -100)[:, 1:]
    count = shift.ne(-100).sum()
    _loss(
        reference(input_ids = ids, attention_mask = mask, use_cache = False).logits, shift, count
    ).backward()

    rows = list(range(rank // size, ids.shape[0], dp))
    manager = _manager(size, group)
    # Every rank refuses here, before any collective, if a layer kind's heads cannot be split.
    cp._check_cp_model(model, size, "ulysses")
    manager.attach_attention_hooks(model)
    attention = [m for n, m in model.named_modules() if n.endswith("self_attn")]
    if control == "shard_local":
        for module in attention:
            module._unsloth_cp_ulysses_attention = False
    elif control == "global_layers_windowed":
        for module in attention:
            module.sliding_window = window
    elif control == "dropped_window":
        for module in attention:
            module.sliding_window = None
    elif control == "default_scale":
        for module in attention:
            module.scaling = None
    calls = []
    real = uh._ulysses_hf_attention

    def counted(*args, **kwargs):
        # The head shard each layer kind sees: (query heads, key / value heads, head_dim).
        calls.append((args[1].shape[1], args[2].shape[1], args[1].shape[-1]))
        return real(*args, **kwargs)

    uh._ulysses_hf_attention = counted
    try:
        inputs = {
            "input_ids": ids[rows].clone(),
            "attention_mask": mask[rows].clone(),
            "labels": labels[rows].clone(),
        }
        with manager.apply(inputs):
            positions = inputs["position_ids"]
            if control == "position_offset":
                positions = positions - positions[:, :1]
            out = model(
                input_ids = inputs["input_ids"],
                attention_mask = inputs["attention_mask"],
                position_ids = positions,
                use_cache = False,
            )
            _loss(out.logits, inputs["shift_labels"], count).backward()
    finally:
        uh._ulysses_hf_attention = real
    if control is None:
        # Engagement: main's Ulysses attention runs once per layer (twice under checkpointing).
        runs = 2 if case.get("checkpoint") else 1
        text = _config(case["kind"])
        assert len(calls) == runs * text.num_hidden_layers, (len(calls), runs)
        expected = {
            (text.num_attention_heads, layer.num_key_value_heads, layer.head_dim)
            for layer in text.per_layer_config
        }
        assert set(calls) == expected, (set(calls), expected)

    ref_params = dict(reference.named_parameters())
    for name, param in model.named_parameters():
        statuses = [None] * world
        dist.all_gather_object(statuses, _grad_status(param.grad))
        expected = ref_params[name].grad
        if expected is None:
            assert statuses == [0] * world, (name, statuses)
            continue
        # Every rank holds a gradient (a rank without labels too), or the reduction desyncs.
        assert 0 not in statuses, (name, statuses)
        grad = param.grad
        dist.all_reduce(grad)
        torch.testing.assert_close(
            grad, expected, rtol = 1e-5, atol = 1e-6, msg = lambda m: f"{name}: {m}"
        )
        assert _grad_status(grad) == _grad_status(expected), name
    # One AdamW step with weight decay: None skips the parameter, zero still decays it. A large eps
    # keeps the first step Lipschitz in the gradient (Gemma 4's RMSNorm runs in float32 even under
    # float64, so gradients carry float32 rounding).
    for net in (reference, model):
        trainable = [p for p in net.parameters() if p.requires_grad]
        torch.optim.AdamW(trainable, lr = 1e-3, eps = 1e-3, weight_decay = 0.1).step()
    for (name, mine), theirs in zip(model.named_parameters(), reference.parameters()):
        torch.testing.assert_close(
            mine, theirs, rtol = 1e-5, atol = 1e-6, msg = lambda m: f"after AdamW {name}: {m}"
        )


def _case_worker(rank, world, case):
    _model_case(rank, world, case)


E2B = dict(kind = "e2b")
MOE = dict(kind = "moe")
HD512 = dict(kind = "hd512")
CASES = {
    # Windows 3 / 9 cross the seam; 64 covers the whole sequence.
    "e2b_cp2": dict(E2B, cp = 2, world = 2, window = 3),
    "e2b_cp2_window9_prompt_masked": dict(E2B, cp = 2, world = 2, window = 9, labels = "prompt_masked"),
    "e2b_cp2_window64": dict(E2B, cp = 2, world = 2, window = 64),
    "e2b_cp4": dict(E2B, cp = 4, world = 4, window = 9, labels = "prompt_masked"),
    "e2b_dp2cp2": dict(E2B, cp = 2, world = 4, window = 5),
    "e2b_cp2_eager_checkpoint": dict(E2B, cp = 2, world = 2, window = 3, attn = "eager", checkpoint = True),
    "e2b_cp2_lora": dict(E2B, cp = 2, world = 2, window = 3, lora = "dense"),
    "moe_cp2": dict(MOE, cp = 2, world = 2, window = 3),
    "moe_cp2_window9_prompt_masked": dict(MOE, cp = 2, world = 2, window = 9, labels = "prompt_masked"),
    "moe_cp4": dict(MOE, cp = 4, world = 4, window = 3),
    "moe_dp2cp2": dict(MOE, cp = 2, world = 4, window = 3, labels = "prompt_masked"),
    "moe_cp2_eager_checkpoint": dict(MOE, cp = 2, world = 2, window = 3, attn = "eager", checkpoint = True),
    "moe_cp2_lora": dict(MOE, cp = 2, world = 2, window = 9, lora = "dense"),
    "moe_cp2_lora_experts": dict(MOE, cp = 2, world = 2, window = 3, lora = "experts"),
    "moe_cp2_lora_experts_eager_checkpoint": dict(
        MOE, cp = 2, world = 2, window = 3, lora = "experts", attn = "eager", checkpoint = True
    ),
    "hd512_cp2": dict(HD512, cp = 2, world = 2, window = 3),
    "hd512_cp2_eager": dict(HD512, cp = 2, world = 2, window = 3, attn = "eager"),
}


@pytest.mark.parametrize("name", list(CASES))
def test_gemma4_cp_matches_fp64_reference(name):
    case = CASES[name]
    _spawn(_case_worker, case["world"], case)


def _controls_worker(rank, world):
    passed = []
    for control in (
        "shard_local",
        "global_layers_windowed",
        "dropped_window",
        "default_scale",
        "position_offset",
    ):
        try:
            _model_case(rank, world, dict(E2B, cp = 2, window = 3), control = control)
            passed.append(control)
        except AssertionError:
            pass
    assert not passed, f"negative controls that matched the reference: {passed}"


def test_gemma4_negative_controls_fail():
    _spawn(_controls_worker, 2)


# ---------------------------------------------------------------------------------------------
# What main lacks: the head split per layer kind (heterogeneous configs).


def test_gemma4_head_counts_are_read_per_layer():
    model = _model("moe", dtype = torch.float32)
    # transformers >= 5.15: the global read raises (a RuntimeError, so getattr's default does not catch it).
    with pytest.raises(RuntimeError, match = "per-layer attribute"):
        model.config.num_key_value_heads
    assert cp._model_heads(model) == [(4, 1), (4, 2)]
    assert cp._model_heads(_model("e2b", dtype = torch.float32)) == [(4, 1)]
    cp._check_cp_model(model, 2, "ulysses")
    cp._check_cp_model(model, 4, "ulysses")
    assert cp.resolve_cp_backend("auto", 2, model) == "ulysses"
    # Gemma 4 configs from before per-layer configs: num_global_key_value_heads gated by attention_k_eq_v.
    old = types.SimpleNamespace(
        config = types.SimpleNamespace(
            num_attention_heads = 6,
            num_key_value_heads = 6,
            num_global_key_value_heads = 3,
            attention_k_eq_v = True,
        )
    )
    assert cp._model_heads(old) == [(6, 3), (6, 6)]
    old.config.attention_k_eq_v = False
    assert cp._model_heads(old) == [(6, 6)]
    # Homogeneous configs keep main's single pair (no key / value heads: one per query head).
    llama = types.SimpleNamespace(config = types.SimpleNamespace(num_attention_heads = 8))
    assert cp._model_heads(llama) == [(8, 8)]
    assert cp._model_heads(types.SimpleNamespace(config = None)) == []


def test_gemma4_global_layer_heads_that_cannot_split_refuse_before_any_collective():
    # Sliding layers: 6 query / 6 KV heads split over 2; global layers: 3 KV heads do not.
    text = _config(
        "moe", num_attention_heads = 6, num_key_value_heads = 6, num_global_key_value_heads = 3
    )
    assert {layer.num_key_value_heads for layer in text.per_layer_config} == {3, 6}
    model = types.SimpleNamespace(config = text)
    assert cp._model_heads(model) == [(6, 3), (6, 6)]
    # No process group exists here: the refusal is raised before any collective.
    with pytest.raises(ValueError, match = r"key/value heads \(3\)"):
        cp._ulysses_head_plans(model, 2)
    # auto falls back to ring, which refuses the window with its own reason.
    assert cp.resolve_cp_backend("auto", 2, model) == "ring"
    assert cp.resolve_cp_backend("auto", 3, model) == "ulysses"


# ---------------------------------------------------------------------------------------------
# Covered by main: the window rules, the fast forward refusal, multimodal refusal.


def _check(
    model,
    backend = "ulysses",
    allow_hybrids = True,
):
    try:
        cp._check_cp_model(model, 2, backend, allow_hybrids)
    except NotImplementedError as error:
        return str(error)
    return None


def test_gemma4_generic_route_gates(monkeypatch):
    from unsloth.models.llama import LlamaAttention_fast_forward

    model = _model("e2b", dtype = torch.float32)
    assert cp._generic_hf_attention(model)
    assert cp._supports_context_parallel(model, "ulysses")
    assert not cp._supports_context_parallel(model, "ring")
    assert _check(model) is None
    assert "sliding" in _check(model, "ring")
    # DPO / GRPO keep the Llama-style rule.
    assert _check(model, allow_hybrids = False) is not None
    # Llama's fast forward ignores the window: Gemma 4 attention running it is not generic, and refuses.
    attention = type(model.model.layers[0].self_attn)
    monkeypatch.setattr(attention, "forward", LlamaAttention_fast_forward)
    assert not cp._generic_hf_attention(model)
    assert "sliding" in _check(model)
    assert "sliding" in _check(model, allow_hybrids = False)
    monkeypatch.undo()
    # Multimodal Gemma 4 stays refused: its vision attention is bidirectional, which main's attach refuses.
    multimodal = _model("mm_e2b", dtype = torch.float32)
    with pytest.raises(NotImplementedError, match = "Gemma4VisionAttention is bidirectional"):
        _manager(2, None, local = 0).attach_attention_hooks(multimodal)
    # Off the generic route, windowed model types stay refused; chunked layers refuse everywhere.
    for model_type in ("gemma3_text", "mistral", "qwen2", "gemma4_text"):
        other = types.SimpleNamespace(
            model_type = model_type,
            sliding_window = 4,
            layer_types = ["sliding_attention", "full_attention"],
        )
        assert cp._unsupported_attention(other, "ulysses") == "sliding_attention layers"
    text = _config("e2b")
    text.layer_types = ["chunked_attention"] * 4
    assert cp._unsupported_attention(text, "ulysses", generic = True) == "chunked_attention layers"


# ---------------------------------------------------------------------------------------------
# Flash: main hands the global layers' head_dim to the model's own flash function, so a head_dim
# flash-attn cannot run fails under CP exactly as in one process (no silent switch to SDPA).

_FLASH_MAX_HEAD_DIM = 256  # flash-attn 2's forward / backward limit


def _fake_flash(module, query, key, value, attention_mask, **kwargs):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    if query.shape[-1] > _FLASH_MAX_HEAD_DIM:
        raise RuntimeError("FlashAttention forward only supports head dimension at most 256")
    kwargs.pop("sliding_window", None)
    return sdpa_attention_forward(module, query, key, value, None, **kwargs)


def _flash_worker(rank, world, kind, results):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = _fake_flash
    model = _model(kind, attn = "flash_attention_2", dtype = torch.float32)
    ids = torch.randint(0, 64, (1, 16), generator = torch.Generator().manual_seed(1))

    def run(fn):
        try:
            fn()
        except RuntimeError as error:
            return str(error)
        return None

    one = run(lambda: model(input_ids = ids, use_cache = False))
    group = _cp_groups(world, world)
    manager = _manager(world, group)
    manager.attach_attention_hooks(model)
    calls = []
    real = uh._ulysses_hf_attention

    def counted(*args, **kwargs):
        calls.append(kwargs["implementation"])
        return real(*args, **kwargs)

    uh._ulysses_hf_attention = counted
    inputs = {"input_ids": ids.clone(), "labels": ids.clone()}

    def cp_forward():
        with manager.apply(inputs):
            model(
                input_ids = inputs["input_ids"], position_ids = inputs["position_ids"], use_cache = False
            )

    many = run(cp_forward)
    uh._ulysses_hf_attention = real
    assert calls and set(calls) == {"flash_attention_2"}, calls
    assert one == many, (one, many)
    results.append((one, many))


@pytest.mark.parametrize("kind, fails", [("hd512", True), ("moe", False)])
def test_flash_head_dim_limit_matches_one_process(kind, fails):
    def worker(rank, world):
        results = []
        _flash_worker(rank, world, kind, results)
        one, _ = results[0]
        assert (one is not None) == fails, one
        if fails:
            assert "head dimension" in one

    _spawn(worker, 2)
