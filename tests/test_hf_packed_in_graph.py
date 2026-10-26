# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Packed rows inside an Unsloth compiled decoder block run torch's varlen_attn inside the block's
graph: no graph break, no recompile per length (the longest segment is a SymInt), same numbers as
the dense block-causal mask. CPU: torch's varlen ops get a math kernel here (they only ship CUDA
kernels), the CUDA-only gate is patched open, as on an sm80+ GPU."""

import importlib.util
import os
import random
import sys
import types

import pytest
import torch
import torch.nn.functional as F

from unsloth.utils import hf_packed_attention as hpa

_DIR = os.path.join(os.path.dirname(__file__), "..", "unsloth", "models")
varlen = pytest.importorskip("torch.nn.attention.varlen")
_HAS_VARLEN = hasattr(torch.ops.torch_attn, "_varlen_attn") and {"scale", "window_size"} <= set(
    __import__("inspect").signature(varlen.varlen_attn).parameters
)
pytestmark = pytest.mark.skipif(not _HAS_VARLEN, reason = "torch varlen_attn without scale / window")


def _load_block_compile():
    # By file path: importing the unsloth package needs an accelerator (as tests/test_block_compile.py).
    spec = importlib.util.spec_from_file_location(
        "_u_block_compile_in_graph", os.path.join(_DIR, "_block_compile.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _segments(query, key, value, cu, is_causal, scale, window_size):
    """Per segment: (slice, q, k, v as (H, n, D) with K / V heads repeated, boolean mask, scale)."""
    bounds = cu.tolist()
    # Flash semantics: keys in [q - left, q + right], -1 unbounded; is_causal sets right = 0.
    left, right = window_size if window_size is not None else (-1, -1)
    if is_causal:
        right = 0
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    for a, b in zip(bounds[:-1], bounds[1:]):
        q, k, v = (t[a:b].transpose(0, 1) for t in (query, key, value))
        groups = q.shape[0] // k.shape[0]
        k, v = k.repeat_interleave(groups, 0), v.repeat_interleave(groups, 0)
        i = torch.arange(b - a)
        mask = torch.ones(b - a, b - a, dtype = torch.bool)
        if right >= 0:
            mask &= i[None, :] <= i[:, None] + right
        if left >= 0:
            mask &= i[None, :] >= i[:, None] - left
        yield slice(a, b), q, k, v, mask, scale, groups


def _probs(q, k, mask, scale):
    scores = (q @ k.transpose(-1, -2)) * scale
    return scores.masked_fill(~mask, float("-inf")).softmax(-1)


def _reference(query, key, value, cu, is_causal, scale, window_size):
    """(T, H, D) math attention per segment."""
    out = torch.empty_like(query)
    for rows, q, k, v, mask, scale, _ in _segments(
        query, key, value, cu, is_causal, scale, window_size
    ):
        out[rows] = (_probs(q, k, mask, scale) @ v).transpose(0, 1)
    return out


def _reference_grads(grad_out, query, key, value, cu, is_causal, scale, window_size):
    """Analytic softmax-attention backward per segment (kernels run below autograd)."""
    dq, dk, dv = torch.zeros_like(query), torch.zeros_like(key), torch.zeros_like(value)
    for rows, q, k, v, mask, scale, groups in _segments(
        query, key, value, cu, is_causal, scale, window_size
    ):
        p = _probs(q, k, mask, scale)
        do = grad_out[rows].transpose(0, 1)
        dv_h = p.transpose(-1, -2) @ do
        dp = do @ v.transpose(-1, -2)
        ds = p * (dp - (dp * p).sum(-1, keepdim = True))
        dq[rows] = ((ds @ k) * scale).transpose(0, 1)
        dk_h = (ds.transpose(-1, -2) @ q) * scale
        h_kv = key.shape[1]
        dk[rows] = dk_h.reshape(h_kv, groups, *dk_h.shape[1:]).sum(1).transpose(0, 1)
        dv[rows] = dv_h.reshape(h_kv, groups, *dv_h.shape[1:]).sum(1).transpose(0, 1)
    return dq, dk, dv


_CALLS = {"fwd": 0, "bwd": 0, "max_q": []}


def _cpu_fwd(
    query,
    key,
    value,
    cu_seq_q,
    cu_seq_k,
    max_q,
    max_k,
    is_causal = False,
    *rest,
    **kw,
):
    names = ("scale", "window_size", "enable_gqa")
    opts = dict(zip(names, rest), **kw)
    _CALLS["fwd"] += 1
    _CALLS["max_q"].append(int(max_q))
    out = _reference(
        query, key, value, cu_seq_q, is_causal, opts.get("scale"), opts.get("window_size")
    )
    lse = torch.zeros(query.shape[1], query.shape[0], dtype = torch.float)
    return out, lse, torch.zeros(2, dtype = torch.uint64)


def _cpu_bwd(
    grad_out, query, key, value, out, lse, cu_seq_q, cu_seq_k, max_q, max_k, is_causal, *rest, **kw
):
    # (rng_state, scale, window_size[, enable_gqa]) after is_causal on 2.11 / 2.12 / 2.13 / 2.14.
    names = ("rng_state", "scale", "window_size", "enable_gqa")
    opts = dict(zip(names, rest), **kw)
    _CALLS["bwd"] += 1
    return _reference_grads(
        grad_out, query, key, value, cu_seq_q, is_causal, opts.get("scale"), opts.get("window_size")
    )


_REGISTERED = []


@pytest.fixture(scope = "module", autouse = True)
def _cpu_varlen_kernels():
    if not _REGISTERED:
        torch.library.register_kernel("torch_attn::_varlen_attn", "cpu", _cpu_fwd)
        torch.library.register_kernel("torch_attn::_varlen_attn_backward", "cpu", _cpu_bwd)
        _REGISTERED.append(True)
    yield


@pytest.fixture
def gate(monkeypatch):
    """The packed wrapper installed, torch varlen available, both block gates as on a B200 with
    torch < 2.13 (kernel in graph possible, #14808 would leave the graph)."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    monkeypatch.delenv("UNSLOTH_HF_PACKED_IN_GRAPH", raising = False)
    saved_entry, saved_orig = ALL_ATTENTION_FUNCTIONS["sdpa"], hpa._ORIG_SDPA[0]
    if not getattr(saved_entry, hpa._FLAG, False):
        hpa._ORIG_SDPA[0] = None
        assert hpa.enable_hf_packed_attention()
    monkeypatch.setattr(hpa, "_TORCH_VARLEN", [varlen.varlen_attn])
    gqa = "enable_gqa" in __import__("inspect").signature(varlen.varlen_attn).parameters
    monkeypatch.setattr(hpa, "_TORCH_VARLEN_GQA", [gqa])
    monkeypatch.setattr(hpa, "_leaves_block", torch._dynamo.assume_constant_result(lambda *a: True))
    # Absent before this change: setting it there is harmless and main keeps leaving the graph.
    monkeypatch.setattr(
        hpa,
        "_in_graph_kernel",
        torch._dynamo.assume_constant_result(lambda *a: True),
        raising = False,
    )
    monkeypatch.setattr(hpa, "_in_graph_device", lambda device: True, raising = False)
    monkeypatch.setattr(hpa, "_layer_leaves", lambda layer, hidden_states: True, raising = False)
    monkeypatch.setattr(hpa, "_PLAN_CACHE", {}, raising = False)
    _CALLS.update(fwd = 0, bwd = 0, max_q = [])
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    yield
    ALL_ATTENTION_FUNCTIONS["sdpa"] = saved_entry
    hpa._ORIG_SDPA[0] = saved_orig if saved_orig is not None else hpa._ORIG_SDPA[0]
    torch._dynamo.reset()


def _tiny(
    arch = "qwen3",
    dtype = torch.float32,
    seed = 0,
):
    import transformers

    torch.manual_seed(seed)
    common = dict(
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 96,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        max_position_embeddings = 4096,
        attn_implementation = "sdpa",
    )
    if arch == "qwen3":
        config = transformers.Qwen3Config(**common)
        cls = transformers.Qwen3ForCausalLM
    elif arch == "llama":
        config = transformers.LlamaConfig(**common)
        cls = transformers.LlamaForCausalLM
    elif arch == "qwen3_sliding":
        config = transformers.Qwen3Config(
            **common,
            use_sliding_window = True,
            sliding_window = 8,
            max_window_layers = 0,
            layer_types = ["sliding_attention", "full_attention"],
        )
        cls = transformers.Qwen3ForCausalLM
    elif arch == "gemma3":
        config = transformers.Gemma3TextConfig(
            **common, sliding_window = 8, layer_types = ["sliding_attention", "full_attention"]
        )
        cls = transformers.Gemma3ForCausalLM
    else:
        raise ValueError(arch)
    model = cls(config).to(dtype)
    model.train()
    return model


def _counting_compile(
    monkeypatch,
    M,
    backend = "aot_eager",
):
    """_block_compile's torch.compile with a counting backend (no Inductor / C++ toolchain on CPU)."""
    from torch._dynamo.testing import CompileCounterWithBackend

    counter = CompileCounterWithBackend(backend)
    graphs = []
    real = torch.compile

    def capture(gm, example_inputs):
        graphs.append(gm)
        return counter(gm, example_inputs)

    def fake_compile(fn, **kwargs):
        kwargs.pop("options", None)
        return real(
            fn,
            backend = capture,
            **{k: v for k, v in kwargs.items() if k in ("fullgraph", "dynamic")},
        )

    monkeypatch.setattr(torch, "compile", fake_compile)
    return counter, graphs


def _wrap_layers(monkeypatch, model):
    M = _load_block_compile()
    counter, graphs = _counting_compile(monkeypatch, M)
    layer_cls = type(model.model.layers[0])
    original = layer_cls.forward
    M._wrap(layer_cls, {})
    return M, counter, graphs, (layer_cls, original)


def _packed_batch(
    lengths,
    vocab = 128,
    seed = 0,
):
    g = torch.Generator().manual_seed(seed)
    total = sum(lengths)
    input_ids = torch.randint(0, vocab, (1, total), generator = g)
    position_ids = torch.cat([torch.arange(n) for n in lengths])[None]
    return dict(
        input_ids = input_ids,
        position_ids = position_ids,
        packed_seq_lengths = torch.tensor(lengths, dtype = torch.int32),
        use_cache = False,
    )


def _loss(model, batch):
    out = model(**batch)
    return out.logits.float().square().mean()


def _graph_breaks():
    return sum(torch._dynamo.utils.counters["graph_break"].values())


def _ops_in(graphs):
    names = []
    for gm in graphs:
        for node in gm.graph.nodes:
            if node.op == "call_function":
                names.append(str(node.target))
    return names


@pytest.mark.parametrize("arch", ["qwen3", "llama", "qwen3_sliding"])
def test_packed_rows_stay_in_the_compiled_block(monkeypatch, gate, arch):
    """Where #14808 leaves the graph per layer (cuDNN over the dense mask, torch < 2.13; forced here),
    the varlen kernel now runs inside it: zero graph
    breaks, torch_attn._varlen_attn in the block graph, and every layer call took it."""
    model = _tiny(arch)
    M, counter, graphs, restore = _wrap_layers(monkeypatch, model)
    try:
        lengths = [5, 9, 3, 7]
        loss = _loss(model, _packed_batch(lengths))
        loss.backward()
    finally:
        restore[0].forward = restore[1]
    assert _graph_breaks() == 0, dict(torch._dynamo.utils.counters["graph_break"])
    assert any("_varlen_attn" in name for name in _ops_in(graphs))
    n_layers = len(model.model.layers)
    assert _CALLS["fwd"] == n_layers and _CALLS["bwd"] == n_layers
    # The exact longest segment, read from the plan (no host sync inside the graph).
    assert set(_CALLS["max_q"]) == {9}


@pytest.mark.parametrize("arch", ["qwen3", "llama", "qwen3_sliding"])
def test_in_graph_rows_match_the_dense_mask_and_an_fp64_oracle(monkeypatch, gate, arch):
    """Loss and every gradient: compiled block with in-graph varlen vs the stock forward over the
    dense block-causal mask (fp32), both against the same model in fp64."""
    lengths = [6, 11, 4, 9]
    batch = _packed_batch(lengths)
    results = {}
    for name, dtype, wrap in (
        ("oracle", torch.float64, False),
        ("stock", torch.float32, False),
        ("ours", torch.float32, True),
    ):
        model = _tiny(arch, dtype = dtype)
        if wrap:
            _, _, graphs, restore = _wrap_layers(monkeypatch, model)
        try:
            loss = _loss(model, batch)
            loss.backward()
        finally:
            if wrap:
                restore[0].forward = restore[1]
        results[name] = (
            loss.detach().double(),
            {n: p.grad.double() for n, p in model.named_parameters()},
        )
    assert _CALLS["fwd"] == len(model.model.layers)  # ours took varlen in every layer
    oracle_loss, oracle_grads = results["oracle"]
    for name in ("stock", "ours"):
        loss, grads = results[name]
        assert abs(loss - oracle_loss) <= 1e-5 * abs(oracle_loss) + 1e-7, (
            name,
            float(loss),
            float(oracle_loss),
        )
        for key, ref in oracle_grads.items():
            err = (grads[key] - ref).norm() / (ref.norm() + 1e-12)
            assert err < 1e-4, (name, key, float(err))
    # Ours is as close to fp64 as the stock path (within 4x of its error, or both tiny).
    for key, ref in oracle_grads.items():
        e_ours = (results["ours"][1][key] - ref).norm()
        e_stock = (results["stock"][1][key] - ref).norm()
        assert e_ours <= 4 * e_stock + 1e-9 * (ref.norm() + 1), key


def test_one_graph_over_50_packed_lengths(monkeypatch, gate):
    """50 packed rows of distinct total length, segment count and longest segment: the block compiles
    once and never breaks (the row length, segment count and longest segment are all symbolic).
    One-document rows (transformers passes no mask, a 1-element lengths tensor is specialised) add
    one graph."""
    model = _tiny("qwen3")
    M, counter, graphs, restore = _wrap_layers(monkeypatch, model)
    rng = random.Random(0)
    longest = set()
    try:
        for documents, phase in (((2, 5), "packed"), ((1, 5), "mixed")):
            totals = set()
            while len(totals) < 50:
                lengths = [rng.randint(2, 70) for _ in range(rng.randint(*documents))]
                if sum(lengths) in totals:
                    continue
                totals.add(sum(lengths))
                longest.add(max(lengths))
                _CALLS["max_q"].clear()
                _loss(model, _packed_batch(lengths, seed = len(totals))).backward()
                model.zero_grad(set_to_none = True)
                assert set(_CALLS["max_q"]) == {max(lengths)}
            if phase == "packed":
                assert len(longest) >= 30
                assert counter.frame_count == 1, counter.frame_count
    finally:
        restore[0].forward = restore[1]
    assert _graph_breaks() == 0, dict(torch._dynamo.utils.counters["graph_break"])
    assert counter.frame_count <= 2, counter.frame_count


def test_kill_switch_restores_the_out_of_graph_path(monkeypatch, gate):
    """UNSLOTH_HF_PACKED_IN_GRAPH=0: #14808's behaviour (the row leaves the graph per layer)."""
    monkeypatch.setenv("UNSLOTH_HF_PACKED_IN_GRAPH", "0")
    model = _tiny("qwen3")
    M, counter, graphs, restore = _wrap_layers(monkeypatch, model)
    try:
        _loss(model, _packed_batch([5, 9, 3])).backward()
    finally:
        restore[0].forward = restore[1]
    assert _graph_breaks() >= 1
    assert not any("_varlen_attn" in name for name in _ops_in(graphs))


@pytest.mark.parametrize(
    "why", ["no_kernel", "bidirectional_mask", "unpacked", "unknown_window", "no_cudnn"]
)
def test_rows_varlen_cannot_take_keep_the_previous_path(monkeypatch, gate, why):
    """No traceable kernel (T4, xformers / flash-attn only, head dim > 256): #14808's gate decides,
    as before. A mask that is not block causal and unpacked rows never take varlen."""
    model = _tiny("qwen3")
    batch = _packed_batch([5, 9, 3])
    if why == "no_kernel":
        monkeypatch.setattr(
            hpa, "_in_graph_kernel", torch._dynamo.assume_constant_result(lambda *a: False)
        )
    elif why == "bidirectional_mask":
        total = 17
        mask = torch.zeros(1, 1, total, total, dtype = torch.bool)
        for a, b in ((0, 5), (5, 14), (14, 17)):
            mask[0, 0, a:b, a:b] = True  # bidirectional inside each document
        batch["attention_mask"] = mask
    elif why == "no_cudnn":
        # Dense-mask SDPA would not pick cuDNN (RTX PRO 6000, H100, A100, T4, torch >= 2.13): the
        # masked SDPA stays in the graph exactly as before and no plan is built.
        monkeypatch.setattr(
            hpa, "_leaves_block", torch._dynamo.assume_constant_result(lambda *a: False)
        )
        monkeypatch.setattr(hpa, "_layer_leaves", lambda layer, hidden_states: False)
        built = []
        real_plan = hpa.block_packed_plan
        monkeypatch.setattr(
            hpa, "block_packed_plan", lambda *a: built.append(real_plan(*a)) or built[-1]
        )
    elif why == "unknown_window":
        # A window none of the layer's modules or configs names (the mask was not checked for it).
        monkeypatch.setattr(hpa, "_layer_windows", lambda layer: (5,))
    else:
        batch.pop("packed_seq_lengths")
    M, counter, graphs, restore = _wrap_layers(monkeypatch, model)
    try:
        _loss(model, batch).backward()
    finally:
        restore[0].forward = restore[1]
    assert _CALLS["fwd"] == 0
    assert not any("_varlen_attn" in name for name in _ops_in(graphs))
    if why == "no_cudnn":
        assert built and not any(built)
        assert "_sdpa_packed_varlen" not in " ".join(torch._dynamo.utils.counters["graph_break"])
    if why == "no_kernel":
        # The forced #14808 gate still sends a packed row without an in-graph kernel out of the graph.
        reasons = " ".join(torch._dynamo.utils.counters["graph_break"])
        assert "_sdpa_packed_varlen" in reasons


def _run_twice(monkeypatch, model, second_policy):
    """One compiled step with the default SDPA policy, then one under second_policy (a context)."""
    M, counter, graphs, restore = _wrap_layers(monkeypatch, model)
    try:
        _loss(model, _packed_batch([5, 9, 3])).backward()
        first = _CALLS["fwd"]
        _CALLS.update(fwd = 0, bwd = 0, max_q = [])
        with second_policy():
            _loss(model, _packed_batch([5, 9, 3])).backward()
    finally:
        restore[0].forward = restore[1]
    return first, _CALLS["fwd"]


def test_flash_disabled_after_a_step_keeps_sdpa(monkeypatch, gate):
    """sdpa_kernel(MATH) after an in-graph step: no plan, so the row takes the eager wrapper, which
    honours the policy (_sdpa_requested) as before this change."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    first, second = _run_twice(monkeypatch, _tiny("qwen3"), lambda: sdpa_kernel([SDPBackend.MATH]))
    assert first > 0 and second == 0, (first, second)


def test_math_first_priority_after_a_step_retraces(monkeypatch, gate):
    """Flash still enabled but math first: the plan carries the policy, Dynamo guards on it, and the
    kernel probe (keyed by it) is asked again, so the traced varlen call is not reused."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    math = int(SDPBackend.MATH)
    monkeypatch.setattr(
        hpa,
        "_in_graph_kernel",
        torch._dynamo.assume_constant_result(lambda *a: not a[-1][-1][:1] == (math,)),
    )
    first, second = _run_twice(
        monkeypatch,
        _tiny("qwen3"),
        lambda: sdpa_kernel([SDPBackend.MATH, SDPBackend.FLASH_ATTENTION], set_priority = True),
    )
    assert first > 0 and second == 0, (first, second)


def test_kernel_verdict_is_cached_per_sdpa_policy(monkeypatch):
    asked = []
    monkeypatch.setattr(hpa, "_IN_GRAPH", {})
    monkeypatch.setattr(hpa, "_in_graph_kernel_probe", lambda *key: asked.append(key) or True)
    args = ("cuda", 0, torch.bfloat16, 64, 8, 2, None)
    for policy in (
        (True, True, True, (1, 2)),
        (True, True, True, (1, 2)),
        (True, True, True, (0, 1)),
    ):
        hpa._in_graph_kernel_eager(*args, policy)
    assert len(asked) == 2 and all(key == args for key in asked)


def test_plan_is_only_set_while_a_compiled_block_runs(monkeypatch, gate):
    model = _tiny("qwen3")
    seen = []
    real = hpa._in_graph_varlen

    def spy(*args, **kwargs):
        seen.append(hpa.PACKED_PLAN[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(hpa, "_in_graph_varlen", spy)
    M, counter, graphs, restore = _wrap_layers(monkeypatch, model)
    try:
        _loss(model, _packed_batch([5, 9])).backward()
    finally:
        restore[0].forward = restore[1]
    assert seen and all(plan is not None and plan[1].shape[0] == 9 for plan in seen)
    assert hpa.PACKED_PLAN == [None]


def _tiny_qwen3_vl():
    import transformers

    torch.manual_seed(0)
    config = transformers.Qwen3VLConfig(
        text_config = dict(
            vocab_size = 128,
            hidden_size = 64,
            intermediate_size = 96,
            num_hidden_layers = 2,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            head_dim = 16,
            max_position_embeddings = 4096,
            rope_parameters = {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [2, 3, 3],
                "mrope_interleaved": True,
            },
        ),
        vision_config = dict(
            depth = 1,
            hidden_size = 32,
            intermediate_size = 64,
            num_heads = 2,
            out_hidden_size = 64,
            deepstack_visual_indexes = [0],
        ),
        image_token_id = 120,
        video_token_id = 121,
        vision_start_token_id = 122,
    )
    config._attn_implementation = "sdpa"
    config.text_config._attn_implementation = "sdpa"
    model = transformers.Qwen3VLForConditionalGeneration(config).float()
    model.train()
    return model


def test_vlm_text_row_takes_varlen_in_the_language_model_block(monkeypatch, gate):
    """A packed text-only row through Qwen3-VL's language model: the compiled text decoder block runs
    varlen inside its graph and matches the uncompiled forward over the dense mask."""
    model = _tiny_qwen3_vl()
    batch = _packed_batch([5, 9, 3, 7], vocab = 100)
    ref = model(**batch).logits.detach()
    layer_cls = type(model.model.language_model.layers[0])
    M = _load_block_compile()
    counter, graphs = _counting_compile(monkeypatch, M)
    original = layer_cls.forward
    _CALLS.update(fwd = 0, bwd = 0, max_q = [])
    M._wrap(layer_cls, {})
    try:
        out = model(**batch).logits
        out.float().square().mean().backward()
    finally:
        layer_cls.forward = original
    assert _CALLS["fwd"] == len(model.model.language_model.layers)
    assert _graph_breaks() == 0, dict(torch._dynamo.utils.counters["graph_break"])
    torch.testing.assert_close(out.detach(), ref, atol = 1e-5, rtol = 1e-4)


def test_gemma3_rows_are_unchanged(monkeypatch, gate):
    """Gemma 3 attention under unsloth_zoo's fp32 patch never reaches the transformers "sdpa" entry
    (eager or compiled): its packed rows give the same loss and gradients with the in-graph path on
    or off, and take no varlen call."""
    batch = _packed_batch([6, 11, 4, 9])
    results = []
    for in_graph in ("1", "0"):
        monkeypatch.setenv("UNSLOTH_HF_PACKED_IN_GRAPH", in_graph)
        torch._dynamo.reset()
        model = _tiny("gemma3")
        _, _, _, restore = _wrap_layers(monkeypatch, model)
        try:
            loss = _loss(model, batch)
            loss.backward()
        finally:
            restore[0].forward = restore[1]
        results.append((loss.detach(), [p.grad.clone() for p in model.parameters()]))
    assert _CALLS["fwd"] == 0
    assert torch.equal(results[0][0], results[1][0])
    assert all(torch.equal(a, b) for a, b in zip(results[0][1], results[1][1]))


# --- GPU: the real gate and kernel (no CPU kernel, no patched probes) --------------------------

from real_accelerator import has_real_cuda  # noqa: E402  tests/_shared, on sys.path via conftest


def _gpu_varlen():
    if not has_real_cuda():
        return None
    device = torch.device("cuda", torch.cuda.current_device())
    return device if hpa._torch_varlen(device) is not None else None


_GPU = pytest.mark.skipif(_gpu_varlen() is None, reason = "needs CUDA sm80+ with torch varlen_attn")


def _gpu_inputs(
    lengths,
    n_heads,
    n_kv,
    head_dim = 64,
    dtype = torch.bfloat16,
    seed = 0,
):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    total = sum(lengths)
    make = lambda h: torch.randn(1, h, total, head_dim, device = "cuda", dtype = dtype, generator = g)
    q, k, v = make(n_heads), make(n_kv), make(n_kv)
    segment = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths)).cuda()
    mask = ((segment[:, None] == segment[None, :]).tril())[None, None]
    return q, k, v, mask, torch.tensor(lengths, dtype = torch.int32)


def _oracle(
    q,
    k,
    v,
    mask,
    window = None,
):
    q64, k64, v64 = (t.double() for t in (q, k, v))
    groups = q.shape[1] // k.shape[1]
    k64, v64 = k64.repeat_interleave(groups, 1), v64.repeat_interleave(groups, 1)
    if window is not None:
        i = torch.arange(q.shape[2], device = q.device)
        mask = mask & (i[None, :] > i[:, None] - window)[None, None]
    out = F.scaled_dot_product_attention(q64, k64, v64, attn_mask = mask)
    return out.transpose(1, 2)


@_GPU
@pytest.mark.parametrize("n_kv,window", [(8, None), (2, None), (2, 24)])
def test_gpu_in_graph_varlen_matches_the_eager_kernel_and_fp64(monkeypatch, n_kv, window):
    """Forward bitwise equal to the eager wrapper's varlen call (same kernel, same max);
    backward within the eager kernel's own run-to-run spread; both as close to fp64 as each other."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    monkeypatch.delenv("UNSLOTH_HF_PACKED_IN_GRAPH", raising = False)
    monkeypatch.setattr(hpa, "_layer_leaves", lambda layer, hidden_states: True)
    monkeypatch.setattr(hpa, "_PLAN_CACHE", {})
    hpa._ORIG_SDPA[0] = sdpa_attention_forward
    lengths = [40, 150, 7, 61]
    q, k, v, mask, psl = _gpu_inputs(lengths, 8, n_kv)
    if window is not None:
        # The mask transformers builds for a sliding layer: kv_idx > q_idx - window as well.
        i = torch.arange(sum(lengths), device = "cuda")
        mask = mask & (i[None, :] > i[:, None] - window)[None, None]
    module = types.SimpleNamespace(
        is_causal = True, training = True, sliding_window = window, num_key_value_groups = 8 // n_kv
    )
    layer = torch.nn.Module()
    layer.sliding_window = window
    plan = hpa.block_packed_plan(
        layer,
        q.new_zeros(1, sum(lengths), 8),
        (),
        {"attention_mask": mask, "packed_seq_lengths": psl},
    )
    assert plan is not None and plan[1].shape[0] == 150

    def eager(q, k, v):
        hpa.PACKED_PLAN[0] = None
        return hpa._sdpa_packed_varlen(module, q, k, v, mask, packed_seq_lengths = psl)[0]

    def in_graph(q, k, v):
        out = hpa._in_graph_varlen(
            module, q, k, v, mask, 0.0, None, None, {"packed_seq_lengths": psl}
        )
        assert out is not None
        return out

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(in_graph, fullgraph = True, dynamic = False)
    results = {}
    for name, fn in (("eager", eager), ("eager2", eager), ("ours", compiled)):
        leaves = [t.detach().clone().requires_grad_() for t in (q, k, v)]
        hpa.PACKED_PLAN[0] = plan
        before = dict(hpa.HF_PACKED_ATTENTION_STATS)
        out = fn(*leaves)
        if name == "eager":
            assert hpa.HF_PACKED_ATTENTION_STATS["fast"] == before["fast"] + 1  # eager varlen ran
        grad = torch.randn_like(out, generator = torch.Generator(device = "cuda").manual_seed(1))
        out.backward(grad)
        results[name] = (out.detach(), [t.grad for t in leaves], grad)
    hpa.PACKED_PLAN[0] = None
    assert sum(torch._dynamo.utils.counters["graph_break"].values()) == 0
    assert torch.equal(results["ours"][0], results["eager"][0])
    ref = _oracle(q, k, v, mask, window)
    err = lambda o: (o.double() - ref).abs().max().item()
    assert err(results["ours"][0]) <= 1.5 * err(results["eager"][0]) + 1e-6
    # Backward: flash's dq accumulates with atomics, so eager vs eager is the noise floor.
    for i in range(3):
        a, b, c = results["eager"][1][i], results["eager2"][1][i], results["ours"][1][i]
        spread = (a - b).abs().max().item()
        scale = a.abs().max().item()
        assert (c - a).abs().max().item() <= max(
            2 * spread, 2 * torch.finfo(torch.bfloat16).eps * scale
        ), i
    q64, k64, v64 = (t.detach().double().requires_grad_() for t in (q, k, v))
    _oracle(q64, k64, v64, mask, window).backward(results["eager"][2].double())
    for i, ref_grad in enumerate((q64.grad, k64.grad, v64.grad)):
        e_ours = (results["ours"][1][i].double() - ref_grad).abs().max().item()
        e_eager = (results["eager"][1][i].double() - ref_grad).abs().max().item()
        assert e_ours <= 1.5 * e_eager + 1e-6, (i, e_ours, e_eager)


@_GPU
@pytest.mark.parametrize("cudnn_dense", [True, False])
def test_gpu_compiled_qwen3_block_takes_varlen_only_where_14808_left(monkeypatch, cudnn_dense):
    """The real kernel probe on this GPU, the cuDNN verdict forced both ways: where #14808 would
    leave the graph (cuDNN over the dense mask, torch < 2.13), a bf16 tiny Qwen3 with Inductor-compiled
    blocks runs the flash varlen kernel inside the graph (zero graph breaks); elsewhere the
    dense-mask SDPA stays in the graph with no flash kernel, as before. Both match the uncompiled
    forward."""
    from torch.profiler import ProfilerActivity, profile

    monkeypatch.delenv("UNSLOTH_HF_PACKED_VARLEN", raising = False)
    monkeypatch.delenv("UNSLOTH_HF_PACKED_IN_GRAPH", raising = False)
    monkeypatch.setattr(hpa, "_leaves_block_probe", lambda *args: cudnn_dense)
    monkeypatch.setattr(hpa, "_LEAVES_BLOCK", {})
    monkeypatch.setattr(hpa, "_IN_GRAPH", {})
    monkeypatch.setattr(hpa, "_PLAN_CACHE", {})
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if not getattr(ALL_ATTENTION_FUNCTIONS["sdpa"], hpa._FLAG, False):
        hpa._ORIG_SDPA[0] = None
        assert hpa.enable_hf_packed_attention()
    model = _tiny("qwen3").cuda().to(torch.bfloat16)
    batch = {
        k: (v.cuda() if isinstance(v, torch.Tensor) and k != "packed_seq_lengths" else v)
        for k, v in _packed_batch([40, 100, 7, 61]).items()
    }
    ref = model(**batch).logits.float().detach()
    M = _load_block_compile()
    layer_cls = type(model.model.layers[0])
    original = layer_cls.forward
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    M._wrap(layer_cls, dict(M._COMPILE_OPTIONS))
    try:
        model(**batch).logits.float().square().mean().backward()  # compile
        with profile(activities = [ProfilerActivity.CUDA]) as prof:
            out = model(**batch).logits.float()
            out.square().mean().backward()
            torch.cuda.synchronize()
    finally:
        layer_cls.forward = original
    breaks = torch._dynamo.utils.counters["graph_break"]
    if cudnn_dense:
        assert sum(breaks.values()) == 0, dict(breaks)
    else:
        # Whatever the wrapped sdpa does in the graph is unchanged; the packed wrapper never breaks.
        assert "_sdpa_packed_varlen" not in " ".join(breaks), dict(breaks)
    names = [e.key.lower() for e in prof.key_averages()]
    # Flash fwd + bwd kernels (cuDNN / memory-efficient SDPA over the dense mask are named otherwise).
    assert (sum("flash" in n for n in names) >= 2) == cudnn_dense, names[:40]
    assert any(hpa._IN_GRAPH.values()) == cudnn_dense
    torch.testing.assert_close(out.detach(), ref, atol = 3e-2, rtol = 3e-2)
