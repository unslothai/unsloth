# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Masked SDPA backward inside a compiled block (unsloth/models/_block_compile.py).

_sdpa_alignment_without_guard answers Inductor's SDPA alignment guards with `statically_known_true`. On
CUDA torch < 2.14 that made the memory-efficient backward copy the attention bias densely as (B, H, S, S),
so a padded batch whose length is not a multiple of 8 faulted on the first training step (gemma-4 in
Studio): `attn_bias is not correctly aligned (strideH / strideM)` or `CUDA error: misaligned address`.
CPU: that backward keeps torch's guard there (and the compile cache key is tagged so graphs cached without it
are not reused), every other SDPA op still skips it. GPU: a compiled layer
trains on right-padded batches of every length 9 to 17, matches the stock layer and does not recompile
per `seq_len % 8`.
"""

import importlib.util
import os
import sys
import types

import pytest
from real_accelerator import has_real_cuda

torch = pytest.importorskip("torch")

_DIR = os.path.join(os.path.dirname(__file__), "..", "unsloth", "models")


def _load(name = "_block_compile"):
    # By file path: importing the unsloth package needs an accelerator.
    spec = importlib.util.spec_from_file_location(f"_u{name}", os.path.join(_DIR, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def stock_lowering(monkeypatch):
    """Hand each test torch's own sdpa_constraint, and put the lowering table back afterwards."""
    import torch._inductor.lowering as lowering

    table = lowering._maybe_layout_constraints
    saved = dict(table)
    stock = lowering.sdpa_constraint
    while getattr(stock, "_unsloth_no_align_guard", False):
        stock = stock.__wrapped__
    monkeypatch.setattr(lowering, "sdpa_constraint", stock)
    for op, fn in list(table.items()):
        if getattr(fn, "_unsloth_no_align_guard", False):
            table[op] = stock
    monkeypatch.delenv("UNSLOTH_BLOCK_SDPA_ALIGN_GUARD", raising = False)
    try:
        yield lowering
    finally:
        table.clear()
        table.update(saved)


def test_efficient_backward_guard_gate(monkeypatch):
    M = _load()
    monkeypatch.setattr(torch.version, "hip", None)
    for version, keeps in (
        ("2.7.1", True),
        ("2.10.0+cu130", True),
        ("2.13.0", True),
        ("2.14.0", False),
        ("2.15.0", False),
    ):
        monkeypatch.setattr(torch, "__version__", version)
        assert M._efficient_backward_keeps_guard() is keeps, version
    # ROCm never faulted here: unchanged.
    monkeypatch.setattr(torch, "__version__", "2.10.0")
    monkeypatch.setattr(torch.version, "hip", "7.1")
    assert M._efficient_backward_keeps_guard() is False


@pytest.mark.parametrize("keeps", [True, False], ids = ["torch_lt_2_14", "torch_2_14"])
def test_efficient_backward_keeps_torch_guard(monkeypatch, stock_lowering, keeps):
    """The memory-efficient backward sees torch's guard_or_false; flash and the efficient forward do not."""
    from torch._inductor.virtualized import V

    M = _load()
    monkeypatch.setattr(M, "_efficient_backward_keeps_guard", lambda: keeps)
    monkeypatch.setattr(torch.compiler.config, "cache_key_tag", "user-tag")
    seen = []

    def stock_guard(expr):
        return "torch guard"

    def static(expr):
        return "static"

    sizevars = types.SimpleNamespace(guard_or_false = stock_guard, statically_known_true = static)

    def recorder(fx_node, *args, **kwargs):
        seen.append((fx_node.target, V.graph.sizevars.guard_or_false(None)))
        return args, kwargs

    lowering = stock_lowering
    monkeypatch.setattr(lowering, "sdpa_constraint", recorder)
    ops = {
        "efficient_backward": torch.ops.aten._scaled_dot_product_efficient_attention_backward.default,
        "efficient_forward": torch.ops.aten._scaled_dot_product_efficient_attention.default,
        "flash_backward": torch.ops.aten._scaled_dot_product_flash_attention_backward.default,
    }
    for op in ops.values():
        monkeypatch.setitem(lowering._maybe_layout_constraints, op, recorder)
    assert M._sdpa_alignment_without_guard() is True
    wrapper = lowering.sdpa_constraint
    assert wrapper is not recorder and wrapper._unsloth_no_align_guard
    for op in ops.values():
        assert lowering._maybe_layout_constraints[op] is wrapper
    with V.set_graph_handler(types.SimpleNamespace(sizevars = sizevars)):
        for op in ops.values():
            wrapper(types.SimpleNamespace(target = op))
    answers = dict(seen)
    assert answers[ops["efficient_backward"]] == ("torch guard" if keeps else "static")
    assert answers[ops["efficient_forward"]] == "static"
    assert answers[ops["flash_backward"]] == "static"
    # The override is undone after each call.
    assert "guard_or_false" in vars(sizevars) and sizevars.guard_or_false is stock_guard
    # Graphs cached without the guard must not be served again: the cache key gets a tag, once.
    tagged = "user-tag," + M._EFFICIENT_BACKWARD_CACHE_TAG
    assert torch.compiler.config.cache_key_tag == (tagged if keeps else "user-tag")
    if keeps:
        assert M._tag_efficient_backward_cache() is True
        assert torch.compiler.config.cache_key_tag == tagged


def test_align_guard_env_keeps_every_guard(monkeypatch, stock_lowering):
    M = _load()
    monkeypatch.setenv("UNSLOTH_BLOCK_SDPA_ALIGN_GUARD", "1")
    stock = stock_lowering.sdpa_constraint
    assert M._sdpa_alignment_without_guard() is False
    assert stock_lowering.sdpa_constraint is stock


@pytest.mark.skipif(not has_real_cuda(), reason = "compiled block needs CUDA")
def test_padded_batches_train_in_a_compiled_block(monkeypatch, stock_lowering):
    """Right-padded batches at every length 9 to 17 through a compiled layer with masked memory-efficient
    SDPA: no fault, the loss and gradients match the stock layer, and no `% 8` recompile."""
    from transformers.models.qwen3 import modeling_qwen3 as modeling
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    M = _load()
    monkeypatch.setattr(torch.compiler.config, "cache_key_tag", torch.compiler.config.cache_key_tag)
    assert M._sdpa_alignment_without_guard() is True
    # A cached graph from a run without the fix carries the dense bias layout.
    monkeypatch.setattr(torch._inductor.config, "fx_graph_cache", False)
    monkeypatch.setattr(torch._functorch.config, "enable_autograd_cache", False)
    import torch._dynamo.convert_frame as convert_frame

    reasons = []
    stock_reasons = convert_frame.get_and_maybe_log_recompilation_reasons

    def record(*a, **k):
        out = stock_reasons(*a, **k)
        reasons.extend(str(r) for r in out)
        return out

    monkeypatch.setattr(convert_frame, "get_and_maybe_log_recompilation_reasons", record)

    cfg = Qwen3Config(
        vocab_size = 64,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 1,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
    )
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    cls = modeling.Qwen3DecoderLayer
    model = modeling.Qwen3Model(cfg).cuda().float().train()

    def step(L):
        g = torch.Generator(device = "cuda").manual_seed(L)
        ids = torch.randint(1, 64, (2, L), device = "cuda", generator = g)
        mask = torch.ones(2, L, dtype = torch.long, device = "cuda")
        mask[1, L - 3 :] = 0
        model.zero_grad(set_to_none = True)
        out = model(input_ids = ids, attention_mask = mask, use_cache = False).last_hidden_state
        loss = (out * mask[..., None]).square().mean()
        loss.backward()
        torch.cuda.synchronize()
        grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        return loss.detach(), grads

    lengths = list(range(9, 18))
    stock = [step(L) for L in lengths]
    original = cls.forward
    try:
        M._wrap(cls, M._COMPILE_OPTIONS)
        torch._dynamo.reset()
        compiled = [step(L) for L in lengths]
        assert model.layers[0].__dict__.get("_unsloth_block_checked", 0) & 2
    finally:
        cls.forward = original
        torch._dynamo.reset()
    for L, (loss, grads), (loss_c, grads_c) in zip(lengths, stock, compiled):
        assert torch.allclose(loss, loss_c, rtol = 1e-4, atol = 1e-6), (L, loss.item(), loss_c.item())
        err = ((grads - grads_c).norm() / grads.norm()).item()
        assert err <= 1e-3, (L, err)
    assert not [r for r in reasons if "% 8" in r], reasons
