# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for ``diffusion_capture_safe.py`` (HunyuanImage-2.1's capture-safe forward).

CPU only. The rewrite is checked against synthetic modules carrying the stock merge block in both
LoRA styles diffusers has shipped (inline in 0.36, a decorator from 0.37), against a drifted block,
and, when the installed diffusers has the class, against a tiny random HunyuanImage transformer.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
import types

import pytest

from core.inference import diffusion_capture_safe as cs
from core.inference import diffusion_cuda_graph as cg

torch = pytest.importorskip("torch")

# The stock merge block as diffusers 0.36.0 through 0.41 ship it (8-space indent inside the if).
_STOCK_BLOCK = """
            # reorder and combine text tokens: combine valid tokens first, then padding
            new_encoder_hidden_states = []
            new_encoder_attention_mask = []

            for text, text_mask, text_2, text_mask_2 in zip(
                encoder_hidden_states, encoder_attention_mask, encoder_hidden_states_2, encoder_attention_mask_2
            ):
                # Concatenate: [valid_mllm, valid_byt5, invalid_mllm, invalid_byt5]
                new_encoder_hidden_states.append(
                    torch.cat(
                        [
                            text_2[text_mask_2],  # valid byt5
                            text[text_mask],  # valid mllm
                            text_2[~text_mask_2],  # invalid byt5
                            text[~text_mask],  # invalid mllm
                        ],
                        dim=0,
                    )
                )

                # Apply same reordering to attention masks
                new_encoder_attention_mask.append(
                    torch.cat(
                        [
                            text_mask_2[text_mask_2],
                            text_mask[text_mask],
                            text_mask_2[~text_mask_2],
                            text_mask[~text_mask],
                        ],
                        dim=0,
                    )
                )

            encoder_hidden_states = torch.stack(new_encoder_hidden_states)
            encoder_attention_mask = torch.stack(new_encoder_attention_mask)
"""

_MODULE_TEMPLATE = """
import functools

import torch

SCALES = []


def apply_lora_scale(kwargs_name):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            SCALES.append((kwargs.get(kwargs_name) or {{}}).get("scale", 1.0))
            return fn(self, *args, **kwargs)

        return wrapper

    return decorator


class HunyuanImageTransformer2DModel:
    {decorator}
    def forward(
        self,
        encoder_hidden_states,
        encoder_attention_mask,
        encoder_hidden_states_2=None,
        encoder_attention_mask_2=None,
        attention_kwargs=None,
        return_dict=True,
    ):
        {inline_lora}
        encoder_attention_mask = encoder_attention_mask.bool()
        if encoder_hidden_states_2 is not None:
            encoder_attention_mask_2 = encoder_attention_mask_2.bool()
{block}
        # the blocks see (states * 2) and a float view of the mask
        return (encoder_hidden_states * 2, encoder_attention_mask.float())
"""


def _load_module(
    tmp_path,
    name,
    *,
    decorator = True,
    block = _STOCK_BLOCK,
):
    source = _MODULE_TEMPLATE.format(
        decorator = '@apply_lora_scale("attention_kwargs")' if decorator else "",
        inline_lora = "SCALES.append((attention_kwargs or {}).get('scale', 1.0))"
        if not decorator
        else "pass",
        block = block,
    )
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(source))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fresh_cache(monkeypatch):
    monkeypatch.setattr(cs, "_CACHE", {})
    monkeypatch.delenv(cs.CAPTURE_SAFE_ENV, raising = False)
    yield
    for name in [n for n in sys.modules if n.startswith("_hyimg_fake_")]:
        sys.modules.pop(name, None)


def _reference_merge(e1, m1, e2, m2):
    states, masks = [], []
    for text, tm, text2, tm2 in zip(e1, m1, e2, m2):
        states.append(torch.cat([text2[tm2], text[tm], text2[~tm2], text[~tm]], dim = 0))
        masks.append(torch.cat([tm2[tm2], tm[tm], tm2[~tm2], tm[~tm]], dim = 0))
    return torch.stack(states), torch.stack(masks)


_MASKS = [
    # (mllm mask rows, byt5 mask rows): right-padded, interleaved, all padding, all valid
    ([[1, 1, 1, 0, 0, 0, 0]], [[1, 1, 0, 0, 0]]),
    ([[1, 0, 1, 1, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1]], [[0, 1, 0, 1, 1], [0, 0, 0, 0, 0]]),
    ([[0] * 7, [1] * 7], [[1] * 5, [0] * 5]),
    ([[1] * 7], [[1] * 5]),
]


@pytest.mark.parametrize("m1, m2", _MASKS)
@pytest.mark.parametrize(
    "dtypes", [(torch.float32, torch.float32), (torch.bfloat16, torch.float32)]
)
def test_merge_text_streams_is_the_stock_permutation_bit_for_bit(m1, m2, dtypes):
    gen = torch.Generator().manual_seed(0)
    m1 = torch.tensor(m1).bool()
    m2 = torch.tensor(m2).bool()
    batch = m1.shape[0]
    e1 = torch.randn(batch, m1.shape[1], 6, generator = gen).to(dtypes[0])
    e2 = torch.randn(batch, m2.shape[1], 6, generator = gen).to(dtypes[1])

    want_states, want_mask = _reference_merge(e1, m1, e2, m2)
    got_states, got_mask = cs.merge_text_streams(e1, m1, e2, m2)

    assert got_states.dtype == want_states.dtype
    assert got_mask.dtype == torch.bool
    assert torch.equal(got_states, want_states)
    assert torch.equal(got_mask, want_mask)


def _aten_ops(fn):
    from torch.utils._python_dispatch import TorchDispatchMode

    class _Record(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops: set = set()

        def __torch_dispatch__(
            self,
            func,
            types,
            args = (),
            kwargs = None,
        ):
            self.ops.add(str(func))
            return func(*args, **(kwargs or {}))

    with _Record() as rec:
        fn()
    return rec.ops


# Ops whose output shape depends on tensor VALUES: each one needs the count on the host.
_DATA_DEPENDENT = (
    "aten.index.Tensor",
    "aten.nonzero",
    "aten.masked_select",
    "aten.unique",
    "aten.item",
)


def test_merge_text_streams_has_no_data_dependent_op_where_the_stock_loop_does():
    """The point of the rewrite: nothing sized by a mask, so nothing reads a count on the host."""
    m1 = torch.tensor([[1, 0, 1]]).bool()
    m2 = torch.tensor([[0, 1]]).bool()
    e1, e2 = torch.randn(1, 3, 2), torch.randn(1, 2, 2)

    stock = _aten_ops(lambda: _reference_merge(e1, m1, e2, m2))
    assert any(op.startswith(_DATA_DEPENDENT) for op in stock), stock

    ours = _aten_ops(lambda: cs.merge_text_streams(e1, m1, e2, m2))
    assert not any(op.startswith(_DATA_DEPENDENT) for op in ours), ours


@pytest.mark.parametrize("decorator", [True, False], ids = ["decorator-lora", "inline-lora"])
def test_rewrite_matches_the_stock_forward_and_keeps_lora_handling(
    tmp_path, fresh_cache, decorator
):
    mod = _load_module(tmp_path, f"_hyimg_fake_{int(decorator)}", decorator = decorator)
    cls = mod.HunyuanImageTransformer2DModel
    safe, why = cs.resolve(cls)
    assert why is None
    assert getattr(safe, "__unsloth_capture_safe__", False) is True

    gen = torch.Generator().manual_seed(1)
    m1 = torch.tensor([[1, 0, 1, 1, 0, 0, 1], [1, 1, 0, 0, 0, 0, 0]])
    m2 = torch.tensor([[0, 1, 0, 1, 1], [1, 1, 1, 0, 0]])
    kw = dict(
        encoder_hidden_states = torch.randn(2, 7, 4, generator = gen),
        encoder_attention_mask = m1,
        encoder_hidden_states_2 = torch.randn(2, 5, 4, generator = gen),
        encoder_attention_mask_2 = m2,
        attention_kwargs = {"scale": 0.5},
    )
    obj = cls()
    want = cls.forward(obj, **kw)
    got = safe(obj, **kw)
    assert torch.equal(want[0], got[0])
    assert torch.equal(want[1], got[1])
    # LoRA scaling ran on both paths: the decorator re-applied, or the inline code kept.
    assert mod.SCALES == [0.5, 0.5]
    # The byt5-free call skips the block on both paths.
    kw2 = dict(encoder_hidden_states = kw["encoder_hidden_states"], encoder_attention_mask = m1)
    assert torch.equal(cls.forward(obj, **kw2)[0], safe(obj, **kw2)[0])


def test_rewrite_keeps_the_signature_and_readable_source(tmp_path, fresh_cache):
    import inspect

    mod = _load_module(tmp_path, "_hyimg_fake_sig")
    safe, _ = cs.resolve(mod.HunyuanImageTransformer2DModel)
    assert inspect.signature(safe) == inspect.signature(mod.HunyuanImageTransformer2DModel.forward)
    src = inspect.getsource(safe)
    assert cs._MERGE_HELPER_NAME in src
    assert "text_2[text_mask_2]" not in src


def test_resolve_is_cached_per_forward(tmp_path, fresh_cache):
    mod = _load_module(tmp_path, "_hyimg_fake_cache")
    first = cs.resolve(mod.HunyuanImageTransformer2DModel)
    assert cs.resolve(mod.HunyuanImageTransformer2DModel) is first


def test_drifted_block_is_not_rewritten_and_names_the_reason(tmp_path, fresh_cache):
    drifted = _STOCK_BLOCK.replace(
        "text_2[~text_mask_2],  # invalid byt5\n                            text[~text_mask],  # invalid mllm",
        "text[~text_mask],  # invalid mllm\n                            text_2[~text_mask_2],  # invalid byt5",
    )
    assert drifted != _STOCK_BLOCK
    mod = _load_module(tmp_path, "_hyimg_fake_drift", block = drifted)
    safe, why = cs.resolve(mod.HunyuanImageTransformer2DModel)
    assert safe is None
    assert "HunyuanImageTransformer2DModel forward is not capture-safe" in why
    assert "merge block changed" in why


def test_kill_switch_declines(tmp_path, fresh_cache, monkeypatch):
    mod = _load_module(tmp_path, "_hyimg_fake_kill")
    monkeypatch.setenv(cs.CAPTURE_SAFE_ENV, "0")
    safe, why = cs.resolve(mod.HunyuanImageTransformer2DModel)
    assert safe is None
    assert cs.CAPTURE_SAFE_ENV in why


def test_unknown_class_needs_nothing(fresh_cache):
    class FluxTransformer2DModel:
        def forward(self, x):
            return x

    assert cs.resolve(FluxTransformer2DModel) == (None, None)
    assert cs.resolve(types.SimpleNamespace) == (None, None)


def test_subclass_overriding_forward_is_left_alone(tmp_path, fresh_cache):
    mod = _load_module(tmp_path, "_hyimg_fake_sub")

    class Custom(mod.HunyuanImageTransformer2DModel):
        def forward(self, *args, **kwargs):
            return "custom"

    assert cs.resolve(Custom) == (None, None)

    class Inherits(mod.HunyuanImageTransformer2DModel):
        pass

    assert cs.resolve(Inherits)[0] is not None


def test_graphed_forward_records_the_capture_safe_forward(tmp_path, fresh_cache):
    mod = _load_module(tmp_path, "_hyimg_fake_gf")
    handle = cg.GraphedForward(mod.HunyuanImageTransformer2DModel())
    try:
        assert handle.capture_safe is True
        assert handle.orig.__func__ is cs.resolve(mod.HunyuanImageTransformer2DModel)[0]
    finally:
        handle.free()


def test_graphed_forward_keeps_the_class_forward_otherwise(tmp_path, fresh_cache):
    mod = _load_module(
        tmp_path, "_hyimg_fake_gf_drift", block = _STOCK_BLOCK.replace("dim=0,", "dim = 0,", 1)
    )
    handle = cg.GraphedForward(mod.HunyuanImageTransformer2DModel())
    try:
        assert handle.capture_safe is False
        assert handle.orig.__func__ is mod.HunyuanImageTransformer2DModel.forward
    finally:
        handle.free()


def _eligible_for(transformer):
    return cg.graph_eligible(
        types.SimpleNamespace(device = "cuda", backend = "cuda"),
        family = types.SimpleNamespace(),
        pipe = types.SimpleNamespace(transformer = transformer),
        offload_active = False,
        cache_active = False,
        speed_mode = "default",
    )


def test_graph_eligible_declines_an_unrewritable_hunyuanimage(tmp_path, fresh_cache, monkeypatch):
    monkeypatch.delenv(cg.CUDA_GRAPH_DISABLE_ENV, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    mod = _load_module(
        tmp_path, "_hyimg_fake_elig_drift", block = _STOCK_BLOCK.replace("dim=0,", "dim = 0,", 1)
    )
    ok, reason = _eligible_for(mod.HunyuanImageTransformer2DModel())
    assert ok is False
    assert "not capture-safe" in reason

    good = _load_module(tmp_path, "_hyimg_fake_elig_ok")
    assert _eligible_for(good.HunyuanImageTransformer2DModel()) == (True, "eligible")


def test_real_diffusers_hunyuanimage_rewrite_is_bit_identical(fresh_cache):
    diffusers = pytest.importorskip("diffusers")
    cls = getattr(diffusers, "HunyuanImageTransformer2DModel", None)
    if cls is None:
        pytest.skip("diffusers has no HunyuanImageTransformer2DModel")
    safe, why = cs.resolve(cls)
    assert why is None, why

    torch.manual_seed(0)
    model = cls(
        in_channels = 4,
        out_channels = 4,
        num_attention_heads = 2,
        attention_head_dim = 16,
        num_layers = 1,
        num_single_layers = 1,
        num_refiner_layers = 1,
        patch_size = (1, 1),
        text_embed_dim = 32,
        text_embed_2_dim = 24,
        rope_axes_dim = (8, 8),
    ).eval()
    kw = dict(
        encoder_hidden_states = torch.randn(2, 7, 32),
        encoder_attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0], [1, 0, 1, 1, 1, 0, 1]]),
        encoder_hidden_states_2 = torch.randn(2, 5, 24),
        encoder_attention_mask_2 = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 0, 0, 0]]),
        return_dict = False,
    )
    latents = torch.randn(2, 4, 4, 4)
    timestep = torch.tensor([500.0, 500.0])
    with torch.no_grad():
        want = cls.forward(model, latents, timestep, **kw)[0]
        got = safe(model, latents, timestep, **kw)[0]
    assert torch.equal(want, got)


def test_qwen_image_21_is_declined_with_the_reason(fresh_cache):
    QwenImage21Transformer2DModel = type(
        "QwenImage21Transformer2DModel", (), {"forward": lambda self, hidden_states: hidden_states}
    )
    forward, why = cs.resolve(QwenImage21Transformer2DModel)
    assert forward is None
    assert why == (
        "QwenImage21Transformer2DModel forward is not capture-safe (its pipeline passes the prefix KV "
        "cache as a Python object on every step and its forward syncs the host)"
    )
    # A subclass that brings its own forward is a different forward.
    Sub = type("Sub", (QwenImage21Transformer2DModel,), {"forward": lambda self, x: x})
    assert cs.resolve(Sub) == (None, None)


def test_graph_eligible_declines_qwen_image_21_at_load(fresh_cache, monkeypatch):
    monkeypatch.delenv(cg.CUDA_GRAPH_DISABLE_ENV, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    QwenImage21Transformer2DModel = type(
        "QwenImage21Transformer2DModel", (), {"forward": lambda self, hidden_states: hidden_states}
    )
    ok, why = _eligible_for(QwenImage21Transformer2DModel())
    assert ok is False
    assert why.startswith("QwenImage21Transformer2DModel forward is not capture-safe")
