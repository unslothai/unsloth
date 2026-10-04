# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Single-file support Unsloth registers for transformer classes diffusers has not.

Without it, every GGUF load of Qwen-Image-2.1 on a CUDA, ROCm or XPU host died before reading a
byte with "FromOriginalModelMixin is currently only compatible with ...", because a GPU backend
routes GGUFs to diffusers and diffusers had no single-file entry for the class.
"""

from __future__ import annotations

import pytest
import torch

from core.inference import diffusion as studio

CLASS = "QwenImage21Transformer2DModel"
PREFIX = "model.diffusion_model."


def _sfm():
    try:
        import diffusers
        from diffusers.loaders import single_file_model as sfm
    except Exception as exc:  # noqa: BLE001 - an optional integration failing is an environment fact
        pytest.skip(f"diffusers is not importable here: {type(exc).__name__}")
    return diffusers, sfm


def test_the_fused_mlp_splits_gate_first_and_the_prefix_goes():
    """The one layout difference between an sd.cpp 2.1 file and diffusers, and its order.

    Gate first, proj second is what the upstream bf16 weights say: both halves of the fused tensor
    are bit-identical to their diffusers tensors that way round, and off by up to 1.08 the other.
    """
    gate = torch.arange(0, 6 * 4, dtype = torch.float32).reshape(6, 4)
    proj = -torch.arange(0, 6 * 4, dtype = torch.float32).reshape(6, 4)
    checkpoint = {
        PREFIX + "transformer_blocks.3.img_mlp.gate_up.weight": torch.cat([gate, proj]),
        PREFIX + "transformer_blocks.3.img_mlp.out.weight": torch.ones(4, 6),
        "img_in.weight": torch.zeros(2, 2),
    }
    out = studio._qwen_image_21_checkpoint_to_diffusers(checkpoint = checkpoint, config = {})

    assert set(out) == {
        "transformer_blocks.3.img_mlp.gate_layer.weight",
        "transformer_blocks.3.img_mlp.proj.weight",
        "transformer_blocks.3.img_mlp.out.weight",
        "img_in.weight",
    }
    assert torch.equal(out["transformer_blocks.3.img_mlp.gate_layer.weight"], gate)
    assert torch.equal(out["transformer_blocks.3.img_mlp.proj.weight"], proj)


def test_an_odd_fused_tensor_is_refused_rather_than_split_wrong():
    with pytest.raises(ValueError, match = "odd row count"):
        studio._qwen_image_21_checkpoint_to_diffusers(
            checkpoint = {"transformer_blocks.0.img_mlp.gate_up.weight": torch.zeros(5, 4)}
        )


def test_a_row_split_keeps_a_gguf_tensor_quantised():
    """GGML packs blocks along the input dimension, so a row slice is whole blocks and has to stay
    a ``GGUFParameter`` of the same quant type, with the logical shape halved."""
    try:
        import gguf
        from diffusers.quantizers.gguf.utils import GGUFParameter
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"gguf support is not importable here: {type(exc).__name__}")
    qtype = gguf.GGMLQuantizationType.Q4_K
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    cols = 2 * block_size
    raw = torch.randint(0, 255, (8, cols // block_size * type_size), dtype = torch.uint8)
    fused = GGUFParameter(raw, quant_type = qtype)

    out = studio._qwen_image_21_checkpoint_to_diffusers(
        checkpoint = {"transformer_blocks.0.img_mlp.gate_up.weight": fused}
    )
    for name, rows in (("gate_layer", slice(0, 4)), ("proj", slice(4, 8))):
        half = out[f"transformer_blocks.0.img_mlp.{name}.weight"]
        assert isinstance(half, GGUFParameter), name
        assert half.quant_type == qtype, name
        assert tuple(half.quant_shape) == (4, cols), name
        assert torch.equal(half.as_tensor(), raw[rows]), name


@pytest.mark.parametrize("qtype_name", ["BF16", "Q8_0"])
def test_a_packed_norm_weight_comes_out_as_real_values(qtype_name):
    """diffusers dequantises only inside the Linear layers it swaps, and a norm reads self.weight
    directly. Measured: the public Q4_K_M keeps txt_in.text_norm in BF16 and the first step died on
    "size of tensor a (4096) must match ... b (8192)" -- the raw bytes, not the values."""
    try:
        import gguf
        from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"gguf support is not importable here: {type(exc).__name__}")
    qtype = getattr(gguf.GGMLQuantizationType, qtype_name)
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    n = 4 * block_size
    raw = torch.randint(0, 255, (n // block_size * type_size,), dtype = torch.uint8)
    packed = GGUFParameter(raw, quant_type = qtype)
    expected = dequantize_gguf_tensor(GGUFParameter(raw.clone(), quant_type = qtype))

    out = studio._qwen_image_21_checkpoint_to_diffusers(
        checkpoint = {PREFIX + "txt_in.text_norm.weight": packed}
    )["txt_in.text_norm.weight"]
    assert not hasattr(out, "quant_type")
    assert tuple(out.shape) == (n,)
    assert torch.equal(out, expected)


def test_registration_fills_the_gap_and_the_real_call_stops_refusing(monkeypatch):
    diffusers, sfm = _sfm()
    if getattr(diffusers, CLASS, None) is None:
        pytest.skip(f"this diffusers does not ship {CLASS}")
    monkeypatch.setattr(sfm, "SINGLE_FILE_LOADABLE_CLASSES", dict(sfm.SINGLE_FILE_LOADABLE_CLASSES))
    sfm.SINGLE_FILE_LOADABLE_CLASSES.pop(CLASS, None)

    assert studio._register_unregistered_single_file_classes() == (CLASS,)
    entry = sfm.SINGLE_FILE_LOADABLE_CLASSES[CLASS]
    assert entry["checkpoint_mapping_fn"] is studio._qwen_image_21_checkpoint_to_diffusers
    assert entry["default_subfolder"] == "transformer"
    # And diffusers' own lookup now resolves the class, which is the check that raised.
    assert sfm._get_single_file_loadable_mapping_class(getattr(diffusers, CLASS)) == CLASS
    # Idempotent.
    assert studio._register_unregistered_single_file_classes() == ()


def test_registration_never_overwrites_diffusers_own_entry(monkeypatch):
    diffusers, sfm = _sfm()
    upstream = {"checkpoint_mapping_fn": lambda checkpoint, **kwargs: checkpoint}
    monkeypatch.setattr(
        sfm, "SINGLE_FILE_LOADABLE_CLASSES", {**sfm.SINGLE_FILE_LOADABLE_CLASSES, CLASS: upstream}
    )
    monkeypatch.setattr(diffusers, CLASS, type(CLASS, (), {}), raising = False)

    assert studio._register_unregistered_single_file_classes() == ()
    assert sfm.SINGLE_FILE_LOADABLE_CLASSES[CLASS] is upstream


def test_a_class_this_diffusers_lacks_is_not_registered(monkeypatch):
    """Not tidiness: diffusers resolves EVERY registry entry with getattr(diffusers, name) on each
    from_single_file call, so one entry for a missing class would break every other family's
    single-file load along with it."""
    diffusers, sfm = _sfm()
    monkeypatch.setattr(sfm, "SINGLE_FILE_LOADABLE_CLASSES", dict(sfm.SINGLE_FILE_LOADABLE_CLASSES))
    assert getattr(diffusers, "NotAClassThisDiffusersShips2DModel", None) is None
    monkeypatch.setattr(
        studio,
        "_UNREGISTERED_SINGLE_FILE_CLASSES",
        {"NotAClassThisDiffusersShips2DModel": studio._qwen_image_21_checkpoint_to_diffusers},
    )

    assert studio._register_unregistered_single_file_classes() == ()
    assert "NotAClassThisDiffusersShips2DModel" not in sfm.SINGLE_FILE_LOADABLE_CLASSES


def test_the_transformer_only_single_file_branch_registers_before_loading():
    """The call site, not just the helper: registration has to precede both the prefix shim (which
    wraps whatever entry it finds) and the from_single_file it exists for."""
    import inspect

    source = inspect.getsource(studio)
    register = source.index("_register_unregistered_single_file_classes(logger)")
    shim = source.index("_install_gguf_prefix_strip(transformer_cls, logger)")
    load = source.index("transformer = transformer_cls.from_single_file(")
    assert register < shim < load
