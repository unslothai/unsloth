# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Krea-2 adapters trained on a fused ``attn.to_qkv`` (SimpleTuner): LoRA, DoRA and LoKR land on to_q / to_k / to_v."""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
diffusers = pytest.importorskip("diffusers")
try:
    from diffusers import Krea2Transformer2DModel
    from diffusers.loaders.lora_pipeline import Krea2LoraLoaderMixin
except ImportError:  # an older diffusers without Krea-2
    pytest.skip("this diffusers has no Krea-2 transformer", allow_module_level = True)

from safetensors.torch import save_file  # noqa: E402

from core.inference import diffusion_lora_files as lf  # noqa: E402

# krea/Krea-2-Raw transformer/config.json shrunk with its ratios kept: 48 query heads over 12 key/value heads in the
# image blocks (q is 4x as wide as k and v), 20 over 20 in the text fusion blocks (equal thirds).
CONFIG = dict(
    in_channels = 4,
    num_layers = 2,
    attention_head_dim = 8,
    num_attention_heads = 8,
    num_key_value_heads = 2,
    intermediate_size = 32,
    timestep_embed_dim = 16,
    text_hidden_dim = 24,
    num_text_layers = 2,
    text_num_attention_heads = 3,
    text_num_key_value_heads = 3,
    text_intermediate_size = 16,
    num_layerwise_text_blocks = 1,
    num_refiner_text_blocks = 1,
    axes_dims_rope = (2, 2, 4),
)
# (attention path, input width, q rows, k / v rows)
ATTNS = (
    ("transformer_blocks.0.attn.", 64, 64, 16),
    ("text_fusion.refiner_blocks.0.attn.", 24, 24, 24),
)
RANK = 4


class _Pipe(Krea2LoraLoaderMixin):
    def __init__(self):
        torch.manual_seed(0)
        self.transformer = Krea2Transformer2DModel(**CONFIG).float().eval()
        self.hf_device_map = None

    @property
    def components(self):
        return {"transformer": self.transformer}


def _ints(*shape, seed):
    # Small integers: every product and sum below is exact in fp32, so equality checks the row mapping, not BLAS.
    return torch.randint(-3, 4, shape, generator = torch.Generator().manual_seed(seed)).float()


@pytest.fixture
def diffusers_log():
    records: list = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())
    log = logging.getLogger("diffusers")
    log.addHandler(handler)
    yield records
    log.removeHandler(handler)


@pytest.fixture(autouse = True)
def _requires_grad_untouched(monkeypatch):
    monkeypatch.setattr(lf, "_refresh_group_offload", lambda roots: None)


def _fused_lora_file(
    path,
    dora = False,
    untrained_siblings = True,
):
    state, fused = {}, {}
    for i, (attn, width, q, kv) in enumerate(ATTNS):
        rows = q + 2 * kv
        a, b = _ints(RANK, width, seed = 10 + i), _ints(rows, RANK, seed = 20 + i)
        state[f"transformer.{attn}to_qkv.lora_A.weight"] = a
        state[f"transformer.{attn}to_qkv.lora_B.weight"] = b
        fused[attn] = {"delta": b @ a}
        if dora:
            m = _ints(rows, seed = 30 + i).abs() + 1.0
            state[f"transformer.{attn}to_qkv.lora_magnitude_vector.weight"] = m
            fused[attn]["magnitude"] = m
        if untrained_siblings:  # what a fused trainer saves for the projections it never ran
            for name, r in (("to_q", q), ("to_k", kv), ("to_v", kv)):
                state[f"transformer.{attn}{name}.lora_A.weight"] = _ints(RANK, width, seed = 40 + i)
                state[f"transformer.{attn}{name}.lora_B.weight"] = torch.zeros(r, RANK)
                if dora:  # DoRA starts from the base weight's row norm, so an unrun sibling still holds it
                    base = _Pipe().transformer.get_submodule(attn + name).weight
                    state[f"transformer.{attn}{name}.lora_magnitude_vector.weight"] = (
                        base.detach().norm(dim = 1)
                    )
    save_file(state, str(path))
    return fused


def _peft_delta(layer):
    return layer.lora_B["a"].weight @ layer.lora_A["a"].weight * layer.scaling["a"]


def _dora_weight(weight, delta, magnitude):
    merged = weight + delta
    norm = (merged * merged).sum(dim = 1).sqrt()
    return merged * (magnitude / norm)[:, None]


@pytest.mark.parametrize("dora", [False, True], ids = ["lora", "dora"])
def test_fused_qkv_lora_splits_into_exact_per_projection_deltas(tmp_path, diffusers_log, dora):
    path = tmp_path / "fused.safetensors"
    fused = _fused_lora_file(path, dora = dora)
    pipe = _Pipe()
    lf.load_adapter(pipe, str(path), "a", "krea-2")
    assert not [m for m in diffusers_log if "unexpected keys" in m]  # every key landed on a layer
    t = pipe.transformer
    for attn, *_ in ATTNS:
        layers = [t.get_submodule(attn + n) for n in ("to_q", "to_k", "to_v")]
        assert torch.equal(torch.cat([_peft_delta(l) for l in layers]), fused[attn]["delta"])
        if dora:
            mags = [l.lora_magnitude_vector["a"].weight for l in layers]
            assert torch.equal(torch.cat(mags), fused[attn]["magnitude"])
            base = torch.cat([l.base_layer.weight for l in layers])
            per = [
                _dora_weight(l.base_layer.weight, _peft_delta(l), m) for l, m in zip(layers, mags)
            ]
            whole = _dora_weight(base, fused[attn]["delta"], fused[attn]["magnitude"])
            assert torch.equal(torch.cat(per), whole)
        assert not hasattr(t.get_submodule(attn[:-1]), "to_qkv")


def test_fused_qkv_lora_alpha_scales_every_split_projection(tmp_path):
    attn, width, q, kv = ATTNS[0]
    a, b = _ints(RANK, width, seed = 1), _ints(q + 2 * kv, RANK, seed = 2)
    path = tmp_path / "alpha.safetensors"
    save_file(
        {
            f"diffusion_model.{attn}to_qkv.lora_down.weight": a,
            f"diffusion_model.{attn}to_qkv.lora_up.weight": b,
            f"diffusion_model.{attn}to_qkv.alpha": torch.tensor(2.0),
        },
        str(path),
    )
    pipe = _Pipe()
    state = lf.krea2_fused_qkv(
        pipe.transformer, lf.normalize_lora(lf.read_lora_file(str(path)), "krea-2")
    )
    pieces = [state[f"diffusion_model.{attn}{n}.lora_B.weight"] for n in ("to_q", "to_k", "to_v")]
    assert [int(p.shape[0]) for p in pieces] == [q, kv, kv]
    assert torch.equal(torch.cat(pieces), b * 0.5)
    shared = [state[f"diffusion_model.{attn}{n}.lora_A.weight"] for n in ("to_q", "to_k", "to_v")]
    assert torch.equal(shared[0], a) and shared[1] is shared[0] and shared[2] is shared[0]


def test_fused_qkv_lokr_targets_carry_row_offsets(tmp_path):
    # LyCORIS on the diffusers model: ``lycoris_`` plus the diffusers path with its dots as underscores.
    attn, width, q, kv = ATTNS[0]
    rows = q + 2 * kv
    w1, w2 = _ints(6, 4, seed = 3), _ints(rows // 6, width // 4, seed = 4)
    lyco = "lycoris_transformer_blocks_0_attn_"
    state = {
        lyco + "to_qkv.lokr_w1": w1,
        lyco + "to_qkv.lokr_w2": w2,
        lyco + "to_qkv.alpha": torch.tensor(1.0),
        lyco + "to_out_0.lokr_w1": _ints(4, 4, seed = 5),
        lyco + "to_out_0.lokr_w2": _ints(16, 16, seed = 6),
    }
    for name, r in (("to_q", q), ("to_k", kv), ("to_v", kv)):
        state[f"{lyco}{name}.lokr_w1"] = _ints(4, 4, seed = 7)
        state[f"{lyco}{name}.lokr_w2"] = torch.zeros(r // 4, width // 4)
    path = tmp_path / "lokr.safetensors"
    save_file(state, str(path))
    pipe = _Pipe()
    lf.load_adapter(pipe, str(path), "k", "krea-2")
    kron = lf.kron_linear_class()
    t = pipe.transformer
    eye = torch.eye(width)
    deltas, spans = [], []
    for name in ("to_q", "to_k", "to_v"):
        wrapper = t.get_submodule(attn + name)
        assert (
            isinstance(wrapper, kron) and len(wrapper.slots) == 1
        )  # the untrained sibling was dropped
        prefix, span = wrapper.slots[0]
        spans.append(span)
        deltas.append(
            lf._kron_delta(
                eye, getattr(wrapper, prefix + "_w1"), getattr(wrapper, prefix + "_w2"), span
            ).T
            * getattr(wrapper, prefix + "_scale")
        )
    assert spans == [(0, q), (q, q + kv), (q + kv, rows)]
    assert torch.equal(torch.cat(deltas), torch.kron(w1, w2))
    assert isinstance(t.get_submodule(attn + "to_out.0"), kron)
    assert len(lf.kron_names(pipe)) == 1 and len(pipe.__dict__[lf.KRON_ATTR]["k"]) == 4


def test_a_trained_projection_beside_a_fused_one_is_refused(tmp_path):
    attn, width, q, kv = ATTNS[0]
    path = tmp_path / "both.safetensors"
    save_file(
        {
            f"transformer.{attn}to_qkv.lora_A.weight": _ints(RANK, width, seed = 1),
            f"transformer.{attn}to_qkv.lora_B.weight": _ints(q + 2 * kv, RANK, seed = 2),
            f"transformer.{attn}to_k.lora_A.weight": _ints(RANK, width, seed = 3),
            f"transformer.{attn}to_k.lora_B.weight": _ints(kv, RANK, seed = 4),
        },
        str(path),
    )
    with pytest.raises(ValueError, match = "trained beside the fused"):
        lf.load_adapter(_Pipe(), str(path), "a", "krea-2")


@pytest.mark.parametrize("extra", ["bias", "magnitude"])
def test_a_zero_delta_sibling_with_a_trained_bias_or_magnitude_is_refused(tmp_path, extra):
    # B @ A is zero, but a bias still adds an output and a trained DoRA magnitude still rescales the base weight.
    attn, width, q, kv = ATTNS[0]
    path = tmp_path / "zero_delta.safetensors"
    state = {
        f"transformer.{attn}to_qkv.lora_A.weight": _ints(RANK, width, seed = 1),
        f"transformer.{attn}to_qkv.lora_B.weight": _ints(q + 2 * kv, RANK, seed = 2),
        f"transformer.{attn}to_k.lora_A.weight": _ints(RANK, width, seed = 3),
        f"transformer.{attn}to_k.lora_B.weight": torch.zeros(kv, RANK),
    }
    if extra == "bias":
        state[f"transformer.{attn}to_k.lora_B.bias"] = _ints(kv, seed = 4)
    else:
        state[f"transformer.{attn}to_k.lora_magnitude_vector.weight"] = torch.ones(kv)
    save_file(state, str(path))
    with pytest.raises(ValueError, match = "trained beside the fused"):
        lf.load_adapter(_Pipe(), str(path), "a", "krea-2")


def test_a_zero_delta_sibling_with_a_zero_bias_is_dropped():
    attn, width, q, kv = ATTNS[0]
    state = {
        f"transformer.{attn}to_qkv.lora_A.weight": _ints(RANK, width, seed = 1),
        f"transformer.{attn}to_qkv.lora_B.weight": _ints(q + 2 * kv, RANK, seed = 2),
        f"transformer.{attn}to_k.lora_A.weight": _ints(RANK, width, seed = 3),
        f"transformer.{attn}to_k.lora_B.weight": torch.zeros(kv, RANK),
        f"transformer.{attn}to_k.lora_B.bias": torch.zeros(kv),
    }
    kept = lf.krea2_drop_unused_projections(state, _Pipe().transformer)
    assert not any(".to_k." in k for k in kept) and len(kept) == 2


def test_a_fused_qkv_dora_scale_is_refused_not_loaded_as_plain_lora(tmp_path):
    # diffusers filters dora_scale out of a Krea-2 file, so splitting it would silently drop the trained magnitudes.
    attn, width, q, kv = ATTNS[0]
    path = tmp_path / "dora_scale.safetensors"
    save_file(
        {
            f"transformer.{attn}to_qkv.lora_A.weight": _ints(RANK, width, seed = 1),
            f"transformer.{attn}to_qkv.lora_B.weight": _ints(q + 2 * kv, RANK, seed = 2),
            f"transformer.{attn}to_qkv.dora_scale": _ints(q + 2 * kv, 1, seed = 3).abs() + 1.0,
        },
        str(path),
    )
    with pytest.raises(ValueError, match = "dora_scale is not supported"):
        lf.load_adapter(_Pipe(), str(path), "a", "krea-2")


def test_a_fused_qkv_lokr_with_the_wrong_rows_is_refused(tmp_path):
    # 192 rows against this model's 96 (64 + 16 + 16): each slice alone would pass the per-target width check.
    attn, width, q, kv = ATTNS[0]
    lyco = "lycoris_transformer_blocks_0_attn_"
    state = {
        lyco + "to_qkv.lokr_w1": _ints(6, 4, seed = 3),
        lyco + "to_qkv.lokr_w2": _ints(192 // 6, width // 4, seed = 4),
        lyco + "to_qkv.alpha": torch.tensor(1.0),
    }
    path = tmp_path / "lokr_rows.safetensors"
    save_file(state, str(path))
    pipe = _Pipe()
    with pytest.raises(
        ValueError, match = "has 192 rows, but the fused projection it splits into here has 96"
    ):
        lf.load_adapter(pipe, str(path), "k", "krea-2")
    assert not lf.kron_names(pipe)


def test_a_fused_qkv_with_the_wrong_rows_is_refused(tmp_path):
    attn, width, q, kv = ATTNS[0]
    path = tmp_path / "bad.safetensors"
    save_file(
        {
            f"transformer.{attn}to_qkv.lora_A.weight": _ints(RANK, width, seed = 1),
            f"transformer.{attn}to_qkv.lora_B.weight": _ints(
                3 * q, RANK, seed = 2
            ),  # equal thirds: no GQA
        },
        str(path),
    )
    with pytest.raises(ValueError, match = r"q / k / v here are \(64, 16, 16\)"):
        lf.load_adapter(_Pipe(), str(path), "a", "krea-2")


def test_untrained_drop_keeps_files_without_a_fused_qkv_unchanged():
    state = {
        "diffusion_model.blocks.0.attn.wq.lora_A.weight": torch.zeros(2, 4),
        "diffusion_model.blocks.0.attn.wq.lora_B.weight": torch.zeros(4, 2),
    }
    assert lf.krea2_drop_unused_projections(state) is state
    assert lf.krea2_fused_qkv(None, state) == state


def test_reference_layout_kohya_keys_still_resolve(tmp_path):
    path = tmp_path / "kohya.safetensors"
    save_file(
        {
            "lycoris_blocks_0_attn_wq.lokr_w1": _ints(4, 4, seed = 1),
            "lycoris_blocks_0_attn_wq.lokr_w2": _ints(16, 16, seed = 2),
        },
        str(path),
    )
    pipe = _Pipe()
    lf.load_adapter(pipe, str(path), "k", "krea-2")
    assert isinstance(pipe.transformer.transformer_blocks[0].attn.to_q, lf.kron_linear_class())
