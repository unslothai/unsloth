# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the ConvRot activation rotation (``diffusion_convrot.py``).

Real torch, CPU only, small dense Linears: the rotation is an orthogonal change of basis, so a
float32 Linear proves every property that matters here -- that the offline and online halves
cancel exactly, that the loader rotates the RECORDED set and nothing else, and that every way the
two halves could disagree is refused rather than run.

That last group is the point of the file. A rotated weight met by an unrotated activation is
finite, raises nothing, and renders quietly wrong, so there is no failure to observe downstream:
the only place it can be caught is here, at the contract.
"""

from __future__ import annotations

import contextlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

import core.inference.diffusion_prequant as pq  # noqa: E402
from core.inference.diffusion_convrot import (  # noqa: E402
    CONVROT_ATTR,
    CONVROT_KIND,
    DEFAULT_CONVROT_GROUPSIZE,
    ROTATION_FQNS_KEY,
    ROTATION_GROUP_KEY,
    ROTATION_KEY,
    apply_activation_rotation,
    build_convrot_hadamard,
    declares_rotation,
    is_power_of_four,
    is_rotated_linear,
    rotatable_fqns,
    rotate_convrot_weight_,
    rotate_linears_,
    rotation_metadata,
    rotation_metadata_error,
)


@pytest.fixture(autouse = True)
def _pin_prequant_safe_globals(real_prequant_safe_globals):
    """Apply the shared stand-in allowlist (see conftest) to every test in this module."""
    return real_prequant_safe_globals


GROUP = 16  # a power of 4, small enough to keep the test model tiny


class _Model(nn.Module):
    """Three Linears: two the group divides, one it does not."""

    def __init__(self, group: int = GROUP) -> None:
        super().__init__()
        self.a = nn.Linear(4 * group, 8)
        self.b = nn.Linear(2 * group, 8)
        self.odd = nn.Linear(group + 1, 8)


def _meta(
    fqns,
    group = GROUP,
    kind = CONVROT_KIND,
):
    return {
        ROTATION_KEY: kind,
        ROTATION_GROUP_KEY: group,
        ROTATION_FQNS_KEY: list(fqns),
    }


# ── the Hadamard itself ───────────────────────────────────────────────────────────


def test_power_of_four_gate():
    assert [n for n in (1, 2, 4, 8, 16, 32, 64, 256, 1024) if is_power_of_four(n)] == [
        4,
        16,
        64,
        256,
        1024,
    ]
    assert not is_power_of_four(0)
    assert not is_power_of_four(-4)
    assert not is_power_of_four("256")
    assert not is_power_of_four(None)


def test_hadamard_is_symmetric_and_orthogonal():
    # Both properties are what make the offline/online pair an identity: symmetric so H.T is H,
    # orthogonal so the two applications cancel.
    for size in (4, 16, 64, 256):
        h = build_convrot_hadamard(size, device = "cpu", dtype = torch.float32)
        assert h.shape == (size, size)
        assert torch.equal(h, h.T)
        assert torch.allclose(h @ h, torch.eye(size), atol = 1e-5)


def test_hadamard_rejects_a_non_power_of_four():
    with pytest.raises(ValueError):
        build_convrot_hadamard(32)


def test_denoiser_and_conditioner_share_one_hadamard():
    # The hosted conditioner (PR 8283) and the denoiser have to agree with the same comfy-kitchen
    # definition down to the normalizer. Sharing the function is how that is guaranteed rather
    # than periodically re-checked; this pins the sharing so a future copy-paste fails here.
    from core.inference import video_minimax_h3_te as te

    from core.inference import diffusion_convrot as cr

    assert te.build_convrot_hadamard is cr.build_convrot_hadamard
    assert te.rotate_convrot_activation is cr.rotate_convrot_activation


# ── the identity ──────────────────────────────────────────────────────────────────


def test_rotate_then_unrotate_is_the_identity():
    # The core invariant: rotating the weight offline and the activation online leaves the float
    # result unchanged. Everything the rotation buys happens inside the quantizer, not here.
    torch.manual_seed(0)
    linear = nn.Linear(1024, 512, dtype = torch.float32)
    x = torch.randn(37, 1024)
    reference = linear(x)

    rotate_convrot_weight_(linear, DEFAULT_CONVROT_GROUPSIZE)
    assert not torch.allclose(linear(x), reference)  # weight side alone is NOT a no-op
    linear.convrot_groupsize = DEFAULT_CONVROT_GROUPSIZE
    from core.inference.diffusion_convrot import convrot_linear_class

    linear.__class__ = convrot_linear_class()

    got = linear(x)
    assert ((got - reference).norm() / reference.norm()).item() < 1e-5


def test_rotation_survives_extra_leading_dims():
    torch.manual_seed(1)
    model = _Model()
    x = torch.randn(2, 5, 4 * GROUP)
    reference = model.a(x)
    rotate_linears_(model, ["a"], GROUP)
    assert torch.allclose(model.a(x), reference, atol = 1e-5)


def test_rotate_weight_refuses_an_indivisible_input_axis():
    with pytest.raises(ValueError):
        rotate_convrot_weight_(nn.Linear(GROUP + 1, 8), GROUP)


def test_rotatable_fqns_splits_on_divisibility():
    model = _Model()
    rotatable, not_divisible = rotatable_fqns(model, lambda m, fqn: True, GROUP)
    assert rotatable == ("a", "b")
    assert not_divisible == ("odd",)
    # A filter that rejects a Linear keeps it out of BOTH lists: it is never quantized, so there
    # is nothing for the rotation to help.
    rotatable, not_divisible = rotatable_fqns(model, lambda m, fqn: fqn != "b", GROUP)
    assert rotatable == ("a",) and not_divisible == ("odd",)


def test_offline_rotation_records_only_what_it_applied():
    model = _Model()
    rotated = rotate_linears_(model, ["a", "b"], GROUP)
    meta = rotation_metadata(GROUP, rotated)
    assert meta[ROTATION_KEY] == CONVROT_KIND
    assert meta[ROTATION_GROUP_KEY] == GROUP
    assert meta[ROTATION_FQNS_KEY] == ["a", "b"]  # sorted, so two builds agree byte for byte
    assert rotation_metadata_error(meta) is None


def test_offline_rotation_raises_rather_than_over_recording():
    model = _Model()
    with pytest.raises(ValueError):
        rotate_linears_(model, ["a", "nope"], GROUP)
    with pytest.raises(ValueError):
        rotate_linears_(model, ["a"], 32)


# ── the loader rotates exactly the recorded set ───────────────────────────────────


def test_apply_rotates_exactly_the_recorded_fqns():
    model = _Model()
    assert apply_activation_rotation(model, _meta(["a"])) == ("a",)
    assert is_rotated_linear(model.a)
    # ... and nothing else, including the other Linear the group WOULD divide. The recorded list
    # is the contract; "everything divisible" is a rule, and a rule can drift away from the
    # weights that were actually baked.
    assert not is_rotated_linear(model.b)
    assert not is_rotated_linear(model.odd)
    assert getattr(model, CONVROT_ATTR)["linears"] == 1
    assert getattr(model, CONVROT_ATTR)["group"] == GROUP


def test_apply_is_inert_without_a_declared_rotation():
    model = _Model()
    for metadata in ({}, None, {"scheme": "int8"}, {ROTATION_KEY: None}, {ROTATION_KEY: ""}):
        assert apply_activation_rotation(model, metadata) == ()
    assert not any(is_rotated_linear(m) for m in (model.a, model.b, model.odd))


def test_rotated_linear_keeps_the_state_dict_unchanged():
    # The swap must not add, rename or drop a key: the hosted checkpoint loads under strict=True.
    before = sorted(_Model().state_dict())
    model = _Model()
    apply_activation_rotation(model, _meta(["a", "b"]))
    assert sorted(model.state_dict()) == before
    assert isinstance(model.a, nn.Linear)  # still an nn.Linear, so torchao treats it as one


@pytest.mark.parametrize(
    "metadata",
    [
        _meta(["a"], kind = "convrot_hadamard_v2"),  # a kind this build does not implement
        _meta(["a"], kind = True),
        _meta(["a"], group = 32),  # not a power of 4
        _meta(["a"], group = "256"),
        _meta([]),  # declared but records nothing
        {ROTATION_KEY: CONVROT_KIND, ROTATION_GROUP_KEY: GROUP},  # no fqn key at all
        _meta(["a", "a"]),  # duplicates
        _meta(["a", 7]),  # not strings
    ],
)
def test_apply_refuses_an_unusable_contract(metadata):
    assert declares_rotation(metadata)
    assert rotation_metadata_error(metadata)
    model = _Model()
    with pytest.raises(ValueError):
        apply_activation_rotation(model, metadata)
    assert not any(is_rotated_linear(m) for m in (model.a, model.b, model.odd))


def test_apply_refuses_an_fqn_this_model_does_not_have():
    model = _Model()
    with pytest.raises(ValueError, match = "does not have"):
        apply_activation_rotation(model, _meta(["a", "missing.linear"]))


def test_apply_refuses_a_target_that_is_not_a_linear():
    model = _Model()
    model.not_a_linear = nn.LayerNorm(GROUP)
    with pytest.raises(ValueError, match = "not an nn.Linear"):
        apply_activation_rotation(model, _meta(["not_a_linear"]))


def test_apply_refuses_an_indivisible_target():
    model = _Model()
    with pytest.raises(ValueError, match = "does not divide"):
        apply_activation_rotation(model, _meta(["odd"]))


def test_apply_refuses_to_rotate_twice():
    model = _Model()
    apply_activation_rotation(model, _meta(["a"]))
    with pytest.raises(ValueError, match = "already rotated"):
        apply_activation_rotation(model, _meta(["a"]))


def test_a_refused_apply_rotates_nothing_at_all():
    # A PARTIAL install is worse than either end state: the rotated half still renders, just
    # wrongly, so nothing downstream fails. Validate every target before swapping any.
    model = _Model()
    with pytest.raises(ValueError):
        apply_activation_rotation(model, _meta(["a", "b", "odd"]))
    assert not any(is_rotated_linear(m) for m in (model.a, model.b, model.odd))


# ── the prequant checkpoint contract ──────────────────────────────────────────────


def test_format_tag_follows_the_rotation():
    assert pq.prequant_format_for({"scheme": "int8"}) == pq.PREQUANT_FORMAT
    assert pq.prequant_format_for(_meta(["a"])) == pq.PREQUANT_FORMAT_ROTATED
    assert pq.PREQUANT_FORMAT_ROTATED != pq.PREQUANT_FORMAT
    assert set(pq.PREQUANT_FORMATS) == {pq.PREQUANT_FORMAT, pq.PREQUANT_FORMAT_ROTATED}


@pytest.mark.parametrize(
    ("fmt", "metadata", "ok"),
    [
        (pq.PREQUANT_FORMAT, {}, True),
        (pq.PREQUANT_FORMAT_ROTATED, _meta(["a"]), True),
        # A rotated artifact tagged v1 loads clean on an Unsloth predating the online half and
        # renders wrong pixels. Refuse it here too: whoever wrote the tag is not to be trusted
        # about the rest of the file either.
        (pq.PREQUANT_FORMAT, _meta(["a"]), False),
        # A v2 tag with nothing to rotate: something was meant to happen and did not.
        (pq.PREQUANT_FORMAT_ROTATED, {}, False),
        (pq.PREQUANT_FORMAT_ROTATED, _meta(["a"], kind = "something_else"), False),
        (pq.PREQUANT_FORMAT_ROTATED, _meta(["a"], group = 32), False),
        (pq.PREQUANT_FORMAT_ROTATED, _meta([]), False),
    ],
)
def test_validator_enforces_the_format_rotation_biconditional(fmt, metadata, ok):
    assert pq._validate_activation_rotation(fmt, metadata, "int8", None) is ok


def test_h3_int8_resolves_to_the_rotated_artifact_with_the_plain_one_behind_it():
    # The wiring the shipped denoiser depends on: this build asks for the ConvRot artifact by
    # name, and the derived name stays as the fallback so an install predating the online half
    # still resolves the plain checkpoint instead of refusing the v2 tag and downloading 66.3 GB
    # of dense weights.
    from core.inference.video_families import detect_video_family

    fam = detect_video_family("MiniMaxAI/MiniMax-H3")
    src = pq.resolve_prequant_source(fam, "int8")
    assert src.location == "unsloth/MiniMax-H3-FP8"
    assert src.filename == "MiniMax-H3-INT8-ConvRot.pt"
    assert src.fallback_filename == "MiniMax-H3-INT8.pt"
    # fp8 is untouched: one artifact, the derived name.
    assert pq.resolve_prequant_source(fam, "fp8").filename == "MiniMax-H3-FP8.pt"


# ── end to end through the prequant loader ────────────────────────────────────────


class _FakeTransformer(nn.Module):
    """Just enough of a diffusers transformer for ``load_prequantized_transformer``."""

    calls: dict = {}

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4 * GROUP, 8)
        self.b = nn.Linear(2 * GROUP, 8)

    @classmethod
    def load_config(cls, base, **kw):
        return {"cfg": True}

    @classmethod
    def from_config(cls, config):
        return cls()

    def load_state_dict(
        self,
        sd,
        strict = True,
        assign = False,
    ):
        _FakeTransformer.calls["load_state_dict"] = {"strict": strict, "assign": assign}


def _load_rotated(monkeypatch, tmp_path, ckpt):
    """Drive ``load_prequantized_transformer`` over ``ckpt`` with real torch."""
    _FakeTransformer.calls = {}
    monkeypatch.setattr(torch, "load", lambda *a, **k: ckpt)
    accelerate = types.ModuleType("accelerate")
    accelerate.init_empty_weights = lambda: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)
    monkeypatch.setenv(pq.ALLOW_LOCAL_PREQUANT_PATH_ENV, str(tmp_path))
    path = tmp_path / "ckpt.pt"
    path.write_bytes(b"x")
    return pq.load_prequantized_transformer(
        _FakeTransformer,
        "Tongyi-MAI/Z-Image-Turbo",
        pq.PrequantSource(kind = "path", location = str(path), filename = None),
        device = "cpu",
        dtype = "bfloat16",
        hf_token = None,
        scheme = "int8",
        logger = None,
    )


def _ckpt(fmt, metadata):
    meta = {"scheme": "int8", "base_model_id": "Tongyi-MAI/Z-Image-Turbo"}
    meta.update(metadata)
    return {"format": fmt, "metadata": meta, "state_dict": {}}


def test_loader_installs_the_rotation_the_checkpoint_records(monkeypatch, tmp_path):
    loaded = _load_rotated(monkeypatch, tmp_path, _ckpt(pq.PREQUANT_FORMAT_ROTATED, _meta(["a"])))
    assert loaded is not None
    assert is_rotated_linear(loaded.a) and not is_rotated_linear(loaded.b)


def test_loader_leaves_a_plain_checkpoint_alone(monkeypatch, tmp_path):
    loaded = _load_rotated(monkeypatch, tmp_path, _ckpt(pq.PREQUANT_FORMAT, {}))
    assert loaded is not None
    assert not is_rotated_linear(loaded.a) and not is_rotated_linear(loaded.b)


@pytest.mark.parametrize(
    ("fmt", "metadata"),
    [
        # Declared but unusable, in each of the ways the loader can tell.
        (pq.PREQUANT_FORMAT_ROTATED, _meta(["a"], kind = "convrot_hadamard_v99")),
        (pq.PREQUANT_FORMAT_ROTATED, _meta(["a"], group = 32)),
        (pq.PREQUANT_FORMAT_ROTATED, _meta(["not_on_this_model"])),
        (pq.PREQUANT_FORMAT_ROTATED, _meta([])),
        (pq.PREQUANT_FORMAT, _meta(["a"])),
    ],
)
def test_loader_refuses_a_rotation_it_cannot_apply(monkeypatch, tmp_path, fmt, metadata):
    # None means "fall back to the dense download": slower and bigger, but never wrong. The
    # alternative -- loading it anyway -- has no symptom at all.
    assert _load_rotated(monkeypatch, tmp_path, _ckpt(fmt, metadata)) is None


def test_every_rotated_projection_shares_one_class():
    """The rotation is a class SWAP, so the class has to be a singleton.

    ``torch.compile`` guards each frame on ``___check_type_id`` of the modules it closes over. A
    class defined inside a function is a new class object per call, so giving each of the 350
    rotated projections its own ConvRotLinear made every one of them look like a different type
    and retraced the block it lives in: 23 recompiles against bfloat16's 1, and 178 s of
    first-call compile against 14 s. Identity, not equality: two classes with identical bodies
    still fail the guard."""
    import torch
    from torch import nn

    from core.inference.diffusion_convrot import _install_rotation, convrot_linear_class

    first, second = nn.Linear(256, 8, bias = False), nn.Linear(256, 8, bias = False)
    _install_rotation(first, 256)
    _install_rotation(second, 256)
    assert type(first) is type(second)
    assert type(first) is convrot_linear_class()
    # Still an nn.Linear, which is what keeps torchao's filter and the checkpoint keys working.
    assert isinstance(first, nn.Linear)
    assert is_rotated_linear(first) and is_rotated_linear(second)
    # And the swap is per instance, so a shared class must not leak one module's group to another.
    third = nn.Linear(512, 8, bias = False)
    _install_rotation(third, 128)
    assert (first.convrot_groupsize, third.convrot_groupsize) == (256, 128)
    assert torch.is_tensor(first.weight)
