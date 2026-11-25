# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An int8 image load on an fp16-only NVIDIA card (Turing) reads the hosted int8 ConvRot DiT as native W8A8 Linears in
the card's dtype instead of quantising the dense shards at load. bf16 cards, weight-only builds, families without a
hosted ConvRot build and the kill switch keep the previous routes."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.inference import diffusion_fp16_hosted_int8 as FH  # noqa: E402
from core.inference import diffusion_rocm_hosted_int8 as RH  # noqa: E402
from core.inference import diffusion_transformer_quant as tq  # noqa: E402
from core.inference.diffusion_families import detect_family  # noqa: E402

QWEN21 = ("Qwen/Qwen-Image-2.1", "qwen-image-2.1")
FLUX3 = ("black-forest-labs/flux-3-image-klein", "flux.3-klein")
T4 = types.SimpleNamespace(device = "cuda", dtype = torch.float16, ordinal = 0)
L4 = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16, ordinal = 0)


def _fam(pair = QWEN21):
    fam = detect_family(pair[0], override = pair[1])
    assert fam is not None and fam.name == pair[1]
    return fam


@pytest.fixture
def turing(monkeypatch):
    for env in (
        FH.FP16_CARD_HOSTED_INT8_ENV,
        "UNSLOTH_NATIVE_INT8_ACT",
        "UNSLOTH_NATIVE_INT8_ROT",
        "UNSLOTH_DIFFUSION_INT8_CONVROT",
    ):
        monkeypatch.delenv(env, raising = False)
    monkeypatch.setattr(tq, "torch_is_rocm", lambda: False)
    monkeypatch.setattr(tq, "is_stubbed", lambda name: False)
    monkeypatch.setattr(tq, "_capability", lambda ordinal = None: (7, 5))


@pytest.mark.parametrize("pair", [QWEN21, FLUX3])
def test_turing_w8a8_build_reads_the_hosted_file(turing, pair):
    assert FH.fp16_card_hosted_int8_reason(_fam(pair), "int8", T4) is not None


def test_unscored_families_keep_the_at_load_build(turing, monkeypatch):
    # A family outside the scored set keeps the at-load build even when it hosts a ConvRot file.
    monkeypatch.setattr(FH, "FP16_CARD_HOSTED_INT8_FAMILIES", frozenset({"qwen-image-2.1"}))
    assert FH.fp16_card_hosted_int8_reason(_fam(FLUX3), "int8", T4) is None


def test_bf16_cards_keep_their_routes(turing):
    assert FH.fp16_card_hosted_int8_reason(_fam(), "int8", L4) is None


@pytest.mark.parametrize("scheme", [None, "fp8", "nvfp4"])
def test_other_native_builds_keep_the_at_load_build(turing, scheme):
    assert FH.fp16_card_hosted_int8_reason(_fam(), scheme, T4) is None


def test_weight_only_turing_build_keeps_the_dense_shards(turing, monkeypatch):
    # The hosted file is W8A8: a weight-only build (activations kept float) is not it.
    monkeypatch.setattr(tq, "fp16_card_int8_act", lambda target: False)
    assert FH.fp16_card_hosted_int8_reason(_fam(), "int8", T4) is None


def test_family_without_a_hosted_convrot_build_is_untouched(turing, monkeypatch):
    monkeypatch.setattr(RH, "_hosted_convrot_file", lambda fam: None)
    assert FH.fp16_card_hosted_int8_reason(_fam(), "int8", T4) is None


def test_kill_switch(turing, monkeypatch):
    monkeypatch.setenv(FH.FP16_CARD_HOSTED_INT8_ENV, "0")
    assert FH.fp16_card_hosted_int8_reason(_fam(), "int8", T4) is None


def test_reader_asks_for_the_cards_dtype(turing, monkeypatch):
    from core.inference import diffusion_denoiser_prequant as DP

    source = types.SimpleNamespace(kind = "repo", location = "unsloth/Qwen-Image-2.1-FP8")
    monkeypatch.setattr(DP, "denoiser_prequant_source", lambda *a, **k: source)
    seen = {}
    module = types.SimpleNamespace(
        _unsloth_prequant_path = "/hub/snap/Qwen-Image-2.1-INT8-ConvRot.safetensors"
    )

    def reader(*a, **k):
        seen.update(k)
        return module

    monkeypatch.setattr(RH, "load_rocm_hosted_transformer", reader)
    got = FH.fp16_card_hosted_int8_transformer(
        _fam(), T4, object, base = QWEN21[0], config_source = "/snap"
    )
    assert got is module
    assert seen["compute_dtype"] is torch.float16
    assert seen["config_source"] == "/snap"
    assert seen["tag"] == "diffusion.fp16_hosted_int8"
    # Cache only: the download plan staged the file, so the load never pulls bytes it did not account for.
    assert seen["local_files_only"] is True
    # The resolved status names the checkpoint that ran.
    assert (
        got._unsloth_prequant_artifact
        == "prequant:unsloth/Qwen-Image-2.1-FP8/Qwen-Image-2.1-INT8-ConvRot.safetensors"
    )


def test_constructor_built_buffers_keep_their_dtype():
    """A float16 ``from_pretrained`` never casts a non-persistent buffer (Qwen-Image-2.1's float32 timestep
    ``freqs``); the hosted read must not either, while stored tensors follow the compute dtype."""
    from core.inference.diffusion_native_hosted import _hold_in_compute_dtype

    class Proj(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "freqs", torch.arange(4, dtype = torch.float32) / 3, persistent = False
            )
            self.register_buffer("shift", torch.ones(4, dtype = torch.bfloat16))
            self.linear = torch.nn.Linear(4, 4, dtype = torch.bfloat16)

    model = Proj()
    want = model.freqs.clone()
    _hold_in_compute_dtype(model, torch.float16, set())
    assert model.freqs.dtype is torch.float32 and torch.equal(model.freqs, want)
    assert model.shift.dtype is torch.float16
    assert model.linear.weight.dtype is torch.float16


def _oom(*_a, **_k):
    raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")


def test_an_allocation_failure_is_not_an_unusable_file(monkeypatch, tmp_path):
    """Out of memory while building the hosted model surfaces through every layer: the dense load + quantise it
    would fall back to needs more."""
    from core.inference import diffusion_denoiser_prequant as DP
    from core.inference import diffusion_native_hosted as NH

    monkeypatch.setattr(NH, "hosted_metadata", _oom)
    with pytest.raises(torch.OutOfMemoryError):
        NH.load_native_from_hosted(
            str(tmp_path / "x.safetensors"), object, "/cfg", spec = {}, family = None
        )

    monkeypatch.setattr(RH, "rocm_convrot_source", _oom)
    source = types.SimpleNamespace(kind = "repo", location = "unsloth/Qwen-Image-2.1-FP8")
    with pytest.raises(torch.OutOfMemoryError):
        RH.load_rocm_hosted_transformer(_fam(), object, QWEN21[0], source)

    monkeypatch.setattr(DP, "denoiser_prequant_source", _oom)
    with pytest.raises(torch.OutOfMemoryError):
        FH.fp16_card_hosted_int8_transformer(_fam(), T4, object, base = QWEN21[0], config_source = None)


def test_a_broken_file_still_keeps_the_at_load_build(monkeypatch, tmp_path):
    from core.inference import diffusion_native_hosted as NH

    def _broken(*_a, **_k):
        raise RuntimeError("header too large")

    monkeypatch.setattr(NH, "hosted_metadata", _broken)
    assert (
        NH.load_native_from_hosted(
            str(tmp_path / "x.safetensors"), object, "/cfg", spec = {}, family = None
        )
        is None
    )


def test_an_online_plan_stages_the_hosted_file_instead_of_the_dense_dit(turing, monkeypatch):
    """The download plan names the file the load reads: on this route the hosted ConvRot file replaces the DiT."""
    from core.inference import diffusion as D
    from core.inference import diffusion_prequant as PQ

    convrot = types.SimpleNamespace(
        kind = "repo",
        location = "unsloth/Qwen-Image-2.1-FP8",
        filename = "Qwen-Image-2.1-INT8-ConvRot.safetensors",
    )
    monkeypatch.setattr(RH, "rocm_convrot_source", lambda source: convrot)
    monkeypatch.setattr(PQ, "hosted_prequant_refusal", lambda *a, **k: None)
    from core.inference import diffusion_denoiser_prequant as DP

    monkeypatch.setattr(
        DP,
        "denoiser_prequant_source",
        lambda *a, **k: types.SimpleNamespace(kind = "repo", location = convrot.location),
    )
    monkeypatch.setattr(D.DiffusionBackend, "_target_for_ordinal", lambda self, fam, ordinal: T4)
    monkeypatch.setattr(
        D.DiffusionBackend,
        "_prequant_source_hub_entry",
        staticmethod(lambda source, *a, **k: (source.location, source.filename, 7_258_370_288)),
    )
    backend = D.DiffusionBackend()
    kwargs = {
        "base_repo": QWEN21[0],
        "transformer_quant": "int8",
        "_pipeline_prequant_planned": None,
    }
    assert backend._dit_prequant_plan_source(_fam(), "pipeline", None, kwargs) == (
        "unsloth/Qwen-Image-2.1-FP8",
        "Qwen-Image-2.1-INT8-ConvRot.safetensors",
        7_258_370_288,
    )
    # A bf16 card or the kill switch keeps the dense shards in the plan.
    monkeypatch.setattr(D.DiffusionBackend, "_target_for_ordinal", lambda self, fam, ordinal: L4)
    assert backend._dit_prequant_plan_source(_fam(), "pipeline", None, kwargs) is None
    monkeypatch.setattr(D.DiffusionBackend, "_target_for_ordinal", lambda self, fam, ordinal: T4)
    monkeypatch.setenv(FH.FP16_CARD_HOSTED_INT8_ENV, "0")
    assert backend._dit_prequant_plan_source(_fam(), "pipeline", None, kwargs) is None


def test_no_hosted_source_keeps_the_at_load_build(turing, monkeypatch):
    from core.inference import diffusion_denoiser_prequant as DP
    monkeypatch.setattr(DP, "denoiser_prequant_source", lambda *a, **k: None)
    assert (
        FH.fp16_card_hosted_int8_transformer(_fam(), T4, object, base = QWEN21[0], config_source = None)
        is None
    )


_CPU_ALLOC = (
    "DefaultCPUAllocator: can't allocate memory: you tried to allocate 1073741824 bytes. "
    "Error code 12 (Cannot allocate memory)"
)


def test_the_cpu_allocators_runtime_error_is_an_allocation_failure(monkeypatch, tmp_path):
    """PyTorch's host allocator raises a plain RuntimeError (errno 12): it must surface, not fall back to the larger
    dense load, from the reader and from the fingerprint check alike."""
    from core.inference import diffusion_native_hosted as NH
    from core.inference import diffusion_prequant as PQ

    def _host_oom(*_a, **_k):
        raise RuntimeError(_CPU_ALLOC)

    assert NH.is_allocation_failure(RuntimeError(_CPU_ALLOC))
    assert NH.is_allocation_failure(MemoryError())
    assert not NH.is_allocation_failure(RuntimeError("header too large"))

    monkeypatch.setattr(NH, "hosted_metadata", _host_oom)
    with pytest.raises(RuntimeError, match = "can't allocate memory"):
        NH.load_native_from_hosted(
            str(tmp_path / "x.safetensors"), object, "/cfg", spec = {}, family = None
        )

    monkeypatch.setattr(PQ, "_fingerprint_mode", lambda: "full")
    monkeypatch.setattr(PQ, "canonical_weight_fingerprint", _oom)
    meta = {"fingerprint": {PQ.CANONICAL_FINGERPRINT_KEY: {"a.weight": "x"}}}
    with pytest.raises(torch.OutOfMemoryError):
        PQ._verify_packed_fingerprint({}, meta)


@pytest.mark.parametrize(
    "base",
    [
        "Qwen/Qwen-Image-2.1-Turbo",  # a sibling distill, not scored on a T4
        "someone/Qwen-Image-2.1",  # a Hub fine-tune
        "LOCAL",  # an edited local copy named like the release
    ],
)
def test_only_the_released_repo_takes_the_hosted_dit(turing, monkeypatch, tmp_path, base):
    """The hosted file stands in for the released DiT only: a local or fine-tuned pipeline keeps its own weights."""
    from core.inference import diffusion_denoiser_prequant as DP

    if base == "LOCAL":
        local = tmp_path / "my-edited" / "Qwen-Image-2.1"
        local.mkdir(parents = True)
        base = str(local)
    source = types.SimpleNamespace(kind = "repo", location = "unsloth/Qwen-Image-2.1-FP8")
    monkeypatch.setattr(DP, "denoiser_prequant_source", lambda *a, **k: source)
    called = []
    monkeypatch.setattr(
        RH,
        "load_rocm_hosted_transformer",
        lambda *a, **k: called.append(1) or types.SimpleNamespace(),
    )
    assert (
        FH.fp16_card_hosted_int8_transformer(_fam(), T4, object, base = base, config_source = base)
        is None
    )
    assert called == []
    assert FH.fp16_card_hosted_int8_plan_source(_fam(), "int8", T4, base = base) is None
    for official in ("Qwen/Qwen-Image-2.1", "unsloth/Qwen-Image-2.1"):
        assert FH._official_hub_base(_fam(), official)
