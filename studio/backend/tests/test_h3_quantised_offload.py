# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 keeps its quantised denoiser on a small card: the fit picks resident, pinned or streamed."""

from __future__ import annotations

import copy
import types

import pytest

torch = pytest.importorskip("torch")


def _h3_family():
    from core.inference.video_families import detect_video_family

    return detect_video_family("minimax-h3")


@pytest.fixture(autouse = True)
def _hosted_checkpoint_readable(monkeypatch):
    import core.inference.diffusion_prequant as pq

    monkeypatch.setattr(pq, "restricted_prequant_load_supported", lambda *a, **k: True)
    monkeypatch.setattr(pq, "torchao_group_offload_supported", lambda: True)




def _precision(vid, fam, tq, te_scheme, free_gb, monkeypatch):
    if tq in ("int8", "fp8"):
        return tq
    if tq == "none":
        return None
    monkeypatch.setattr(vid, "_h3_auto_precision_ok", lambda target = None: True, raising = False)
    monkeypatch.setattr(vid, "_h3_free_device_bytes", lambda device: int(free_gb * 1000**3))
    return vid._h3_auto_denoiser_scheme(
        fam,
        target = None,
        dtype = torch.bfloat16,
        device = "cuda",
        te_scheme = te_scheme,
        task = "fl2va",
        base_repo = fam.base_repo,
    )


# (free GB, conditioner, request) -> (engaged denoiser, tier). Quantised denoiser 20.3 GB, released 66.3 GB,
# conditioner 27.2 GB int8 / 66.7 GB dense, VAEs 11.1 GB, plus the default-shape activation headroom.
_LADDER = {
    (24, "int8", "auto"): ("int8", "stream"),
    (32, "int8", "auto"): ("int8", "stream"),
    (48, "int8", "auto"): ("int8", "stream"),
    (64, "int8", "auto"): ("int8", "pinned"),
    (80, "int8", "auto"): ("int8", "resident"),
    (141, "int8", "auto"): ("int8", "resident"),
    (24, None, "auto"): ("int8", "stream"),
    (32, None, "auto"): ("int8", "stream"),
    (48, None, "auto"): ("int8", "stream"),
    (64, None, "auto"): ("int8", "stream"),
    (80, None, "auto"): ("int8", "stream"),
    (141, None, "auto"): ("int8", "resident"),
    (24, "int8", "none"): (None, "rotation"),
    (32, "int8", "none"): (None, "rotation"),
    (48, "int8", "none"): (None, "rotation"),
    (64, "int8", "none"): (None, "rotation"),
    (80, "int8", "none"): (None, "rotation"),
    (141, "int8", "none"): (None, "resident"),
    (24, None, "none"): (None, "rotation"),
    (80, None, "none"): (None, "rotation"),
    (141, None, "none"): (None, "rotation"),
}


@pytest.mark.parametrize("free_gb", [24, 32, 48, 64, 80, 141])
@pytest.mark.parametrize("te_scheme", ["int8", None])
@pytest.mark.parametrize("tq", ["auto", "int8", "fp8", "none"])
def test_the_placement_ladder(monkeypatch, free_gb, te_scheme, tq):
    from core.inference import video as vid

    fam = _h3_family()
    engaged = _precision(vid, fam, tq, te_scheme, free_gb, monkeypatch)
    assert engaged == (None if tq == "none" else ("int8" if tq == "auto" else tq))

    from core.inference.video_minimax_h3 import h3_transformer_resident_gb

    denoiser = int(h3_transformer_resident_gb(engaged) * 1000**3)
    whole = vid._h3_planned_denoiser_bytes(fam, te_scheme = te_scheme, dtype = torch.bfloat16)
    pinned = vid._h3_planned_denoiser_bytes(
        fam, te_scheme = te_scheme, dtype = torch.bfloat16, rotating = True
    )
    tier = vid._h3_placement_tier(
        quantised = engaged is not None,
        whole_set_sizes = (denoiser, whole[1]),
        pinned_sizes = (denoiser, pinned[1]),
        free_bytes = int(free_gb * 1000**3),
        speed_off = False,
    )
    expected = _LADDER.get((free_gb, te_scheme, "none" if engaged is None else "auto"))
    if expected is not None:
        assert tier == expected[1]
    if engaged is not None:
        assert tier in ("resident", "pinned", "stream")


def test_speed_off_and_unreadable_cards_keep_their_placements():
    from core.inference import video as vid

    sizes = (20_300_000_000, 36_000_000_000)
    whole = (20_300_000_000, 47_000_000_000)
    assert (
        vid._h3_placement_tier(
            quantised = True, whole_set_sizes = whole, pinned_sizes = sizes,
            free_bytes = 500 * 1000**3, speed_off = True,
        )
        == "pinned"
    )
    assert (
        vid._h3_placement_tier(
            quantised = False, whole_set_sizes = whole, pinned_sizes = sizes,
            free_bytes = 500 * 1000**3, speed_off = True,
        )
        == "rotation"
    )
    for quantised, tier in ((True, "pinned"), (False, "rotation")):
        assert (
            vid._h3_placement_tier(
                quantised = quantised, whole_set_sizes = whole, pinned_sizes = sizes,
                free_bytes = None, speed_off = False,
            )
            == tier
        )


def test_the_quantised_denoiser_is_sized_on_its_payload_not_its_logical_shape():
    """A torchao weight reports the bf16 shape through numel / element_size, which priced the
    20.3 GB hosted denoiser at ~40 GB and pushed a card that could pin it onto the streamed tier."""
    from core.inference.diffusion_prequant import tensor_payload_bytes

    class _Inner:
        def __init__(self, n):
            self._n = n

        def numel(self):
            return self._n

        def element_size(self):
            return 1

    class QuantisedWeight:
        def __init__(self):
            self.qdata = _Inner(1000)
            self.scale = types.SimpleNamespace(numel = lambda: 10, element_size = lambda: 4)

        def __tensor_flatten__(self):
            return ["qdata", "scale"], None

        def numel(self):
            return 1000

        def element_size(self):
            return 2

    assert tensor_payload_bytes(QuantisedWeight()) == 1000 + 40
    assert tensor_payload_bytes(torch.empty(10, dtype = torch.bfloat16)) == 20




def test_a_32gb_card_is_admitted_with_the_streamed_quantised_denoiser():
    from core.inference.video_minimax_h3 import estimate_h3_diffusers_vram_gb

    free_5090_gb = 33.56
    streamed = estimate_h3_diffusers_vram_gb(
        960, 544, 124, text_encoder_gb = 27.2, transformer_gb = 20.3, transformer_streamed = True
    )
    assert streamed + 0 <= free_5090_gb + 0.25
    assert streamed == pytest.approx(29.0, abs = 0.01)
    pinned = estimate_h3_diffusers_vram_gb(
        960, 544, 124, text_encoder_gb = 27.2, transformer_gb = 20.3, transformer_pinned = True
    )
    rotating_bf16 = estimate_h3_diffusers_vram_gb(960, 544, 124, text_encoder_gb = 27.2)
    assert pinned > free_5090_gb + 0.25 and rotating_bf16 > free_5090_gb + 0.25
    assert estimate_h3_diffusers_vram_gb(960, 544, 124) == pytest.approx(73.68, abs = 0.02)
    long_clip = estimate_h3_diffusers_vram_gb(
        1344, 768, 345, text_encoder_gb = 27.2, transformer_gb = 20.3, transformer_streamed = True
    )
    assert long_clip > streamed


def test_the_host_floor_counts_the_pinned_staging_copy_of_a_streamed_denoiser():
    """Group offloading stages the streamed denoiser in pinned host memory beside the copy it was
    built as: measured 80.2 GB peak host RSS against the 64.5 GB a single count asks for."""
    from core.inference.video_minimax_h3 import estimate_h3_diffusers_host_ram_gb

    single = estimate_h3_diffusers_host_ram_gb(33.5, text_encoder_gb = 27.2, transformer_gb = 20.3)
    streamed = estimate_h3_diffusers_host_ram_gb(
        33.5, text_encoder_gb = 27.2, transformer_gb = 20.3, transformer_streamed = True
    )
    assert single == pytest.approx(64.5, abs = 0.01)
    assert streamed == pytest.approx(84.8, abs = 0.01) and streamed >= 80.2
    assert estimate_h3_diffusers_host_ram_gb(33.5) == pytest.approx(150.0, abs = 0.01)


def test_the_generate_preflight_reads_the_streamed_fact_off_the_state():
    import inspect

    from core.inference import video as vid

    assert "denoiser_streamed" in {f for f in vid._VideoLoadState.__dataclass_fields__}
    source = inspect.getsource(vid.VideoBackend.generate)
    assert source.count('transformer_streamed = bool(getattr(state, "denoiser_streamed", False))') == 2




class _Net(torch.nn.Module):
    def __init__(self, d = 256, n = 4):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(torch.nn.Linear(d, d), torch.nn.GELU(), torch.nn.Linear(d, d))
                for _ in range(n)
            ]
        )
        self.proj_out = torch.nn.Linear(d, d)

    def pad_small_m(self):
        # What the hosted H3 load does to its small-M int8 linears: a wrapper that forwards ``weight`` to the Linear.
        from core.inference.diffusion_quant_pad import PadToMinM

        self.proj_out = PadToMinM(self.proj_out)
        return self

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return self.proj_out(x)


class _UserHook:
    def __init__(self, model):
        self.model = model
        self.hook = types.SimpleNamespace(other_hooks = [])
        self.removed = False

    def remove(self):
        self.removed = True

    def offload(self):
        self.model.to("cpu")


def _torchao_configs():
    torchao = pytest.importorskip("torchao")
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Int8DynamicActivationInt8WeightConfig,
        PerRow,
    )

    configs = {}
    import inspect

    int8_params = inspect.signature(Int8DynamicActivationInt8WeightConfig).parameters
    if "version" in int8_params:
        configs["int8_v2"] = lambda: Int8DynamicActivationInt8WeightConfig(version = 2)
        try:
            Int8DynamicActivationInt8WeightConfig(version = 1)
            from torchao.quantization import linear_activation_quantized_tensor  # noqa: F401

            configs["int8_v1"] = lambda: Int8DynamicActivationInt8WeightConfig(version = 1)
        except Exception:  # noqa: BLE001 -- torchao >= 0.18 removed the v1 classes
            pass
    else:
        configs["int8_v1"] = lambda: Int8DynamicActivationInt8WeightConfig()
    if torch.cuda.get_device_capability() >= (8, 9):
        configs["fp8_row"] = lambda: Float8DynamicActivationFloat8WeightConfig(
            granularity = PerRow()
        )
    del torchao
    return configs


def _stream(module, stream_prequantized_module):
    rotating = torch.nn.Linear(8, 8).to("cuda")
    hooks = [_UserHook(module), _UserHook(rotating)]
    manager = types.SimpleNamespace(model_hooks = list(hooks))
    mode = stream_prequantized_module(manager, module, "cuda")
    assert hooks[0].removed and [h.model for h in manager.model_hooks] == [rotating]
    return mode, rotating


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
@pytest.mark.parametrize("grad_mode", ["inference_mode", "no_grad"])
def test_a_streamed_torchao_denoiser_matches_the_resident_one_bit_for_bit(monkeypatch, grad_mode):
    """Studio renders under inference_mode, and under no_grad where a torchao denoiser streams."""
    pytest.importorskip("diffusers")
    pytest.importorskip("torchao")
    monkeypatch.undo()
    from torchao.quantization import quantize_

    import core.inference.prequant_legacy_int8 as legacy
    from core.inference.diffusion_prequant import (
        stream_prequantized_module,
        torchao_group_offload_supported,
    )

    if not torchao_group_offload_supported():
        pytest.skip("this diffusers cannot group-offload torchao weights")
    render = torch.inference_mode if grad_mode == "inference_mode" else torch.no_grad

    torch.manual_seed(0)
    base = _Net().to(torch.bfloat16)
    x = torch.randn(64, 256, dtype = torch.bfloat16, device = "cuda")
    ran = []
    for name, config in _torchao_configs().items():
        resident = copy.deepcopy(base)
        quantize_(resident, config())
        resident.pad_small_m()
        v1 = type(resident.blocks[0][0].weight).__name__ == "LinearActivationQuantizedTensor"
        if v1:
            assert legacy.convert_legacy_int8_weights(resident) > 0
        resident.to("cuda")
        with render():
            expected = resident(x)
        del resident

        streamed = copy.deepcopy(base)
        quantize_(streamed, config())  # built on the CPU, like the seeded H3 load
        streamed.pad_small_m()
        mode, rotating = _stream(streamed, stream_prequantized_module)
        assert mode == "stream", (name, mode)
        with render():
            outs = [streamed(x) for _ in range(3)]
        for out in outs:
            assert torch.equal(out, expected), name
        assert rotating.weight.device.type == "cpu"
        ran.append((name, type(streamed.blocks[0][0].weight).__name__, mode))
    print("RAN", ran)
    assert ran


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_a_v1_int8_denoiser_that_cannot_be_rebuilt_moves_synchronously_and_exactly(monkeypatch):
    """The fallback when the Int8Tensor rebuild declines: synchronous copies, same values as the
    resident v1 module, including under inference_mode (the first thing it used to fail on)."""
    pytest.importorskip("diffusers")
    pytest.importorskip("torchao")
    monkeypatch.undo()
    from torchao.quantization import quantize_

    import core.inference.prequant_legacy_int8 as legacy
    from core.inference.diffusion_prequant import (
        stream_prequantized_module,
        torchao_group_offload_supported,
    )

    configs = _torchao_configs()
    if "int8_v1" not in configs or not torchao_group_offload_supported():
        pytest.skip("needs torchao <= 0.17 (v1 int8) and torchao-aware group offload")
    monkeypatch.setattr(legacy, "convert_legacy_int8_weights", lambda module: 0)
    torch.manual_seed(0)
    base = _Net().to(torch.bfloat16)
    x = torch.randn(64, 256, dtype = torch.bfloat16, device = "cuda")
    resident = copy.deepcopy(base)
    quantize_(resident, configs["int8_v1"]())
    resident.to("cuda")
    with torch.inference_mode():
        expected = resident(x)
    streamed = copy.deepcopy(base)
    quantize_(streamed, configs["int8_v1"]())
    mode, _ = _stream(streamed, stream_prequantized_module)
    assert mode == "sync"
    assert type(streamed.blocks[0][0].weight).__name__ == "LinearActivationQuantizedTensor"
    with torch.inference_mode():
        for _ in range(2):
            assert torch.equal(streamed(x), expected)
    assert streamed.blocks[-1][0].weight.device.type == "cpu"


def test_a_streamed_torchao_render_runs_under_no_grad_and_the_rest_keep_inference_mode():
    """torchao's aliasing check fails a weight move under inference_mode, and a streamed denoiser
    moves weights inside the render; the image loader switches for the same reason."""
    import inspect

    from core.inference import video as vid

    source = inspect.getsource(vid.VideoBackend.generate)
    assert (
        'if state.transformer_quant and getattr(state, "denoiser_streamed", False)\n'
        "                    else torch.inference_mode()"
    ) in source
    assert "with grad_ctx, protect_ctx, progress_ctx(), sigma_ctx:" in source


def test_a_failed_streaming_setup_raises_instead_of_pinning(monkeypatch):
    """Streaming is only chosen when pinning does not fit, so a failed setup must not fall back to pinning."""
    pytest.importorskip("diffusers")
    import diffusers.hooks
    import core.inference.diffusion_prequant as prequant

    def _boom(*args, **kwargs):
        raise ValueError("no group offload here")

    monkeypatch.setattr(prequant, "torchao_group_offload_supported", lambda: True)
    monkeypatch.setattr(diffusers.hooks, "apply_group_offloading", _boom)
    module = torch.nn.Linear(8, 8)
    hook = _UserHook(module)
    manager = types.SimpleNamespace(model_hooks = [hook])
    with pytest.raises(RuntimeError, match = "group offloading could not be set up"):
        prequant.stream_prequantized_module(manager, module, "cpu")
    assert hook.removed and module.weight.device.type == "cpu"
