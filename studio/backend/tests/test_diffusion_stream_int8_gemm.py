# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A block-streamed image denoiser takes the fused int8 GEMM: the speed layer installs it against the onload device
when the plan moves the denoiser, and the weights that arrive through the event-fenced prefetch run it bit-identically
to stock torchao. The CPU tests need no GPU; the last ones need NVIDIA sm80+ CUDA, Triton and torchao."""

from __future__ import annotations

import ast
import pathlib
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_int8_gemm as g8  # noqa: E402
from core.inference import diffusion_speed as ds  # noqa: E402

_INFERENCE = pathlib.Path(ds.__file__).resolve().parent


class _DiT:
    def compile_repeated_blocks(self, **kwargs):
        return None


def _record_install(monkeypatch):
    seen = []

    def _install(
        transformer,
        logger = None,
        offload_active = False,
        device = None,
    ):
        seen.append({"offload_active": offload_active, "device": device})
        return 7

    monkeypatch.setattr(g8, "install", _install)
    monkeypatch.setattr(ds, "_denoiser_dits", lambda pipe: [_DiT()])
    return seen


@pytest.mark.parametrize(
    "offload_active, denoiser_offloaded, onload, expected",
    [
        (True, True, "cuda:1", {"offload_active": False, "device": "cuda:1"}),
        (True, None, "cuda", {"offload_active": False, "device": "cuda"}),
        # no onload device (video backend, ROCm, CPU): still refused
        (True, True, None, {"offload_active": True, "device": None}),
        (True, False, "cuda", {"offload_active": False, "device": None}),
        (False, None, "cuda", {"offload_active": False, "device": None}),
    ],
)
def test_moved_denoiser_installs_against_the_onload_device(
    monkeypatch, offload_active, denoiser_offloaded, onload, expected
):
    seen = _record_install(monkeypatch)
    ds._compile_repeated_blocks(
        types.SimpleNamespace(),
        None,
        offload_active = offload_active,
        denoiser_offloaded = denoiser_offloaded,
        onload_device = onload,
    )
    assert seen == [expected]


@pytest.mark.parametrize(
    "target, expected",
    [
        (types.SimpleNamespace(device = "cuda", backend = "cuda", torch_device = "cuda:1"), "cuda:1"),
        (types.SimpleNamespace(device = "cuda", backend = "cuda", torch_device = "cuda"), "cuda"),
        (types.SimpleNamespace(device = "cuda", backend = "cuda"), "cuda"),
        (types.SimpleNamespace(device = "cuda", backend = "rocm", torch_device = "cuda"), None),
        (types.SimpleNamespace(device = "cpu", backend = "cpu", torch_device = "cpu"), None),
        (types.SimpleNamespace(device = "mps", backend = "mps", torch_device = "mps"), None),
    ],
)
def test_onload_device_is_nvidia_cuda_only(target, expected):
    assert ds._onload_device(target) == expected


def _speed_calls(path: pathlib.Path) -> list:
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None)) == "apply_speed_optims"
    ]


def test_image_backend_asks_for_the_streamed_install():
    """Both image-backend speed passes (load, and the deferred compile on the 3rd image) ask for it; the video
    backend keeps its own H3-only streamed install."""
    calls = _speed_calls(_INFERENCE / "diffusion.py")
    assert len(calls) == 2
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert (
            isinstance(kw.get("stream_int8_gemm"), ast.Constant)
            and kw["stream_int8_gemm"].value is True
        )
    for call in _speed_calls(_INFERENCE / "video.py"):
        assert "stream_int8_gemm" not in {k.arg for k in call.keywords}


def test_apply_speed_optims_passes_the_onload_device_only_when_asked(monkeypatch):
    seen = []

    def _compile(pipe, logger, **kwargs):
        seen.append(kwargs.get("onload_device"))
        return False

    monkeypatch.setattr(ds, "_compile_repeated_blocks", _compile)
    monkeypatch.setattr(ds, "compile_eligible", lambda *a, **k: True)
    monkeypatch.setattr(ds, "fp16_unet_offloaded", lambda *a, **k: False)
    monkeypatch.setattr(ds, "_denoiser_unet", lambda pipe: None)
    target = types.SimpleNamespace(device = "cuda", backend = "cuda", torch_device = "cuda:2", dtype = None)
    pipe = types.SimpleNamespace()
    for mode in (ds.SPEED_DEFAULT, ds.SPEED_MAX):
        for ask in (True, False):
            try:
                ds.apply_speed_optims(
                    pipe,
                    target,
                    is_gguf = False,
                    family = types.SimpleNamespace(supports_cuda_graph = False),
                    speed_mode = mode,
                    offload_active = True,
                    denoiser_offloaded = True,
                    stream_int8_gemm = ask,
                )
            except Exception:  # noqa: BLE001 - only the compile call is under test
                pass
    assert seen == ["cuda:2", None, "cuda:2", None]


def _cuda_ready() -> bool:
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return False
    if torch.cuda.get_device_capability() < (8, 0):
        return False
    try:
        import torchao  # noqa: F401
        import triton  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


needs_cuda = pytest.mark.skipif(
    not _cuda_ready(), reason = "needs NVIDIA sm80+ CUDA, Triton and torchao"
)


@pytest.fixture
def forced(monkeypatch):
    monkeypatch.setenv(g8.INT8_GEMM_ENV, "1")
    monkeypatch.delenv(g8.INT8_GEMM_STREAMED_ENV, raising = False)
    g8._DEVICE_CFG.clear()
    cfg = g8.device_config(torch.cuda.current_device())
    if cfg is None:
        pytest.skip("int8 GEMM probe refused this device")
    yield cfg
    g8._DEVICE_CFG.clear()


def _blocks(n, k, version):
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.blocks = torch.nn.ModuleList(
                [torch.nn.Sequential(torch.nn.Linear(k, k, bias = i % 2 == 0)) for i in range(n)]
            )

        def forward(self, x):
            for block in self.blocks:
                x = block(x) * 0.5
            return x

    holder = Holder().cuda().to(torch.bfloat16)
    cfg = Int8DynamicActivationInt8WeightConfig(set_inductor_config = False)
    if version is not None:
        if not hasattr(cfg, "version"):
            pytest.skip("torchao without config versions")
        cfg.version = version
    quantize_(holder, cfg)
    return holder.requires_grad_(False)


@needs_cuda
@pytest.mark.parametrize("version", [None, 2])
def test_prefetched_weights_run_the_fused_gemm_bit_identically(forced, version):
    """Studio's block streaming (diffusers group offload driven by the event-fenced prefetch, two groups ahead): the
    weights reach each block through the copy stream, every int8 Linear still takes the fused GEMM, and the output
    equals stock torchao on resident weights, forward after forward."""
    pytest.importorskip("diffusers.hooks")
    from core.inference import diffusion_memory as dm
    from core.inference.diffusion_offload_prefetch import module_prefetcher

    n, k = 6, 1024
    stock = _blocks(n, k, version)
    streamed = _blocks(n, k, version).cpu()
    pipe = types.SimpleNamespace(transformer = streamed, components = {"transformer": streamed})
    assert dm._apply_group_offload(pipe, "cuda", None)
    pf = module_prefetcher(streamed)
    if pf is None:
        pytest.skip("this diffusers keeps its own stream prefetch")
    assert all(lin.weight.device.type == "cpu" for block in streamed.blocks for lin in block)
    assert g8.install(streamed, device = "cuda") == n
    xs = [torch.randn(300, k, device = "cuda", dtype = torch.bfloat16) * 3 for _ in range(4)]
    with torch.no_grad():  # group offload's swap_tensors onload cannot run on inference tensors
        for x in xs:
            before = g8.call_count()
            out = streamed(x)
            assert g8.call_count() == before + n
            assert torch.equal(out, stock(x))
    assert pf.stats["prefetched"] > 0 and pf.stats["missed"] == 0
