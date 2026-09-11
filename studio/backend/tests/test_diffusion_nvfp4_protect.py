# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the NVFP4 per-step precision lever (W4A16 at protected denoising steps)."""

from __future__ import annotations

import os
import types

import pytest

from core.inference import diffusion_nvfp4_linear as nl
from core.inference import diffusion_nvfp4_ops as ops
from core.inference import diffusion_nvfp4_protect as pr

REAL_SHAPES = ((3072, 3072), (18432, 3072), (15360, 256), (5120, 3072))


class _CapableLayer:
    """A stand-in for a converted flashinfer Linear: weak-referenceable, which is all the
    controller's registry asks of it."""


def _capable(ctl):
    """Register one, which is what makes a controller willing to arm. The caller must keep the
    returned object alive: the registry is weak."""
    layer = _CapableLayer()
    ctl.register_layer(layer)
    return layer


def _cuda_or_skip():
    torch = pytest.importorskip("torch")
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        pytest.skip("CUDA devices are masked off for this process")
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    capability = tuple(torch.cuda.get_device_capability(0))
    if capability not in ops.NVFP4_FLASHINFER_CAPS:
        pytest.skip("sm_%d%d has no flashinfer NVFP4 kernels" % capability)
    pytest.importorskip("flashinfer")
    pytest.importorskip("torchao")
    return torch


@pytest.mark.parametrize(
    "spec,total,want",
    [
        ("0,1,2,3,-1", 50, (0, 1, 2, 3, 49)),
        ("0,-1", 50, (0, 49)),
        ("0,1,2,3,4,5,6,7,-1", 50, (0, 1, 2, 3, 4, 5, 6, 7, 49)),
        ("auto", 50, (0, 1, 2, 3, 49)),
        ("auto", 8, (0, 7)),
        ("auto", 100, tuple(range(8)) + (99,)),
        ("auto", 1, (0,)),
        (" 0 , 0 , -1 , 49 ", 50, (0, 49)),
        ("", 50, ()),
        ("off", 50, ()),
        ("none", 50, ()),
        (None, 50, ()),
        ("0,1,2,3,-1", 4, (0, 1, 2, 3)),
        ("0,1,2,3,-1", 2, (0, 1)),
        ("60", 50, ()),
        ("0", 0, ()),
    ],
)
def test_the_schedule_parses(spec, total, want):
    assert pr.parse_protect_steps(spec, total) == want


def test_a_typo_in_the_schedule_raises_rather_than_protecting_nothing():
    with pytest.raises(ValueError) as excinfo:
        pr.parse_protect_steps("0,fisrt,-1", 50)
    assert "not a step index" in str(excinfo.value)
    assert pr.PROTECT_STEPS_ENV in str(excinfo.value)


def test_the_env_reader_treats_blank_and_off_as_off(monkeypatch):
    monkeypatch.delenv(pr.PROTECT_STEPS_ENV, raising = False)
    assert pr.protect_steps_env() == ""
    monkeypatch.setenv(pr.PROTECT_STEPS_ENV, "  ")
    assert pr.protect_steps_env() == ""
    monkeypatch.setenv(pr.PROTECT_STEPS_ENV, "OFF")
    assert pr.protect_steps_env() == ""
    monkeypatch.setenv(pr.PROTECT_STEPS_ENV, " 0,1,2,3,-1 ")
    assert pr.protect_steps_env() == "0,1,2,3,-1"
    monkeypatch.setenv(pr.PROTECT_STEPS_ENV, "AUTO")
    assert pr.protect_steps_env() == "auto"


def test_an_unarmed_controller_never_protects():
    ctl = pr.NVFP4StepController("")
    assert ctl.armed is False
    assert ctl.begin(50) == ()
    for _ in range(50):
        assert ctl.protected is False
        ctl.advance()
    assert ctl.protected_steps_seen == 0


def test_the_controller_protects_exactly_the_named_steps():
    ctl = pr.NVFP4StepController("0,1,2,3,-1")
    assert ctl.armed is True
    assert ctl.begin(50) == (0, 1, 2, 3, 49)
    fired = []
    for step in range(50):
        assert ctl.index == step
        if ctl.protected:
            fired.append(step)
        ctl.advance()
    assert fired == [0, 1, 2, 3, 49]
    assert ctl.protected_steps_seen == 5
    assert ctl.protected is False


def test_reset_leaves_a_controller_protecting_nothing():
    ctl = pr.NVFP4StepController("auto")
    ctl.begin(50)
    assert ctl.protected is True
    ctl.reset()
    assert (ctl.protected, ctl.index, ctl.steps, ctl.total) == (False, 0, (), 0)
    assert ctl.armed is True


def test_begin_clears_the_previous_generation():
    ctl = pr.NVFP4StepController("-1")
    ctl.begin(4)
    for _ in range(3):
        ctl.advance()
    assert (ctl.index, ctl.protected) == (3, True)
    assert ctl.begin(50) == (49,)
    assert (ctl.index, ctl.protected, ctl.generations) == (0, False, 2)


def test_a_bad_schedule_is_logged_and_costs_no_render():
    class _Logger:
        def __init__(self):
            self.text = ""

        def warning(self, message, *args):
            self.text += (message % args if args else message) + "\n"

        def info(self, message, *args):
            self.text += (message % args if args else message) + "\n"

    logger = _Logger()
    ctl = pr.NVFP4StepController("0,oops")
    assert ctl.begin(50, logger = logger) == ()
    assert "protect schedule ignored" in logger.text
    assert ctl.protected is False


def test_configure_rearms_and_disarms():
    ctl = pr.NVFP4StepController("")
    assert ctl.armed is False
    ctl.configure("0,-1")
    assert (ctl.armed, ctl.spec) == (True, "0,-1")
    ctl.configure("off")
    assert (ctl.armed, ctl.spec) == (False, "")


class _FakeScheduler:
    def __init__(self) -> None:
        self.calls = 0

    def step(self, *args, **kwargs):
        self.calls += 1
        return "stepped"


class _FakePipe:
    """A denoise loop the shape every diffusers pipeline has: forward, then scheduler.step."""

    def __init__(self) -> None:
        self.scheduler = _FakeScheduler()

    def run(self, steps: int, observe) -> None:
        for _ in range(steps):
            observe()  # the transformer forward for this step
            self.scheduler.step()


def test_the_scheduler_wrapper_gives_the_forward_the_right_step_index():
    ctl = pr.NVFP4StepController("0,1,2,3,-1")
    layer = _capable(ctl)
    pipe = _FakePipe()
    seen: list = []
    with pr.protect_generation(pipe, 50, controller = ctl):
        pipe.run(50, lambda: seen.append((ctl.index, ctl.protected)))
    assert [index for index, _ in seen] == list(range(50))
    assert [index for index, protected in seen if protected] == [0, 1, 2, 3, 49]
    assert pipe.scheduler.calls == 50


def test_the_wrapper_is_removed_and_the_controller_reset_afterwards():
    """Removed by DELETING the instance attribute it added, not by assigning the method back."""
    ctl = pr.NVFP4StepController("0")
    layer = _capable(ctl)
    pipe = _FakePipe()
    assert "step" not in pipe.scheduler.__dict__
    with pr.protect_generation(pipe, 8, controller = ctl):
        assert "step" in pipe.scheduler.__dict__
        assert ctl.protected is True
    assert "step" not in pipe.scheduler.__dict__
    assert (ctl.protected, ctl.steps, ctl.index) == (False, (), 0)


def test_the_wrapper_is_removed_when_the_denoise_loop_raises():
    ctl = pr.NVFP4StepController("0,-1")
    layer = _capable(ctl)
    pipe = _FakePipe()
    with pytest.raises(RuntimeError):
        with pr.protect_generation(pipe, 8, controller = ctl):
            pipe.scheduler.step()
            raise RuntimeError("cancelled")
    assert "step" not in pipe.scheduler.__dict__
    assert ctl.protected is False


def test_an_unarmed_controller_wraps_nothing_at_all():
    """Default OFF has to be free: no wrapper on the scheduler at all."""
    ctl = pr.NVFP4StepController("")
    pipe = _FakePipe()
    with pr.protect_generation(pipe, 50, controller = ctl):
        assert "step" not in pipe.scheduler.__dict__
        pipe.run(50, lambda: None)
    assert ctl.protected is False


def test_a_pipeline_with_no_scheduler_protects_nothing():
    class _NoScheduler:
        scheduler = None

    class _Logger:
        text = ""

        def warning(self, message, *args):
            type(self).text += message % args if args else message

    ctl = pr.NVFP4StepController("0,-1")
    layer = _capable(ctl)
    logger = _Logger()
    with pr.protect_generation(_NoScheduler(), 50, controller = ctl, logger = logger):
        assert ctl.protected is False
    assert "no scheduler.step" in _Logger.text


def test_a_second_wrapper_over_the_first_still_counts_once():
    """The gate harness and the video backend both wrap ``scheduler.step`` for progress."""
    ctl = pr.NVFP4StepController("0,-1")
    layer = _capable(ctl)
    pipe = _FakePipe()
    with pr.protect_generation(pipe, 10, controller = ctl):
        inner = pipe.scheduler.step
        ticks: list = []
        progress = pipe.scheduler.step

        def outer(*a, **kw):
            ticks.append(1)
            return progress(*a, **kw)

        pipe.scheduler.step = outer
        seen: list = []
        pipe.run(10, lambda: seen.append(ctl.index))
        pipe.scheduler.step = inner
    assert seen == list(range(10))
    assert len(ticks) == 10
    assert "step" not in pipe.scheduler.__dict__


def test_the_graph_key_is_empty_when_the_lever_is_off(monkeypatch):
    monkeypatch.delenv(pr.PROTECT_STEPS_ENV, raising = False)
    pr.reset_protect_controller()
    assert pr.protect_graph_key() == ()
    assert pr.protect_graph_key(True) == ()


def test_the_graph_key_separates_the_two_branches(monkeypatch):
    monkeypatch.setenv(pr.PROTECT_STEPS_ENV, "0,-1")
    ctl = pr.reset_protect_controller()
    try:
        ctl.begin(50)
        assert pr.protect_graph_key() == (("nvfp4_protect", True),)
        ctl.advance()
        assert pr.protect_graph_key() == (("nvfp4_protect", False),)
        assert pr.protect_graph_key(True) != pr.protect_graph_key(False)
    finally:
        monkeypatch.delenv(pr.PROTECT_STEPS_ENV, raising = False)
        pr.reset_protect_controller()


def test_the_wrapper_does_not_key_a_model_with_no_nvfp4_layers(monkeypatch):
    """An fp8 load in a process that names protect steps must not double its graph count."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from core.inference import diffusion_cuda_graph as cg

    monkeypatch.setenv(pr.PROTECT_STEPS_ENV, "0,-1")
    pr.reset_protect_controller()
    try:
        assert cg._protect_keyed(nn.Sequential(nn.Linear(8, 8))) is False
    finally:
        monkeypatch.delenv(pr.PROTECT_STEPS_ENV, raising = False)
        pr.reset_protect_controller()
    assert torch is not None


def _cpu_layer(
    torch,
    *,
    in_features = 64,
    out_features = 32,
):
    return nl.nvfp4_linear_class()(
        in_features,
        out_features,
        wq = torch.zeros(out_features, in_features // 2, dtype = torch.uint8),
        w_sf = torch.ones(128, 4, dtype = torch.uint8),
        alpha = torch.tensor([0.25]),
        a_gsf = torch.tensor([1344.0]),
        bias = None,
    )


def test_a_layer_takes_the_process_controller_and_derives_its_weight_scale():
    torch = pytest.importorskip("torch")
    layer = _cpu_layer(torch)
    assert layer.protect is pr.protect_controller()
    assert float(layer.w_scale) == pytest.approx(0.25 * 1344.0)
    assert "protect=" not in layer.extra_repr()


def test_the_layer_repr_names_the_schedule_when_armed(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv(pr.PROTECT_STEPS_ENV, "0,1,2,3,-1")
    pr.reset_protect_controller()
    try:
        assert "protect=0,1,2,3,-1" in _cpu_layer(torch).extra_repr()
    finally:
        monkeypatch.delenv(pr.PROTECT_STEPS_ENV, raising = False)
        pr.reset_protect_controller()


def test_attach_controller_reaches_every_nvfp4_layer():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    tree = nn.Sequential(_cpu_layer(torch), nn.SiLU(), _cpu_layer(torch), nn.Linear(4, 4))
    ctl = pr.NVFP4StepController("0")
    assert pr.attach_controller(tree, ctl) == 2
    assert tree[0].protect is ctl and tree[2].protect is ctl
    assert [name for name, _ in pr.protect_layers(tree)] == ["0", "2"]


@pytest.mark.parametrize("out_features,in_features", REAL_SHAPES)
def test_the_dequantiser_matches_torchao_bit_for_bit(out_features, in_features):
    """The protected step has to read the SAME weight the unprotected step's GEMM reads."""
    torch = _cuda_or_skip()
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    torch.manual_seed(7)
    with torch.cuda.device(0):
        w = torch.randn(out_features, in_features, device = "cuda", dtype = torch.bfloat16) * 0.02
        amax = w.float().abs().amax().clamp(min = 1e-8)
        per_tensor_scale = (amax / (6.0 * 448.0)).reshape(1)
        tensor = NVFP4Tensor.to_nvfp4(w, per_tensor_scale = per_tensor_scale, is_swizzled_scales = True)
        wq = tensor.qdata.view(torch.uint8).contiguous()
        cols = in_features // 16
        w_sf = (
            tensor.scale.reshape(-1)
            .view(torch.uint8)
            .reshape(ops.sf_matrix_shape(out_features, cols))
            .contiguous()
        )

        want = tensor.dequantize(torch.bfloat16)
        got = ops.dequantize_nvfp4_weight(wq, w_sf, per_tensor_scale, dtype = torch.bfloat16)

    assert tuple(got.shape) == (out_features, in_features)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, want), (
        f"{int((got != want).sum())} of {got.numel()} elements differ from "
        f"NVFP4Tensor.dequantize"
    )
    with torch.cuda.device(0):
        assert torch.equal(
            ops.dequantize_nvfp4_weight(wq, w_sf, per_tensor_scale, dtype = torch.float32),
            tensor.dequantize(torch.float32),
        )


def test_the_dequantiser_matches_flashinfers_own_packing():
    """The other producer of these bytes."""
    torch = _cuda_or_skip()
    import flashinfer

    out_features, in_features = 3072, 3072
    torch.manual_seed(11)
    with torch.cuda.device(0):
        w = torch.randn(out_features, in_features, device = "cuda", dtype = torch.bfloat16) * 0.02
        amax = w.float().abs().amax().clamp(min = 1e-8)
        w_gsf = (6.0 * 448.0 / amax).reshape(1)
        wq, w_sf = flashinfer.nvfp4_quantize(w, w_gsf, do_shuffle = False)
        cols = in_features // 16
        deq = ops.dequantize_nvfp4_weight(
            wq.view(torch.uint8),
            w_sf.reshape(-1).view(torch.uint8).reshape(ops.sf_matrix_shape(out_features, cols)),
            (1.0 / w_gsf),
            dtype = torch.float32,
        )
        rel = float((deq - w.float()).norm() / w.float().norm())
    assert rel < 0.12, rel
    assert bool(torch.isfinite(deq).all())


def _torchao_linear(
    torch,
    out_features,
    in_features,
    *,
    bias = True,
    seed = 0,
):
    import torch.nn as nn
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
    from torchao.quantization import quantize_

    torch.manual_seed(seed)
    linear = nn.Linear(in_features, out_features, bias = bias).to("cuda", torch.bfloat16)
    quantize_(linear, NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel = False))
    return linear


@pytest.mark.parametrize("m", [1, 512, 4096])
def test_a_protected_forward_is_the_dense_bf16_gemm(m):
    torch = _cuda_or_skip()
    import torch.nn.functional as F

    with torch.cuda.device(0):
        linear = _torchao_linear(torch, 3072, 3072)
        x = torch.randn(m, 3072, device = "cuda", dtype = torch.bfloat16) * 0.05
        converted = nl.nvfp4_linear_from_torchao(linear, ops.global_scale(x))
        ctl = pr.NVFP4StepController("0")
        converted.protect = ctl
        ctl.begin(2)

        with torch.inference_mode():
            got = converted(x)
            weight = ops.dequantize_nvfp4_weight(
                converted.wq, converted.w_sf, converted.w_scale, dtype = torch.bfloat16
            )
            want = F.linear(x, weight)
            want = want.to(x.dtype)
            want.add_(converted.bias)
            ctl.advance()
            assert ctl.protected is False
            w4a4 = converted(x)

    assert torch.equal(got, want)
    assert bool(torch.isfinite(w4a4).all())
    rel = float((got.float() - w4a4.float()).norm() / w4a4.float().norm())
    assert 0.0 < rel < 0.2, rel


def test_the_protected_branch_leaves_no_resident_weight_behind():
    """The whole claim of this lever: the dense weight is transient."""
    torch = _cuda_or_skip()

    with torch.cuda.device(0):
        linear = _torchao_linear(torch, 18432, 3072, seed = 3)
        x = torch.randn(1024, 3072, device = "cuda", dtype = torch.bfloat16) * 0.05
        converted = nl.nvfp4_linear_from_torchao(linear, ops.global_scale(x))
        ctl = pr.NVFP4StepController("1")
        converted.protect = ctl
        ctl.begin(4)

        resident = lambda: sum(b.numel() * b.element_size() for b in converted.buffers())
        before_bytes = resident()
        with torch.inference_mode():
            converted(x)  # unprotected
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        ctl.advance()
        assert ctl.protected is True
        with torch.inference_mode():
            out = converted(x)
        del out
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()

    assert resident() == before_bytes
    assert after == baseline, f"{(after - baseline) / 2 ** 20:.1f} MiB left resident"


def test_the_switch_costs_two_compiled_variants_over_a_fifty_step_render():
    """Under torch.compile the branch is a host bool, so Dynamo compiles one variant per VALUE."""
    torch = _cuda_or_skip()
    import torch._dynamo as dynamo

    with torch.cuda.device(0):
        linear = _torchao_linear(torch, 3072, 3072, seed = 5)
        x = torch.randn(512, 3072, device = "cuda", dtype = torch.bfloat16) * 0.05
        converted = nl.nvfp4_linear_from_torchao(linear, ops.global_scale(x))
        ctl = pr.NVFP4StepController("0,1,2,3,-1")
        converted.protect = ctl

        dynamo.reset()
        dynamo.utils.counters.clear()
        compiled = torch.compile(converted, fullgraph = True, dynamic = False)
        ctl.begin(50)
        with torch.inference_mode():
            for _ in range(50):
                compiled(x)
                ctl.advance()
        frames = dynamo.utils.counters["stats"]["unique_graphs"]

    assert frames == 2, f"{frames} compiled variants for one bool; expected 2"
    assert ctl.protected_steps_seen == 5


def test_a_captured_block_gets_one_graph_per_branch(monkeypatch):
    """A graph recorded at a W4A4 step must never replay at a W4A16 one."""
    torch = _cuda_or_skip()
    import torch.nn as nn
    from core.inference import diffusion_cuda_graph as cg

    monkeypatch.setenv(pr.PROTECT_STEPS_ENV, "0")
    ctl = pr.reset_protect_controller()
    try:
        with torch.cuda.device(0):
            linear = _torchao_linear(torch, 3072, 3072, seed = 9)
            x = torch.randn(512, 3072, device = "cuda", dtype = torch.bfloat16) * 0.05
            converted = nl.nvfp4_linear_from_torchao(linear, ops.global_scale(x))

            class Block(nn.Module):
                def __init__(self, inner):
                    super().__init__()
                    self.inner = inner

                def forward(
                    self,
                    hidden_states,
                    return_dict = True,
                ):
                    out = self.inner(hidden_states)
                    return types.SimpleNamespace(sample = out) if return_dict else (out,)

            block = Block(converted).eval()
            handle = cg.GraphedForward(block).install().enable()
            ctl.begin(4)
            call = lambda: block(hidden_states = x, return_dict = False)[0].clone()
            with torch.inference_mode():
                protected = call()
                ctl.advance()
                assert ctl.protected is False
                plain = call()
                ctl.advance()
                again = call()
            captures = handle.stats["captures"]
            replays = handle.stats["replays"]
            cg.uninstall_all([handle])
    finally:
        monkeypatch.delenv(pr.PROTECT_STEPS_ENV, raising = False)
        pr.reset_protect_controller()

    assert captures == 2, handle.stats
    assert replays == 3, handle.stats
    assert torch.equal(plain, again)
    assert not torch.equal(protected, plain)


def test_an_armed_controller_with_no_protect_capable_layer_refuses_and_says_so():
    """Only ``NVFP4FlashInferLinear`` reads the controller, so a load that stayed on torchao runs
    W4A4 at every step. Arming there would count protected steps nothing ran."""

    class _Logger:
        def __init__(self):
            self.text = ""

        def warning(self, message, *args):
            self.text += (message % args if args else message) + "\n"

        def info(self, message, *args):
            self.text += (message % args if args else message) + "\n"

    ctl = pr.NVFP4StepController("0,-1")
    assert ctl.capable_layers() == 0
    pipe = _FakePipe()
    logger = _Logger()
    with pr.protect_generation(pipe, 10, controller = ctl, logger = logger):
        assert "step" not in pipe.scheduler.__dict__
        pipe.run(10, lambda: None)
    assert "no protect-capable NVFP4 layer" in logger.text
    assert (ctl.steps, ctl.protected, ctl.protected_steps_seen) == ((), False, 0)


def test_a_registered_layer_lets_the_same_controller_arm():
    ctl = pr.NVFP4StepController("0,-1")
    layer = _capable(ctl)
    assert ctl.capable_layers() == 1
    pipe = _FakePipe()
    seen: list = []
    with pr.protect_generation(pipe, 10, controller = ctl):
        pipe.run(10, lambda: seen.append(ctl.protected))
    assert [i for i, protected in enumerate(seen) if protected] == [0, 9]
    assert layer is not None


def test_the_layer_registry_is_weak_so_an_unloaded_model_stops_counting():
    import gc

    ctl = pr.NVFP4StepController("0,-1")
    layer = _capable(ctl)
    assert ctl.capable_layers() == 1
    del layer
    gc.collect()
    assert ctl.capable_layers() == 0
