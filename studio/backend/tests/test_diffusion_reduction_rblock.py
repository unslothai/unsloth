# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Left on, dynamic_scale_rblock benchmarks a second launcher per process, changing the bits."""

from __future__ import annotations

import hashlib
import types

import pytest

torch = pytest.importorskip("torch")
import torch._inductor.config  # noqa: E402

from core.inference import diffusion_compile_cache as cache  # noqa: E402
from core.inference import diffusion_compile_config as cc  # noqa: E402
from core.inference import diffusion_speed as ds  # noqa: E402

_INDUCTOR = torch._inductor.config
_KILL_SWITCH = "UNSLOTH_DIFFUSION_DYNAMIC_SCALE_RBLOCK"


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    before = (_INDUCTOR.dynamic_scale_rblock, _INDUCTOR.emulate_precision_casts)
    test_cfg = _INDUCTOR.test_configs
    before_filter = getattr(test_cfg, "force_filter_reduction_configs", None)
    monkeypatch.delenv(_KILL_SWITCH, raising = False)
    cc._reset_for_tests()
    yield
    cc._reset_for_tests()
    _INDUCTOR.dynamic_scale_rblock, _INDUCTOR.emulate_precision_casts = before
    if before_filter is not None:
        test_cfg.force_filter_reduction_configs = before_filter


class _Transformer:
    _repeated_blocks = ()

    def compile_repeated_blocks(self, **kwargs):
        self.kwargs = kwargs

    def named_modules(self):
        return iter(())

    def modules(self):
        return iter(())

    def parameters(self):
        return iter(())


def _compile(max_autotune = False):
    pipe = types.SimpleNamespace(transformer = _Transformer())
    assert ds._compile_repeated_blocks(pipe, None, max_autotune = max_autotune) is True


@pytest.mark.parametrize("max_autotune", [False, True])
def test_regional_compile_pins_dynamic_scale_rblock_off(max_autotune):
    _INDUCTOR.dynamic_scale_rblock = True
    _compile(max_autotune)
    assert _INDUCTOR.dynamic_scale_rblock is False
    assert cc.get_knob("torch._inductor.config", "dynamic_scale_rblock") is False


def test_kill_switch_keeps_inductor_default(monkeypatch):
    monkeypatch.setenv(_KILL_SWITCH, "1")
    _INDUCTOR.dynamic_scale_rblock = True
    _compile()
    assert _INDUCTOR.dynamic_scale_rblock is True


def test_unload_restores_the_process_value():
    _INDUCTOR.dynamic_scale_rblock = True
    snap = ds.snapshot_backend_flags()
    assert snap["inductor_dynamic_scale_rblock"] is True
    _compile()
    assert _INDUCTOR.dynamic_scale_rblock is False
    ds.restore_backend_flags(snap)
    assert _INDUCTOR.dynamic_scale_rblock is True


def _fingerprint():
    return cache.model_fingerprint(
        family = "flux.1",
        transformer = None,
        dtype = "torch.bfloat16",
        quant = "int8",
        attention_backend = "_native_cudnn",
        compile_kwargs = {"fullgraph": True, "dynamic": None, "mode": "default"},
    )


def test_bundle_key_tracks_the_pin(monkeypatch):
    pinned = _fingerprint()
    assert pinned["inductor"] == {"dynamic_scale_rblock": False}
    monkeypatch.setenv(_KILL_SWITCH, "1")
    unpinned = _fingerprint()
    # kill switch = pre-pin key, so bundles compiled with inductor's default still hit
    assert "inductor" not in unpinned
    env = cache.environment_fingerprint()
    assert cache.cache_key(env, pinned) != cache.cache_key(env, unpinned)


def _flux_like_block():
    """Single-stream FLUX block head: LayerNorm + AdaLN modulation feeding four torchao int8 dynamic-activation
    Linears (q, k, v, mlp). Inductor fuses the norm and the four activation quantisations into one looped reduction."""
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(3072, elementwise_affine = False, eps = 1e-6)
            self.q = torch.nn.Linear(3072, 3072)
            self.k = torch.nn.Linear(3072, 3072)
            self.v = torch.nn.Linear(3072, 3072)
            self.mlp = torch.nn.Linear(3072, 12288)

        def forward(self, x, shift, scale):
            h = self.norm(x) * (1 + scale) + shift
            return self.q(h), self.k(h), self.v(h), self.mlp(h)

    torch.manual_seed(0)
    block = Block().cuda().to(torch.bfloat16)
    quantize_(block, Int8DynamicActivationInt8WeightConfig(set_inductor_config = False))
    x = torch.randn(4096, 3072, device = "cuda", dtype = torch.bfloat16)
    shift = torch.randn(3072, device = "cuda", dtype = torch.bfloat16)
    scale = torch.randn(3072, device = "cuda", dtype = torch.bfloat16)
    return block, (x, shift, scale)


def _run_counting_reduction_autotunes(
    monkeypatch,
    tmp_path,
    *,
    force_r0_block = None,
):
    """Compile + run the block in a fresh inductor cache. Returns (R0_BLOCK sets benchmarked at runtime, output sha).
    ``force_r0_block`` makes a two-launcher reduction take that block size instead of benchmarking."""
    from torch._inductor.runtime import triton_heuristics

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    from torch._inductor.codecache import PyCodeCache

    torch._dynamo.reset()
    PyCodeCache.cache_clear()
    seen = []
    real = triton_heuristics.CachingAutotuner.autotune_to_one_config

    def autotune(self, *args, **kwargs):
        blocks = sorted(
            {launcher.config.kwargs.get("R0_BLOCK") for launcher in self.launchers} - {None}
        )
        if blocks:
            seen.append(blocks)
            if force_r0_block in blocks:
                self.launchers = [
                    l for l in self.launchers if l.config.kwargs.get("R0_BLOCK") == force_r0_block
                ]
                return None
        return real(self, *args, **kwargs)

    monkeypatch.setattr(triton_heuristics.CachingAutotuner, "autotune_to_one_config", autotune)
    block, inputs = _flux_like_block()
    with torch.no_grad():
        out = torch.compile(block, fullgraph = True, dynamic = False)(*inputs)
    torch.cuda.synchronize()
    digest = hashlib.sha256()
    for t in out:
        digest.update(t.float().cpu().numpy().tobytes())
    torch._dynamo.reset()
    return seen, digest.hexdigest()


def _cuda_torchao_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("inductor's dynamic_scale_rblock is sm80+ only")
    pytest.importorskip("torchao")
    pytest.importorskip("triton")


@pytest.mark.gpu
def test_pinned_reduction_has_one_block_size_and_the_default_has_two(monkeypatch, tmp_path):
    _cuda_torchao_or_skip()
    _INDUCTOR.emulate_precision_casts = True
    _INDUCTOR.dynamic_scale_rblock = True
    default_seen, _ = _run_counting_reduction_autotunes(monkeypatch, tmp_path / "default")
    if not default_seen:
        pytest.skip(
            "this GPU does not give the fused norm kernel a second R0_BLOCK (register budget not exceeded)"
        )
    assert any(len(b) == 2 and b[1] == 2 * b[0] for b in default_seen), default_seen
    small, large = next(b for b in default_seen if len(b) == 2)
    _, sha_small = _run_counting_reduction_autotunes(
        monkeypatch, tmp_path / "small", force_r0_block = small
    )
    _, sha_large = _run_counting_reduction_autotunes(
        monkeypatch, tmp_path / "large", force_r0_block = large
    )
    # the per-process benchmark winner changes the bits: the cross-server drift
    assert sha_small != sha_large
    _compile()
    pinned_seen, sha_pinned = _run_counting_reduction_autotunes(monkeypatch, tmp_path / "pinned")
    assert pinned_seen == []
    assert sha_pinned in (sha_small, sha_large)


# per compile only, never the global inductor knob

_FILTER = "test_configs.force_filter_reduction_configs"
_HAS_FILTER = hasattr(_INDUCTOR.test_configs, "force_filter_reduction_configs")
_needs_filter = pytest.mark.skipif(
    not _HAS_FILTER, reason = "torch has no inductor reduction-config filter"
)


def _family(name):
    from core.inference import diffusion_families, video_families
    for fam in (*video_families._FAMILIES, *diffusion_families._FAMILIES):
        if fam.name == name:
            return fam
    raise AssertionError(f"no family {name}")


def _filter_reaching_compile(monkeypatch, family):
    """What apply_speed_optims hands the regional compile for ``family`` on a bf16 CUDA target, default tier."""
    seen = {}

    def fake_compile(pipe, logger, **kwargs):
        seen.update(kwargs)
        return False

    monkeypatch.setattr(ds, "compile_eligible", lambda *a, **k: True)
    monkeypatch.setattr(ds, "_compile_repeated_blocks", fake_compile)
    target = types.SimpleNamespace(device = "cpu", dtype = torch.bfloat16, backend = "cuda")
    ds.apply_speed_optims(
        types.SimpleNamespace(), target, is_gguf = False, family = family, speed_mode = ds.SPEED_DEFAULT
    )
    return seen["filter_reductions"]


def test_only_ltx_opts_into_the_reduction_filter(monkeypatch):
    # off sm120, where the image families' arch-scoped opt-in is inert
    monkeypatch.setattr(cc, "_device_capability", lambda: (10, 0))
    assert _filter_reaching_compile(monkeypatch, _family("ltx-2")) is True
    for name in ("hunyuanvideo-1.5", "wan2.2-ti2v-5b", "flux.1"):
        assert _filter_reaching_compile(monkeypatch, _family(name)) is False, name


def _compiled_kwargs(max_autotune = False, filter_reductions = False):
    pipe = types.SimpleNamespace(transformer = _Transformer())
    assert (
        ds._compile_repeated_blocks(
            pipe, None, max_autotune = max_autotune, filter_reductions = filter_reductions
        )
        is True
    )
    return pipe.transformer.kwargs


@_needs_filter
@pytest.mark.parametrize("max_autotune", [False, True])
def test_opted_in_compile_carries_the_filter_and_leaves_the_process_knob(max_autotune):
    _INDUCTOR.test_configs.force_filter_reduction_configs = False
    kwargs = _compiled_kwargs(max_autotune, filter_reductions = True)
    assert kwargs["options"][_FILTER] is True
    # torch.compile takes mode or options, so max's mode is folded into options
    assert "mode" not in kwargs
    if max_autotune:
        assert kwargs["options"]["max_autotune"] is True
    assert _INDUCTOR.test_configs.force_filter_reduction_configs is False
    assert not cc.is_recorded(
        "torch._inductor.config.test_configs", "force_filter_reduction_configs"
    )


@pytest.mark.parametrize("max_autotune", [False, True])
def test_other_families_compile_with_inductor_default(max_autotune):
    kwargs = _compiled_kwargs(max_autotune, filter_reductions = False)
    assert "options" not in kwargs
    assert ("mode" in kwargs) is max_autotune


@_needs_filter
def test_kill_switch_drops_the_filter(monkeypatch):
    monkeypatch.setenv(_KILL_SWITCH, "1")
    kwargs = _compiled_kwargs(filter_reductions = True)
    assert "options" not in kwargs


@_needs_filter
def test_bundle_key_carries_the_filter_only_when_set(monkeypatch):
    plain = _fingerprint()
    assert plain["inductor"] == {"dynamic_scale_rblock": False}
    filtered = cache.model_fingerprint(
        family = "ltx-2",
        transformer = None,
        dtype = "torch.bfloat16",
        quant = "int8",
        attention_backend = "_native_cudnn",
        compile_kwargs = {"fullgraph": True, "dynamic": None, "mode": "default"},
        reduction_filter = True,
    )
    assert filtered["inductor"] == {
        "dynamic_scale_rblock": False,
        "force_filter_reduction_configs": True,
    }
    monkeypatch.setenv(_KILL_SWITCH, "1")
    assert "inductor" not in cache.model_fingerprint(
        family = "ltx-2",
        transformer = None,
        dtype = "torch.bfloat16",
        quant = "int8",
        attention_backend = "_native_cudnn",
        compile_kwargs = {"fullgraph": True, "dynamic": None, "mode": "default"},
        reduction_filter = True,
    )


def _wan_like_block_head():
    """Wan TI2V-5B block head: fp32 LayerNorm of the bf16 stream, per-token AdaLN modulation, then a Linear."""

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(3072, elementwise_affine = False, eps = 1e-6)
            self.table = torch.nn.Parameter(torch.randn(1, 6, 3072) / 3072**0.5)
            self.proj = torch.nn.Linear(3072, 3072)

        def forward(self, x, temb):
            shift, scale, *_ = (self.table.unsqueeze(0) + temb.float()).chunk(6, dim = 2)
            h = (self.norm(x.float()) * (1 + scale.squeeze(2)) + shift.squeeze(2)).type_as(x)
            return self.proj(h)

    torch.manual_seed(0)
    head = Head().cuda()
    head.proj.to(torch.bfloat16)
    x = (torch.randn(1, 6160, 3072, device = "cuda") * 4 + 0.3).to(torch.bfloat16)
    temb = (torch.randn(1, 6160, 6, 3072, device = "cuda") * 0.5).to(torch.bfloat16)
    return head, (x, temb)


def _ltx_like_block_norm():
    """LTX-2.3 block head at 768x512x121: fp32-stat RMSNorm over hidden 4096, per-token AdaLN modulation, a Linear.
    The norm weight keeps both summation orders visible in the output on every torch."""

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Parameter(torch.randn(6, 4096) / 64)
            self.weight = torch.nn.Parameter(1 + 0.1 * torch.randn(4096))
            self.proj = torch.nn.Linear(4096, 4096)

        def forward(self, x, temb):
            shift, scale, *_ = (
                self.table[None, None] + temb.reshape(1, temb.shape[1], 6, -1)
            ).unbind(dim = 2)
            h = x.float()
            normed = (
                h * torch.rsqrt(h.pow(2).mean(-1, keepdim = True) + 1e-6) * self.weight.float()
            ).to(x.dtype)
            return self.proj(normed * (1 + scale) + shift)

    torch.manual_seed(0)
    head = Head().cuda().to(torch.bfloat16)
    x = (torch.randn(1, 6144, 4096, device = "cuda") * 3 + 0.2).to(torch.bfloat16)
    temb = (torch.randn(1, 6144, 6 * 4096, device = "cuda") * 0.5).to(torch.bfloat16)
    return head, (x, temb)


def _qwen_like_text_norm():
    """The modulated output stays fp32 so the two statistics orders show on every GPU."""

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(3072, elementwise_affine = False, eps = 1e-6)

        def forward(self, x, mod):
            shift, scale = mod.float().chunk(2, dim = -1)
            return torch.addcmul(shift.unsqueeze(1), self.norm(x.float()), 1 + scale.unsqueeze(1))

    torch.manual_seed(0)
    head = Head().cuda()
    x = (torch.randn(1, 64, 3072, device = "cuda") * 3 + 0.2).to(torch.bfloat16)
    mod = (torch.randn(1, 6144, device = "cuda") * 0.5).to(torch.bfloat16)
    return head, (x, mod)


def _run_norm(
    monkeypatch,
    tmp_path,
    *,
    force_r0_block = None,
    build = None,
    compile_kwargs = None,
):
    from torch._inductor.runtime import triton_heuristics

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    from torch._inductor.codecache import PyCodeCache

    torch._dynamo.reset()
    PyCodeCache.cache_clear()
    seen = []
    real = triton_heuristics.CachingAutotuner.autotune_to_one_config

    def autotune(self, *args, **kwargs):
        blocks = sorted(
            {launcher.config.kwargs.get("R0_BLOCK") for launcher in self.launchers} - {None}
        )
        if len(blocks) > 1:
            seen.append(blocks)
            if force_r0_block in blocks:
                self.launchers = [
                    l for l in self.launchers if l.config.kwargs.get("R0_BLOCK") == force_r0_block
                ]
                return None
        return real(self, *args, **kwargs)

    monkeypatch.setattr(triton_heuristics.CachingAutotuner, "autotune_to_one_config", autotune)
    head, inputs = (build or _wan_like_block_head)()
    with torch.no_grad():
        out = torch.compile(head, **(compile_kwargs or {"fullgraph": True, "dynamic": False}))(
            *inputs
        )
    torch.cuda.synchronize()
    torch._dynamo.reset()
    return seen, hashlib.sha256(out.float().cpu().numpy().tobytes()).hexdigest()


@pytest.mark.gpu
@_needs_filter
@pytest.mark.parametrize(
    "build",
    [_wan_like_block_head, _ltx_like_block_norm, _qwen_like_text_norm],
    ids = ["wan_layernorm", "ltx23_rmsnorm", "qwen_image_text_layernorm"],
)
def test_filter_option_gives_the_norm_one_config(monkeypatch, tmp_path, build):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("needs an sm80+ CUDA GPU")
    pytest.importorskip("triton")
    _INDUCTOR.emulate_precision_casts = True
    _INDUCTOR.dynamic_scale_rblock = False
    _INDUCTOR.test_configs.force_filter_reduction_configs = False
    default_seen, _ = _run_norm(monkeypatch, tmp_path / "default", build = build)
    if not default_seen:
        pytest.skip("the reduction heuristic gives this GPU's norm a single config")
    small, large = min(default_seen[0]), max(default_seen[0])
    _, sha_small = _run_norm(monkeypatch, tmp_path / "small", force_r0_block = small, build = build)
    _, sha_large = _run_norm(monkeypatch, tmp_path / "large", force_r0_block = large, build = build)
    assert sha_small != sha_large
    kwargs = {"fullgraph": True, "dynamic": False}
    assert ds.pin_reduction_configs(kwargs) is True
    pinned_seen, sha_pinned = _run_norm(
        monkeypatch, tmp_path / "pinned", build = build, compile_kwargs = kwargs
    )
    assert pinned_seen == []
    assert sha_pinned in (sha_small, sha_large)
    assert _INDUCTOR.test_configs.force_filter_reduction_configs is False
    after_seen, _ = _run_norm(monkeypatch, tmp_path / "after", build = build)
    assert after_seen == default_seen
