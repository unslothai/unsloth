# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""sm_89 fp8 unit-scale dispatch: install gating, kill switch, fallback, and the epilogue math (CPU, emulated GEMM)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("torchao")
from torchao.quantization.quantize_.workflows.float8 import float8_tensor as f8t  # noqa: E402

from core.inference import diffusion_fp8_unit_scale as us  # noqa: E402


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    us.uninstall()
    monkeypatch.delenv(us.UNIT_SCALE_ENV, raising = False)
    monkeypatch.delenv(us.MIN_ROWS_ENV, raising = False)
    for k in us._COUNTS:
        us._COUNTS[k] = 0
    original = f8t.addmm_float8_unwrapped_inference
    yield
    us.uninstall()
    f8t.addmm_float8_unwrapped_inference = original


def _caps(monkeypatch, caps):
    monkeypatch.setattr(us, "_device_caps", lambda: dict(caps))


def test_kill_switch_never_installs(monkeypatch):
    _caps(monkeypatch, {0: (8, 9)})
    monkeypatch.setenv(us.UNIT_SCALE_ENV, "0")
    before = f8t.addmm_float8_unwrapped_inference
    assert us.install() is False
    assert f8t.addmm_float8_unwrapped_inference is before


@pytest.mark.parametrize("cap", [(8, 0), (8, 6), (9, 0), (10, 0), (12, 0)])
def test_auto_leaves_other_arches_untouched(monkeypatch, cap):
    _caps(monkeypatch, {0: cap})
    before = f8t.addmm_float8_unwrapped_inference
    assert us.install() is False
    assert f8t.addmm_float8_unwrapped_inference is before
    assert us.stats()["installed"] is False


def test_auto_installs_on_ada_idempotent_and_reversible(monkeypatch):
    _caps(monkeypatch, {0: (8, 9)})
    before = f8t.addmm_float8_unwrapped_inference
    assert us.install() is True
    wrapped = f8t.addmm_float8_unwrapped_inference
    assert wrapped is not before and wrapped._unsloth_orig is before
    assert us.install() is True
    assert f8t.addmm_float8_unwrapped_inference is wrapped
    us.uninstall()
    assert f8t.addmm_float8_unwrapped_inference is before


def _operands(
    m,
    k = 64,
    n = 48,
    seed = 0,
):
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn(m, k, generator = g) * 3).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, generator = g) * 3).to(torch.float8_e4m3fn).t()
    sa = torch.rand(m, 1, generator = g) + 0.1
    sb = torch.rand(1, n, generator = g) + 0.1
    return a, sa, b, sb


def test_non_cuda_calls_fall_back_to_torchao(monkeypatch):
    _caps(monkeypatch, {0: (8, 9)})
    seen = []
    monkeypatch.setattr(
        f8t, "addmm_float8_unwrapped_inference", lambda *a, **k: seen.append((a, k)) or "orig"
    )
    assert us.install() is True
    a, sa, b, sb = _operands(2048)
    out = f8t.addmm_float8_unwrapped_inference(a, sa, b, sb, torch.bfloat16, use_fast_accum = True)
    assert out == "orig" and len(seen) == 1 and seen[0][1]["use_fast_accum"] is True
    assert us.stats()["unit_scale"] == 0 and us.stats()["rowwise"] == 1


class _FakeDevice:
    type = "cuda"
    index = 0


class _Dev:
    def __init__(self, t):
        self.t = t
        self.device = _FakeDevice()
        self.shape = t.shape

    def dim(self):
        return self.t.dim()


def _emulated_scaled_mm(
    a,
    b,
    scale_a,
    scale_b,
    out_dtype,
    use_fast_accum = False,
    **_,
):
    a = a.t if isinstance(a, _Dev) else a
    acc = (a.float() * scale_a.float()) @ (b.float() * scale_b.float())
    return acc.to(out_dtype)


@pytest.mark.parametrize("bias", [False, True])
def test_epilogue_matches_rowwise_math(monkeypatch, bias):
    _caps(monkeypatch, {0: (8, 9)})
    rowwise_calls = []

    def rowwise(
        a,
        sa,
        b,
        sb,
        output_dtype,
        output_scale = None,
        bias = None,
        use_fast_accum = False,
    ):
        rowwise_calls.append(a.shape[0])
        a = a.t if isinstance(a, _Dev) else a
        out = (a.float() @ b.float()) * sa * sb
        return (out + bias.float() if bias is not None else out).to(output_dtype)

    monkeypatch.setattr(f8t, "addmm_float8_unwrapped_inference", rowwise)
    monkeypatch.setattr(torch, "_scaled_mm", _emulated_scaled_mm)
    monkeypatch.setattr(
        torch, "ones", lambda *a, device = None, dtype = None: torch.tensor(1.0, dtype = dtype)
    )
    monkeypatch.setenv(us.MIN_ROWS_ENV, "1024")
    assert us.install() is True
    fn = f8t.addmm_float8_unwrapped_inference
    a, sa, b, sb = _operands(2048)
    bv = torch.randn(48).to(torch.bfloat16) if bias else None
    got = fn(_Dev(a), sa, b, sb, torch.bfloat16, bias = bv)
    want = rowwise(a, sa, b, sb, torch.bfloat16, bias = bv)
    assert us.stats()["unit_scale"] == 1
    # One extra bf16 rounding of the unscaled accumulator (bias adds after it, so measure against the output range).
    err = (got.float() - want.float()).abs().max()
    assert err <= 2**-7 * want.float().abs().max()
    assert torch.isfinite(got).all()

    small = _operands(256)
    fn(_Dev(a), sa, b, sb, torch.bfloat16, use_fast_accum = True)
    fn(_Dev(small[0]), small[1], small[2], small[3], torch.bfloat16)
    fn(_Dev(a), sa, b, sb, torch.float32)
    fn(_Dev(a), sa, b, sb, torch.bfloat16, output_scale = torch.tensor(1.0))
    assert us.stats()["unit_scale"] == 1 and len(rowwise_calls) == 5


def test_min_rows_env(monkeypatch):
    monkeypatch.setenv(us.MIN_ROWS_ENV, "4096")
    assert us._min_rows() == 4096
    monkeypatch.setenv(us.MIN_ROWS_ENV, "junk")
    assert us._min_rows() == us._DEFAULT_MIN_ROWS


@pytest.mark.parametrize("container", ["pickle", "safetensors"])
def test_every_prequant_container_installs_the_dispatch(monkeypatch, container):
    # The pickle container and the safetensors rebuild-off path never reach prequant_native's torchao API.
    from core.inference import diffusion_fp8_unit_scale as us
    from core.inference import diffusion_prequant as dp
    from core.inference import prequant_safetensors as ps

    calls = []
    monkeypatch.setattr(us, "install", lambda logger = None, **kw: calls.append(container) or True)
    monkeypatch.setattr(ps, "is_safetensors_checkpoint", lambda path: container == "safetensors")
    monkeypatch.setattr(ps, "load_prequant_safetensors", lambda path, mmap = False: {"format": "fp8"})
    monkeypatch.setattr(dp, "_torch_load_prequant", lambda path, **kwargs: {"format": "fp8"})
    assert dp._load_prequant_checkpoint("model.bin", map_location = "cpu") == {"format": "fp8"}
    assert calls == [container]


def _install_emulated(monkeypatch, scaled_mm = _emulated_scaled_mm):
    _caps(monkeypatch, {0: (8, 9)})
    monkeypatch.setattr(f8t, "addmm_float8_unwrapped_inference", lambda *a, **k: "orig")
    monkeypatch.setattr(torch, "_scaled_mm", scaled_mm)
    monkeypatch.setattr(
        torch, "ones", lambda *a, device = None, dtype = None: torch.tensor(1.0, dtype = dtype)
    )
    assert us.install() is True
    return f8t.addmm_float8_unwrapped_inference


def test_counters_stay_untouched_while_compiling(monkeypatch):
    # A module-global mutated inside a traced region would make Dynamo guard on it and recompile every call.
    fn = _install_emulated(monkeypatch)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    a, sa, b, sb = _operands(2048)
    fn(_Dev(a), sa, b, sb, torch.bfloat16)
    fn(_Dev(a), sa, b, sb, torch.bfloat16, use_fast_accum = True)
    assert us.stats()["unit_scale"] == 0 and us.stats()["rowwise"] == 0
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    fn(_Dev(a), sa, b, sb, torch.bfloat16)
    assert us.stats()["unit_scale"] == 1


def test_legacy_scaled_mm_tuple_is_unwrapped(monkeypatch):
    def legacy(*args, **kwargs):
        return _emulated_scaled_mm(*args, **kwargs), torch.tensor(0.0)

    fn = _install_emulated(monkeypatch, legacy)
    a, sa, b, sb = _operands(2048)
    out = fn(_Dev(a), sa, b, sb, torch.bfloat16)
    assert isinstance(out, torch.Tensor) and out.shape == (2048, 48)


def test_unit_scale_switches_key_the_compile_cache(monkeypatch):
    from core.inference import diffusion_compile_cache as cc

    assert cc.graph_switches() == {}
    monkeypatch.setenv(us.UNIT_SCALE_ENV, "0")
    monkeypatch.setenv(us.MIN_ROWS_ENV, "1024")
    assert cc.graph_switches() == {us.UNIT_SCALE_ENV: "off", us.MIN_ROWS_ENV: "1024"}
    monkeypatch.setenv(us.MIN_ROWS_ENV, "256")
    assert us.MIN_ROWS_ENV not in cc.graph_switches()
    # Keyed on the value _min_rows() uses: 0 runs as 1, so it must not collide with the 256 default.
    for raw, keyed in (("0", "1"), ("1", "1"), ("junk", None)):
        monkeypatch.setenv(us.MIN_ROWS_ENV, raw)
        assert cc.graph_switches().get(us.MIN_ROWS_ENV) == keyed
        assert us._min_rows() == (int(keyed) if keyed else us._DEFAULT_MIN_ROWS)


def test_comfy_prequant_load_installs_the_dispatch():
    # The ComfyUI branch of load_prequantized_transformer bypasses _load_prequant_checkpoint.
    import ast
    import inspect

    from core.inference import diffusion_prequant as dp

    tree = ast.parse(inspect.getsource(dp.load_prequantized_transformer))
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "comfy_scan is not None" in ast.unparse(node.test):
            calls = [ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)]
            assert "load_comfy_prequant" in calls and "_install_fp8_unit_scale" in calls
            assert calls.index("load_comfy_prequant") < calls.index("_install_fp8_unit_scale")
            return
    raise AssertionError("ComfyUI branch not found")


def test_install_tags_the_torch_compile_cache_key(monkeypatch):
    # torch's FX graph / AOTAutograd caches key on the Dynamo graph, which is the same with or without the patch.
    import torch.compiler.config as ccfg

    monkeypatch.setattr(ccfg, "cache_key_tag", "user-tag")
    _caps(monkeypatch, {0: (8, 9)})
    monkeypatch.setenv(us.MIN_ROWS_ENV, "512")
    assert us.install() is True
    assert ccfg.cache_key_tag == f"user-tag;{us._CACHE_TAG_PREFIX}512"
    us.uninstall()
    assert ccfg.cache_key_tag == "user-tag"
    monkeypatch.setenv(us.UNIT_SCALE_ENV, "1")
    assert us.install() is True
    assert ccfg.cache_key_tag == f"user-tag;{us._CACHE_TAG_PREFIX}512:force"
    us.uninstall()
    monkeypatch.setenv(us.UNIT_SCALE_ENV, "0")
    assert us.install() is False
    assert ccfg.cache_key_tag == "user-tag"


_COMPILE_CHILD = r"""
import json, sys, torch, torch.nn as nn
from torchao.float8 import Float8MMConfig
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_
from torch._inductor.utils import run_and_get_code
from core.inference import diffusion_fp8_unit_scale as us
installed = us.install() if sys.argv[1] == "head" else False


def scaled_mm_scale_args(code):
    # Positional args 3 and 4 (scale_a, scale_b) of each generated extern _scaled_mm call, split paren-aware.
    out = []
    for line in code.splitlines():
        if "extern_kernels._scaled_mm(" not in line or line.lstrip().startswith("#"):
            continue
        body, depth, args, cur = line.split("extern_kernels._scaled_mm(", 1)[1], 0, [], ""
        for ch in body:
            if ch == "," and depth == 0:
                args.append(cur.strip()); cur = ""; continue
            if ch == ")" and depth == 0:
                break
            depth += ch in "([" ; depth -= ch in ")]"
            cur += ch
        args.append(cur.strip())
        out.append(args[2:4])
    return out
torch.manual_seed(0)
lin = nn.Linear(256, 512, device = "cuda", dtype = torch.bfloat16)
kw = {}
try:  # Studio pins the plain-torch kernel (diffusion_transformer_quant._make_quant_config)
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
    kw["kernel_preference"] = KernelPreference.TORCH
except Exception:
    pass
quantize_(lin, Float8DynamicActivationFloat8WeightConfig(
    granularity = PerRow(), mm_config = Float8MMConfig(use_fast_accum = False), **kw))
x = torch.randn(512, 256, device = "cuda", dtype = torch.bfloat16)
with torch.no_grad():
    _, codes = run_and_get_code(torch.compile(lin), x)
calls = [a for c in codes for a in scaled_mm_scale_args(c)]
# The unit-scale graph feeds one shared ones() scalar as both scales; the row-wise graph feeds two distinct scales.
print(json.dumps({"installed": installed, "calls": len(calls), "unit": all(a == b for a, b in calls)}))
"""


def _fp8_gpu() -> bool:
    try:
        # ROCm capability is a gfx version, and Studio never runs torchao fp8 on ROCm.
        if getattr(torch.version, "hip", None):
            return False
        return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _fp8_gpu(), reason = "needs an fp8 _scaled_mm GPU (sm_89+)")
def test_compiled_graph_after_a_cached_rowwise_one_uses_the_dispatcher(tmp_path):
    # Base process compiles first into a shared inductor cache; the installed process must not reuse that graph.
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    backend = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "inductor"),
        us.UNIT_SCALE_ENV: "1",
        us.MIN_ROWS_ENV: "1",
        "PYTHONPATH": os.pathsep.join(filter(None, [str(backend), os.environ.get("PYTHONPATH")])),
    }
    env.pop("TORCH_COMPILE_CACHE_KEY_TAG", None)
    runs = {}
    for arm in ("base", "head"):
        proc = subprocess.run(
            [sys.executable, "-c", _COMPILE_CHILD, arm],
            env = env,
            capture_output = True,
            text = True,
            timeout = 600,
            cwd = str(backend),
        )
        assert proc.returncode == 0, proc.stderr[-3000:]
        runs[arm] = json.loads(proc.stdout.strip().splitlines()[-1])
    assert (
        runs["head"]["installed"] is True and runs["base"]["calls"] and runs["head"]["calls"]
    ), runs
    assert runs["base"]["unit"] is False, runs
    # Without its own cache key the installed run reloads the base run's row-wise graph from the shared cache.
    assert runs["head"]["unit"] is True, runs


def test_installed_dispatcher_keys_studio_compile_bundles(monkeypatch):
    # A bundle saved without the dispatcher holds artifacts torch no longer looks up once the tag is set.
    import torch.compiler.config as ccfg

    from core.inference import diffusion_compile_cache as cc

    monkeypatch.setattr(ccfg, "cache_key_tag", "")
    plain = cc.environment_fingerprint()
    assert "torch_cache_key_tag" not in plain
    _caps(monkeypatch, {0: (8, 9)})
    assert us.install() is True
    tagged = cc.environment_fingerprint()
    assert tagged["torch_cache_key_tag"] == f"{us._CACHE_TAG_PREFIX}{us._DEFAULT_MIN_ROWS}"
    assert cc.cache_key(plain, {}) != cc.cache_key(tagged, {})
    us.uninstall()
    assert cc.environment_fingerprint() == plain


# ── installed only where it can engage, consistent kill switch, eager epilogue memory ─────────


def test_auto_never_installs_for_fast_accumulate(monkeypatch):
    # Studio's default is fast accumulate, which the dispatcher never takes: installing would only tag every later
    # compile in the process (a bf16 or GGUF model too) with a cache key that splits its bundles.
    import torch.compiler.config as ccfg

    monkeypatch.setattr(ccfg, "cache_key_tag", "")
    _caps(monkeypatch, {0: (8, 9)})
    before = f8t.addmm_float8_unwrapped_inference
    assert us.install(fast_accum = True) is False
    assert f8t.addmm_float8_unwrapped_inference is before and ccfg.cache_key_tag == ""
    assert us.install(fast_accum = False) is True
    us.uninstall()
    assert us.install(fast_accum = None) is True  # unknown accumulate keeps the speed-up
    us.uninstall()
    monkeypatch.setenv(us.UNIT_SCALE_ENV, "1")
    assert us.install(fast_accum = True) is True  # force covers both accumulate modes


def test_fp8_quant_config_installs_only_for_precise_accumulate(monkeypatch):
    from core.inference import diffusion_transformer_quant as tq

    _caps(monkeypatch, {0: (8, 9)})
    tq._make_quant_config(tq.TQ_FP8)
    assert us.stats()["installed"] is False
    tq._make_quant_config(tq.TQ_FP8, fast_accum = True)
    assert us.stats()["installed"] is False
    tq._make_quant_config(tq.TQ_FP8, fast_accum = False)
    assert us.stats()["installed"] is True


@pytest.mark.parametrize(
    "metadata, installs",
    [
        ({"scheme": "int8"}, False),
        ({"scheme": "fp8", "fast_accum": True}, False),
        ({"scheme": "fp8", "fast_accum": False}, True),
        ({"scheme": "nvfp4"}, True),  # a policy over fp8, accumulate unrecorded
        ({}, True),
    ],
)
def test_prequant_checkpoint_installs_only_when_its_fp8_layers_can_engage(
    monkeypatch, metadata, installs
):
    from core.inference import diffusion_prequant as dp
    from core.inference import prequant_safetensors as ps

    _caps(monkeypatch, {0: (8, 9)})
    ckpt = {"format": dp.PREQUANT_FORMAT, "state_dict": {}, "metadata": metadata}
    monkeypatch.setattr(ps, "is_safetensors_checkpoint", lambda path: True)
    monkeypatch.setattr(ps, "load_prequant_safetensors", lambda path, mmap = False: ckpt)
    assert dp._load_prequant_checkpoint("model.safetensors") is ckpt
    assert us.stats()["installed"] is installs


def test_every_off_word_of_the_compile_cache_key_turns_the_dispatcher_off(monkeypatch):
    from core.inference import diffusion_compile_cache as cc

    _caps(monkeypatch, {0: (8, 9)})
    for word in cc._OFF_WORDS:
        monkeypatch.setenv(us.UNIT_SCALE_ENV, word.upper())
        assert cc.graph_switches().get(us.UNIT_SCALE_ENV) == "off"
        assert us._mode() == "off" and us.install(fast_accum = False) is False
    for word in cc._ON_WORDS:
        monkeypatch.setenv(us.UNIT_SCALE_ENV, word)
        assert us._mode() == "force"


@pytest.mark.parametrize("bias", [False, True])
def test_eager_epilogue_writes_into_the_accumulator_bitwise_as_before(monkeypatch, bias):
    # The fused expression materialized an fp32 [M, N] buffer plus a second bf16 output eagerly (the PR's own 662 vs
    # 486 MiB). Chunked into the accumulator it allocates neither, and rounds every element the same way.
    produced = []

    def scaled_mm(*args, **kwargs):
        acc = _emulated_scaled_mm(*args, **kwargs)
        produced.append(acc)
        return acc

    fn = _install_emulated(monkeypatch, scaled_mm)
    monkeypatch.setattr(
        us, "_EAGER_EPILOGUE_ELEMS", 48 * 300
    )  # 300-row chunks: several, the last one partial
    a, sa, b, sb = _operands(2048)
    bv = torch.randn(48).to(torch.bfloat16) if bias else None
    got = fn(_Dev(a), sa, b, sb, torch.bfloat16, bias = bv)
    acc = produced[-1]
    assert got.data_ptr() == acc.data_ptr()
    unit = _emulated_scaled_mm(a, b, torch.tensor(1.0), torch.tensor(1.0), torch.bfloat16)
    want = unit * sa.float()
    want.mul_(sb.float())
    if bv is not None:
        want.add_(bv.float())
    assert torch.equal(got, want.to(torch.bfloat16))
