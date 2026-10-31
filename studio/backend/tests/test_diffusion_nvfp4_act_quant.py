# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The one-pass NVFP4 activation quantiser: kill switch, eligibility, and (on Blackwell) bit identity."""

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_nvfp4_act_quant as aq  # noqa: E402


def _blackwell() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
    except Exception:  # noqa: BLE001
        return False


def test_kill_switch(monkeypatch):
    monkeypatch.setenv(aq.FAST_ACT_QUANT_ENV, "0")
    assert not aq.fast_act_quant_enabled()
    assert aq.install_nvfp4_act_quant_patch() is False
    assert aq.describe()["reason"].endswith("=0")
    monkeypatch.delenv(aq.FAST_ACT_QUANT_ENV)
    assert aq.fast_act_quant_enabled()


def test_eligibility_is_structural():
    s = torch.tensor([0.01])
    x = torch.zeros(4, 32, dtype = torch.bfloat16)
    # CPU tensors, wrong dtype, unswizzled scales, missing per-tensor scale, odd K: all stay on torchao.
    assert not aq._eligible(x, 16, s, True, False)
    assert not aq._eligible(x.float(), 16, s, True, False)
    assert not aq._eligible(x, 16, None, True, False)
    assert not aq._eligible(x, 16, s, False, False)
    assert not aq._eligible(x, 16, s, True, True)


def test_host_scalar_scale_falls_back_to_torchao():
    # torchao accepts a CPU 0-d per-tensor scale with a CUDA activation; the kernel would read a host pointer.
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode(allow_non_fake_inputs = True):
        x = torch.empty(4, 32, dtype = torch.bfloat16, device = "cuda")
        assert aq._eligible(x, 16, torch.empty((), device = "cuda"), True, False)
        assert not aq._eligible(x, 16, torch.empty(()), True, False)


def test_install_without_cuda_is_a_noop(monkeypatch):
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    monkeypatch.setitem(aq._STATE, "installed", False)
    assert aq.install_nvfp4_act_quant_patch() is False
    assert aq.describe()["reason"] == "cuda unavailable"


@pytest.mark.skipif(not _blackwell(), reason = "needs sm_100+ for torchao NVFP4")
def test_bit_identical_to_compiled_torchao():
    pytest.importorskip("triton")
    ok, reason = aq.verify()
    assert ok, reason


def test_fake_tensors_count_as_traced():
    # AOTAutograd traces torchao's subclass dispatch with fake tensors while is_compiling() is False.
    from torch._subclasses.fake_tensor import FakeTensorMode
    assert not aq._traced(torch.zeros(2, 16))
    with FakeTensorMode():
        assert aq._traced(torch.empty(2, 16))


@pytest.mark.skipif(not _blackwell(), reason = "needs sm_100+ for torchao NVFP4")
def test_compiled_nvfp4_linear_launches_the_kernel():
    pytest.importorskip("triton")
    from torch.profiler import ProfilerActivity, profile
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
    from torchao.quantization import quantize_

    assert aq.install_nvfp4_act_quant_patch()
    m = torch.nn.Sequential(torch.nn.Linear(512, 512), torch.nn.GELU(), torch.nn.Linear(512, 512))
    m = m.cuda().to(torch.bfloat16)
    quantize_(m, NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel = False))
    f = torch.compile(m, fullgraph = True)
    x = torch.randn(256, 512, device = "cuda", dtype = torch.bfloat16)
    with torch.inference_mode():
        f(x)
        with profile(activities = [ProfilerActivity.CUDA]) as p:
            f(x)
            torch.cuda.synchronize()
    assert sum(1 for e in p.events() if "act_quant_kernel" in e.name) == 2


def test_install_tags_the_compile_cache_and_uninstall_clears_it(monkeypatch):
    # The Dynamo graph is the same with or without the patch, so torch's persistent FX/AOT caches must be keyed on it.
    cc = pytest.importorskip("torch.compiler.config")
    if not isinstance(getattr(cc, "cache_key_tag", None), str):
        pytest.skip("torch without compiler cache_key_tag")
    nvfp4 = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    monkeypatch.setattr(nvfp4.NVFP4Tensor, "to_nvfp4", nvfp4.NVFP4Tensor.__dict__["to_nvfp4"])
    monkeypatch.setattr(cc, "cache_key_tag", "other=1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(aq, "verify", lambda: (True, "stub"))
    monkeypatch.setattr(
        aq, "_STATE", {"installed": False, "verified": None, "reason": "", "fallback": 0}
    )
    monkeypatch.setattr(aq, "_ORIGINAL", None)
    monkeypatch.delenv(aq.FAST_ACT_QUANT_ENV, raising = False)
    assert aq.install_nvfp4_act_quant_patch() is True
    assert cc.cache_key_tag == "other=1;unsloth_nvfp4_act_quant=1"
    assert aq.install_nvfp4_act_quant_patch() is True
    assert cc.cache_key_tag.count("unsloth_nvfp4_act_quant") == 1
    aq.uninstall_nvfp4_act_quant_patch()
    assert cc.cache_key_tag == "other=1"


def test_failed_verification_tags_the_cache_off(monkeypatch):
    cc = pytest.importorskip("torch.compiler.config")
    if not isinstance(getattr(cc, "cache_key_tag", None), str):
        pytest.skip("torch without compiler cache_key_tag")
    nvfp4 = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    monkeypatch.setattr(nvfp4.NVFP4Tensor, "to_nvfp4", nvfp4.NVFP4Tensor.__dict__["to_nvfp4"])
    monkeypatch.setattr(cc, "cache_key_tag", "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(aq, "verify", lambda: (False, "stub mismatch"))
    monkeypatch.setattr(
        aq, "_STATE", {"installed": False, "verified": None, "reason": "", "fallback": 0}
    )
    monkeypatch.setattr(aq, "_ORIGINAL", None)
    monkeypatch.delenv(aq.FAST_ACT_QUANT_ENV, raising = False)
    assert aq.install_nvfp4_act_quant_patch() is False
    assert cc.cache_key_tag == "unsloth_nvfp4_act_quant=0"


def _isolated_tag_state(monkeypatch, tag = ""):
    cc = pytest.importorskip("torch.compiler.config")
    if not isinstance(getattr(cc, "cache_key_tag", None), str):
        pytest.skip("torch without compiler cache_key_tag")
    nvfp4 = pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    monkeypatch.setattr(nvfp4.NVFP4Tensor, "to_nvfp4", nvfp4.NVFP4Tensor.__dict__["to_nvfp4"])
    monkeypatch.setattr(cc, "cache_key_tag", tag)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(aq, "verify", lambda: (True, "stub"))
    monkeypatch.setattr(
        aq, "_STATE", {"installed": False, "verified": None, "reason": "", "fallback": 0}
    )
    monkeypatch.setattr(aq, "_ORIGINAL", None)
    monkeypatch.delenv(aq.FAST_ACT_QUANT_ENV, raising = False)
    return cc, nvfp4.NVFP4Tensor


def test_kill_switch_key_differs_from_the_legacy_untagged_key(monkeypatch):
    # Caches from before the tag are untagged and may hold the patched graph: "off" must not land on that key.
    cc, _ = _isolated_tag_state(monkeypatch, "other=1")
    monkeypatch.setenv(aq.FAST_ACT_QUANT_ENV, "0")
    assert aq.install_nvfp4_act_quant_patch() is False
    assert cc.cache_key_tag == "other=1;unsloth_nvfp4_act_quant=0"
    monkeypatch.delenv(aq.FAST_ACT_QUANT_ENV)
    assert aq.install_nvfp4_act_quant_patch() is True
    assert cc.cache_key_tag == "other=1;unsloth_nvfp4_act_quant=1"


def test_kill_switch_after_install_restores_torchao(monkeypatch):
    cc, NVFP4Tensor = _isolated_tag_state(monkeypatch)
    original = NVFP4Tensor.to_nvfp4
    assert aq.install_nvfp4_act_quant_patch() is True
    assert NVFP4Tensor.to_nvfp4 is not original
    monkeypatch.setenv(aq.FAST_ACT_QUANT_ENV, "0")
    assert aq.install_nvfp4_act_quant_patch() is False
    assert NVFP4Tensor.to_nvfp4 is original
    assert cc.cache_key_tag == "unsloth_nvfp4_act_quant=0"


def test_per_load_reset_drops_the_tag_for_a_non_nvfp4_load(monkeypatch):
    cc, NVFP4Tensor = _isolated_tag_state(monkeypatch, "other=1")
    assert aq.install_nvfp4_act_quant_patch() is True
    aq.reset_compile_cache_tag()
    assert cc.cache_key_tag == "other=1"
    assert aq.install_nvfp4_act_quant_patch() is True
    assert cc.cache_key_tag == "other=1;unsloth_nvfp4_act_quant=1"


def test_load_pipeline_resets_the_tag_after_unload():
    import ast
    import pathlib

    src = pathlib.Path(aq.__file__).with_name("diffusion.py").read_text(encoding = "utf-8")
    fn = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "load_pipeline"
    )
    body = ast.get_source_segment(src, fn)
    assert "reset_compile_cache_tag()" in body
    assert body.index("self._unload_locked()") < body.index("reset_compile_cache_tag()")


_CROSS_PROCESS_PROBE = """
import os, sys
import torch
from torch.profiler import ProfilerActivity, profile
from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
from torchao.quantization import quantize_
from core.inference import diffusion_nvfp4_act_quant as aq
aq.install_nvfp4_act_quant_patch()
torch.manual_seed(0)
m = torch.nn.Sequential(torch.nn.Linear(512, 512)).cuda().to(torch.bfloat16)
quantize_(m, NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel = False))
f = torch.compile(m, fullgraph = True)
x = torch.randn(256, 512, device = "cuda", dtype = torch.bfloat16)
with torch.inference_mode():
    f(x)
    with profile(activities = [ProfilerActivity.CUDA]) as p:
        f(x)
        torch.cuda.synchronize()
print("LAUNCHES", sum(1 for e in p.events() if "act_quant_kernel" in e.name))
"""


@pytest.mark.skipif(not _blackwell(), reason = "needs sm_100+ for torchao NVFP4")
@pytest.mark.parametrize("order", [("1", "0"), ("0", "1")])
def test_warm_compile_cache_follows_the_kill_switch(tmp_path, order):
    # A fresh process sharing a warm Inductor cache must run the quantiser its own switch selects.
    pytest.importorskip("triton")
    import os
    import subprocess
    import sys
    from pathlib import Path

    backend = str(Path(aq.__file__).resolve().parents[2])
    env = dict(os.environ)
    env["PYTHONPATH"] = backend + os.pathsep + env.get("PYTHONPATH", "")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(tmp_path / "inductor")
    env["TRITON_CACHE_DIR"] = str(tmp_path / "triton")
    launches = {}
    for switch in order:
        env[aq.FAST_ACT_QUANT_ENV] = switch
        out = subprocess.run(
            [sys.executable, "-c", _CROSS_PROCESS_PROBE],
            env = env,
            capture_output = True,
            text = True,
            timeout = 600,
        )
        assert out.returncode == 0, out.stderr[-2000:]
        launches[switch] = int(out.stdout.split("LAUNCHES")[-1].split()[0])
    assert launches == {"1": 1, "0": 0}
