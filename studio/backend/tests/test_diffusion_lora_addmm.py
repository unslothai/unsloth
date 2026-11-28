# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unfolded LoRA adapters add their delta in the B GEMM's epilogue; every other case keeps PEFT's own forward."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from core.inference import diffusion_lora_addmm as la  # noqa: E402


def _model(
    dtype = torch.float32,
    dropout = 0.0,
    adapters = ("a",),
    use_dora = False,
):
    from peft import LoraConfig, inject_adapter_in_model

    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(64, 96)).to(dtype)
    for i, name in enumerate(adapters):
        cfg = LoraConfig(
            r = 8,
            lora_alpha = 16 + 8 * i,
            lora_dropout = dropout,
            target_modules = ["0"],
            use_dora = use_dora,
        )
        m = inject_adapter_in_model(cfg, m, adapter_name = name)
        torch.nn.init.normal_(m[0].lora_B[name].weight)
    m[0].set_adapter(list(adapters))
    return m.eval()


def _stock(m, x):
    from peft.tuners.lora import layer as lora_layer
    patched = lora_layer.Linear.forward
    try:
        lora_layer.Linear.forward = patched._unsloth_original
        with torch.no_grad():
            return m(x)
    finally:
        lora_layer.Linear.forward = patched


@pytest.fixture(autouse = True)
def _installed(monkeypatch):
    monkeypatch.delenv(la.LORA_ADDMM_ENV, raising = False)
    assert la.install()
    la.CALLS[:] = [0, 0]


@pytest.mark.parametrize("adapters", [("a",), ("a", "b")])
def test_epilogue_matches_peft_fp32(adapters):
    m = _model(adapters = adapters)
    x = torch.randn(3, 5, 64)
    ref = _stock(m, x)
    with torch.no_grad():
        out = m(x)
    assert la.CALLS[0] == 1
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol = 1e-5, atol = 1e-5)


def test_epilogue_bf16_within_one_rounding():
    m = _model(dtype = torch.bfloat16)
    x = torch.randn(4, 64, dtype = torch.bfloat16)
    ref = _stock(m, x)
    with torch.no_grad():
        out = m(x)
    assert la.CALLS[0] == 1 and out.dtype == torch.bfloat16
    # one fp32 sum rounded once vs PEFT's three bf16 roundings: within a bf16 ulp of each other
    torch.testing.assert_close(out.float(), ref.float(), rtol = 2**-7, atol = 2**-6)


def test_base_output_not_written():
    m = _model()
    x = torch.randn(4, 64)
    base = m[0].base_layer
    held = []
    orig = base.forward

    def keep(inp):
        y = orig(inp)
        held.append(y)
        return y

    base.forward = keep
    with torch.no_grad():
        before = orig(x).clone()
        out = m(x)
    assert la.CALLS[0] == 1
    torch.testing.assert_close(held[0], before)
    assert out.data_ptr() != held[0].data_ptr()


@pytest.mark.parametrize("case", ["dropout", "dora", "grad", "disabled", "kill_switch", "kwargs"])
def test_other_cases_run_peft(case, monkeypatch):
    m = _model(dropout = 0.1 if case == "dropout" else 0.0, use_dora = case == "dora")
    x = torch.randn(4, 64)
    if case == "grad":
        x.requires_grad_(True)
        m(x).sum().backward()
        assert la.CALLS == [0, 1]
        return
    if case == "disabled":
        m[0].enable_adapters(False)
    if case == "kill_switch":
        monkeypatch.setenv(la.LORA_ADDMM_ENV, "0")
    ref = _stock(m, x)
    with torch.no_grad():
        out = m[0](x, adapter_names = ["a"] * 4) if case == "kwargs" else m(x)
    assert la.CALLS[0] == 0 and la.CALLS[1] == 1
    if case != "dropout":
        torch.testing.assert_close(out, ref)


def test_install_idempotent():
    from peft.tuners.lora import layer as lora_layer

    first = lora_layer.Linear.forward
    assert la.install()
    assert lora_layer.Linear.forward is first
