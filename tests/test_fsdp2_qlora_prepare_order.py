# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The FSDP2 prepare hook falls back to uint8 4-bit storage before unifying frozen dtypes (torch < 2.8,
forced here), so float-packed weights from an explicit bnb_4bit_quant_storage do not trip it."""

import types

import pytest

torch = pytest.importorskip("torch")
bnb = pytest.importorskip("bitsandbytes")
pytest.importorskip("accelerate")

from unsloth.models import fsdp2, fsdp2_qlora  # noqa: E402


def test_old_torch_float_packed_qlora_without_mixed_precision(monkeypatch):
    import accelerate.accelerator as accelerator_module

    torch.manual_seed(0)
    lin = bnb.nn.Linear4bit(
        128,
        64,
        bias = False,
        compute_dtype = torch.bfloat16,
        quant_type = "nf4",
        quant_storage = torch.bfloat16,
    )
    lin.weight = bnb.nn.Params4bit(
        torch.randn(64, 128) * 0.05,
        requires_grad = False,
        quant_type = "nf4",
        quant_storage = torch.bfloat16,
        blocksize = 64,
        module = lin,
    )
    model = torch.nn.Sequential(lin.to("cpu"), torch.nn.Linear(64, 8))
    assert lin.weight.dtype == torch.bfloat16 and lin.weight.bnb_quantized

    seen = {}

    def stub(accelerator, model):
        seen["dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
        return model

    monkeypatch.setattr(accelerator_module, "fsdp2_prepare_model", stub)
    monkeypatch.setattr(fsdp2, "_fsdp2_checks_frozen_dtypes", lambda: True)
    monkeypatch.setattr(fsdp2_qlora, "_fsdp2_checks_frozen_dtypes", lambda: True)
    assert fsdp2.install_fsdp2_prepare_hook()
    plugin = types.SimpleNamespace(mixed_precision_policy = None)
    accelerator = types.SimpleNamespace(
        state = types.SimpleNamespace(fsdp_plugin = plugin), mixed_precision = "no"
    )
    accelerator_module.fsdp2_prepare_model(accelerator, model)
    assert seen["dtypes"] == ["torch.float32", "torch.uint8"], seen
