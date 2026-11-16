# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""The grouped fp8 eval forward traces as one graph: whether transformers' grouped kernel is usable (it reads
the module source) is settled at import, not inside the compiled forward."""

import pytest

torch = pytest.importorskip("torch")


def test_grouped_fp8_eval_forward_compiles_fullgraph(monkeypatch):
    from unsloth.kernels import fp8

    if getattr(fp8, "FP8GroupedLinear", None) is None:
        pytest.skip("this transformers has no FP8GroupedLinear")
    if hasattr(
        fp8._fp8_grouped_eval_kernel_ok, "cache_clear"
    ):  # cold cache on the pre-fix lru_cache helper
        fp8._fp8_grouped_eval_kernel_ok.cache_clear()
    reached = []

    def _upstream(layer, x):  # traceable stand-in for transformers' eval kernel
        reached.append(1)
        return x * 2

    monkeypatch.setattr(fp8, "_fp8_grouped_forward_orig", _upstream)
    monkeypatch.setattr(fp8, "_FP8_GROUPED_EVAL_KERNEL_OK", True, raising = False)

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros(8, 8).to(torch.float8_e4m3fn), requires_grad = False
    )
    layer.eval()
    x = torch.randn(2, 8)

    torch._dynamo.reset()
    out = torch.compile(
        lambda t: fp8._fp8_grouped_forward(layer, t), backend = "eager", fullgraph = True
    )(x)
    torch.testing.assert_close(out, x * 2)
    assert reached, "eval did not take the upstream kernel path"
