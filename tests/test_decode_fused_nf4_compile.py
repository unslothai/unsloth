# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import importlib
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch

from real_accelerator import has_real_cuda


def _probe():
    from packaging.version import Version
    from torch._dynamo.utils import counters

    zoo = types.ModuleType("unsloth_zoo")
    zoo.__path__ = []
    utils = types.ModuleType("unsloth_zoo.utils")
    utils.Version = Version
    package = types.ModuleType("nfk")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "unsloth" / "kernels")]
    sys.modules.update({"unsloth_zoo": zoo, "unsloth_zoo.utils": utils, "nfk": package})
    fused = importlib.import_module("nfk.nf4_decode_fused")
    with torch.inference_mode():
        residual = torch.linspace(-2, 2, 256, device = "cuda")
        partials = residual.square().sum().reshape(1)
        weight = torch.linspace(0.5, 1.5, 256, device = "cuda", dtype = torch.float16)
        eager = torch.empty_like(weight)
        output = torch.empty_like(weight)
        fused.final_norm(residual, partials, 1, weight, 1e-5, eager)
        compiled = torch.compile(fused.final_norm, fullgraph = True)
        compiled(residual, partials, 1, weight, 1e-5, output)
        torch.testing.assert_close(output, eager, rtol = 0, atol = 0)
        graphs = counters["stats"]["unique_graphs"]
        assert graphs > 0
        for _ in range(3):
            compiled(residual, partials, 1, weight, 1e-5, output)
            torch.testing.assert_close(output, eager, rtol = 0, atol = 0)
        assert counters["stats"]["unique_graphs"] == graphs

        def plain(site, x, out):
            fused.fused_site(site, x, out, epi = 0, capability = (8, 0))
            return out

        compiled_plain = torch.compile(plain, fullgraph = True, dynamic = True)
        for k in (128, 256, 512):
            site = fused.FusedSite(
                "o",
                torch.zeros(128 * k // 2, device = "cuda", dtype = torch.uint8),
                torch.ones(128 * k // 64, device = "cuda"),
                torch.linspace(-1, 1, 16, device = "cuda"),
                [128],
                k,
                64,
            )
            x = torch.linspace(-1, 2, k, device = "cuda", dtype = torch.float16)
            eager = torch.empty(128, device = "cuda", dtype = torch.float16)
            output = torch.empty_like(eager)
            plain(site, x, eager)
            expected = (-x.float().sum()).to(eager.dtype).expand_as(eager)
            torch.testing.assert_close(eager, expected, rtol = 0, atol = 0)
            compiled_plain(site, x, output)
            torch.testing.assert_close(output, eager, rtol = 0, atol = 0)
            graphs = counters["stats"]["unique_graphs"]
            compiled_plain(site, x, output)
            torch.testing.assert_close(output, eager, rtol = 0, atol = 0)
            assert counters["stats"]["unique_graphs"] == graphs
        assert not counters["graph_break"], dict(counters["graph_break"])
    print("FUSED COMPILE OK")


@pytest.mark.skipif(not has_real_cuda(), reason = "requires a real CUDA GPU")
@pytest.mark.skipif(torch.version.hip is not None, reason = "the fused path is gated off on ROCm")
def test_final_norm_compiles_without_graph_breaks():
    env = dict(os.environ)
    env.pop("TRITON_INTERPRET", None)
    shared = str(Path(__file__).resolve().parent / "_shared")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (shared, env.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, __file__, "--probe"],
        env = env,
        capture_output = True,
        text = True,
        timeout = 300,
    )
    assert result.returncode == 0 and "FUSED COMPILE OK" in result.stdout, (
        result.stdout + result.stderr
    )


if __name__ == "__main__":
    _probe()
