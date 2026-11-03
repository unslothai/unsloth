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
    capability = torch.cuda.get_device_capability()
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
            fused.fused_site(site, x, out, epi = 0, capability = capability)
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

        heads, kv_heads, head_dim, width, length = 2, 1, 64, 256, 16
        cos = torch.ones((length, head_dim), device = "cuda", dtype = torch.float16)
        sin = torch.zeros_like(cos)
        position = torch.tensor([3], device = "cuda")
        slot = torch.tensor([5], device = "cuda")

        def qkv_bias(site, x, query, cache):
            key, value = cache[:, 0], cache[:, 1]
            fused.fused_site(
                site,
                x,
                query,
                epi = fused.EPI_ROPE_KV,
                capability = capability,
                k_cache = key,
                v_cache = value,
                cache_strides = (key.stride(2), key.stride(0)),
                cos = cos,
                sin = sin,
                pos = position,
                slot = slot,
                head_dim = head_dim,
                bias = site.bias,
            )
            return query, cache

        compiled_bias = torch.compile(qkv_bias, fullgraph = True, dynamic = True)
        for separate in (False, True):
            position.fill_(3)
            slot.fill_(5)
            sizes = (heads * head_dim, kv_heads * head_dim, kv_heads * head_dim)
            weights = [torch.zeros(n * width // 2, device = "cuda", dtype = torch.uint8) for n in sizes]
            scales = [torch.ones(n * width // 64, device = "cuda") for n in sizes]
            biases = tuple(
                torch.linspace(0.125 + i, 0.75 + i, n, device = "cuda", dtype = torch.float16)
                for i, n in enumerate(sizes)
            )
            site = fused.FusedSite(
                "qkv",
                torch.cat(weights),
                torch.cat(scales),
                torch.linspace(-1, 1, 16, device = "cuda"),
                [128, 192, 256],
                width,
                64,
                bias = biases,
                segments = [(w, s, None) for w, s in zip(weights, scales)] if separate else None,
            )
            x = torch.zeros(width, device = "cuda", dtype = torch.float16)
            query = torch.empty(heads * head_dim, device = "cuda", dtype = torch.float16)
            eager_query = torch.empty_like(query)
            cache = torch.zeros(
                (length, 2, 1, kv_heads, head_dim), device = "cuda", dtype = torch.float16
            )
            eager_cache = torch.zeros_like(cache)
            qkv_bias(site, x, eager_query, eager_cache)
            compiled_bias(site, x, query, cache)
            torch.testing.assert_close(query, biases[0], rtol = 0, atol = 0)
            torch.testing.assert_close(query, eager_query, rtol = 0, atol = 0)
            torch.testing.assert_close(cache, eager_cache, rtol = 0, atol = 0)
            torch.testing.assert_close(cache[5, 0, 0].reshape(-1), biases[1], rtol = 0, atol = 0)
            torch.testing.assert_close(cache[5, 1, 0].reshape(-1), biases[2], rtol = 0, atol = 0)
            graphs = counters["stats"]["unique_graphs"]
            for p, s in ((7, 8), (9, 11), (4, 6)):
                position.fill_(p)
                slot.fill_(s)
                qkv_bias(site, x, eager_query, eager_cache)
                compiled_bias(site, x, query, cache)
                torch.testing.assert_close(query, eager_query, rtol = 0, atol = 0)
                torch.testing.assert_close(cache, eager_cache, rtol = 0, atol = 0)
                assert counters["stats"]["unique_graphs"] == graphs

        angle = (
            torch.arange(length, device = "cuda")[:, None]
            * (10000 ** (-torch.arange(0, head_dim, 2, device = "cuda").float() / head_dim))[None]
        )
        angle = torch.cat((angle, angle), -1)
        cos.copy_(angle.cos())
        sin.copy_(angle.sin())
        raw = torch.linspace(-2, 3, 256, device = "cuda", dtype = torch.float16)
        raw_before = raw.clone()
        wq = torch.linspace(0.5, 1.5, head_dim, device = "cuda", dtype = torch.float16)
        wk = torch.linspace(1.5, 0.5, head_dim, device = "cuda", dtype = torch.float16)

        def qk_norm(raw, query, cache):
            fused.qk_norm_rope_kv(
                raw,
                query,
                wq,
                wk,
                1e-6,
                3e-6,
                cos,
                sin,
                position,
                slot,
                cache[:, 0],
                cache[:, 1],
                heads,
                kv_heads,
                head_dim,
            )
            return query, cache

        compiled_qk = torch.compile(qk_norm, fullgraph = True, dynamic = True)
        eager_cache.zero_()
        cache.zero_()
        qk_norm(raw, eager_query, eager_cache)
        compiled_qk(raw, query, cache)
        torch.testing.assert_close(query, eager_query, rtol = 0, atol = 0)
        torch.testing.assert_close(cache, eager_cache, rtol = 0, atol = 0)
        torch.testing.assert_close(raw, raw_before, rtol = 0, atol = 0)
        graphs = counters["stats"]["unique_graphs"]
        for p, s in ((3, 4), (8, 10), (11, 12)):
            position.fill_(p)
            slot.fill_(s)
            qk_norm(raw, eager_query, eager_cache)
            compiled_qk(raw, query, cache)
            torch.testing.assert_close(query, eager_query, rtol = 0, atol = 0)
            torch.testing.assert_close(cache, eager_cache, rtol = 0, atol = 0)
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
