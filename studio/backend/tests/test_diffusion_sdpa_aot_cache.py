# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AOTAutogradCache for denoiser graphs that hold an ``sdpa_kernel`` block.

CPU only, checked with a real inductor compile (an AOTAutograd cache hit after a dynamo reset).
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_compile_config as compile_config  # noqa: E402
from core.inference import diffusion_speed as speed  # noqa: E402

HELPERS = ("torch.nn.attention._backend_from_string", "torch.nn.attention._sdpa_kernel")


@pytest.fixture(autouse = True)
def _reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.fixture
def _restore_cacheable():
    cfg = torch._inductor.config
    before = dict(cfg.unsafe_marked_cacheable_functions)
    recorded = compile_config.recorded()
    yield
    cfg.unsafe_marked_cacheable_functions = before
    compile_config._reset_for_tests()
    for (module_name, attr), value in recorded.items():
        compile_config.set_knob(module_name, attr, value)


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(16, 16)

    def forward(self, x):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q = self.lin(x)
        with sdpa_kernel(SDPBackend.MATH):
            out = torch.nn.functional.scaled_dot_product_attention(q, q, q)
        return out + x


class _Dit(torch.nn.Module):
    _repeated_blocks = ["_Block"]

    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block(), _Block()])
        self.compiled_kwargs = None

    def compile_repeated_blocks(self, **kwargs):
        self.compiled_kwargs = kwargs


def test_regional_compile_marks_the_sdpa_helpers_cacheable(_restore_cacheable, monkeypatch):
    # Every regional compile goes through _compile_repeated_blocks; that is where the helpers must become cacheable.
    monkeypatch.delenv("UNSLOTH_DIFFUSION_SDPA_AOT_CACHE", raising = False)
    torch._inductor.config.unsafe_marked_cacheable_functions = {}
    pipe = types.SimpleNamespace(transformer = _Dit())
    speed._compile_repeated_blocks(pipe, None)
    marked = torch._inductor.config.unsafe_marked_cacheable_functions
    for name in HELPERS:
        assert name in marked
        assert torch.__version__ in marked[name]


def test_sdpa_aot_cache_kill_switch(_restore_cacheable, monkeypatch):
    from core.inference import diffusion_aot_cache

    monkeypatch.setenv("UNSLOTH_DIFFUSION_SDPA_AOT_CACHE", "0")
    torch._inductor.config.unsafe_marked_cacheable_functions = {}
    assert diffusion_aot_cache.install() is False
    assert torch._inductor.config.unsafe_marked_cacheable_functions == {}


def test_sdpa_aot_cache_key_carries_the_priority_order(_restore_cacheable, monkeypatch):
    from core.inference import diffusion_aot_cache

    monkeypatch.delenv("UNSLOTH_DIFFUSION_SDPA_AOT_CACHE", raising = False)
    torch._inductor.config.unsafe_marked_cacheable_functions = {"user.fn": "1"}
    assert diffusion_aot_cache.install() is True
    marked = dict(torch._inductor.config.unsafe_marked_cacheable_functions)
    assert marked["user.fn"] == "1"  # a user's own entries survive
    priority = tuple(int(b) for b in torch._C._get_sdp_priority_order())
    assert f"sdp_priority={priority}" in marked[HELPERS[0]]


def test_sdpa_kernel_graph_hits_the_aot_cache_after_a_restart(
    _restore_cacheable, monkeypatch, tmp_path
):
    # A dynamo reset is a fresh process for dynamo; the second compile must be served by AOTAutogradCache, not bypass it.
    from torch._dynamo.utils import counters
    from core.inference import diffusion_aot_cache

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("UNSLOTH_DIFFUSION_SDPA_AOT_CACHE", raising = False)
    torch._inductor.config.unsafe_marked_cacheable_functions = {}
    assert diffusion_aot_cache.install() is True
    compile_config.apply()
    block = _Block().eval()
    x = torch.randn(1, 4, 16)
    with torch.no_grad():
        ref = block(x)
        counters.clear()
        first = torch.compile(block, backend = "inductor")(x)
        assert counters["aot_autograd"]["autograd_cache_bypass"] == 0
        assert counters["aot_autograd"]["autograd_cache_miss"] >= 1
        torch._dynamo.reset()
        counters.clear()
        second = torch.compile(block, backend = "inductor")(x)
    assert counters["aot_autograd"]["autograd_cache_hit"] >= 1
    assert counters["aot_autograd"]["autograd_cache_bypass"] == 0
    # The cache hit must serve the artifact the first compile built.
    assert torch.equal(second, first)
    torch.testing.assert_close(first, ref)
