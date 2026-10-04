# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The inductor divisibility backport against the exact FLUX.1 CantSplit expression (needs a real torch)."""

import pytest

sizevars = pytest.importorskip("torch._inductor.sizevars")
sympy = pytest.importorskip("sympy")

from core.inference import diffusion_inductor_backports as bp  # noqa: E402


@pytest.fixture
def backport():
    bp.uninstall()
    yield bp
    bp.uninstall()


def test_flux_single_block_split_is_proven(backport):
    s31, s87 = sympy.symbols("s31 s87", integer = True, positive = True)
    backport.install()
    assert backport.proof_available() is True
    allocator = sizevars.SizeVarAllocator()
    # FluxSingleTransformerBlock: cat([attn, mlp], dim=2) over the merged text+image sequence.
    assert allocator.statically_known_multiple_of(15360 * s31 + 15360 * s87, s31 + s87)
    # With the text length static (mark_static on the text dim): still exact.
    assert allocator.statically_known_multiple_of(15360 * s87 + 7864320, s87 + 512)


def test_backport_stays_sound(backport):
    s31, s87 = sympy.symbols("s31 s87", integer = True, positive = True)
    backport.install()
    allocator = sizevars.SizeVarAllocator()
    assert not allocator.statically_known_multiple_of(15360 * s31 + 15359 * s87, s31 + s87)
    assert not allocator.statically_known_multiple_of(15360 * s87 + 7864321, s87 + 512)
    assert not allocator.statically_known_multiple_of(s31 + 1, s31 + s87)


def test_kill_switch_keeps_the_static_fallback(backport, monkeypatch):
    monkeypatch.setenv(bp.BACKPORTS_ENV, "0")
    if bp._stock_proves(sizevars.SizeVarAllocator):
        pytest.skip("this torch proves the split natively")
    assert bp.proof_available() is False
