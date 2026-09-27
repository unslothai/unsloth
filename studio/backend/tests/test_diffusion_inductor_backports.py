# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``diffusion_inductor_backports.py`` (the torch 2.14 symbolic divisibility backport). CPU only."""

from __future__ import annotations

import pytest

from core.inference import diffusion_inductor_backports as bp

torch = pytest.importorskip("torch")
sympy = pytest.importorskip("sympy")
sizevars = pytest.importorskip("torch._inductor.sizevars")


@pytest.fixture(autouse = True)
def _restore(monkeypatch):
    monkeypatch.delenv(bp.BACKPORTS_ENV, raising = False)
    bp.uninstall()
    # Exercise the patched method on every torch, including ones whose stock check already proves the probe.
    monkeypatch.setattr(bp, "_stock_proves", lambda _cls: False)
    yield
    bp.uninstall()


def _syms():
    return sympy.symbols("s87 s89", integer = True, positive = True)


def _proves(num, den) -> bool:
    return bool(sizevars.SizeVarAllocator().statically_known_multiple_of(num, den))


def test_proves_the_qwen21_and_flux_cantsplit_expressions():
    a, b = _syms()
    bp.install()
    # Qwen-Image-2.1 torchao (text + target attention output) and FLUX.1 single-block (attn + mlp cat) shapes.
    assert _proves(4096 * a - 4096 * b, a - b)
    assert _proves(15360 * a + 15360 * b, a + b)
    assert _proves(4096 * a - 4096 * b, b - a)  # sign of the group does not matter for divisibility


def test_never_proves_a_non_multiple():
    a, b = _syms()
    bp.install()
    assert not _proves(4096 * a - 4095 * b, a - b)
    assert not _proves(4096 * a, a + b)
    assert not _proves(a * b + 1, a)


def test_integer_denominators_keep_the_stock_answer():
    a, _b = _syms()
    bp.install()
    assert _proves(4096 * a, 64)
    assert not _proves(4096 * a + 3, 64)


def test_install_is_idempotent_and_uninstall_restores():
    stock = sizevars.SizeVarAllocator.statically_known_multiple_of
    first = bp.install()
    assert first is True
    patched = sizevars.SizeVarAllocator.statically_known_multiple_of
    assert bp.install() is first
    assert sizevars.SizeVarAllocator.statically_known_multiple_of is patched
    bp.uninstall()
    assert sizevars.SizeVarAllocator.statically_known_multiple_of is stock
    assert not bp.is_installed()


def test_kill_switch(monkeypatch):
    monkeypatch.setenv(bp.BACKPORTS_ENV, "0")
    stock = sizevars.SizeVarAllocator.statically_known_multiple_of
    assert bp.install() is False
    assert sizevars.SizeVarAllocator.statically_known_multiple_of is stock


def test_torch_that_already_proves_it_is_left_alone(monkeypatch):
    monkeypatch.setattr(bp, "_stock_proves", lambda _cls: True)
    stock = sizevars.SizeVarAllocator.statically_known_multiple_of
    assert bp.install() is False
    assert sizevars.SizeVarAllocator.statically_known_multiple_of is stock
    assert not bp.is_installed()


def test_stock_probe_matches_torch_version(monkeypatch):
    """Regressed in 2.12.0 (pytorch#177051), fixed in 2.14.0 (pytorch#184566); the probe, not the version, decides."""
    monkeypatch.undo()
    major_minor = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    proves = bp._stock_proves(sizevars.SizeVarAllocator)
    if (2, 12) <= major_minor < (2, 14):
        assert not proves
    else:
        assert proves


def test_a_failing_proof_returns_the_stock_answer(monkeypatch):
    a, b = _syms()
    stock = _proves(4096 * a - 4096 * b, a - b)
    bp.install()

    def boom(*_a, **_k):
        raise RuntimeError("sympy exploded")

    monkeypatch.setattr(bp, "_gcd_proves_multiple", boom)
    assert _proves(4096 * a - 4096 * b, a - b) == stock


def test_regional_compile_and_vae_compile_install_the_backport(monkeypatch):
    import types

    from core.inference import diffusion_speed as ds_mod

    calls: list = []
    monkeypatch.setattr(bp, "install", lambda logger = None: calls.append("install") or True)

    class _Dit(torch.nn.Module):
        _repeated_blocks: list = []

        def compile_repeated_blocks(self, **kwargs):
            calls.append(("compile", kwargs.get("dynamic")))

    pipe = types.SimpleNamespace(transformer = _Dit())
    monkeypatch.setattr(ds_mod, "_inductor_config", lambda: None)
    monkeypatch.setattr(ds_mod, "guard_compiled_blocks", lambda transformer, logger = None: 0)
    assert ds_mod._compile_repeated_blocks(pipe, None) is True
    # Installed BEFORE the (lazy) compile is requested, so the first forward already lowers with it.
    assert calls[0] == "install" and calls[1][0] == "compile"

    calls.clear()
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)
    vae = types.SimpleNamespace(decode = lambda x: x)
    assert ds_mod._compile_vae_decode(types.SimpleNamespace(vae = vae), None) is True
    assert calls == ["install"]


def test_a_broken_backport_never_fails_the_compile(monkeypatch):
    import types

    from core.inference import diffusion_speed as ds_mod

    def boom(logger = None):
        raise ImportError("no inductor")

    monkeypatch.setattr(bp, "install", boom)
    assert ds_mod._install_inductor_backports(None) is False
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: fn)
    vae = types.SimpleNamespace(decode = lambda x: x)
    assert ds_mod._compile_vae_decode(types.SimpleNamespace(vae = vae), None) is True
