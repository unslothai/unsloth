# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""torchao < 0.15 names ``FqnToConfig`` ``ModuleFqnToConfig``; diffusers 0.41 imports the new name unguarded."""

from __future__ import annotations

import importlib.metadata
import sys
import types

import pytest

from core import _torchao_stub


class _ModuleFqnToConfig:
    pass


@pytest.fixture
def fake_torchao(monkeypatch):
    """A torchao package whose quantization module has only the pre-0.15 name, at a chosen version."""
    pkg = types.ModuleType("torchao")
    quant = types.ModuleType("torchao.quantization")
    quant.ModuleFqnToConfig = _ModuleFqnToConfig
    pkg.quantization = quant
    monkeypatch.setitem(sys.modules, "torchao", pkg)
    monkeypatch.setitem(sys.modules, "torchao.quantization", quant)

    def use(version):
        def _version(name):
            if name != "torchao":
                return importlib.metadata.version(name)
            if version is None:
                raise importlib.metadata.PackageNotFoundError(name)
            return version

        monkeypatch.setattr(_torchao_stub.importlib.metadata, "version", _version)
        return quant

    return use


@pytest.mark.parametrize("version", ["0.14.0", "0.14.1", "0.13.0", "0.14.0+cu126", "0.14.0.dev20251001"])
def test_an_old_torchao_gains_the_new_name(fake_torchao, version):
    quant = fake_torchao(version)
    _torchao_stub.alias_torchao_fqn_to_config()
    from torchao.quantization import FqnToConfig

    assert FqnToConfig is quant.ModuleFqnToConfig
    _torchao_stub.alias_torchao_fqn_to_config()  # idempotent
    assert quant.FqnToConfig is quant.ModuleFqnToConfig


@pytest.mark.parametrize("version", ["0.15.0", "0.17.0+cu130", "0.18.0", "1.0.0"])
def test_a_new_torchao_is_left_alone(fake_torchao, version):
    quant = fake_torchao(version)
    _torchao_stub.alias_torchao_fqn_to_config()
    assert not hasattr(quant, "FqnToConfig")


def test_a_new_torchao_is_not_even_imported(monkeypatch):
    monkeypatch.setattr(_torchao_stub.importlib.metadata, "version", lambda name: "0.18.0")
    monkeypatch.delitem(sys.modules, "torchao", raising = False)
    monkeypatch.delitem(sys.modules, "torchao.quantization", raising = False)
    _torchao_stub.alias_torchao_fqn_to_config()
    assert "torchao.quantization" not in sys.modules


def test_absent_or_stubbed_torchao_is_a_no_op(fake_torchao, monkeypatch):
    quant = fake_torchao(None)
    _torchao_stub.alias_torchao_fqn_to_config()
    assert not hasattr(quant, "FqnToConfig")
    fake_torchao("0.14.0")
    monkeypatch.setattr(_torchao_stub, "is_stubbed", lambda name: name == "torchao")
    _torchao_stub.alias_torchao_fqn_to_config()
    assert not hasattr(quant, "FqnToConfig")


def test_an_existing_fqn_to_config_is_kept(fake_torchao):
    quant = fake_torchao("0.14.0")
    sentinel = type("FqnToConfig", (), {})
    quant.FqnToConfig = sentinel
    _torchao_stub.alias_torchao_fqn_to_config()
    assert quant.FqnToConfig is sentinel


@pytest.mark.parametrize("module", ["run", "core.inference.diffusion", "core.training.diffusion_train_common"])
def test_every_diffusers_importer_aliases_after_the_rocm_stub(module):
    """Static order check: each module-level hook block runs the alias right after the torchao stub."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.find_spec(module)
    src = Path(spec.origin).read_text(encoding = "utf-8")
    assert "\ninstall_torchao_windows_rocm_stub()\nalias_torchao_fqn_to_config()\n" in src
