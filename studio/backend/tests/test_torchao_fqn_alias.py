# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""torchao < 0.15 names ``FqnToConfig`` ``ModuleFqnToConfig``; diffusers 0.41 imports the new name unguarded."""

from __future__ import annotations

import importlib.metadata
import sys
import types

import pytest

from core import _torchao_stub


_PACKAGES = ("diffusers", "torchao")


def _loaded_packages() -> dict:
    return {n: m for n, m in sys.modules.items() if n.split(".")[0] in _PACKAGES}


@pytest.fixture(autouse = True)
def _restore_diffusers_and_torchao_modules():
    """Restore sys.modules for diffusers/torchao: monkeypatch.delitem misses modules not yet imported."""
    before = _loaded_packages()
    yield
    for name in list(_loaded_packages()):
        if before.get(name) is not sys.modules.get(name):
            sys.modules.pop(name, None)
    sys.modules.update(before)


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


@pytest.mark.parametrize(
    "version", ["0.14.0", "0.14.1", "0.13.0", "0.14.0+cu126", "0.14.0.dev20251001"]
)
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
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    monkeypatch.setattr(_torchao_stub.importlib.metadata, "version", lambda name: "0.18.0")
    monkeypatch.delitem(sys.modules, "torchao", raising = False)
    monkeypatch.delitem(sys.modules, "torchao.quantization", raising = False)
    monkeypatch.setitem(sys.modules, "diffusers", types.ModuleType("diffusers"))
    _torchao_stub.alias_torchao_fqn_to_config()
    assert "torchao.quantization" not in sys.modules


def _drop_diffusers_and_torchao(monkeypatch):
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    for name in [n for n in sys.modules if n.split(".")[0] in ("diffusers", "torchao")]:
        monkeypatch.delitem(sys.modules, name)


def test_an_old_torchao_is_aliased_on_the_first_diffusers_import_not_before(monkeypatch, tmp_path):
    """run.py and every spawn child call this; importing torchao there would load torch for nothing."""
    _drop_diffusers_and_torchao(monkeypatch)
    pkg = tmp_path / "torchao"
    (pkg / "quantization").mkdir(parents = True)
    (pkg / "__init__.py").write_text("")
    (pkg / "quantization" / "__init__.py").write_text("class ModuleFqnToConfig:\n    pass\n")
    (tmp_path / "diffusers").mkdir()
    (tmp_path / "diffusers" / "__init__.py").write_text(
        "from torchao.quantization import FqnToConfig\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(_torchao_stub.importlib.metadata, "version", lambda name: "0.14.0")

    _torchao_stub.alias_torchao_fqn_to_config()
    _torchao_stub.alias_torchao_fqn_to_config()
    assert "torchao" not in sys.modules
    assert sum(isinstance(f, _torchao_stub._FqnAliasBeforeDiffusers) for f in sys.meta_path) == 1

    import diffusers

    assert diffusers.FqnToConfig is sys.modules["torchao.quantization"].ModuleFqnToConfig


def test_a_failing_alias_never_breaks_the_diffusers_import(monkeypatch):
    _drop_diffusers_and_torchao(monkeypatch)

    def boom():
        raise RuntimeError("unreadable metadata")

    monkeypatch.setattr(_torchao_stub, "_alias_fqn_to_config_now", boom)
    assert _torchao_stub._FqnAliasBeforeDiffusers().find_spec("diffusers") is None


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


@pytest.mark.parametrize(
    "module", ["run", "core.inference.diffusion", "core.training.diffusion_train_common"]
)
def test_every_diffusers_importer_aliases_after_the_rocm_stub(module):
    """Static order check: each module-level hook block runs the alias right after the torchao stub."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.find_spec(module)
    src = Path(spec.origin).read_text(encoding = "utf-8")
    assert "\ninstall_torchao_windows_rocm_stub()\nalias_torchao_fqn_to_config()\n" in src
