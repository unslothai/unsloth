# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A torchao too new for the installed torch reads as absent instead of failing `import unsloth`.

`pip install unsloth` on torch 2.6 resolves torchao 0.18 (unsloth_zoo pins only `torchao>=0.13`),
and transformers' quantizer import then dies on `torch.float4_e2m1fn_x2`. The torchao and torch
versions are simulated, so this runs on any torch, CPU only.
"""

import functools
import os
import subprocess
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import unsloth.import_fixes as import_fixes  # noqa: E402

GPU_INIT = ROOT / "unsloth" / "_gpu_init.py"


def _write_fake_torchao(root, quantization_body):
    package = root / "torchao"
    (package / "quantization").mkdir(parents = True)
    (package / "__init__.py").write_text("")
    (package / "quantization" / "__init__.py").write_text(quantization_body)
    return root


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """Fake torchao on sys.path, real torchao modules and our flag restored afterwards."""
    saved = {n: m for n, m in sys.modules.items() if n == "torchao" or n.startswith("torchao.")}
    for name in saved:
        del sys.modules[name]
    monkeypatch.setattr(import_fixes, "TORCHAO_DISABLED_FOR_OLD_TORCH", False)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    def configure(torch_version, torchao_version, quantization_body):
        import torch

        _write_fake_torchao(tmp_path, quantization_body)
        importlib.invalidate_caches()
        monkeypatch.setattr(torch, "__version__", torch_version)
        monkeypatch.setattr(import_fixes, "importlib_version", lambda name: torchao_version)

    yield configure
    for name in [n for n in list(sys.modules) if n == "torchao" or n.startswith("torchao.")]:
        del sys.modules[name]
    sys.modules.update(saved)
    importlib.invalidate_caches()


MISSING_TORCH_DTYPE = "import torch\nX = torch.float4_e2m1fn_x2_not_in_any_torch\n"


def test_torchao_018_on_torch_26_is_disabled_not_fatal(isolated, capsys):
    isolated("2.6.0+cu124", "0.18.0", MISSING_TORCH_DTYPE)
    assert import_fixes.disable_torchao_if_torch_too_old() is True
    assert sys.modules["torchao"] is None
    assert importlib.util.find_spec("torchao") is None
    with pytest.raises(ImportError):
        import torchao  # noqa: F401
    out = capsys.readouterr().out
    assert "torchao==0.18.0" in out and "torch==2.6.0+cu124" in out
    assert 'pip install "torchao<0.14"' in out, "names the torchao that supports torch 2.6"


def test_torchao_that_imports_is_left_alone(isolated):
    """torchao 0.15 / 0.16 / 0.17 do import on torch 2.6 (with the existing shims): never disable them."""
    isolated("2.6.0", "0.18.0", "")
    assert import_fixes.disable_torchao_if_torch_too_old() is False
    assert sys.modules.get("torchao") is not None
    assert import_fixes.TORCHAO_DISABLED_FOR_OLD_TORCH is False


@pytest.mark.parametrize(
    "torch_version, torchao_version",
    [
        ("2.6.0", "0.13.0"),
        ("2.7.1", "0.15.0"),
        ("2.8.0", "0.16.0"),
        ("2.9.0", "0.18.0"),
        ("2.14.1", "0.18.0"),
    ],
)
def test_supported_pairs_are_not_even_probed(isolated, torch_version, torchao_version):
    isolated(torch_version, torchao_version, MISSING_TORCH_DTYPE)
    assert import_fixes.disable_torchao_if_torch_too_old() is False
    assert "torchao" not in sys.modules, "a supported pair must not import torchao early"


def test_a_real_torchao_failure_is_not_hidden(isolated):
    """Only torch's missing-name wording counts; anything else must surface where it did before."""
    isolated("2.6.0", "0.18.0", "raise RuntimeError('broken torchao build')\n")
    assert import_fixes.disable_torchao_if_torch_too_old() is False
    assert "torchao" not in sys.modules or sys.modules["torchao"] is not None
    assert import_fixes.TORCHAO_DISABLED_FOR_OLD_TORCH is False


def test_missing_torch_import_name_counts_as_skew(isolated):
    isolated("2.7.1", "0.18.0", "from torch.nn.functional import NotInAnyTorchUnslothTest\n")
    assert import_fixes.disable_torchao_if_torch_too_old() is True
    assert sys.modules["torchao"] is None


def test_cached_availability_checks_are_cleared(isolated, monkeypatch):
    """A library imported before unsloth cached True; it must answer False afterwards."""

    @functools.lru_cache
    def is_torchao_available(*args, **kwargs):
        return True

    fake = types.ModuleType("peft.import_utils")
    fake.is_torchao_available = is_torchao_available
    monkeypatch.setitem(sys.modules, "peft.import_utils", fake)
    isolated("2.6.0", "0.18.0", MISSING_TORCH_DTYPE)
    assert import_fixes.disable_torchao_if_torch_too_old() is True
    assert fake.is_torchao_available() is False


def test_calling_twice_is_stable(isolated, capsys):
    isolated("2.6.0", "0.18.0", MISSING_TORCH_DTYPE)
    assert import_fixes.disable_torchao_if_torch_too_old() is True
    assert import_fixes.disable_torchao_if_torch_too_old() is True
    assert sys.modules["torchao"] is None
    assert capsys.readouterr().out.count("disabled for this process") == 1


def test_it_runs_after_the_torchao_shims_and_before_unsloth_zoo():
    source = GPU_INIT.read_text(encoding = "utf-8")
    call = source.index("\ndisable_torchao_if_torch_too_old()\n")
    assert source.index("\nfix_torchao_torch_symbol_skew()\n") < call
    assert source.index("\nfix_torchao_pytree_register_constant()\n") < call
    assert source.index("\nfix_torchao_without_torch_distributed()\n") < call
    assert call < source.index("    import unsloth_zoo\n")
    assert "\ndel disable_torchao_if_torch_too_old\n" in source


def test_disable_records_the_pair_for_subprocesses(isolated, monkeypatch):
    monkeypatch.delenv(import_fixes._TORCHAO_DISABLED_ENV, raising = False)
    isolated("2.6.0", "0.18.0", MISSING_TORCH_DTYPE)
    assert import_fixes.disable_torchao_if_torch_too_old() is True
    assert os.environ[import_fixes._TORCHAO_DISABLED_ENV] == "0.18.0|0.18.0"


def test_propagation_is_staged_when_torchao_was_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(import_fixes, "TORCHAO_DISABLED_FOR_OLD_TORCH", True)
    monkeypatch.setattr(import_fixes, "_subprocess_fix_directory", lambda: str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", "")
    assert import_fixes.propagate_torchao_fix_to_subprocesses() == str(tmp_path)
    assert (tmp_path / "sitecustomize.py").exists()


def _child_sees_torchao(tmp_path, marker):
    (tmp_path / "sitecustomize.py").write_text(import_fixes._subprocess_sitecustomize_source())
    env = dict(os.environ, PYTHONPATH = str(tmp_path))
    env.pop(import_fixes._TORCHAO_DISABLED_ENV, None)
    if marker is not None:
        env[import_fixes._TORCHAO_DISABLED_ENV] = marker
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util; print(importlib.util.find_spec('torchao') is not None)",
        ],
        env = env,
        capture_output = True,
        text = True,
        check = True,
    ).stdout.strip()
    return out == "True"


def test_child_hides_torchao_only_for_the_same_pair(tmp_path):
    from importlib.metadata import version

    if importlib.util.find_spec("torchao") is None:
        pytest.skip("torchao not installed")
    pair = f"{version('torchao')}|{version('torch')}"
    assert _child_sees_torchao(tmp_path, None) is True
    assert _child_sees_torchao(tmp_path, pair) is False
    assert (
        _child_sees_torchao(tmp_path, "0.0.1|" + version("torch")) is True
    ), "another env keeps its torchao"


def test_torch_25_is_told_to_upgrade_torch_not_pin_below_the_zoo_floor(isolated, capsys):
    isolated("2.5.1", "0.18.0", MISSING_TORCH_DTYPE)
    assert import_fixes.disable_torchao_if_torch_too_old() is True
    out = capsys.readouterr().out
    assert "torchao<0.13" not in out and "torch>=2.6" in out
