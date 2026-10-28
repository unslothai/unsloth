# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A torchao too new for the installed torch reads as absent instead of failing `import unsloth`.

`pip install unsloth` on torch 2.6 resolves torchao 0.18 (unsloth_zoo pins only `torchao>=0.13`),
and transformers' quantizer import then dies on `torch.float4_e2m1fn_x2`. The torchao and torch
versions are simulated, so this runs on any torch, CPU only.
"""

import functools
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
    source = GPU_INIT.read_text()
    call = source.index("\ndisable_torchao_if_torch_too_old()\n")
    assert source.index("\nfix_torchao_torch_symbol_skew()\n") < call
    assert source.index("\nfix_torchao_pytree_register_constant()\n") < call
    assert source.index("\nfix_torchao_without_torch_distributed()\n") < call
    assert call < source.index("    import unsloth_zoo\n")
    assert "\ndel disable_torchao_if_torch_too_old\n" in source
