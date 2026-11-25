# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Native FSDP2 / TP resume must refuse torch.load of pickled checkpoint files on torch < 2.6
(CVE-2025-32434), like transformers' check_torch_load_is_safe does on the Trainer's own paths."""

import importlib.util
import os
import sys
import types

import pytest
import torch


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    """The module from this tree without importing the unsloth package (which needs a GPU)."""
    for pkg_name, sub in (("unsloth", ("unsloth",)), ("unsloth.models", ("unsloth", "models"))):
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [os.path.join(REPO, *sub)]
            sys.modules[pkg_name] = pkg
    if name not in sys.modules:
        path = os.path.join(REPO, *name.split(".")) + ".py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    assert sys.modules[name].__file__.startswith(REPO)
    return sys.modules[name]


@pytest.fixture
def old_torch(monkeypatch):
    calls = []
    monkeypatch.setattr(torch, "__version__", "2.5.1+cu124")
    monkeypatch.setattr(torch, "load", lambda *a, **k: calls.append(a) or {})
    return calls


def test_fsdp2_read_checkpoint_weights_refuses(tmp_path, old_torch):
    m = _load("unsloth.models.fsdp2_native")
    (tmp_path / "pytorch_model.bin").write_bytes(b"x")
    with pytest.raises(RuntimeError, match = "CVE-2025-32434"):
        m._read_checkpoint_weights(str(tmp_path))
    assert old_torch == []


def test_fsdp2_optimizer_refuses(tmp_path, old_torch):
    m = _load("unsloth.models.fsdp2_native")
    p = tmp_path / "optimizer.pt"
    p.write_bytes(b"x")
    with pytest.raises(RuntimeError, match = "CVE-2025-32434"):
        m.native_load_optimizer(None, None, str(p))
    assert old_torch == []


def test_tp_resume_payload_refuses(tmp_path, old_torch, monkeypatch):
    m = _load("unsloth.models.tp_native")
    p = tmp_path / "rank0.pt"
    p.write_bytes(b"x")
    monkeypatch.setattr(m, "_ckpt_path", lambda checkpoint, rank: str(p))

    class T:
        pass

    with pytest.raises(RuntimeError, match = "CVE-2025-32434"):
        m._tp_resume_payload(T(), str(tmp_path))
    assert old_torch == []


def test_new_torch_still_loads(tmp_path, monkeypatch):
    m = _load("unsloth.models.fsdp2_native")
    monkeypatch.setattr(torch, "__version__", "2.6.0")
    torch.save({"a": torch.zeros(1)}, tmp_path / "pytorch_model.bin")
    sd, is_adapter = m._read_checkpoint_weights(str(tmp_path))
    assert "a" in sd and is_adapter is False
