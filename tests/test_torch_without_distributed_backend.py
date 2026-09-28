# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""torch builds without a distributed backend (AMD's Windows ROCm torch 2.11).

Such a torch ships no ``torch._C._distributed_c10d``, so ``torch.distributed.is_available()`` is
False and anything importing ``torch.distributed.distributed_c10d`` raises. accelerate 1.15.0's
``model_has_dtensor`` does that at Trainer start (huggingface/accelerate#4249). Simulated on any
torch; on one without the backend `import unsloth` has already applied the fix, so it is unwrapped.
"""

from __future__ import annotations

import importlib.abc
import sys

import pytest

import unsloth  # noqa: F401
from unsloth import import_fixes

torch = pytest.importorskip("torch")

_C10D = "torch._C._distributed_c10d"


def _missing_c10d():
    return ModuleNotFoundError(
        f"No module named '{_C10D}'; 'torch._C' is not a package", name = _C10D
    )


class _Raises(importlib.abc.MetaPathFinder):
    def __init__(self, prefix, error):
        self.prefix, self.error, self.hits = prefix, error, 0

    def find_spec(
        self,
        name,
        path = None,
        target = None,
    ):
        if name == self.prefix or name.startswith(self.prefix + "."):
            self.hits += 1
            raise self.error
        return None


@pytest.fixture
def no_backend(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: False)


@pytest.fixture
def accelerate_dtensor(monkeypatch):
    other = pytest.importorskip("accelerate.utils.other")
    import accelerate.accelerator
    import accelerate.utils

    original = getattr(other.model_has_dtensor, "__wrapped__", other.model_has_dtensor)
    for module in (other, accelerate.utils, accelerate.accelerator):
        monkeypatch.setattr(module, "model_has_dtensor", original)
    # What `from torch.distributed.tensor import DTensor` does on a torch without the backend.
    for name in [
        n
        for n in sys.modules
        if n.startswith("torch.distributed.tensor") or n.startswith("torch.distributed._tensor")
    ]:
        monkeypatch.delitem(sys.modules, name)
    for attr in ("tensor", "_tensor"):
        monkeypatch.delattr(torch.distributed, attr, raising = False)
    monkeypatch.setattr(
        sys,
        "meta_path",
        [
            _Raises("torch.distributed.tensor", _missing_c10d()),
            _Raises("torch.distributed._tensor", _missing_c10d()),
            *sys.meta_path,
        ],
    )
    return original


def test_accelerate_dtensor_check_without_backend(accelerate_dtensor, no_backend):
    import accelerate.accelerator
    import accelerate.utils
    import accelerate.utils.other as other

    model = torch.nn.Linear(2, 2)
    try:
        unpatched = accelerate_dtensor(model)
    except ImportError:
        unpatched = "raises"
    assert import_fixes.fix_accelerate_dtensor_check_without_torch_distributed() is True
    assert other.model_has_dtensor(model) is False
    assert accelerate.utils.model_has_dtensor is other.model_has_dtensor
    assert accelerate.accelerator.model_has_dtensor is other.model_has_dtensor
    assert other.model_has_dtensor.__wrapped__ is accelerate_dtensor
    assert import_fixes.fix_accelerate_dtensor_check_without_torch_distributed() is True
    assert other.model_has_dtensor.__wrapped__ is accelerate_dtensor
    # accelerate releases before huggingface/accelerate#4250 raise here; later ones return False.
    assert unpatched in ("raises", False)


def test_accelerate_untouched_with_backend(accelerate_dtensor, monkeypatch):
    import accelerate.utils.other as other

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)

    assert import_fixes.fix_accelerate_dtensor_check_without_torch_distributed() is False
    assert other.model_has_dtensor is accelerate_dtensor


def test_other_import_errors_still_raise(monkeypatch, no_backend):
    other = pytest.importorskip("accelerate.utils.other")

    def broken(model):
        raise ModuleNotFoundError("No module named 'numpy'", name = "numpy")

    monkeypatch.setattr(other, "model_has_dtensor", broken)
    assert import_fixes.fix_accelerate_dtensor_check_without_torch_distributed() is True
    with pytest.raises(ModuleNotFoundError, match = "numpy"):
        other.model_has_dtensor(torch.nn.Linear(2, 2))
