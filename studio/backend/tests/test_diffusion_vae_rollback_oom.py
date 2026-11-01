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

"""An OOM during the fp32 upcast must roll every weight back, even when the first cast back itself runs out of room."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_vae_upcast as vu  # noqa: E402


class _Budget:
    """A device with ``cap`` units; ``Tensor.to`` charges the copy before the source is freed."""

    def __init__(self, cap: int):
        self.cap, self.used = cap, 0

    def alloc(self, n: int) -> None:
        if self.used + n > self.cap:
            raise torch.OutOfMemoryError("simulated")
        self.used += n


class _Tensor:
    def __init__(self, budget: _Budget, n: int, dtype):
        self.budget, self.n, self.dtype = budget, n, dtype
        self.device = torch.device("cuda:0")
        self.size = n * (2 if dtype == torch.float16 else 4)
        self.shape = (n,)
        self.live = False
        budget.alloc(self.size)
        self.live = True

    def __del__(self):
        if self.live:
            self.budget.used -= self.size

    def is_pinned(self):
        return False

    def is_inference(self):
        return False

    def data_ptr(self):
        return id(self)

    def to(
        self,
        *args,
        dtype = None,
        **kwargs,
    ):
        dtype = dtype or next(a for a in args if isinstance(a, torch.dtype))
        return self if dtype == self.dtype else _Tensor(self.budget, self.n, dtype)


class _Param:
    def __init__(self, t):
        self.data = t

    @property
    def dtype(self):
        return self.data.dtype

    def is_floating_point(self):
        return True


class _Model:
    def __init__(self, budget: _Budget):
        self._buffers = {}
        self._parameters = {
            name: _Param(_Tensor(budget, n, torch.float16))
            for name, n in [("large0", 4), ("bias", 1), ("large1", 4)]
        }

    def modules(self):
        return [self]


def test_rollback_retries_a_cast_back_that_ran_out_of_room():
    # fp16 8 + 2 + 8 = 18 of 34: large0 and bias upcast (28 used), large1 needs 16 -> OOM. Casting large0 back needs 8
    # with 6 free; only after bias goes back (frees 2) does it fit.
    budget = _Budget(34)
    model = _Model(budget)
    with pytest.raises(torch.OutOfMemoryError):
        vu.upcast_keeping_storage(model)
    dtypes = {name: p.dtype for name, p in model._parameters.items()}
    assert dtypes == {name: torch.float16 for name in dtypes}
    assert budget.used == 18
