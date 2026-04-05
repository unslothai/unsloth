# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Optional attention residual hooks used by attention dispatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from torch import Tensor

AttnResKernelHook = Callable[[Tensor], Tensor]


@dataclass
class AttnResConfig:
    enabled: bool = False
    kernel_hook: Optional[AttnResKernelHook] = None


@dataclass
class AttnResState:
    config: AttnResConfig = field(default_factory = AttnResConfig)
    kernel_hook: Optional[AttnResKernelHook] = None
    enabled: bool = False

    def __post_init__(self):
        if self.kernel_hook is None:
            self.kernel_hook = self.config.kernel_hook
        self.enabled = bool(
            self.enabled or self.config.enabled or self.kernel_hook is not None
        )

    def begin(self) -> "AttnResState":
        self.enabled = bool(
            self.enabled or self.config.enabled or self.kernel_hook is not None
        )
        return self

    def end(self) -> None:
        self.enabled = False

    def apply(self, output: Tensor) -> Tensor:
        return apply_attnres(output, self)


def begin_attnres(
    config: Optional[AttnResConfig] = None,
    *,
    kernel_hook: Optional[AttnResKernelHook] = None,
) -> AttnResState:
    if config is None:
        config = AttnResConfig()
    state = AttnResState(config = config, kernel_hook = kernel_hook)
    return state.begin()


def end_attnres(state: Optional[AttnResState]) -> None:
    if state is None:
        return
    state.end()


def apply_attnres(
    output: Tensor,
    state: Optional[AttnResState] = None,
    *,
    kernel_hook: Optional[AttnResKernelHook] = None,
) -> Tensor:
    if state is None or not state.enabled:
        return output

    hook = kernel_hook
    if hook is None:
        hook = state.kernel_hook
    if hook is None:
        hook = state.config.kernel_hook
    if hook is None:
        return output
    return hook(output)


__all__ = [
    "AttnResConfig",
    "AttnResState",
    "begin_attnres",
    "end_attnres",
    "apply_attnres",
]
