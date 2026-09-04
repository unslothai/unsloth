# Copyright 2026-present the Unforgettable contributors.
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

"""MemoryWheels §6 default act/sim policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unforgettable.loop.context import EpisodeRequest, EpisodeState


class Action:
    WORLD_ACT = "world_act"
    ENTER_SIM = "enter_sim"
    CONTINUE_SIM = "continue_sim"
    RETRY_WORLD = "retry_world"
    ESCALATE = "escalate"
    FINISH = "finish"


@dataclass(frozen = True)
class Policy:
    max_clones: int = 1
    max_sim_turns: int = 8
    require_confirm_retry: bool = False


def default_policy() -> Policy:
    return Policy()


def require_confirm_retry(
    *, stakes: str | None, permission_mode: str | None, confirm_retry: bool | None
) -> bool:
    if confirm_retry is False:
        return False
    if confirm_retry is True:
        return True
    if stakes == "high":
        return True
    if permission_mode == "ask":
        return True
    return False


def policy_from_request(request: "EpisodeRequest") -> Policy:
    max_clones = 1
    max_sim_turns = 8
    requested_clones = getattr(request, "max_clones", None)
    requested_turns = getattr(request, "max_sim_turns", None)
    if requested_clones is not None and requested_clones >= 1:
        max_clones = requested_clones
    if requested_turns is not None and requested_turns >= 1:
        max_sim_turns = requested_turns
    return Policy(
        max_clones = max_clones,
        max_sim_turns = max_sim_turns,
        require_confirm_retry = require_confirm_retry(
            stakes = getattr(request, "stakes", None),
            permission_mode = getattr(request, "permission_mode", None),
            confirm_retry = getattr(request, "confirm_retry", None),
        ),
    )


def decide(
    event: str,
    state: "EpisodeState",
    policy: Policy | None = None,
) -> str:
    """event: failure | success | finished."""
    pol = policy or default_policy()
    mode = state.contact
    if event == "finished":
        return Action.FINISH
    if event == "success":
        if mode == "sim" and state.had_world_failure:
            return Action.RETRY_WORLD
        return Action.FINISH
    # failure
    if mode == "world":
        if state.clone_count >= pol.max_clones:
            return Action.ESCALATE
        return Action.ENTER_SIM
    if state.sim_turns >= pol.max_sim_turns:
        return Action.ESCALATE
    return Action.CONTINUE_SIM
