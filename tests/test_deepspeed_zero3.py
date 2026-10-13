# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""DeepSpeed ZeRO-3 (#225): the load must not pass a device_map transformers refuses."""

import pytest

transformers = pytest.importorskip("transformers")

from unsloth.models import loader_utils


@pytest.fixture
def zero3(monkeypatch):
    def set_enabled(enabled):
        monkeypatch.setattr(
            transformers.integrations, "is_deepspeed_zero3_enabled", lambda: enabled
        )

    return set_enabled


def test_zero3_drops_the_placement_unsloth_picked(zero3):
    zero3(True)
    for device_map in (loader_utils.DEFAULT_DEVICE_MAP, "sequential", "auto", "cuda"):
        assert loader_utils.deepspeed_zero3_device_map(device_map) is None, device_map


@pytest.mark.parametrize(
    "device_map", [{"model.layers.0": 0, "lm_head": 1}, {"": "cuda:1"}, "cuda:1", None]
)
def test_zero3_keeps_a_map_the_caller_chose(zero3, device_map):
    """transformers rejects these under ZeRO-3; silently dropping one would move the run to another card."""
    zero3(True)
    assert loader_utils.deepspeed_zero3_device_map(device_map) is device_map


def test_without_zero3_the_device_map_is_untouched(zero3):
    zero3(False)
    for device_map in (loader_utils.DEFAULT_DEVICE_MAP, "sequential", {"": "cuda:1"}, None):
        assert loader_utils.deepspeed_zero3_device_map(device_map) is device_map
