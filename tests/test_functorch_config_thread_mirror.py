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

"""torch >= 2.12 keeps config writes per thread. pp_schedules.split_backward_scope turns AOTAutograd's
donated_buffer off during the zero bubble schedule's retain_graph backward, so that write must reach every
thread (autograd's device threads compile there), like the dynamo / inductor writes already do."""

import threading
from pathlib import Path

import pytest
import torch

import unsloth  # noqa: F401  (applies fix_dynamo_config_thread_visibility)
from unsloth import import_fixes
from unsloth_zoo.utils import Version

functorch_config = pytest.importorskip("torch._functorch.config")


def _read_in_thread(name):
    seen = []
    thread = threading.Thread(target = lambda: seen.append(getattr(functorch_config, name)))
    thread.start()
    thread.join()
    return seen[0]


@pytest.mark.skipif(
    Version(torch.__version__) < Version("2.12.0"), reason = "config is process-wide before 2.12"
)
def test_donated_buffer_write_reaches_other_threads():
    assert Path(import_fixes.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1])
    import_fixes.fix_dynamo_config_thread_visibility()
    if not hasattr(functorch_config, "donated_buffer"):
        pytest.skip("no donated_buffer on this torch")
    before = functorch_config.donated_buffer
    try:
        functorch_config.donated_buffer = not before
        assert _read_in_thread("donated_buffer") is (not before)
    finally:
        functorch_config.donated_buffer = before
    assert _read_in_thread("donated_buffer") is before
