"""torch >= 2.12 keeps config writes per thread. pp_schedules.allow_split_backward turns AOTAutograd's
donated_buffer off for the zero bubble schedule's retain_graph backward, so that write must reach every
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
