# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Worker entrypoints for test_inference_worker_stderr_tail.py.

A separate module because ``multiprocessing``'s spawn start method imports the target by
name in the child, so the target cannot live in the test module pytest collected.
"""

import sys


def exit_one_after_writing_to_stderr(**_kwargs) -> None:
    """The reported shape: the worker logs, then dies from an unhandled exception (#7843)."""
    print("Received command: generate", file = sys.stderr)
    print("Starting text generation", file = sys.stderr)
    sys.stderr.flush()
    raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")


def record_received_kwargs(**kwargs) -> None:
    """Write the keyword arguments the entrypoint was actually handed, then exit 1."""
    print(f"entrypoint kwargs: {sorted(kwargs)}", file = sys.stderr)
    sys.stderr.flush()
    raise SystemExit(1)


def hold_a_second_handle_on_stderr_then_exit(**_kwargs) -> None:
    """A worker carrying something that dup'd fd 2, which is the ordinary shape of a logging
    handler or a native library. The mirror's pipe then has a writer the teardown cannot
    close, so the pump never sees EOF and the join at exit has to time out."""
    import os

    _keep_alive = os.dup(2)                                 # noqa: F841
    print("held a second handle on stderr", file = sys.stderr)
    sys.stderr.flush()
    raise SystemExit(1)


def write_non_ascii_far_past_the_read_window(**_kwargs) -> None:
    """More stderr than the parent reads back, with non-ASCII on both sides of the cut.

    The parent reads only the last TAIL_READ_BYTES of the sink, at whatever byte offset that
    lands on. This is the worker that makes that offset fall inside a multi-byte character.
    """
    from utils.worker_stderr import TAIL_READ_BYTES

    # Two-byte characters, so some window offset is guaranteed to sever one.
    padding = "é" * 40
    for index in range((TAIL_READ_BYTES // len(padding.encode("utf-8"))) + 200):
        sys.stderr.write(f"{padding} warmup {index}\n")
    sys.stderr.write("RuntimeError: modellädt nicht — CUDA out of memory\n")
    sys.stderr.flush()
    raise SystemExit(1)


def write_far_more_than_the_cap(**_kwargs) -> None:
    """Flood stderr, then die, so the compaction path is exercised for real."""
    from utils.worker_stderr import MIRROR_FILE_CAP_BYTES

    line = "x" * 1023 + "\n"
    for _ in range((MIRROR_FILE_CAP_BYTES * 3) // len(line)):
        sys.stderr.write(line)
    sys.stderr.flush()
    raise RuntimeError("the last line before the worker died")
