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


def write_far_more_than_the_cap(**_kwargs) -> None:
    """Flood stderr, then die, so the compaction path is exercised for real."""
    from utils.worker_stderr import MIRROR_FILE_CAP_BYTES

    line = "x" * 1023 + "\n"
    for _ in range((MIRROR_FILE_CAP_BYTES * 3) // len(line)):
        sys.stderr.write(line)
    sys.stderr.flush()
    raise RuntimeError("the last line before the worker died")
