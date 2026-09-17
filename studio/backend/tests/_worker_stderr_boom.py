# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys


def exit_one_after_writing_to_stderr(**_kwargs) -> None:
    print("Received command: generate", file = sys.stderr)
    print("Starting text generation", file = sys.stderr)
    sys.stderr.flush()
    raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")


def record_received_kwargs(**kwargs) -> None:
    print(f"entrypoint kwargs: {sorted(kwargs)}", file = sys.stderr)
    sys.stderr.flush()
    raise SystemExit(1)


def hold_a_second_handle_on_stderr_then_exit(**_kwargs) -> None:
    import os

    _keep_alive = os.dup(2)  # noqa: F841
    print("held a second handle on stderr", file = sys.stderr)
    sys.stderr.flush()
    raise SystemExit(1)


def write_non_ascii_far_past_the_read_window(**_kwargs) -> None:
    from utils.worker_stderr import TAIL_READ_BYTES

    padding = "é" * 40
    for index in range((TAIL_READ_BYTES // len(padding.encode("utf-8"))) + 200):
        sys.stderr.write(f"{padding} warmup {index}\n")
    sys.stderr.write("RuntimeError: modellädt nicht — CUDA out of memory\n")
    sys.stderr.flush()
    raise SystemExit(1)


def write_far_more_than_the_cap(**_kwargs) -> None:
    from utils.worker_stderr import MIRROR_FILE_CAP_BYTES

    line = "x" * 1023 + "\n"
    for _ in range((MIRROR_FILE_CAP_BYTES * 3) // len(line)):
        sys.stderr.write(line)
    sys.stderr.flush()
    raise RuntimeError("the last line before the worker died")
