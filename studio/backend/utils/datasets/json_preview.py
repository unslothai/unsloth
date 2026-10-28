# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep native JSON reader failures outside the HTTP server process."""

from __future__ import annotations

import json
import multiprocessing as mp
import tempfile
import time
from contextlib import closing
from pathlib import Path

from core.training.account_jobs import init_job_owner, owned_job
from loggers import get_logger

logger = get_logger(__name__)
_CTX = mp.get_context("spawn")
_PREVIEW_TIMEOUT_S = 60.0
_REAP_TIMEOUT_S = 3.0
_NO_ACCELERATOR_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
    "ONEAPI_DEVICE_SELECTOR": "opencl:cpu",
    "UNSLOTH_ALLOW_CPU": "1",
}


class _JsonPreviewJob:
    def __init__(self):
        self._proc = None
        init_job_owner(self, self._alive, self._stop)

    def _alive(self):
        return self._proc is not None and self._proc.is_alive()

    def _stop(self):
        proc = self._proc
        if proc is None or proc.pid is None:
            return
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout = _REAP_TIMEOUT_S)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout = _REAP_TIMEOUT_S)

    @owned_job()
    def read(self, path: Path, preview_size: int, load_kwargs: dict):
        from core.training.account_jobs import account_process_spec
        from utils.hf_cache_settings import child_environment_for_spawn, get_hf_cache_paths
        from utils.native_path_leases import (
            STDERR_MIRROR_KWARG,
            native_path_secret_removed_for_child_start,
            run_without_native_path_secret,
        )
        from utils.paths import tmp_root
        from utils.process_lifetime import (
            adopt_pid,
            forget_pid,
            is_process_shutting_down,
            spawn_on_lifetime_thread,
        )
        from utils.worker_stderr import WorkerStderrCapture, first_crash_line, format_exit_code

        if is_process_shutting_down():
            raise ValueError("Unsloth is shutting down.")
        scratch_root = tmp_root()
        scratch_root.mkdir(parents = True, exist_ok = True)
        # File IPC lets the parent wait for exit without blocking a child writing a large preview.
        # The Arrow stream is read only after a successful exit, never after a partial native write.
        with tempfile.TemporaryDirectory(
            prefix = "unsloth-json-preview-", dir = scratch_root
        ) as scratch:
            with closing(WorkerStderrCapture(directory = scratch)) as stderr:
                cache_env = get_hf_cache_paths().child_env({})
                worker_env = {**cache_env, **_NO_ACCELERATOR_ENV}
                if load_kwargs.get("features") is None:
                    # Inferred JSON columns contain ordinary values, never Audio/Video features.
                    # datasets otherwise imports torch even when no tensor formatting is requested.
                    worker_env.update(USE_TORCH = "0", USE_TF = "0")
                process_args, process_kwargs = account_process_spec(
                    "utils.datasets.json_preview_worker",
                    "read_json_preview",
                    worker_env,
                    {
                        "path": str(path),
                        "preview_size": preview_size,
                        "load_kwargs": load_kwargs,
                        "output_dir": scratch,
                        STDERR_MIRROR_KWARG: stderr.path,
                    },
                )
                proc = self._proc = _CTX.Process(
                    target = run_without_native_path_secret,
                    args = process_args,
                    kwargs = process_kwargs,
                    daemon = True,
                )
                try:
                    with (
                        child_environment_for_spawn(cache_env),
                        native_path_secret_removed_for_child_start(),
                    ):
                        spawn_on_lifetime_thread(proc.start)
                    adopt_pid(proc.pid)
                    deadline = time.monotonic() + _PREVIEW_TIMEOUT_S
                    while proc.exitcode is None:
                        if is_process_shutting_down():
                            raise ValueError("Unsloth is shutting down.")
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise ValueError("The JSON dataset preview timed out.")
                        proc.join(timeout = min(remaining, 0.1))

                    if proc.exitcode != 0:
                        logger.error(
                            "JSON dataset preview worker exited with code %s: %s",
                            format_exit_code(proc.exitcode),
                            first_crash_line(stderr.text()),
                        )
                        raise ValueError(
                            "The JSON dataset reader stopped unexpectedly. "
                            "Check the file's structure and nested field types."
                        )
                    try:
                        reply = json.loads(
                            (Path(scratch) / "result.json").read_text(encoding = "utf-8")
                        )
                    except (OSError, ValueError) as exc:
                        raise ValueError(
                            "The JSON dataset reader returned no usable result."
                        ) from exc
                    if not reply["ok"]:
                        raise ValueError(reply["error"])
                    if reply["empty"]:
                        return None

                    import pyarrow as pa
                    from datasets import Dataset

                    # Preserve the child's schema (including Features metadata), without inferring it again.
                    data = (Path(scratch) / "preview.arrow").read_bytes()
                    return Dataset.from_buffer(pa.py_buffer(data)), None
                finally:
                    self._stop()
                    if proc.pid is not None and proc.exitcode is not None:
                        forget_pid(proc.pid)
                        proc.close()
                    self._proc = None


def load_json_preview_slice(path: Path, preview_size: int, **load_kwargs):
    return _JsonPreviewJob().read(path, preview_size, load_kwargs)
