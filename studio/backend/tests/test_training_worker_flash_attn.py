# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import builtins
import subprocess
import sys
from unittest import mock

from core.training import worker

_REAL_IMPORT = builtins.__import__


def _raise_only_for_flash_attn(name, *args, **kwargs):
    if name == "flash_attn":
        raise ImportError
    return _REAL_IMPORT(name, *args, **kwargs)


def test_runtime_flash_attn_prefers_prebuilt_wheel():
    statuses: list[str] = []

    with (
        mock.patch.object(
            worker,
            "flash_attn_wheel_url",
            return_value = "https://example.com/fa.whl",
        ),
        mock.patch.object(worker, "url_exists", return_value = True),
        mock.patch.object(
            worker,
            "_send_status",
            side_effect = lambda queue, message: statuses.append(message),
        ),
        mock.patch.object(
            worker,
            "install_wheel",
            return_value = [("pip", subprocess.CompletedProcess(["pip"], 0, ""))],
        ),
        mock.patch.object(worker, "probe_torch_wheel_env", return_value = None),
        mock.patch.object(
            builtins,
            "__import__",
            side_effect = _raise_only_for_flash_attn,
        ),
    ):
        worker._ensure_flash_attn_for_long_context(
            event_queue = object(), max_seq_length = 32768
        )

    assert statuses == ["Installing prebuilt flash-attn wheel..."]


def test_runtime_flash_attn_falls_back_to_pypi():
    statuses: list[str] = []
    run_calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        run_calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "")

    with (
        mock.patch.object(
            worker,
            "flash_attn_wheel_url",
            return_value = "https://example.com/fa.whl",
        ),
        mock.patch.object(worker, "url_exists", return_value = False),
        mock.patch.object(
            worker,
            "_send_status",
            side_effect = lambda queue, message: statuses.append(message),
        ),
        mock.patch.object(worker.shutil, "which", return_value = None),
        mock.patch.object(worker, "install_wheel") as install_wheel,
        mock.patch.object(worker._sp, "run", side_effect = fake_run),
        mock.patch.object(worker, "probe_torch_wheel_env", return_value = None),
        mock.patch.object(
            builtins,
            "__import__",
            side_effect = _raise_only_for_flash_attn,
        ),
    ):
        worker._ensure_flash_attn_for_long_context(
            event_queue = object(), max_seq_length = 32768
        )

    install_wheel.assert_not_called()
    assert statuses == ["Installing flash-attn from PyPI for long-context training..."]
    assert run_calls == [[sys.executable, "-m", "pip", "install", "flash-attn"]]


def test_runtime_flash_attn_skip_env_avoids_all_install_work():
    with (
        mock.patch.dict(
            "os.environ",
            {worker._FLASH_ATTN_SKIP_ENV: "1"},
            clear = False,
        ),
        mock.patch.object(worker, "probe_torch_wheel_env") as probe_torch_wheel_env,
        mock.patch.object(worker, "install_wheel") as install_wheel,
        mock.patch.object(worker._sp, "run") as run,
    ):
        worker._ensure_flash_attn_for_long_context(
            event_queue = object(), max_seq_length = 32768
        )

    probe_torch_wheel_env.assert_not_called()
    install_wheel.assert_not_called()
    run.assert_not_called()
