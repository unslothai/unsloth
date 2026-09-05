# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Post-drop Python diagnostics for the one-process Windows bootstrap profile.

The native Job Object enforces the policy. These guards are intentionally not a
security boundary: native calls and saved references must still be denied by the
Job. Never install this module in the Studio broker, Limited, Full or Terminal.
"""

from __future__ import annotations


class WindowsSandboxChildProcessDisabled(RuntimeError):
    code = "WINDOWS_SANDBOX_CHILD_PROCESS_DISABLED"

    def __init__(self):
        super().__init__(
            f"{self.code}: This Windows Python sandbox supports one process per tool. "
            "Worker processes and subprocess launches are disabled. Use the library's "
            "single-process or threading mode."
        )


def _deny_process(*_args, **_kwargs):
    raise WindowsSandboxChildProcessDisabled()


async def _deny_async_process(*_args, **_kwargs):
    raise WindowsSandboxChildProcessDisabled()


def install_single_process_policy() -> None:
    """Called explicitly by the native host after its verified token-drop gate."""
    import asyncio
    import concurrent.futures.process
    import multiprocessing
    import multiprocessing.context
    import multiprocessing.managers
    import multiprocessing.pool
    import multiprocessing.process
    import subprocess

    # Guard constructors before they allocate queues/global named pipes. Keep
    # imports and thread-based pools usable; do not substitute algorithms.
    multiprocessing.process.BaseProcess.start = _deny_process
    multiprocessing.context.BaseContext.Pool = _deny_process
    multiprocessing.context.BaseContext.Manager = _deny_process
    multiprocessing.Pool = _deny_process
    multiprocessing.Manager = _deny_process
    multiprocessing.managers.BaseManager.start = _deny_process
    concurrent.futures.process.ProcessPoolExecutor.__init__ = _deny_process
    subprocess.Popen.__init__ = _deny_process
    asyncio.create_subprocess_exec = _deny_async_process
    asyncio.create_subprocess_shell = _deny_async_process
    asyncio.BaseEventLoop.subprocess_exec = _deny_async_process
    asyncio.BaseEventLoop.subprocess_shell = _deny_async_process

    # ThreadPool inherits Pool.__init__. Preserve that path while rejecting the
    # process-backed constructor, including direct multiprocessing.pool imports.
    original_pool_init = multiprocessing.pool.Pool.__init__
    if getattr(original_pool_init, "_unsloth_single_process_guard", False):
        return

    def pool_init(self, *args, **kwargs):
        if not isinstance(self, multiprocessing.pool.ThreadPool):
            _deny_process()
        return original_pool_init(self, *args, **kwargs)

    pool_init._unsloth_single_process_guard = True
    multiprocessing.pool.Pool.__init__ = pool_init
