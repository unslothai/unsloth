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


def _install_private_asyncio_wakeup():
    """Keep CPython 3.11-3.13's IOCP loop, replacing only its TCP self-pipe."""
    import _winapi
    import asyncio
    import asyncio.windows_events
    from asyncio.windows_utils import PipeHandle
    import msvcrt
    import os
    import secrets
    import threading

    if getattr(asyncio.ProactorEventLoop, "_unsloth_private_wakeup", False):
        return

    class WakeupWriter:
        def __init__(self, handle):
            self._lock = threading.RLock()
            self._fd = msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_BINARY)

        def fileno(self):
            return self._fd

        def send(self, data):
            with self._lock:
                if self._fd < 0:
                    raise OSError("asyncio wakeup pipe is closed")
                return os.write(self._fd, data)

        def close(self):
            with self._lock:
                if self._fd >= 0:
                    fd, self._fd = self._fd, -1
                    os.close(fd)

    def make_self_pipe(loop):
        address = "\\\\.\\pipe\\LOCAL\\unsloth-asyncio-" + secrets.token_hex(16)
        reader = writer = None
        try:
            reader = _winapi.CreateNamedPipe(
                address,
                _winapi.PIPE_ACCESS_INBOUND
                | _winapi.FILE_FLAG_OVERLAPPED
                | _winapi.FILE_FLAG_FIRST_PIPE_INSTANCE,
                _winapi.PIPE_WAIT,
                1,
                0,
                8192,
                0,
                _winapi.NULL,
            )
            writer = _winapi.CreateFile(
                address,
                _winapi.GENERIC_WRITE,
                0,
                _winapi.NULL,
                _winapi.OPEN_EXISTING,
                0,
                _winapi.NULL,
            )
            _winapi.ConnectNamedPipe(reader, overlapped = True).GetOverlappedResult(True)
            # Reads use ordinary IOCP overlapped I/O. The synchronous byte writer
            # must never wait for buffer space (including from a signal handler).
            # A full buffer already contains the wakeup; no queued write/Future
            # is registered from the calling worker thread.
            _winapi.SetNamedPipeHandleState(writer, 1, None, None)  # PIPE_NOWAIT
            sender = WakeupWriter(writer)
            writer = None
            loop._ssock, loop._csock = PipeHandle(reader), sender
            reader = None
            loop._internal_fds += 1
        finally:
            if writer is not None:
                _winapi.CloseHandle(writer)
            if reader is not None:
                _winapi.CloseHandle(reader)

    class PrivateProactorEventLoop(asyncio.windows_events.ProactorEventLoop):
        _unsloth_private_wakeup = True

        def __init__(self, proactor = None):
            # CPython's base constructor registers a socket with signal. Its
            # getsockopt(CRT-fd) check fails with native INVALID_HANDLE in LPAC.
            # This embedded, no-console profile uses broker/Job cancellation,
            # not console signal wakeups. Do not disable native handle checks.
            asyncio.BaseEventLoop.__init__(self)
            self._proactor = proactor or asyncio.windows_events.IocpProactor()
            self._selector = self._proactor
            self._self_reading_future = None
            self._accept_futures = {}
            self._ssock = self._csock = None
            self._proactor.set_loop(self)
            try:
                make_self_pipe(self)
            except BaseException:
                self._proactor.close()
                self._proactor = self._selector = None
                asyncio.BaseEventLoop.close(self)
                raise

        def close(self):
            if self.is_running():
                raise RuntimeError("Cannot close a running event loop")
            if self.is_closed():
                return
            self._stop_accept_futures()
            self._close_self_pipe()
            self._proactor.close()
            self._proactor = self._selector = None
            asyncio.BaseEventLoop.close(self)

    # Default and explicitly selected Proactor loops use the same private IPC.
    # SelectorEventLoop is not relabelled or given a network exemption.
    asyncio.ProactorEventLoop = PrivateProactorEventLoop
    asyncio.windows_events.ProactorEventLoop = PrivateProactorEventLoop
    asyncio.WindowsProactorEventLoopPolicy._loop_factory = PrivateProactorEventLoop


def install_single_process_policy() -> None:
    """Called explicitly by the native host after its verified token-drop gate."""
    import asyncio
    import concurrent.futures.process
    import multiprocessing
    import multiprocessing.context
    import multiprocessing.connection
    import multiprocessing.managers
    import multiprocessing.pool
    import multiprocessing.process
    import subprocess
    import os

    if os.name == "nt":
        import secrets

        _install_private_asyncio_wakeup()
        previous_address = multiprocessing.connection.arbitrary_address

        if not getattr(previous_address, "_unsloth_private_pipe", False):

            def private_address(family):
                if family == "AF_PIPE":
                    # AppContainer named pipes must use its LOCAL namespace.
                    # No host pipe is made visible and no broker is involved.
                    return "\\\\.\\pipe\\LOCAL\\unsloth-" + secrets.token_hex(16)
                return previous_address(family)

            private_address._unsloth_private_pipe = True
            multiprocessing.connection.arbitrary_address = private_address

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
