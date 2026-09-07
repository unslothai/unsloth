# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Broker-side experimental launch ownership, independent of logging failures."""

import shutil
from ..os_sandbox import PreparedSandboxLaunch as _PreparedSandboxLaunch


class PreparedPythonLaunch(_PreparedSandboxLaunch):
    def cleanup(self) -> None:
        # Finish resource cleanup before invoking the application logging stack.
        # Fixed native workers may not have it, and handlers can themselves fail.
        failures: list[tuple[str, Exception]] = []
        while self.cleanup_callbacks:
            callback = self.cleanup_callbacks.pop()
            try:
                callback()
            except Exception as exc:  # noqa: BLE001 - cleanup continues in LIFO order
                diagnostic = f"{type(exc).__name__}: {exc}"
                self.cleanup_diagnostics.append(diagnostic)
                failures.append((diagnostic, exc))
        while self.owned_files:
            try:
                self.owned_files.pop().close()
            except Exception as exc:  # noqa: BLE001 - cleanup must continue
                diagnostic = f"{type(exc).__name__}: {exc}"
                self.cleanup_diagnostics.append(diagnostic)
                failures.append((diagnostic, exc))
        while self.cleanup_paths:
            path = self.cleanup_paths.pop()
            try:
                shutil.rmtree(path)
            except OSError as exc:
                diagnostic = f"could not remove private sandbox path: {path}"
                self.cleanup_diagnostics.append(diagnostic)
                failures.append((diagnostic, exc))
        for diagnostic, exc in failures:
            try:
                from loggers import get_logger
                get_logger(__name__).warning(
                    "Sandbox cleanup failed: %s",
                    diagnostic,
                    exc_info = (type(exc), exc, exc.__traceback__),
                )
            except Exception as logging_error:  # noqa: BLE001 - diagnostics must not abort cleanup
                self.cleanup_diagnostics.append(
                    f"Cleanup logging failed: {type(logging_error).__name__}: {logging_error}"
                )
