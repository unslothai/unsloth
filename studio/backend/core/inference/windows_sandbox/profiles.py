# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Checked-in bootstrap policy and ABI build targets, not qualification evidence.

Only reviewed source changes may extend the privileged startup action set.
Discovering a package or passing a compatibility test does not authorize its
initializer to run with the temporary startup token.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


class WindowsRuntimeError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen = True)
class AbiAdapter:
    major: int
    minor: int
    architecture: str
    configuration_api: str = "PyConfig"

    @property
    def identity(self) -> str:
        return f"cpython-{self.major}{self.minor}-{self.architecture}-release"


# One source for helper build targets and runtime ABI admission. Membership says
# an adapter must be built/tested, not that a runtime is available or qualified.
ABI_ADAPTERS = tuple(AbiAdapter(3, minor, "x64") for minor in (11, 12, 13))


@dataclass(frozen = True)
class BootstrapProfile:
    profile_id: str
    schema_version: int
    protocol_version: int
    abi_adapters: tuple[AbiAdapter, ...]
    runtime_families: tuple[str, ...]
    startup_capabilities: tuple[str, ...]
    startup_actions: tuple[str, ...]
    payload_capabilities: tuple[str, ...]
    active_process_limit: int
    limitations: tuple[str, ...]

    @property
    def digest(self) -> str:
        from dataclasses import asdict
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys = True, separators = (",", ":")).encode("utf-8")
        ).hexdigest()


PYTHON_PROFILE = BootstrapProfile(
    profile_id = "windows-lpac-python-single-process-v2",
    schema_version = 1,
    protocol_version = 1,
    abi_adapters = ABI_ADAPTERS,
    runtime_families = ("cpython", "venv"),
    startup_capabilities = ("registryRead",),
    startup_actions = (
        "initialize_isolated_cpython",
        "initialize_winsock_providers",
        "initialize_overlapped",
    ),
    payload_capabilities = (),
    active_process_limit = 1,
    limitations = ("python_single_process", "gpu_execution_unqualified"),
)


def select_abi_adapter(
    *,
    implementation: str,
    version: tuple[int, int, int],
    architecture: str,
    debug: bool = False,
    free_threaded: bool = False,
) -> AbiAdapter:
    """Match explicit discovered metadata; never execute an interpreter to guess."""
    if (
        implementation != "cpython"
        or debug
        or free_threaded
        or len(version) != 3
        or any(type(part) is not int or part < 0 for part in version)
    ):
        raise WindowsRuntimeError(
            "WINDOWS_SANDBOX_ABI_UNSUPPORTED",
            "The Python bootstrap requires an explicitly supported release CPython ABI.",
        )
    for adapter in ABI_ADAPTERS:
        if (adapter.major, adapter.minor, adapter.architecture) == (*version[:2], architecture):
            return adapter
    raise WindowsRuntimeError(
        "WINDOWS_SANDBOX_ABI_UNSUPPORTED",
        f"No bootstrap adapter exists for CPython {version[0]}.{version[1]} ({architecture}).",
    )
