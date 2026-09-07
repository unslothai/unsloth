# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Per-kind Windows qualification and owned launch dispatch, without probe caching.

Compatibility measurements cannot enable this backend. Each launch requires a
fresh complete qualification and matching prepared runtime/content identities.
Limited mode is owned by the separate existing backend.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

from . import launch, probe, terminal_launch, terminal_qualification
from .dependencies import checked_path
from .profiles import PYTHON_PROFILE, WindowsRuntimeError


def _failure(message, code = "WINDOWS_SANDBOX_QUALIFICATION_INCOMPLETE"):
    return WindowsRuntimeError(code, message)


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _os_identity():
    if sys.platform != "win32":
        raise _failure(
            "The Windows bootstrap requires a Windows host.", "WINDOWS_SANDBOX_UNSUPPORTED"
        )
    version = sys.getwindowsversion()
    return tuple(version), tuple(version.platform_version), sys.maxsize


def _executable_identity(selected):
    path = checked_path(selected)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 128 * 1024 * 1024:
        raise _failure("Invalid selected executable identity.")
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise _failure("Selected executable changed before qualification.")
        while block := stream.read(1024 * 1024):
            total += len(block)
            if total > 128 * 1024 * 1024:
                raise _failure("Selected executable exceeds the identity bound.")
            digest.update(block)
        after = os.fstat(stream.fileno())
    current = checked_path(path).stat()
    fields = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_nlink)
    if (
        total != before.st_size
        or fields(before) != fields(after)
        or fields(after) != fields(current)
        or before.st_ctime_ns != current.st_ctime_ns
        or opened.st_ctime_ns != after.st_ctime_ns
    ):
        raise _failure("Selected executable changed during qualification.")
    return str(path), digest.hexdigest(), before.st_dev, before.st_ino, total


def _complete_checks(module, observations, *, python):
    required = getattr(module, "REQUIRED_QUALIFICATION_CHECKS", None)
    complete = (
        getattr(observations, "qualification_complete", False)
        if python
        else getattr(observations, "qualified", False)
    )
    checks = getattr(observations, "checks", None)
    return (
        complete is True
        and type(required) is tuple
        and bool(required)
        and all(type(name) is str and name for name in required)
        and len(set(required)) == len(required)
        and type(checks) is tuple
        and all(type(name) is str for name in checks)
        and len(set(checks)) == len(checks)
        and set(required) <= set(checks)
    )


@dataclass(frozen = True)
class _Qualification:
    kind: str
    profile_id: str
    executable: tuple
    os_identity: tuple
    measurements: object
    generation: str


class WindowsBootstrapBackend:
    identity = "windows-lpac"
    supports_network_allowlist = False
    # The outer capability layer must not reuse its OS-name/path cache here.
    requires_fresh_qualification = True

    def __init__(
        self,
        store_root = None,
        *,
        timeout = 90,
        cancel = None,
    ):
        if store_root is None:
            local = os.environ.get("LOCALAPPDATA")
            store_root = Path(local) / "UnslothSandboxRuntime" if local else None
        self.store_root = store_root
        self.timeout = timeout
        self.cancel = cancel

    def _qualify(self, kind, selected):
        if kind not in ("python", "terminal") or not selected:
            raise _failure("Select an explicit Python interpreter or Terminal executable.")
        if self.store_root is None:
            raise _failure("The trusted Windows runtime store location is unavailable.")
        os_identity = _os_identity()
        executable = _executable_identity(selected)
        if kind == "python":
            measured = probe.run_python_probe(
                executable[0], self.store_root, timeout = self.timeout, cancel = self.cancel
            )
            if type(measured) is not probe.ProbeObservations:
                raise _failure("Invalid Python qualification result.")
            if not _complete_checks(probe, measured, python = True):
                raise _failure(
                    "Python compatibility measurements do not satisfy complete isolation qualification."
                )
            if (
                not all(
                    _digest(value)
                    for value in (
                        measured.runtime_digest,
                        measured.content_digest,
                        measured.artifact_digest,
                        measured.profile_digest,
                        getattr(measured, "catalog_binding_digest", ""),
                    )
                )
                or measured.profile_digest != PYTHON_PROFILE.digest
            ):
                raise _failure("Python qualification identities do not match the reviewed profile.")
            profile_id = PYTHON_PROFILE.profile_id
        else:
            measured = terminal_qualification.qualify_terminal_runtime(
                executable[0], store_root = self.store_root, timeout = self.timeout, cancel = self.cancel
            )
            if type(measured) is not terminal_qualification.TerminalQualification:
                raise _failure("Invalid Terminal qualification result.")
            if not _complete_checks(terminal_qualification, measured, python = False):
                raise _failure(
                    measured.reason or "Selected Terminal runtime remains unqualified.",
                    measured.failure_code or "WINDOWS_SANDBOX_QUALIFICATION_INCOMPLETE",
                )
            if (
                not _digest(measured.runtime_digest)
                or (measured.content_digest and not _digest(measured.content_digest))
                or os.path.normcase(measured.selected_executable) != os.path.normcase(executable[0])
                or measured.profile_id != terminal_qualification.TERMINAL_PROFILE_ID
            ):
                raise _failure(
                    "Terminal qualification identities do not match the selected runtime."
                )
            profile_id = measured.profile_id
        if _os_identity() != os_identity or _executable_identity(selected) != executable:
            raise _failure("The selected executable or OS changed during qualification.")
        measurements = asdict(measured)
        measurements.pop("elapsed_seconds", None)
        generation = hashlib.sha256(
            json.dumps(
                {
                    "kind": kind,
                    "profile": profile_id,
                    "executable": executable,
                    "os": os_identity,
                    "measurements": measurements,
                },
                sort_keys = True,
                separators = (",", ":"),
            ).encode()
        ).hexdigest()
        return _Qualification(kind, profile_id, executable, os_identity, measured, generation)

    def probe(self):
        return self.probe_for_kind("python", sys.executable)

    def probe_for_kind(
        self,
        execution_kind,
        selected_executable = None,
    ):
        from ..os_sandbox import SandboxCapability

        if selected_executable is None and execution_kind == "python":
            selected_executable = sys.executable
        try:
            result = self._qualify(execution_kind, selected_executable)
        except Exception as error:
            code = getattr(error, "code", "WINDOWS_SANDBOX_QUALIFICATION_FAILED")
            transient = code in {
                "WINDOWS_SANDBOX_CANCELLED",
                "WINDOWS_SANDBOX_STARTUP_TIMEOUT",
                "WINDOWS_SANDBOX_PREPARATION_TIMEOUT",
                "WINDOWS_SANDBOX_SCAN_LIMIT",
                "WINDOWS_SANDBOX_CLEANUP_FAILED",
                "WINDOWS_SANDBOX_STORE_BUSY",
            }
            # Failed observations still bind consent to the selected executable.
            # Diagnostic text and temporary paths must not rotate valid grants.
            try:
                executable = _executable_identity(selected_executable)
            except Exception:
                executable = (str(selected_executable), "unavailable")
            try:
                os_identity = _os_identity()
            except Exception:
                os_identity = ("unavailable",)
            generation = hashlib.sha256(
                json.dumps(
                    {
                        "kind": execution_kind,
                        "executable": executable,
                        "os": os_identity,
                        "python_profile": PYTHON_PROFILE.digest,
                        "terminal_profile": terminal_qualification.TERMINAL_PROFILE_ID,
                    },
                    sort_keys = True,
                    separators = (",", ":"),
                ).encode()
            ).hexdigest()
            return SandboxCapability(
                self.identity,
                False,
                f"{code}: {error}",
                available = False,
                transient = transient,
                environment = "windows",
                limitations = (f"{execution_kind}_runtime_unqualified",),
                probe_generation = generation,
            )
        limitations = (
            PYTHON_PROFILE.limitations
            if execution_kind == "python"
            else result.measurements.limitations
        )
        return SandboxCapability(
            self.identity,
            True,
            "The selected runtime passed complete live isolation qualification.",
            available = True,
            environment = "windows",
            protection_state = "preview",
            profile_id = result.profile_id,
            limitations = limitations,
            probe_generation = result.generation,
            qualification_generation = result.generation,
        )

    def prepare(self, spec):
        profile = (
            PYTHON_PROFILE.profile_id
            if spec.execution_kind == "python"
            else terminal_qualification.TERMINAL_PROFILE_ID
        )
        return self.prepare_for_profile(spec, profile)

    def prepare_for_profile(
        self,
        spec,
        profile_id,
        *,
        expected_generation = None,
    ):
        from ..os_sandbox import ToolLaunchPlan

        if (
            type(spec) is not ToolLaunchPlan
            or spec.execution_kind not in ("python", "terminal")
            or not spec.argv
        ):
            raise _failure("Windows isolation requires an explicit execution kind.")
        if spec.requested_mode != "os_isolation_required" or spec.network_policy != "deny":
            raise _failure(
                "The qualified Windows backend requires isolated execution with network denial."
            )
        expected_profile = (
            PYTHON_PROFILE.profile_id
            if spec.execution_kind == "python"
            else terminal_qualification.TERMINAL_PROFILE_ID
        )
        if profile_id != expected_profile:
            raise _failure("The selected execution kind does not match the qualified profile.")
        qualified = self._qualify(spec.execution_kind, spec.argv[0])
        if expected_generation is not None and qualified.generation != expected_generation:
            raise _failure(
                "Runtime qualification changed before preparation.",
                "WINDOWS_SANDBOX_RUNTIME_CHANGED",
            )
        if spec.execution_kind == "python":
            prepared = launch.prepare_python_launch(
                spec, self.store_root, timeout = self.timeout, cancel = self.cancel
            )
        else:
            prepared = terminal_launch.prepare_terminal_launch(
                spec, store_root = self.store_root, timeout = self.timeout, cancel = self.cancel
            )
        try:
            owner = getattr(prepared.spawn_callback, "__self__", None)
            if owner is None:
                raise _failure("The prepared Windows launch lost its native owner.")
            measured = qualified.measurements
            if spec.execution_kind == "python":
                published = owner.published
                actual = (
                    published.core.digest,
                    published.content_digest,
                    published.artifacts.digest,
                    PYTHON_PROFILE.digest,
                )
                expected = (
                    measured.runtime_digest,
                    measured.content_digest,
                    measured.artifact_digest,
                    measured.profile_digest,
                )
            else:
                from .terminal_probe import _runtime_identity

                selected, _, runtime_digest, content_digest = _runtime_identity(owner)
                actual = (os.path.normcase(selected), runtime_digest, content_digest)
                expected = (
                    os.path.normcase(measured.selected_executable),
                    measured.runtime_digest,
                    measured.content_digest,
                )
            if (
                actual != expected
                or _os_identity() != qualified.os_identity
                or _executable_identity(spec.argv[0]) != qualified.executable
            ):
                raise _failure("Prepared runtime identity differs from its complete qualification.")
            if spec.execution_kind == "python":
                owner.expected_catalog_binding = measured.catalog_binding_digest
            return prepared
        except BaseException as original:
            prepared.cleanup()
            if prepared.cleanup_diagnostics or (
                owner is not None and not getattr(owner, "closed", False)
            ):
                error = _failure(
                    "Unqualified launch cleanup retained native ownership.",
                    "WINDOWS_SANDBOX_CLEANUP_FAILED",
                )
                error.retained_launch = owner
                raise error from original
            raise
