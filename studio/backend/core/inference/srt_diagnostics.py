# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Public diagnostics contain reviewed text, never workload output or configuration."""

import hashlib
import os
from pathlib import Path
import sys


REASONS = {
    "proxy_port_unavailable": (
        "Windows cannot bind any port in the configured tool isolation proxy range.",
        "Choose Repair existing setup to select usable proxy ports and replace the matching sandbox firewall rules. Other SRT sessions may need restarting. Studio will verify isolation afterward.",
    ),
    "setup_conflict": (
        "The existing Windows tool isolation setup has conflicting network settings.",
        "Open Windows tool isolation setup and choose Repair existing setup. This replaces its sandbox network filters; other SRT sessions may need restarting. If repaired externally, choose Check again.",
    ),
    "runtime_missing": (
        "The pinned SRT runtime is missing.",
        "Run studio/install_srt_runtime.py with Studio's selected Python, then Check again.",
    ),
    "runtime_invalid": (
        "The installed SRT runtime failed integrity verification.",
        "Repair the pinned runtime with Studio's selected Python, then Check again.",
    ),
    "dependency_missing": (
        "A required isolation dependency is missing or unavailable.",
        "Install or repair the named dependency in this environment, then Check again. Keep Studio's selected Python environment.",
    ),
    "policy_invalid": (
        "Studio could not validate the isolation policy.",
        "Repair the reported policy field or runtime layout, then Check again. Changing isolation mode does not repair this policy.",
    ),
    "policy_oversized": (
        "The isolation policy exceeds a supported limit.",
        "Update or repair Studio's policy generation, then Check again. Do not broaden or discard paths to fit.",
    ),
    "operation_unsupported": (
        "This environment cannot perform a required isolation operation.",
        "Use an environment that supports the required isolation operations. Studio will not change host security settings.",
    ),
    "probe_timeout": (
        "The isolation check timed out.",
        "Check again after this environment recovers. The original command will not run automatically.",
    ),
    "enforcement_failed": (
        "The isolation check did not establish the required protections.",
        "Review the diagnostic stage and repair the isolation environment, then Check again.",
    ),
    "probe_failed": (
        "Studio could not verify isolation in this environment.",
        "Check again or report the diagnostic code and stage. Required remains blocked.",
    ),
}
STAGES = {"installation", "dependency", "policy", "launch", "enforcement", "probe"}
DEPENDENCIES = {
    "node",
    "bubblewrap",
    "ripgrep",
    "ripgrep",
    "socat",
    "selected_interpreter",
    "selected_shell",
    "seccomp_helper",
}


class ProbeReason(str):
    """Keep the existing (available, reason) probe interface for its callers."""

    def __new__(
        cls,
        code,
        stage = "probe",
        dependency = None,
        details = None,
    ):
        code = code if isinstance(code, str) and code in REASONS else "probe_failed"
        value = super().__new__(cls, REASONS[code][0])
        value.code = code
        value.stage = stage if isinstance(stage, str) and stage in STAGES else "probe"
        value.dependency = (
            dependency if isinstance(dependency, str) and dependency in DEPENDENCIES else None
        )
        value.details = {}
        if isinstance(details, dict) and details.get("field") in (
            "readRoots",
            "writeRoots",
            "denyReadRoots",
            "denyWriteRoots",
        ):
            if all(
                type(details.get(key)) is int and 0 <= details[key] <= 262144
                for key in ("count", "limit")
            ):
                value.details = {key: details[key] for key in ("field", "count", "limit")}
        return value

    def fields(self):
        return {
            "reason_code": self.code,
            "diagnostic": {
                "code": self.code,
                "stage": self.stage,
                "dependency": self.dependency,
                **self.details,
            },
            "remediation": REASONS[self.code][1],
            "retryable": True,
        }


def environment_context():
    # Both facts are read on the server. Browser location and request labels are irrelevant.
    if (
        sys.platform == "linux"
        and os.path.isdir("/var/colab")
        and os.environ.get("COLAB_RELEASE_TAG")
    ):
        return "colab"
    return sys.platform


def boundary_identity():
    """Invalidate cached probes and consent on runtime/container recycling."""
    digest = hashlib.sha256(environment_context().encode())
    for name in (
        "/proc/self/ns/user",
        "/proc/self/ns/mnt",
        "/proc/self/ns/pid",
        "/proc/self/ns/net",
    ):
        try:
            digest.update(os.readlink(name).encode())
        except OSError:
            digest.update(b"unknown")
    try:
        digest.update(Path("/proc/sys/kernel/random/boot_id").read_bytes()[:128])
    except OSError:
        pass
    return digest.hexdigest()


def limited_disclosure(environment):
    location = "inside this Colab runtime" if environment == "colab" else "in this environment"
    return (
        f"Python and Terminal run with Studio's permissions {location}. "
        "They can access files and credentials available to Studio and use its network connection. "
        "Studio does not add an OS sandbox in Limited mode."
    )
