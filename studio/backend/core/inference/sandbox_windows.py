# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Windows-only adapter for Studio's pinned SRT runtime."""

import hashlib
import sys
from dataclasses import replace

from . import srt_adapter, srt_probe
from .os_sandbox import (
    SandboxCapability,
    SandboxBuildError,
    PreparedSandboxLaunch,
    _record,
    _OS_ISOLATION_SAFEGUARDS,
    scan_workdir_for_host_channels,
)


def capability_snapshot(
    *,
    force = False,
    execution_kind = None,
    selected_executable = None,
):
    if sys.platform != "win32":
        raise SandboxBuildError("Studio SRT is Windows-only")
    available, reason = srt_probe.probe(
        force = force,
        execution_kind = execution_kind,
        selected_executable = selected_executable,
    )
    identity = hashlib.sha256(
        repr(
            (
                srt_adapter.installation_identity(),
                srt_probe.runtime_inputs(),
                execution_kind,
                selected_executable,
            )
        ).encode()
    ).hexdigest()
    diagnostic = reason.fields() if hasattr(reason, "fields") else {}
    return SandboxCapability(
        backend = "srt",
        available = available,
        reason = str(reason),
        environment = "win32",
        protection_state = "preview" if available else "unavailable",
        profile_id = "srt-0.0.75-native-v1",
        limitations = ("srt_windows_system_dns_unfenced", "srt_windows_shared_account_grants"),
        probe_generation = hashlib.sha256((identity + str(available)).encode()).hexdigest(),
        environment_fingerprint = identity,
        remediation = diagnostic.get(
            "remediation",
            "Windows setup is optional in Auto. Use Set up Windows sandbox to install or repair isolation; this never retries a Python or Terminal call.",
        ),
        reason_code = diagnostic.get("reason_code"),
        diagnostic = diagnostic.get("diagnostic"),
    )


def prepare(plan, capability):
    try:
        scan_workdir_for_host_channels(plan.workdir)
        request = srt_adapter.request_for(plan.argv, plan.workdir, plan.env, plan.timeout_seconds)
    except Exception as exc:
        raise SandboxBuildError(f"Windows sandbox preparation failed: {exc}") from exc
    record = _record(
        plan,
        capability,
        effective_mode = "os_isolated",
        os_isolation = True,
        backend = "srt",
        profile_id = capability.profile_id,
        safeguards = tuple(s for s in _OS_ISOLATION_SAFEGUARDS if s != "resource_limits"),
        limitations = capability.limitations,
    )
    prepared = PreparedSandboxLaunch(
        argv = plan.argv,
        workdir = plan.workdir,
        env = plan.env,
        preexec_fn = None,
        backend = "srt",
        timeout_seconds = plan.timeout_seconds,
        execution_record = replace(record, network_policy = "deny_with_system_dns"),
    )

    def launch(_prepared, kwargs):
        try:
            proc = srt_adapter.spawn(request, cancel_event = plan.cancel_event, **kwargs)
        except Exception as exc:
            # Recheck native state on the next request; never replay this launch.
            srt_probe.invalidate_cache()
            raise SandboxBuildError(f"Windows sandbox launch failed: {exc}") from exc
        prepared.cleanup_callbacks.append(lambda: srt_adapter.release_control(proc))
        return proc

    prepared.spawn_callback = launch
    return prepared
