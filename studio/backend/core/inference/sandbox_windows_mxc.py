# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Windows-only MXC backend for the shared Studio tool launch contract."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import logging
import os
import subprocess
import sys

from . import mxc_adapter, mxc_drive_alias, mxc_policy, mxc_probe, mxc_read_grants, mxc_runtime
from .os_sandbox import (
    PreparedSandboxLaunch,
    SandboxBuildError,
    SandboxCapability,
    _OS_ISOLATION_SAFEGUARDS,
    _SOFTWARE_SAFEGUARDS,
    _record,
    with_session_packages,
)


logger = logging.getLogger(__name__)

CMD_PROFILE_LIMITATION = "terminal_cmd_profile_git_hooks_pager_editor_disabled"


def _workdir_alias(plan):
    """(lease, limitations) for a cmd Terminal launch: stock git needs the workdir as a drive root."""
    if plan.execution_kind != "terminal" or not mxc_policy.is_cmd_argv(plan.argv):
        return None, ()
    try:
        workdir = mxc_policy._safe_canonical_path(plan.workdir, directory = True)
    except Exception:
        return None, ()  # build_launch_request refuses the same workdir with the real reason
    lease = mxc_drive_alias.acquire(workdir)
    if lease is None:
        return None, (mxc_drive_alias.LIMITATION_UNAVAILABLE,)
    return lease, (CMD_PROFILE_LIMITATION,)


def _capability_fingerprint(identity: str, execution_kind: str, selected_executable: str) -> str:
    return hashlib.sha256(
        repr(
            (
                identity,
                execution_kind,
                os.path.abspath(selected_executable),
                mxc_runtime.PROFILE_ID,
                mxc_policy.dacl_fallback_enabled(),
            )
        ).encode()
    ).hexdigest()


def capability_snapshot(
    *,
    force = False,
    execution_kind = None,
    selected_executable = None,
    cancel_event = None,
):
    if sys.platform != "win32":
        raise SandboxBuildError("MXC is Windows-only")
    if execution_kind is None:
        execution_kind = "python"
        selected_executable = selected_executable or sys.executable
    if execution_kind not in {"python", "terminal"} or not selected_executable:
        return SandboxCapability(
            backend = "mxc-processcontainer",
            available = False,
            reason = "the MXC profile requires a selected Python or Terminal executable",
            environment = "win32",
            remediation = "Select a supported native Windows runtime.",
            limitations = ("unsupported_execution_kind",),
        )
    if not (mxc_policy.dacl_fallback_enabled() and mxc_read_grants.enabled()):
        # Here too: a host with only the DACL tier stops at this capability and never builds a launch.
        mxc_read_grants.revoke_recorded()
    available, reason = mxc_probe.probe(
        selected_executable,
        execution_kind = execution_kind,
        force = force,
        cancel_event = cancel_event,
    )
    try:
        identity = mxc_runtime.installation_identity()
    except Exception:
        identity = "missing"
    fingerprint = _capability_fingerprint(identity, execution_kind, selected_executable)
    dacl = mxc_policy.dacl_fallback_enabled()
    limitations = (
        "mxc_preview_not_a_security_boundary",
        "wxc_exec_does_not_report_execution_tier",
        "workload_start_not_structured_by_wxc_exec",
        "network_posture_requested_not_attested",
        "nested_path_identity_may_change_during_mxc_grant_resolution",
    )
    if dacl:
        limitations += ("mxc_tier3_dacl_host_permission_changes",)
        if mxc_read_grants.enabled():
            limitations += ("mxc_tier3_persistent_runtime_read_grants",)
    shell_incompatible = reason == mxc_probe.MSYS_NAMESPACE_REASON
    if shell_incompatible:
        remediation = (
            "Set tool isolation to auto to run Terminal commands with software safeguards, or use "
            "the Python tool, which is sandboxed separately."
        )
    elif dacl:
        remediation = (
            "Install the pinned Microsoft WXC runtime and prepare this host once as an "
            "administrator (Settings > Sandbox > Prepare this PC); the null device step repeats "
            "after every reboot."
        )
    else:
        remediation = (
            "Install the pinned Microsoft WXC runtime and enable BaseContainer/PSEC. On Windows "
            "builds without it, turn on Settings > Sandbox > Use the fallback sandbox (or "
            f"set {mxc_policy.DACL_FALLBACK_ENV}=1) to use the AppContainer "
            "tier: it adds temporary permission entries to the granted host folders, removed "
            "on exit, and needs a one-time administrator host preparation plus one per reboot. "
            "Studio's own Python runtime folders get a permanent read-only entry instead, so "
            "launches stay fast; turn that off in the same place (or set "
            f"{mxc_read_grants.PERSISTENT_GRANTS_ENV}=0) to keep every entry temporary."
        )
    # Only in DACL mode: a bare --probe allows the fallback, so it warns on hosts Studio never uses it on.
    host_prep = (
        mxc_probe.host_prep_remediation()
        if dacl and not available and not shell_incompatible
        else None
    )
    if host_prep:
        remediation = f"{remediation} {host_prep}"
    return SandboxCapability(
        backend = "mxc-processcontainer",
        available = available,
        reason = reason,
        environment = "win32",
        protection_state = "preview" if available else "unavailable",
        profile_id = mxc_runtime.PROFILE_ID,
        limitations = limitations,
        probe_generation = hashlib.sha256((fingerprint + str(available)).encode()).hexdigest(),
        environment_fingerprint = fingerprint,
        remediation = remediation,
    )


def prepare(plan, capability):
    lease, alias_limitations = _workdir_alias(plan)
    grant_lease = None
    try:
        # Before the request is built, so a revocation racing this launch either finishes first or waits.
        grant_lease = mxc_read_grants.hold_if_needed()
        request = (
            mxc_policy.build_launch_request(plan, cwd_alias = lease.root)
            if lease
            else mxc_policy.build_launch_request(plan)
        )
    except Exception as exc:
        if lease is not None:
            lease.release()
        if grant_lease is not None:
            grant_lease.release()
        raise SandboxBuildError(f"Windows MXC policy construction failed: {exc}") from exc
    launch_limitations = tuple(request.get("launchLimitations", ())) + alias_limitations
    record = _record(
        plan,
        capability,
        effective_mode = "os_isolated",
        os_isolation = True,
        backend = "mxc-processcontainer",
        profile_id = capability.profile_id,
        safeguards = _OS_ISOLATION_SAFEGUARDS + ("ui_isolation",),
        limitations = capability.limitations + launch_limitations,
    )
    record = replace(
        record,
        network_policy = "mxc_compatibility_requested_unverified",
        backend_tier = "unknown",
        runtime_revision = mxc_runtime.MXC_REVISION,
        runtime_artifact_digest = f"sha256:{mxc_runtime.WXC_EXEC_SHA256}",
        schema_version = mxc_runtime.MXC_SCHEMA_VERSION,
        policy_hash = request["policyHash"],
        execution_status = "planned",
        cleanup_status = "pending",
    )
    prepared = PreparedSandboxLaunch(
        argv = plan.argv,
        workdir = plan.workdir,
        env = plan.env,
        preexec_fn = None,
        backend = "mxc-processcontainer",
        timeout_seconds = plan.timeout_seconds,
        execution_record = record,
        launch_limitations = launch_limitations,
    )
    if grant_lease is not None:
        # Appended first so it runs last (LIFO): the grants outlive the workload that may use them.
        prepared.cleanup_callbacks.append(grant_lease.release)
    if lease is not None:
        # Runs on every exit path, spawned or not; release_runtime drops it first once the workload is gone.
        prepared.cleanup_callbacks.append(lease.release)

    def launch(_prepared, kwargs):
        try:
            current_identity = mxc_runtime.installation_identity()
            current_fingerprint = _capability_fingerprint(
                current_identity, plan.execution_kind, plan.argv[0]
            )
            if current_fingerprint != capability.environment_fingerprint:
                raise RuntimeError(
                    "the selected MXC runtime changed after capability qualification"
                )
            proc = mxc_adapter.spawn(request, cancel_event = plan.cancel_event, popen_kwargs = kwargs)
        except Exception as exc:
            may_have_started = bool(getattr(exc, "may_have_started", False))
            if lease is not None and not may_have_started:
                lease.release()
            cancelled = isinstance(exc, mxc_adapter.MxcLaunchCancelled)
            refused = getattr(exc, "stage", None) == "policy"
            if plan.requested_mode == "auto" and not (may_have_started or cancelled or refused):
                # The same environment as any other unisolated launch, session packages included.
                kwargs = {
                    **kwargs,
                    "env": with_session_packages(kwargs.get("env") or plan.env, plan.workdir),
                }
                try:
                    proc = subprocess.Popen(mxc_policy.host_spawn_args(plan.argv), **kwargs)
                except OSError as fallback_exc:
                    raise SandboxBuildError(
                        f"Windows MXC and software-safeguard launches both failed: {fallback_exc}"
                    ) from fallback_exc
                prepared.backend = "software-safeguards"
                prepared.execution_record = _record(
                    plan,
                    capability,
                    effective_mode = "software_safeguards",
                    os_isolation = False,
                    backend = "software-safeguards",
                    profile_id = "software-safeguards-v1",
                    safeguards = _SOFTWARE_SAFEGUARDS,
                    limitations = (
                        "no_os_isolation",
                        "host_files_readable",
                        "unrestricted_network",
                        "mxc_launch_failed_before_dispatch",
                    ),
                )
                prepared.execution_record = replace(
                    prepared.execution_record,
                    execution_status = "started",
                    completion_status = "pending",
                    cleanup_status = "pending",
                )
                mxc_probe.invalidate_cache()
                return proc
            prepared.execution_record = replace(
                prepared.execution_record,
                execution_status = "unknown_start" if may_have_started else "not_started",
                completion_status = (
                    "uncertain"
                    if may_have_started
                    else ("cancelled" if cancelled else "not_started")
                ),
                cleanup_status = "uncertain" if may_have_started else "complete",
            )
            if not cancelled:
                mxc_probe.invalidate_cache()
            raise SandboxBuildError(
                f"Windows MXC launch failed without host replay: {exc}"
            ) from exc
        prepared.execution_record = replace(
            prepared.execution_record,
            execution_status = "dispatched",
            backend_tier = str(getattr(proc, "_mxc_backend_tier", "unknown")),
        )
        if lease is not None:
            proc._mxc_drive_alias = lease
        prepared.cleanup_callbacks.append(lambda: mxc_adapter.release_runtime(proc))
        return proc

    prepared.spawn_callback = launch
    return prepared


def verify_success(prepared, proc) -> dict:
    try:
        result = mxc_adapter.completion_result(proc)
    except Exception as exc:
        if prepared.execution_record is not None:
            prepared.execution_record = replace(
                prepared.execution_record,
                completion_status = "uncertain",
                cleanup_status = "uncertain",
            )
        mxc_probe.invalidate_cache()
        raise SandboxBuildError(f"MXC completion state is uncertain: {exc}") from exc
    prepared.execution_record = replace(
        prepared.execution_record,
        execution_status = "completed",
        completion_status = (
            "timed_out"
            if result.get("timedOut")
            else ("cancelled" if result.get("cancelled") else "finished")
        ),
        cleanup_status = str(result.get("cleanup") or "unknown"),
    )
    if result.get("cleanup") != "complete":
        forced = bool(result.get("timedOut") or result.get("cancelled"))
        if not forced:
            raise SandboxBuildError(
                "MXC cleanup did not complete cleanly; execution state is uncertain"
            )
        # Studio killed the tree, so the workload is gone; only ACE restore is unconfirmed, and every
        # later wxc-exec start replays the journal. Report the timeout or cancel, not an error.
        logger.warning(
            "MXC could not confirm the DACL restore after a forced exit; it retries at the next start"
        )
        mxc_probe.invalidate_cache()
    return result
