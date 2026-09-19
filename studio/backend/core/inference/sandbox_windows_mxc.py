# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Windows-only MXC backend for the shared Studio tool launch contract."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import os
import sys

from . import mxc_adapter, mxc_policy, mxc_probe, mxc_runtime
from .os_sandbox import (
    PreparedSandboxLaunch,
    SandboxBuildError,
    SandboxCapability,
    _OS_ISOLATION_SAFEGUARDS,
    _record,
)


def _capability_fingerprint(identity: str, execution_kind: str, selected_executable: str) -> str:
    return hashlib.sha256(
        repr(
            (
                identity,
                execution_kind,
                os.path.abspath(selected_executable),
                mxc_policy.PROFILE_ID,
            )
        ).encode()
    ).hexdigest()


def capability_snapshot(
    *, force=False, execution_kind=None, selected_executable=None, cancel_event=None
):
    if sys.platform != "win32":
        raise SandboxBuildError("MXC is Windows-only")
    if execution_kind is None:
        execution_kind = "python"
        selected_executable = selected_executable or sys.executable
    if execution_kind not in {"python", "terminal"} or not selected_executable:
        return SandboxCapability(
            backend="mxc-processcontainer",
            available=False,
            reason="the MXC profile requires a selected Python or Terminal executable",
            environment="win32",
            remediation="Select a supported native Windows runtime.",
            limitations=("unsupported_execution_kind",),
        )
    available, reason = mxc_probe.probe(
        selected_executable,
        execution_kind=execution_kind,
        force=force,
        cancel_event=cancel_event,
    )
    try:
        runtime_info = mxc_runtime.selected_runtime()
        identity = runtime_info.identity
    except Exception:
        runtime_info = None
        identity = "missing"
    fingerprint = _capability_fingerprint(identity, execution_kind, selected_executable)
    return SandboxCapability(
        backend="mxc-processcontainer",
        available=available,
        reason=reason,
        environment="win32",
        protection_state="preview" if available else "unavailable",
        profile_id=mxc_policy.PROFILE_ID,
        limitations=(
            "mxc_preview_not_a_security_boundary",
            "effective_tier_requires_pinned_mxc_extension",
            "network_posture_requested_not_attested",
            "nested_path_identity_not_atomic_with_mxc_grant_resolution",
            *(
                ("development_runtime_not_packaged",)
                if runtime_info and runtime_info.development
                else ()
            ),
        ),
        probe_generation=hashlib.sha256((fingerprint + str(available)).encode()).hexdigest(),
        environment_fingerprint=fingerprint,
        remediation=(
            "Install the pinned MXC supervisor and ensure BaseContainer/PSEC is enabled; "
            "Unsloth does not enable the AppContainer DACL fallback in this Preview."
        ),
    )


def prepare(plan, capability):
    try:
        request = mxc_policy.build_launch_request(plan)
        runtime_info = mxc_runtime.selected_runtime()
    except Exception as exc:
        raise SandboxBuildError(f"Windows MXC policy construction failed: {exc}") from exc
    record = _record(
        plan,
        capability,
        effective_mode="os_isolated",
        os_isolation=True,
        backend="mxc-processcontainer",
        profile_id=capability.profile_id,
        safeguards=_OS_ISOLATION_SAFEGUARDS,
        limitations=capability.limitations,
    )
    record = replace(
        record,
        network_policy="mxc_compatibility_requested_unverified",
        backend_tier="unknown",
        runtime_revision=mxc_runtime.MXC_REVISION,
        runtime_generation=runtime_info.generation,
        runtime_artifact_digest=f"sha256:{runtime_info.runner_sha256}",
        runtime_api_revision=f"patch-sha256:{mxc_runtime.MXC_PATCH_SHA256}",
        schema_version=mxc_runtime.MXC_SCHEMA_VERSION,
        policy_hash=request["policyHash"],
        execution_status="planned",
        cleanup_status="pending",
    )
    prepared = PreparedSandboxLaunch(
        argv=plan.argv,
        workdir=plan.workdir,
        env=plan.env,
        preexec_fn=None,
        backend="mxc-processcontainer",
        timeout_seconds=plan.timeout_seconds,
        execution_record=record,
    )

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
            proc = mxc_adapter.spawn(request, cancel_event=plan.cancel_event, popen_kwargs=kwargs)
        except Exception as exc:
            may_have_started = bool(getattr(exc, "may_have_started", False))
            prepared.execution_record = replace(
                prepared.execution_record,
                execution_status="unknown_start" if may_have_started else "not_started",
                completion_status="uncertain" if may_have_started else "not_started",
                cleanup_status="uncertain" if may_have_started else "complete",
            )
            mxc_probe.invalidate_cache()
            raise SandboxBuildError(
                f"Windows MXC launch failed without host replay: {exc}"
            ) from exc
        prepared.execution_record = replace(prepared.execution_record, execution_status="started")
        prepared.execution_record = replace(
            prepared.execution_record,
            backend_tier=str(getattr(proc, "_mxc_backend_tier", "unknown")),
            runtime_generation=proc._mxc_runtime_info.generation,
            runtime_artifact_digest=f"sha256:{proc._mxc_runtime_info.runner_sha256}",
        )
        prepared.cleanup_callbacks.append(lambda: mxc_adapter.release_control(proc))
        return proc

    prepared.spawn_callback = launch
    return prepared


def verify_success(prepared, proc) -> dict:
    try:
        receipt = mxc_adapter.completion_receipt(proc)
    except Exception as exc:
        if prepared.execution_record is not None:
            prepared.execution_record = replace(
                prepared.execution_record,
                completion_status="uncertain",
                cleanup_status="uncertain",
            )
        mxc_probe.invalidate_cache()
        raise SandboxBuildError(f"MXC completion state is uncertain: {exc}") from exc
    prepared.execution_record = replace(
        prepared.execution_record,
        completion_status=(
            "timed_out"
            if receipt.get("timedOut")
            else ("cancelled" if receipt.get("exitCode") == 130 else "finished")
        ),
        cleanup_status=str(receipt.get("cleanup") or "unknown"),
    )
    if receipt.get("cleanup") != "complete":
        raise SandboxBuildError(
            "MXC cleanup did not complete cleanly; execution state is uncertain"
        )
    return receipt
