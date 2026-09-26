# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Direct adapter for Microsoft's pinned ``wxc-exec.exe`` release."""

from __future__ import annotations

import base64
import os
import subprocess

from . import mxc_policy, mxc_runtime

MAX_CONFIG_BASE64 = 24_000


class MxcAdapterError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        stage: str,
        may_have_started: bool = False,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.may_have_started = may_have_started


class MxcLaunchCancelled(MxcAdapterError):
    """The caller cancelled before WXC dispatch, so host replay is forbidden."""


def _control_environment() -> dict[str, str]:
    allowed = {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "PROGRAMDATA"}
    env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    # Always set, even with the fallback off: recovery at every start must read the same journal.
    env["MXC_DACL_STATE_DIR"] = str(mxc_runtime.dacl_state_dir())
    return env


def spawn(
    request: dict,
    *,
    cancel_event = None,
    popen_kwargs: dict | None = None,
):
    """Verify the request and start the approved WXC executable directly."""
    if cancel_event is not None and cancel_event.is_set():
        raise MxcLaunchCancelled(
            "MXC launch cancelled before dispatch",
            stage = "startup",
        )

    config = request.get("config")
    if not isinstance(config, dict):
        raise MxcAdapterError("the MXC configuration is missing", stage = "policy")
    encoded_config = mxc_policy.canonical_config_bytes(config)
    if encoded_config != request.get("configBytes"):
        raise MxcAdapterError("the MXC configuration changed before dispatch", stage = "policy")
    if mxc_policy.compute_policy_hash(config) != request.get("policyHash"):
        raise MxcAdapterError("the MXC configuration hash is invalid", stage = "policy")
    fallback = config.get("fallback")
    if (
        not isinstance(fallback, dict)
        or set(fallback) != {"allowDaclMutation"}
        or type(fallback["allowDaclMutation"]) is not bool
    ):
        raise MxcAdapterError("the MXC DACL fallback policy is malformed", stage = "policy")
    if fallback["allowDaclMutation"] and not mxc_policy.dacl_fallback_enabled():
        raise MxcAdapterError("MXC DACL fallback is not enabled on this host", stage = "policy")
    if config.get("ui", {}).get("disable") is not False:
        raise MxcAdapterError("the Studio MXC UI policy is not enabled", stage = "policy")

    encoded = base64.b64encode(encoded_config).decode("ascii")
    if len(encoded) > MAX_CONFIG_BASE64:
        raise MxcAdapterError(
            "the MXC configuration exceeds the safe Windows command-line bound",
            stage = "policy",
        )

    runtime_lease = None
    proc = None
    try:
        runtime_lease = mxc_runtime.acquire_runtime()
        mxc_policy.verify_launch_identities(request)
        if cancel_event is not None and cancel_event.is_set():
            raise MxcLaunchCancelled(
                "MXC launch cancelled before dispatch",
                stage = "startup",
            )

        options = dict(popen_kwargs or {})
        options.pop("preexec_fn", None)
        options.pop("pass_fds", None)
        options.update(
            stdin = subprocess.DEVNULL,
            cwd = str(runtime_lease.info.path.parent),
            env = _control_environment(),
            close_fds = True,
        )
        try:
            proc = subprocess.Popen(
                [str(runtime_lease.info.path), "--config-base64", encoded],
                **options,
            )
        except OSError as exc:
            raise MxcAdapterError(
                f"could not start the approved wxc-exec.exe: {exc}",
                stage = "spawn",
            ) from exc

        # From this point WXC may have created the workload. No caller may replay
        # the original command on the host.
        proc._mxc_dispatched = True
        proc._mxc_backend_tier = "unknown"
        proc._mxc_policy_hash = request["policyHash"]
        proc._mxc_dacl = bool(fallback["allowDaclMutation"])
        proc._mxc_runtime_lease = runtime_lease
        runtime_lease = None
        return proc
    except MxcAdapterError:
        raise
    except mxc_policy.MxcPolicyError as exc:
        # The workdir or runtime changed since the policy was built: a refusal, never a host replay.
        raise MxcAdapterError(str(exc), stage = "policy", may_have_started = proc is not None) from exc
    except Exception as exc:
        raise MxcAdapterError(
            str(exc),
            stage = "dispatch" if proc is not None else "startup",
            may_have_started = proc is not None,
        ) from exc
    finally:
        if runtime_lease is not None:
            runtime_lease.release()


def completion_result(proc) -> dict:
    if not getattr(proc, "_mxc_dispatched", False):
        raise MxcAdapterError(
            "the process has no WXC dispatch evidence",
            stage = "completion",
            may_have_started = True,
        )
    if proc.poll() is None:
        raise MxcAdapterError(
            "wxc-exec.exe has not completed",
            stage = "completion",
            may_have_started = True,
        )
    reason = getattr(proc, "_unsloth_completion_reason", None)
    if reason not in {"finished", "timed_out", "cancelled"}:
        raise MxcAdapterError(
            "wxc-exec.exe exited without a trusted Studio completion state",
            stage = "completion",
            may_have_started = True,
        )
    cleanup = "complete"
    if reason != "finished" and getattr(proc, "_mxc_dacl", False):
        # A forced kill skips wxc-exec's own ACE restore; replay the journal now, not at the next start.
        cleanup = (
            "complete" if mxc_runtime.recover_dacl_state(_control_environment()) else "uncertain"
        )
    return {
        "exitCode": int(proc.returncode),
        "timedOut": reason == "timed_out",
        "cancelled": reason == "cancelled",
        "cleanup": cleanup,
        "backendTier": "unknown",
        "policyHash": getattr(proc, "_mxc_policy_hash", ""),
    }


def release_runtime(proc) -> None:
    runtime_lease = getattr(proc, "_mxc_runtime_lease", None)
    if runtime_lease is not None:
        proc._mxc_runtime_lease = None
        runtime_lease.release()


def abort(proc, *, grace_seconds: float = 1) -> None:
    """Reclaim WXC and its descendants for capability-probe failures."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdin = subprocess.DEVNULL,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                timeout = 5,
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
                check = False,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout = grace_seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout = 5)
