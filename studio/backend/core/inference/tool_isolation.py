# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Capability and session-only consent primitives for local tool isolation."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import hmac
import secrets
import threading
import time
from typing import Any, Literal, Mapping


ToolExecutionMode = Literal["os_isolation_required", "limited", "full"]


@dataclass(frozen = True)
class ToolIsolationCapability:
    environment: str
    backend: str
    protection_state: str
    profile_id: str
    probe_generation: str
    environment_fingerprint: str
    reason: str
    remediation: str
    retryable: bool
    qualified: bool
    available: bool = False
    limitations: tuple[str, ...] = ()


@dataclass(frozen = True)
class IssuedLimitedGrant:
    token: str
    expires_at: float
    probe_generation: str
    mode: Literal["limited"] = "limited"


@dataclass(frozen = True)
class ValidatedLimitedGrant:
    current_subject: str
    tool_ui_session_id: str
    probe_generation: str
    expires_at: float
    mode: Literal["limited"] = "limited"


@dataclass(frozen = True)
class _StoredLimitedGrant:
    token_digest: bytes
    current_subject: str
    tool_ui_session_id: str
    probe_generation: str
    expires_monotonic: float
    expires_at: float


class LimitedGrantError(ValueError):
    """A Limited grant is missing, stale, expired, or bound to another request."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class LimitedGrantStore:
    """Small, bounded, process-local store for UI-authorized Limited mode."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 300.0,
        max_entries: int = 1024,
    ):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._ttl_seconds = float(ttl_seconds)
        self._max_entries = int(max_entries)
        self._records: OrderedDict[bytes, _StoredLimitedGrant] = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def _digest(token: str) -> bytes:
        return hashlib.sha256(token.encode("utf-8", errors = "strict")).digest()

    def _cleanup_locked(self, now: float) -> None:
        expired = [
            digest for digest, record in self._records.items() if record.expires_monotonic <= now
        ]
        for digest in expired:
            self._records.pop(digest, None)

    def issue(
        self,
        *,
        current_subject: str,
        tool_ui_session_id: str,
        probe_generation: str,
        requested_mode: ToolExecutionMode = "limited",
    ) -> IssuedLimitedGrant:
        if requested_mode != "limited":
            raise LimitedGrantError(
                "LIMITED_MODE_REQUIRED", "A Limited grant can authorize only Limited mode."
            )
        if not current_subject:
            raise LimitedGrantError("INVALID_SUBJECT", "An authenticated subject is required.")
        if not tool_ui_session_id:
            raise LimitedGrantError(
                "INVALID_UI_SESSION", "A non-empty tool UI session identifier is required."
            )
        if not probe_generation:
            raise LimitedGrantError(
                "INVALID_PROBE_GENERATION", "A capability probe generation is required."
            )

        now_monotonic = time.monotonic()
        expires_at = time.time() + self._ttl_seconds
        with self._lock:
            self._cleanup_locked(now_monotonic)
            while len(self._records) >= self._max_entries:
                self._records.popitem(last = False)
            while True:
                token = secrets.token_urlsafe(32)
                digest = self._digest(token)
                if digest not in self._records:
                    break
            self._records[digest] = _StoredLimitedGrant(
                token_digest = digest,
                current_subject = current_subject,
                tool_ui_session_id = tool_ui_session_id,
                probe_generation = probe_generation,
                expires_monotonic = now_monotonic + self._ttl_seconds,
                expires_at = expires_at,
            )
        return IssuedLimitedGrant(
            token = token,
            expires_at = expires_at,
            probe_generation = probe_generation,
        )

    def validate(
        self,
        token: str | None,
        *,
        current_subject: str,
        tool_ui_session_id: str | None,
        probe_generation: str,
        requested_mode: ToolExecutionMode,
    ) -> ValidatedLimitedGrant:
        if requested_mode != "limited":
            raise LimitedGrantError(
                "LIMITED_MODE_REQUIRED", "A Limited grant can authorize only Limited mode."
            )
        if not token or not tool_ui_session_id:
            raise LimitedGrantError(
                "LIMITED_GRANT_REQUIRED",
                "Limited mode requires a grant from the current Studio UI session.",
            )

        supplied_digest = self._digest(token)
        now_monotonic = time.monotonic()
        with self._lock:
            record = self._records.get(supplied_digest)
            expected_digest = record.token_digest if record is not None else secrets.token_bytes(32)
            token_matches = hmac.compare_digest(supplied_digest, expected_digest)
            self._cleanup_locked(now_monotonic)
            if record is None or not token_matches:
                raise LimitedGrantError(
                    "INVALID_LIMITED_GRANT",
                    "The Limited-mode grant is invalid or no longer available.",
                )
            if record.expires_monotonic <= now_monotonic:
                raise LimitedGrantError(
                    "EXPIRED_LIMITED_GRANT", "The Limited-mode grant has expired."
                )
            if record.current_subject != current_subject:
                raise LimitedGrantError(
                    "LIMITED_GRANT_SCOPE_MISMATCH",
                    "The Limited-mode grant does not belong to this authenticated subject.",
                )
            if record.tool_ui_session_id != tool_ui_session_id:
                raise LimitedGrantError(
                    "LIMITED_GRANT_SCOPE_MISMATCH",
                    "The Limited-mode grant does not belong to this Studio UI session.",
                )
            if record.probe_generation != probe_generation:
                raise LimitedGrantError(
                    "CAPABILITY_CHANGED",
                    "Tool-isolation capability changed; request Limited access again.",
                )
            return ValidatedLimitedGrant(
                current_subject = record.current_subject,
                tool_ui_session_id = record.tool_ui_session_id,
                probe_generation = record.probe_generation,
                expires_at = record.expires_at,
            )


_LIMITED_GRANTS = LimitedGrantStore()


def _snapshot_value(snapshot: object, name: str) -> Any:
    if isinstance(snapshot, Mapping):
        return snapshot[name]
    return getattr(snapshot, name)


def capability_snapshot(*, force: bool = False) -> ToolIsolationCapability:
    """Return the OS backend's capability using a stable API-facing shape."""

    from core.inference.os_sandbox import capability_snapshot as os_capability_snapshot

    snapshot = os_capability_snapshot(force = force)
    return ToolIsolationCapability(
        environment = str(_snapshot_value(snapshot, "environment")),
        backend = str(_snapshot_value(snapshot, "backend")),
        protection_state = str(_snapshot_value(snapshot, "protection_state")),
        profile_id = str(_snapshot_value(snapshot, "profile_id")),
        probe_generation = str(_snapshot_value(snapshot, "probe_generation")),
        environment_fingerprint = str(_snapshot_value(snapshot, "environment_fingerprint")),
        reason = str(_snapshot_value(snapshot, "reason")),
        remediation = str(_snapshot_value(snapshot, "remediation")),
        retryable = bool(_snapshot_value(snapshot, "retryable")),
        available = bool(_snapshot_value(snapshot, "available")),
        qualified = bool(_snapshot_value(snapshot, "qualified")),
        limitations = tuple(str(item) for item in _snapshot_value(snapshot, "limitations")),
    )


def issue_limited_grant(
    *,
    current_subject: str,
    tool_ui_session_id: str,
    probe_generation: str,
    requested_mode: ToolExecutionMode = "limited",
) -> IssuedLimitedGrant:
    return _LIMITED_GRANTS.issue(
        current_subject = current_subject,
        tool_ui_session_id = tool_ui_session_id,
        probe_generation = probe_generation,
        requested_mode = requested_mode,
    )


def validate_limited_grant(
    token: str | None,
    *,
    current_subject: str,
    tool_ui_session_id: str | None,
    probe_generation: str,
    requested_mode: ToolExecutionMode,
) -> ValidatedLimitedGrant:
    """Validate the grant against the caller's immediately re-probed generation."""

    return _LIMITED_GRANTS.validate(
        token,
        current_subject = current_subject,
        tool_ui_session_id = tool_ui_session_id,
        probe_generation = probe_generation,
        requested_mode = requested_mode,
    )
