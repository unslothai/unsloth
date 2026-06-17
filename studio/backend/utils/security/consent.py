# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Consent gate for loads that would execute model repo code.

This is the LOAD-path counterpart to the capability probes. Detection never
calls this: it reads raw ``config.json`` and never needs remote code. A
deliberate load (train / infer / export) calls ``evaluate_remote_code_consent``
right before it would pass ``trust_remote_code=True`` to the loader.

The gate answers: *is it safe to run this model's repo code now?*

* No ``auto_map`` in ``config.json`` -> ``trust_remote_code`` executes nothing;
  allow (``has_remote_code=False``).
* ``auto_map`` present -> statically scan the repo's ``.py`` (reusing
  ``remote_code_scan``) and decide by severity and provenance:
    - CRITICAL findings (reverse shell, cloud-metadata/IMDS, credential theft,
      remote-code loaders, droppers) -> block even for first-party repos
      (defense against a compromised trusted repo), unless pinned-approved.
    - HIGH findings (``subprocess``/``exec``/``eval``/network/``b64decode``) are
      common in legitimate modeling code (DeepSeek-OCR, Kimi, etc. all trip
      HIGH), so they block only for UNTRUSTED third-party repos. First-party
      ``unsloth``/``nvidia`` repos (``is_trusted_org_repo``) load through -- the
      org is the trust anchor.
  In every blocking case the caller surfaces ``findings_summary`` +
  ``fingerprint`` so the user can make an informed, pinned decision and retry.
* ``auto_map`` present but the code cannot be fetched (gated/offline) -> cannot
  verify; allow with a warning, since the load only reaches here on an explicit
  opt-in (user toggle or trusted-org auto-enable). We never silently *run* code
  the scan flagged, but we do not break legitimate gated repos either.

The gate is hardening + consent-driven, not a hard sandbox: a determined attacker
can obfuscate past static patterns. Its job is to raise the bar and inform
consent; subprocess/venv isolation remains the containment layer.
"""

from dataclasses import dataclass, field
from typing import Optional

from loggers import get_logger

from utils.security.remote_code_scan import (
    CRITICAL,
    HIGH,
    remote_code_fingerprint,
    repo_remote_code_files,
    scan_remote_code_files,
)

logger = get_logger(__name__)


@dataclass
class RemoteCodeDecision:
    """Outcome of the consent gate for one (model, trust_remote_code) load."""

    model_name: str
    has_remote_code: bool
    blocked: bool
    fingerprint: Optional[str]
    max_severity: Optional[str]
    findings_summary: str
    reason: str
    findings: list = field(default_factory = list)  # structured [{severity,file,check,evidence}]
    approvable: bool = True  # False only for CRITICAL (user cannot override)

    def response_payload(self) -> dict:
        """Machine-readable detail for the frontend to render + pin approval.

        ``error_kind`` distinguishes a user-approvable prompt from a hard block:
        CRITICAL is ``remote_code_blocked`` (no override); anything else
        approvable is ``remote_code_consent_required``.
        """
        return {
            "error_kind": (
                "remote_code_consent_required" if self.approvable else "remote_code_blocked"
            ),
            "model_name": self.model_name,
            "has_remote_code": self.has_remote_code,
            "approvable": self.approvable,
            "fingerprint": self.fingerprint,
            "max_severity": self.max_severity,
            "findings": self.findings,
            "findings_summary": self.findings_summary,
            "reason": self.reason,
        }


# ``trust_remote_code`` executes custom code for BOTH the model architecture
# (config.json auto_map) and the tokenizer (tokenizer_config.json auto_map), so
# either declaring auto_map must gate the consent flow.
_REMOTE_CODE_CONFIG_FILES = ("config.json", "tokenizer_config.json")


def _config_has_auto_map(model_name: str, hf_token: Optional[str] = None) -> Optional[bool]:
    """Whether ``config.json`` OR ``tokenizer_config.json`` declares an
    ``auto_map`` (repo code).

    Reads raw JSON only (never executes), using ``hf_token`` so private/gated
    repos resolve with the same auth the later load will use. Returns ``None``
    when no config could be read, so the caller treats it as "unknown" and scans
    rather than assuming there is no remote code.
    """
    configs = _load_remote_code_configs(model_name, hf_token)
    if configs is None:
        return None
    return any(bool((cfg or {}).get("auto_map")) for cfg in configs)


def _load_remote_code_configs(model_name: str, hf_token: Optional[str] = None) -> Optional[list]:
    """Read every config that can declare ``auto_map`` (model + tokenizer) as raw
    dicts. Returns the configs that exist (possibly ``[]`` for a local dir with
    none), or ``None`` when nothing could be read at all (e.g. an unreachable
    remote repo) so the caller treats the repo as "unknown" and scans it."""
    import json
    from pathlib import Path

    try:
        from utils.paths import is_local_path, normalize_path

        if is_local_path(model_name):
            root = Path(normalize_path(model_name)).expanduser()
            configs = []
            for name in _REMOTE_CODE_CONFIG_FILES:
                p = root / name
                if p.is_file():
                    configs.append(json.loads(p.read_text()))
            return configs

        from huggingface_hub import hf_hub_download

        configs, read_any = [], False
        for name in _REMOTE_CODE_CONFIG_FILES:
            try:
                p = hf_hub_download(repo_id = model_name, filename = name, token = hf_token)
            except Exception:
                continue  # file absent (404) or a transient miss for this name
            read_any = True
            configs.append(json.loads(Path(p).read_text()))
        return configs if read_any else None
    except Exception as exc:
        logger.debug("auto_map check could not read config for %s: %s", model_name, exc)
        return None


def evaluate_remote_code_consent(
    model_name: str,
    hf_token: Optional[str] = None,
    *,
    trust_remote_code: bool,
    approved_fingerprint: Optional[str] = None,
    trusted_org: Optional[bool] = None,
) -> RemoteCodeDecision:
    """Decide whether a ``trust_remote_code=True`` load may proceed.

    Call this immediately before a load that would execute repo code. When the
    returned decision is ``blocked``, the caller must refuse the load and surface
    ``response_payload()`` so the user can review the findings and, if they
    accept, re-issue the load with ``approved_fingerprint`` set to
    ``decision.fingerprint``.

    ``trusted_org`` may be supplied if the caller already computed
    ``is_trusted_org_repo``; otherwise it is resolved lazily, and only when it
    matters (a HIGH-severity result), to avoid an extra Hub call.
    """
    if not trust_remote_code:
        return RemoteCodeDecision(
            model_name, False, False, None, None, "", "trust_remote_code disabled"
        )

    # ``trust_remote_code`` only executes code when the repo defines an auto_map.
    # Only skip when the config is readable and definitively has no auto_map; an
    # unreadable config (private/gated/offline) is "unknown" and still scanned.
    if _config_has_auto_map(model_name, hf_token) is False:
        return RemoteCodeDecision(
            model_name,
            False,
            False,
            None,
            None,
            "",
            "no auto_map; trust_remote_code is a no-op",
        )

    files = repo_remote_code_files(model_name, hf_token = hf_token)
    if not files:
        # auto_map present but unscannable (gated/offline). Reached only on an
        # explicit opt-in; allow but record that we could not verify.
        logger.warning(
            "Remote code for '%s' could not be fetched to scan; allowing on "
            "explicit opt-in but it was NOT verified.",
            model_name,
        )
        return RemoteCodeDecision(
            model_name,
            True,
            False,
            None,
            None,
            "Remote code present (auto_map) but could not be downloaded to scan.",
            "unscannable; allowed via explicit opt-in",
        )

    result = scan_remote_code_files(files)
    fingerprint = remote_code_fingerprint(files)
    sev = result.max_severity

    # CRITICAL is never user-approvable; a supplied fingerprint only pins
    # approval for approvable severities and must never unblock CRITICAL.
    approvable = sev != CRITICAL
    approved = (
        approvable and approved_fingerprint is not None and approved_fingerprint == fingerprint
    )

    if sev == CRITICAL:
        # High-confidence malicious patterns: block even first-party repos,
        # and never allow a fingerprint to override.
        blocked, reason = True, "blocked: scan found CRITICAL patterns"
    elif approved:
        blocked, reason = False, "approved by fingerprint"
    elif sev == HIGH:
        # exec/eval/subprocess/b64decode are common in legitimate modeling code,
        # so HIGH blocks only for untrusted third-party repos. First-party repos
        # (the org is the trust anchor) load through.
        trusted = trusted_org
        if trusted is None:
            from utils.security.trusted_org import is_trusted_org_repo
            trusted = is_trusted_org_repo(model_name, hf_token = hf_token)
        if trusted:
            blocked, reason = False, "allowed: first-party repo (HIGH findings hardening)"
        else:
            blocked, reason = True, "blocked: scan found HIGH patterns in third-party repo"
    else:
        blocked, reason = False, "allowed: no high-risk patterns"

    if blocked:
        logger.warning(
            "Blocking trust_remote_code load of '%s': scan severity %s (fingerprint %s)",
            model_name,
            sev,
            fingerprint[:12],
        )

    return RemoteCodeDecision(
        model_name,
        True,
        blocked,
        fingerprint,
        sev,
        result.summary(),
        reason,
        findings = result.findings_payload(),
        approvable = approvable,
    )
