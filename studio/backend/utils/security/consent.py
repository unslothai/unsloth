# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Consent gate for loads that would execute model repo code.

This is the LOAD-path counterpart to the capability probes. Detection never
calls this: it reads raw ``config.json`` and never needs remote code. A
deliberate load (train / infer / export) calls ``evaluate_remote_code_consent``
right before it would pass ``trust_remote_code=True`` to the loader.

The gate answers: *is it safe to run this model's repo code now?*

* No ``auto_map`` in any config (model/tokenizer/processor) -> ``trust_remote_code``
  executes nothing; allow (``has_remote_code=False``).
* ``auto_map`` present -> statically scan the repo's ``.py`` (reusing
  ``remote_code_scan``) and decide by severity:
    - CRITICAL findings (reverse shell, cloud-metadata/IMDS, credential theft,
      remote-code loaders, droppers) -> hard block (never approvable), even for
      first-party repos (defense against a compromised trusted repo).
    - HIGH findings (``subprocess``/``exec``/``eval``/network/``b64decode``) ->
      block, but user-approvable: the consent dialog pins approval to the scanned
      ``fingerprint``. This applies to EVERY repo, including first-party
      ``unsloth``/``nvidia`` -- the org is not a blanket bypass, since a
      compromised first-party repo with HIGH code still warrants per-version
      review. (A first-party model whose remote code scans clean, like
      DeepSeek-OCR, still prompts for consent because it has an ``auto_map``, then
      loads; one that tripped HIGH would load only after a pinned-fingerprint
      approval.)
  In every blocking case the caller surfaces ``findings_summary`` +
  ``fingerprint`` so the user can make an informed, pinned decision and retry.
* ``auto_map`` present but the code cannot be fully fetched/listed to scan
  (gated/offline/transient, or a repo-listing failure that could hide an imported
  helper) -> fail closed: hard block, no approval path, since we cannot verify or
  fingerprint code we cannot see. Retry when the repo is reachable with the token.

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
    REMOTE_CODE_CONFIG_FILES,
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


# ``trust_remote_code`` executes custom code from EVERY config that can carry an
# ``auto_map`` (model architecture, tokenizer, image/feature processor, processor,
# video processor). transformers' AutoImageProcessor / AutoFeatureExtractor /
# AutoProcessor read auto_map from these and run the referenced .py under
# trust_remote_code, so any of them declaring auto_map must gate the consent flow
# -- scanning only config.json/tokenizer would miss a custom-processor VLM
# entirely. The list lives in ``remote_code_scan`` (the scanner) so the gate and
# the scanner stay in lockstep.
_REMOTE_CODE_CONFIG_FILES = REMOTE_CODE_CONFIG_FILES


def _config_has_auto_map(model_name: str, hf_token: Optional[str] = None) -> Optional[bool]:
    """Whether ANY config that can carry one (model/tokenizer/processor) declares
    an ``auto_map`` (repo code).

    Reads raw JSON only (never executes), using ``hf_token`` so private/gated
    repos resolve with the same auth the later load will use. Returns ``None``
    only when a config could not be read due to a transient/auth error, so the
    caller treats it as "unknown" and scans rather than assuming there is no
    remote code. A repo that genuinely ships none of these configs returns
    ``False`` (definitively no remote code).
    """
    configs = _load_remote_code_configs(model_name, hf_token)
    if configs is None:
        return None
    return any(bool((cfg or {}).get("auto_map")) for cfg in configs)


def _load_remote_code_configs(model_name: str, hf_token: Optional[str] = None) -> Optional[list]:
    """Read every config that can declare ``auto_map`` (model/tokenizer/processor)
    as raw dicts.

    Returns the configs that exist (possibly ``[]`` when the repo genuinely ships
    none of them -- a 404 on every one -- which definitively means no config-based
    auto_map), or ``None`` only when a config could NOT be read due to a transient/
    auth/network failure, so the caller treats that repo as "unknown" and scans it.
    The 404-vs-error split matters: a real absence is "no remote code" (allow); an
    unreadable config is "unknown" (fail closed to a scan)."""
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
        from huggingface_hub.utils import EntryNotFoundError

        configs = []
        for name in _REMOTE_CODE_CONFIG_FILES:
            try:
                p = hf_hub_download(repo_id = model_name, filename = name, token = hf_token)
            except EntryNotFoundError:
                continue  # this config genuinely does not exist (404) -> truly absent
            except Exception:
                # A transient/auth/network failure is NOT "absent": treating it as
                # absent could let a tokenizer/processor-only auto_map slip through
                # as "no remote code". Fail closed to "unknown" so the caller scans.
                return None
            configs.append(json.loads(Path(p).read_text()))
        # Reaching here means every config was either read or a genuine 404, so an
        # empty list is a definitive "no config-based auto_map", not "unknown".
        return configs
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

    ``trusted_org`` is accepted for backward compatibility but no longer changes
    the decision: first-party is not a blanket bypass, so a HIGH-severity result
    requires per-version approval for EVERY repo and CRITICAL is a hard block for
    every repo. The parameter is retained so existing call sites keep working.
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
        # auto_map present but the code could NOT be fully fetched/listed to scan
        # (gated/offline/transient, or a repo-listing failure that could hide an
        # imported helper module). We cannot verify -- or fingerprint -- code we
        # can't see, so fail closed: block with no approval path. The load can be
        # retried once the repo is reachable with the right token.
        logger.warning(
            "Blocking trust_remote_code load of '%s': remote code present (auto_map) "
            "but could not be downloaded and scanned.",
            model_name,
        )
        return RemoteCodeDecision(
            model_name,
            True,
            True,
            None,
            None,
            "Remote code is present (auto_map) but could not be downloaded and "
            "scanned. Retry when the repo is reachable and the correct Hugging Face "
            "token is set.",
            "blocked: remote code could not be scanned",
            approvable = False,
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
        # HIGH (exec/eval/subprocess/network/b64decode) is user-approvable, but
        # EVERY repo -- including first-party unsloth/nvidia -- must pin approval to
        # the scanned fingerprint through the consent dialog. The org is no longer a
        # blanket bypass: a compromised first-party repo with HIGH code still
        # requires explicit, per-version review. CRITICAL stays a hard block above.
        blocked, reason = True, "blocked: scan found HIGH patterns; approval required"
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
