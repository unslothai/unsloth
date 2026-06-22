# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Consent gate for loads that would execute model repo code.

The LOAD-path counterpart to the capability probes (which read raw config and
never need remote code). A deliberate load calls ``evaluate_remote_code_consent``
right before passing ``trust_remote_code=True``, and decides by the severity of a
static scan of the repo's ``auto_map`` ``.py``:

* No ``auto_map`` in any config (model/tokenizer/processor) -> nothing runs; allow.
* CRITICAL (reverse shell, IMDS, credential theft, droppers) -> hard block, never
  approvable, even first-party (defends a compromised trusted repo).
* HIGH/MEDIUM (subprocess/exec/eval/network/b64decode, or a large embedded blob) ->
  block but user-approvable: the dialog pins approval to the scanned ``fingerprint``.
  Applies to EVERY repo; first-party is not a blanket bypass.
* ``auto_map`` present but unscannable (gated/offline/listing failure) -> fail
  closed: hard block, since we cannot verify or fingerprint unseen code.

Hardening + consent, not a sandbox: static patterns are evadable, so subprocess /
venv isolation remains the containment layer.
"""

from dataclasses import dataclass, field
from typing import Optional

from loggers import get_logger

from utils.security.remote_code_scan import (
    CRITICAL,
    HIGH,
    MEDIUM,
    REMOTE_CODE_CONFIG_FILES,
    RemoteCodeUnscannable,
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
        """Machine-readable detail for the frontend. ``error_kind`` splits a
        user-approvable prompt (``remote_code_consent_required``) from a CRITICAL hard
        block (``remote_code_blocked``).
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


# trust_remote_code runs auto_map from ANY of these configs (model/tokenizer/
# processor), so all of them gate consent (scanning only config.json/tokenizer would
# miss a custom-processor VLM). The list lives in remote_code_scan so the gate and
# scanner stay in lockstep.
_REMOTE_CODE_CONFIG_FILES = REMOTE_CODE_CONFIG_FILES


def _config_has_auto_map(model_name: str, hf_token: Optional[str] = None) -> Optional[bool]:
    """Whether any config (model/tokenizer/processor) declares an ``auto_map`` the load
    would execute. Reads raw JSON with ``hf_token``; returns None when a config is
    unreadable (transient/auth) so the caller treats it as "unknown" and scans, False
    when the repo genuinely ships none. GGUF is False (llama.cpp never runs auto_map);
    this is the single chokepoint for that rule, shared by validate / scan / worker.
    """
    # A direct .gguf FILE loads via llama.cpp (auto_map inert). A bare repo id ending in
    # .gguf can still ship safetensors + auto_map, so it falls through to the scan.
    if _is_direct_gguf_file_ref(model_name):
        return False
    configs = _load_remote_code_configs(model_name, hf_token)
    if configs is None:
        return None
    if not any(bool((cfg or {}).get("auto_map")) for cfg in configs):
        return False
    # auto_map present but a GGUF repo -> inert. Checked only when auto_map exists, so
    # normal models skip the extra listing.
    if _is_gguf_repo(model_name, hf_token):
        logger.debug("Ignoring auto_map for GGUF repo '%s' (llama.cpp never runs it).", model_name)
        return False
    return True


def _is_direct_gguf_file_ref(model_name: str) -> bool:
    """Whether ``model_name`` names a specific ``.gguf`` FILE (llama.cpp), not a repo:
    a local ``.gguf`` path or a remote ``org/repo/.../file.gguf`` (>= 2 slashes). A bare
    ``org/name.gguf`` is a repo id that can still ship safetensors + auto_map, so it
    falls through to the scan.
    """
    name = model_name or ""
    if not name.lower().endswith(".gguf"):
        return False
    try:
        from utils.paths import is_local_path
        if is_local_path(name):
            return True
    except Exception:
        pass
    # Remote: a file reference is repo_id ("org/name") + filename => >= 2 slashes.
    return name.count("/") >= 2


# Weight formats transformers can load (and thus run auto_map for). A repo shipping any
# of these is not GGUF-only -- the user could load it through transformers -- so consent
# still applies even if it also ships a .gguf.
_TRANSFORMERS_WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".h5",
    ".msgpack",
    ".onnx",
    ".ckpt",
)


def _is_gguf_repo(model_name: str, hf_token: Optional[str] = None) -> bool:
    """Whether a remote repo loads only through llama.cpp (GGUF weights and NO
    transformers-loadable weights), making its config inert. A repo that also ships
    transformers weights is NOT GGUF (auto_map could run, so still gate). A listing
    failure is treated as "not known-GGUF" (fall through to scan).
    """
    try:
        from utils.paths import is_local_path

        if is_local_path(model_name):
            return False
        from huggingface_hub import list_repo_files

        files = [f.lower() for f in list_repo_files(model_name, token = hf_token)]
        has_gguf = any(f.endswith(".gguf") for f in files)
        has_transformers_weights = any(f.endswith(_TRANSFORMERS_WEIGHT_SUFFIXES) for f in files)
        return has_gguf and not has_transformers_weights
    except Exception:
        return False


def _load_remote_code_configs(model_name: str, hf_token: Optional[str] = None) -> Optional[list]:
    """Read every config that can declare ``auto_map`` (model/tokenizer/processor) as
    raw dicts. Returns the configs present (``[]`` when all 404, a definitive "no
    auto_map"), or None when one is unreadable (transient/auth) so the caller scans.
    The 404-vs-error split matters: real absence is "allow"; unreadable is "unknown".
    """
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
                continue  # genuine 404 -> truly absent
            except Exception:
                # Transient/auth failure is not "absent" -> fail closed to "unknown" so
                # the caller scans (a tokenizer/processor-only auto_map must not slip by).
                return None
            configs.append(json.loads(Path(p).read_text()))
        # Every config was read or a genuine 404 -> an empty list is a definitive
        # "no auto_map", not "unknown".
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
    subject: Optional[str] = None,
) -> RemoteCodeDecision:
    """Single-repo consent; thin wrapper over the for_targets form. ``trusted_org`` is
    accepted for backward compatibility but no longer changes the decision.
    """
    return evaluate_remote_code_consent_for_targets(
        [model_name],
        hf_token,
        trust_remote_code = trust_remote_code,
        approved_fingerprint = approved_fingerprint,
        subject = subject,
    )


def _fingerprint_target_key(target: str) -> str:
    """Namespace key for a target in the combined fingerprint. The pin is over CODE
    BYTES, not the repo-id spelling: the scan canonicalizes a cached repo's casing while
    workers pass raw input, so lowercase Hub ids (keep local paths as-is) or ``Org/Model``
    vs ``org/model`` would fingerprint differently and reject a valid approval.
    """
    try:
        from utils.paths import is_local_path
        if is_local_path(target):
            return target
    except Exception:
        return target
    return target.lower()


def _auto_approved_decision(model_name: str, stored) -> RemoteCodeDecision:
    """A not-blocked decision built from a cached approval (no scan/download needed)."""
    return RemoteCodeDecision(
        model_name,
        True,
        False,
        stored.fingerprint,
        stored.max_severity,
        "",
        "approved by cache (sha match)",
        approvable = True,
    )


def evaluate_remote_code_consent_for_targets(
    targets,
    hf_token: Optional[str] = None,
    *,
    trust_remote_code: bool,
    approved_fingerprint: Optional[str] = None,
    subject: Optional[str] = None,
) -> RemoteCodeDecision:
    """Decide whether a ``trust_remote_code=True`` load may proceed, over every repo whose
    code the load would execute. A LoRA load runs adapter AND base code, so all targets
    are scanned as ONE unit and pinned by ONE fingerprint over the union of their ``.py``
    -- one approval covers every repo, and a base-only fingerprint can't leave an
    adapter's own ``auto_map`` unreviewed. On ``blocked``, the caller surfaces
    ``response_payload()`` and retries with ``approved_fingerprint`` if the user accepts.

    When ``subject`` is given, a prior approval by that user is honored: a commit-SHA match
    auto-approves without re-downloading; otherwise the stored fingerprint seeds the
    authoritative content check below, and a genuine approval is recorded for next time.
    """
    targets = [t for t in dict.fromkeys(targets) if t]
    primary = targets[0] if targets else ""

    if not trust_remote_code:
        return RemoteCodeDecision(
            primary, False, False, None, None, "", "trust_remote_code disabled"
        )

    # Persistent per-user approval fast path. A SHA match means a byte-identical tree to
    # the approved revision, so skip the scan/download entirely; otherwise (local/offline,
    # or the SHA moved) seed the fingerprint so the content check still auto-approves an
    # unchanged repo and re-prompts a changed one.
    caller_approved_fingerprint = approved_fingerprint
    if subject:
        from utils.security import remote_code_approvals

        _ak = remote_code_approvals.approval_target_key(targets)
        _stored = remote_code_approvals.lookup(subject, _ak)
        if _stored is not None:
            _sha = remote_code_approvals.resolve_combined_sha(targets, hf_token)
            if _sha is not None and _sha == _stored.commit_sha:
                logger.info("trust_remote_code approved from cache (sha match) for '%s'", primary)
                return _auto_approved_decision(primary, _stored)
            approved_fingerprint = approved_fingerprint or _stored.fingerprint

    # Gather executable .py from every target that ships auto_map. A definitively
    # auto_map-free target contributes nothing; an unreadable config is scanned anyway.
    # If ANY target's code is present but unscannable, fail the whole load closed.
    combined: dict = {}
    has_remote_code = False
    for target in targets:
        if _config_has_auto_map(target, hf_token) is False:
            continue
        has_remote_code = True
        try:
            files = repo_remote_code_files(target, hf_token = hf_token)
        except RemoteCodeUnscannable:
            logger.warning(
                "Blocking trust_remote_code load of '%s': remote code present (auto_map) "
                "but could not be downloaded and scanned.",
                target,
            )
            return RemoteCodeDecision(
                target,
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
        # Namespace filenames by (casing-normalized) target so two repos' same-named
        # files stay distinct and the pin tracks code, not the repo-id spelling.
        target_key = _fingerprint_target_key(target)
        for filename, body in files.items():
            combined[f"{target_key}\0{filename}"] = body

    if not has_remote_code:
        return RemoteCodeDecision(
            primary, False, False, None, None, "", "no auto_map; trust_remote_code is a no-op"
        )

    if not combined:
        # auto_map declared but no executable .py (e.g. a GGUF repo's vestigial
        # auto_map) -> nothing to run -> allow.
        return RemoteCodeDecision(
            primary,
            False,
            False,
            None,
            None,
            "",
            "auto_map declared but no executable code present; trust_remote_code is a no-op",
        )

    result = scan_remote_code_files(combined)
    fingerprint = remote_code_fingerprint(combined)
    sev = result.max_severity

    # CRITICAL is never approvable; a fingerprint pins approval for lower severities only.
    approvable = sev != CRITICAL
    approved = (
        approvable and approved_fingerprint is not None and approved_fingerprint == fingerprint
    )

    if sev == CRITICAL:
        blocked, reason = True, "blocked: scan found CRITICAL patterns"
    elif approved:
        blocked, reason = False, "approved by fingerprint"
    elif sev == HIGH:
        # HIGH is user-approvable but must pin the fingerprint via the dialog, for every
        # repo including first-party (a compromised trusted repo still needs review).
        blocked, reason = True, "blocked: scan found HIGH patterns; approval required"
    elif sev == MEDIUM:
        # MEDIUM (e.g. a big embedded base64 blob) also pins approval like HIGH, so a
        # direct API caller can't run flagged code by just setting trust_remote_code=True.
        blocked, reason = True, "blocked: scan found MEDIUM patterns; approval required"
    else:
        blocked, reason = False, "allowed: no high-risk patterns"

    if blocked:
        logger.warning(
            "Blocking trust_remote_code load of '%s': scan severity %s (fingerprint %s)",
            primary,
            sev,
            fingerprint[:12],
        )

    # Persist a genuine user approval (the caller supplied the matching fingerprint, not a
    # cache seed) so the same unchanged repo is not re-prompted next session.
    if approved and subject and caller_approved_fingerprint == fingerprint:
        from utils.security import remote_code_approvals
        remote_code_approvals.record(
            subject,
            remote_code_approvals.approval_target_key(targets),
            commit_sha = remote_code_approvals.resolve_combined_sha(targets, hf_token),
            fingerprint = fingerprint,
            max_severity = sev,
        )

    return RemoteCodeDecision(
        primary,
        True,
        blocked,
        fingerprint,
        sev,
        result.summary(),
        reason,
        findings = result.findings_payload(),
        approvable = approvable,
    )
