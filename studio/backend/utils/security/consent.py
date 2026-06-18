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
    an ``auto_map`` (repo code) THAT THE LOAD WOULD EXECUTE.

    Reads raw JSON only (never executes), using ``hf_token`` so private/gated
    repos resolve with the same auth the later load will use. Returns ``None``
    only when a config could not be read due to a transient/auth error, so the
    caller treats it as "unknown" and scans rather than assuming there is no
    remote code. A repo that genuinely ships none of these configs returns
    ``False`` (definitively no remote code).

    A GGUF load is special-cased to ``False``: it runs through llama.cpp, which
    NEVER executes ``auto_map``, so a GGUF repo's ``config.json`` (often copied
    verbatim from the original transformers model, ``auto_map`` and all) is inert.
    This is the single chokepoint for that rule -- validate's
    ``requires_trust_remote_code``, the scan endpoint, and the worker consent gate
    all go through here, so a GGUF model never triggers the consent flow.
    """
    # A direct .gguf FILE reference loads via llama.cpp -> config/auto_map is inert.
    # Only an actual file reference (local path, or a remote repo_id + filename)
    # qualifies; a bare repo id whose name merely ends in ".gguf" can still ship
    # safetensors + auto_map and is scanned via _is_gguf_repo below.
    if _is_direct_gguf_file_ref(model_name):
        return False
    configs = _load_remote_code_configs(model_name, hf_token)
    if configs is None:
        return None
    if not any(bool((cfg or {}).get("auto_map")) for cfg in configs):
        return False
    # auto_map is declared, but if this is a GGUF repo (loads via llama.cpp), the
    # config is inert and the auto_map never runs -- ignore it. Only checked once an
    # auto_map is actually present, so the extra listing is avoided for normal models.
    if _is_gguf_repo(model_name, hf_token):
        logger.debug("Ignoring auto_map for GGUF repo '%s' (llama.cpp never runs it).", model_name)
        return False
    return True


def _is_direct_gguf_file_ref(model_name: str) -> bool:
    """Whether ``model_name`` names a specific ``.gguf`` FILE (a llama.cpp load),
    not a transformers repo.

    True for a local ``.gguf`` path, or a remote ``org/repo/.../file.gguf`` (repo id
    plus filename, i.e. three or more ``/``-separated segments). A bare two-segment
    ``org/name.gguf`` is a REPO id whose name merely ends in ``.gguf`` -- it can still
    ship ``safetensors`` + ``auto_map`` Python that transformers would execute, so it
    is NOT treated as a direct file reference and must fall through to the scan.
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


# Weight formats transformers can load (and therefore run ``auto_map`` for). If a
# repo ships ANY of these, it is NOT GGUF-only: the user could load that weight set
# through transformers, where the custom code WOULD execute, so consent still applies.
# ``.safetensors`` is listed, but so are the pickle/legacy formats -- a repo with a
# ``.gguf`` plus a ``pytorch_model.bin`` is a transformers-loadable repo, not a
# llama.cpp-only one, and must still be gated.
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
    """Whether a remote repo loads through llama.cpp (GGUF), making its config inert.

    True only when the repo ships ``.gguf`` weights and NO transformers-loadable
    weights (``.safetensors``/``.bin``/``.pt``/``.pth``/``.h5``/``.msgpack``/
    ``.onnx``/``.ckpt``). A repo that also ships any of those is NOT treated as GGUF:
    the user could load that weight set through transformers, where ``auto_map``
    WOULD run, so the consent gate must still apply. Direct ``.gguf`` file references
    are handled by ``_is_direct_gguf_file_ref`` in the caller; a listing failure here
    is treated as "not known-GGUF" (fall through to scan).
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
    """Decide whether a ``trust_remote_code=True`` load of a single repo may proceed.

    Thin wrapper over ``evaluate_remote_code_consent_for_targets`` for the common
    single-repo case. ``trusted_org`` is accepted for backward compatibility but no
    longer changes the decision (first-party is not a blanket bypass).
    """
    return evaluate_remote_code_consent_for_targets(
        [model_name],
        hf_token,
        trust_remote_code = trust_remote_code,
        approved_fingerprint = approved_fingerprint,
    )


def _fingerprint_target_key(target: str) -> str:
    """Namespace key for a target in the combined fingerprint.

    The fingerprint pins the CODE BYTES, not one caller's spelling of the repo id.
    Hub repo ids are case-insensitive, but the scan endpoint canonicalizes a cached
    repo's casing (``resolve_cached_repo_id_case``) while the workers pass the raw
    user input, so the same code under ``Org/Model`` vs ``org/model`` would otherwise
    fingerprint differently and reject a valid approval. Lowercase Hub ids so the pin
    is casing-robust; keep local paths as-is (case-sensitive filesystems).
    """
    try:
        from utils.paths import is_local_path
        if is_local_path(target):
            return target
    except Exception:
        return target
    return target.lower()


def evaluate_remote_code_consent_for_targets(
    targets,
    hf_token: Optional[str] = None,
    *,
    trust_remote_code: bool,
    approved_fingerprint: Optional[str] = None,
) -> RemoteCodeDecision:
    """Decide whether a ``trust_remote_code=True`` load may proceed, over EVERY repo
    whose code that one load would execute.

    A LoRA load runs the adapter's AND the base's repo code, so all targets are
    scanned here as a SINGLE combined unit and pinned by ONE fingerprint over the
    union of their ``.py``. Approving the load therefore approves every repo's code
    together: a base-only approval can no longer leave an adapter's own ``auto_map``
    code unreviewed, nor make it impossible to approve with the base's fingerprint.

    When the returned decision is ``blocked``, the caller must refuse the load and
    surface ``response_payload()`` so the user can review the findings and, if they
    accept, re-issue the load with ``approved_fingerprint`` set to
    ``decision.fingerprint``.
    """
    targets = [t for t in dict.fromkeys(targets) if t]
    primary = targets[0] if targets else ""

    if not trust_remote_code:
        return RemoteCodeDecision(
            primary, False, False, None, None, "", "trust_remote_code disabled"
        )

    # ``trust_remote_code`` only executes code when a repo defines an auto_map. Gather
    # the executable .py from every target that ships auto_map code. A target whose
    # config is readable and definitively has no auto_map contributes nothing; an
    # unreadable config (private/gated/offline) is "unknown" and still scanned. If ANY
    # target's code is present but unscannable, fail closed for the whole load: we
    # cannot verify -- or fingerprint -- code we cannot see.
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
        # Namespace filenames by target so two repos' same-named files stay distinct
        # in the combined scan + fingerprint. The key uses a casing-normalized target
        # so an approval pins the code, not one caller's spelling of the repo id.
        target_key = _fingerprint_target_key(target)
        for filename, body in files.items():
            combined[f"{target_key}\0{filename}"] = body

    if not has_remote_code:
        return RemoteCodeDecision(
            primary, False, False, None, None, "", "no auto_map; trust_remote_code is a no-op"
        )

    if not combined:
        # auto_map declared but the repos ship NO executable .py (and no fetchable
        # external code): nothing to run (e.g. a GGUF repo's vestigial auto_map) -> allow.
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
    elif sev == MEDIUM:
        # MEDIUM (e.g. a large embedded base64 blob) is a weaker but real signal of
        # hidden code/data. It is also user-approvable, but must pin per-version
        # approval like HIGH so a direct API caller cannot run flagged code by simply
        # setting trust_remote_code=True without consenting. Only a CLEAN scan
        # (no findings) loads without a pinned fingerprint.
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
