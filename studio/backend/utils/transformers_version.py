# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Automatic transformers version switching.

Some newer model architectures (Ministral-3, GLM-4.7-Flash, Qwen3-30B-A3B MoE,
tiny_qwen3_moe) require transformers>=5.3.0, while Gemma 4 models require a
newer 5.x sidecar.  Dense NemotronH models (e.g. NVIDIA-Nemotron-3-Nano-4B) use
MLP layers that only transformers>=5.10 can parse natively, so they go on the
5.10 sidecar too.  Everything else needs the default 4.57.x that ships with
Unsloth.

Two separate target directories are maintained:
  - .venv_t5_530/  — transformers 5.3.0 (Ministral-3, GLM, Qwen3 MoE, etc.)
  - .venv_t5_550/  — transformers 5.5.0 (Gemma 4)
  - .venv_t5_510/  — transformers 5.10.2 (Gemma 4 Unified / 12B)

When loading a LoRA adapter with a custom name, we resolve the base model from
``adapter_config.json`` and check *that* against the model list.

Strategy:
  Training and inference run in subprocesses that activate the correct version
  via sys.path (prepending the appropriate .venv_t5_*/ directory). See:
    - core/training/worker.py
    - core/inference/worker.py

  For export (still in-process), ensure_transformers_version() does a lightweight
  sys.path swap using the same directories pre-installed by setup.sh.
"""

import ast
import importlib
import importlib.util
import json
import structlog
from loggers import get_logger
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from utils.native_path_leases import child_env_without_native_path_secret
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)

logger = get_logger(__name__)


_OFFLINE_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_offline() -> bool:
    """True if an HF offline env var is truthy (canonical strip+lower parse); gates the urllib fetches below."""
    return (
        os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in _OFFLINE_TRUE_VALUES
        or os.environ.get("TRANSFORMERS_OFFLINE", "").strip().lower() in _OFFLINE_TRUE_VALUES
    )


def hf_endpoint_unreachable(timeout: int = 3) -> bool:
    """Bounded reachability probe to the HF endpoint. A HEAD request runs in a daemon thread
    joined with a deadline, so a resolver blackhole cannot block past ~timeout+1s. True if
    unreachable. urllib natively honors *_PROXY / NO_PROXY, so this verifies real egress
    (the proxy can reach HF), not just that the proxy is up. No ML imports, so it is safe to
    call before transformers version activation. Mirrors the probe in export._hf_offline."""
    import ssl
    import threading
    import urllib.error
    import urllib.request

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    if "://" not in endpoint:
        endpoint = "https://" + endpoint

    result = {"online": False}

    def _probe():
        try:
            req = urllib.request.Request(endpoint, method = "HEAD")
            with urllib.request.urlopen(req, timeout = timeout):
                result["online"] = True
        except urllib.error.HTTPError as exc:
            # The server/proxy answered: reachable unless it is a gateway error.
            result["online"] = exc.code not in (502, 503, 504)
        except urllib.error.URLError as exc:
            # A TLS/cert failure means we DID reach the server; treat as reachable so the real
            # load surfaces it (consistent with _is_offline_related_error not retrying TLS).
            result["online"] = isinstance(exc.reason, ssl.SSLError)
        except ssl.SSLError:
            result["online"] = True
        except Exception:
            result["online"] = False

    t = threading.Thread(target = _probe, daemon = True)
    t.start()
    t.join(timeout + 1)
    return t.is_alive() or not result["online"]


def _safe_is_file(p: Path) -> bool:
    """``p.is_file()`` returning False instead of raising on a bad path."""
    try:
        return p.is_file()
    except (OSError, ValueError):
        return False


def _safe_is_dir(p: Path) -> bool:
    """``p.is_dir()`` returning False instead of raising on a bad path."""
    try:
        return p.is_dir()
    except (OSError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

# Lowercase substrings — any match in the lowered model name needs transformers 5.3.0.
TRANSFORMERS_5_MODEL_SUBSTRINGS: tuple[str, ...] = (
    "ministral-3-",  # Ministral-3-{3,8,14}B-{Instruct,Reasoning,Base}-2512
    "glm-4.7-flash",  # GLM-4.7-Flash
    "qwen3-30b-a3b",  # Qwen3-30B-A3B-Instruct-2507 and variants
    "qwen3.5",  # Qwen3.5 family (35B-A3B, etc.)
    "qwen3-next",  # Qwen3-Next and variants
    "tiny_qwen3_moe",  # imdatta0/tiny_qwen3_moe_2.8B_0.7B
    "lfm2.5-vl-450m",  # LiquidAI/LFM2.5-VL-450M
)

# Lowercase substrings for models that require transformers 5.10.x (checked first).
TRANSFORMERS_510_MODEL_SUBSTRINGS: tuple[str, ...] = (
    "gemma-4-12b",  # Gemma 4 Unified 12B
    "gemma4-12b",
)

# Lowercase substrings for models that require the Gemma 4 transformers 5.5 sidecar.
TRANSFORMERS_550_MODEL_SUBSTRINGS: tuple[str, ...] = (
    "gemma-4",  # Gemma-4 (E2B-it, E4B-it, 31B-it, 26B-A4B-it)
    "gemma4",  # Gemma-4 alternate naming
    "qwen3.6",
)

# Architecture classes / model_type values that require transformers 5.10.x.
# Checked via config.json (local or HuggingFace).
_TRANSFORMERS_510_ARCHITECTURES: set[str] = {
    "Gemma4UnifiedForConditionalGeneration",
    "Gemma4AssistantForCausalLM",
    "Gemma4UnifiedAssistantForCausalLM",
    "AstralForCausalLM",
}
_TRANSFORMERS_510_MODEL_TYPES: set[str] = {
    "gemma4_unified",
    "gemma4_assistant",
    "gemma4_unified_assistant",
    "astral",
}

# Architecture classes / model_type values that require transformers 5.5.0.
# Checked via config.json (local or HuggingFace).
_TRANSFORMERS_550_ARCHITECTURES: set[str] = {
    "Gemma4ForConditionalGeneration",
}
_TRANSFORMERS_550_MODEL_TYPES: set[str] = {
    "gemma4",
}

# Architecture classes / model_type values that require transformers 5.3.0.
# Checked via config.json (local or HuggingFace).
_TRANSFORMERS_530_ARCHITECTURES: set[str] = {
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3MoeForCausalLM",
    "Qwen3NextForCausalLM",
    "Glm4MoeLiteForCausalLM",
    "Lfm2MoeForCausalLM",
    "Lfm2VlForConditionalGeneration",
}
_TRANSFORMERS_530_MODEL_TYPES: set[str] = {
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen3_moe",
    "qwen3_next",
    "glm4_moe_lite",
    "lfm2_moe",
    "lfm2_vl",
}

# Tokenizer classes that only exist in transformers>=5.x.
_TRANSFORMERS_5_TOKENIZER_CLASSES: set[str] = {
    "TokenizersBackend",
}

# Import strings in auto_map remote .py that 4.57.x cannot satisfy but the 5.3 sidecar can.
# Fully qualified so a repo's own same-named helper cannot match. Only names ABSENT from
# 4.57.x belong here: ``modeling_layers`` (4.52+) and ``use_kernel_forward_from_hub`` (4.51+)
# import fine on default, so listing them would push ordinary custom-code models (EXAONE,
# MiniMax, Molmo2, ...) onto a sidecar. ``transformers.TokenizersBackend`` is the 5.x-only
# public re-export, listed because remote code writing ``from transformers import
# TokenizersBackend`` never names the defining submodule; safe since 4.57.x has no such name
# (only 5.x-only public re-exports belong here). The ``utils.output_capturing`` names have no
# top-level re-export in 5.3.0, so the module spelling is their only spelling.
_TRANSFORMERS_5_REMOTE_IMPORT_MARKERS: tuple[str, ...] = (
    "transformers.tokenization_utils_tokenizers",
    "transformers.utils.output_capturing",
    "transformers.TokenizersBackend",
)

# Remote modeling imports that not even the 5.3 sidecar can satisfy. A module belongs here
# only when it is absent from 5.3.0/5.5.0 as well; anything merely newer than 4.57.x goes in
# the set above, where the 5.3 floor plus the AutoConfig probe escalates to 5.5/5.10 when the
# model really needs it. Empty today: every marker found so far imports fine on 5.3.0.
_TRANSFORMERS_510_REMOTE_IMPORT_MARKERS: tuple[str, ...] = ()

# Caches keyed on (model_name, token-hash) so authed/unauthed reads stay separate (a
# gated/private repo's unauthenticated miss must not poison a later authenticated lookup).
# Offline negatives are NOT written (see the _env_offline branches) so they cannot poison a
# later online read in this persistent worker.
_tokenizer_class_cache: dict[tuple[str, str | None], bool] = {}
_config_json_cache: dict[tuple[str, str | None], dict | None] = {}
_config_needs_510_cache: dict[tuple[str, str | None], bool] = {}
_config_needs_550_cache: dict[tuple[str, str | None], bool] = {}
_config_needs_530_cache: dict[tuple[str, str | None], bool] = {}
_remote_auto_map_tier_cache: dict[tuple[str, str | None, str | None], str] = {}

# Models whose config.json got a definitive 401/403/404. Kept apart from _config_json_cache
# (configs we could read) so an unread config is never mistaken for "no auto_map".
_config_json_absent: set[tuple[str, str | None]] = set()

# AutoConfig-probe tier cache for the process lifetime (cleared on restart), keyed by
# model_name plus a local config.json signature (see _probe_cache_key) so an overwritten
# checkpoint re-probes. Not keyed by Hub sha, so the probe never imports huggingface_hub
# before a worker's sidecar venv is activated (which would pin the wrong hub).
_probe_tier_cache: dict[str, str] = {}

# Versions
TRANSFORMERS_510_VERSION = "5.10.2"
TRANSFORMERS_550_VERSION = "5.5.0"
TRANSFORMERS_530_VERSION = "5.3.0"
TRANSFORMERS_DEFAULT_VERSION = "4.57.6"
# Backwards-compat alias — points to the highest 5.x tier.
# Consumers should prefer TRANSFORMERS_510_VERSION / TRANSFORMERS_550_VERSION /
# TRANSFORMERS_530_VERSION.
TRANSFORMERS_5_VERSION = TRANSFORMERS_510_VERSION

# Pre-installed directories — created by setup.sh / setup.ps1.
from utils.paths.storage_roots import studio_root as _studio_root  # noqa: E402

_VENV_T5_530_DIR = str(_studio_root() / ".venv_t5_530")
_VENV_T5_550_DIR = str(_studio_root() / ".venv_t5_550")
_VENV_T5_510_DIR = str(_studio_root() / ".venv_t5_510")
# Backwards-compat alias
_VENV_T5_DIR = _VENV_T5_550_DIR

# llm-compressor-main shadow for FP8/FP4 export of newer-transformers models. Like the .venv_t5_*
# sidecars but also shadows llm-compressor main + compressed-tensors; installed --no-deps so it
# reuses the workspace torch (torch-agnostic).
_VENV_LLMCOMPRESSOR_DIR = str(_studio_root() / ".venv_llmcompressor")

# User-consented "latest transformers" sidecar (utils/transformers_latest.py); pinned version in a marker file.
_VENV_T5_LATEST_DIR = str(_studio_root() / ".venv_t5_latest")
_LATEST_PIN_MARKER = ".unsloth_pinned_transformers"

# Tier precedence: higher rank wins in _higher_tier. "latest" outranks every fixed tier.
_TIER_RANK = {"default": 0, "530": 1, "550": 2, "510": 3, "latest": 4}


def _higher_tier(a: str, b: str) -> str:
    return a if _TIER_RANK.get(a, 0) >= _TIER_RANK.get(b, 0) else b


def activate_transformers_for_subprocess(model_name: str, hf_token: str | None = None) -> None:
    """Activate the correct transformers version in a subprocess worker.

    Call BEFORE any ML imports. Resolves LoRA adapters to their base model,
    determines the required tier, prepends the appropriate ``.venv_t5_*`` dir to
    ``sys.path``, and propagates it via ``PYTHONPATH`` for child processes
    (e.g. GGUF converter). Used by training, inference, and export workers.

    ``hf_token`` is forwarded to tier detection so a gated/private model whose only 5.x
    signal is an authenticated config/tokenizer reaches the right sidecar, not the default.
    """
    # Pre-resolve LoRA adapters (local dir or remote adapter repo); full checkpoints
    # go to get_transformers_tier so their local config.json drives the tier (a full
    # checkpoint with a private/offline _name_or_path must not resolve to an
    # unreachable HF id and skip its own config). Remote adapters activate for their
    # BASE model, matching latest_tier_active_for and the inference worker.
    if _is_lora_adapter_dir(Path(model_name)):
        resolved = _resolve_base_model(model_name)
    else:
        resolved = _remote_lora_base(model_name, hf_token = hf_token) or model_name
    tier = get_transformers_tier(resolved, hf_token)
    if model_name != resolved and _safe_is_file(Path(model_name) / "config.json"):
        # Gate on a real local config.json: a checkpoint carries config the base may not
        # surface, but path names alone must not upgrade a plain adapter.
        tier = _higher_tier(tier, get_transformers_tier(model_name, hf_token))

    if tier == "latest":
        pinned = latest_venv_pinned_version()
        if pinned is None or not _ensure_venv_t5_latest_exists():
            raise RuntimeError(
                f"Cannot activate the latest-transformers sidecar: "
                f".venv_t5_latest missing or unpinned at {_VENV_T5_LATEST_DIR}"
            )
        if _VENV_T5_LATEST_DIR not in sys.path:
            sys.path.insert(0, _VENV_T5_LATEST_DIR)
        logger.info(
            "Prepended transformers %s venv to sys.path from %s "
            "(path only; the loaded version is confirmed later by "
            "'Subprocess loaded transformers ...' on first import)",
            pinned,
            _VENV_T5_LATEST_DIR,
        )
        _pp = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = _VENV_T5_LATEST_DIR + (os.pathsep + _pp if _pp else "")
    elif tier == "510":
        if not _ensure_venv_t5_510_exists():
            raise RuntimeError(
                f"Cannot activate transformers {TRANSFORMERS_510_VERSION}: "
                f".venv_t5_510 missing at {_VENV_T5_510_DIR}"
            )
        if _VENV_T5_510_DIR not in sys.path:
            sys.path.insert(0, _VENV_T5_510_DIR)
        logger.info(
            "Prepended transformers %s venv to sys.path from %s "
            "(path only; the loaded version is confirmed later by "
            "'Subprocess loaded transformers ...' on first import)",
            TRANSFORMERS_510_VERSION,
            _VENV_T5_510_DIR,
        )
        _pp = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = _VENV_T5_510_DIR + (os.pathsep + _pp if _pp else "")
    elif tier == "550":
        if not _ensure_venv_t5_550_exists():
            raise RuntimeError(
                f"Cannot activate transformers {TRANSFORMERS_550_VERSION}: "
                f".venv_t5_550 missing at {_VENV_T5_550_DIR}"
            )
        if _VENV_T5_550_DIR not in sys.path:
            sys.path.insert(0, _VENV_T5_550_DIR)
        logger.info(
            "Prepended transformers %s venv to sys.path from %s "
            "(path only; the loaded version is confirmed later by "
            "'Subprocess loaded transformers ...' on first import)",
            TRANSFORMERS_550_VERSION,
            _VENV_T5_550_DIR,
        )
        _pp = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = _VENV_T5_550_DIR + (os.pathsep + _pp if _pp else "")
    elif tier == "530":
        if not _ensure_venv_t5_530_exists():
            raise RuntimeError(
                f"Cannot activate transformers 5.3.0: "
                f".venv_t5_530 missing at {_VENV_T5_530_DIR}"
            )
        if _VENV_T5_530_DIR not in sys.path:
            sys.path.insert(0, _VENV_T5_530_DIR)
        logger.info(
            "Prepended transformers %s venv to sys.path from %s "
            "(path only; the loaded version is confirmed later by "
            "'Subprocess loaded transformers ...' on first import)",
            TRANSFORMERS_530_VERSION,
            _VENV_T5_530_DIR,
        )
        _pp = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = _VENV_T5_530_DIR + (os.pathsep + _pp if _pp else "")
    else:
        logger.info("Using default transformers (4.57.x) for %s", model_name)


def latest_tier_active_for(model_name: str, hf_token: str | None = None) -> bool:
    """True when *model_name* routes to the consented latest-transformers sidecar.

    Mirrors the inference worker's pre-activation resolution (local adapter dir,
    then a remote adapter's Hub adapter_config.json). ``latest`` only wins when
    the sidecar exists with a valid pin, i.e. exactly the loads that will import
    the newest release. Never raises: any resolution failure returns False so
    callers treat the model as a known tier.
    """
    try:
        # No consented sidecar pin means nothing routes to latest; return before
        # any resolution so the common case costs no config or network reads.
        if latest_venv_pinned_version() is None:
            return False
        if _is_lora_adapter_dir(Path(model_name)):
            resolved = _resolve_base_model(model_name)
        else:
            # A remote LoRA activates the sidecar for its BASE model; sizing and the
            # worker's 4-bit guard must see that base too, not the adapter repo.
            resolved = _remote_lora_base(model_name, hf_token = hf_token) or model_name
        tier = get_transformers_tier(resolved, hf_token)
        if model_name != resolved and _safe_is_file(Path(model_name) / "config.json"):
            tier = _higher_tier(tier, get_transformers_tier(model_name, hf_token))
        return tier == "latest"
    except Exception:
        return False


def _has_adapter_weights(path: Path) -> bool:
    """True if *path* holds LoRA adapter weight files (``adapter_model.*``)."""
    try:
        return any(path.glob("adapter_model*.safetensors")) or any(path.glob("adapter_model*.bin"))
    except OSError:
        return False


def _is_lora_adapter_dir(path: Path) -> bool:
    """True if *path* is a local LoRA dir (adapter_config.json or adapter_model-only
    weights). Import-light so it can run during subprocess activation."""
    try:
        if not path.is_dir():
            return False
        return (path / "adapter_config.json").is_file() or _has_adapter_weights(path)
    except OSError:
        return False


def _is_same_path(value: str, local_path: Path) -> bool:
    """True if *value* resolves to *local_path* (relative/absolute/symlink)."""
    if value == str(local_path):
        return True
    try:
        return os.path.realpath(value) == os.path.realpath(str(local_path))
    except OSError:
        return False


def _resolve_base_model(model_name: str) -> str:
    """If *model_name* points to a LoRA adapter, return its base model.

    Checks ``adapter_config.json`` locally first. Only calls the heavier
    ``get_base_model_from_lora`` for real local directories (avoids noisy
    warnings for plain HF model IDs). Returns *model_name* unchanged if not a
    LoRA adapter.
    """
    # --- Fast local check ---------------------------------------------------
    local_path = Path(model_name)
    adapter_cfg_path = local_path / "adapter_config.json"
    if _safe_is_file(adapter_cfg_path):
        try:
            with open(adapter_cfg_path) as f:
                cfg = json.load(f)
            base = cfg.get("base_model_name_or_path")
            if base:
                logger.info(
                    "Resolved LoRA adapter '%s' → base model '%s'",
                    model_name,
                    base,
                )
                return base
        except Exception as exc:
            logger.debug("Could not read %s: %s", adapter_cfg_path, exc)

    # --- config.json fallback (works for both LoRA and full fine-tune) ------
    config_json_path = local_path / "config.json"
    if _safe_is_file(config_json_path):
        try:
            with open(config_json_path) as f:
                cfg = json.load(f)
            # Unsloth writes model_name, HF writes _name_or_path; skip a self-reference.
            for _key in ("model_name", "_name_or_path"):
                base = cfg.get(_key)
                if isinstance(base, str) and base and not _is_same_path(base, local_path):
                    logger.info(
                        "Resolved checkpoint '%s' → base model '%s' (via config.json)",
                        model_name,
                        base,
                    )
                    return base
        except Exception as exc:
            logger.debug("Could not read %s: %s", config_json_path, exc)

    # Gate the heavy resolver on adapter_config.json: importing utils.models pulls
    # in transformers, which would pin the default into sys.modules before the
    # sidecar venv is prepended during activation.
    if _safe_is_file(adapter_cfg_path):
        try:
            from utils.models import get_base_model_from_lora
            base = get_base_model_from_lora(model_name)
            if base:
                logger.info(
                    "Resolved LoRA adapter '%s' → base model '%s' "
                    "(via get_base_model_from_lora)",
                    model_name,
                    base,
                )
                return base
        except Exception as exc:
            logger.debug(
                "get_base_model_from_lora failed for '%s': %s",
                model_name,
                exc,
            )

    # adapter_model-only LoRA: no config to resolve from, so use the
    # unsloth_<model>_<timestamp> dir-name convention (pure string parse).
    if local_path.name.startswith("unsloth_") and _has_adapter_weights(local_path):
        parts = local_path.name.split("_")
        if len(parts) >= 2:  # unsloth_<model...>_<timestamp>
            base = "unsloth/" + "_".join(parts[1:-1])
            logger.info(
                "Resolved adapter-only LoRA '%s' → base model '%s' (via directory name)",
                model_name,
                base,
            )
            return base

    return model_name


def _token_cache_key(model_name: str, hf_token: str | None) -> tuple[str, str | None]:
    """Cache key that keeps authenticated and unauthenticated reads separate, so an
    unauthenticated miss on a gated/private repo never poisons a later authed lookup."""
    import hashlib

    tok = hashlib.sha256(hf_token.encode()).hexdigest()[:16] if hf_token else None
    return (model_name, tok)


def _is_canonical_repo_id(model_name: str) -> bool:
    """True for a canonical ``owner/repo`` Hub id (not a local or relative path)."""
    return bool(
        model_name
        and model_name.count("/") == 1
        and model_name[0] not in "/.~"
        and "\\" not in model_name
    )


def _adapter_base_from_hf_cache(model_name: str) -> str | None:
    """``base_model_name_or_path`` from a remote adapter's cached ``adapter_config.json``.

    Stdlib path resolution of the HF hub cache (no ``huggingface_hub`` import); the newest
    snapshot wins. Lets an offline cached LoRA still resolve its base.
    """
    if not _is_canonical_repo_id(model_name):
        return None
    hub = (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.path.join(
            os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub"
        )
    )
    repo_dir = Path(hub) / ("models--" + model_name.replace("/", "--"))
    candidates = []
    ref_main = repo_dir / "refs" / "main"

    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    try:
        if ref_main.is_file():
            candidates.append(
                repo_dir / "snapshots" / ref_main.read_text().strip() / "adapter_config.json"
            )
        candidates += sorted(
            repo_dir.glob("snapshots/*/adapter_config.json"), key = _mtime, reverse = True
        )
        for cfg_path in candidates:
            if cfg_path.is_file():
                base = json.loads(cfg_path.read_text()).get("base_model_name_or_path")
                return base or None
    except Exception as exc:
        logger.debug("HF cache adapter_config.json lookup failed for '%s': %s", model_name, exc)
    return None


def _redirect_drops_auth(old_url: str, new_url: str) -> bool:
    """True when a redirect from *old_url* to *new_url* must not carry ``Authorization``.

    Same rule as ``requests`` (``SessionRedirectMixin.should_strip_auth``), curl and browsers:
    a different host drops the header, a same-host http-to-https upgrade keeps it (that token
    was already on the wire), any other scheme/port change or an unparseable URL drops it.
    """
    from urllib.parse import urlsplit

    try:
        old, new = urlsplit(old_url), urlsplit(new_url)
        if (old.hostname or "").lower() != (new.hostname or "").lower():
            return True
        defaults = {"http": 80, "https": 443}
        old_scheme, new_scheme = old.scheme.lower(), new.scheme.lower()
        old_port = old.port or defaults.get(old_scheme)
        new_port = new.port or defaults.get(new_scheme)
    except ValueError:
        return True
    if old_scheme == "http" and old_port == 80 and new_scheme == "https" and new_port == 443:
        return False
    return (old_scheme, old_port) != (new_scheme, new_port)


_hub_opener_cache = None


def _hub_opener():
    """Opener used by every Hub fetch here that can carry ``Authorization``.

    Stock ``HTTPRedirectHandler`` copies every header onto the redirect target without
    comparing hosts, so a 3xx replays ``Authorization: Bearer <hf_token>`` to whatever host
    ``Location`` names, and ``HF_ENDPOINT`` is user-configurable. Dropping the header outright
    is not an option either: the Hub 307s ``/gpt2/...`` to ``/openai-community/gpt2/...`` and
    301s http to https, both same-origin, and a renamed private repo answers 401
    unauthenticated (cached here as a definitive "absent"). So keep it same-origin, strip it
    cross-origin.

    Built once and never installed, so the rest of the process keeps stock ``urlopen``;
    ``build_opener`` retains the default proxy/HTTPS handlers, so ``*_PROXY`` / ``NO_PROXY``
    still work. Racing builders are harmless (equivalent, stateless). ``urllib.request`` stays
    a lazy import, hence the nested handler.
    """
    global _hub_opener_cache
    if _hub_opener_cache is not None:
        return _hub_opener_cache

    import urllib.request

    class _AuthStrippingRedirectHandler(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            new = super().redirect_request(req, fp, code, msg, headers, newurl)
            if new is not None and _redirect_drops_auth(req.full_url, new.full_url):
                for key in [k for k in new.headers if k.lower() == "authorization"]:
                    del new.headers[key]
            return new

    _hub_opener_cache = urllib.request.build_opener(_AuthStrippingRedirectHandler)
    return _hub_opener_cache


def _hub_urlopen(req, timeout = 10):
    """``urlopen`` for the Hub fetches here, via the redirect-aware opener above.

    ``hf_endpoint_unreachable`` stays on plain ``urlopen``: it sends no credentials.
    """
    return _hub_opener().open(req, timeout = timeout)


def _remote_lora_base(model_name: str, hf_token: str | None = None) -> str | None:
    """``base_model_name_or_path`` from a remote adapter's ``adapter_config.json``, or None.

    Raw HTTP (no huggingface_hub / transformers import), so a remote LoRA's base is known
    before any ML import. Offline (or on a transient failure) it reads the local hub cache,
    since a cached adapter is still loadable; a definitive 404 returns None (the repo is not
    a LoRA) rather than a stale cached base. Skipped for local/non-canonical ids.
    """
    if not _is_canonical_repo_id(model_name):
        return None
    try:
        from utils.paths import is_local_path
        if is_local_path(model_name):
            return None  # an existing relative path is a local checkpoint, not a Hub repo
    except Exception:
        pass
    if _env_offline():
        return _adapter_base_from_hf_cache(model_name)

    import urllib.error
    import urllib.request

    endpoint = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    url = f"{endpoint}/{model_name}/raw/main/adapter_config.json"
    headers = {"User-Agent": "unsloth-studio"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    try:
        req = urllib.request.Request(url, headers = headers)
        with _hub_urlopen(req, timeout = 10) as resp:
            cfg = json.loads(resp.read().decode())
        base = cfg.get("base_model_name_or_path")
        if base:
            logger.info("Resolved remote LoRA adapter '%s' → base model '%s'", model_name, base)
        return base or None
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None  # definitively not a LoRA; do not serve a stale cached base
        logger.debug("adapter_config.json fetch failed for '%s': %s", model_name, exc)
        return _adapter_base_from_hf_cache(model_name)
    except Exception as exc:
        logger.debug("No remote adapter_config.json for '%s': %s", model_name, exc)
        return _adapter_base_from_hf_cache(model_name)


def _hf_endpoint_base() -> str:
    """Hub base URL honoring ``HF_ENDPOINT`` (mirrors ``_remote_lora_base``)."""
    return (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")


def _hf_raw_file_url(model_name: str, filename: str) -> str:
    """Raw Hub file URL honoring ``HF_ENDPOINT`` (mirrors ``_remote_lora_base``)."""
    return f"{_hf_endpoint_base()}/{model_name}/raw/main/{filename}"


def _hf_api_url(path: str) -> str:
    """Hub API URL honoring ``HF_ENDPOINT``."""
    return f"{_hf_endpoint_base()}{path}"


def _is_same_hub_origin(url: str) -> bool:
    """True when *url* has the same scheme and authority as the configured Hub endpoint.

    Pagination cursors are echoed by the endpoint and followed with the caller's
    ``Authorization: Bearer <hf_token>``, so a cross-origin ``rel="next"`` would hand the Hub
    token to an arbitrary host (and make the backend an SSRF probe). The real Hub answers with
    an absolute same-origin cursor, so requiring one costs nothing.

    Compared as the RFC 6454 ``(scheme, host, effective port)`` tuple, not as raw ``netloc``
    text, so an endpoint ``https://mirror.internal`` echoing ``https://mirror.internal:443``
    still matches instead of turning a multi-page listing inconclusive; normalizing the default
    port only accepts authorities that already denote the configured origin, and an explicit
    differing port is still cross-origin. Userinfo is rejected outright: ``hostname`` ignores
    it, so ``https://user@host`` would compare equal to ``https://host`` while urllib connects
    to the whole ``user@host`` string.
    """
    from urllib.parse import urlsplit

    base = urlsplit(_hf_endpoint_base())
    nxt = urlsplit(url)
    nxt_scheme = nxt.scheme.lower()
    if nxt_scheme not in ("http", "https") or "@" in nxt.netloc:
        return False
    defaults = {"http": 80, "https": 443}
    base_scheme = base.scheme.lower()
    try:
        base_port = base.port or defaults.get(base_scheme)
        nxt_port = nxt.port or defaults.get(nxt_scheme)
    except ValueError:
        return False  # out-of-range or non-numeric port
    return (nxt_scheme, (nxt.hostname or "").lower(), nxt_port) == (
        base_scheme,
        (base.hostname or "").lower(),
        base_port,
    )


def _iter_link_entries(header: str):
    """Yield ``(uri, [(lowercased param name, value), ...])`` per RFC 8288 ``Link`` entry.

    Written out rather than split on ``;`` because a parameter value is a quoted string that
    may contain ``;`` or ``,``, and ``rel`` may sit anywhere in the list. Lenient about the
    ``<>`` around the URI: an unread entry stops pagination and reports a truncated listing as
    complete, which is worse than accepting an odd cursor (still rejected by
    :func:`_is_same_hub_origin`).
    """
    i, n = 0, len(header)
    while i < n:
        while i < n and header[i] in ", \t":
            i += 1
        if i >= n:
            return
        if header[i] == "<":
            end = header.find(">", i)
            if end == -1:
                return  # unterminated URI: the rest of the header cannot be split reliably
            uri = header[i + 1 : end].strip()
            i = end + 1
        else:
            start = i
            while i < n and header[i] not in ";,":
                i += 1
            uri = header[start:i].strip()
        params: list[tuple[str, str]] = []
        while i < n and header[i] != ",":
            while i < n and header[i] in " \t":
                i += 1
            if i >= n or header[i] == ",":
                break
            if header[i] != ";":
                return  # junk between parameters: stop rather than mis-attribute a rel
            i += 1
            start = i
            while i < n and header[i] not in "=;,":
                i += 1
            name = header[start:i].strip().lower()
            value = ""
            if i < n and header[i] == "=":
                i += 1
                while i < n and header[i] in " \t":
                    i += 1
                if i < n and header[i] == '"':
                    i += 1
                    buf: list[str] = []
                    while i < n and header[i] != '"':
                        if header[i] == "\\" and i + 1 < n:
                            i += 1
                        buf.append(header[i])
                        i += 1
                    value = "".join(buf)
                    i += 1  # closing quote
                    while i < n and header[i] not in ";,":
                        i += 1  # trailing junk after a quoted value
                else:
                    start = i
                    while i < n and header[i] not in ";,":
                        i += 1
                    value = header[start:i].strip()
            if name:
                params.append((name, value))
        if uri:
            yield uri, params


def _parse_link_next(link_header: "str | list[str] | None") -> str | None:
    """Next-page URL from RFC 8288 ``Link`` header field(s), or None.

    ``rel`` may sit at any position, be unquoted, and carry a case-insensitive space-separated
    list. Matching only a leading ``rel="next"`` would drop the valid
    ``<...>; type="application/json"; rel="next"``, and that is not safe:
    :func:`_hf_api_get_json` reads "no next cursor" as "listing finished" and returns the
    truncated page with ``success=True``, caching a later-page 5.x-only import as default tier.

    A list is accepted because RFC 7230 lets a server repeat ``Link`` as separate field lines,
    and ``HTTPMessage.get`` returns only the first: a cursor in a later line hits that same
    truncated-page-cached-as-default failure.
    """
    fields = [link_header] if isinstance(link_header, str) else list(link_header or ())
    for field in fields:
        for uri, params in _iter_link_entries(field or ""):
            for name, value in params:
                if name != "rel":
                    continue
                if "next" in value.lower().split():
                    return uri
                break  # RFC 8288: only an entry's first rel parameter is significant
    return None


# A repo tree needs one request per 1000 entries; stop well past any real repo rather
# than follow a cursor loop forever.
_HF_API_MAX_PAGES = 200


def _hf_api_get_json(path: str, hf_token: str | None = None) -> tuple[object | None, bool]:
    """GET a Hub API path via stdlib urllib, following ``Link`` pagination.

    The tree endpoint caps each response (1000 entries) and advertises the next cursor as
    ``Link: rel="next"``, the way ``huggingface_hub``'s ``list_repo_tree`` walks it, so list
    payloads are concatenated across pages instead of silently truncating a large repo.
    Returns ``(parsed_json, success)``; success is False if any page fails, so a partial
    listing is never mistaken for a complete one.
    """
    if _env_offline():
        return None, False

    import urllib.request

    headers = {"User-Agent": "unsloth-studio"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    url = _hf_api_url(path)
    merged: list = []
    try:
        for _ in range(_HF_API_MAX_PAGES):
            req = urllib.request.Request(url, headers = headers)
            with _hub_urlopen(req, timeout = 10) as resp:
                payload = json.loads(resp.read().decode())
                next_url = _parse_link_next(resp.headers.get_all("Link"))
            if not isinstance(payload, list):
                return payload, True
            merged.extend(payload)
            if not next_url:
                return merged, True
            # The next request replays the Authorization header, so only follow a cursor back
            # to the same origin; success=False keeps the truncated tree from passing as whole.
            if not _is_same_hub_origin(next_url):
                logger.debug("Refusing cross-origin Hub pagination link for %s: %s", path, next_url)
                return None, False
            url = next_url
        logger.debug("Hub API pagination exceeded %s pages for %s", _HF_API_MAX_PAGES, path)
        return None, False
    except Exception as exc:
        logger.debug("HF API request failed for %s: %s", path, exc)
        return None, False


def _list_hub_repo_py_files(repo_id: str, hf_token: str | None = None) -> tuple[set[str], bool]:
    """List ``.py`` paths in a Hub repo via the REST API (no huggingface_hub import)."""
    data, ok = _hf_api_get_json(f"/api/models/{repo_id}/tree/main?recursive=1", hf_token)
    if not ok or not isinstance(data, list):
        return set(), False
    return {
        item["path"]
        for item in data
        if isinstance(item, dict)
        and item.get("type") == "file"
        and str(item.get("path", "")).endswith(".py")
    }, True


# A scan downloads every ``.py`` the ``auto_map`` closure reaches, and workers run it BEFORE
# the remote-code consent gate, so unbounded a repo could stall or OOM a worker before Studio
# can reject the model. Survey of the 300 most-downloaded ``trust_remote_code`` repos: largest
# is 26 ``.py`` (p50 3, p95 11, p99 21), 933 KiB aggregate, 216 KiB single file, so these caps
# sit ~5x / 17x / 19x above the worst real repo and truncate none of them.
_REMOTE_SCAN_MAX_FILES = 128
_REMOTE_SCAN_MAX_TOTAL_CHARS = 16 * 1024 * 1024
_REMOTE_SCAN_MAX_FILE_BYTES = 4 * 1024 * 1024


class _RemoteScanBudget:
    """Cap on the remote ``.py`` downloads of ONE scan, shared across every repo it reaches.

    Spanning own repo plus every external ``auto_map`` repo keeps the ceiling on the whole
    activation, so N referenced repos cannot multiply it. Running out is reported as an
    incomplete closure, never as "no 5.x import", so a truncated scan is not memoized as a
    confirmed default tier. Sources read before the cap are kept: the scan already trusts a
    positive from a partial closure.
    """

    __slots__ = ("files_left", "chars_left", "truncated")

    def __init__(self) -> None:
        self.files_left = _REMOTE_SCAN_MAX_FILES
        self.chars_left = _REMOTE_SCAN_MAX_TOTAL_CHARS
        self.truncated = False

    def take_file(self) -> bool:
        """Reserve one download slot. False once the file or aggregate budget is spent."""
        if self.files_left <= 0 or self.chars_left <= 0:
            self.truncated = True
            return False
        self.files_left -= 1
        return True

    def spend(self, size: int) -> None:
        self.chars_left -= size


def _fetch_hub_py_sources(
    repo_id: str,
    hf_token: str | None = None,
    budget: "_RemoteScanBudget | None" = None,
) -> tuple[list[str], bool]:
    """Fetch every present ``.py`` in a Hub repo. Returns (sources, definitive).

    A file that fails to fetch (or a spent *budget*) only makes the closure incomplete; the
    sources already read are still returned, since discarding them would throw away an
    already-observed 5.x-only import and route the worker to a sidecar that cannot import it.
    """
    py_files, listing_ok = _list_hub_repo_py_files(repo_id, hf_token)
    if not listing_ok:
        return [], False
    sources: list[str] = []
    complete = True
    for fn in sorted(py_files):
        if budget is not None and not budget.take_file():
            logger.debug(
                "Remote scan budget spent at '%s'; %d .py left unread",
                repo_id,
                len(py_files),
            )
            complete = False
            break
        text = _read_repo_text_file(repo_id, fn, hf_token)
        if text is None:
            complete = False
            continue
        if budget is not None:
            budget.spend(len(text))
        sources.append(text)
    return sources, complete


def _snapshot_py_sources(snapshot: Path, budget: "_RemoteScanBudget | None" = None) -> list[str]:
    """Every readable ``.py`` under a hub-cache snapshot dir, charged to *budget*.

    Stands in for a repo the Hub API would not list: the snapshot is the exact code an
    offline ``trust_remote_code`` load executes. Each read draws on the caller's scan-wide
    budget, so substituting snapshots for N repos cannot lift one scan's ceiling.
    """
    sources: list[str] = []
    try:
        paths = sorted(snapshot.rglob("*.py"))
    except Exception as exc:
        logger.debug("Could not walk hub cache snapshot %s: %s", snapshot, exc)
        return sources
    for py_path in paths:
        if not _safe_is_file(py_path):
            continue
        if budget is not None and not budget.take_file():
            logger.debug("Remote scan budget spent walking hub cache snapshot %s", snapshot)
            break
        try:
            text = py_path.read_text(encoding = "utf-8", errors = "replace")
        except Exception as exc:
            logger.debug("Could not read %s: %s", py_path, exc)
            continue
        if budget is not None:
            budget.spend(len(text))
        sources.append(text)
    return sources


def _collect_external_py_sources(
    refs: set,
    hf_token: str | None = None,
    budget: "_RemoteScanBudget | None" = None,
) -> tuple[list[str], bool]:
    """Fetch all ``.py`` from external repos referenced by ``auto_map``.

    Like :func:`_fetch_hub_py_sources`, an unreachable repo marks the closure incomplete
    without dropping what the repos that did answer returned. *budget* is the caller's
    scan-wide cap, threaded through so every referenced repo draws on the same ceiling.

    A repo the Hub would not answer for falls back to its own hub-cache snapshot, the same
    substitution :func:`_remote_auto_map_tier` makes for the primary model id. Without it an
    offline model whose ``auto_map`` names a separate repo is scanned with that repo's code
    missing, so a 5.x-only import living only there is never seen and the worker activates
    4.57.x for code it cannot import. ``complete`` stays False whenever the Hub was not read,
    snapshot or not: an on-disk copy is not proof the Hub repo still matches it, so the
    closure never becomes definitive and no negative is memoized.
    """
    external_repos = {repo for repo, _ in refs if repo is not None}
    if not external_repos:
        return [], True
    sources: list[str] = []
    complete = True
    for repo in sorted(external_repos):
        repo_sources, ok = _fetch_hub_py_sources(repo, hf_token, budget)
        sources.extend(repo_sources)
        if not ok:
            complete = False
            snapshot = _hf_cache_snapshot_dir(repo)
            if snapshot is not None:
                sources.extend(_snapshot_py_sources(snapshot, budget))
    return sources, complete


def _auto_map_config_files(model_name: str) -> tuple[str, ...]:
    """Configs to read for ``auto_map``, the same set for a local dir and a Hub id.

    A processor declared only in ``preprocessor_config.json`` / ``processor_config.json`` /
    ``video_preprocessor_config.json`` (e.g. TencentARC/TimeLens-7B, whose config.json is a
    stock ``qwen2_5_vl``) is still executed by ``AutoProcessor.from_pretrained(...,
    trust_remote_code=True)``, so reading only ``config.json`` for a Hub id would report a
    definitive negative for remote code that really runs. The extra reads land only on the
    probe=True activation path, once per model, and only when the repo is scanned.
    """
    from utils.security.remote_code_scan import REMOTE_CODE_CONFIG_FILES
    return REMOTE_CODE_CONFIG_FILES


def _repo_auto_map_refs(model_name: str, hf_token: str | None = None) -> tuple[set, bool, bool]:
    """``(refs, has_auto_map, definitive)`` from the repo's remote-code configs.

    Each config is read once. ``definitive`` is False when a config could not be read at all
    (transient Hub failure, offline Hub id), so "declares no auto_map" is never concluded from
    a config nobody managed to look at.
    """
    from utils.security.remote_code_scan import _auto_map_refs

    refs: set = set()
    has_auto_map = False
    definitive = True
    for cfg_name in _auto_map_config_files(model_name):
        cfg, cfg_definitive = _load_repo_json_checked(model_name, cfg_name, hf_token)
        if not cfg_definitive:
            definitive = False
        if isinstance(cfg, dict):
            found = _auto_map_refs(cfg)
            if found:
                has_auto_map = True
                refs |= found
    return refs, has_auto_map, definitive


def _decode_source_bytes(raw: bytes) -> str:
    """Decode Python source bytes the way CPython does.

    PEP 263 (a BOM or a ``# -*- coding: <enc> -*-`` cookie in the first two lines), via
    ``tokenize.detect_encoding``, so a legitimate non-UTF-8 module does not fail to decode and
    turn a real 5.x-only import into unreadable source. Undecodable bytes fall back to
    replacement, matching how local checkpoint files are read.
    """
    import io
    import tokenize

    encoding = "utf-8"
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
    except Exception:
        pass
    try:
        return raw.decode(encoding)
    except Exception:
        return raw.decode(encoding, errors = "replace")


def _read_repo_text_file(
    model_name: str,
    filename: str,
    hf_token: str | None = None,
) -> str | None:
    """Return a repo-relative text file's contents; local first, else HuggingFace raw fetch.

    The remote read stops at ``_REMOTE_SCAN_MAX_FILE_BYTES``: :class:`_RemoteScanBudget` can
    only be charged after a file is read, so bounding the read itself is what keeps one
    enormous ``.py`` from exhausting a worker. Over-cap is reported as an unread file
    (``None``), which the caller counts as an incomplete closure.
    """
    local_path = Path(model_name) / filename
    if _safe_is_file(local_path):
        try:
            return local_path.read_text(encoding = "utf-8")
        except Exception as exc:
            logger.debug("Could not read %s: %s", local_path, exc)
            return None
    if _safe_is_dir(Path(model_name)):
        return None
    if _env_offline() or not _looks_like_hf_id(model_name):
        return None

    import urllib.request

    url = _hf_raw_file_url(model_name, filename)
    headers = {"User-Agent": "unsloth-studio"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    try:
        req = urllib.request.Request(url, headers = headers)
        with _hub_urlopen(req, timeout = 10) as resp:
            raw = resp.read(_REMOTE_SCAN_MAX_FILE_BYTES + 1)
        if len(raw) > _REMOTE_SCAN_MAX_FILE_BYTES:
            logger.debug(
                "Skipping oversized remote source %s for '%s' (over %d bytes)",
                filename,
                model_name,
                _REMOTE_SCAN_MAX_FILE_BYTES,
            )
            return None
        return _decode_source_bytes(raw)
    except Exception as exc:
        logger.debug("Could not fetch %s for '%s': %s", filename, model_name, exc)
        return None


def _load_repo_json_checked(
    model_name: str,
    filename: str,
    hf_token: str | None = None,
) -> tuple[dict | None, bool]:
    """``(parsed_json, definitive)`` for a repo-relative JSON file; local first, else raw Hub.

    ``definitive`` is False only when the answer is genuinely unknown (transient remote
    failure, offline Hub id); a local read, a 401/403/404 and a parse error are all definitive,
    so only a read that never happened blocks caching.

    A local checkpoint's ``config.json`` is read straight from disk, NOT through
    ``_load_config_json``: that process-lifetime cache would serve pre-rewrite contents to the
    rescan a changed scan signature just forced (a staged checkpoint writing code first and
    ``auto_map`` afterwards), memoizing a fresh negative under the new signature.
    """
    if filename == "config.json" and not _safe_is_dir(Path(model_name)):
        cfg = _load_config_json(model_name, hf_token)
        if cfg is not None or not _looks_like_hf_id(model_name):
            return cfg, True
        return None, _config_json_answer_is_definitive(model_name, hf_token)
    local_path = Path(model_name) / filename
    if _safe_is_file(local_path):
        try:
            with open(local_path) as f:
                return json.load(f), True
        except Exception as exc:
            logger.debug("Could not read %s: %s", local_path, exc)
            return None, True
    if _safe_is_dir(Path(model_name)) or not _looks_like_hf_id(model_name):
        return None, True
    if _env_offline():
        return None, False

    import urllib.error
    import urllib.request

    url = _hf_raw_file_url(model_name, filename)
    headers = {"User-Agent": "unsloth-studio"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    try:
        req = urllib.request.Request(url, headers = headers)
        with _hub_urlopen(req, timeout = 10) as resp:
            return json.loads(resp.read().decode()), True
    except urllib.error.HTTPError as exc:
        # 404 = genuinely absent; 401/403 = definitively unreadable with this token.
        # Anything else (5xx, rate limit) is transient and must stay inconclusive.
        logger.debug("Could not fetch %s for '%s': %s", filename, model_name, exc)
        return None, exc.code in (401, 403, 404)
    except Exception as exc:
        logger.debug("Could not fetch %s for '%s': %s", filename, model_name, exc)
        return None, False


def _remote_auto_map_py_contents(
    model_name: str,
    hf_token: str | None = None,
    known_refs: set | None = None,
) -> tuple[list[str], bool]:
    """Executable remote-code sources transformers may load (own, helper, external).

    Stdlib-only (urllib + HF REST API) so tier detection never imports ``huggingface_hub``
    before a sidecar is activated. ``known_refs`` are ``auto_map`` refs the caller already
    parsed, which both proves remote code exists and saves re-reading that config. Returns
    ``(sources, definitive)``; when ``definitive`` is False negatives must not be cached.
    """
    own_refs, has_auto_map, cfg_definitive = _repo_auto_map_refs(model_name, hf_token)
    if not known_refs and not has_auto_map:
        # An unread config proves nothing: inconclusive, not "declares no auto_map", so the
        # negative is not cached until a real read.
        return [], cfg_definitive

    ext_refs: set = set(known_refs or ()) | own_refs
    # One ceiling for the whole closure, so N referenced repos cannot multiply it.
    budget = _RemoteScanBudget()

    local_root = Path(model_name)
    if _safe_is_dir(local_root):
        sources: list[str] = []
        for py_path in local_root.rglob("*.py"):
            if py_path.is_file():
                try:
                    sources.append(py_path.read_text(encoding = "utf-8", errors = "replace"))
                except Exception as exc:
                    logger.debug("Could not read %s: %s", py_path, exc)
                    return [], False
        ext_sources, ext_ok = _collect_external_py_sources(ext_refs, hf_token, budget)
        sources.extend(ext_sources)
        if any(repo is not None for repo, _ in ext_refs) and not ext_ok:
            return sources, False
        return sources, cfg_definitive

    if _env_offline() or not _looks_like_hf_id(model_name):
        return [], False

    # An incomplete own-repo listing does not end the scan: the external repos may still hold
    # the marker, and whatever was read stays available to prove the tier.
    sources, repo_ok = _fetch_hub_py_sources(model_name, hf_token, budget)
    ext_sources, ext_ok = _collect_external_py_sources(ext_refs, hf_token, budget)
    sources.extend(ext_sources)
    if not repo_ok or (any(repo is not None for repo, _ in ext_refs) and not ext_ok):
        return sources, False
    return sources, cfg_definitive


def _parsed_dotted_imports(src: str) -> set[str] | None:
    """Absolute module paths *src* imports, resolved with ``ast``; ``None`` if it does not parse.

    Catches what a per-line substring scan cannot: parenthesized multi-line imports and the
    ``from <parent> import <submodule>`` spelling. Relative imports are skipped (a repo's own
    same-named helper must not match). ``None`` (unparseable) and an empty set (parsed, imports
    nothing) differ, so the caller only falls back to the line scan for unreadable source.

    ``if TYPE_CHECKING:`` bodies are skipped: that guard is false at runtime, so a type-only
    import must not promote an otherwise 4.57.x-loadable model onto a sidecar. Only the guard's
    own body is dropped; its ``else``/``elif`` branches and ``if not TYPE_CHECKING:`` do run
    and are collected. The name must resolve to ``typing``/``typing_extensions``'
    ``TYPE_CHECKING`` (aliases included) or to a literal ``False``: a module is free to bind
    that spelling to its own truthy value, and dropping a branch that really executes would
    hide a 5.x-only import behind the default tier.

    ``importlib.import_module("pkg.mod")`` / ``__import__("pkg.mod")`` count, since that call
    really imports at runtime and would raise on the wrong sidecar. Only literal string args
    are read, so a marker in a docstring, comment or other inert literal cannot promote the
    tier, and computed or f-string names are left alone.

    Qualified access counts too: ``import transformers`` then ``transformers.X`` is the usual
    spelling of a public-export marker. Aliases (including an aliased ``import_module``)
    resolve through the file's own bindings, so a never-imported name contributes nothing.
    """
    import ast

    _TYPING_GUARD_MODULES = ("typing", "typing_extensions")

    def _type_checking_bindings(tree) -> tuple[set[str], set[str]]:
        """``(names meaning typing's TYPE_CHECKING, names bound to a typing module)``.

        A pre-pass, because the walk below pops off a stack seeded with the module and so
        reaches an ``if TYPE_CHECKING:`` before the import that binds the name. ``deferred``
        cannot help: pruning decides whether the children are walked at all.

        A name qualifies only when the file binds it to ``typing``/``typing_extensions``'
        ``TYPE_CHECKING`` (aliases included) or to a literal ``False``, the spelling that keeps
        the guard without importing ``typing`` at runtime. Any other binding of that name is a
        shadow whose branch really does execute, so it is subtracted. Erring toward keeping a
        branch only costs a needless sidecar; dropping a live one hides a 5.x-only import.
        """
        direct: set[str] = set()
        modules: set[str] = set()
        shadowed: set[str] = set()
        # Identities, so the sweep below can tell a dead ``X = False`` target apart from the
        # same name reappearing as an ordinary store.
        dead_targets: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                if node.value.value is False:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            direct.add(target.id)
                            dead_targets.add(id(target))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if alias.asname:
                        # ``import typing.io as tio`` binds the submodule, not ``typing``.
                        if alias.name in _TYPING_GUARD_MODULES:
                            modules.add(alias.asname)
                    elif root in _TYPING_GUARD_MODULES:
                        modules.add(root)
            elif isinstance(node, ast.ImportFrom):
                # A relative ``from .flags import TYPE_CHECKING`` is the repo's own name.
                if node.level or node.module not in _TYPING_GUARD_MODULES:
                    continue
                for alias in node.names:
                    if alias.name == "TYPE_CHECKING":
                        direct.add(alias.asname or alias.name)
                    elif alias.name == "*":
                        # TYPE_CHECKING is in typing.__all__, so a star import binds it.
                        direct.add("TYPE_CHECKING")
            elif isinstance(node, ast.arg):
                shadowed.add(node.arg)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                shadowed.add(node.name)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                shadowed.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                # Covers assignment, for/with targets, walrus and comprehensions.
                if id(node) not in dead_targets:
                    shadowed.add(node.id)
        return direct - shadowed, modules - shadowed

    def _is_type_checking(test) -> bool:
        if isinstance(test, ast.Name):
            return test.id in type_checking_names
        if isinstance(test, ast.Attribute):
            return (
                test.attr == "TYPE_CHECKING"
                and isinstance(test.value, ast.Name)
                and test.value.id in type_checking_modules
            )
        return False

    def _dynamic_import_call(node) -> tuple[str, str] | None:
        """``(callee, module)`` for a constant-string import call, else ``None``."""
        func = node.func
        if isinstance(func, ast.Attribute):
            called = func.attr
        elif isinstance(func, ast.Name):
            called = func.id
        else:
            return None
        if not node.args:
            return None
        arg = node.args[0]
        if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
            return None
        # A leading dot is relative, skipped for the same reason as ImportFrom.
        return None if arg.value.startswith(".") else (called, arg.value)

    def _attribute_chain(node) -> tuple[str, str] | None:
        """``(root name, dotted attribute path)`` for ``a.b.c``, else ``None``."""
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        return node.id, ".".join(reversed(parts))

    try:
        tree = ast.parse(src)
    except Exception:
        return None
    type_checking_names, type_checking_modules = _type_checking_bindings(tree)
    names: set[str] = set()
    bound: dict[str, str] = {}
    chains: set[tuple[str, str]] = set()
    deferred: list[tuple[str, str]] = []
    stack: list = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
                root = alias.name.split(".", 1)[0]
                bound[alias.asname or root] = alias.name if alias.asname else root
            continue
        if isinstance(node, ast.ImportFrom):
            if not node.level and node.module:
                names.add(node.module)
                for alias in node.names:
                    names.add(f"{node.module}.{alias.name}")
                    bound[alias.asname or alias.name] = f"{node.module}.{alias.name}"
            continue
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            stack.extend(node.orelse)
            continue
        if isinstance(node, ast.Attribute):
            chain = _attribute_chain(node)
            if chain is not None:
                chains.add(chain)
        elif isinstance(node, ast.Call):
            call = _dynamic_import_call(node)
            if call is not None:
                called, module = call
                if called in ("import_module", "__import__"):
                    names.add(module)
                elif isinstance(node.func, ast.Name):
                    deferred.append((called, module))
            # Not terminal: keep walking so a nested call or import still counts.
        stack.extend(ast.iter_child_nodes(node))
    # Resolved after the walk: a binding can sit below the use that needs it.
    for root, attr in chains:
        module = bound.get(root)
        if module:
            names.add(f"{module}.{attr}")
    for called, module in deferred:
        if bound.get(called) == "importlib.import_module":
            names.add(module)
    return names


def _imported_dotted_names(src: str) -> set[str]:
    """Absolute module paths *src* imports; empty when the source does not parse."""
    return _parsed_dotted_imports(src) or set()


def _remote_auto_map_py_matches(markers: tuple[str, ...], sources: list[str]) -> bool:
    """True when a source *imports* one of *markers*.

    Import statements only, resolved by ``ast``, so a marker in a docstring, comment or other
    string literal cannot promote the tier. Source ``ast`` cannot parse falls back to the raw
    line scan (the only place a substring match is trusted), so unparseable remote code never
    becomes a silent negative that crashes on the default tier.
    """
    if not markers:
        return False
    for src in sources:
        names = _parsed_dotted_imports(src)
        if names is None:
            for line in src.splitlines():
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")) and any(
                    m in stripped for m in markers
                ):
                    return True
            continue
        for name in names:
            if any(name == m or name.startswith(m + ".") for m in markers):
                return True
    return False


def _local_scan_signature(model_name: str) -> str | None:
    """Signature of a local checkpoint's scan inputs (path + size + mtime + ctime + inode),
    or ``""`` for a Hub id. ``None`` when the walk fails, which means "do not cache".

    Covers the ``.py`` files AND every config that can declare an ``auto_map``, since both
    decide the answer: a checkpoint materialized in stages can land its code first and gain the
    ``auto_map`` later, and signing only the ``.py`` would keep serving the earlier "no remote
    code" negative. Folded into the scan cache keys (mirrors ``_probe_cache_key``) so a
    replaced checkpoint is re-scanned instead of reusing a stale answer.

    Size and mtime alone miss a same-sized replacement that restores the timestamp, i.e. an
    archival redeploy (``rsync --archive``, ``tar -xp``), leaving a long-lived worker serving a
    tier computed from code no longer on disk. ``st_ctime_ns`` moves on any write to an
    existing inode and ``st_ino`` moves on replace-by-rename, so together they cover both
    spellings for free (same ``stat`` call); a digest would work but would re-read every ``.py``
    on this hot path. On Windows ``st_ctime_ns`` is a creation time, so only the ``st_ino`` half
    applies, still more than before. A spurious miss costs one local re-scan, so this errs
    toward re-reading.
    """
    root = Path(model_name)
    if not _safe_is_dir(root):
        return ""
    try:
        paths = sorted(root.rglob("*.py"))
        paths += [root / name for name in _auto_map_config_files(model_name)]
        parts = []
        for path in paths:
            try:
                st = path.stat()
            except OSError:
                # Absent contributes nothing, and starts contributing the moment it appears.
                continue
            parts.append(f"{path}:{st.st_size}:{st.st_mtime_ns}:{st.st_ctime_ns}:{st.st_ino}")
        return "\0".join(parts)
    except Exception as exc:
        logger.debug("Could not signature local scan inputs under %s: %s", model_name, exc)
        return None


def _hf_cache_snapshot_dir(model_name: str) -> Path | None:
    """Newest local HF hub cache snapshot directory for a Hub id, or None.

    Stdlib-only path resolution mirroring :func:`_config_json_from_hf_cache`, so tier detection
    never loads the default-env ``huggingface_hub`` before a sidecar venv is activated.
    """
    # Only a canonical ``owner/repo`` Hub id maps to a cache dir; reject local paths.
    if not model_name or model_name.count("/") != 1 or model_name[0] in "/.~" or "\\" in model_name:
        return None
    hub = (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.path.join(
            os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub"
        )
    )
    repo_dir = Path(hub) / ("models--" + model_name.replace("/", "--"))
    candidates: list[Path] = []
    ref_main = repo_dir / "refs" / "main"
    try:
        if ref_main.is_file():
            candidates.append(repo_dir / "snapshots" / ref_main.read_text().strip())
        # No refs/main (commit-pinned downloads): newest snapshot by mtime, not a stale
        # lexicographically-first SHA, matching what the Hub cache would load.
        candidates += sorted(repo_dir.glob("snapshots/*"), key = _safe_mtime, reverse = True)
        for path in candidates:
            if path.is_dir():
                return path
    except Exception as exc:
        logger.debug("HF cache snapshot lookup failed for '%s': %s", model_name, exc)
    return None


def _remote_auto_map_tier(model_name: str, hf_token: str | None = None) -> tuple[str, bool]:
    """Return ``(tier, scan_was_definitive)`` for auto_map remote code.

    ``"510"`` when the closure imports a symbol no sidecar below 5.10 has, ``"530"`` when it
    imports a module absent from 4.57.x but present in 5.3, else ``"default"``. Classified here
    and not only in the tokenizer check because remote code reached through ``config.json`` or
    a processor config is never scanned by that check unless ``tokenizer_config.json`` itself
    declares the module, leaving the worker on 4.57.x for code that needs 5.x.
    """
    signature = _local_scan_signature(model_name)
    cache_key = (*_token_cache_key(model_name, hf_token), signature)
    if signature is not None and cache_key in _remote_auto_map_tier_cache:
        return _remote_auto_map_tier_cache[cache_key], True

    # Offline Hub id: a downloaded snapshot is exactly the code an offline load would execute,
    # so scan it rather than assume no 5.x-only import (which would activate 4.57.x and crash
    # on a repo whose files are all local). Still non-definitive, since external auto_map repos
    # are unreachable offline, and nothing is cached under the Hub id, so a later online read
    # re-fetches (mirrors _check_tokenizer_config_needs_v5).
    if _env_offline() and not _safe_is_dir(Path(model_name)):
        snapshot = _hf_cache_snapshot_dir(model_name)
        if snapshot is None:
            return "default", False
        tier, _ = _remote_auto_map_tier(str(snapshot), hf_token)
        return tier, False

    sources, definitive = _remote_auto_map_py_contents(model_name, hf_token)
    if _remote_auto_map_py_matches(_TRANSFORMERS_510_REMOTE_IMPORT_MARKERS, sources):
        tier = "510"
        logger.info("Remote auto_map check: %s needs transformers 5.10.x", model_name)
    elif _remote_auto_map_py_matches(_TRANSFORMERS_5_REMOTE_IMPORT_MARKERS, sources):
        tier = "530"
        logger.info("Remote auto_map check: %s requires transformers 5.x", model_name)
    else:
        tier = "default"
    if not definitive and tier == "default" and not _safe_is_dir(Path(model_name)):
        # An inconclusive scan is not a negative: the caller only sees the tier, so a bare
        # "default" here is indistinguishable from a repo that really has no 5.x import and
        # would activate 4.57.x for code needing 5.x. A hub-cache snapshot is the same code an
        # offline load would execute, so read it rather than guess -- the _env_offline fallback
        # above, now also for a transient failure while nominally online. Only ever raises the
        # tier, and stays non-definitive so a recovered Hub re-scans. Deliberately not a blanket
        # 5.3 floor: an unread config is inconclusive for every model, so flooring would push
        # plain non-remote-code repos onto the 5.3 sidecar for the length of any Hub outage.
        snapshot = _hf_cache_snapshot_dir(model_name)
        if snapshot is not None:
            tier = _higher_tier(tier, _remote_auto_map_tier(str(snapshot), hf_token)[0])
    if definitive and signature is not None:
        _remote_auto_map_tier_cache[cache_key] = tier
    return tier, definitive


def _remote_auto_map_scan_result(model_name: str, hf_token: str | None = None) -> tuple[bool, bool]:
    """Return ``(needs_510, scan_was_definitive)`` for auto_map remote code."""
    tier, definitive = _remote_auto_map_tier(model_name, hf_token)
    return tier == "510", definitive


def _check_remote_auto_map_needs_510(model_name: str, hf_token: str | None = None) -> bool:
    """True when auto_map remote code imports transformers>=5.10-only symbols.

    While ``_TRANSFORMERS_510_REMOTE_IMPORT_MARKERS`` is empty the answer is False for every
    repo, so the scan is skipped rather than run: it would read each remote-code config, page
    the repo tree and fetch every ``.py`` to reach an answer it already has, and that cost lands
    on the name fast path where common activations cannot move the tier anyway.

    Gated on the tuple rather than at the call site, so re-adding a 5.10-only module restores
    the scan everywhere by itself. The 5.3 classification is not lost: callers that need it go
    through :func:`_remote_auto_map_tier`, which :func:`_check_config_needs_510` also calls
    for the ``definitive`` flag and the scanned tier.
    """
    if not _TRANSFORMERS_510_REMOTE_IMPORT_MARKERS:
        return False
    needs, _ = _remote_auto_map_scan_result(model_name, hf_token)
    return needs


def _tokenizer_auto_map_needs_v5(
    data: dict, model_name: str, hf_token: str | None
) -> tuple[bool, bool]:
    """``(needs_v5, definitive)`` for the tokenizer ``auto_map`` closure.

    ``definitive`` is False when the closure could not be read in full (missing local file,
    unreachable external repo, incomplete Hub listing), so the caller must not cache the
    negative and keep routing to the default sidecar once the Hub or checkpoint recovers.
    """
    from utils.security.remote_code_scan import _auto_map_refs

    refs = _auto_map_refs(data)
    if not refs:
        return False, True

    local_root = Path(model_name)
    if _safe_is_dir(local_root):
        sources: list[str] = []
        for py_path in local_root.rglob("*.py"):
            if py_path.is_file():
                try:
                    sources.append(py_path.read_text(encoding = "utf-8", errors = "replace"))
                except Exception as exc:
                    logger.debug("Could not read %s: %s", py_path, exc)
                    return False, False
        # An unwritten own-repo entry file (in-progress checkpoint) means the closure is
        # incomplete, not that it is 4.x-safe.
        definitive = all(repo is not None or _safe_is_file(local_root / fn) for repo, fn in refs)
        ext_sources, ext_ok = _collect_external_py_sources(refs, hf_token, _RemoteScanBudget())
        if any(repo is not None for repo, _ in refs) and not ext_ok:
            # An unreachable external repo leaves the closure incomplete (negative uncached),
            # but the local sources already read can still PROVE 5.x, so keep them and let the
            # marker scan below decide instead of returning early and routing a checkpoint
            # whose own tokenizer imports the 5.x-only backend to the default sidecar.
            definitive = False
        sources.extend(ext_sources)
    else:
        # The tokenizer refs already prove remote code exists, so a repo whose only auto_map
        # is the tokenizer's is still scanned.
        sources, definitive = _remote_auto_map_py_contents(model_name, hf_token, known_refs = refs)
        if not definitive and not sources:
            return False, False

    if _remote_auto_map_py_matches(_TRANSFORMERS_5_REMOTE_IMPORT_MARKERS, sources):
        logger.info("Remote tokenizer auto_map check: %s requires transformers 5.x", model_name)
        return True, True
    return False, definitive


def _check_tokenizer_config_needs_v5(
    model_name: str,
    hf_token: str | None = None,
    scan_auto_map: bool = True,
) -> bool:
    """True if the model's tokenizer_class requires transformers 5.x.

    Checks local tokenizer_config.json, else fetches from HuggingFace (authenticated with
    ``hf_token`` so gated/private repos resolve). Cached by (model, token) so an unauthenticated
    miss cannot poison a later authed read, plus the local scan signature so a replaced
    checkpoint is re-read. Returns False on any network/parse error (fail-open to default).

    ``scan_auto_map=False`` answers from the tokenizer class alone and drops the ``auto_map``
    closure scan, mirroring :func:`_check_config_needs_510`. The skip happens before the cache
    write, so a probe=False negative never poisons a later activation.
    """
    signature = _local_scan_signature(model_name)
    cache_key = (*_token_cache_key(model_name, hf_token), signature)
    if signature is not None and cache_key in _tokenizer_class_cache:
        return _tokenizer_class_cache[cache_key]

    # --- Check local tokenizer_config.json first ---------------------------
    local_path = Path(model_name)
    local_tc = local_path / "tokenizer_config.json"
    if _safe_is_file(local_tc):
        try:
            with open(local_tc) as f:
                data = json.load(f)
            tokenizer_class = data.get("tokenizer_class", "")
            result = tokenizer_class in _TRANSFORMERS_5_TOKENIZER_CLASSES
            definitive = True
            if not result:
                if not scan_auto_map:
                    return False
                result, definitive = _tokenizer_auto_map_needs_v5(data, model_name, hf_token)
            if result:
                logger.info(
                    "Local check: %s uses tokenizer_class=%s (requires transformers 5.x)",
                    model_name,
                    tokenizer_class,
                )
            if definitive:
                _tokenizer_class_cache[cache_key] = result
            return result
        except Exception as exc:
            logger.debug("Could not read %s: %s", local_tc, exc)

    # Local checkpoint without the file yet: don't fetch it as a Hub id or cache the miss,
    # so a file written later this process (in-progress checkpoint) is read next call.
    if _safe_is_dir(local_path):
        return False

    # Offline: skip the 10s urllib fetch (fail-open to lower tier). Do NOT cache this
    # assumed negative, so a later online read of the same id re-fetches the real value.
    if _env_offline():
        return False

    # --- Fall back to fetching from HuggingFace ----------------------------
    import urllib.request

    # Endpoint-aware: a mirror-only deployment (HF_ENDPOINT set, huggingface.co blocked) must
    # still read this, or the auto_map scan it gates is never reached.
    url = _hf_raw_file_url(model_name, "tokenizer_config.json")
    headers = {"User-Agent": "unsloth-studio"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    try:
        req = urllib.request.Request(url, headers = headers)
        with _hub_urlopen(req, timeout = 10) as resp:
            data = json.loads(resp.read().decode())
        tokenizer_class = data.get("tokenizer_class", "")
        result = tokenizer_class in _TRANSFORMERS_5_TOKENIZER_CLASSES
        definitive = True
        if not result:
            if not scan_auto_map:
                return False
            result, definitive = _tokenizer_auto_map_needs_v5(data, model_name, hf_token)
        if result:
            logger.info(
                "Dynamic check: %s uses tokenizer_class=%s (requires transformers 5.x)",
                model_name,
                tokenizer_class,
            )
        if definitive and signature is not None:
            _tokenizer_class_cache[cache_key] = result
        return result
    except Exception as exc:
        logger.debug("Could not fetch tokenizer_config.json for '%s': %s", model_name, exc)
        if signature is not None:
            _tokenizer_class_cache[cache_key] = False
        return False


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _config_json_from_hf_cache(model_name: str) -> dict | None:
    """Parsed ``config.json`` from the local HF hub cache, or None.

    Stdlib-only path resolution (no ``huggingface_hub`` import) so tier detection never
    loads the default-env hub before a sidecar venv is activated.
    """
    # Only a canonical ``owner/repo`` Hub id maps to a cache dir; reject local paths.
    if not model_name or model_name.count("/") != 1 or model_name[0] in "/.~" or "\\" in model_name:
        return None
    hub = (
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.path.join(
            os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub"
        )
    )
    repo_dir = Path(hub) / ("models--" + model_name.replace("/", "--"))
    candidates = []
    ref_main = repo_dir / "refs" / "main"
    try:
        if ref_main.is_file():
            candidates.append(repo_dir / "snapshots" / ref_main.read_text().strip() / "config.json")
        # No refs/main (e.g. commit-pinned downloads): newest snapshot by mtime, not a stale
        # lexicographically-first SHA, matching what the Hub cache would actually load.
        candidates += sorted(
            repo_dir.glob("snapshots/*/config.json"), key = _safe_mtime, reverse = True
        )
        for cfg_path in candidates:
            if cfg_path.is_file():
                with open(cfg_path) as f:
                    return json.load(f)
    except Exception as exc:
        logger.debug("HF cache config.json lookup failed for '%s': %s", model_name, exc)
    return None


def _load_config_json(model_name: str, hf_token: str | None = None) -> dict | None:
    """Return parsed ``config.json`` for *model_name*, checking local files first.

    ``hf_token`` authenticates the raw fetch so gated/private repos resolve. The
    cache is keyed on the token so an unauthenticated miss never poisons a later
    authenticated read. The HF hub cache is consulted only offline or after a failed
    network fetch, so an online read never serves stale metadata.
    """
    import hashlib

    tok = hashlib.sha256(hf_token.encode()).hexdigest()[:16] if hf_token else None
    cache_key = (model_name, tok)
    if cache_key in _config_json_cache:
        return _config_json_cache[cache_key]

    local_cfg = Path(model_name) / "config.json"
    if _safe_is_file(local_cfg):
        try:
            with open(local_cfg) as f:
                cfg = json.load(f)
            _config_json_cache[cache_key] = cfg
            return cfg
        except Exception as exc:
            logger.debug("Could not read %s: %s", local_cfg, exc)
            _config_json_cache[cache_key] = None
            return None

    # Local checkpoint without the file yet: don't fetch it as a Hub id or cache the miss,
    # so a file written later this process (in-progress checkpoint) is read next call.
    if _safe_is_dir(Path(model_name)):
        return None

    if _env_offline():
        # No network: a previously downloaded repo can still tier from the hub cache. Cache a
        # real hit, but never the miss (None) so a later online read still fetches the config.
        cfg = _config_json_from_hf_cache(model_name)
        if cfg is not None:
            _config_json_cache[cache_key] = cfg
        return cfg

    import urllib.error
    import urllib.request

    # Endpoint-aware: a mirror-only deployment (HF_ENDPOINT set, huggingface.co blocked) must
    # still read the config that gates the whole auto_map scan.
    url = _hf_raw_file_url(model_name, "config.json")
    headers = {"User-Agent": "unsloth-studio"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    try:
        req = urllib.request.Request(url, headers = headers)
        with _hub_urlopen(req, timeout = 10) as resp:
            cfg = json.loads(resp.read().decode())
        _config_json_cache[cache_key] = cfg
        return cfg
    except urllib.error.HTTPError as exc:
        # A definitive access answer: never serve cached private metadata to a wrong-token
        # or unauthenticated request.
        if exc.code in (401, 403, 404):
            logger.debug("config.json access denied for '%s': %s", model_name, exc)
            _config_json_absent.add(cache_key)
            return None
        logger.debug("Could not fetch config.json for '%s': %s", model_name, exc)
        return _config_json_from_hf_cache(model_name)
    except Exception as exc:
        logger.debug("Could not fetch config.json for '%s': %s", model_name, exc)
        # Transient: serve the hub cache uncached so the next call retries the network.
        return _config_json_from_hf_cache(model_name)


def _config_json_is_definitive(model_name: str, hf_token: str | None = None) -> bool:
    """True if the last ``_load_config_json`` read for this model+token was cached
    (definitive), not a transient fallback (not stored, so callers re-check next call)."""
    return _token_cache_key(model_name, hf_token) in _config_json_cache


def _config_json_answer_is_definitive(model_name: str, hf_token: str | None = None) -> bool:
    """True when ``_load_config_json`` established an answer for this model+token: it read a
    config (cached) or the Hub said 401/403/404. A transient failure stores neither, so "no
    config" stays inconclusive and no negative gets cached."""
    key = _token_cache_key(model_name, hf_token)
    return key in _config_json_cache or key in _config_json_absent


def _config_matches_tier(cfg: dict, architectures: set[str], model_types: set[str]) -> bool:
    # Defensive: a malformed config may carry non-string values (e.g. list model_type).
    archs = cfg.get("architectures")
    if isinstance(archs, (list, tuple)) and any(a in architectures for a in archs):
        return True
    mt = cfg.get("model_type")
    return isinstance(mt, str) and mt in model_types


def _config_needs_550(cfg: dict) -> bool:
    return _config_matches_tier(
        cfg,
        _TRANSFORMERS_550_ARCHITECTURES,
        _TRANSFORMERS_550_MODEL_TYPES,
    )


_NESTED_CONFIG_KEYS = ("llm_config", "text_config", "language_config", "thinker_config")


def _nemotron_h_needs_mlp_support(cfg: dict) -> bool:
    """True for a dense NemotronH config using MLP (``-``) layers.

    transformers only gained ``-`` -> ``mlp`` in 5.10; 5.3/5.5 raise ``KeyError: '-'``.
    Read from ``hybrid_override_pattern`` or ``layers_block_type``, recursing into nested
    language configs (VL wrappers hold the dense LM under ``llm_config``/``text_config``).
    """
    if not isinstance(cfg, dict):
        return False
    if cfg.get("model_type") == "nemotron_h":
        pattern = cfg.get("hybrid_override_pattern")
        if isinstance(pattern, str) and "-" in pattern:
            return True
        block_types = cfg.get("layers_block_type")
        if isinstance(block_types, (list, tuple)) and "mlp" in block_types:
            return True
    return any(_nemotron_h_needs_mlp_support(cfg.get(key)) for key in _NESTED_CONFIG_KEYS)


def _config_needs_510(cfg: dict) -> bool:
    if _config_matches_tier(
        cfg,
        _TRANSFORMERS_510_ARCHITECTURES,
        _TRANSFORMERS_510_MODEL_TYPES,
    ):
        return True
    return _nemotron_h_needs_mlp_support(cfg)


def _config_needs_530(cfg: dict) -> bool:
    return _config_matches_tier(
        cfg,
        _TRANSFORMERS_530_ARCHITECTURES,
        _TRANSFORMERS_530_MODEL_TYPES,
    )


def _check_config_needs_550(model_name: str, hf_token: str | None = None) -> bool:
    """True if ``config.json`` needs transformers 5.5.0 (e.g. Gemma 4). Local first, else
    fetched (authenticated with ``hf_token``); cached by (model, token) only for a definitive
    read so a transient miss retries. False on error.
    """
    cache_key = _token_cache_key(model_name, hf_token)
    if cache_key in _config_needs_550_cache:
        return _config_needs_550_cache[cache_key]

    cfg = _load_config_json(model_name, hf_token)
    result = bool(cfg) and _config_needs_550(cfg)
    if result:
        logger.info(
            "config.json check: %s needs transformers %s (architectures=%s, model_type=%s)",
            model_name,
            TRANSFORMERS_550_VERSION,
            cfg.get("architectures", []),
            cfg.get("model_type"),
        )
    if _config_json_is_definitive(model_name, hf_token):
        _config_needs_550_cache[cache_key] = result
    return result


def _check_config_needs_530(model_name: str, hf_token: str | None = None) -> bool:
    """True if ``config.json`` needs transformers 5.3.0 (Qwen3.5, Qwen3 MoE, GLM-4.7, LFM2.5-VL).
    Local first, else fetched (authenticated with ``hf_token``); cached by (model, token) only
    for a definitive read so a transient miss retries. False on error.
    """
    cache_key = _token_cache_key(model_name, hf_token)
    if cache_key in _config_needs_530_cache:
        return _config_needs_530_cache[cache_key]

    cfg = _load_config_json(model_name, hf_token)
    result = bool(cfg) and _config_needs_530(cfg)
    if result:
        logger.info(
            "config.json check: %s needs transformers %s (architectures=%s, model_type=%s)",
            model_name,
            TRANSFORMERS_530_VERSION,
            cfg.get("architectures", []),
            cfg.get("model_type"),
        )
    if _config_json_is_definitive(model_name, hf_token):
        _config_needs_530_cache[cache_key] = result
    return result


def _check_config_needs_510(
    model_name: str,
    hf_token: str | None = None,
    scan_auto_map: bool = True,
    scan_out: dict | None = None,
) -> bool:
    """Check ``config.json`` for Gemma 4 Unified / 12B architectures (authenticated with
    ``hf_token``; cached by (model, token) only for a definitive read).

    ``scan_auto_map=False`` drops the remote-code scan for the cheap parent-side check
    (``probe=False``), which only asks "is this 5.x at all" and never activates a sidecar.

    ``scan_out`` receives ``{"tier": ...}`` from the remote-code scan this call runs, for a
    caller resolving the full tier. The return value is a 5.10 boolean, so a ``"530"`` scan
    collapses to False here, and re-deriving the tier from a second scan is not equivalent:
    only a definitive scan is memoized, so precisely when the first scan was inconclusive does
    the second one refetch, and a 5.3 marker read on the first pass is lost if the ``.py``
    carrying it fails on the second.
    """
    cache_key = _token_cache_key(model_name, hf_token)
    if cache_key in _config_needs_510_cache:
        return _config_needs_510_cache[cache_key]

    cfg = _load_config_json(model_name, hf_token)
    config_definitive = _config_json_is_definitive(model_name, hf_token)
    if cfg and _config_needs_510(cfg):
        if config_definitive:
            _config_needs_510_cache[cache_key] = True
        logger.info(
            "config.json check: %s needs transformers %s (architectures=%s, model_type=%s)",
            model_name,
            TRANSFORMERS_510_VERSION,
            cfg.get("architectures", []),
            cfg.get("model_type"),
        )
        return True

    if not scan_auto_map:
        # The scan costs a read of every remote-code config, a paged tree listing and one raw
        # request per .py, too much for the cheap parent-side path (model inspection, load
        # logging) where a large repo or degraded Hub would stall it. Never cached: this
        # negative only says "no config match", and probe=True still scans.
        return False

    remote_tier, remote_definitive = _remote_auto_map_tier(model_name, hf_token)
    remote_needs = remote_tier == "510"
    if scan_out is not None:
        scan_out["tier"] = remote_tier
    if remote_needs:
        logger.info(
            "config.json check: %s needs transformers %s (auto_map remote code)",
            model_name,
            TRANSFORMERS_510_VERSION,
        )
    if config_definitive and remote_definitive:
        _config_needs_510_cache[cache_key] = remote_needs
    return remote_needs


def _config_saved_by_transformers_5(cfg: dict | None) -> bool:
    """True if ``config.json``'s ``transformers_version`` is >= 5. Only a cheap "worth
    probing" hint (the saving version, not the minimum to load); the default-first probe
    decides the actual tier."""
    if not isinstance(cfg, dict):
        return False
    ver = cfg.get("transformers_version")
    if not isinstance(ver, str):
        return False
    try:
        return int(ver.strip().split(".", 1)[0]) >= 5
    except ValueError:
        return False


def _cached_config_json(model_name: str, hf_token: str | None) -> dict | None:
    """Already-fetched config.json from the in-process cache (no new fetch); the tier checks
    above populate it, and a miss just skips the version-field probe."""
    return _config_json_cache.get(_token_cache_key(model_name, hf_token))


# --- Static tier from CONFIG_MAPPING_NAMES (AST only: no import/network/exec) ---
# A model_type absent from an overlay's mapping can't load there. Parse each sidecar's
# config map from source and pick the lowest tier that ships it, so a new arch routes
# correctly with no per-model table edit. Only ever upgrades default, never lowers.
_config_mapping_cache: dict[str, frozenset[str]] = {}


def _latest_tier_disabled() -> bool:
    """Kill switch shared with utils.transformers_latest: lets operators roll
    back a provisioned latest sidecar without deleting files."""
    return os.environ.get("UNSLOTH_STUDIO_NO_LATEST_TRANSFORMERS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# Failed lazy repairs back off so a broken sidecar can't turn every routing
# call into a pip install attempt.
_latest_repair_failed_at: float = 0.0
_LATEST_REPAIR_BACKOFF_SECS = 5 * 60


def _latest_sidecar_intact() -> bool:
    """The pinned latest sidecar exists with its transformers dir and every pinned
    package. False when the pin itself is gone: a cached 'latest' mapping must then be
    dropped (routing re-resolves to no latest tier), not trusted, and a sidecar that kept
    transformers/ but lost a pinned package must self-heal rather than route models to a
    latest tier that fails activation in workers, which refuse parent-only repairs.

    (_overlay_transformers_dir only calls this after gating on a present pin, so the
    pin-missing case here is the cache-revalidation caller whose pin was deleted after
    the mapping was first cached.)"""
    pin = _latest_pin_data()
    if pin is None:
        return False
    return _venv_dir_is_valid(_VENV_T5_LATEST_DIR, tuple(pin["packages"]))


def _overlay_transformers_dir(tier: str) -> str | None:
    """transformers source dir for a tier, located without importing it."""
    global _latest_repair_failed_at
    if tier != "default":
        # latest requires a valid pin and the kill switch off.
        if tier == "latest" and (_latest_tier_disabled() or latest_venv_pinned_version() is None):
            return None
        root = {
            "530": _VENV_T5_530_DIR,
            "550": _VENV_T5_550_DIR,
            "510": _VENV_T5_510_DIR,
            "latest": _VENV_T5_LATEST_DIR,
        }.get(tier)
        src = os.path.join(root, "transformers") if root else None
        if src and tier == "latest" and not _latest_sidecar_intact():
            # A valid pin whose sidecar vanished or lost a pinned package (partial
            # deletion, disk issue, interrupted external edits) must self-heal, or
            # latest-only models either silently route to older tiers or reach a
            # worker that cannot repair, failing every load until a manual
            # reinstall. Repair under the swap reservation; back off after a
            # failure so routing calls don't hammer pip.
            repaired = False
            if time.time() - _latest_repair_failed_at >= _LATEST_REPAIR_BACKOFF_SECS:
                if _ensure_venv_t5_latest_exists():
                    _latest_repair_failed_at = 0.0
                    repaired = True
                else:
                    _latest_repair_failed_at = time.time()
            if not repaired:
                # Still broken: treat the overlay as unavailable rather than route
                # models to a tier whose worker activation is known to fail. Models
                # an older tier supports keep loading there until a repair succeeds,
                # matching the behavior when the sidecar dir is missing entirely.
                return None
        return src if src and _safe_is_dir(Path(src)) else None
    # default: the base 4.x transformers. find_spec resolves to a 5.x sidecar if one
    # is already on sys.path, so skip any .venv_t5_* / llmcompressor overlay dir.
    sidecars = tuple(
        os.path.abspath(d) + os.sep
        for d in (
            _VENV_T5_530_DIR,
            _VENV_T5_550_DIR,
            _VENV_T5_510_DIR,
            _VENV_T5_LATEST_DIR,
            _VENV_LLMCOMPRESSOR_DIR,
        )
    )
    candidates = []
    try:
        spec = importlib.util.find_spec("transformers")
        if spec and spec.origin:
            candidates.append(os.path.dirname(spec.origin))
    except Exception:
        pass
    candidates += [os.path.join(e, "transformers") for e in sys.path if e]
    for c in candidates:
        if _safe_is_dir(Path(c)) and not os.path.abspath(c).startswith(sidecars):
            return c
    return None


def _mapping_first_keys(value: ast.AST) -> set[str]:
    """First keys of a dict literal, or of an OrderedDict(...)/dict(...)/.update(...)
    built from 2-tuple lists and **{...} unpacking."""

    def keys_of(node):
        if isinstance(node, ast.Dict):
            return list(node.keys)
        if isinstance(node, (ast.List, ast.Tuple)):
            return [
                el.elts[0] for el in node.elts if isinstance(el, (ast.Tuple, ast.List)) and el.elts
            ]
        return []

    nodes = keys_of(value)
    if isinstance(value, ast.Call):
        for a in value.args:
            nodes += keys_of(a)
        for kw in value.keywords:  # **{...} unpacking has kw.arg is None
            if kw.arg is None:
                nodes += keys_of(kw.value)
    return {n.value for n in nodes if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def _model_types_from_source(source: str) -> set[str]:
    """model_type keys of CONFIG_MAPPING_NAMES in *source* (AST only, no execution).

    Handles the direct ``CONFIG_MAPPING_NAMES = ...`` binding (dict literal or
    OrderedDict/dict call over 2-tuple lists and **{...} unpacking) and any
    ``CONFIG_MAPPING_NAMES.update({...})`` mutation. Shared by the on-disk overlay
    reader below and the remote latest-release checker (utils/transformers_latest.py).
    """
    keys: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "CONFIG_MAPPING_NAMES" for t in node.targets
        ):
            keys |= _mapping_first_keys(node.value)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            fn = node.value.func
            if (
                isinstance(fn, ast.Attribute)
                and fn.attr == "update"
                and isinstance(fn.value, ast.Name)
                and fn.value.id == "CONFIG_MAPPING_NAMES"
            ):
                keys |= _mapping_first_keys(node.value)
    return keys


def _config_model_types(tier: str) -> frozenset[str]:
    """model_type keys in a tier's CONFIG_MAPPING_NAMES (5.10 moved it to auto_mappings.py)."""
    # Kill switch beats the cache: a stale mapping must not keep routing latest-only models until restart.
    if tier == "latest" and _latest_tier_disabled():
        return frozenset()
    cached = _config_mapping_cache.get(tier)
    if cached is not None:
        # A cached 'latest' mapping can outlive the sidecar it was parsed from: if the
        # pinned sidecar was since deleted or lost a package in this process, drop the
        # cache so routing re-resolves through _overlay_transformers_dir (which self-heals)
        # instead of routing latest-only models to a broken tier until restart.
        if tier != "latest" or _latest_sidecar_intact():
            return cached
        _config_mapping_cache.pop("latest", None)
    tdir = _overlay_transformers_dir(tier)
    if tdir is None:
        return frozenset()  # overlay not provisioned yet; do not cache so a later call re-reads
    keys: set[str] = set()
    for rel in ("models/auto/configuration_auto.py", "models/auto/auto_mappings.py"):
        path = Path(tdir) / rel
        if not _safe_is_file(path):
            continue
        try:
            keys |= _model_types_from_source(path.read_text(encoding = "utf-8"))
        except Exception:
            continue
    result = frozenset(keys)
    _config_mapping_cache[tier] = result
    return result


def _model_types_from_config(cfg: dict) -> list[str]:
    """All model_types in the config: the primary (top-level, else first nested)
    first, then every other nested sub-config. Wrappers instantiate sub-configs
    through CONFIG_MAPPING, so nested types matter for routing too."""
    seen: list[str] = []

    def add(value):
        if isinstance(value, str) and value and value not in seen:
            seen.append(value)

    add(cfg.get("model_type"))
    for key in _NESTED_CONFIG_KEYS:
        sub = cfg.get(key)
        if isinstance(sub, dict):
            add(sub.get("model_type"))
    for value in cfg.values():
        if isinstance(value, dict):
            add(value.get("model_type"))
    return seen


def _lowest_tier_for(model_type: str) -> str | None:
    for tier in sorted(_TIER_RANK, key = _TIER_RANK.get):
        if model_type in _config_model_types(tier):
            return tier
    return None


def _tier_from_config_mapping(cfg: dict) -> str | None:
    """Lowest tier able to load every model_type in cfg, or None when the
    primary type is unknown everywhere. A nested type can raise the tier (its
    sub-config is built through CONFIG_MAPPING); an unknown nested type never
    vetoes, since no installed tier could load it either way (the latest
    checker handles surfacing the install prompt for it)."""
    types = _model_types_from_config(cfg)
    if not types:
        return None
    best = _lowest_tier_for(types[0])
    if best is None:
        return None
    for model_type in types[1:]:
        tier = _lowest_tier_for(model_type)
        if tier is not None and _TIER_RANK[tier] > _TIER_RANK[best]:
            best = tier
    return best


def _raise_tier_for_nested(cfg: dict | None, tier: str) -> str:
    """Raise *tier* when the mapping resolver needs a higher one for *cfg*.

    A wrapper's top-level model_type can match a hardcoded fast path while a
    nested text/vision config's type only exists in a newer sidecar (e.g. the
    installed latest); its sub-config is built through CONFIG_MAPPING, so the
    fast-path tier would fail to load it. Raise-only: never lowers a fast-path
    match, so name overrides (Qwen3.6) keep their tier. Never raises an
    exception: a resolution failure keeps the fast-path tier."""
    if not isinstance(cfg, dict):
        return tier
    try:
        mapped = _tier_from_config_mapping(cfg)
        if mapped is not None and _TIER_RANK.get(mapped, 0) > _TIER_RANK.get(tier, 0):
            return mapped
    except Exception:
        pass
    return tier


# --- AutoConfig probe: general tier resolution for ambiguous models ----------
# When the cheap signals only say "needs some 5.x", parse config.json with the built-in
# parser in each candidate sidecar (lowest first) instead of guessing. Generalizes beyond
# the hardcoded lists, e.g. dense NemotronH whose '-' (MLP) layer only 5.10 can parse.
_PROBE_TIER_ORDER = ("530", "550", "510")
_PROBE_TIMEOUT_SECS = 60

# config.json-only parse in a sidecar (--target dir on sys.path, no per-venv python).
# Built-in parser only, no repo code, no weights. Exit 0 = parses; token via env, not argv.
_PROBE_CONFIG_SCRIPT = r"""
import sys, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
target_dir, model_name = sys.argv[1], sys.argv[2]
if target_dir:  # empty = probe the ambient (default 4.57.x) transformers, no sidecar prepend
    sys.path.insert(0, target_dir)
try:
    from transformers import AutoConfig
    AutoConfig.from_pretrained(model_name, trust_remote_code=False)
    sys.exit(0)
except Exception as exc:
    # stderr encoding may not be UTF-8 (e.g. cp1252 on Windows); write bytes so a
    # non-ASCII error message cannot itself raise UnicodeEncodeError.
    sys.stderr.buffer.write((type(exc).__name__ + ": " + str(exc)).encode("utf-8", "replace"))
    sys.exit(1)
"""

# stderr fragments meaning "couldn't fetch/auth", NOT "needs a newer parser".
_PROBE_TRANSIENT_MARKERS = (
    "ConnectionError",
    "HTTPError",
    "Timeout",
    "Max retries",
    "Temporary failure",
    "GatedRepoError",
    "RepositoryNotFoundError",
    "LocalEntryNotFoundError",
    "OfflineModeIsEnabled",
    "401",
    "403",
    "404",
)


def _stderr_is_transient(err: str) -> bool:
    return any(marker in err for marker in _PROBE_TRANSIENT_MARKERS)


def _probe_tier_venvs():
    """tier -> (target_dir, ensure_fn), a function so the later _ensure_* defs resolve. The
    ``default`` entry (empty target_dir = ambient 4.57.x) is only probed with include_default."""
    return {
        "default": ("", lambda: True),
        "530": (_VENV_T5_530_DIR, _ensure_venv_t5_530_exists),
        "550": (_VENV_T5_550_DIR, _ensure_venv_t5_550_exists),
        "510": (_VENV_T5_510_DIR, _ensure_venv_t5_510_exists),
        "latest": (_VENV_T5_LATEST_DIR, _ensure_venv_t5_latest_exists),
    }


def _probe_tier_order() -> tuple[str, ...]:
    """Sidecar probe order. The consented "latest" sidecar joins only once it is
    provisioned (pin marker present): an absent optional tier must not flip the probe's
    skipped-tier bookkeeping, keeping pre-latest behavior byte-identical."""
    if not _latest_tier_disabled() and latest_venv_pinned_version() is not None:
        return _PROBE_TIER_ORDER + ("latest",)
    return _PROBE_TIER_ORDER


def _probe_autoconfig(target_dir: str, model_name: str, hf_token: str | None) -> bool | None:
    """Parse config.json with the built-in parser inside *target_dir*'s sidecar.
    True = parses, False = parse/version failure (escalate), None = transient
    (auth/network/offline/spawn) so the caller fails safe and does not cache.
    """
    env = child_env_without_native_path_secret()
    if hf_token:
        env["HF_TOKEN"] = hf_token
        # The probe relies on the implicit HF_TOKEN env (no token= arg). Clear any inherited
        # HF_HUB_DISABLE_IMPLICIT_TOKEN=1 so a gated repo authenticates instead of 401ing
        # into the 530 fail-safe.
        env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "0"
    if _env_offline():
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE_CONFIG_SCRIPT, target_dir, model_name],
            capture_output = True,
            text = True,
            errors = "replace",
            timeout = _PROBE_TIMEOUT_SECS,
            env = env,
            **_windows_hidden_subprocess_kwargs(),
        )
    except subprocess.TimeoutExpired:
        logger.warning("AutoConfig probe timed out for '%s' in %s", model_name, target_dir)
        return None
    except Exception as exc:
        logger.warning("AutoConfig probe could not spawn for '%s': %s", model_name, exc)
        return None
    if result.returncode == 0:
        return True
    err = (result.stderr or "").strip()
    if _stderr_is_transient(err):
        logger.warning("AutoConfig probe transient failure for '%s': %s", model_name, err)
        return None
    logger.info("AutoConfig probe parse failure for '%s' in %s: %s", model_name, target_dir, err)
    return False


def _probe_cache_key(model_name: str) -> str:
    """Cache key for the probe result. A local checkpoint can be overwritten in place, so
    fold in a cheap config.json signature (size + mtime) and re-probe when it changes.
    Remote ids key by name alone (resolving a Hub revision would need a pre-activation hub
    import that pins the wrong env)."""
    try:
        config_path = (Path(model_name) / "config.json").resolve()
        st = config_path.stat()
    except OSError:
        return model_name
    return f"{config_path}\0{st.st_size}:{st.st_mtime_ns}"


def _probe_tier(
    model_name: str,
    hf_token: str | None,
    reason: str,
    *,
    include_default: bool = False,
    floor: str = "530",
) -> str:
    """Lowest tier whose built-in parser loads the config; *floor* is the fail-safe.

    Escalates ``_PROBE_TIER_ORDER`` (prefixed with the ambient ``default`` tier when
    ``include_default``), returning the first that parses; never raises or escalates on
    uncertainty:
      - first success wins (cached unless a lower tier was skipped);
      - transient failure (auth/network/offline) -> *floor*, uncached;
      - a skipped/uninstallable sidecar -> uncached (a lower tier may yet be the answer);
      - all tiers probed, none parse -> remote-code/custom model_type; keep *floor*.

    Known-5.x callers use ``floor='530'``; weak-signal callers (config saved by transformers
    5.x) use ``include_default=True, floor='default'`` so a model that still parses on 4.57.x
    stays on the default. Cached per _probe_cache_key (process lifetime). No Hub sha is
    resolved: that would import huggingface_hub before the sidecar is on sys.path.
    """
    if os.environ.get("UNSLOTH_DISABLE_TIER_PROBE", "").lower() in ("1", "true", "yes", "on"):
        return floor
    key = _probe_cache_key(model_name)
    # Key by probe mode: the default-first path can return 'default', which must not be
    # reused for a tokenizer/known-5.x caller (floor='530'). Legacy 530 keeps the bare key.
    if include_default or floor != "530":
        key = f"{key}\0floor={floor}:def={int(include_default)}"
    if key in _probe_tier_cache:
        cached = _probe_tier_cache[key]
        # Kill switch beats the cache (like _config_model_types): a stale 'latest' probe must not keep activating it.
        if cached != "latest" or not _latest_tier_disabled():
            return cached

    def _cache(tier: str, *, skipped: bool) -> str:
        # Do not pin a result that depended on a skipped lower tier: once that sidecar is
        # available the lowest valid tier may differ, so re-probe next call.
        if not skipped:
            _probe_tier_cache[key] = tier
        return tier

    venvs = _probe_tier_venvs()
    sidecar_order = _probe_tier_order()
    order = (("default",) + sidecar_order) if include_default else sidecar_order
    probed_count = 0
    skipped_any = False
    for tier in order:
        target_dir, ensure_fn = venvs[tier]
        try:
            available = ensure_fn()
        except Exception:
            available = False
        if not available:
            skipped_any = True
            continue
        probed_count += 1
        ok = _probe_autoconfig(target_dir, model_name, hf_token)
        if ok is True:
            logger.info(
                "Transformers tier %s selected for %s (AutoConfig probe; %s)",
                tier,
                model_name,
                reason,
            )
            return _cache(tier, skipped = skipped_any)
        if ok is None:
            logger.info("Tier probe inconclusive for %s (%s); using %s", model_name, reason, floor)
            return floor  # transient: retry next load

    # Nothing parsed. Only treat it as conclusive (and cache) when every tier was actually
    # probed; a skipped sidecar means the environment is incomplete, so retry uncached.
    if skipped_any or probed_count == 0:
        logger.info(
            "Tier probe incomplete for %s (%s); using %s (uncached)", model_name, reason, floor
        )
        return floor
    logger.info(
        "Transformers tier %s selected for %s (AutoConfig probe found no higher tier; %s)",
        floor,
        model_name,
        reason,
    )
    return _cache(floor, skipped = False)


def _norm_separators(s: str) -> str:
    """Collapse ``_``/whitespace to ``-`` (underscore aliases) but keep ``.`` so a
    version dot (``qwen3.5``) isn't conflated with a size separator (``Qwen3-5B``)."""
    return "".join("-" if ch in "_ \t" else ch for ch in s)


def _looks_like_hf_id(value: str) -> bool:
    """True if *value* looks like a Hub id (``org/name``), not a local path. An
    existing path is treated as a path, mirroring transformers' own resolution."""
    if not value or not value.strip():
        return False
    if os.path.isabs(value) or value.startswith((".", "~")) or "\\" in value:
        return False
    if os.path.exists(value):
        return False
    return value.count("/") <= 1


def _tier_from_name(name: str) -> tuple[str, str] | None:
    """``(tier, reason)`` from name substrings (order 510 > 550 > 530), or ``None``.

    Underscore aliases match (``Qwen3_5`` == ``Qwen3.5``); a dot-version substring
    matches only the dot/underscore form, never a hyphen, so ``Qwen3-6B`` size names
    aren't promoted.
    """
    lowered = name.lower()
    norm = _norm_separators(lowered)
    dotted = lowered.replace("_", ".")
    if "assistant" in lowered and ("gemma-4" in norm or "gemma4" in norm):
        return "510", "gemma-4 assistant variant"
    for substrings, tier in (
        (TRANSFORMERS_510_MODEL_SUBSTRINGS, "510"),
        (TRANSFORMERS_550_MODEL_SUBSTRINGS, "550"),
        (TRANSFORMERS_5_MODEL_SUBSTRINGS, "530"),
    ):
        for s in substrings:
            if "." in s:
                if s in lowered or s in dotted:
                    return tier, s
            elif s in lowered or _norm_separators(s) in norm:
                return tier, s
    return None


def _higher_tier_name_override(name_hint: str | None) -> str | None:
    """510/550 tier if *name_hint* names a higher-tier model, else ``None``. Qwen3.6
    reuses Qwen3.5 config ids but needs the 5.5 sidecar, so a name hint overrides 530."""
    if not name_hint:
        return None
    hint = _tier_from_name(name_hint)
    return hint[0] if hint is not None and hint[0] in ("510", "550") else None


def get_transformers_tier(
    model_name: str,
    hf_token: str | None = None,
    probe: bool = True,
) -> str:
    """Return the transformers tier required for *model_name*.

    Returns ``"510"`` for models needing transformers 5.10.x (Gemma 4 Unified),
    ``"550"`` for models needing transformers 5.5.0 (Gemma 4),
    ``"530"`` for models needing transformers 5.3.0 (e.g. Ministral-3, Qwen3 MoE),
    or ``"default"`` for everything else (4.57.x).

    Strong signals (architecture/model_type, name substrings) are fast paths. For local paths,
    ``config.json`` is checked before name heuristics to avoid false-positives from directory
    name fragments. When the only signal is the 5.x tokenizer class, the exact tier is resolved
    by probing AutoConfig in each sidecar; a config saved by transformers 5.x with no fast-path
    match is probed default-first, catching a new 5.x-only arch while 4.57.x-loadable models
    stay on default.

    ``probe=False`` skips the sidecar subprocesses (used by the cheap
    :func:`needs_transformers_5`); it still classifies via cheap signals (a 5.x-saved config
    returns ``"530"``). ``probe=True`` (the activation path) resolves the exact tier.

    Higher 5.x tiers run first.
    """
    # Local path: trust config.json. If its arch matches a known sidecar, return;
    # else fall back to the HF id in the config (not the folder name) for renamed dirs.
    local_cfg = Path(model_name) / "config.json"
    if _safe_is_file(local_cfg):
        cfg = _load_config_json(model_name, hf_token)
        if cfg is not None:
            if _config_needs_510(cfg):
                tier = _raise_tier_for_nested(cfg, "510")
                logger.info(
                    "Transformers tier %s selected for %s (local config.json check)",
                    tier,
                    model_name,
                )
                return tier
            remote_tier, _ = _remote_auto_map_tier(model_name, hf_token)
            if remote_tier == "510":
                tier = _raise_tier_for_nested(cfg, "510")
                logger.info(
                    "Transformers tier %s selected for %s (local auto_map needs 5.10.x)",
                    tier,
                    model_name,
                )
                return tier
            if _config_needs_550(cfg):
                tier = _raise_tier_for_nested(cfg, "550")
                logger.info(
                    "Transformers tier %s selected for %s (local config.json check)",
                    tier,
                    model_name,
                )
                return tier
            if _config_needs_530(cfg):
                # Qwen3.6 reuses Qwen3.5 config ids but needs 5.5 by name. Only a real
                # Hub id (or the folder basename) may override 530, so a stale local
                # path in _name_or_path can't flip a correct 530 config to 550.
                base = _resolve_base_model(model_name)
                hint_src = (
                    base
                    if (base != model_name and _looks_like_hf_id(base))
                    else Path(model_name).name
                )
                override = _higher_tier_name_override(hint_src)
                if override is not None:
                    override = _raise_tier_for_nested(cfg, override)
                    logger.info(
                        "Transformers tier %s selected for %s (name overrides 530 config)",
                        override,
                        model_name,
                    )
                    return override
                tier = _raise_tier_for_nested(cfg, "530")
                logger.info(
                    "Transformers tier %s selected for %s (local config.json check)",
                    tier,
                    model_name,
                )
                return tier
            # Unknown arch: resolve the base id from config. A resolved local dir
            # recurses (config check); a Hub id uses name rules only (no network).
            resolved = _resolve_base_model(model_name)
            if resolved != model_name:
                if _safe_is_dir(Path(resolved)):
                    tier = get_transformers_tier(resolved, hf_token, probe = probe)
                    if tier != "default":
                        logger.info(
                            "Transformers tier %s selected for %s (resolved local path: %s)",
                            tier,
                            model_name,
                            resolved,
                        )
                        return tier
                elif _looks_like_hf_id(resolved):
                    result = _tier_from_name(resolved)
                    if result is not None:
                        tier, match = result
                        logger.info(
                            "Transformers tier %s selected for %s (resolved HF ID: %s, match: %s)",
                            tier,
                            model_name,
                            resolved,
                            match,
                        )
                        return tier
            static = _tier_from_config_mapping(cfg)
            if static is not None and static != "default":
                logger.info(
                    "Transformers tier %s selected for %s (config mapping: model_type absent below)",
                    static,
                    model_name,
                )
                return static
            local_tc = Path(model_name) / "tokenizer_config.json"
            if _safe_is_file(local_tc) and _check_tokenizer_config_needs_v5(model_name, hf_token):
                if not probe:
                    return "530"
                return _probe_tier(model_name, hf_token, "local tokenizer needs 5.x")
            # Remote code declared by config.json / a processor config that imports the
            # 5.x-only tokenizers backend: the check above only walks the closure
            # tokenizer_config.json declares. 510 already returned above.
            if remote_tier != "default":
                if not probe:
                    return "530"
                return _probe_tier(model_name, hf_token, "local auto_map needs 5.x")
            if _config_saved_by_transformers_5(cfg):
                if not probe:
                    return "530"  # cheap 5.x hint; the real path resolves the exact tier
                tier = _probe_tier(
                    model_name,
                    hf_token,
                    "local config saved by transformers 5.x",
                    include_default = True,
                    floor = "default",
                )
                if tier != "default":
                    return tier
            logger.info(
                "Transformers tier default (4.57.x) selected for %s (local config.json no match)",
                model_name,
            )
            return "default"

    # --- Fast substring checks (no I/O) ------------------------------------
    result = _tier_from_name(model_name)
    if result is not None:
        tier, match = result
        # With a consented latest sidecar pinned, a name that matches a fixed
        # tier can still carry a latest-only model_type (e.g. a newer variant
        # reusing a family name); consult the config so an accepted upgrade
        # actually routes to the sidecar it installed. Costs a config read only
        # in the pinned case, keeping the pre-latest path I/O-free.
        if latest_venv_pinned_version() is not None:
            tier = _raise_tier_for_nested(_load_config_json(model_name, hf_token), tier)
        # Activation path only: the name already resolved a 5.x tier, so the scan can only
        # pick 510 over 530/550, and probe=False must not pay a repo listing plus a fetch
        # per remote .py just to answer "is this 5.x at all".
        if probe and _check_remote_auto_map_needs_510(model_name, hf_token):
            tier = _higher_tier(tier, "510")
            tier = _raise_tier_for_nested(_load_config_json(model_name, hf_token), tier)
        logger.info(
            "Transformers tier %s selected for %s (substring match: %s)",
            tier,
            model_name,
            match,
        )
        return tier

    # --- Slow config fallbacks (network for HF IDs; authenticated with hf_token) --------
    # The remote-code scan this call runs already resolves a full tier; keep it so the 5.3
    # check below reuses what this resolution observed instead of scanning a second time.
    remote_scan: dict = {}
    if _check_config_needs_510(model_name, hf_token, scan_auto_map = probe, scan_out = remote_scan):
        tier = _raise_tier_for_nested(_load_config_json(model_name, hf_token), "510")
        logger.info("Transformers tier %s selected for %s (config.json check)", tier, model_name)
        return tier
    if _check_config_needs_550(model_name, hf_token):
        tier = _raise_tier_for_nested(_load_config_json(model_name, hf_token), "550")
        logger.info("Transformers tier %s selected for %s (config.json check)", tier, model_name)
        return tier
    if _check_config_needs_530(model_name, hf_token):
        # Qwen3.6 reuses Qwen3.5 config ids but needs 5.5 by name; honor a real Hub-id name
        # hint from _name_or_path before selecting 530.
        remote_cfg = _load_config_json(model_name, hf_token) or {}
        base = remote_cfg.get("_name_or_path") or remote_cfg.get("model_name")
        override = _higher_tier_name_override(
            base if isinstance(base, str) and base != model_name else None
        )
        if override is not None:
            override = _raise_tier_for_nested(remote_cfg, override)
            logger.info(
                "Transformers tier %s selected for %s (name overrides 530 config)",
                override,
                model_name,
            )
            return override
        tier = _raise_tier_for_nested(remote_cfg, "530")
        logger.info("Transformers tier %s selected for %s (config.json check)", tier, model_name)
        return tier
    # _load_config_json (not the cache-only reader) so a config served from the hub
    # cache during a transient outage still feeds the mapping resolver.
    remote_cfg = _load_config_json(model_name, hf_token)
    if remote_cfg is not None:
        static = _tier_from_config_mapping(remote_cfg)
        if static is not None and static != "default":
            logger.info(
                "Transformers tier %s selected for %s (config mapping: model_type absent below)",
                static,
                model_name,
            )
            return static
    # Same probe gate as the config fallback above: probe=False must not pay a repo listing
    # plus a fetch per remote .py, which this check reaches through the tokenizer auto_map
    # closure. The skipped result is never cached, so the activation path still resolves it.
    if _check_tokenizer_config_needs_v5(model_name, hf_token, scan_auto_map = probe):
        if not probe:
            return "530"
        return _probe_tier(model_name, hf_token, "tokenizer needs 5.x")

    # Same gap as the local branch, for a Hub id: remote code declared outside
    # tokenizer_config.json that imports the 5.x-only tokenizers backend. Activation path
    # only (see _check_config_needs_510's scan_auto_map note). 510 already returned above, so
    # this can only add the 5.3 floor. Reuse the tier that call's scan resolved: a definitive
    # scan is memoized and would be a cache hit anyway, while a non-definitive one is exactly
    # the scan that must not be repeated, since the .py carrying the marker can be the one
    # that fails on the second pass, dropping a 5.3 answer this activation already read.
    if probe:
        remote_tier = remote_scan.get("tier")
        if remote_tier is None:
            # No scan ran here: config.json answered on its own, or a memoized 5.10 answer
            # short-circuited the call, so ask for the tier directly.
            remote_tier = _remote_auto_map_tier(model_name, hf_token)[0]
        if remote_tier != "default":
            return _probe_tier(model_name, hf_token, "auto_map needs 5.x")

    if _config_saved_by_transformers_5(_cached_config_json(model_name, hf_token)):
        if not probe:
            return "530"  # cheap 5.x hint; the real path resolves the exact tier
        tier = _probe_tier(
            model_name,
            hf_token,
            "config saved by transformers 5.x",
            include_default = True,
            floor = "default",
        )
        if tier != "default":
            return tier

    logger.info("Transformers tier default (4.57.x) selected for %s (no match)", model_name)
    return "default"


def needs_transformers_5(model_name: str) -> bool:
    """Return True if *model_name* requires any transformers 5.x version.

    Convenience wrapper around :func:`get_transformers_tier`. Passes ``probe=False`` so a
    log-only parent caller never spawns sidecar probes (the worker re-resolves the exact
    tier with ``probe=True`` on the real activation path).
    """
    return get_transformers_tier(model_name, probe = False) != "default"


# ---------------------------------------------------------------------------
# Version switching (in-process — used only by export)
# ---------------------------------------------------------------------------


def _get_in_memory_version() -> str | None:
    """Return the transformers version currently loaded in this process."""
    tf = sys.modules.get("transformers")
    if tf is not None:
        return getattr(tf, "__version__", None)
    return None


# All top-level prefixes that hold references to transformers internals.
_PURGE_PREFIXES = (
    "transformers",
    "huggingface_hub",
    "unsloth",
    "unsloth_zoo",
    "peft",
    "trl",
    "accelerate",
    "auto_gptq",
    # NOTE: bitsandbytes is intentionally EXCLUDED -- it registers torch custom
    # operators via torch.library.define() into torch's global registry, which
    # survives module purge; re-importing after purge -> duplicate registration
    # -> crash.
    # Our own modules that import from transformers at module level.
    "utils.models",
    "core.training",
    "core.inference",
    "core.export",
)


def _purge_modules() -> int:
    """Remove all cached modules for transformers and its dependents.

    Returns the number of modules purged.
    """
    importlib.invalidate_caches()
    to_remove = [
        k
        for k in list(sys.modules.keys())
        if any(k == p or k.startswith(p + ".") for p in _PURGE_PREFIXES)
    ]
    for key in to_remove:
        del sys.modules[key]
    return len(to_remove)


_VENV_T5_530_PACKAGES = (
    f"transformers=={TRANSFORMERS_530_VERSION}",
    "huggingface_hub==1.8.0",
    "hf_xet==1.4.2",
    "tiktoken",
)

_VENV_T5_510_PACKAGES = (
    f"transformers=={TRANSFORMERS_510_VERSION}",
    "huggingface_hub==1.8.0",
    "hf_xet==1.4.2",
    "tiktoken",
)

_VENV_T5_550_PACKAGES = (
    f"transformers=={TRANSFORMERS_550_VERSION}",
    "huggingface_hub==1.8.0",
    "hf_xet==1.4.2",
    "tiktoken",
)

# Backwards-compat alias
_VENV_T5_PACKAGES = _VENV_T5_550_PACKAGES


def _venv_dir_is_valid(venv_dir: str, packages: tuple[str, ...]) -> bool:
    """Return True if *venv_dir* has all *packages* at the correct versions."""
    if not os.path.isdir(venv_dir) or not os.listdir(venv_dir):
        return False
    for pkg_spec in packages:
        parts = pkg_spec.split("==")
        pkg_name = parts[0]
        pkg_version = parts[1] if len(parts) > 1 else None
        pkg_name_norm = pkg_name.replace("-", "_")
        # Directory must exist.
        if not any(
            (Path(venv_dir) / d).is_dir() for d in (pkg_name_norm, pkg_name_norm.replace("_", "-"))
        ):
            return False
        # Unpinned packages: existence is enough.
        if pkg_version is None:
            continue
        # Check version via .dist-info metadata.
        dist_info_found = False
        for di in Path(venv_dir).glob(f"{pkg_name_norm}-*.dist-info"):
            metadata = di / "METADATA"
            if not metadata.is_file():
                continue
            for line in metadata.read_text(errors = "replace").splitlines():
                if line.startswith("Version:"):
                    installed_ver = line.split(":", 1)[1].strip()
                    if installed_ver != pkg_version:
                        logger.warning(
                            "%s has %s==%s but need %s -- venv will be wiped and reinstalled",
                            venv_dir,
                            pkg_name,
                            installed_ver,
                            pkg_version,
                        )
                        return False
                    dist_info_found = True
                    break
            if dist_info_found:
                break
        if not dist_info_found:
            return False
    return True


def _venv_t5_is_valid() -> bool:
    """Backwards-compat: check the Gemma 4 sidecar venv."""
    return _venv_dir_is_valid(_VENV_T5_550_DIR, _VENV_T5_550_PACKAGES)


def _install_to_dir(pkg: str, target_dir: str) -> bool:
    """Install a single package into *target_dir*, preferring uv then pip."""
    # Try uv first (faster) if on PATH -- do NOT install uv at runtime.
    if shutil.which("uv"):
        result = subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--target",
                target_dir,
                "--no-deps",
                "--upgrade",
                pkg,
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            env = child_env_without_native_path_secret(),
            **_windows_hidden_subprocess_kwargs(),
        )
        if result.returncode == 0:
            return True
        logger.warning("uv install of %s failed, falling back to pip", pkg)

    # Fallback to pip.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            target_dir,
            "--no-deps",
            "--upgrade",
            pkg,
        ],
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        env = child_env_without_native_path_secret(),
        **_windows_hidden_subprocess_kwargs(),
    )
    if result.returncode != 0:
        logger.error("install failed:\n%s", result.stdout)
        return False
    return True


def _ensure_venv_dir(venv_dir: str, packages: tuple[str, ...], label: str) -> bool:
    """Ensure *venv_dir* exists with all *packages*. Install if missing."""
    if _venv_dir_is_valid(venv_dir, packages):
        return True

    logger.warning("%s not found or incomplete at %s -- installing at runtime", label, venv_dir)
    shutil.rmtree(venv_dir, ignore_errors = True)
    os.makedirs(venv_dir, exist_ok = True)
    total = len(packages)
    for idx, pkg in enumerate(packages, start = 1):
        logger.info("Installing %s (%d/%d) into %s ...", pkg, idx, total, venv_dir)
        if not _install_to_dir(pkg, venv_dir):
            return False
    logger.info("Installed %s to %s", label, venv_dir)
    return True


def _ensure_venv_t5_530_exists() -> bool:
    """Ensure .venv_t5_530/ exists with transformers 5.3.0."""
    return _ensure_venv_dir(_VENV_T5_530_DIR, _VENV_T5_530_PACKAGES, "transformers 5.3.0")


def _ensure_venv_t5_550_exists() -> bool:
    """Ensure .venv_t5_550/ exists with transformers 5.5.0."""
    return _ensure_venv_dir(
        _VENV_T5_550_DIR,
        _VENV_T5_550_PACKAGES,
        f"transformers {TRANSFORMERS_550_VERSION}",
    )


def _ensure_venv_t5_510_exists() -> bool:
    """Ensure .venv_t5_510/ exists with transformers 5.10.x."""
    return _ensure_venv_dir(
        _VENV_T5_510_DIR,
        _VENV_T5_510_PACKAGES,
        f"transformers {TRANSFORMERS_510_VERSION}",
    )


def _ensure_venv_t5_exists() -> bool:
    """Backwards-compat: ensure the Gemma 4 5.5 sidecar venv exists."""
    return _ensure_venv_t5_550_exists()


# --- User-consented "latest transformers" sidecar (.venv_t5_latest) --------------------------
# Provisioned via ensure_latest_transformers_venv() after the user confirms the upgrade popup
# (utils/transformers_latest.py); pinned in a marker file so restarts revalidate and routing auto-picks it.

# PEP 440-ish release strings only (guards the pip install spec against injection).
_LATEST_VERSION_RE = r"[0-9]+(\.[0-9]+)*((a|b|rc)[0-9]+)?(\.post[0-9]+)?(\.dev[0-9]+)?"


def _is_valid_version_string(version: str) -> bool:
    import re
    return isinstance(version, str) and re.fullmatch(_LATEST_VERSION_RE, version) is not None


# Only the sidecar recipe's own packages, as plain (optionally ==pinned) specs, may
# come from the on-disk pin marker; anything else (URLs, extras, options) is rebuilt.
_PIN_SPEC_RE = re.compile(r"^[A-Za-z0-9_.-]+(==[A-Za-z0-9_.+-]+)?$")
_PIN_ALLOWED_NAMES = frozenset(
    {
        "transformers",
        "huggingface_hub",
        "huggingface-hub",
        "hf_xet",
        "hf-xet",
        "tiktoken",
        "tokenizers",
        "safetensors",
    }
)


def _is_safe_pin_spec(spec: str) -> bool:
    if not _PIN_SPEC_RE.match(spec):
        return False
    name = spec.split("==", 1)[0].lower().replace("_", "-")
    return name in {n.replace("_", "-") for n in _PIN_ALLOWED_NAMES}


def _recover_stranded_latest_sidecar() -> None:
    """Restore a sidecar stranded at ``.old`` by a swap whose activation rename AND its
    rollback both failed (e.g. a lingering worker file handle on Windows blocked both).

    That double failure leaves no live dir and the pin marker gone with it, so the
    sidecar reads as unprovisioned and never self-heals. Recover only when no live dir
    exists and no swap is in flight: the reservation is held throughout the swap, so the
    transient live-absent window of a legitimate swap never triggers a restore."""
    live = Path(_VENV_T5_LATEST_DIR)
    retired = Path(_VENV_T5_LATEST_DIR + ".old")
    try:
        if live.exists() or not retired.is_dir() or sidecar_swap_in_progress():
            return
        os.rename(retired, live)
        logger.info("Recovered .venv_t5_latest from a stranded .old after a failed swap")
    except OSError:
        pass


def _latest_pin_data() -> dict | None:
    """Parsed pin marker: {"version": str, "packages": [specs...]}, or None.

    The marker is JSON; a plain version string (older/simpler writers) is tolerated and
    expanded with the default package set.
    """
    _recover_stranded_latest_sidecar()
    marker = Path(_VENV_T5_LATEST_DIR) / _LATEST_PIN_MARKER
    try:
        if not marker.is_file():
            return None
        raw = marker.read_text(encoding = "utf-8").strip()
    except Exception:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        data = raw
    if isinstance(data, str):
        if not _is_valid_version_string(data):
            return None
        return {"version": data, "packages": list(_venv_t5_latest_packages(data))}
    if not isinstance(data, dict):
        return None
    version = data.get("version")
    if not _is_valid_version_string(version):
        return None
    packages = data.get("packages")
    if not (
        isinstance(packages, list)
        and packages
        and all(isinstance(p, str) and _is_safe_pin_spec(p) for p in packages)
    ):
        # Malformed or unexpected specs (the pin is user-writable on disk) never
        # reach pip: rebuild the canonical set for the pinned version instead.
        packages = list(_venv_t5_latest_packages(version))
    return {"version": version, "packages": packages}


def latest_venv_pinned_version() -> str | None:
    """Exact transformers version pinned in .venv_t5_latest's marker, or None if the
    sidecar was never provisioned (or the marker is unreadable/invalid)."""
    data = _latest_pin_data()
    return data["version"] if data else None


def _venv_t5_latest_packages(version: str, extra_packages: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Package set for the latest sidecar; mirrors the fixed .venv_t5_* sidecars.
    *extra_packages* carries dep-compat shadows (e.g. a newer tokenizers) computed by
    utils.transformers_latest before install."""
    return (
        f"transformers=={version}",
        "huggingface_hub==1.8.0",
        "hf_xet==1.4.2",
        "tiktoken",
    ) + tuple(extra_packages)


# Single reservation for ANY .venv_t5_latest replacement (consented install or lazy repair),
# checked by training/export starts so no worker spawns mid-swap. Backed by a lock FILE (not just
# this flag) so a lazy repair running in a worker subprocess stays visible to the parent's route
# checks; the in-process flag marks ownership (only the owner unlinks the file).
_sidecar_swap_lock = threading.Lock()
_sidecar_swap_active = False
_sidecar_swap_token: str | None = None
_sidecar_swap_kind: str | None = None
# An install is minutes; a lock this old is a crashed owner, not a live swap.
_SWAP_LOCK_STALE_SECS = 2 * 60 * 60


def _swap_lock_path() -> Path:
    return Path(_VENV_T5_LATEST_DIR + ".swaplock")


def _pid_alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        pass
    if os.name == "nt":
        # os.kill(pid, 0) is NOT a POSIX signal-0 liveness probe on Windows: signal 0
        # is CTRL_C_EVENT, so CPython routes it through GenerateConsoleCtrlEvent (a real
        # Ctrl+C to that console group) rather than a harmless check. Probe via OpenProcess.
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            # PROCESS_QUERY_LIMITED_INFORMATION: minimal right, granted across integrity levels.
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            # ERROR_ACCESS_DENIED means the process exists but we may not query it.
            return ctypes.get_last_error() == 5
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
    except Exception:
        return False


def _swap_lock_is_stale(path: Path) -> bool:
    """Stale when the recorded owner is provably dead: a crashed installer is reclaimed
    at once, not after the long cutoff, so `/load`, training, export, and repair are not
    wedged for hours after a crash. A live but slow pip install keeps its lock (its PID
    is alive), so breaking it and racing two swaps on the same staging dirs stays
    impossible. Only a lock whose PID can't be read (mid-write or corrupt) falls back to
    the age cutoff, so the create-before-metadata-write window is never mistaken for dead."""
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    data = _read_swap_lock(path) or {}
    pid = data.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return age > _SWAP_LOCK_STALE_SECS
    return not _pid_alive(pid)


class SidecarSwapInProgress(RuntimeError):
    """A worker start lost the race to a .venv_t5_latest install/repair; retryable."""


def _read_swap_lock(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return None
    except OSError:
        return {}
    except Exception:
        return {}


def try_begin_sidecar_swap(kind: str = "install") -> bool:
    """Reserve the sidecar swap window; False when one is already reserved
    (in this process or, via the lock file, in any worker subprocess).
    *kind* is "install" (consented route) or "repair" (lazy venv repair)."""
    global _sidecar_swap_active, _sidecar_swap_token, _sidecar_swap_kind
    with _sidecar_swap_lock:
        if _sidecar_swap_active:
            return False
        token = f"{os.getpid()}-{time.time_ns()}"
        path = _swap_lock_path()
        try:
            path.parent.mkdir(parents = True, exist_ok = True)
        except OSError:
            pass
        for attempt in range(2):
            try:
                fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if attempt or not _swap_lock_is_stale(path):
                    return False
                try:
                    path.unlink()
                except OSError:
                    return False
            except OSError:
                # Lock file not creatable (odd filesystem): fall back to the process-local reservation.
                fd = None
                break
        if fd is not None:
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(
                        json.dumps(
                            {"pid": os.getpid(), "at": time.time(), "token": token, "kind": kind}
                        )
                    )
            except OSError:
                pass
        _sidecar_swap_active = True
        _sidecar_swap_token = token
        _sidecar_swap_kind = kind
        return True


def end_sidecar_swap() -> None:
    """Release the reservation taken by :func:`try_begin_sidecar_swap`."""
    global _sidecar_swap_active, _sidecar_swap_token, _sidecar_swap_kind
    with _sidecar_swap_lock:
        if _sidecar_swap_active:
            # Only the file WE wrote is removed: if this reservation was declared
            # stale and superseded, unlinking blindly would drop the new owner's
            # live lock and unguard its in-flight swap.
            path = _swap_lock_path()
            data = _read_swap_lock(path)
            if data is not None and data.get("token", _sidecar_swap_token) == _sidecar_swap_token:
                try:
                    path.unlink()
                except OSError:
                    pass
        _sidecar_swap_active = False
        _sidecar_swap_token = None
        _sidecar_swap_kind = None


def sidecar_swap_in_progress() -> bool:
    """True while a .venv_t5_latest install or repair holds the reservation,
    in this process or any other Unsloth process (lock file)."""
    return sidecar_swap_kind() is not None


def sidecar_swap_kind() -> str | None:
    """The active reservation's kind ("install" / "repair"), or None when idle.
    Lets guards that rely on the install route's own abort-on-active-worker
    checks keep refusing for repairs, which have no such checks."""
    with _sidecar_swap_lock:
        if _sidecar_swap_active:
            return _sidecar_swap_kind or "install"
    path = _swap_lock_path()
    try:
        if not path.is_file() or _swap_lock_is_stale(path):
            return None
    except OSError:
        return None
    data = _read_swap_lock(path) or {}
    kind = data.get("kind")
    return kind if kind in ("install", "repair") else "install"


def _stage_and_swap_latest_venv(
    version: str,
    packages: tuple[str, ...],
    before_swap = None,
) -> bool:
    """Stage-and-swap: build the new sidecar next to the live one and swap only
    once complete, so a failed install or marker write never destroys a
    previously working .venv_t5_latest or its pin. Shared by the consented
    install and the lazy repair path. *before_swap* (optional callable) runs
    after the staging build succeeds and immediately before the live dir is
    replaced, so callers can tear down workers only when the swap is certain;
    if it raises, the previous sidecar is left untouched."""
    staging = _VENV_T5_LATEST_DIR + ".staging"
    retired = _VENV_T5_LATEST_DIR + ".old"
    shutil.rmtree(staging, ignore_errors = True)
    try:
        if not _ensure_venv_dir(staging, packages, f"transformers {version} (latest)"):
            # No exception, so the except cleanup below never runs; drop the partial dir.
            shutil.rmtree(staging, ignore_errors = True)
            return False
        (Path(staging) / _LATEST_PIN_MARKER).write_text(
            json.dumps({"version": version, "packages": list(packages)}), encoding = "utf-8"
        )
        if before_swap is not None:
            before_swap()
        shutil.rmtree(retired, ignore_errors = True)
        if os.path.isdir(_VENV_T5_LATEST_DIR):
            os.rename(_VENV_T5_LATEST_DIR, retired)
        try:
            os.rename(staging, _VENV_T5_LATEST_DIR)
        except OSError:
            # Restore the previous sidecar if the final swap fails.
            if not os.path.isdir(_VENV_T5_LATEST_DIR) and os.path.isdir(retired):
                os.rename(retired, _VENV_T5_LATEST_DIR)
            raise
    except Exception as exc:
        logger.error("Could not provision transformers %s into .venv_t5_latest: %s", version, exc)
        shutil.rmtree(staging, ignore_errors = True)
        return False
    shutil.rmtree(retired, ignore_errors = True)
    # CONFIG_MAPPING_NAMES may have changed: drop the cached key set.
    _config_mapping_cache.pop("latest", None)
    logger.info("Provisioned .venv_t5_latest with transformers %s", version)
    return True


def _workers_active_for_repair() -> bool:
    """Best-effort: any parent-visible chat/training/export worker alive. Never
    raises; unavailable backends (worker subprocess, early startup) count idle."""
    try:
        from core.training import get_training_backend
        if get_training_backend().is_training_active():
            return True
    except Exception:
        pass
    try:
        from core.export import get_export_backend

        _export = get_export_backend()
        if _export.is_export_active():
            return True
        _alive = getattr(_export, "is_worker_alive", None)
        if callable(_alive) and _alive():
            return True
    except Exception:
        pass
    try:
        from core.inference import get_inference_backend

        backend = get_inference_backend()
        if getattr(backend, "active_model_name", None):
            return True
        # An in-flight load counts too: its worker spawns moments later.
        if getattr(backend, "loading_models", None):
            return True
        _alive = getattr(backend, "is_worker_alive", None)
        if callable(_alive) and _alive():
            return True
    except Exception:
        pass
    return False


def _ensure_venv_t5_latest_exists() -> bool:
    """Ensure .venv_t5_latest/ holds its pinned transformers version.

    Never installs without a pin: an unprovisioned sidecar (no marker) returns False so
    routing and probing behave exactly as before the feature existed. With a pin present
    it repairs a broken dir the same way the fixed sidecars do.
    """
    pin = _latest_pin_data()
    if pin is None:
        return False
    version = pin["version"]
    packages = tuple(pin["packages"])
    if _venv_dir_is_valid(_VENV_T5_LATEST_DIR, packages):
        return True
    if _env_offline():
        logger.warning(
            ".venv_t5_latest (transformers %s) is incomplete and offline mode is set; "
            "cannot repair it.",
            version,
        )
        return False
    # Repairs are a parent-process action: a worker child's backend singletons are
    # empty, so it cannot see live siblings that may still lazy-import from the
    # sidecar. Fail activation in the child instead; the parent's routing
    # self-heal (guarded below) performs the actual repair.
    try:
        import multiprocessing as _mp
        if _mp.parent_process() is not None:
            logger.warning(
                ".venv_t5_latest is incomplete; repairs run in the parent process. "
                "Retry after the parent repairs the sidecar."
            )
            return False
    except Exception:
        pass
    # Same stage-and-swap as the install, under the same reservation so training/export starts
    # (which check sidecar_swap_in_progress) wait out a lazy repair; a failed repair keeps the pin.
    if not try_begin_sidecar_swap(kind = "repair"):
        logger.warning(
            "Cannot repair .venv_t5_latest: another sidecar install or repair is in progress."
        )
        return False
    try:
        # Worker check UNDER the reservation (the install route quiesces workers;
        # a repair has none): worker starts set their active markers BEFORE
        # rechecking the reservation, so either this check sees them and aborts,
        # or their recheck sees this reservation and aborts -- no interleaving
        # lets a worker spawn against a mid-swap sidecar.
        if _workers_active_for_repair():
            logger.warning(
                "Cannot repair .venv_t5_latest: active chat/training/export workers "
                "may be importing from it. Retry when they are idle."
            )
            return False
        return _stage_and_swap_latest_venv(version, packages)
    finally:
        end_sidecar_swap()


def ensure_latest_transformers_venv(
    version: str,
    extra_packages: tuple[str, ...] = (),
    before_swap = None,
) -> bool:
    """Provision .venv_t5_latest/ pinned to *version* (user-consented install path).

    Reuses the same --target/--no-deps installer as the fixed sidecars, then writes the pin
    marker (version + full package set) so the venv persists across restarts and
    :func:`latest_venv_pinned_version` / routing pick it up automatically.
    *extra_packages* carries dep-compat shadows (see utils.transformers_latest).
    Returns True on success.
    """
    if not _is_valid_version_string(version):
        logger.error("Refusing to install invalid transformers version %r", version)
        return False
    if _env_offline():
        logger.warning(
            "Cannot install transformers %s: HF/transformers offline mode is set.", version
        )
        return False
    packages = _venv_t5_latest_packages(version, extra_packages)
    pin = _latest_pin_data()
    if (
        pin is not None
        and pin["version"] == version
        and tuple(pin["packages"]) == packages
        and _venv_dir_is_valid(_VENV_T5_LATEST_DIR, packages)
    ):
        return True
    return _stage_and_swap_latest_venv(version, packages, before_swap = before_swap)


# --- llm-compressor-main shadow (FP8/FP4 export of newer-transformers models) ---------------------
# Exact, reproducible pins (bump deliberately in review). Full 40-char SHA validated to FP8-quantize
# Qwen3.5 / Gemma-4 / Llama.
_LLMC_MAIN_TRANSFORMERS = "5.10.2"
_LLMC_MAIN_SHA = "973c9c539a84dd9efaf74e115ede5ca419704c18"
_LLMC_MAIN_COMPRESSED_TENSORS = "0.17.2a20260702"
# Installed --no-deps (torch untouched); the full runtime set llm-compressor main needs, pinned.
_VENV_LLMCOMPRESSOR_SPECS = (
    f"transformers=={_LLMC_MAIN_TRANSFORMERS}",
    f"llmcompressor @ git+https://github.com/vllm-project/llm-compressor@{_LLMC_MAIN_SHA}",
    f"compressed-tensors=={_LLMC_MAIN_COMPRESSED_TENSORS}",
    "huggingface-hub==1.21.0",
    "hf-xet==1.5.1",
    "tokenizers==0.22.2",
    "safetensors==0.8.0",
    "accelerate==1.14.0",
    "datasets==5.0.0",
    "pydantic==2.13.4",
    "pydantic-core==2.46.4",
    "typing-inspection==0.4.2",
    "loguru==0.7.3",
    "pyyaml==6.0.3",
    "nvidia-ml-py==13.610.43",
    "pillow==12.3.0",
    "auto-round==0.13.1",
    "regex==2026.6.28",
)
# Fingerprint of the pin set; bump the trailing schema version to force a rebuild on layout changes.
_LLMC_SHADOW_FINGERPRINT = (
    f"{_LLMC_MAIN_SHA}|{_LLMC_MAIN_TRANSFORMERS}|{_LLMC_MAIN_COMPRESSED_TENSORS}|schema=1"
)
_LLMC_SHADOW_MARKER = ".unsloth_llmc_fingerprint"


def _llmcompressor_main_disabled() -> bool:
    """True if the operator forbids the llm-compressor-main shadow (air-gapped / locked-down)."""
    return os.environ.get("UNSLOTH_DISABLE_LLMCOMPRESSOR_MAIN", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _llmcompressor_shadow_is_valid() -> bool:
    """True if the shadow dir exists with a marker matching the current pin fingerprint."""
    marker = Path(_VENV_LLMCOMPRESSOR_DIR) / _LLMC_SHADOW_MARKER
    try:
        return marker.is_file() and marker.read_text().strip() == _LLMC_SHADOW_FINGERPRINT
    except Exception:
        return False


def _ensure_venv_llmcompressor_exists() -> bool:
    """Ensure .venv_llmcompressor/ has the pinned llm-compressor-main stack. Install if missing.

    All specs are installed with --no-deps into a --target dir (mirrors the transformers sidecars),
    so the workspace torch is never touched. Returns True on success.
    """
    if _llmcompressor_shadow_is_valid():
        return True
    if _llmcompressor_main_disabled():
        logger.warning(
            "llm-compressor-main shadow needed but UNSLOTH_DISABLE_LLMCOMPRESSOR_MAIN is set; "
            "compressed export of newer-transformers models will fail fast."
        )
        return False
    if _env_offline():
        logger.warning(
            "llm-compressor-main shadow missing and HF/offline mode is set; cannot provision it."
        )
        return False

    logger.warning(
        "Provisioning llm-compressor-main shadow at %s (one-time, ~a few hundred MB, no torch) ...",
        _VENV_LLMCOMPRESSOR_DIR,
    )
    shutil.rmtree(_VENV_LLMCOMPRESSOR_DIR, ignore_errors = True)
    os.makedirs(_VENV_LLMCOMPRESSOR_DIR, exist_ok = True)

    # Prefer uv (faster) then pip; install every spec at once, --no-deps, prereleases allowed
    # (compressed-tensors ships as a pre-release).
    base = [
        "--target",
        _VENV_LLMCOMPRESSOR_DIR,
        "--no-deps",
        "--prerelease=allow",
        *_VENV_LLMCOMPRESSOR_SPECS,
    ]
    cmds = []
    if shutil.which("uv"):
        cmds.append(["uv", "pip", "install", "--python", sys.executable, *base])
    cmds.append(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            *[a for a in base if a != "--prerelease=allow"],
            "--pre",
        ]
    )

    last_out = ""
    for cmd in cmds:
        result = subprocess.run(
            cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            env = child_env_without_native_path_secret(),
            **_windows_hidden_subprocess_kwargs(),
        )
        last_out = result.stdout or ""
        if result.returncode == 0:
            try:
                (Path(_VENV_LLMCOMPRESSOR_DIR) / _LLMC_SHADOW_MARKER).write_text(
                    _LLMC_SHADOW_FINGERPRINT
                )
            except Exception:
                pass
            logger.info("Provisioned llm-compressor-main shadow at %s", _VENV_LLMCOMPRESSOR_DIR)
            return True
        logger.warning("llm-compressor-main shadow install failed with %s; trying next", cmd[0])

    logger.error(
        "Failed to provision llm-compressor-main shadow (spec: llmcompressor@%s). Output:\n%s",
        _LLMC_MAIN_SHA,
        last_out[-4000:],
    )
    return False


def llmcompressor_shadow_pythonpath() -> str | None:
    """Provision (lazily) the llm-compressor-main shadow and return its sys.path entry, or None.

    Returns None when the shadow is disabled (UNSLOTH_DISABLE_LLMCOMPRESSOR_MAIN), offline, or
    provisioning failed - callers then fall back to the fail-fast path.
    """
    if _llmcompressor_main_disabled():
        return None
    if _ensure_venv_llmcompressor_exists():
        return _VENV_LLMCOMPRESSOR_DIR
    return None


def _activate_venv(venv_dir: str, label: str) -> None:
    """Prepend *venv_dir* to sys.path, purge stale modules, reimport."""
    if venv_dir not in sys.path:
        sys.path.insert(0, venv_dir)
        logger.info("Prepended %s to sys.path", venv_dir)

    count = _purge_modules()
    logger.info("Purged %d cached modules", count)

    import transformers

    logger.info("Loaded transformers %s (%s)", transformers.__version__, label)


def _deactivate_5x() -> None:
    """Remove all .venv_t5_*/ dirs from sys.path, purge stale modules, reimport."""
    for d in (_VENV_T5_530_DIR, _VENV_T5_550_DIR, _VENV_T5_510_DIR, _VENV_T5_LATEST_DIR):
        while d in sys.path:
            sys.path.remove(d)
    logger.info("Removed venv_t5 dirs from sys.path")

    count = _purge_modules()
    logger.info("Purged %d cached modules", count)

    import transformers

    logger.info("Reverted to transformers %s", transformers.__version__)


def ensure_transformers_version(model_name: str) -> None:
    """Ensure the correct ``transformers`` version is active for *model_name*.

    Uses sys.path with .venv_t5_510/, .venv_t5_550/, or .venv_t5_530/
    (pre-installed by setup.sh):
      • Need 5.10.x → prepend .venv_t5_510/ to sys.path, purge modules.
      • Need 5.5.0 → prepend .venv_t5_550/ to sys.path, purge modules.
      • Need 5.3.0 → prepend .venv_t5_530/ to sys.path, purge modules.
      • Need 4.x  → remove all .venv_t5_*/ from sys.path, purge modules.

    For custom-named LoRA adapters, the base model is resolved before checking
    (from ``adapter_config.json`` or, for adapter_model-only LoRAs, the directory
    name).

    NOTE: Training and inference use subprocess isolation instead. Used only by
    the export path (routes/export.py).
    """
    # Only pre-resolve for LoRA adapter dirs; see activate_transformers_for_subprocess.
    if _is_lora_adapter_dir(Path(model_name)):
        resolved = _resolve_base_model(model_name)
    else:
        # A remote adapter's tier is its BASE model's (see activation above).
        resolved = _remote_lora_base(model_name) or model_name
    tier = get_transformers_tier(resolved)
    if model_name != resolved and _safe_is_file(Path(model_name) / "config.json"):
        # Gate on a real local config.json: a checkpoint carries config the base may not
        # surface, but path names alone must not upgrade a plain adapter.
        tier = _higher_tier(tier, get_transformers_tier(model_name))

    if tier == "latest":
        pinned = latest_venv_pinned_version()
        if pinned is None:
            raise RuntimeError(
                f"Cannot activate the latest-transformers sidecar: "
                f"no pin marker at {_VENV_T5_LATEST_DIR}"
            )
        target_version = pinned
        venv_dir = _VENV_T5_LATEST_DIR
        ensure_fn = _ensure_venv_t5_latest_exists
    elif tier == "510":
        target_version = TRANSFORMERS_510_VERSION
        venv_dir = _VENV_T5_510_DIR
        ensure_fn = _ensure_venv_t5_510_exists
    elif tier == "550":
        target_version = TRANSFORMERS_550_VERSION
        venv_dir = _VENV_T5_550_DIR
        ensure_fn = _ensure_venv_t5_550_exists
    elif tier == "530":
        target_version = TRANSFORMERS_530_VERSION
        venv_dir = _VENV_T5_530_DIR
        ensure_fn = _ensure_venv_t5_530_exists
    else:
        target_version = TRANSFORMERS_DEFAULT_VERSION
        venv_dir = None
        ensure_fn = None

    target_major = int(target_version.split(".")[0])

    # Check what's actually loaded in memory
    in_memory = _get_in_memory_version()

    logger.info(
        "Version check for '%s' (resolved: '%s'): need=%s, in_memory=%s",
        model_name,
        resolved,
        target_version,
        in_memory,
    )

    # --- Already correct? ---------------------------------------------------
    if in_memory is not None:
        if in_memory == target_version:
            logger.info(
                "transformers %s already loaded — correct for '%s'",
                in_memory,
                model_name,
            )
            return
        # Different 5.x -> need to switch (e.g. 5.3.0 loaded but need 5.10.x).
        in_memory_major = int(in_memory.split(".")[0])
        if in_memory_major == target_major and venv_dir is None:
            # Both are default (4.x) — close enough.
            logger.info(
                "transformers %s already loaded — correct for '%s'",
                in_memory,
                model_name,
            )
            return

    # --- Switch version -----------------------------------------------------
    if venv_dir is not None:
        # First remove any other 5.x venv from sys.path.
        _deactivate_5x()
        if not ensure_fn():
            raise RuntimeError(
                f"Cannot activate transformers {target_version}: " f"venv missing at {venv_dir}"
            )
        logger.info("Activating transformers %s…", target_version)
        _activate_venv(venv_dir, f"transformers {target_version}")
    else:
        logger.info("Reverting to default transformers %s…", TRANSFORMERS_DEFAULT_VERSION)
        _deactivate_5x()

    final = _get_in_memory_version()
    logger.info("✓ transformers version is now %s", final)
