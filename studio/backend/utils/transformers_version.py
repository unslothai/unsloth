# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Automatic transformers version switching.

Some newer model architectures (Ministral-3, GLM-4.7-Flash, Qwen3-30B-A3B MoE,
tiny_qwen3_moe) require transformers>=5.3.0, while Gemma 4 models require a
newer 5.x sidecar.  Everything else needs the default 4.57.x that ships with
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

import importlib
import json
import structlog
from loggers import get_logger
import os
import shutil
import subprocess
import sys
from pathlib import Path

from utils.native_path_leases import child_env_without_native_path_secret
from utils.subprocess_compat import (
    windows_hidden_subprocess_kwargs as _windows_hidden_subprocess_kwargs,
)

logger = get_logger(__name__)


def _env_offline() -> bool:
    """True if HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE is set to a truthy value."""
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in (
        "1",
        "true",
        "yes",
    ) or os.environ.get("TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes")


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
}
_TRANSFORMERS_510_MODEL_TYPES: set[str] = {
    "gemma4_unified",
    "gemma4_assistant",
    "gemma4_unified_assistant",
}

# Architecture classes / model_type values that require transformers 5.5.0.
# Checked via config.json (local or HuggingFace).
_TRANSFORMERS_550_ARCHITECTURES: set[str] = {
    "Gemma4ForConditionalGeneration",
}
_TRANSFORMERS_550_MODEL_TYPES: set[str] = {
    "gemma4",
}

# Tokenizer classes that only exist in transformers>=5.x.
_TRANSFORMERS_5_TOKENIZER_CLASSES: set[str] = {
    "TokenizersBackend",
}

# Caches keyed on (model_name, token-hash) so authed/unauthed reads stay separate (a
# gated/private repo's unauthenticated miss must not poison a later authenticated lookup).
_tokenizer_class_cache: dict[tuple[str, str | None], bool] = {}
_config_json_cache: dict[tuple[str, str | None], dict | None] = {}
_config_needs_510_cache: dict[tuple[str, str | None], bool] = {}
_config_needs_550_cache: dict[tuple[str, str | None], bool] = {}

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


def activate_transformers_for_subprocess(model_name: str, hf_token: str | None = None) -> None:
    """Activate the correct transformers version in a subprocess worker.

    Call BEFORE any ML imports. Resolves LoRA adapters to their base model,
    determines the required tier, prepends the appropriate ``.venv_t5_*`` dir to
    ``sys.path``, and propagates it via ``PYTHONPATH`` for child processes
    (e.g. GGUF converter). Used by training, inference, and export workers.

    ``hf_token`` is forwarded to tier detection so a gated/private model whose only 5.x
    signal is an authenticated config/tokenizer reaches the right sidecar, not the default.
    """
    resolved = _resolve_base_model(model_name)
    tier = get_transformers_tier(resolved, hf_token)

    if tier == "510":
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
    if adapter_cfg_path.is_file():
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
    if config_json_path.is_file():
        try:
            with open(config_json_path) as f:
                cfg = json.load(f)
            # Unsloth writes "model_name"; HF writes "_name_or_path"
            base = cfg.get("model_name") or cfg.get("_name_or_path")
            if base and base != str(local_path):
                logger.info(
                    "Resolved checkpoint '%s' → base model '%s' (via config.json)",
                    model_name,
                    base,
                )
                return base
        except Exception as exc:
            logger.debug("Could not read %s: %s", config_json_path, exc)

    # --- Only try the heavier fallback for local directories ----------------
    if local_path.is_dir():
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

    return model_name


def _token_cache_key(model_name: str, hf_token: str | None) -> tuple[str, str | None]:
    """Cache key that keeps authenticated and unauthenticated reads separate, so an
    unauthenticated miss on a gated/private repo never poisons a later authed lookup."""
    import hashlib

    tok = hashlib.sha256(hf_token.encode()).hexdigest()[:16] if hf_token else None
    return (model_name, tok)


def _check_tokenizer_config_needs_v5(model_name: str, hf_token: str | None = None) -> bool:
    """True if the model's tokenizer_class requires transformers 5.x.

    Checks local tokenizer_config.json, else fetches from HuggingFace (authenticated
    with ``hf_token`` so gated/private repos resolve). Cached in
    ``_tokenizer_class_cache``, keyed by (model, token) so an unauthenticated miss does
    not poison a later authed read. Returns False on any network/parse error
    (fail-open to default version).
    """
    cache_key = _token_cache_key(model_name, hf_token)
    if cache_key in _tokenizer_class_cache:
        return _tokenizer_class_cache[cache_key]

    # --- Check local tokenizer_config.json first ---------------------------
    local_path = Path(model_name)
    local_tc = local_path / "tokenizer_config.json"
    if local_tc.is_file():
        try:
            with open(local_tc) as f:
                data = json.load(f)
            tokenizer_class = data.get("tokenizer_class", "")
            result = tokenizer_class in _TRANSFORMERS_5_TOKENIZER_CLASSES
            if result:
                logger.info(
                    "Local check: %s uses tokenizer_class=%s (requires transformers 5.x)",
                    model_name,
                    tokenizer_class,
                )
            _tokenizer_class_cache[cache_key] = result
            return result
        except Exception as exc:
            logger.debug("Could not read %s: %s", local_tc, exc)

    # Offline: skip the 10s urllib fetch (fail-open to lower tier).
    if _env_offline():
        _tokenizer_class_cache[cache_key] = False
        return False

    # --- Fall back to fetching from HuggingFace ----------------------------
    import urllib.request

    url = f"https://huggingface.co/{model_name}/raw/main/tokenizer_config.json"
    headers = {"User-Agent": "unsloth-studio"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    try:
        req = urllib.request.Request(url, headers = headers)
        with urllib.request.urlopen(req, timeout = 10) as resp:
            data = json.loads(resp.read().decode())
        tokenizer_class = data.get("tokenizer_class", "")
        result = tokenizer_class in _TRANSFORMERS_5_TOKENIZER_CLASSES
        if result:
            logger.info(
                "Dynamic check: %s uses tokenizer_class=%s (requires transformers 5.x)",
                model_name,
                tokenizer_class,
            )
        _tokenizer_class_cache[cache_key] = result
        return result
    except Exception as exc:
        logger.debug("Could not fetch tokenizer_config.json for '%s': %s", model_name, exc)
        _tokenizer_class_cache[cache_key] = False
        return False


def _load_config_json(model_name: str, hf_token: str | None = None) -> dict | None:
    """Return parsed ``config.json`` for *model_name*, checking local files first.

    ``hf_token`` authenticates the raw fetch so gated/private repos resolve. The
    cache is keyed on the token so an unauthenticated miss never poisons a later
    authenticated read.
    """
    import hashlib

    tok = hashlib.sha256(hf_token.encode()).hexdigest()[:16] if hf_token else None
    cache_key = (model_name, tok)
    if cache_key in _config_json_cache:
        return _config_json_cache[cache_key]

    local_cfg = Path(model_name) / "config.json"
    if local_cfg.is_file():
        try:
            with open(local_cfg) as f:
                cfg = json.load(f)
            _config_json_cache[cache_key] = cfg
            return cfg
        except Exception as exc:
            logger.debug("Could not read %s: %s", local_cfg, exc)
            _config_json_cache[cache_key] = None
            return None

    if _env_offline():
        _config_json_cache[cache_key] = None
        return None

    import urllib.request

    url = f"https://huggingface.co/{model_name}/raw/main/config.json"
    headers = {"User-Agent": "unsloth-studio"}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    try:
        req = urllib.request.Request(url, headers = headers)
        with urllib.request.urlopen(req, timeout = 10) as resp:
            cfg = json.loads(resp.read().decode())
        _config_json_cache[cache_key] = cfg
        return cfg
    except Exception as exc:
        logger.debug("Could not fetch config.json for '%s': %s", model_name, exc)
        _config_json_cache[cache_key] = None
        return None


def _config_matches_tier(cfg: dict, architectures: set[str], model_types: set[str]) -> bool:
    archs = cfg.get("architectures", [])
    if any(a in architectures for a in archs):
        return True
    if cfg.get("model_type") in model_types:
        return True
    return False


def _config_needs_550(cfg: dict) -> bool:
    return _config_matches_tier(
        cfg,
        _TRANSFORMERS_550_ARCHITECTURES,
        _TRANSFORMERS_550_MODEL_TYPES,
    )


def _config_needs_510(cfg: dict) -> bool:
    return _config_matches_tier(
        cfg,
        _TRANSFORMERS_510_ARCHITECTURES,
        _TRANSFORMERS_510_MODEL_TYPES,
    )


def _check_config_needs_550(model_name: str, hf_token: str | None = None) -> bool:
    """True if ``config.json`` has architectures/model_type needing transformers
    5.5.0 (e.g. Gemma 4).

    Checks locally first, else fetches from HuggingFace (authenticated with
    ``hf_token``). Cached in ``_config_needs_550_cache``, keyed by (model, token).
    Returns False on any error (fail-open to lower tier).
    """
    cache_key = _token_cache_key(model_name, hf_token)
    if cache_key in _config_needs_550_cache:
        return _config_needs_550_cache[cache_key]

    cfg = _load_config_json(model_name, hf_token)
    if cfg is None:
        _config_needs_550_cache[cache_key] = False
        return False

    result = _config_needs_550(cfg)
    if result:
        logger.info(
            "config.json check: %s needs transformers %s (architectures=%s, model_type=%s)",
            model_name,
            TRANSFORMERS_550_VERSION,
            cfg.get("architectures", []),
            cfg.get("model_type"),
        )
    _config_needs_550_cache[cache_key] = result
    return result


def _check_config_needs_510(model_name: str, hf_token: str | None = None) -> bool:
    """Check ``config.json`` for Gemma 4 Unified / 12B architectures (authenticated
    with ``hf_token``; cache keyed by (model, token))."""
    cache_key = _token_cache_key(model_name, hf_token)
    if cache_key in _config_needs_510_cache:
        return _config_needs_510_cache[cache_key]

    cfg = _load_config_json(model_name, hf_token)
    if cfg is None:
        _config_needs_510_cache[cache_key] = False
        return False

    result = _config_needs_510(cfg)
    if result:
        logger.info(
            "config.json check: %s needs transformers %s (architectures=%s, model_type=%s)",
            model_name,
            TRANSFORMERS_510_VERSION,
            cfg.get("architectures", []),
            cfg.get("model_type"),
        )
    _config_needs_510_cache[cache_key] = result
    return result


def _config_saved_by_transformers_5(cfg: dict | None) -> bool:
    """True if ``config.json`` records a transformers >= 5 ``transformers_version``.

    This field is the *saving* version, not the minimum to load, so it never decides the
    tier on its own. It is only a cheap "worth probing" hint: a model saved by 5.x and
    not matched by any fast path may carry a new architecture the default 4.57.x parser
    cannot read. The probe (default-first) makes the authoritative call.
    """
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
    """Return an already-fetched config.json from the in-process cache, without a new
    network round-trip. The tier checks above populate it in the real flow; a miss
    (mocked checks, transient fetch, no config) just means "skip the version-field probe".
    """
    import hashlib

    tok = hashlib.sha256(hf_token.encode()).hexdigest()[:16] if hf_token else None
    return _config_json_cache.get((model_name, tok))


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
    """tier -> (target_dir, ensure_fn), built here so the later _ensure_* defs resolve.
    The ``default`` entry has an empty target_dir (ambient 4.57.x, no sidecar) and is
    always available; it is only probed when a caller passes ``include_default``."""
    return {
        "default": ("", lambda: True),
        "530": (_VENV_T5_530_DIR, _ensure_venv_t5_530_exists),
        "550": (_VENV_T5_550_DIR, _ensure_venv_t5_550_exists),
        "510": (_VENV_T5_510_DIR, _ensure_venv_t5_510_exists),
    }


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
        st = (Path(model_name) / "config.json").stat()
    except OSError:
        return model_name
    return f"{model_name}\0{st.st_size}:{st.st_mtime_ns}"


def _probe_tier(
    model_name: str,
    hf_token: str | None,
    reason: str,
    *,
    include_default: bool = False,
    floor: str = "530",
) -> str:
    """Lowest tier whose built-in parser loads the config; *floor* is the fail-safe.

    Escalates through ``_PROBE_TIER_ORDER`` (optionally prefixed with the ambient
    ``default`` tier when ``include_default`` is set) and returns the first that parses;
    never raises, never escalates on uncertainty:
      - first success wins (cached unless a lower tier was skipped);
      - transient failure (auth/network/offline) -> *floor*, uncached;
      - a skipped/uninstallable sidecar -> uncached (a lower tier may yet be the answer);
      - all tiers probed, none parse -> a remote-code/custom model_type that loads via its
        own code; keep *floor* rather than jumping to a higher 5.x sidecar.

    Callers that already know the model needs 5.x use ``floor='530'`` (never default).
    Callers probing on a weak signal (config saved by transformers 5.x) use
    ``include_default=True, floor='default'`` so a model that still parses on 4.57.x is
    left on the stable default instead of being pushed onto a sidecar.

    Cached per _probe_cache_key for the process lifetime (a model's tier follows its config).
    No Hub commit sha is resolved: that would import huggingface_hub into the worker before
    the sidecar is on sys.path (activation does not purge), pinning the default-env hub.
    """
    if os.environ.get("UNSLOTH_DISABLE_TIER_PROBE", "").lower() in ("1", "true", "yes"):
        return floor
    key = _probe_cache_key(model_name)
    # The probe mode changes what a cached result means: the default-first path
    # (floor='default') may legitimately return 'default', which is wrong to hand back to a
    # tokenizer/known-5.x caller (floor='530', no default tier). Key those modes separately
    # so a 'default' result never leaks across; the legacy 530 mode keeps the bare key.
    if include_default or floor != "530":
        key = f"{key}\0floor={floor}:def={int(include_default)}"
    if key in _probe_tier_cache:
        return _probe_tier_cache[key]

    def _cache(tier: str, *, skipped: bool) -> str:
        # Do not pin a result that depended on a skipped lower tier: once that sidecar is
        # available the lowest valid tier may differ, so re-probe next call.
        if not skipped:
            _probe_tier_cache[key] = tier
        return tier

    venvs = _probe_tier_venvs()
    order = (("default",) + _PROBE_TIER_ORDER) if include_default else _PROBE_TIER_ORDER
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

    Strong signals (architecture/model_type, name substrings) are fast paths. When the
    only signal is "needs some 5.x" (the tokenizer class), the exact tier is resolved by
    probing AutoConfig in each sidecar instead of guessing the lowest one. A model whose
    ``config.json`` was saved by transformers 5.x but matches no fast path is probed
    default-first, so a new 5.x-only architecture is caught while 4.57.x-loadable models
    stay on the default.

    ``probe=False`` skips every AutoConfig probe (the sidecar subprocesses) and is used by
    the cheap :func:`needs_transformers_5` boolean so a parent/log-only caller never spawns
    probes; it still classifies via the cheap signals (a config saved by transformers 5.x
    returns ``"530"`` without probing). The real activation path keeps ``probe=True`` and
    resolves the exact tier (which may be ``"default"`` if the model still parses on 4.57.x).

    Higher 5.x tiers run first.
    """
    lowered = model_name.lower()

    # Local checkpoint names can contain architecture substrings in their
    # directory names (for example a pytest temp dir). If config.json exists,
    # trust it before using name heuristics.
    local_cfg = Path(model_name) / "config.json"
    if local_cfg.is_file():
        cfg = _load_config_json(model_name, hf_token)
        if cfg is not None and _config_needs_510(cfg):
            logger.info(
                "Transformers tier 510 selected for %s (local config.json check)",
                model_name,
            )
            return "510"
        if cfg is not None and _config_needs_550(cfg):
            logger.info(
                "Transformers tier 550 selected for %s (local config.json check)",
                model_name,
            )
            return "550"
        if cfg is not None:
            local_tc = Path(model_name) / "tokenizer_config.json"
            if local_tc.is_file() and _check_tokenizer_config_needs_v5(model_name, hf_token):
                if not probe:
                    return "530"
                return _probe_tier(model_name, hf_token, "local tokenizer needs 5.x")
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
    if "assistant" in lowered and ("gemma-4" in lowered or "gemma4" in lowered):
        logger.info(
            "Transformers tier 510 selected for %s (gemma-4 assistant variant)",
            model_name,
        )
        return "510"
    match = next((sub for sub in TRANSFORMERS_510_MODEL_SUBSTRINGS if sub in lowered), None)
    if match is not None:
        logger.info(
            "Transformers tier 510 selected for %s (substring match: %s)",
            model_name,
            match,
        )
        return "510"
    match = next((sub for sub in TRANSFORMERS_550_MODEL_SUBSTRINGS if sub in lowered), None)
    if match is not None:
        logger.info(
            "Transformers tier 550 selected for %s (substring match: %s)",
            model_name,
            match,
        )
        return "550"
    match = next((sub for sub in TRANSFORMERS_5_MODEL_SUBSTRINGS if sub in lowered), None)
    if match is not None:
        logger.info(
            "Transformers tier 530 selected for %s (substring match: %s)",
            model_name,
            match,
        )
        return "530"

    # --- Slow config fallbacks (network for HF IDs; authenticated with hf_token) --------
    if _check_config_needs_510(model_name, hf_token):
        logger.info("Transformers tier 510 selected for %s (config.json check)", model_name)
        return "510"
    if _check_config_needs_550(model_name, hf_token):
        logger.info("Transformers tier 550 selected for %s (config.json check)", model_name)
        return "550"
    if _check_tokenizer_config_needs_v5(model_name, hf_token):
        if not probe:
            return "530"
        return _probe_tier(model_name, hf_token, "tokenizer needs 5.x")

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
    for d in (_VENV_T5_530_DIR, _VENV_T5_550_DIR, _VENV_T5_510_DIR):
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

    For custom-named LoRA adapters, the base model is resolved from
    ``adapter_config.json`` before checking.

    NOTE: Training and inference use subprocess isolation instead. Used only by
    the export path (routes/export.py).
    """
    # Resolve LoRA adapters to their base model for accurate detection.
    resolved = _resolve_base_model(model_name)
    tier = get_transformers_tier(resolved)

    if tier == "510":
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
