# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted opt-in controls for OpenAI-compatible model auto-switching.

All off by default so existing API behavior is unchanged:
- ``openai_api_auto_switch_model``: when on, a ``/v1`` request whose ``model``
  names a downloaded local GGUF different from the loaded one transparently
  loads it before serving (llama-swap-style). Unknown names pass through.
- ``openai_api_auto_download_model``: when on, a ``/v1`` request naming an
  undownloaded GGUF repo starts a background download instead of failing.
  Gated on auto-switch, which is what serves the model once it lands.
- ``openai_api_auto_unload_idle_seconds``: when > 0, the loaded GGUF is
  unloaded after this many idle seconds to free VRAM. Enabled values have a
  60s floor (0 stays "off"): a tiny TTL tears the model down between turns of
  an active chat, forcing a full weight reload + prompt re-prefill per turn.

The idle TTL can also be set at startup via the ``UNSLOTH_MODEL_IDLE_TTL`` env
var. Unlike the stored setting (which stays gated on auto-switch), the env value
is a standalone default that enables idle-unload even with auto-switch off, for
headless/container deploys; an explicit UI/API value still overrides it.

Reads are cached for a short window because these are consulted on the
per-request hot path; writes invalidate the cache.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Optional

OPENAI_AUTO_SWITCH_SETTING_KEY = "openai_api_auto_switch_model"
OPENAI_AUTO_DOWNLOAD_SETTING_KEY = "openai_api_auto_download_model"
AUTO_UNLOAD_IDLE_SETTING_KEY = "openai_api_auto_unload_idle_seconds"
AUTO_UNLOAD_KEEP_KV_SETTING_KEY = "openai_api_auto_unload_keep_kv"
MODEL_OVERRIDES_SETTING_KEY = "openai_api_auto_switch_overrides"
MODEL_IDLE_TTL_ENV_VAR = "UNSLOTH_MODEL_IDLE_TTL"

DEFAULT_OPENAI_AUTO_SWITCH_ENABLED = False
DEFAULT_OPENAI_AUTO_DOWNLOAD_ENABLED = False
DEFAULT_AUTO_UNLOAD_IDLE_SECONDS = 0
DEFAULT_AUTO_UNLOAD_KEEP_KV = True
MIN_AUTO_UNLOAD_IDLE_SECONDS = 60

_CACHE_TTL_S = 2.0
_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, Any]] = {}


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return None


def _coerce_int(value: Any) -> int | None:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _apply_idle_floor(seconds: int) -> int:
    return 0 if seconds <= 0 else max(MIN_AUTO_UNLOAD_IDLE_SECONDS, seconds)


def _cached_setting(key: str, default: Any) -> Any:
    """Read an app setting, memoized for _CACHE_TTL_S to spare the hot path."""
    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None and now - hit[0] < _CACHE_TTL_S:
            return hit[1]
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(key, None)
    except Exception:
        stored = None
    value = default if stored is None else stored
    with _cache_lock:
        _cache[key] = (now, value)
    return value


def _invalidate(key: str) -> None:
    with _cache_lock:
        _cache.pop(key, None)


def get_openai_auto_switch_enabled() -> bool:
    parsed = _coerce_bool(_cached_setting(OPENAI_AUTO_SWITCH_SETTING_KEY, None))
    return parsed if parsed is not None else DEFAULT_OPENAI_AUTO_SWITCH_ENABLED


def get_stored_openai_auto_download_enabled() -> bool:
    """The persisted auto-download flag, independent of auto-switch, so the UI
    round-trips the saved value across an auto-switch toggle instead of erasing it."""
    parsed = _coerce_bool(_cached_setting(OPENAI_AUTO_DOWNLOAD_SETTING_KEY, None))
    return parsed if parsed is not None else DEFAULT_OPENAI_AUTO_DOWNLOAD_ENABLED


def get_openai_auto_download_enabled() -> bool:
    """Whether a /v1 request may download a GGUF repo it names but doesn't have.

    Gated on auto-switch: that is what loads the model once it lands, so without
    it we would fetch gigabytes nothing can serve.
    """
    return get_stored_openai_auto_download_enabled() and get_openai_auto_switch_enabled()


def _stored_idle_seconds() -> Optional[int]:
    """The persisted idle TTL as an int, or None when never set."""
    return _coerce_int(_cached_setting(AUTO_UNLOAD_IDLE_SETTING_KEY, None))


_env_floor_warned = False


def _env_idle_seconds() -> Optional[int]:
    """UNSLOTH_MODEL_IDLE_TTL as a non-negative seconds value, or None if unset/invalid.

    Floored to MIN_AUTO_UNLOAD_IDLE_SECONDS here (with a one-time warning) since
    headless/container deploys have no UI to surface a validation error."""
    raw = os.environ.get(MODEL_IDLE_TTL_ENV_VAR)
    if raw is None or not raw.strip():
        return None
    parsed = _coerce_int(raw)
    if parsed is None:
        return None
    floored = _apply_idle_floor(parsed)
    if floored != parsed:
        global _env_floor_warned
        if not _env_floor_warned:
            _env_floor_warned = True
            from loggers import get_logger
            get_logger(__name__).warning(
                "%s=%s is below the %ss minimum; using %ss",
                MODEL_IDLE_TTL_ENV_VAR,
                parsed,
                MIN_AUTO_UNLOAD_IDLE_SECONDS,
                floored,
            )
    return floored


def get_stored_auto_unload_idle_seconds() -> int:
    """The persisted idle-unload TTL, independent of whether auto-switch is on.

    The settings UI reads this so it can display and round-trip the saved value;
    toggling auto-switch off must not erase it. Falls back to the env override so
    the UI shows the startup default. The idle loop uses the gated reader below.
    """
    stored = _stored_idle_seconds()
    if stored is not None:
        # Floor legacy values persisted before the minimum existed, so the UI
        # displays the effective TTL and round-trips it cleanly.
        return _apply_idle_floor(stored)
    env = _env_idle_seconds()
    return env if env is not None else DEFAULT_AUTO_UNLOAD_IDLE_SECONDS


def get_auto_unload_idle_seconds() -> int:
    """Effective idle TTL the idle loop runs on (0 = never unload)."""
    stored = _stored_idle_seconds()
    if stored is not None:
        # An explicit UI/API value stays gated on auto-switch: off reports 0 so the
        # off state is identical to pre-feature. Floored to cover values persisted
        # before the minimum existed.
        return _apply_idle_floor(stored) if get_openai_auto_switch_enabled() else 0
    # No stored value: UNSLOTH_MODEL_IDLE_TTL is a standalone startup default that
    # enables idle-unload even with auto-switch off (headless/container deploys).
    env = _env_idle_seconds()
    return env if env is not None else 0


def get_auto_unload_keep_kv() -> bool:
    """Whether the idle unload persists slot KV to disk for restore on reload."""
    parsed = _coerce_bool(_cached_setting(AUTO_UNLOAD_KEEP_KV_SETTING_KEY, None))
    return parsed if parsed is not None else DEFAULT_AUTO_UNLOAD_KEEP_KV


def set_openai_auto_switch(
    enabled: Any,
    idle_seconds: Any,
    keep_kv: Any = None,
    auto_download: Any = None,
) -> tuple[bool, int, bool, bool]:
    """One-transaction write; ``None`` leaves a stored value untouched."""
    parsed_enabled = _coerce_bool(enabled)
    if parsed_enabled is None:
        raise ValueError("OpenAI auto-switch must be true or false.")
    parsed_idle = None
    if idle_seconds is not None:
        parsed_idle = _coerce_int(idle_seconds)
        if parsed_idle is None:
            raise ValueError("Auto-unload idle seconds must be a non-negative integer.")
        if 0 < parsed_idle < MIN_AUTO_UNLOAD_IDLE_SECONDS:
            raise ValueError(
                f"Auto-unload idle seconds must be 0 (off) or at least "
                f"{MIN_AUTO_UNLOAD_IDLE_SECONDS}."
            )
    parsed_keep_kv = None
    if keep_kv is not None:
        parsed_keep_kv = _coerce_bool(keep_kv)
        if parsed_keep_kv is None:
            raise ValueError("Keep KV on idle unload must be true or false.")
    parsed_auto_download = None
    if auto_download is not None:
        parsed_auto_download = _coerce_bool(auto_download)
        if parsed_auto_download is None:
            raise ValueError("Auto-download missing models must be true or false.")
    from storage.studio_db import upsert_app_settings

    updates: dict[str, Any] = {OPENAI_AUTO_SWITCH_SETTING_KEY: parsed_enabled}
    if parsed_idle is not None:
        updates[AUTO_UNLOAD_IDLE_SETTING_KEY] = parsed_idle
    if parsed_keep_kv is not None:
        updates[AUTO_UNLOAD_KEEP_KV_SETTING_KEY] = parsed_keep_kv
    if parsed_auto_download is not None:
        updates[OPENAI_AUTO_DOWNLOAD_SETTING_KEY] = parsed_auto_download
    upsert_app_settings(updates)
    _invalidate(OPENAI_AUTO_SWITCH_SETTING_KEY)
    if parsed_idle is not None:
        _invalidate(AUTO_UNLOAD_IDLE_SETTING_KEY)
    if parsed_keep_kv is not None:
        _invalidate(AUTO_UNLOAD_KEEP_KV_SETTING_KEY)
    if parsed_auto_download is not None:
        _invalidate(OPENAI_AUTO_DOWNLOAD_SETTING_KEY)
    return (
        parsed_enabled,
        parsed_idle if parsed_idle is not None else get_stored_auto_unload_idle_seconds(),
        parsed_keep_kv if parsed_keep_kv is not None else get_auto_unload_keep_kv(),
        (
            parsed_auto_download
            if parsed_auto_download is not None
            else get_stored_openai_auto_download_enabled()
        ),
    )


# --- Per-model launch config -------------------------------------------------
#
# An override is the server-side twin of the UI's per-model config (the browser
# localStorage map behind features/model-picker/model-config). The UI mirrors
# every save here so a model loaded by an OpenAI-compatible API request gets the
# same launch settings a user would get loading it from the picker; without this
# the API path could only ever apply the two legacy fields below.
#
# Legacy entries hold just {llama_extra_args, max_seq_length}; every field is
# optional and absent means "fall back to the app default", so old entries keep
# loading correctly. A write is a full replace of the fields it expresses, so the
# route carries `llama_extra_args` over when the payload omits it (the settings
# UI has no control for launch flags and must not wipe them).
#
# Known gap: the picker resolves a couple of knobs as "per-model value, else the
# user's global preference" -- GPU memory mode and speculative decoding, whose
# globals live in browser localStorage. An override deliberately stores only an
# explicit per-model choice (so the model keeps following later global changes),
# and the server cannot see the globals at all. So for a model that follows the
# global on one of those two, an API load falls back to the app default rather
# than the user's global. Every other field matches the picker exactly.

# Mirrors _valid_cache_types in core/inference/llama_cpp.py.
VALID_KV_CACHE_DTYPES = frozenset(
    {"f16", "bf16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "iq4_nl", "f32"}
)
# Canonical values plus the legacy spellings LoadRequest still accepts.
VALID_SPECULATIVE_TYPES = frozenset(
    {
        "auto",
        "mtp",
        "ngram",
        "mtp+ngram",
        "off",
        "default",
        "draft-mtp",
        "ngram-mod",
        "ngram-simple",
    }
)
# Only these two consume spec_draft_n_max (mirrors MTP_SPECULATIVE_TYPES in the UI).
MTP_SPECULATIVE_TYPES = frozenset({"mtp", "mtp+ngram", "draft-mtp"})
VALID_GPU_MEMORY_MODES = frozenset({"auto", "manual"})

MAX_SEQ_LENGTH_CEILING = 1048576
MAX_CHAT_TEMPLATE_OVERRIDE_BYTES = 65_536


def _clean_str(value: Any, allowed: frozenset[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in allowed else None


def _bounded_int(value: Any, *, minimum: int, maximum: int) -> Optional[int]:
    # bool is a subclass of int, so `gpu_ids: [true, false]` would otherwise pin
    # the model to GPUs 1 and 0.
    if isinstance(value, bool):
        return None
    # int(1.5) is 1, which would silently turn a fractional context into a
    # useless one. Only exact integers count.
    if isinstance(value, float) and not value.is_integer():
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        # OverflowError is float("inf"), which json.loads accepts as `Infinity`.
        return None
    if parsed < minimum or parsed > maximum:
        return None
    return parsed


def normalize_model_override(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate one per-model launch config, dropping anything unusable.

    Silently drops rather than raising: an override is a convenience mirror of the
    UI's config, so one stale field (a KV dtype this llama.cpp build lost, a GPU id
    from another host) must not block persisting the rest or fail the API load that
    reads it. ``validate_extra_args`` is the caller's job -- it lives in the
    llama_server_args allow-list module, which this one must not import.
    """
    entry: dict[str, Any] = {}

    extra_args = payload.get("llama_extra_args")
    if isinstance(extra_args, (list, tuple)) and extra_args:
        entry["llama_extra_args"] = [str(arg) for arg in extra_args]

    # 0 / negative means "unset"; the loader reads absence as the app default.
    for key in ("max_seq_length", "custom_context_length"):
        parsed = _bounded_int(payload.get(key), minimum = 1, maximum = MAX_SEQ_LENGTH_CEILING)
        if parsed:
            entry[key] = parsed

    kv_cache_dtype = _clean_str(payload.get("kv_cache_dtype"), VALID_KV_CACHE_DTYPES)
    if kv_cache_dtype:
        entry["kv_cache_dtype"] = kv_cache_dtype

    speculative_type = _clean_str(payload.get("speculative_type"), VALID_SPECULATIVE_TYPES)
    if speculative_type:
        entry["speculative_type"] = speculative_type
        # Only meaningful for the MTP modes; storing it otherwise would resurface
        # in the UI as an edit the loader silently ignores.
        if speculative_type in MTP_SPECULATIVE_TYPES:
            spec_draft_n_max = _bounded_int(payload.get("spec_draft_n_max"), minimum = 1, maximum = 16)
            if spec_draft_n_max:
                entry["spec_draft_n_max"] = spec_draft_n_max

    if _coerce_bool(payload.get("tensor_parallel")):
        entry["tensor_parallel"] = True

    template = payload.get("chat_template_override")
    if isinstance(template, str) and template.strip():
        # JSON can carry lone surrogates, which encode() rejects outright. Such a
        # template can never render, so it is dropped like any other bad field.
        try:
            template_bytes = len(template.encode("utf-8"))
        except UnicodeEncodeError:
            template_bytes = MAX_CHAT_TEMPLATE_OVERRIDE_BYTES + 1
        if template_bytes <= MAX_CHAT_TEMPLATE_OVERRIDE_BYTES:
            entry["chat_template_override"] = template

    # Only "manual" is a real override: persisting "auto" would pin the model and
    # stop it following later changes to the global GPU memory preference.
    if _clean_str(payload.get("gpu_memory_mode"), VALID_GPU_MEMORY_MODES) == "manual":
        entry["gpu_memory_mode"] = "manual"

    # -1 is Auto (llama.cpp --fit owns layer sizing), which is also the default,
    # so only a pinned count >= 0 is worth storing.
    gpu_layers = _bounded_int(payload.get("gpu_layers"), minimum = 0, maximum = 1024)
    if gpu_layers is not None:
        entry["gpu_layers"] = gpu_layers

    n_cpu_moe = _bounded_int(payload.get("n_cpu_moe"), minimum = 1, maximum = 1024)
    if n_cpu_moe:
        entry["n_cpu_moe"] = n_cpu_moe

    gpu_ids = payload.get("gpu_ids")
    if isinstance(gpu_ids, (list, tuple)) and gpu_ids:
        # De-duplicate, preserving order: resolve_requested_gpu_ids rejects a
        # repeated id outright, so storing [0, 0] would make every later API load
        # of this model fail with a 400 that the picker never hits.
        cleaned_ids: list[int] = []
        for gid in gpu_ids:
            parsed = _bounded_int(gid, minimum = 0, maximum = 1024)
            if parsed is not None and parsed not in cleaned_ids:
                cleaned_ids.append(parsed)
        if cleaned_ids:
            entry["gpu_ids"] = cleaned_ids

    return entry


def resolve_fit_max_seq_length(override: dict[str, Any], *, is_gguf: bool) -> Optional[int]:
    """The ``max_seq_length`` an API load should send for this override.

    Mirrors resolveFitMaxSeqLength in the UI (features/chat/presets/preset-policy.ts):
    under Manual GPU memory with Auto layers, llama.cpp's ``--fit`` owns context
    sizing, so the load sends the explicit context pin (or 0 to hand sizing over)
    rather than the stored max sequence length. Returns None to leave the field
    at the loader's default.
    """
    manual_auto_layers = (
        is_gguf
        and override.get("gpu_memory_mode") == "manual"
        and override.get("gpu_layers") is None
    )
    if manual_auto_layers:
        return override.get("custom_context_length") or 0
    # max_seq_length wins where both are set. The UI only ever sends it for a
    # non-GGUF model (a GGUF's context is `custom_context_length`), so in
    # practice the two never collide from that path; a hand-written or legacy
    # entry that sets it on a GGUF is honoured, which is this API's contract.
    return override.get("max_seq_length") or override.get("custom_context_length")


def model_override_load_kwargs(override: dict[str, Any], *, is_gguf: bool) -> dict[str, Any]:
    """Map a stored per-model config onto ``LoadRequest`` keyword arguments.

    Mirrors the UI's load payload (features/chat/api/chat-adapter.ts) so an API
    auto-switch load and a picker load of the same model produce the same command
    line. GPU placement is GGUF-only there, so it is gated the same way here: a
    safetensors model loads through HF auto-placement and must not inherit a
    hidden GGUF GPU pin.
    """
    if not override:
        return {}
    kwargs: dict[str, Any] = {}

    max_seq_length = resolve_fit_max_seq_length(override, is_gguf = is_gguf)
    if max_seq_length is not None:
        kwargs["max_seq_length"] = max_seq_length
    for source, target in (
        ("llama_extra_args", "llama_extra_args"),
        ("kv_cache_dtype", "cache_type_kv"),
        ("speculative_type", "speculative_type"),
        ("spec_draft_n_max", "spec_draft_n_max"),
        ("tensor_parallel", "tensor_parallel"),
        ("chat_template_override", "chat_template_override"),
    ):
        if override.get(source) is not None:
            kwargs[target] = override[source]

    if is_gguf:
        if override.get("gpu_memory_mode") is not None:
            kwargs["gpu_memory_mode"] = override["gpu_memory_mode"]
        if override.get("gpu_layers") is not None:
            kwargs["gpu_layers"] = override["gpu_layers"]
        if override.get("n_cpu_moe") is not None:
            kwargs["n_cpu_moe"] = override["n_cpu_moe"]
        if override.get("gpu_ids") is not None:
            kwargs["gpu_ids"] = override["gpu_ids"]
    return kwargs


def _looks_like_filesystem_path(model_id: str) -> bool:
    """True for an absolute path id, as the ./models and LM Studio scanners emit."""
    if model_id.startswith(("/", "\\")):
        return True
    # Windows drive letter, e.g. "C:\models\x.gguf".
    return len(model_id) >= 3 and model_id[1] == ":" and model_id[2] in ("\\", "/")


# The three path shapes whose filesystem is case-insensitive, matching the rule
# the browser applies in features/hub/lib/model-identity.ts. Kept in step with
# it: the browser folds these before storing, so the two sides have to agree on
# which paths fold or a stored key becomes unreachable.
_WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_WSL_DRIVE_PATH = re.compile(r"^/mnt/[A-Za-z](?:/|$)")


def _fold_case_insensitive_path(model_id: str) -> Optional[str]:
    """``model_id`` folded for comparison, or None when the path is case-sensitive.

    A Windows drive path, a UNC share and a WSL drive path all name one file
    whatever the casing, and the separator is interchangeable on Windows. A
    POSIX path is not: folding "/models/Foo.gguf" onto "/models/foo.gguf" would
    replay another model's context and GPU pin.
    """
    slashed = model_id.replace("\\", "/")
    if _WINDOWS_DRIVE_PATH.match(model_id):
        minimum = 3
    elif slashed.startswith("//"):
        minimum = 2
    elif _WSL_DRIVE_PATH.match(slashed):
        minimum = 6
    else:
        return None
    trimmed = slashed
    while len(trimmed) > minimum and trimmed.endswith("/"):
        trimmed = trimmed[:-1]
    return trimmed.casefold()


# A quant label may carry a bits-per-weight modifier, because two files at the
# same base quant are kept distinct by it ("IQ4_XS-3.53bpw"). The two label
# helpers disagree on whether to keep it, so anything reading a stored key has
# to accept both forms.
_BPW_SUFFIX = re.compile(r"-[0-9]+(?:\.[0-9]+)?bpw$", re.IGNORECASE)
_MAX_QUANT_SUFFIX_LEN = 64


def split_quant_suffix(value: str) -> Optional[tuple[str, str]]:
    """``(head, quant)`` for a ``head:QUANT`` key, or None when there is none.

    The suffix has to be a real quant label, so an ordinary colon inside a POSIX
    filename is left alone: "/models/foo:bar.gguf" is one valid filename, and
    splitting it would graft /models/foo's launch flags onto a different model.
    """
    from core.inference.llama_cpp import _GGUF_KNOWN_QUANT_RE
    from hub.utils.gguf import extract_quant_label

    head, sep, tail = value.rpartition(":")
    if not sep or not head or not tail:
        return None
    if "/" in tail or "\\" in tail:
        return None
    if len(tail) <= _MAX_QUANT_SUFFIX_LEN and _GGUF_KNOWN_QUANT_RE.fullmatch(
        _BPW_SUFFIX.sub("", tail)
    ):
        return head, tail
    # A .gguf whose filename holds no recognizable quant token is still labelled
    # by the scanner, which falls back to the stem, so keys like
    # "/models/CustomModel.gguf:custommodel" exist. Storage lowercases that
    # label while the scanner probes with the filename's own casing, so the
    # comparison is case-insensitive. Requiring the suffix to be exactly that
    # label is what keeps an ordinary colon out: "/models/foo:bar.gguf" splits
    # to a head that is not a .gguf at all.
    if not head.lower().endswith(".gguf"):
        return None
    filename = head.replace("\\", "/").rsplit("/", 1)[-1]
    return (head, tail) if tail.casefold() == extract_quant_label(filename).casefold() else None


def _fold_posix_path_variant(value: str) -> str:
    """A POSIX path id with only its quant suffix folded.

    The browser lowercases the variant but keeps the path casing, so a stored
    "/models/Foo:q4_k_m" has to be reachable from "/models/Foo:Q4_K_M" without
    also making "/models/Foo.gguf" reachable from "/models/foo.gguf".
    """
    split = split_quant_suffix(value)
    if split is None:
        return value
    head, quant = split
    return f"{head}:{quant.casefold()}"


def get_model_overrides() -> dict[str, dict]:
    """Per-model launch configs keyed by model id (see normalize_model_override)."""
    raw = _cached_setting(MODEL_OVERRIDES_SETTING_KEY, None)
    return raw if isinstance(raw, dict) else {}


def get_model_override(model_id: str) -> dict:
    """The launch override applied when auto-switch loads ``model_id`` (or empty).

    Falls back to a case-insensitive match when nothing matches exactly. Repo ids
    and quants are case-insensitive in practice ("Q4_K_M" and "q4_k_m" name one
    file), and the browser normalizes them to lowercase before storing, so an
    exact-only lookup misses entries written from that side. Exact still wins, and
    an ambiguous fallback matches nothing, so two POSIX paths differing only in
    case stay distinct.
    """
    key = resolve_model_override_key(model_id)
    if key is None:
        return {}
    override = get_model_overrides().get(key)
    return override if isinstance(override, dict) else {}


def resolve_model_override_key(model_id: str) -> Optional[str]:
    """The stored key an override lookup for ``model_id`` would actually hit.

    Shared by read and remove so "what a load applies" and "what forgetting this
    model clears" can never disagree.
    """
    overrides = get_model_overrides()
    if isinstance(overrides.get(model_id), dict):
        return model_id
    if not isinstance(model_id, str):
        return None
    # A POSIX path is case-sensitive and names a different file, so matching
    # "/models/Foo.gguf" against an entry saved for "/models/foo.gguf" would
    # replay another model's context and GPU pin. A Windows drive path, a UNC
    # share and a WSL drive path are not case-sensitive, and the browser folds
    # exactly those before storing, so refusing to fold them here would leave
    # every migrated Windows entry unreachable until the user saved it again.
    if _looks_like_filesystem_path(model_id):
        folded = _fold_case_insensitive_path(model_id)
        if folded is not None:

            def fold(key: str) -> Optional[str]:
                return _fold_case_insensitive_path(key)
        else:
            # POSIX: the path itself stays case-sensitive, but the browser
            # lowercases the quant suffix while keeping the path casing, so a
            # migrated "/models/Foo:q4_k_m" has to stay reachable from the
            # scanner's "/models/Foo:Q4_K_M".
            folded = _fold_posix_path_variant(model_id)

            def fold(key: str) -> Optional[str]:
                # A path only ever folds onto another path.
                if not _looks_like_filesystem_path(key):
                    return None
                return None if _fold_case_insensitive_path(key) else _fold_posix_path_variant(key)
    else:
        folded = model_id.casefold()

        def fold(key: str) -> Optional[str]:
            # A path never folds onto a repo id: the shapes cannot collide.
            return None if _looks_like_filesystem_path(key) else key.casefold()

    matches = [
        key
        for key, value in overrides.items()
        if isinstance(key, str) and fold(key) == folded and isinstance(value, dict)
    ]
    return matches[0] if len(matches) == 1 else None


def set_model_override(
    model_id: str,
    llama_extra_args: Optional[list[str]] = None,
    max_seq_length: Optional[int] = None,
    **config: Any,
) -> dict:
    """Upsert one model's launch config; a config with no usable fields removes it.

    The two legacy parameters stay positional for existing callers; every other
    per-model field is passed by keyword and normalized together.
    """
    if not model_id or not model_id.strip():
        raise ValueError("model_id is required.")
    entry = normalize_model_override(
        {
            **config,
            "llama_extra_args": llama_extra_args,
            "max_seq_length": max_seq_length,
        }
    )

    from storage.studio_db import upsert_app_setting_map_entry

    # Atomic per-entry merge so two PUTs for different models can't drop each other.
    upsert_app_setting_map_entry(MODEL_OVERRIDES_SETTING_KEY, model_id.strip(), entry or None)
    _invalidate(MODEL_OVERRIDES_SETTING_KEY)
    return entry
