# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sliding-window attention metadata cache."""

import json
import os
import re
import tempfile
import threading

from pathlib import Path
from typing import Optional
from hub.utils.hf_tokens import call_hub_with_anonymous_retry

# ── Sliding-window-pattern resolver ───────────────────────────
# Resolves the per-layer SWA mask when a GGUF reports a sliding window but
# no `sliding_window_pattern` field. Tier order in `_resolve_swa_pattern`:
# GGUF metadata, on-disk cache, bootstrap dict below, transformers
# introspection, HF Hub config.json, legacy 1/4 fallback. Period N means
# layer i is SWA iff `(i + 1) % N != 0`, matching transformers. Skipped on
# purpose: phi3 (no key/val length in GGUF, window >= ctx anyway), qwen2
# family (converter strips sliding_window when use_sliding_window=False),
# mistral v0.1/v0.2 (all-SWA can't be a period).
_BOOTSTRAP_SWA_DEFAULTS: dict[str, int] = {
    "gemma2": 2,  # Gemma2Config.sliding_window_pattern
    "gemma3": 6,  # Gemma3TextConfig.sliding_window_pattern
    "gemma3n": 5,  # text_config.layer_types: SWA*4 + FULL
    "gpt_oss": 2,  # text_config.layer_types: alternating
    "cohere2": 4,  # Cohere2Config.sliding_window_pattern
}

# Process-wide cache backed by JSON on disk. Values are int period or
# list[bool] mask; `__missed_repos__` lists casefolded repos with no SWA field. Lazy-loaded.
_SWA_CACHE: Optional[dict] = None
_SWA_CACHE_LOCK = threading.Lock()


def _swa_cache_path() -> Path:
    home = os.environ.get("UNSLOTH_STUDIO_HOME") or os.environ.get("STUDIO_HOME")
    base = Path(home) if home else Path.home() / ".unsloth" / "studio"
    return base / "swa_cache.json"


def _load_swa_cache() -> dict:
    global _SWA_CACHE
    with _SWA_CACHE_LOCK:
        if _SWA_CACHE is not None:
            return _SWA_CACHE
        try:
            with open(_swa_cache_path(), encoding = "utf-8-sig") as f:
                _SWA_CACHE = json.load(f)
                if not isinstance(_SWA_CACHE, dict):
                    _SWA_CACHE = {}
        except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeDecodeError):
            _SWA_CACHE = {}
        return _SWA_CACHE


def _save_swa_cache(cache: dict) -> None:
    try:
        path = _swa_cache_path()
        path.parent.mkdir(parents = True, exist_ok = True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding = "utf-8") as f:
            json.dump(cache, f, indent = 2, sort_keys = True)
        tmp.replace(path)
    except (OSError, UnicodeDecodeError):
        pass


def _period_from_layer_types(layer_types: list) -> Optional[int]:
    """Smallest period N where `(i+1) % N != 0` matches the SWA mask, else None."""
    if not layer_types:
        return None
    is_swa = ["full" not in str(t).lower() for t in layer_types]
    n = len(is_swa)
    for N in range(1, n + 1):
        if all(((i + 1) % N != 0) == is_swa[i] for i in range(n)):
            return N
    return None


def _swa_entry_from_layer_types(lt) -> Optional[object]:
    """Period int, or per-layer bool mask, from a transformers ``layer_types`` list."""
    if isinstance(lt, list) and lt:
        return _period_from_layer_types(lt) or ["full" not in str(t).lower() for t in lt]
    return None


_SWA_CONFIRMED_MISS = False
_SWA_MISSED_REPOS_KEY = "__missed_repos__"


def _swa_missed_repos(cache: dict) -> set:
    missed = cache.get(_SWA_MISSED_REPOS_KEY)
    if not isinstance(missed, list):
        return set()
    return {item.casefold() for item in missed if isinstance(item, str)}


def _remember_swa_repo_miss(cache: dict, repo_id: str) -> None:
    folded = repo_id.casefold()
    with _SWA_CACHE_LOCK:
        missed = _swa_missed_repos(cache)
        if folded in missed:
            return
        missed.add(folded)
        cache[_SWA_MISSED_REPOS_KEY] = sorted(missed)
    _save_swa_cache(cache)


def _fetch_swa_entry_from_hf(repo_id: str) -> Optional[object]:
    try:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache
        from utils.hf_cache_settings import active_hf_hub_cache
        from utils.hf_probe import hf_file_definitely_absent

        cached = try_to_load_from_cache(repo_id, "config.json", cache_dir = active_hf_hub_cache())
        if isinstance(cached, str):
            with open(cached, encoding = "utf-8-sig") as f:
                cfg = json.load(f)
        else:
            # Only a confirmed 404 is a miss; timeouts, 429 and gated stay None so a later load retries.
            if hf_file_definitely_absent(repo_id, "config.json"):
                return _SWA_CONFIRMED_MISS
            # Throwaway cache: a config-only repo in the user's cache lists as a phantom model (#10047).
            with tempfile.TemporaryDirectory(
                prefix = "unsloth-swa-", ignore_cleanup_errors = True
            ) as tmp:
                cfg_path = call_hub_with_anonymous_retry(
                    hf_hub_download,
                    None,
                    repo_id,
                    "config.json",
                    repo_type = "model",
                    cache_dir = tmp,
                )
                with open(cfg_path, encoding = "utf-8-sig") as f:
                    cfg = json.load(f)
    except Exception:
        return None

    src = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    period = src.get("sliding_window_pattern")
    if isinstance(period, int) and period > 0:
        return period
    entry = _swa_entry_from_layer_types(src.get("layer_types"))
    return entry if entry is not None else _SWA_CONFIRMED_MISS


def _arch_aliases(arch: str) -> tuple:
    # GGUF emits `falcon-h1`; HF model_type is `falcon_h1`. Normalise both ways.
    seen = []
    for a in (arch, arch.replace("-", "_"), arch.replace("_", "-")):
        if a and a not in seen:
            seen.append(a)
    return tuple(seen)


def _swa_entry_from_config_obj(cfg) -> Optional[object]:
    src = getattr(cfg, "text_config", None) or cfg
    period = getattr(src, "sliding_window_pattern", None)
    if isinstance(period, int) and period > 0:
        return period
    return _swa_entry_from_layer_types(getattr(src, "layer_types", None))


_SWA_PATTERN_SOURCE_RE = re.compile(r"sliding_window_pattern\s*(?::\s*[\w\[\], ]*)?\s*=\s*(\d+)")


def _resolve_swa_entry_from_transformers(arch: str) -> Optional[object]:
    """Default-instantiate the matching Config; on failure, regex-parse its
    source for `sliding_window_pattern = N`."""
    try:
        from transformers.models.auto.configuration_auto import (
            CONFIG_MAPPING,
            CONFIG_MAPPING_NAMES,
        )
    except Exception:
        return None

    cfg_class = None
    for alias in _arch_aliases(arch):
        if alias in CONFIG_MAPPING_NAMES:
            try:
                cfg_class = CONFIG_MAPPING[alias]
                break
            except Exception:
                cfg_class = None
    if cfg_class is None:
        return None

    try:
        if (entry := _swa_entry_from_config_obj(cfg_class())) is not None:
            return entry
    except Exception:
        pass

    import inspect

    candidates = [cfg_class]
    text_cfg_class = getattr(cfg_class, "sub_configs", {}).get("text_config")
    if text_cfg_class is not None:
        candidates.append(text_cfg_class)
    for cls in candidates:
        try:
            src = inspect.getsource(cls)
        except (OSError, TypeError):
            continue
        if m := _SWA_PATTERN_SOURCE_RE.search(src):
            period = int(m.group(1))
            if period > 0:
                return period
    return None


def _resolve_swa_pattern(
    arch: Optional[str],
    n_layers: Optional[int],
    source_repo_candidates: tuple = (),
    *,
    allow_network: Optional[bool] = None,
) -> Optional[list]:
    if not arch or not n_layers:
        return None
    if allow_network is None:
        allow_network = os.environ.get("UNSLOTH_STUDIO_OFFLINE", "0") not in (
            "1",
            "true",
            "True",
            "yes",
        )

    cache = _load_swa_cache()

    def _entry_to_mask(entry):
        if isinstance(entry, int) and entry > 0:
            return [(i + 1) % entry != 0 for i in range(n_layers)]
        if isinstance(entry, list) and entry:
            return [bool(entry[i % len(entry)]) for i in range(n_layers)]
        return None

    def _persist(entry):
        with _SWA_CACHE_LOCK:
            cache[arch] = entry
        _save_swa_cache(cache)

    if (entry := cache.get(arch)) is not None:
        if (mask := _entry_to_mask(entry)) is not None:
            return mask

    if (entry := _BOOTSTRAP_SWA_DEFAULTS.get(arch)) is not None:
        return _entry_to_mask(entry)

    entry = _resolve_swa_entry_from_transformers(arch)
    if entry is not None:
        _persist(entry)
        return _entry_to_mask(entry)

    # Tier 3: live HF fetch. Misses are keyed per repo, not arch: a new GGUF may name a new source.
    if allow_network:
        seen = set()
        missed = _swa_missed_repos(cache)
        for repo_id in source_repo_candidates:
            if not repo_id:
                continue
            folded = repo_id.casefold()
            if folded in seen or folded in missed:
                continue
            seen.add(folded)
            entry = _fetch_swa_entry_from_hf(repo_id)
            if entry is _SWA_CONFIRMED_MISS:
                _remember_swa_repo_miss(cache, folded)
                continue
            if entry is not None:
                _persist(entry)
                return _entry_to_mask(entry)

    return None
