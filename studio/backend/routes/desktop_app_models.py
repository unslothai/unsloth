# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The models the Desktop apps cards (Claude Desktop, Codex Desktop) offer in their pickers."""

from __future__ import annotations

import asyncio
from typing import Optional

from pydantic import BaseModel


class DesktopAppModel(BaseModel):
    id: str
    loaded: bool = False
    quant: Optional[str] = None


async def available_models() -> list[DesktopAppModel]:
    """Studio's chat catalog for the caller: loaded models first, then every downloaded one (GGUF
    with one row per on-disk quant, and other local weights). The same list, and the same ids,
    POST /models/load accepts, so a pick can always be loaded."""
    from routes.llama_compat import _chat_catalog

    rows = await _chat_catalog()
    models = [
        DesktopAppModel(
            id = row["id"],
            loaded = bool(row.get("loaded")),
            quant = row.get("quant") if isinstance(row.get("quant"), str) else None,
        )
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"]
    ]
    models.sort(key = lambda model: (not model.loaded, model.id.lower()))
    return models


# The context window a gateway load can ask for. Below Codex's need (~25k tokens a turn) the
# model cannot answer at all, so the Agents slider starts here.
AGENT_CONTEXT_MIN = 8192
# Steps the guard estimate is rounded down to, as the slider moves.
_GUARD_STEP = 1024


class DesktopAppModelSettings(BaseModel):
    """What the Agents panel needs to show one model's shared run settings.

    The values themselves (context length, speculative mode) are NOT stored here: they are the
    per-model settings Run settings saves with Remember (browser copy + the server override a
    gateway load applies). This only says where they live and what bounds them."""

    model: str
    # The id and quant Run settings keys this model's remembered settings by when the model is
    # loaded (residentModelConfigTarget): the load path, which a load's override lookup also
    # tries first. alias_id is the catalog spelling an older picker record may sit under.
    config_id: Optional[str] = None
    alias_id: Optional[str] = None
    gguf_variant: Optional[str] = None
    is_gguf: bool = False
    # The server-side remembered values a gateway load of this model applies right now.
    custom_context_length: Optional[int] = None
    speculative_type: Optional[str] = None
    # The context the model was trained for (GGUF header), the slider's top.
    trained_context: Optional[int] = None
    # Embedded MTP (NextN) heads: True has them, False is a headless verdict, None unknown.
    mtp_available: Optional[bool] = None
    # Apple Silicon memory guard: the unified-memory budget a load fits its context under, and
    # about how much context fits in it now. guard_context is None when the full context fits
    # or the guard does not apply (no Metal budget, unsizable model).
    guard_budget_bytes: Optional[int] = None
    guard_context: Optional[int] = None
    loaded: bool = False
    loaded_context: Optional[int] = None


def guard_fitted_context(
    trained: int, low_ctx: int, low_bytes: int, full_bytes: int, budget_bytes: int
) -> Optional[int]:
    """About how much context fits ``budget_bytes``, from two priced points of the same load.

    KV and the context-linear compute grow linearly with the window, so a straight line through
    (low_ctx, low_bytes) and (trained, full_bytes) is the loader's own cost curve to within its
    rounding. None when the full context fits (nothing to cap). Never below AGENT_CONTEXT_MIN:
    the loader itself does not fit below that floor (_FIT_MIN_CTX)."""
    if budget_bytes <= 0 or trained <= 0 or full_bytes <= budget_bytes:
        return None
    if trained <= low_ctx or full_bytes <= low_bytes:
        return max(AGENT_CONTEXT_MIN, min(low_ctx, trained))
    per_token = (full_bytes - low_bytes) / (trained - low_ctx)
    fitted = low_ctx + (budget_bytes - low_bytes) / per_token
    fitted = int(fitted) // _GUARD_STEP * _GUARD_STEP
    return max(AGENT_CONTEXT_MIN, min(trained, fitted))


def _gguf_header_facts(
    model_path: str, variant: Optional[str]
) -> tuple[Optional[int], Optional[bool]]:
    """(trained context, MTP heads present) from the main GGUF on disk; Nones when unreadable."""
    try:
        from routes.inference import (
            _cached_estimate_config,
            _ESTIMATE_NOT_ON_DISK,
            _local_gguf_main_path,
        )
        from utils.models.gguf_metadata import (
            read_gguf_context_length,
            read_gguf_nextn_predict_layers,
        )

        config = _cached_estimate_config(model_path, variant, None, False)
        if config is None or config is _ESTIMATE_NOT_ON_DISK:
            return None, None
        path = _local_gguf_main_path(config)
        if not path:
            return None, None
        trained = read_gguf_context_length(path)
        heads = read_gguf_nextn_predict_layers(path)
        return (
            trained if isinstance(trained, int) and trained > 0 else None,
            None if heads is None else heads > 0,
        )
    except Exception:
        return None, None


# Headroom the guard leaves macOS and other apps on top of what they hold right now, since they
# can grow while llama-server spends seconds mapping weights: 5 % of the Metal working set, and
# never under 4 GiB (23 GiB on a 464 GiB working set, 4 GiB on a 16 GB Mac's ~10.7 GiB).
_GUARD_MARGIN_FRACTION = 0.05
_GUARD_MARGIN_MIN_BYTES = 4 * 1024**3


def _vm_stat_bytes() -> Optional[dict[str, int]]:
    """macOS ``vm_stat`` counters in bytes, by their printed name; None when unreadable."""
    import re
    import subprocess

    try:
        out = subprocess.run(
            ["vm_stat"],
            capture_output = True,
            text = True,
            timeout = 5,
            encoding = "utf-8",
            errors = "replace",
        ).stdout
    except Exception:
        return None
    page = re.search(r"page size of (\d+) bytes", out or "")
    if not page:
        return None
    size = int(page.group(1))
    stats = {}
    for name, value in re.findall(r"^([^:\n]+):\s+(\d+)\.?\s*$", out, re.MULTILINE):
        stats[name.strip().strip('"')] = int(value) * size
    return stats


def unified_memory_held_bytes(stats: Optional[dict[str, int]]) -> Optional[int]:
    """Unified memory macOS cannot hand a new Metal load without taking it from someone.

    Wired pages (the kernel, GPU allocations, a resident model's Metal buffers), what the
    compressor occupies, and anonymous (process-private) pages less the purgeable ones. File-
    backed pages, active or inactive, are a cache of files on disk that macOS drops on demand;
    after an unload most of a big GGUF stays there as "active" file cache, which psutil's
    available (free + inactive) leaves out, so a reload looked hundreds of GB short. None when a
    counter is missing."""
    if not stats:
        return None
    try:
        wired = stats["Pages wired down"]
        anonymous = stats["Anonymous pages"]
    except KeyError:
        return None
    compressor = stats.get("Pages occupied by compressor", 0)
    purgeable = stats.get("Pages purgeable", 0)
    return wired + compressor + max(0, anonymous - purgeable)


def metal_guard_budget(
    working_set: int,
    total: int,
    held: int,
    reclaimable: int = 0,
) -> int:
    """Unified memory a Metal load can count on: what nobody else holds, under Metal's ceiling.

    ``held`` is unified_memory_held_bytes; ``reclaimable`` the part of it the load frees first
    (a resident server it replaces), so only other processes' memory is charged. The result is
    min(Metal working set, total RAM - memory others hold) less the safety margin."""
    if working_set <= 0 or total <= 0:
        return 0
    others = max(0, int(held) - max(0, int(reclaimable)))
    margin = max(_GUARD_MARGIN_MIN_BYTES, int(working_set * _GUARD_MARGIN_FRACTION))
    return max(0, min(int(working_set), int(total) - others) - margin)


def _guard_budget_bytes(reclaimable: int) -> int:
    """The Agents guard's budget. 0 where the Metal guard does not apply (off Apple Silicon,
    unresolvable), exactly as LlamaCppBackend._apple_metal_memory_budget_bytes decides. Where
    vm_stat or the working set cannot be read, that function's figure is used unchanged."""
    from core.inference.llama_cpp import LlamaCppBackend

    if reclaimable > 0:
        fallback = LlamaCppBackend._apple_metal_memory_budget_bytes(reclaimable)
    else:
        fallback = LlamaCppBackend._apple_metal_memory_budget_bytes()
    if not fallback:
        return 0
    try:
        import psutil

        from utils.hardware.hardware import _metal_recommended_working_set_bytes

        working_set = int(_metal_recommended_working_set_bytes())
        total = int(psutil.virtual_memory().total)
    except Exception:
        return int(fallback)
    held = unified_memory_held_bytes(_vm_stat_bytes())
    if held is None or working_set <= 0 or total <= 0:
        return int(fallback)
    return metal_guard_budget(working_set, total, held, reclaimable)


def _serves(llama, model_path: str, variant: Optional[str]) -> bool:
    """Whether a resident llama backend is serving this model (and quant, when both are known)."""
    from core.inference.model_ids import public_model_id

    ident = getattr(llama, "model_identifier", None)
    if not isinstance(ident, str) or not ident:
        return False
    if ident != model_path and public_model_id(ident) != public_model_id(model_path):
        return False
    held = getattr(llama, "hf_variant", None)
    return not (variant and held and str(held).lower() != str(variant).lower())


def _replaced_resident_bytes(model_path: str, variant: Optional[str]) -> int:
    """Unified memory the resident llama-server(s) a load of this model stops first hold now.

    The guard budget is what macOS has available right now, and a loaded model's weights and KV
    are not available: on a 512 GB Mac serving a 300+ GB GGUF the budget read ~150 GB and the
    guard said only 8k of context fit, for the very model already running at its full context.
    A load of a model that is already resident restarts (or keeps) that server, and with Keep
    multiple models loaded Off any load replaces what runs, so that memory comes back before
    the new server allocates. Best effort: 0 when nothing is resident or it cannot be read."""
    try:
        from core.inference import model_slots
        from routes.inference import get_llama_cpp_backend
        from utils.multi_model_settings import get_multi_model_mode

        replace_all = get_multi_model_mode() == "off"
        slots = (None, *model_slots.visible())
    except Exception:
        return 0
    total = 0
    for slot in slots:
        try:
            llama = model_slots.in_slot(slot, get_llama_cpp_backend)
            if not getattr(llama, "is_loaded", False):
                continue
            if not replace_all and not _serves(llama, model_path, variant):
                continue
            total += int(llama.resident_memory_mib() or 0) * 1024 * 1024
        except Exception:
            continue
    return total


async def _guard_context(
    load_request, trained: int, fastapi_request, current_subject
) -> tuple[Optional[int], Optional[int]]:
    """(budget, fitted context) of the Metal memory guard for this load, priced by the same
    /api/inference/estimate-memory arithmetic the Run settings memory bar uses, against the
    budget of ``_guard_budget_bytes`` (which counts a resident server this load replaces, see
    ``_replaced_resident_bytes``, as available)."""
    from models.inference import EstimateMemoryRequest
    from routes.inference import estimate_memory

    reclaimable = await asyncio.to_thread(
        _replaced_resident_bytes, load_request.model_path, load_request.gguf_variant
    )
    budget = await asyncio.to_thread(_guard_budget_bytes, reclaimable)
    if not budget:
        return None, None
    shared = {
        name: getattr(load_request, name)
        for name in EstimateMemoryRequest.model_fields
        if name not in {"model_path", "gguf_variant", "n_ctx", "max_seq_length", "hf_token"}
        and getattr(load_request, name, None) is not None
    }
    # A load spells some choices the estimator does not take (gpu_layers -1 = Auto, where the
    # estimator wants the field left out): drop whatever it would refuse so it prices Auto.
    from pydantic import ValidationError

    try:
        EstimateMemoryRequest(model_path = load_request.model_path, n_ctx = trained, **shared)
    except ValidationError as exc:
        for error in exc.errors():
            if error.get("loc"):
                shared.pop(error["loc"][0], None)
    low_ctx = min(AGENT_CONTEXT_MIN, trained)

    async def price(n_ctx: int) -> Optional[int]:
        try:
            estimate = await estimate_memory(
                EstimateMemoryRequest(
                    model_path = load_request.model_path,
                    gguf_variant = load_request.gguf_variant,
                    n_ctx = n_ctx,
                    **shared,
                ),
                fastapi_request,
                current_subject,
            )
        except Exception:
            return None
        gpu = getattr(estimate, "gpu_bytes", None) or getattr(estimate, "total_bytes", None)
        return int(gpu) if getattr(estimate, "available", False) and gpu else None

    full = await price(trained)
    low = await price(low_ctx) if full is not None else None
    if full is None or low is None:
        return int(budget), None
    return int(budget), guard_fitted_context(trained, low_ctx, low, full, int(budget))


async def model_settings(
    model_id: str, fastapi_request, current_subject: str
) -> DesktopAppModelSettings:
    """Where one catalog model's shared run settings live, and what bounds them."""
    from routes.llama_compat import _catalog_row, _resolve_load_target
    from utils.openai_auto_switch_settings import resolve_override_for_load

    _, row, _ = await _catalog_row(model_id)
    load_request, public_id = await _resolve_load_target(model_id)
    alias = getattr(load_request, "_override_alias_id", None)
    variant = load_request.gguf_variant
    loaded = bool(row.get("loaded")) if isinstance(row, dict) else False
    loaded_context = row.get("context_length") if loaded and isinstance(row, dict) else None
    settings = DesktopAppModelSettings(
        model = public_id,
        loaded = loaded,
        loaded_context = loaded_context
        if isinstance(loaded_context, int) and loaded_context > 0
        else None,
    )
    if alias is None:
        return settings
    _, override = await asyncio.to_thread(
        resolve_override_for_load, load_request.model_path, alias, variant
    )
    trained, mtp = await asyncio.to_thread(_gguf_header_facts, load_request.model_path, variant)
    settings.config_id = load_request.model_path
    settings.alias_id = alias
    settings.gguf_variant = variant
    settings.is_gguf = variant is not None or trained is not None
    custom = override.get("custom_context_length") if isinstance(override, dict) else None
    settings.custom_context_length = custom if isinstance(custom, int) and custom > 0 else None
    spec = override.get("speculative_type") if isinstance(override, dict) else None
    settings.speculative_type = spec if isinstance(spec, str) and spec else None
    settings.trained_context = trained
    settings.mtp_available = mtp
    if trained:
        settings.guard_budget_bytes, settings.guard_context = await _guard_context(
            load_request, trained, fastapi_request, current_subject
        )
    return settings
