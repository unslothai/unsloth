# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Source-contract guards for the model-picker per-model-config feature.

These are cheap, CPU-only, no-browser checks that read the frontend source and
assert the specific fixes that got the predecessor PR reverted stay in place. If
a future edit reverts one of them (e.g. rounds the context ceiling up again, or
puts the HF token back in the URL), the matching assertion reddens. They pair
with the runtime Playwright checks (which prove the behavior end to end) and the
backend pytest checks (which prove the backend logic).
"""

from __future__ import annotations

import re
from pathlib import Path

WORKDIR = Path(__file__).resolve().parents[2]
FRONTEND = WORKDIR / "studio" / "frontend" / "src"


def _read(rel: str) -> str:
    path = FRONTEND / rel
    assert path.exists(), f"missing source file: {path}"
    return path.read_text(encoding = "utf-8")


def test_models_api_sends_token_via_header_not_query():
    """getModelConfig / checkVisionModel / checkEmbeddingModel must pass the HF
    token through hubTokenHeader, never as a ?hf_token= query param (which leaks
    the credential into server/proxy access logs)."""
    src = _read("features/training/api/models-api.ts")
    assert src.count("hubTokenHeader(") >= 3
    assert "hf_token=" not in src
    assert '"hf_token"' not in src and "'hf_token'" not in src


def test_model_metadata_probe_never_puts_token_in_query():
    src = _read("features/model-picker/api/model-metadata.ts")
    assert "hf_token=" not in src
    assert '"hf_token"' not in src and "'hf_token'" not in src


def test_model_config_page_floors_the_context_ceiling():
    """The model's native max-context must be FLOORED to the step grid, never
    rounded up (rounding up can offer/persist a length above the model's real
    ceiling and break loading)."""
    src = _read("features/model-picker/components/model-config-page.tsx")
    assert "floorMaxSeqLength(modelMaxPosition.maxPositionEmbeddings)" in src
    assert "normalizeMaxSeqLength(modelMaxPosition.maxPositionEmbeddings)" not in src


def test_compare_load_clears_stale_native_lease():
    """A compare-pane load never comes from the desktop file picker, so it must
    clear any prior picked file's lease token + expiry, otherwise a reload can
    send a stale lease for the now-active model."""
    src = _read("features/chat/shared-composer.tsx")
    assert "activeNativePathToken: null" in src
    assert "activeNativePathExpiresAtMs: null" in src


def test_autoload_records_backend_loaded_model_identity():
    """An inactive-cache inventory row loads by local path, so startup autoload
    must key both the active checkpoint and its summary by the backend's loaded
    model identity instead of the catalog repo id."""
    src = _read("features/chat/api/chat-adapter.ts")
    autoload = src.split("async function loadAutoLoadCandidate", 1)[1]
    autoload = autoload.split("\n  try {", 1)[0]
    assert "const loadedModelId = loadResp.model || modelPath" in autoload
    assert "setCheckpoint(loadedModelId," in autoload
    assert "id: loadedModelId" in autoload
    assert "m.id === loadedModelId" in autoload


def test_rollback_restores_native_lease_expiry_with_token():
    """A failed model switch that rolls back to a previously loaded picked GGUF
    must restore the lease expiry paired with the token, never the token alone
    (which would look non-expiring and skip the expiry guard)."""
    src = _read("features/chat/hooks/use-chat-model-runtime.ts")
    assert "previousActiveNativePathExpiresAtMs" in src
    assert re.search(
        r"activeNativePathExpiresAtMs:\s*previousActiveNativePathToken", src
    ), "rollback must restore the expiry alongside the token"


def test_default_caches_keyed_on_inventory_version():
    """The chat-template and max-position caches must key on the inventory
    version so a model update in the same session invalidates the cached value
    instead of showing the stale revision."""
    src = _read("features/model-picker/hooks/use-model-defaults.ts")
    # Both cache keys (template + max-position) end with the inventory version.
    assert src.count("${inventoryVersion}") >= 2


def test_hidden_infra_model_needles_present():
    """The frontend static needle list must keep hiding the RAG embedder and the
    llama.cpp validation probe."""
    src = _read("features/hub/lib/hidden-models.ts")
    assert '"bge-small-en-v1.5"' in src
    assert '"ggml-org/models"' in src
    assert '"stories260k.gguf"' in src


def test_hidden_models_dynamic_exact_ids_wired():
    """The configured embedder arrives from /api/hub/hidden-models as exact
    repo ids; a substring needle would let a generic basename like "model"
    hide unrelated chat models."""
    src = _read("features/hub/lib/hidden-models.ts")
    assert "toLowerStrings(data.exact_ids)" in src
    assert "dynamicExactIds.includes(lower)" in src


def test_hidden_model_matchers_refresh_with_inventory_version():
    src = _read("features/hub/lib/hidden-models.ts")
    assert "const version = getInventoryVersion()" in src
    assert "matchersFetchVersion === version" in src
    assert "getInventoryVersion() !== version" in src


def test_diffusion_capability_labeled_image_generation():
    """The diffusion capability detects image GENERATORS (FLUX, SDXL,
    text-to-image tags); labeling it "Image to text" showed generators when
    users asked for captioning models."""
    for rel in (
        "features/hub/lib/model-capabilities.ts",
        "features/hub/lib/model-type-filter.ts",
        "features/hub/lib/view-models.ts",
    ):
        src = _read(rel)
        assert "Image to text" not in src, rel
        assert "Image generation" in src, rel


def test_active_model_config_round_trips_gpu_fields():
    """The active model's config must carry the GPU Memory knobs (GGUF only) so
    a sidebar/hub-gear reload cannot silently reset manual GPU settings, and
    "Remember settings" cannot persist a GPU-less config over a saved one."""
    src = _read("features/model-picker/hooks/use-active-model-config.ts")
    for field in ("gpuMemoryMode", "gpuLayers", "nCpuMoe", "selectedGpuIds"):
        assert field in src, field
    assert "if (!isGguf)" in src and "return base" in src
    for rel in (
        "features/chat/chat-page.tsx",
        "features/hub/catalog/sampling-settings-dialog.tsx",
    ):
        assert "useActiveModelConfig(" in _read(rel), rel
    signature = _read("features/model-picker/components/sidebar-model-config.tsx")
    assert "gpuFieldsSignature(config)" in signature
    shared = _read("features/model-picker/model-config/apply-per-model-config.ts")
    assert "export function gpuFieldsSignature" in shared


def test_gpu_picker_round_trips_requested_pool_not_fitted_subset():
    """A GGUF fit may narrow [0, 1] to [0], but load/status hydration must keep
    [0, 1] as the editable pool so a later reload can grow back onto GPU 1."""
    types = _read("features/chat/types/api.ts")
    assert types.count("requested_gpu_ids?: number[] | null") >= 2

    store = _read("features/chat/stores/chat-runtime-store.ts")
    assert "resp.requested_gpu_ids ?? resp.gpu_ids ?? null" in store

    status = _read("features/chat/lib/apply-inference-status-to-store.ts")
    assert "status.requested_gpu_ids ?? status.gpu_ids ?? null" in status


def test_compare_load_uses_each_models_gpu_config():
    src = _read("features/chat/shared-composer.tsx")
    assert "ownConfig.gpuMemoryMode ?? compareLoadKnobs.gpuMemoryMode" in src
    assert "ownConfig.gpuLayers ?? compareLoadKnobs.gpuLayers" in src
    assert "ownConfig.nCpuMoe ?? compareLoadKnobs.nCpuMoe" in src
    assert "if (ownConfig.selectedGpuIds != null)" in src
    assert "reconcilePersistedGpuIds(ownConfig.selectedGpuIds)" in src
    for field in (
        "gpu_memory_mode: effectiveGpuMemoryMode",
        "gpu_layers: effectiveGpuLayers",
        "n_cpu_moe: effectiveNCpuMoe",
        "gpu_ids: effectiveSelectedGpuIds ?? undefined",
    ):
        assert field in src


def test_active_native_gguf_metadata_uses_path_token():
    src = _read("features/model-picker/components/model-config-page.tsx")
    assert "(isActiveModel ? activeNativePathToken : null)" in src
    assert "target.meta.nativePathToken ??" in src
    assert "nativePathToken," in src
    assert '${nativePathToken ?? ""}' in src


def test_model_default_hooks_do_not_reset_state_in_effect():
    src = _read("features/model-picker/hooks/use-model-defaults.ts")
    assert "setFetched(null)" not in src


def test_variant_expander_refreshes_after_delete():
    """Deleting a downloaded quant from an expanded repo that still has other
    cached quants must bump the expander refresh key, or the deleted quant stays
    shown as downloaded and clickable and tries to reload the removed file."""
    src = _read("features/model-picker/components/model-selector/pickers.tsx")
    del_confirm = re.search(
        r"await onDeleteVariant\(v\.quant\);.*?setRefreshKey\(\(key\) => key \+ 1\)",
        src,
        re.S,
    )
    assert del_confirm, "delete onConfirm must bump refreshKey after a successful delete"


def test_local_picker_rows_require_chat_capability():
    """Local inventory rows can be classified non-chat (canChat false, e.g. a
    folder with only config.json). The picker must filter those out, or selecting
    one loads a weightless path; toLocalModelInfo drops capabilities so the memo
    is the only place the guard can live."""
    src = _read("features/model-picker/inventory/use-chat-picker-inventory.ts")
    memo = re.search(r"const localModels = useMemo\(.*?\[inventory\.localRows\]", src, re.S)
    assert memo, "localModels memo not found"
    assert "row.capabilities.canChat" in memo.group(0)


def test_model_picker_toolbar_reflows_before_crossing_picker_edge():
    """The content-sized section tabs and fixed-width dropdowns must reflow,
    while an oversized tab group must shrink labels but preserve its icons."""
    picker = _read("features/model-picker/components/model-selector/pickers.tsx")
    assert '"flex flex-wrap items-center gap-2"' in picker
    assert 'hasConnected ? "-mr-4" : "-mr-2"' in picker
    assert '"flex max-w-full min-w-0 flex-wrap items-center gap-2"' in picker

    tabs = _read("features/model-picker/components/model-selector/pill-tabs.tsx")
    assert 'fit ? "min-w-0 shrink" : "min-w-0 flex-1"' in tabs
    assert '<span className="min-w-0 truncate">{tab.label}</span>' in tabs

    selector = _read("features/model-picker/components/model-selector.tsx")
    assert 'icon={StarIcon} className="size-3.5 shrink-0"' in selector
    assert 'icon={Download01Icon} className="size-3.5 shrink-0"' in selector
    assert 'icon={CloudIcon} className="size-3.5 shrink-0"' in selector


def test_native_picked_gguf_template_read_through_lease():
    """A native (picked / drag-drop) GGUF's path lives only in its signed lease,
    and the picker chat-template GET has no lease plumbing, so the default
    template must be read through the lease-aware validate probe: mint a
    validate-model lease and post include_chat_template. The native token also
    has to reach the fetch (threaded through the hook) and be part of the cache
    key so two picks of the same basename don't share a template."""
    api = _read("features/model-picker/api/templates.ts")
    assert 'consumeNativePathToken(nativePathToken, "validate-model")' in api
    assert "include_chat_template: true" in api
    assert "/api/inference/validate" in api
    hook = _read("features/model-picker/hooks/use-model-defaults.ts")
    assert "nativePathToken," in hook
    assert '${nativePathToken ?? ""}' in hook


def test_model_load_guard_is_cross_instance():
    """The in-flight load guard must consult the shared store pick (not only the
    per-hook ref) and ejectModel must refuse while any instance is loading:
    three live useChatModelRuntime instances exist (chat page, hub page, hub
    gear dialog)."""
    src = _read("features/chat/hooks/use-chat-model-runtime.ts")
    assert "useChatRuntimeStore.getState().loadingModelPick" in src
    assert "clearLoadingModelPick" in src
    eject_body = src.split("const ejectModel", 1)[1]
    assert "loadingModelPick" in eject_body.split("ejectModel,", 1)[0]


def test_partial_safetensors_download_keeps_delete_menu():
    """A stopped partial safetensors download must keep its options menu (the
    Delete affordance) like the GGUF card does, or partial downloads can only
    be cleaned up by finishing or leaving them. During an ACTIVE download the
    menu stays hidden (every item would be disabled: no Copy path while not
    downloaded, no Delete while downloading, pin suppressed in the run bar)."""
    src = _read("features/hub/catalog/safetensors-download-card.tsx")
    assert "(isDownloaded || (isPartial && !downloading))" in src


def test_pinned_validation_uses_cached_local_variant_listing():
    """Pinned-quant validation must use the TTL-cached hub client with
    preferLocalCache (downloaded-ness is local state) instead of one uncached
    round-trip per pinned repo on every picker open. Picker deletes must go
    through the hub inventory client, whose delete invalidates both the
    variants TTL cache and the server-side HF cache scan (the legacy
    /api/models/delete-cached route invalidates neither, so a post-delete
    inventory refresh would resurrect the deleted row until the scan TTL)."""
    src = _read("features/model-picker/components/model-selector/pickers.tsx")
    assert "listGgufVariantsCached(" in src
    assert "preferLocalCache: true" in src
    assert re.search(r'import \{[^}]*\bdeleteCachedModel\b[^}]*\} from "@/features/hub"', src)
    hub_api = _read("features/hub/inventory/api.ts")
    delete_fn = hub_api.split("export async function deleteCachedModel", 1)[1]
    delete_fn = delete_fn.split("export ", 1)[0]
    assert "invalidateGgufVariantsCache(" in delete_fn
    assert "bumpInventoryVersion(" in delete_fn


def test_chat_autoload_scopes_variant_lookup_to_cached_repo_path():
    """Autoload must probe the exact cache row it will load, including rows
    retained from a previously selected Hugging Face cache."""
    src = _read("features/chat/api/chat-adapter.ts")
    auto_load = src.split("async function autoLoadSmallestModel", 1)[1]
    assert auto_load.count("preferLocalCache: true") >= 2
    assert auto_load.count("localPath: repo.cache_path") >= 2

    chat_api = _read("features/chat/api/chat-api.ts")
    variants_fn = chat_api.split("export async function listGgufVariants", 1)[1]
    variants_fn = variants_fn.split("export interface KvCacheEstimate", 1)[0]
    assert 'params.set("prefer_local_cache", "true")' in variants_fn
    assert 'params.set("local_path", localPath)' in variants_fn


def test_cache_location_update_invalidates_frontend_inventory():
    """A successful cache switch must refresh both inventory rows and cached
    GGUF variant results before any stale active-cache identity can be reused."""
    src = _read("features/settings/api/hugging-face-cache.ts")
    update_fn = src.split("export async function updateHuggingFaceCacheSettings", 1)[1]
    assert "bumpInventoryVersion();" in update_fn
    assert "invalidateGgufVariantsCache();" in update_fn


def test_downloaded_list_offsets_virtual_rows():
    """The On Device virtualized list sits below the Pinned block in the same
    scroll element, so it must pass its measured offset as scrollMargin or rows
    past the overscan render blank."""
    src = _read("features/hub/catalog/models-catalog-lists.tsx")
    assert "scrollMargin={scrollMargin}" in src


def test_local_gguf_diagnostics_gate_on_broad_is_gguf():
    """The MTP fallback note and the context/VRAM warning must gate on the broad
    isGguf (variant, loaded gguf context, or .gguf suffix), not the variant-only
    isLoadedGguf, so direct-file and custom-folder GGUF loads keep those
    diagnostics."""
    src = _read("features/chat/chat-settings-sheet.tsx")
    spec = re.search(r"const showSpecFallback =.*?;", src, re.S)
    vram = re.search(r"const showContextVramWarning =.*?;", src, re.S)
    assert spec and "isGguf &&" in spec.group(0) and "isLoadedGguf" not in spec.group(0)
    assert vram and "isGguf &&" in vram.group(0) and "isLoadedGguf" not in vram.group(0)


def test_fixed_layer_gguf_pins_displayed_context():
    """An already-loaded auto-fit GGUF saved with Manual fixed GPU layers must
    pin the shown context, so a later fresh load keeps the fitted placement
    instead of sending native/0 and recreating the OOM."""
    src = _read("features/model-picker/components/model-config-page.tsx")
    assert "const pinFixedLayerContext =" in src
    assert 'config.gpuMemoryMode === "manual"' in src
    assert "customContextLength: activeLoadedContext" in src


def test_fixed_layer_pin_recomputed_after_committing_gpu_layers():
    """pinFixedLayerContext is computed from the render-time config, before a
    same-click GPU Layers draft is committed. handleRun must recompute it from the
    committed effectiveConfig; otherwise typing a positive GPU Layers value on an
    auto-fit GGUF and clicking Reload saves customContextLength: null, so a later
    fresh load sends the native context with fixed layers (the OOM the pin avoids)."""
    src = _read("features/model-picker/components/model-config-page.tsx")
    assert "const effectivePinFixedLayerContext =" in src
    assert 'effectiveConfig.gpuMemoryMode === "manual"' in src
    assert "effectiveConfig.gpuLayers != null" in src
    assert "effectiveConfig.customContextLength == null" in src
    assert "{ ...effectiveConfig, customContextLength: activeLoadedContext }" in src


def test_blur_cache_cleared_on_every_settled_render():
    """The lastBlurCommittedRef bridge is valid only across the single synchronous
    same-click gesture that set it. Keying its clear on [value] missed a Reset (or
    external edit) that restores the shown value unchanged after the blur dispatched
    onChange: value nets back to its prior number, the effect never re-ran, and a
    later Load/Save replayed the override Reset removed. Clear it on every settled
    render instead."""
    src = _read("features/model-picker/components/numeric-value-input.tsx")
    # The clearing effect must run on every commit, not be gated on [value] alone.
    assert not re.search(r"lastBlurCommittedRef\.current = null;\s*\}, \[value\]\);", src)
    assert re.search(
        r"useEffect\(\(\) => \{\s*lastBlurCommittedRef\.current = null;\s*\}\);",
        src,
    )


def test_auto_defaults_not_persisted_as_overrides():
    """Auto GPU memory mode and Auto/default speculative type are follow-global
    defaults; normalization must not persist them as per-model overrides, else a
    model stops following later changes to the global preference."""
    src = _read("features/model-picker/model-config/per-model-config.ts")
    assert 'if (partial.gpuMemoryMode === "manual") {' in src
    assert 'partial.gpuMemoryMode === "auto" || partial.gpuMemoryMode === "manual"' not in src
    spec = re.search(r'if \(s === "auto" \|\| s === "default"\) \{\s*return ([^;]+);', src)
    assert spec and spec.group(1).strip() == "null"


def test_compare_pane_context_from_own_config_only():
    """A compare pane's context comes from its own config only (a saved pin, else
    null for Auto/native); it must not inherit the active model's shared snapshot,
    which resolveFitMaxSeqLength would treat as an explicit pin (VRAM/OOM)."""
    src = _read("features/chat/shared-composer.tsx")
    assert "const effectiveCustomContextLength = ownConfig.customContextLength;" in src
    assert "compareLoadKnobs.customContextLength" not in src


def test_reset_max_seq_length_falls_back_to_app_default():
    """After Reset clears maxSeqLength (null), a non-GGUF active model's shown
    max sequence length must fall back to the app default, never the loaded
    runtime snapshot, or a remembered/active override can never be cleared."""
    src = _read("features/model-picker/components/model-config-page.tsx")
    # The null fallback resolves to the app-default constant, not a runtime value.
    assert "clampMaxSeqLength(DEFAULT_MAX_SEQ_LENGTH, nativeMaxSeqLength)" in src
    # The buggy runtime-seeded fallback must not come back.
    assert "clampMaxSeqLength(initialMaxSeqLength" not in src


def test_reset_persists_null_max_length_and_substitutes_only_for_load():
    """The persisted per-model record must keep config.maxSeqLength (null after
    Reset) so isDefaultConfig can clear a remembered override; the concrete
    fallback is substituted only into the load request, not the saved record."""
    src = _read("features/model-picker/components/model-config-page.tsx")
    # Load-only substitution of the resolved value (recomputed from any committed
    # same-click Max Seq Length draft, so it is never dropped).
    assert "maxSeqLength: effectiveMaxSeqLengthValue" in src
    assert "const effectiveLoadConfig" in src
    # The persisted record is saved from effectiveRuntimeConfig; the load request
    # carries effectiveLoadConfig (with any committed context input).
    assert "onRun(effectiveLoadConfig)" in src
    assert "savePerModelConfig(" in src


def test_initial_load_uses_staged_config_payload():
    """Run-settings Load must pass the staged config through to /load even when
    React has not flushed NumericValueInput blur commits into the store yet."""
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    assert "const pendingLoadConfig =" in runtime
    assert "pendingLoadConfig?.kvCacheDtype" in runtime
    assert "pendingLoadConfig?.customContextLength" in runtime
    page = _read("features/model-picker/components/model-config-page.tsx")
    assert "contextInputRef" in page
    assert "contextInputRef.current?.commit()" in page
    numeric = _read("features/model-picker/components/numeric-value-input.tsx")
    assert "export type NumericValueInputHandle" in numeric
    assert "commit:" in numeric
    # P1: commit returns null unless the user actually edited the field,
    # so Load/Save with untouched Auto does not pin native context.
    assert "dirtyRef.current" in numeric
    assert "return null;" in numeric
    # P2: blur clears dirtyRef after commit so Reset/slider cannot be
    # overwritten by a stale draft on a later Load.
    assert "dirtyRef.current = false;" in numeric
    assert "draftRef.current = String(final);" in numeric
    # Same-click Load after blur still sees the committed draft.
    assert "lastBlurCommittedRef" in numeric
    # Invalid drafts must not turn Auto into an explicit pin.
    assert "const commitDraft = (raw: string): number | null" in numeric
    assert re.search(r"if \(!Number\.isFinite\(parsed\)\) \{\s*return null;", numeric)
    assert re.search(
        r"if \(final == null\) \{\s*"
        r"draftRef\.current = String\(value\);\s*"
        r"lastBlurCommittedRef\.current = null;",
        numeric,
    )
    # handleRun only promotes commit() when non-null.
    assert "committedContext != null" in page
    assert "pendingPatch.customContextLength = committedContext;" in page


def test_same_click_commit_covers_all_numeric_inputs():
    """The same-click blur bridge must flush every NumericValueInput-backed
    setting, not just Context Length. Max Seq Length (non-GGUF), GPU Layers and
    MoE Layers (GGUF) also stage their draft only on blur, so handleRun must
    imperatively commit each and fold the value into the staged load config;
    otherwise a value the user typed right before clicking Load/Reload is lost."""
    page = _read("features/model-picker/components/model-config-page.tsx")
    # Each numeric input owns an imperative handle that handleRun commits, and the
    # handle is forwarded down to the actual NumericValueInput.
    for ref in ("maxSeqLengthInputRef", "gpuLayersInputRef", "moeLayersInputRef"):
        assert f"const {ref} = useRef<NumericValueInputHandle>(null);" in page
        assert f"{ref}.current?.commit()" in page
        assert f"inputRef={{{ref}}}" in page
    # The leaf sub-components accept and forward the handle as a ref.
    assert page.count("inputRef?: Ref<NumericValueInputHandle>;") >= 2
    assert "ref={inputRef}" in page
    # Committed drafts are folded into the staged config, gated on non-null so an
    # untouched field never fabricates an override.
    assert "committedMaxSeqLength != null" in page
    assert "committedGpuLayers != null" in page
    assert "committedMoeLayers != null" in page
    assert "pendingPatch.gpuLayers = committedGpuLayers;" in page
    assert "pendingPatch.nCpuMoe = committedMoeLayers;" in page
    # The non-GGUF load path substitutes the committed Max Seq Length draft.
    assert "const effectiveMaxSeqLengthValue =" in page
    assert "maxSeqLength: effectiveMaxSeqLengthValue" in page


def test_context_commit_rechecks_persistence_only_shortcut():
    """Committed context changes must bypass persistence-only saves."""
    src = _read("features/model-picker/components/model-config-page.tsx")
    assert "const effectiveConfig =" in src
    assert "perModelConfigsEqual(effectiveConfig, baseline)" in src
    assert "const effectivePersistenceOnly =" in src
    assert "if (effectivePersistenceOnly)" in src


def test_reset_enabled_for_explicit_context_pin_at_native():
    """An explicit customContextLength that equals the native ceiling is still a
    user override, so contextAtDefault must require customContextLength == null.
    The buggy form treated `contextValue === native` alone as default, wedging
    the Reset button disabled for a deliberate pin-to-native."""
    src = " ".join(_read("features/model-picker/components/model-config-page.tsx").split())
    assert (
        "const contextAtDefault = !target.isGguf || "
        "(config.customContextLength == null && "
        "(nativeContextLength == null || contextValue === nativeContextLength));" in src
    )
    # The old form that ignored an explicit pin equal to native must not return.
    assert (
        "(nativeContextLength == null ? config.customContextLength == null : "
        "contextValue === nativeContextLength)" not in src
    )
    # The app-default constant is the single source of truth (imported, not local).
    assert "DEFAULT_MAX_SEQ_LENGTH," in src
    assert "const DEFAULT_MAX_SEQ_LENGTH = 4096" not in src


def test_compare_pane_non_gguf_falls_back_to_app_default():
    """A non-GGUF compare pane with no saved maxSeqLength must fall back to the
    shared app default, not the active model's runtime snapshot; otherwise an
    unconfigured pane inherits a saved 128K neighbor's context and can OOM."""
    per_model = _read("features/model-picker/model-config/per-model-config.ts")
    assert "export const DEFAULT_MAX_SEQ_LENGTH = 4096;" in per_model
    barrel = _read("features/model-picker/index.ts")
    assert "DEFAULT_MAX_SEQ_LENGTH," in barrel
    src = " ".join(_read("features/chat/shared-composer.tsx").split())
    assert "DEFAULT_MAX_SEQ_LENGTH," in src
    assert (
        "const effectiveMaxSeqLength = ownConfig.customContextLength ?? "
        "normalizeMaxSeqLength(ownConfig.maxSeqLength) ?? "
        "(isGgufLoad ? 0 : DEFAULT_MAX_SEQ_LENGTH);" in src
    )
    # The buggy fallback to the active model's shared runtime value must not return.
    assert "(isGgufLoad ? 0 : maxSeqLength)" not in src
    assert "const maxSeqLength = store.params.maxSeqLength;" not in src


def test_default_gpu_mode_clears_manual_knobs():
    """Switching GPU Memory back to Default must clear the Manual-only knobs
    (gpuLayers/nCpuMoe/selectedGpuIds); otherwise a remembered config keeps stale
    pins that a later load re-applies when the global preference is Manual."""
    src = _read("features/model-picker/components/model-config-page.tsx")
    assert 'gpuMemoryMode: "auto",' in src
    assert "gpuLayers: undefined," in src
    assert "nCpuMoe: undefined," in src
    assert "selectedGpuIds: undefined," in src


def test_legacy_migration_is_idempotent_and_non_destructive():
    """The v1->v2 localStorage migration (unsloth_load_settings ->
    unsloth_model_configs) is invoked on every store read, so it must be
    idempotent: repeated reads, browser reloads, and Studio restarts must never
    re-migrate, duplicate records, or overwrite a newer per-model config. This
    was the class of regression that reverted the predecessor PR, so pin all
    three idempotency layers at source level; dropping any of them reddens here.
    """
    raw = _read("features/model-picker/model-config/per-model-config.ts")
    src = " ".join(raw.split())
    # Migration runs from readMap (every store read), so it must be safe to repeat.
    assert (
        "function readMap(): StoredMap { migrateLegacyLoadSettingsOnce(); "
        "return readMapRaw(); }" in src
    )
    # Layer 1: in-memory once-per-session guard so repeated readMap() calls
    # migrate at most once.
    assert "let legacyMigrationChecked = false;" in src
    assert "if (legacyMigrationChecked || !canUseStorage()) {" in src
    assert "legacyMigrationChecked = true;" in src
    # Layer 2: persistent cross-session flag so a completed migration is never
    # redone. Set in every terminal branch (malformed data, nothing to migrate,
    # successful write); a failed quota write leaves it unset so the next session
    # retries. Three set-sites encode exactly that.
    assert 'const LEGACY_MIGRATION_FLAG = "unsloth_model_configs_migrated";' in src
    assert "if (localStorage.getItem(LEGACY_MIGRATION_FLAG)) {" in src
    assert src.count('localStorage.setItem(LEGACY_MIGRATION_FLAG, "1");') >= 3
    # Layer 3: non-overwriting merge skips an existing (or default) key, so even a
    # forced re-run cannot duplicate or clobber a user's config.
    assert "if (isDefaultConfig(migrated) || Object.hasOwn(map, key)) {" in src


def test_vulkan_inference_devices_are_the_pickable_set():
    """GGUF loads run through llama-server, so on a Vulkan build the picker must
    offer the inference inventory (ggml ordinals, the space `--device Vulkan<i>`
    pins) rather than the torch view, which can miss cards llama-server drives.
    The XPU ban must not apply there: it is about torch-xpu ordinals no
    applicator speaks, and a Vulkan pick does not use them.
    """
    src = " ".join(_read("hooks/use-gpu-info.ts").split())
    # The Vulkan inventory is consulted first, and only when it has devices.
    assert (
        "const inference = data?.inference_gpu; "
        'if (inference?.backend === "vulkan" && (inference.devices ?? []).length) {' in src
    )
    # Pinnable on the ggml ordinal space, gated on the backend's own support flag.
    assert "const picksAccepted = inference.gguf_gpu_ids_supported !== false;" in src
    assert 'physicalIndex: picksAccepted && d.index_kind === "vulkan",' in src
    # The torch fallback keeps its physical-only gate and the XPU ban.
    assert 'data?.device_backend !== "xpu" &&' in src
    assert 'physicalIndex: pinnableBackend && d.index_kind === "physical",' in src


def test_training_picker_pagination_preserves_scanned_progress():
    adapter = _read("components/resource-picker/use-picker-hub-pagination.ts")
    assert "const canFetch = enabled && hasMore;" in adapter
    assert "return false;" in adapter
    assert "return fetchMore();" in adapter
    assert "signal: scannedCount" in adapter
    for needle in ("enabled: canFetch", "isFetching", "resetKey", "resultCount"):
        assert needle in adapter

    for rel in (
        "features/model-picker/components/train-model-selector.tsx",
        "features/dataset-picker/components/dataset-selector.tsx",
    ):
        src = _read(rel)
        assert "usePickerHubPagination({" in src
        assert "scannedCount: scannedHfCount" in src
        assert "hasMore: hasMoreHf" in src
        assert "isFetching: isLoadingHf || isLoadingHfMore" in src
        assert "resetKey: picker.debouncedHubQuery" in src
        assert "resultCount: hfResults.length" in src
        assert "hubPagination.fetchMore" in src
        assert "hubPagination.signal" in src
        assert "hubPagination.options" in src
        assert "useLatestRef" not in src


def test_seamless_training_model_picker_has_no_hidden_task_filter():
    src = _read("features/model-picker/components/train-model-selector.tsx")
    search_options = src.split("useHubModelSearch(", 1)[1].split("});", 1)[0]
    assert "MODEL_TYPE_TO_HF_TASKS" not in src
    assert re.search(r"\btask\b", search_options) is None


def test_cached_training_rows_select_canonical_repo_identity():
    src = _read(
        "features/model-picker/components/train-model-picker-view-model.ts"
    )
    cached = src.split("function toCachedTrainModelDeviceItem", 1)[1]
    cached = cached.split("function toLocalTrainModelDeviceItem", 1)[0]
    assert "id: row.repoId" in cached
    assert "localPath: row.cachePath ?? null" in cached

    local = src.split("function toLocalTrainModelDeviceItem", 1)[1]
    local = local.split("function hubTrainingModelCandidate", 1)[0]
    assert 'row.source === "hf_cache" ? row.repoId?.trim() : null' in local
    assert "id: cachedRepoId || row.loadId" in local
    assert 'knownCached: row.source === "hf_cache"' in local
    assert "localPath: row.path" in local

    selector = _read("features/model-picker/components/train-model-selector.tsx")
    assert "toCachedTrainModelDeviceItem(" in selector
    assert ".map(toLocalTrainModelDeviceItem)" in selector


def test_training_picker_controls_keep_visible_keyboard_focus():
    focus = _read("components/resource-picker/picker-focus.ts")
    assert "focus-visible:ring-2" in focus
    assert "focus-visible:ring-ring" in focus
    assert "focus-visible:ring-offset-2" in focus
    assert "focus-visible:ring-offset-background" in focus
    assert "focus-visible:ring-inset" in focus

    trigger = _read("features/model-picker/components/train-picker-trigger.ts")
    dataset = _read("features/dataset-picker/components/dataset-selector.tsx")
    options = _read("components/resource-picker/selectable-picker-item.tsx")
    token = _read("features/hub/components/hf-token-indicator.tsx")
    dataset_controls = _read(
        "features/studio/sections/dataset-panel-controls.tsx"
    )
    dataset_section = _read("features/studio/sections/dataset-section.tsx")
    assert "PICKER_FOCUS_VISIBLE_CLASS" in trigger
    assert "PICKER_FOCUS_VISIBLE_CLASS" in dataset
    assert "PICKER_OPTION_FOCUS_VISIBLE_CLASS" in options
    assert "PICKER_FOCUS_VISIBLE_CLASS" in token
    assert "PICKER_FOCUS_VISIBLE_CLASS" in dataset_controls
    assert "PICKER_FOCUS_VISIBLE_CLASS" in dataset_section
    assert dataset_controls.count("aria-label={t(") >= 4
    assert 't("studio.dataset.streamingInfoAriaLabel")' in dataset_controls
    assert "focus-visible:ring-0" not in trigger
    assert "focus-visible:ring-0" not in dataset
    assert "focus-visible:ring-0" not in token

    hub_css = _read("features/hub/hub.css")
    focus_rule = hub_css.split(".field-soft:focus-visible", 1)[1]
    focus_rule = focus_rule.split("}", 1)[0]
    assert "box-shadow: none !important" not in focus_rule


def test_streaming_dataset_omits_full_download_notice():
    src = _read("features/training/hooks/use-training-resource-notices.ts")
    resolver = src.split("function resolveDatasetNotice", 1)[1]
    resolver = resolver.split("export function useTrainingResourceNotices", 1)[0]
    assert re.search(r"if \(streaming\) \{\s*return null;\s*\}", resolver)
    assert "completeSet: streaming ?" not in resolver
    assert "partialSet: streaming ?" not in resolver


def test_local_dataset_picker_uses_cross_platform_path_identity():
    selector = _read("features/dataset-picker/components/dataset-selector.tsx")
    for needle in (
        "cacheLocalPathMatchesSelection(item.path, query)",
        "cacheLocalPathMatchesSelection(item.path, uploadedFile)",
    ):
        assert needle in selector
    assert "item.path === query" not in selector
    assert "item.path === uploadedFile" not in selector

    lists = _read(
        "features/dataset-picker/components/dataset-selector-lists.tsx"
    )
    assert "export type DatasetDeviceItem" in lists
    assert "export function DatasetDeviceList" in lists
    assert "export function DatasetHubList" in lists
    assert "cacheLocalPathMatchesSelection(selectedLocalPath, item.path)" in lists
    assert "selectedLocalPath === item.path" not in lists
    assert "function DeviceList" not in selector
    assert "function HubList" not in selector

    section = _read("features/studio/sections/dataset-section.tsx")
    assert "cacheLocalPathMatchesSelection(item.path, uploadedFile)" in section
    assert "item.path === uploadedFile" not in section


def test_local_dataset_keyboard_commit_uses_canonical_path_identity():
    shell = _read("components/resource-picker/picker-shell.tsx")
    selector = _read("features/dataset-picker/components/dataset-selector.tsx")
    model_selector = _read(
        "features/model-picker/components/train-model-selector.tsx"
    )
    model_view_model = _read(
        "features/model-picker/components/train-model-picker-view-model.ts"
    )

    assert "onExactQueryCommit?: (query: string) => boolean;" in shell
    assert "if (onExactQueryCommit?.(activeQuery))" in shell
    assert shell.index("if (onExactQueryCommit?.(activeQuery))") < shell.index(
        "const exactMatch = scrollRef.current"
    )
    assert "values.includes(query)" in shell

    exact_commit = selector.split("const commitExactQuery = useCallback", 1)[1]
    exact_commit = exact_commit.split("const selectedLocalDatasetTitle", 1)[0]
    assert 'if (tab === "hub")' in exact_commit
    assert "hubResourceIdsEqual(candidate.id, query)" in exact_commit
    assert "findExactLocalDataset(query, deviceItems)" in exact_commit
    assert "selectLocalDataset(item.path)" in exact_commit
    assert "onExactQueryCommit={commitExactQuery}" in selector

    model_exact_commit = model_selector.split(
        "function commitExactQuery", 1
    )[1]
    model_exact_commit = model_exact_commit.split(
        "const display = selectedModel", 1
    )[0]
    assert 'if (tab === "hub")' in model_exact_commit
    assert "findCanonicalHubResourceId(query, hubResultIds)" in model_exact_commit
    assert (
        "findExactTrainModelDeviceItem(query, trainableLocalModels)"
        in model_exact_commit
    )
    assert "pickDeviceModel(model)" in model_exact_commit
    device_picker = model_selector.split("function pickDeviceModel", 1)[1]
    device_picker = device_picker.split("function commitExactQuery", 1)[0]
    assert "knownCached: model.knownCached" in device_picker
    assert "localPath: model.localPath" in device_picker
    assert "modelFormat: model.modelFormat" in device_picker
    assert "onExactQueryCommit={commitExactQuery}" in model_selector
    assert "cacheLocalPathMatchesSelection(item.path, query)" in model_view_model


def test_non_explicit_picker_tab_infers_both_connectivity_directions():
    source = _read("components/resource-picker/use-picker-state.ts")
    resolver = source.split("function resolvePickerTab", 1)[1]
    resolver = resolver.split("export function usePickerState", 1)[0]
    compact = " ".join(resolver.split())

    assert (
        "const inferredTab = hasExplicitTabPreference ? selectedTab : "
        "shouldUseDeviceTab ? PICKER_TAB.device : PICKER_TAB.hub;" in compact
    )
    assert "return lockedInferredTab ?? inferredTab;" in compact


def test_s3_round_trip_restores_source_qualified_browse_dataset_selection():
    types = _read("features/training/types/config.ts")
    browse_type = types.split("export type BrowseDatasetSelection =", 1)[1]
    browse_type = browse_type.split("export type DatasetManualMapping", 1)[0]
    assert 'source: "huggingface";' in browse_type
    assert "dataset: string | null;" in browse_type
    assert "knownCached: boolean;" in browse_type
    assert "localPath: string | null;" in browse_type
    assert 'source: "upload";' in browse_type
    assert "uploadedFile: string | null;" in browse_type

    store = _read("features/training/stores/training-config-store.ts")
    hf_selection = store.split("const selectHfDatasetInternal", 1)[1]
    hf_selection = hf_selection.split("const selectLocalDatasetInternal", 1)[0]
    assert "createHfBrowseDatasetSelection(" in hf_selection
    assert "datasetKnownCached: browseDatasetSelection.knownCached" in hf_selection
    assert "datasetLocalPath: browseDatasetSelection.localPath" in hf_selection

    upload_selection = store.split("const selectLocalDatasetInternal", 1)[1]
    upload_selection = upload_selection.split("const selectS3SourceInternal", 1)[0]
    assert "createUploadBrowseDatasetSelection(uploadedFile)" in upload_selection

    s3_selection = store.split("const selectS3SourceInternal", 1)[1]
    s3_selection = s3_selection.split(
        "const restoreBrowseDatasetSourceInternal", 1
    )[0]
    for needle in (
        'datasetSource: "s3",',
        "browseDatasetSelection,",
        "dataset: null,",
        "uploadedFile: null,",
        "datasetKnownCached: false,",
        "datasetLocalPath: null,",
    ):
        assert needle in s3_selection

    restore = store.split("const restoreBrowseDatasetSourceInternal", 1)[1]
    restore = restore.split("const selectModelInternal", 1)[0]
    assert 'selection.source === "upload"' in restore
    assert "selectLocalDatasetInternal(selection.uploadedFile)" in restore
    assert "selectHfDatasetInternal(selection.dataset" in restore
    assert "knownCached: selection.knownCached" in restore
    assert "localPath: selection.localPath" in restore

    assert "version: 14," in store
    migration = store.split("if (version < 14)", 1)[1]
    migration = migration.split("return s as unknown as TrainingConfigStore", 1)[0]
    assert "createUploadBrowseDatasetSelection(uploadedFile)" in migration
    assert "createHfBrowseDatasetSelection(dataset" in migration

    toggle = _read("features/studio/sections/dataset-panel-controls.tsx")
    assert "restoreBrowseDatasetSource();" in toggle
    assert "selectLocalDataset(uploadedFile)" not in toggle
    assert "selectHfDataset(dataset)" not in toggle


def test_s3_training_payload_excludes_remembered_browse_sources():
    mapper = " ".join(_read("features/training/api/mappers.ts").split())
    assert (
        'const s3 = config.datasetSource === "s3" ? config.s3Config : null;'
        in mapper
    )
    assert (
        'const hfDataset = config.datasetSource === "huggingface" '
        "? config.dataset : null;" in mapper
    )
    assert (
        'const localDatasets = config.datasetSource === "upload" && '
        "config.uploadedFile ? [config.uploadedFile] : [];" in mapper
    )
    assert (
        "dataset_known_cached: hfDataset && !config.datasetStreaming "
        "? config.datasetKnownCached : false," in mapper
    )
    assert (
        "dataset_local_path: hfDataset && !config.datasetStreaming "
        "? config.datasetLocalPath : null," in mapper
    )
    assert (
        'config.datasetSource === "upload" && config.uploadedEvalFile '
        "? [config.uploadedEvalFile] : []" in mapper
    )
    assert "browseDatasetSelection" not in mapper


def test_dataset_hub_list_retains_the_active_hf_selection():
    selector = _read("features/dataset-picker/components/dataset-selector.tsx")
    hub_items = selector.split("const hubItems = useMemo", 1)[1]
    hub_items = hub_items.split("const hubPagination", 1)[0]
    assert 'datasetSource !== "huggingface"' in hub_items
    assert (
        "hfResults.some((item) => hubResourceIdsEqual(item.id, dataset))"
        in hub_items
    )
    assert "return [...hfResults, { id: dataset }];" in hub_items
    assert "hasExactDatasetMatch(" in selector
    assert re.search(
        r"hasExactDatasetMatch\(\s*activeQuery,\s*tab,\s*hubItems,",
        selector,
    )
    assert "<DatasetHubList" in selector
    assert "items={hubItems}" in selector


def test_filtered_hub_pages_keep_the_pagination_sentinel_mounted():
    for rel, export_name, empty_collection in (
        (
            "features/dataset-picker/components/dataset-selector-lists.tsx",
            "DatasetHubList",
            "items",
        ),
        (
            "features/model-picker/components/train-model-picker-lists.tsx",
            "TrainModelHubList",
            "ids",
        ),
    ):
        source = _read(rel)
        hub_list = source.split(f"export function {export_name}", 1)[1]
        assert f"if ({empty_collection}.length === 0)" in hub_list
        assert "if (!hasQuery)" in hub_list
        assert "<PickerSearchError" in hub_list
        assert "compact={true}" in hub_list
        assert re.search(r"<div ref=\{sentinelRef\} className=\"h-px\" />", hub_list)
        assert not re.search(
            rf"if \({empty_collection}\.length === 0\).*?"
            r"if \(hasQuery\)\s*\{\s*return null;",
            hub_list,
            re.S,
        )


def test_infinite_scroll_retries_an_intersecting_filtered_page():
    source = _read("features/hub/hooks/use-hub-infinite-scroll.ts")
    assert "const PREFETCH_MARGIN_PX = 200;" in source
    assert "function isSentinelWithinPrefetchRange(" in source
    assert "root.getBoundingClientRect()" in source
    assert "sentinel.getBoundingClientRect()" in source
    assert (
        "lastRequestedSignalRef.current !== null &&\n"
        "            signal <= lastRequestedSignalRef.current"
        in source
    )
    assert "lastRequestedSignalRef.current = signalRef.current;" in source
    assert "lastRequestedSignalRef.current = null;" in source
    assert "isFetching," in source
    assert "signal," in source


def test_infinite_scroll_observes_a_late_mounted_sentinel():
    source = _read("features/hub/hooks/use-hub-infinite-scroll.ts")
    assert "const [sentinelNode, setSentinelNode]" in source
    assert "const sentinelRef = useCallback(" in source
    assert "observer.observe(sentinelNode);" in source
    assert "if (!root) {" in source
    assert "if (!sentinelNode) {" in source
    assert "}, [enabled, requestAutomaticPage, sentinelNode]);" in source
    assert "sentinelNode,\n    setManualFetchAvailable," in source


def test_train_hub_selections_preserve_canonical_identity():
    helper = _read("components/resource-picker/hub-resource-id.ts")
    model_selector = _read(
        "features/model-picker/components/train-model-selector.tsx"
    )
    model_view = _read(
        "features/model-picker/components/train-model-picker-view-model.ts"
    )
    dataset_selector = _read(
        "features/dataset-picker/components/dataset-selector.tsx"
    )

    assert "first?.trim().toLowerCase()" in helper
    assert "second?.trim().toLowerCase()" in helper
    assert "findCanonicalHubResourceId(query, hubIds)" in model_view
    assert "findCanonicalHubResourceId(query, hubResultIds)" in model_selector
    assert "hubTrainingModelCandidate(canonicalId," in model_selector
    assert "pick(\n      canonicalId," in model_selector
    assert "const canonicalId = cached?.repoId ?? id.trim();" in dataset_selector
    assert "hubResourceIdsEqual(candidate.id, query)" in dataset_selector


def test_new_model_selection_replaces_the_previous_model_type():
    inference = _read("features/training/lib/model-type-inference.ts")
    selector = _read(
        "features/model-picker/components/train-model-selector.tsx"
    )

    assert "export function inferTrainingModelTypeFromFlags" in inference
    assert "resolvePickerInferredModelType" not in inference
    assert "inferTrainingModelTypeFromFlags(inferredFlags)" in selector
    assert "modelType: s.modelType" not in selector


def test_picker_shell_handles_ime_focus_and_short_viewports():
    shell = _read("components/resource-picker/picker-shell.tsx")

    assert "function isImeCompositionKey(" in shell
    assert "event.nativeEvent.isComposing" in shell
    assert "event.keyCode === 229" in shell
    assert shell.count("isImeCompositionKey(") >= 3
    assert "onCompositionStart" in shell
    assert "onCompositionEnd" in shell
    assert 'aria-label={t("picker.searchAriaLabel", { noun })}' in shell
    assert "max-h-(--radix-popover-content-available-height)" in shell
    assert '"mt-2.5 flex min-h-0 flex-1 flex-col gap-2"' in shell
    assert '"min-h-0 max-h-[320px] flex-1 overflow-y-auto' in shell
    assert "function switchToDevice()" in shell
    assert "window.requestAnimationFrame" in shell
    assert "onSwitchDevice={switchToDevice}" in shell
