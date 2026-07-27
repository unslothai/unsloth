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
    return path.read_text()


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
    auto_load = src.split("async function autoLoadOnDeviceModel", 1)[1]
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


# ---------------------------------------------------------------------------
# Send-with-no-model auto-load (issue #7374): on-device discovery must cover
# every picker inventory source, the remembered model must survive local
# (non-cache) loads, and the send path must never start a remote download.
# ---------------------------------------------------------------------------


def _autoload_section() -> str:
    src = _read("features/chat/api/chat-adapter.ts")
    return src.split("async function autoLoadOnDeviceModel", 1)[1]


def test_send_path_cannot_reach_hardcoded_default_download():
    """Pressing Send with no model loaded must never fetch the hard-coded
    default repo from Hugging Face (the unconsented download in the bug
    report). Any recommended download must stay an explicit user action."""
    src = _read("features/chat/api/chat-adapter.ts")
    assert "Qwen3.5-4B-MTP-GGUF" not in src
    assert "Downloading a small model" not in src
    assert "No downloaded models found" not in src
    # The old entry point must not linger anywhere.
    assert "autoLoadSmallestModel" not in src
    # The renamed entry point runs exactly once per send, so the submitted
    # prompt executes exactly once after a successful load.
    assert src.count("await autoLoadOnDeviceModel())") == 1


def test_autoload_no_model_error_is_actionable():
    """With no valid on-device candidate the user is told to select or
    explicitly download a model instead of getting a silent remote load."""
    src = _read("features/chat/api/chat-adapter.ts")
    assert "Select a model in the top bar, or download one from the Hub, then retry." in src


def test_autoload_inventory_failure_is_not_empty_inventory():
    """A failed cached/local inventory request must stop the automatic
    selection path, not be swallowed into an empty list that used to fall
    through to the remote default download."""
    src = _read("features/chat/api/chat-adapter.ts")
    assert ".catch(() => [])" not in src
    auto_load = _autoload_section()
    assert "inventoryErrorSurfaced: true" in auto_load
    # All three inventory sources are queried together and fail closed.
    for needle in (
        "listCachedGguf(hfToken)",
        "listCachedModels(hfToken)",
        "listLocalModels()",
    ):
        assert needle in auto_load, needle


def test_autoload_uses_unified_backend_inventory():
    """Auto-load must consume the same non-React backend inventory the
    unified picker uses (no second frontend filesystem scanner), covering
    the models dir, LM Studio dirs, and custom scan folders."""
    src = _read("features/chat/api/chat-adapter.ts")
    assert re.search(
        r'import \{[^}]*listLocalModels[^}]*\} from "@/features/hub/inventory/api"',
        src,
        re.S,
    )
    sources = re.search(r"const AUTO_LOAD_LOCAL_SOURCES[^;]*;", src, re.S)
    assert sources, "AUTO_LOAD_LOCAL_SOURCES not found"
    for source in ('"models_dir"', '"lmstudio"', '"custom"'):
        assert source in sources.group(0), source


def test_autoload_filters_match_picker_policy():
    """Only complete, chat-capable, non-hidden rows may auto-load: partial
    downloads, weightless/non-chat folders, and infrastructure models are
    excluded with the same policy the picker applies."""
    src = _read("features/chat/api/chat-adapter.ts")
    local_fn = src.split("function isAutoLoadableLocalRow", 1)[1]
    local_fn = local_fn.split("\nfunction ", 1)[0]
    assert "row.capabilities?.can_chat !== true" in local_fn
    assert "row.partial" in local_fn
    # Adapters resolve their base model on load; a Hub-id base would start
    # the implicit remote fetch a background auto-load must never trigger.
    assert 'row.model_format === "adapter"' in local_fn
    assert "isHiddenModelId(row.model_id, row.id, row.path)" in local_fn
    assert "hasBigEndianGgufMarker(row.path, row.format_variant)" in local_fn
    cached_fn = src.split("function isAutoLoadableCachedRepo", 1)[1]
    cached_fn = cached_fn.split("\nconst ", 1)[0]
    assert "repo.partial" in cached_fn
    # Cached adapter repos are chat-capable too and resolve a base model on
    # load, so they must be excluded exactly like local adapter rows.
    assert 'repo.model_format === "adapter"' in cached_fn
    assert "repo.capabilities?.can_chat === false" in cached_fn
    assert "isHiddenModelId(repo.repo_id)" in cached_fn


def test_autoload_local_rows_load_via_backend_target():
    """Indexed local rows (models dir, LM Studio, custom scan folders) must
    load through the backend-provided target and record their stable
    inventory identity, never a reconstructed path or synthetic variant."""
    src = _read("features/chat/api/chat-adapter.ts")
    assert "function localRowLoadTarget" in src
    assert "row.load_id || row.id" in src
    candidate_fn = src.split("function localRowToCandidate", 1)[1]
    candidate_fn = candidate_fn.split("\n/**", 1)[0]
    assert "loadId: localRowLoadTarget(row)" in candidate_fn
    assert "inventoryId: row.inventory_id ?? null" in candidate_fn
    # Inactive-cache rows keep loading by their backend load_id.
    auto_load = _autoload_section()
    assert auto_load.count("loadId: repo.load_id") >= 3


def test_autoload_remembers_last_model_across_all_sources():
    """The remembered model resolves against managed caches AND the indexed
    local inventory; a stale entry only falls through to other on-device
    candidates (there is no remote branch left to reach)."""
    auto_load = _autoload_section()
    assert "isManagedCacheSource(lastLoaded.source)" in auto_load
    assert "matchesRememberedLocalRow(candidateRow, lastLoaded)" in auto_load
    assert "await resolveLocalRowCandidate(" in auto_load
    assert "lastLoaded.ggufVariant," in auto_load
    # Managed-cache candidates record their provenance for later resolution.
    assert auto_load.count('source: "hf_cache"') >= 4


def test_autoload_deduplicates_cached_and_local_candidates():
    """Candidates resolving to the same load target (e.g. a custom scan
    folder pointing into an HF cache) must not be tried twice, but the
    dedupe must key on actual load targets/paths only: a local copy that
    merely shares a repo model_id is a distinct set of files and must stay
    available when the cached copy fails or has no usable quant."""
    auto_load = _autoload_section()
    assert "const seenLoadTargets = new Set<string>()" in auto_load
    # Keys carry the model kind: a folder emitting both GGUF and safetensors
    # rows shares a path while holding two different models.
    assert "seenLoadTargets.add(`${kind}:${normalizeLoadTargetKey(value)}`)" in auto_load
    assert 'markSeen("gguf", repo.load_id || repo.repo_id, repo.cache_path)' in auto_load
    assert 'markSeen("model", repo.load_id || repo.repo_id, repo.cache_path)' in auto_load
    assert "isSeen(localCandidate.kind, row.load_id, row.id, row.path)" in auto_load
    # The repo-id-based dedupe that shadowed distinct local copies is gone.
    assert "isSeen(row.load_id, row.id, row.path, row.model_id)" not in auto_load
    assert "markSeen(repo.repo_id," not in auto_load


def test_local_quant_resolution_skips_failed_quants():
    """When a folder's smallest quant already failed or was blocked, the
    resolver must return the next complete quant instead of abandoning the
    whole folder (which made Send falsely report no model)."""
    src = _read("features/chat/api/chat-adapter.ts")
    resolve_fn = src.split("async function resolveLocalRowCandidate", 1)[1]
    resolve_fn = resolve_fn.split("\nfunction ", 1)[0]
    assert "for (const entry of downloaded)" in resolve_fn
    assert "if (isSkippedCandidate?.(candidate)) continue;" in resolve_fn
    # The fallback loop feeds the skip set into resolution.
    auto_load = _autoload_section()
    assert "isSkippedAutoLoadCandidate," in auto_load


def test_autoload_trust_guard_still_blocks_background_loads():
    """A model needing custom-code approval or a security review is never
    silently auto-loaded, and a blocked candidate can only cascade to other
    on-device candidates."""
    auto_load = _autoload_section()
    assert "validation.requires_trust_remote_code" in auto_load
    assert "validation.requires_security_review" in auto_load
    assert "MAX_AUTO_LOAD_ATTEMPTS" in auto_load


def test_remembered_model_record_supports_local_sources():
    """last-local-model-load must represent managed-cache models AND
    backend-indexed local models: a local GGUF is valid with a null variant,
    legacy v1 records keep resolving as managed-cache entries, and no
    secrets (tokens/leases) are ever persisted."""
    src = _read("features/chat/utils/last-local-model-load.ts")
    # Same storage key: v1 records parse backward-compatibly, no migration.
    assert 'const STORAGE_KEY = "unsloth.last-local-model-load.v1";' in src
    # Legacy records carry no source and default to the managed cache.
    assert "isLastLocalModelSource(parsed.source)" in src
    assert ': "hf_cache";' in src
    # The GGUF-variant requirement is scoped to managed-cache records; a
    # local GGUF's load target identifies the file, so null stays valid.
    assert src.count('source === "hf_cache" && !ggufVariant') == 2
    # Indexed local scan sources are representable.
    for source in ('"models_dir"', '"lmstudio"', '"custom"'):
        assert source in src, source
    # Identity only: never tokens, native path leases, or approvals.
    assert "nativePath" not in src
    assert "hfToken" not in src and "hf_token" not in src
    assert "fingerprint" not in src


def test_interactive_local_loads_are_remembered_without_lease_bypass():
    """A successful interactive load of a backend-indexed local model
    (picker source "local") must be remembered so auto-load can reuse it,
    while native-picked files (signed, expiring path lease) and other
    arbitrary paths must never be recorded."""
    src = _read("features/chat/hooks/use-chat-model-runtime.ts")
    record_block = src.split("const indexedLocalSelection", 1)[1]
    record_block = record_block.split("} catch (error) {", 1)[0]
    assert 'selection.source === "local"' in src
    assert "!nativePathToken &&" in record_block
    assert "(indexedLocalSelection || !isLocalModelPath(modelId))" in record_block
    assert 'source: "local",' in record_block


def test_remembered_local_row_match_requires_kind_agreement():
    """A folder holding both GGUF and safetensors weights yields two inventory
    rows with the same path/load target, so the remembered kind must gate the
    identifier match or a remembered safetensors load can resolve to the GGUF
    row (and vice versa)."""
    src = _read("features/chat/api/chat-adapter.ts")
    match_fn = src.split("function matchesRememberedLocalRow", 1)[1]
    match_fn = match_fn.split("\nasync function ", 1)[0].split("\nfunction ", 1)[0]
    assert '(row.model_format === "gguf") !== (remembered.kind === "gguf")' in match_fn


def test_directory_gguf_rows_resolve_variant_like_picker():
    """Directory-based local GGUFs (LM Studio, models dir, custom folders) are
    flagged requires_variant by the backend, so the fallback must resolve a
    quant through the variants API (as the picker card does) instead of
    silently dropping every directory row; non-GGUF variant-requiring rows
    have no background resolution and stay excluded."""
    src = _read("features/chat/api/chat-adapter.ts")
    resolve_fn = src.split("async function resolveLocalRowCandidate", 1)[1]
    resolve_fn = resolve_fn.split("\nfunction ", 1)[0]
    assert "row.capabilities?.requires_variant === true" in resolve_fn
    assert "if (!isGguf) return null;" in resolve_fn
    # Quants must be resolved from the folder the row will load from, not
    # from a same-id HF cache repo whose quants may be absent locally.
    assert "const variantScanTarget = isLocalModelPath(row.id) ? row.id : row.path;" in resolve_fn
    assert "listGgufVariants(variantScanTarget" in resolve_fn
    assert "localPath: row.path" in resolve_fn
    assert "entry.downloaded && !entry.partial && isAutoLoadableGgufVariant(entry)" in resolve_fn
    # The cascade must keep directory GGUF rows as candidates.
    auto_load = src.split("async function autoLoadOnDeviceModel", 1)[1]
    assert 'row.model_format === "gguf" ||' in auto_load
    assert "await resolveLocalRowCandidate(" in auto_load


def test_remembered_local_failure_does_not_block_folder_fallback():
    """A failed remembered model must exclude only that exact candidate key,
    not mark the whole row or repo as seen; otherwise a folder or cache repo
    with another complete quant can never fall back and Send falsely reports
    no model. Applies to local rows and managed-cache repos alike."""
    src = _read("features/chat/api/chat-adapter.ts")
    auto_load = src.split("async function autoLoadOnDeviceModel", 1)[1]
    remembered_block = auto_load.split("if (lastLoaded) {", 1)[1]
    remembered_block = remembered_block.split("// On-device fallback", 1)[0]
    assert (
        "markSeen(" not in remembered_block
    ), "remembered paths must not pre-mark their row/repo as deduped"
    assert "autoLoadSkipKey(rememberedCandidate)" in remembered_block


def test_fallback_orders_by_resolved_quant_size():
    """A GGUF folder or cache repo row's size_bytes sums every quant in it, so
    the smallest-first fallback must order both local rows and cached repos by
    the resolved quant's own size; otherwise a repo holding one small quant
    loses to a larger single-quant model."""
    src = _read("features/chat/api/chat-adapter.ts")
    resolve_fn = src.split("async function resolveLocalRowCandidate", 1)[1]
    resolve_fn = resolve_fn.split("\nfunction ", 1)[0]
    assert "sizeBytes: sizeOrUnknownBytes(entry.size_bytes)" in resolve_fn
    auto_load = _autoload_section()
    # Local rows order on the resolved quant size.
    assert "sizeBytes: resolved.sizeBytes" in auto_load
    # Cached GGUF repos order on the resolved quant size too.
    assert "const resolveCachedGgufEntry" in auto_load
    assert "sizeBytes: sizeOrUnknownBytes(variant.size_bytes)" in auto_load
    # The all-variant row sum only orders non-GGUF cached repos, whose
    # snapshot loads whole.
    seed_block = auto_load.split("for (const repo of platform.chatOnly ? [] : modelRepos)", 1)[1]
    seed_block = seed_block.split("const resolveCachedGgufEntry", 1)[0]
    assert "sizeOrUnknownBytes(repo.size_bytes)" in seed_block
    assert auto_load.count("sizeOrUnknownBytes(repo.size_bytes)") == 1


def test_cascade_retries_next_quant_after_load_failure():
    """A failed /api/inference/load (not just a blocked validation) must mark
    that quant skipped and re-enter the folder's or repo's next complete quant
    into the GLOBAL size order (still ahead of the safetensors group) instead
    of retrying inline, so one folder of failing quants cannot starve a
    smaller model elsewhere; single-candidate rows resolve to null once
    skipped, and the attempt cap bounds total loads."""
    src = _read("features/chat/api/chat-adapter.ts")
    auto_load = src.split("async function autoLoadOnDeviceModel", 1)[1]
    # No inline retry loop: retries re-enter the shared ordered pool.
    assert "while (localCandidate" not in auto_load
    assert "retry: true," in auto_load
    assert "insertReady({ ...next, retry: true })" in auto_load
    # Ordered insertion respects the GGUF-before-safetensors group boundary
    # and the ascending size order among the remaining candidates.
    assert "!isModelKindEntry(readyPool[at])" in auto_load
    assert "readyPool[at].sizeBytes <= entry.sizeBytes" in auto_load
    # Requeued entries bypass the seen gate; fresh rows still dedupe.
    assert "if (!candidate.retry) {" in auto_load
    # The cascade catch records the failed quant before requeueing.
    assert "skippedAutoLoadCandidates.add(skipKey)" in auto_load
    assert "skippedAutoLoadCandidates.add(autoLoadSkipKey(localCandidate))" in auto_load
    # Termination guard: a skipped single candidate resolves to null.
    resolve_fn = src.split("async function resolveLocalRowCandidate", 1)[1]
    resolve_fn = resolve_fn.split("\nfunction ", 1)[0]
    assert "if (isSkippedCandidate?.(candidate)) return null;" in resolve_fn


def test_cached_rows_deduped_against_local_aliases():
    """A cached repo and an indexed local row can alias the same files (e.g.
    a scan folder pointing into an HF cache). The fallback must not spend a
    second load attempt re-trying files already visited or failed through the
    other row: cached branches apply the same seen gate local rows use, and
    skip keys are scoped to the backend load target both rows share."""
    src = _read("features/chat/api/chat-adapter.ts")
    auto_load = src.split("async function autoLoadOnDeviceModel", 1)[1]
    assert 'if (isSeen("gguf", repo.load_id || repo.repo_id, repo.cache_path))' in auto_load
    assert 'if (isSeen("model", repo.load_id || repo.repo_id, repo.cache_path))' in auto_load
    key_fn = src.split("function autoLoadSkipKey", 1)[1]
    key_fn = key_fn.split("\nfunction ", 1)[0]
    assert "candidate.loadId ?? candidate.id" in key_fn


def test_send_not_blocked_by_full_inventory_resolution():
    """Pressing Send must not wait for every /gguf-variants folder scan before
    the first load attempt: candidates resolve through a bounded worker pool
    and are consumed incrementally after a short settle grace, so one slow
    folder cannot stall the send path behind the transport timeout."""
    src = _read("features/chat/api/chat-adapter.ts")
    assert "const AUTO_LOAD_RESOLVE_GRACE_MS" in src
    auto_load = src.split("async function autoLoadOnDeviceModel", 1)[1]
    # The consumer never awaits full resolution; it waits for the grace
    # window (cut short when resolution finishes) and then per-completion.
    assert "await resolutionDone" not in auto_load
    assert "clearTimeout(graceTimer)" in auto_load
    assert "await nextProgress();" in auto_load
    assert "if (pendingJobs <= 0) {" in auto_load
    # Rows that need no backend scan seed the pool before the workers start,
    # so they can never queue behind slow folder scans.
    assert "const needsVariantScan" in auto_load
    assert ".filter((row) => !needsVariantScan(row))" in auto_load
    assert auto_load.index(".filter((row) => !needsVariantScan(row))") < auto_load.index(
        "const cachedScanJobs"
    )
    # Cached and local scans interleave so one slow source cannot
    # monopolize every worker.
    assert "resolutionJobs.push(cachedScanJobs[jobIndex])" in auto_load
    assert "resolutionJobs.push(localScanJobs[jobIndex])" in auto_load


def test_pending_gguf_scans_gate_safetensors_candidates():
    """GGUF-first is the documented preference order, and incremental
    consumption must not let an instantly-resolved safetensors row claim a
    load slot while a pending folder scan can still yield a GGUF candidate;
    resolved GGUF entries are never gated."""
    src = _read("features/chat/api/chat-adapter.ts")
    auto_load = src.split("async function autoLoadOnDeviceModel", 1)[1]
    assert "if (isModelKindEntry(candidate) && pendingJobs > 0) {" in auto_load


def test_only_first_attempt_leapfrogs_pending_scans():
    """Fast-resolving candidates whose loads fail must not exhaust the attempt
    cap while pending scans can still yield smaller loadable quants: only the
    first (latency-critical) attempt may run ahead of pending scans; once any
    budget is spent, the remaining attempts wait for the settled global
    smallest-first order."""
    src = _read("features/chat/api/chat-adapter.ts")
    auto_load = src.split("async function autoLoadOnDeviceModel", 1)[1]
    assert "if (pendingJobs > 0 && loadAttempts > 0) {" in auto_load


def test_hidden_matcher_fetch_never_blocks_send_unbounded():
    """ensureHiddenModelMatchers has no timeout of its own (unlike the
    30s-bounded inventory calls), so the send path must not await it serially:
    it runs alongside inventory discovery and is only awaited through a short
    grace, after which the static needles filter alone."""
    src = _read("features/chat/api/chat-adapter.ts")
    auto_load = src.split("async function autoLoadOnDeviceModel", 1)[1]
    assert "await ensureHiddenModelMatchers()" not in auto_load
    assert "const hiddenMatchersReady = ensureHiddenModelMatchers().catch(" in auto_load
    assert "const HIDDEN_MATCHERS_GRACE_MS" in src
    assert "setTimeout(resolve, HIDDEN_MATCHERS_GRACE_MS)" in auto_load
    assert "clearTimeout(matcherTimer)" in auto_load


def test_resolution_workers_stop_on_terminal_result():
    """A successful early load (or any other terminal outcome) must stop the
    workers from claiming further folder scans, so autoload cannot leave
    background scans contending with inference for the backend and disk."""
    src = _read("features/chat/api/chat-adapter.ts")
    auto_load = src.split("async function autoLoadOnDeviceModel", 1)[1]
    assert "let resolutionStopped = false;" in auto_load
    assert "while (!resolutionStopped && nextJob < resolutionJobs.length)" in auto_load
    # The flag is set in a finally so every exit path (return, break, throw)
    # stops the workers.
    assert "resolutionStopped = true;" in auto_load
    assert auto_load.index("} finally {") < auto_load.index("resolutionStopped = true;")


def test_local_rows_apply_picker_platform_gate():
    """Chat-only installs run GGUF (any host) and MLX (Mac only); the picker
    hides other local formats and all cached non-GGUF rows there, so the
    background cascade must not load a row the user could not have picked.
    The remembered path stays ungated: a recorded load is user precedent."""
    src = _read("features/chat/api/chat-adapter.ts")
    local_fn = src.split("function isAutoLoadableLocalRow", 1)[1]
    local_fn = local_fn.split("\nfunction ", 1)[0]
    assert "platform.chatOnly" in local_fn
    assert "localRowIsGgufLike(row)" in local_fn
    assert "platform.isMac && localRowIsMlxNamed(row)" in local_fn
    auto_load = src.split("async function autoLoadOnDeviceModel", 1)[1]
    # The platform snapshot hydrates through the cached, non-throwing fetch.
    assert "await fetchDeviceType();" in auto_load
    # Cascade seeding of cached non-GGUF repos mirrors the picker's
    # chat-only exclusion; the remembered lookup above it stays unfiltered.
    assert "platform.chatOnly ? [] : modelRepos" in auto_load


def test_autoload_keys_preserve_posix_path_case():
    """Linux filesystems distinguish /models/Foo from /models/foo, so seen
    keys and remembered-model matching must not fold case on POSIX paths;
    Windows-style paths and Hub repo ids keep case-insensitive matching."""
    src = _read("features/chat/api/chat-adapter.ts")
    norm_fn = src.split("function normalizeLoadTargetKey", 1)[1]
    norm_fn = norm_fn.split("\nfunction ", 1)[0].split("\nconst ", 1)[0]
    assert "looksWindowsPath" in norm_fn
    assert 'value.startsWith("/") || value.startsWith("~")' in norm_fn
    assert "return value;" in norm_fn
    assert "return value.toLowerCase();" in norm_fn
    match_fn = src.split("function matchesRememberedLocalRow", 1)[1]
    match_fn = match_fn.split("\nfunction ", 1)[0]
    assert "normalizeLoadTargetKey" in match_fn
    # Inventory ids compare exactly (same backend generator on both sides).
    assert "row.inventory_id === remembered.inventoryId" in match_fn
    assert "row.inventory_id.toLowerCase()" not in match_fn
    # Skip keys use the same semantics: a failure recorded for /models/Foo
    # must not also skip /models/foo.
    key_fn = src.split("function autoLoadCandidateKey", 1)[1]
    key_fn = key_fn.split("\nfunction ", 1)[0]
    assert "normalizeLoadTargetKey(id)" in key_fn
    assert "id.toLowerCase()" not in key_fn


def test_local_variant_scans_bounded_concurrency():
    """Each /gguf-variants call triggers a recursive backend directory scan,
    so pre-resolution must not fan out unbounded over every indexed folder
    at once."""
    src = _read("features/chat/api/chat-adapter.ts")
    assert "const AUTO_LOAD_VARIANT_SCAN_CONCURRENCY" in src
    auto_load = src.split("async function autoLoadOnDeviceModel", 1)[1]
    assert "Math.min(AUTO_LOAD_VARIANT_SCAN_CONCURRENCY, resolutionJobs.length)" in auto_load
    assert "await Promise.all(\n        cascadeLocalRows.map(" not in auto_load
