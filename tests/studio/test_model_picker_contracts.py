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


def test_chat_autoload_toast_is_persistent_and_dismissible():
    """Send-triggered autoload stays visible until it settles but remains
    dismissible, matching the explicit model-loading toast's lifetime."""
    src = _read("features/chat/api/chat-adapter.ts")
    auto_load = src.split("async function autoLoadSmallestModel", 1)[1]
    auto_load = auto_load.split("export function createOpenAIStreamAdapter", 1)[0]
    assert "toast.loading(" not in auto_load
    assert "const updateAutoLoadToast =" in auto_load
    assert "if (autoLoadToastDismissed) return;" in auto_load
    assert auto_load.count("toast.message(") == 2
    assert auto_load.count("updateAutoLoadToast(") >= 4
    assert "duration: Number.POSITIVE_INFINITY" in auto_load
    assert "closeButton: true" in auto_load
    assert "icon: createLoadingToastIcon()" in auto_load
    assert "onDismiss:" in auto_load
    # Terminal success uses a fresh finite toast after manual progress dismissal.
    assert "showAutoLoadSuccess" in auto_load
    assert "description: undefined" in auto_load
    assert "icon: undefined" in auto_load
    assert "duration: 5000" in auto_load
    assert "duration: 30000" not in auto_load
    assert auto_load.count("toast.dismiss(toastId)") >= 4

    explicit_load = _read("features/chat/hooks/use-chat-model-runtime.ts")
    assert "duration: Infinity" in explicit_load


def test_recipe_model_load_toast_is_persistent_and_dismissible():
    """Recipe model loading uses the same dismissible persistent lifecycle as
    chat loading because both call the non-abortable loadModel API."""
    src = _read("features/recipe-studio/hooks/use-recipe-executions.ts")
    model_load = src.split("async function loadLocalModelSelection", 1)[1]
    model_load = model_load.split("function getLocalModelLoadPlanForPayload", 1)[0]
    assert "toast.loading(" not in model_load
    assert "toast.message(" in model_load
    assert "duration: Number.POSITIVE_INFINITY" in model_load
    assert "closeButton: true" in model_load
    assert "icon: createLoadingToastIcon()" in model_load
    assert "onDismiss:" in model_load
    assert "description: undefined" in model_load
    assert "icon: undefined" in model_load
    assert "duration: 2000" in model_load

    toast_lib = _read("lib/toast.ts")
    assert "createElement(Spinner" in toast_lib
    assert 'className: "size-4 text-muted-foreground"' in toast_lib

    sonner = _read("components/ui/sonner.tsx")
    assert "loading: createLoadingToastIcon()" in sonner


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
    # The GPU knobs are part of the editor's instance key, so a reload that lands
    # on different placement re-seeds the editor instead of leaving it on the old
    # values. Shared by every host that mounts ModelConfigPage.
    shared = _read("features/model-picker/model-config/config-signature.ts")
    assert "export function gpuFieldsSignature" in shared
    assert "gpuFieldsSignature(config)," in shared
    assert "export function modelConfigInstanceKey" in shared
    for rel in (
        "features/model-picker/components/sidebar-model-config.tsx",
        "features/hub/catalog/hub-model-settings-view.tsx",
    ):
        assert "modelConfigInstanceKey(" in _read(rel), rel
    # apply-per-model-config re-exports it, so its own callers are unchanged.
    reexport = _read("features/model-picker/model-config/apply-per-model-config.ts")
    assert "export { gpuFieldsSignature };" in reexport


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


def test_parallel_slots_setting_wired_end_to_end():
    """The per-load Parallel Slots knob (llama-server --parallel) must flow from
    the run-settings form through persistence, every /load builder, the validate
    preflight and the cross-model reset; a lost hop silently reverts the model to
    the server-wide slot default."""
    config = _read("features/model-picker/model-config/per-model-config.ts")
    # Persisted per model, clamped on every read/write, and null (= server
    # default) counts as default so blank configs are not stored.
    assert '"nParallel",' in config
    assert "N_PARALLEL_MAX, Math.round(partial.nParallel)" in config
    assert "config.nParallel == null &&" in config
    page = _read("features/model-picker/components/model-config-page.tsx")
    # Rendered in the GGUF advanced section, which a remembered override reopens.
    assert "Parallel Slots" in page
    assert "config.nParallel != null ||" in page
    assert 'aria-label="Parallel decode slots"' in page
    api_types = _read("features/chat/types/api.ts")
    assert "n_parallel?: number | null;" in api_types
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    # Click-time snapshot, /load body, validate preflight, cross-model reset and
    # failed-switch rollback all carry the value.
    assert "pendingLoadConfig?.nParallel" in runtime
    # GGUF-gated, like the compare pane: a transformers load has no slots.
    assert "n_parallel: isGguf ? loadNParallel : null," in runtime
    assert "n_parallel: validateNParallel," in runtime
    assert "loadNParallel = pendingLoadConfig?.nParallel ?? null;" in runtime
    assert "n_parallel: stateBeforeUnload.loadedNParallel," in runtime
    chat_api = _read("features/chat/api/chat-api.ts")
    assert "n_parallel: payload.n_parallel," in chat_api
    composer = _read("features/chat/shared-composer.tsx")
    # The compare pane is a second /load builder; its preflight sizes like its load.
    assert composer.count("n_parallel: ownConfig.nParallel ?? null,") == 2
    adapter = _read("features/chat/api/chat-adapter.ts")
    # The startup auto-load is a third builder reading the remembered config.
    assert adapter.count("n_parallel: config.nParallel ?? null,") == 2
    # ... and records it as loaded through the diffusion-gated local below.
    assert "loadedNParallel: committedSlots," in adapter
    status = _read("features/chat/lib/apply-inference-status-to-store.ts")
    # Hydration seeds the rollback BASELINE only; adopting the resolved echo into
    # the control would pin a blank "server default" to a number.
    assert "loadedNParallel: status.requested_parallel_slots," in status
    assert "nParallel: status.requested_parallel_slots," not in status
    # The sidebar form remounts when an external change lands. The signature it
    # keys on is shared with the hub and the model config page now, so the slot
    # count has to be in that one definition rather than the sidebar's own copy.
    signature = _read("features/model-picker/model-config/config-signature.ts")
    assert 'config.nParallel ?? "",' in signature
    sidebar = " ".join(_read("features/model-picker/components/sidebar-model-config.tsx").split())
    assert "key={modelConfigInstanceKey(modelId, settingsGgufVariant, loadedConfig)}" in sidebar


def test_parallel_slots_reach_an_api_load_through_the_server_mirror():
    """The server mirror is the hop an OpenAI-compatible auto-switch load reads,
    and it is the browser's only way to express a per-model setting to a load no
    browser makes. A slot count missing from it silently reverts that load to the
    server-wide --parallel default, and llama_extra_args cannot stand in because
    --parallel is denylisted. A config whose only change is the slot count also
    serializes to an empty payload, so the one-time backfill sends nothing and
    still marks itself done."""
    api = " ".join(_read("features/model-picker/api/model-overrides.ts").split())
    assert "n_parallel?: number;" in api
    assert (
        "if (config.nParallel && config.nParallel > 0) { payload.n_parallel = config.nParallel; }"
        in api
    )
    # The monitor lists what a remote load applies, so an entry holding only slots
    # must not read as "App defaults".
    monitor = " ".join(_read("features/api-monitor/components/saved-model-settings.tsx").split())
    assert "if (override.n_parallel) {" in monitor

    route = (WORKDIR / "studio" / "backend" / "routes" / "settings.py").read_text(encoding = "utf-8")
    assert "n_parallel: Optional[int] = Field(" in route
    assert "n_parallel = payload.n_parallel," in route
    store = (WORKDIR / "studio" / "backend" / "utils" / "openai_auto_switch_settings.py").read_text(
        encoding = "utf-8"
    )
    assert 'entry["n_parallel"] = n_parallel' in store
    # GGUF-only, like the picker: a safetensors load has no llama-server slots.
    gguf_block = store.split("    if is_gguf:", 1)[1]
    assert 'kwargs["n_parallel"] = override["n_parallel"]' in gguf_block


def test_parallel_slots_control_cleared_when_the_load_never_sent_them():
    """`nParallel` is the editable control ("blank = follow the server default")
    and `loadedNParallel` the rollback baseline. A success path that sends no
    slot count must blank the control, or a value staged for another model shows
    as applied, is persisted into this model's config (`isDefaultConfig` keys on
    nParallel) and is re-sent by the next Apply. Each assertion below is the only
    thing pinning one such path."""
    status = " ".join(_read("features/chat/lib/apply-inference-status-to-store.ts").split())
    # A model/variant swap underneath this tab must reset the control like
    # performLoad's cross-model reset, or model A's count follows onto model B.
    # Narrowly gated -- see test_hydration_keeps_the_slot_control_when_readopting_the_running_model.
    assert "...(seedLoadParams && slotsModelChanged && { nParallel: null })," in status
    # ... while still never adopting the RESOLVED echo into the control.
    assert "nParallel: status.requested_parallel_slots," not in status

    adapter = _read("features/chat/api/chat-adapter.ts")
    # Slice the two success branches apart, bounding the second at the shared tail
    # so it cannot swallow the fresh-default path below and stay green.
    candidate = adapter.split("async function loadAutoLoadCandidate", 1)[1]
    gguf_branch, non_gguf_rest = candidate.split('if (candidate.kind === "gguf") {', 1)[1].split(
        "\n    } else {\n", 1
    )
    non_gguf_branch = non_gguf_rest.split("if (!(loadResp.is_lora ?? false)) {", 1)[0]
    # The cached-GGUF branch keeps the remembered override via the gated local...
    assert "nParallel: committedSlots," in gguf_branch
    assert "nParallel: null," not in gguf_branch
    # ... the safetensors fallback sends no slots, so it clears both, or the count
    # survives on a model whose form does not even render the field.
    assert "nParallel: null," in non_gguf_branch
    assert "loadedNParallel: null," in non_gguf_branch

    fresh_default = adapter.split("No downloaded models found. Fetching", 1)[1].split(
        'showAutoLoadSuccess("Loaded Qwen', 1
    )[0]
    # The fresh-default download omits the slots, so its success state clears both,
    # or the control reads as an unapplied edit against the seeded baseline.
    assert "n_parallel" not in fresh_default.split("saveSpeculativeType", 1)[0]
    assert "nParallel: null," in fresh_default
    assert "loadedNParallel: null," in fresh_default


def test_hydration_clears_the_slot_baseline_for_a_slotless_model():
    """The baseline is what a rollback re-sends and what preset capture reads, so
    a model that cannot have slots must not inherit the previous GGUF's count.
    /status omits the echo for non-GGUF and sends an explicit null for diffusion;
    an absent field on a GGUF is an older backend and must NOT wipe it."""
    src = _read("features/chat/lib/apply-inference-status-to-store.ts")
    assert (
        "(status.is_gguf === false || status.requested_parallel_slots === null) && {" in src
    ), "the slotless clear must key on is_gguf or an explicit null echo"
    clear = src.index("status.is_gguf === false || status.requested_parallel_slots === null")
    assert "loadedNParallel: null," in src[clear : clear + 200]
    # Never `!= null`: that also matches the absent field an older backend sends.
    assert "status.requested_parallel_slots !== null && {" not in src


def test_hydration_keeps_the_slot_control_when_readopting_the_running_model():
    """`hydratingExistingModel` is true whenever the incoming status disagrees
    with what this tab last recorded, which includes RE-ADOPTING a model the tab
    never lost: the resident-adopt branch restores the model's own per-model
    config and only then hydrates, passing the EXTERNAL id as
    `previousCheckpoint`. An ungated clear there wipes the slot count that branch
    just restored, and the blank persists into `savePerModelConfig`, so a Save
    the user reads as a no-op erases their remembered override.

    Only that branch knows the model is unchanged, so it says so explicitly.
    Slot counts cannot stand in: the echo falls back to the server-wide default,
    so a genuine A->B swap can echo exactly A's explicit count."""
    status = " ".join(_read("features/chat/lib/apply-inference-status-to-store.ts").split())
    assert (
        "const slotsModelChanged = hydratingExistingModel && !options.readoptingSameModel;"
        in status
    )
    assert "...(seedLoadParams && slotsModelChanged && { nParallel: null })," in status
    # Never a slot-count proxy for "same model".
    assert "prevState.loadedNParallel === (status.requested_parallel_slots" not in status
    # The baseline seed stays ungated, or a rollback after a tab reload restores
    # the model at the server default slots.
    assert "loadedNParallel: status.requested_parallel_slots," in status

    runtime = " ".join(_read("features/chat/hooks/use-chat-model-runtime.ts").split())
    resident = runtime.split("if (!forceReload && isExternalModelId(selectedCheckpoint)) {", 1)[
        1
    ].split("const stopDecision", 1)[0]
    # What makes the scenario reachable: the branch restores the model's own
    # config, then hydrates against the external id.
    assert "applyPerModelConfigToRuntime(selection.previousConfig);" in resident
    assert "previousCheckpoint: selectedCheckpoint," in resident
    # Only reachable because the branch matched the id AND the variant first.
    assert "resolveInferenceCheckpointId(residentStatus) === modelId" in resident
    assert "readoptingSameModel: true," in resident
    # The refresh() hydrate must NOT claim it: there the model really can change.
    poll = runtime.split("setModels(listRes.models.map(toChatModelSummary));", 1)[1].split(
        "} else if (!statusRes.active_model", 1
    )[0]
    assert "applyActiveModelStatusToStore(statusRes, {" in poll
    assert "readoptingSameModel" not in poll


def test_parallel_slots_are_never_recorded_for_a_diffusion_load():
    """A DiffusionGemma GGUF answers ``is_gguf: true``, but its runner ignores
    ``--parallel``, so ``_parallel_slot_echo`` reports null slots for it. The
    three load success paths must gate on ``is_diffusion`` too, or they record a
    click-time count the load never committed.

    That phantom does not stay put: ``capturePresetLoadConfig`` snapshots
    ``nParallel`` with no model gate and a preset carries no model identity, so
    applying it over a TEXT GGUF sends the count as a real ``n_parallel``.
    """
    runtime = " ".join(_read("features/chat/hooks/use-chat-model-runtime.ts").split())
    # One gated local feeds the control and the baseline, so they cannot drift.
    assert "(loadResponse.is_gguf ?? false) && !(loadResponse.is_diffusion ?? false)" in runtime
    assert "nParallel: committedSlots," in runtime
    assert "loadedNParallel: committedSlots," in runtime

    adapter = " ".join(_read("features/chat/api/chat-adapter.ts").split())
    assert (
        "const committedSlots = (loadResp.is_diffusion ?? false) ? null "
        ": (config.nParallel ?? null);" in adapter
    )
    assert "nParallel: committedSlots," in adapter
    assert "loadedNParallel: committedSlots," in adapter

    composer = " ".join(_read("features/chat/shared-composer.tsx").split())
    assert "targetIsGguf && !(resp.is_diffusion ?? false)" in composer
    assert "nParallel: committedSlots," in composer
    assert "loadedNParallel: committedSlots," in composer


def test_hydration_restores_a_remembered_slot_override():
    """The control is never seeded from the status echo, so a model running on a
    remembered override shows a BLANK slot control after a browser reload or a
    tab move to another GGUF. `ModelConfigPage.resolveInitial` prefers the live
    store for the active model, so that blank is what the form edits: the next
    Apply reloads at the server default and a Save writes the blank over the
    remembered count.

    The seed is deliberately narrow: storage is read only on a fresh store or a
    model change, never on a steady poll, and the value is adopted only when the
    server already runs that exact count, which proves it is this model's own.
    """
    src = _read("features/chat/lib/apply-inference-status-to-store.ts")
    status = " ".join(src.split())
    assert (
        "resolveInitialConfig(checkpointId, status.gguf_variant ?? null)" in status
    ), "the remembered override comes from per-model storage, not the echo"
    assert (
        "const slotsUnseeded = prevState.loadedNParallel === null && "
        "prevState.nParallel === null;" in status
    )
    assert (
        "status.is_gguf && (slotsUnseeded || slotsModelChanged)" in status
    ), "storage is read on a fresh store or a model change, never on a steady poll"
    assert (
        "...(seedLoadParams && (slotsUnseeded || slotsModelChanged) &&" in status
    ), "the seed fires in both cases the clear leaves the control blank"
    assert (
        "rememberedNParallel != null && rememberedNParallel === "
        "status.requested_parallel_slots && { nParallel: rememberedNParallel, }" in status
    )
    # Both cases trip the model-change clear, so the seed only survives by
    # being spread after it.
    assert src.index("slotsModelChanged && { nParallel: null }") < src.index(
        "nParallel: rememberedNParallel,"
    )


def test_failed_switch_rollback_restores_the_slot_intent_not_the_resolved_count():
    """`loadedNParallel` holds a RESOLVED count even for a load that sent no
    slots (the echo falls back to the server-wide default), so it is the right
    value to re-send when recreating the previous server and the wrong one to put
    back in the control: it turns "follow the server default" into an explicit
    override that a later Save or preset capture pins. The outer catch only
    repairs that for a staged config, so a plain string pick keeps the phantom.

    The intent comes from the picker's own pre-switch snapshot when there is one:
    chat-page pre-applies the TARGET's config before calling selectModel, so the
    live control describes the outgoing model only for a bare pick."""
    runtime = " ".join(_read("features/chat/hooks/use-chat-model-runtime.ts").split())
    assert (
        'const previousNParallel = typeof selection !== "string" && '
        "selection.previousConfig ? (selection.previousConfig.nParallel ?? null) "
        ": useChatRuntimeStore.getState().nParallel;" in runtime
    )
    assert runtime.index("const previousNParallel") < runtime.index(
        "applyPerModelConfigToRuntime(pendingLoadConfig);"
    ), "a config staged on the selection must not replace it either"
    picker = " ".join(_read("features/chat/chat-page.tsx").split())
    assert (
        "const previousConfig = currentRuntimePerModelConfig({ includeMaxSeqLength: true, }); "
        "const hasAppliedConfig = applyModelLoadConfigToRuntime(" in picker
    ), "the snapshot must be taken before the target's config is applied"
    rollback = runtime.split("const rollbackSpeculativeType", 1)[1]
    assert "nParallel: previousNParallel," in rollback
    # Baseline and reload payload keep the resolved count, or the rollback
    # recreates the previous model at a different slot count.
    assert "loadedNParallel: stateBeforeUnload.loadedNParallel ?? null," in rollback
    assert "n_parallel: stateBeforeUnload.loadedNParallel," in runtime


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


def test_only_gguf_configs_are_mirrored_to_the_server():
    """The server override map is read by the OpenAI-compatible auto-switch, and
    its resolver indexes GGUFs only. Mirroring a safetensors config there would
    advertise settings on the monitor's applied-on-API-load list that no API
    request can ever apply. The local write stays unconditional: the picker
    loads safetensors models and must honour their config.
    """
    src = " ".join(_read("features/model-picker/components/model-config-page.tsx").split())
    assert (
        "if ( !saveFailed && (target.apiLoadable ?? target.isGguf) && !nativePathToken ) "
        "{ syncModelOverride(" in src
    )
    # The local save is not behind the same gate.
    assert "if (remember) { saveFailed = !savePerModelConfig(" in src


def test_a_native_leased_gguf_is_not_mirrored_to_the_server():
    """A dropped or file-picked GGUF loads through a signed native-path lease, and
    /api/inference/status reports model_identifier as null for it, so the checkpoint
    the browser keys settings by is the bare file name the backend echoes back.
    _build_index keys a standalone GGUF by its on-disk path and by its .gguf-stripped
    stem, so that name is never an index key: mirroring it wrote an override no load
    can read, which the monitor's applied-on-API-load list then advertised as live.

    The live save gates on the lease token rather than the name, because the label
    falls back to a plain string with no suffix when the host reports none.
    """
    page = " ".join(_read("features/model-picker/components/model-config-page.tsx").split())
    assert "&& !nativePathToken ) { syncModelOverride(" in page
    assert (
        "const nativePathToken = target.meta.nativePathToken ?? "
        "(isActiveModel ? activeNativePathToken : null);" in page
    ), "the token this gate reads"
    # The one-time backfill has no token to read, so it goes by the identity shape.
    backfill = " ".join(_read("features/model-picker/api/migrate-model-overrides.ts").split())
    assert "!isNativeFileLabel(entry.modelId) &&" in backfill
    identity = " ".join(_read("features/hub/lib/model-identity.ts").split())
    assert "export function isNativeFileLabel(" in identity
    # A bare file name: no separator, and the .gguf the resolver's keys never carry.
    assert "const NATIVE_FILE_LABEL_RE = /^[^/\\\\]+\\.gguf$/i;" in identity

    backend = WORKDIR / "studio" / "backend"
    inference = (backend / "routes" / "inference.py").read_text(encoding = "utf-8")
    assert (
        "model_identifier = None if _native_grant_backed else _model_id" in inference
    ), "why the checkpoint is only a display name"
    models = (backend / "routes" / "models.py").read_text(encoding = "utf-8")
    assert "display_name = gguf_file.stem," in models, "why the name is never an index key"


def test_evicted_local_configs_drop_their_server_overrides():
    """savePerModelConfig evicts older models when the map exceeds its budget.
    Those models keep a server override that API loads still apply, with nothing
    left in the UI showing it or able to forget it, so eviction has to propagate.

    It propagates as a clear, not a Forget: saving one model silently drops the
    oldest OTHER model, and sending the full remove would also take llama_extra_args
    that only the settings API writes and no UI can show or restore.
    """
    src = " ".join(_read("features/model-picker/components/model-config-page.tsx").split())
    assert "const evicted: { modelId: string; ggufVariant: string | null }[] = [];" in src
    assert (
        "for (const dropped of evicted) { syncModelOverride(dropped.modelId, "
        "dropped.ggufVariant, null, { keepLaunchFlags: true, }); }" in src
    )
    api = " ".join(_read("features/model-picker/api/model-overrides.ts").split())
    assert "keepLaunchFlags?: boolean;" in api
    # remove=false with no fields: the route re-supplies the stored flags and drops
    # the row outright once nothing server-owned is left in it.
    assert "remove: config === null && !options?.keepLaunchFlags," in api
    assert (
        "...(config === null && !options?.keepLaunchFlags ? { llama_extra_args: [] } : {}),"
        in api.replace("// biome-ignore lint/style/useNamingConvention: API schema ", "")
    )
    route = (WORKDIR / "studio" / "backend" / "routes" / "settings.py").read_text(encoding = "utf-8")
    assert 'requested_extra_args = stored.get("llama_extra_args")' in route, "the rule this mirrors"

    # The eviction path must report what it dropped, decoded back into a model id
    # and variant rather than the normalized storage key.
    store = " ".join(_read("features/model-picker/model-config/per-model-config.ts").split())
    assert "evicted?: { modelId: string; ggufVariant: string | null }[]" in store
    assert "modelIdFromStorageKey(" in store and "ggufVariantFromStorageKey(" in store


def test_backfill_compares_server_keys_by_normalized_identity():
    """app_settings has no schema version, so an install predating identity
    normalization holds rows keyed by whatever id was typed, e.g.
    "Unsloth/Repo-GGUF:Q4_K_M". This browser only ever stores the folded form,
    so an exact property lookup reports "not on the server" for a row that is,
    and the one-time backfill then overwrites settings it documents as the newer
    authority. The comparison has to fold the same way the backend resolves.
    """
    src = " ".join(_read("features/model-picker/api/migrate-model-overrides.ts").split())
    assert "function normalizedOverrideKey(" in src
    # Folded on both sides: the older `id::variant` local keys are not.
    assert "known.set(normalizedOverrideKey(storedKey), storedEntry);" in src
    assert "const stored = known.get(key);" in src
    # A quant-aware split, so a Windows drive letter is not read as a separator.
    assert "const split = splitQuantSuffix(key);" in src
    # Repo ids fold and POSIX paths do not, which is what these do.
    assert "normalizeModelIdentity(" in src and "normalizeGgufVariantIdentity(" in src


def test_monitor_stats_exclude_model_lifecycle_rows():
    """A load, unload or download is recorded as a monitor entry but is not an
    HTTP call. It reads as "running" for as long as the load takes, so counting
    it reports an in-flight request with no client waiting and folds a
    multi-minute download into "Avg latency". The backend already leaves these
    out of active_count, so counting them here also makes the page disagree with
    the number the API itself reports.
    """
    src = " ".join(_read("features/api-monitor/use-api-monitor.ts").split())
    assert 'if (entry.kind === "lifecycle") { continue; }' in src
    # "Requests" is a request count too, so it cannot stay entries.length.
    assert "total: requests," in src
    assert "total: entries.length" not in src

    backend = (WORKDIR / "studio" / "backend" / "core" / "inference" / "api_monitor.py").read_text(
        encoding = "utf-8"
    )
    assert 'entry.kind != "lifecycle"' in backend, "the rule this mirrors"


def test_api_reach_copy_is_limited_to_gguf_models():
    """The Hub opens this page for every downloaded model, but ModelConfigPage
    mirrors settings to the server only when target.isGguf, because API
    auto-switch indexes GGUFs only. Telling a safetensors user the settings
    apply to an API request describes a load that cannot happen.
    """
    src = " ".join(_read("features/hub/catalog/hub-model-settings-view.tsx").split())
    assert "{(target.apiLoadable ?? target.isGguf)" in src
    assert "Saved settings apply everywhere Studio loads this model." in src


def test_backfill_includes_a_standalone_gguf_with_no_variant():
    """A standalone .gguf picked directly has no quant to choose between, so it
    is stored with a null variant. The quant filter classified it like
    safetensors and skipped it, and since the done flag is set on the same pass
    those settings stayed browser-only permanently while API auto-switch, which
    does resolve that model, kept loading it with defaults.
    """
    src = " ".join(_read("features/model-picker/api/migrate-model-overrides.ts").split())
    assert 'entry.modelId.toLowerCase().endsWith(".gguf")' in src
    # Still excluded for safetensors, which auto-switch does not resolve.
    assert "entry.ggufVariant != null ||" in src


def test_monitor_overlay_does_not_pull_in_the_lazy_page():
    """The overlay is mounted from __root.tsx, so a static import of the page
    for two label helpers drags the whole 900-line page and its dependency
    graph into the eagerly loaded bundle and undoes the route's
    lazyRouteComponent. Measured: the async api-monitor chunk was 0.20 kB with
    the page in the main bundle, and 18.83 kB after the helpers moved, with the
    main bundle 18 kB smaller.
    """
    overlay = _read("features/api-monitor/api-monitor-overlay.tsx")
    assert 'from "./lifecycle"' in overlay
    assert "api-monitor-page" not in overlay, "the overlay must not reach the page"
    # The helpers live in their own module, not re-exported through the page.
    shared = _read("features/api-monitor/lifecycle.ts")
    assert "export function isLifecycleEntry(" in shared
    assert "export function lifecycleLabel(" in shared


def test_override_writes_are_ordered_per_model():
    """Two saves for one model, or a save racing the one-time backfill, started
    independent requests with no sequencing, so the older response could commit
    last and resurrect the entry the newer one meant to replace. An API load
    then applies settings the user has already changed. Different models still
    overlap, so a slow write for one cannot hold up another.
    """
    src = " ".join(_read("features/model-picker/api/model-overrides.ts").split())
    assert "const writesByKey = new Map<string, Promise<void>>();" in src
    # Keyed by the same override key the server stores under.
    # Folded, not literal: the backfill uses a legacy casing and a UI save the
    # normalized one, and the backend resolves both to one row.
    assert (
        "const key = modelOverrideKey( normalizeModelIdentity(modelId), normalizeGgufVariantIdentity(ggufVariant), );"
        in src
    )
    # Chained on the settled tail, so one failed write cannot cancel the next.
    assert "previous .catch(() => {}) .then(() => sendModelOverride(" in src
    # Only the last writer clears the slot, or a queue still building loses order.
    assert "if (writesByKey.get(key) === write) { writesByKey.delete(key); }" in src


def test_backfill_skips_future_schema_local_records():
    """loadPerModelConfig refuses to apply a record written by a newer Studio and
    eviction refuses to drop one, because this client cannot interpret that
    schema. The enumeration the backfill uses had no such guard, so it would
    persist this client's partial reading server-side and an API-triggered load
    would then apply settings the same client will not apply locally.
    """
    src = " ".join(_read("features/model-picker/model-config/per-model-config.ts").split())
    listing = src[src.index("export function listPerModelConfigs()") :]
    assert "storedConfigVersion(raw) > STORAGE_SCHEMA_VERSION" in listing[:900]


def test_detail_settings_need_a_resolved_quant():
    """The on-device card passes a null variant while its own lookup is pending
    or after it failed. Opening the editor then saves a bare-model config, which
    the picker never finds because it matches variants exactly, while the API's
    bare-key fallback would apply it. openModelSettings already refuses; this
    entry point has to refuse the same way."""
    src = " ".join(_read("features/hub/hub-page.tsx").split())
    guard = "if ( !ggufVariant && selectedModel.isGguf && selectedModel.requiresVariant )"
    assert guard in src
    assert src.count("Couldn't determine which quant to configure.") == 2


def test_a_failed_detail_fetch_is_retried():
    """A terminal row's updated_at never advances and selectedIsMissing stays
    true, so a fetch that failed had nothing left to re-run the effect and the
    payload stayed unavailable until another row was selected. The retry is
    bounded because the usual failure is an entry aged out of the ring buffer,
    which never arrives however often it is asked for."""
    src = " ".join(_read("features/api-monitor/api-monitor-page.tsx").split())
    assert "const DETAIL_FETCH_ATTEMPTS = 3;" in src
    assert "const detailInFlight = selectedId_ != null && loadingDetails.has(selectedId_);" in src
    assert "if (attemptsRef.current.count >= DETAIL_FETCH_ATTEMPTS) { return; }" in src


def test_ollama_models_are_not_advertised_as_api_loadable():
    """local_model_resolver skips Ollama's scanner, so an Ollama GGUF is never in
    the auto-switch index and no OpenAI request can resolve it. target.isGguf is
    still true for one, so gating on that alone mirrored settings the API can
    never apply and told the user the opposite."""
    types_src = " ".join(_read("features/model-picker/components/model-selector/types.ts").split())
    assert "apiLoadable?: boolean;" in types_src
    hub = " ".join(_read("features/hub/hub-page.tsx").split())
    assert "row.source !== LOCAL_MODEL_SOURCE.OLLAMA" in hub
    assert "apiLoadable:" in hub
    backend = (
        WORKDIR / "studio" / "backend" / "core" / "inference" / "local_model_resolver.py"
    ).read_text(encoding = "utf-8")
    assert (
        "Ollama's\n    scanner is skipped" in backend or "scanner is skipped" in backend
    ), "the rule this mirrors"


def test_cached_repo_settings_are_keyed_by_the_repo_id():
    """A repo cached outside the active HF cache reports load_id = the snapshot
    path (hub/services/cache_inventory.py), while the chat picker and the
    auto-switch index key it by repo_id. Keying the Hub's settings by the load id
    saved them where no other load looks, so they silently never applied."""
    config_page = " ".join(_read("features/model-picker/components/model-config-page.tsx").split())
    assert "const configId = target.configId ?? target.id;" in config_page
    for call in (
        "resolveInitialConfig(configId, target.ggufVariant)",
        "savePerModelConfig( configId, target.ggufVariant,",
        "deletePerModelConfig(configId, target.ggufVariant)",
        "syncModelOverride( configId, target.ggufVariant,",
    ):
        assert call in config_page, call
    # The probes have to open the model, so they keep the load id.
    assert "useDefaultChatTemplate( target.id," in config_page
    assert "model_path: target.id," in config_page

    hub = " ".join(_read("features/hub/hub-page.tsx").split())
    assert 'if (kind !== "cache") return resource.runId;' in hub
    assert "return resource.repoId ?? resource.runId;" in hub
    # Both openers and the Hub's own load resolve through it.
    assert hub.count("modelConfigIdentity(") == 3
    assert 'const configId = row.kind === "cache" ? row.repoId : id;' in hub
    assert hub.count("configId,") >= 2

    backend = (
        WORKDIR / "studio" / "backend" / "hub" / "tests" / "test_model_services.py"
    ).read_text(encoding = "utf-8")
    assert 'fields["load_id"] == str(snapshot)' in backend, "the rule this mirrors"


def test_backfill_splits_a_quant_suffix_the_way_the_backend_does():
    """The backfill compared server keys under an identity taken by splitting on
    the last colon, so a Windows drive letter and an ordinary colon inside a POSIX
    filename were read as quant separators: `/models/foo:Bar.gguf` and
    `/models/foo:bar.gguf` folded to one key, and whichever was already on the
    server made the other look migrated."""
    identity = " ".join(_read("features/model-picker/model-config/model-identity.ts").split())
    assert "export function splitQuantSuffix(" in identity
    # The two rules that keep a path out: no separator in the tail, and a head
    # that is not a .gguf cannot carry a free-form label.
    assert 'if (tail.includes("/") || tail.includes("\\\\"))' in identity
    assert 'if (!head.toLowerCase().endsWith(".gguf")) { return null; }' in identity
    # A .gguf head is not enough on its own: the suffix has to be the label the
    # scanner derives from that filename, or a name that itself contains ".gguf:"
    # ("/models/llama.gguf:Bar.gguf" and its lowercase sibling, two real POSIX
    # files) folds onto one key and one file's settings never migrate.
    assert "tail.toLowerCase() === ggufQuantLabel(filename).toLowerCase()" in identity

    migrate = " ".join(_read("features/model-picker/api/migrate-model-overrides.ts").split())
    assert "const split = splitQuantSuffix(key);" in migrate
    assert 'key.lastIndexOf(":")' not in migrate, "the unconditional split is gone"

    backend = (
        WORKDIR / "studio" / "backend" / "utils" / "openai_auto_switch_settings.py"
    ).read_text(encoding = "utf-8")
    assert "def split_quant_suffix(" in backend, "the rule this mirrors"
    assert "_BPW_SUFFIX" in backend and "bpw" in identity
    assert "extract_quant_label(filename).casefold()" in backend, "the label rule"
    # Both sides accept the same quant vocabulary; the regex lives with the loader.
    quants = (WORKDIR / "studio" / "backend" / "core" / "inference" / "llama_cpp.py").read_text(
        encoding = "utf-8"
    )
    for token in ("MXFP", "IQ", "TQ", "BF16", "F16", "F32"):
        assert token in quants and token in identity, token
    # The label helpers the .gguf branch leans on are ported too, shard suffix and
    # float-precision fallback included, or the two sides label a filename apart.
    gguf = (WORKDIR / "studio" / "backend" / "hub" / "utils" / "gguf.py").read_text(
        encoding = "utf-8"
    )
    assert "def extract_quant_label(" in gguf and "def _gguf_stem(" in gguf
    assert "function ggufQuantLabel(" in identity and "function ggufStem(" in identity
    assert "_GGUF_SPLIT_SUFFIX_RE" in gguf and "GGUF_SPLIT_SUFFIX" in identity
    assert "_FLOAT_PRECISION_QUANTS" in gguf and "FLOAT_PRECISION_QUANTS" in identity
    # The executable half of this contract, checked case by case against the
    # answers split_quant_suffix gives.
    assert (WORKDIR / "studio" / "frontend" / "tests" / "quant-suffix-split.test.ts").is_file()


def test_the_detail_card_also_gates_ollama_out_of_the_api_promise():
    """Settings opens from two places in the Hub. The row menu gated Ollama out of
    the server mirror and the "API loads use these" copy; the detail card did not,
    so the same model made the same false promise from the other entry point."""
    hub = " ".join(_read("features/hub/hub-page.tsx").split())
    assert hub.count("LOCAL_MODEL_SOURCE.OLLAMA") == 2
    assert "selectedModel.localSource !== LOCAL_MODEL_SOURCE.OLLAMA" in hub
    assert "row.source !== LOCAL_MODEL_SOURCE.OLLAMA" in hub


def test_the_settings_page_judges_the_config_storage_actually_keeps():
    """savePerModelConfig normalizes before deciding, and the runtime hands this
    page Speculative Decoding "auto", which canonicalizes to null. Judging the raw
    object called a default config non-default, so Remember reported saved while
    the local write had dropped the entry, and the mirror sent the server an
    "auto" override the browser did not have."""
    src = " ".join(_read("features/model-picker/components/model-config-page.tsx").split())
    assert (
        "const normalizedRuntimeConfig = normalizePerModelConfig( effectiveRuntimeConfig, );" in src
    )
    assert "const defaultConfig = isDefaultConfig(normalizedRuntimeConfig);" in src
    # The same object goes to storage and to the server, or they disagree again.
    assert "target.ggufVariant, normalizedRuntimeConfig, evicted," in src
    assert "remember ? normalizedRuntimeConfig : null," in src
    assert "isDefaultConfig(effectiveRuntimeConfig)" not in src

    store = " ".join(_read("features/model-picker/model-config/per-model-config.ts").split())
    assert "export function normalizePerModelConfig(" in store
    assert "const normalized = normalize(config);" in store
    assert "if (isDefaultConfig(normalized)) {" in store, "the rule this mirrors"


def test_the_chat_picker_marks_ollama_targets_unloadable_by_the_api():
    """A settings target opened from the Chat model picker carried no apiLoadable,
    so the `?? target.isGguf` fallback mirrored an Ollama GGUF to the server.
    local_model_resolver.py refuses every path under a .studio_links/ollama_links
    link dir, which is exactly how Ollama's blobs reach this picker, so the mirror
    advertised a load the API can never make."""
    picker = " ".join(_read("features/model-picker/components/model-selector.tsx").split())
    assert "apiLoadable: isGguf && !isOllamaLinkPath(id)," in picker
    sidebar = " ".join(_read("features/model-picker/components/sidebar-model-config.tsx").split())
    assert "apiLoadable: isGguf && !isOllamaLinkPath(modelId)," in sidebar
    # The same classification gates the one-time backfill, or a config saved before
    # the upgrade still reaches the server on the next start.
    backfill = " ".join(_read("features/model-picker/api/migrate-model-overrides.ts").split())
    assert "!isOllamaLinkPath(entry.modelId) &&" in backfill

    identity = _read("features/hub/lib/model-identity.ts")
    assert 'new Set([".studio_links", "ollama_links"])' in identity
    resolver = (
        WORKDIR / "studio" / "backend" / "core" / "inference" / "local_model_resolver.py"
    ).read_text(encoding = "utf-8")
    assert 'seg in (".studio_links", "ollama_links")' in resolver, "the rule this mirrors"


def test_the_backfill_fills_in_fields_rather_than_skipping_known_keys():
    """The backfill reads the override map once and then writes each model in turn,
    so a save by another tab during that pass was overwritten by this browser's
    older localStorage copy. The server reads and writes under one transaction
    rather than this re-fetching per model, and it does so field by field: the
    override map shipped before this browser mirror did, holding only
    llama_extra_args and max_seq_length, so an entry-level skip would strand the
    context, KV cache, speculative and GPU settings this migration exists to carry."""
    backfill = " ".join(_read("features/model-picker/api/migrate-model-overrides.ts").split())
    assert "{ fillAbsentFields: true }," in backfill
    # Key presence alone is not "done": what the server lacks decides.
    assert "const stored = known.get(key);" in backfill
    assert (
        "if (stored && absentFields(stored, current.config).length === 0) { continue; }" in backfill
    )
    assert "const fields = Object.keys(toApiOverride(config));" in backfill
    assert "return fields.filter((field) => !(field in stored));" in backfill

    api = " ".join(_read("features/model-picker/api/model-overrides.ts").split())
    assert "fillAbsentFields?: boolean;" in api
    assert "options?.fillAbsentFields ? { fill_absent_fields: true } : {}" in api.replace(
        "// biome-ignore lint/style/useNamingConvention: API schema ", ""
    )
    # Ordinary saves must stay unconditional, or a settings edit would never land.
    assert "syncModelOverride" in api and "fill_absent_fields: true" in api

    route = (WORKDIR / "studio" / "backend" / "routes" / "settings.py").read_text(encoding = "utf-8")
    assert "fill_absent_fields: bool = False" in route, "the rule this mirrors"
    assert "fill_absent_fields = payload.fill_absent_fields," in route
    # A write mode must not leak into the saved fields, or "only model_id means
    # forget this model" stops working.
    assert '"remove", "fill_absent_fields"' in route

    # The merge is the server's, under the write's own transaction: a client-side
    # read-modify-write would reopen the race the conditional write closed.
    db = (WORKDIR / "studio" / "backend" / "storage" / "studio_db.py").read_text(encoding = "utf-8")
    assert "merged = {**entry_value, **stored}" in db
    assert "BEGIN IMMEDIATE" in db


def test_the_hub_settings_page_matches_a_resident_path_loaded_model():
    """A GGUF loaded from an inactive HF cache or straight off disk loads by path,
    but /status reports the clean public id, so comparing it to settingsTarget.id
    said "not loaded" and the page showed saved or default values instead of the
    live launch config."""
    hub = " ".join(_read("features/hub/hub-page.tsx").split())
    assert (
        "residentModelIdMatches( activeCheckpoint, settingsTarget.id, settingsTarget.configId, )"
        in hub
    )
    assert "loadedConfig={settingsTargetIsResident ? activeModelConfig : null}" in hub
    assert "settingsTargetIsResident ? activeGgufContextLength : null" in hub
    # The loadable identifier, as every other status reader records it. active_model
    # is the clean public id, and two files sharing a filename collapse onto one, so
    # storing it would let the wrong catalog row look loaded.
    assert "checkpointId: resolveInferenceCheckpointId(status)," in hub
    assert "setCheckpoint(status.active_model" not in hub
    chat = " ".join(_read("features/chat/lib/apply-inference-status-to-store.ts").split())
    assert "return status.model_identifier ?? status.active_model;" in chat, "the rule this mirrors"
    # The alias is the backend's own public id rule, not a private heuristic.
    identity = _read("features/hub/lib/model-identity.ts")
    assert "export function publicModelId(" in identity
    assert "models--" in identity and "snapshots" in identity
    # Only a namespaced repo id names one model; a filename or directory stem does
    # not, so it must never stand in for the loaded model's identity.
    assert 'return publicId.includes("/") && modelIdsMatch(active, publicId);' in identity
    backend = (WORKDIR / "studio" / "backend" / "core" / "inference" / "model_ids.py").read_text(
        encoding = "utf-8"
    )
    assert "def public_model_id(" in backend, "the rule this mirrors"


def test_the_hub_hydrates_the_live_settings_before_it_offers_them():
    """The Hub builds activeModelConfig out of the chat runtime store, and landing
    straight on /hub is the one entry point where nothing has applied
    /api/inference/status yet: useChatModelRuntime has no mount sync and the chat
    page is a different route. Pinning only the checkpoint left every other field
    at its default, so the settings page passed those defaults on as the resident
    model's live config and Apply reloaded the model with them."""
    hub = " ".join(_read("features/hub/hub-page.tsx").split())
    assert "adoptResidentModelStatus(" in hub
    assert "applyActiveModelStatusToStore(status, {" in hub
    assert "previousCheckpoint: previous.checkpoint ?? undefined," in hub
    assert "previousGgufVariant: previous.ggufVariant," in hub
    assert "modelLoading: store.modelLoading," in hub

    adopt = " ".join(_read("features/hub/lib/adopt-inference-status.ts").split())
    # Unconditional: a persisted checkpoint rehydrates from localStorage on its
    # own, carrying none of the fields that say how the model was launched.
    assert "actions.applyStatus(previous); return true;" in adopt
    # Never fight the load that owns the store, and never describe an external
    # provider's model with the resident GGUF's launch settings.
    assert "if (state.checkpointIsExternal) { return false; }" in adopt
    assert "if (state.modelLoading) { return false; }" in adopt

    # Same call the chat runtime's own refresh makes, which is the rule this mirrors.
    runtime = " ".join(_read("features/chat/hooks/use-chat-model-runtime.ts").split())
    assert "applyActiveModelStatusToStore(statusRes, {" in runtime


def test_the_hub_settings_editor_reseeds_when_the_live_config_lands():
    """ModelConfigPage reads loadedConfig in a useState initializer, so it seeds
    once per mounted instance. Opening the Hub's settings page before
    /api/inference/status has hydrated (or while the target is still loading)
    flips loadedConfig from null to the live config after mount, and without the
    config in the React key the editor kept the saved/default values for a model
    running with something else, which Apply then wrote back over it."""
    view = " ".join(_read("features/hub/catalog/hub-model-settings-view.tsx").split())
    assert "key={modelConfigInstanceKey( target.id, target.ggufVariant, loadedConfig, )}" in view
    # Same key the sidebar entry uses; that parity is the point. It keys by the
    # settings variant, which is the loader's filename label nulled out for a
    # standalone .gguf, so both surfaces name one config per file.
    sidebar = " ".join(_read("features/model-picker/components/sidebar-model-config.tsx").split())
    assert "key={modelConfigInstanceKey(modelId, settingsGgufVariant, loadedConfig)}" in sidebar

    signature = " ".join(_read("features/model-picker/model-config/config-signature.ts").split())
    # "No live config yet" has to be its own value: that transition is exactly
    # the one that must remount.
    assert 'if (!config) { return "none"; }' in signature
    for field in (
        "config.customContextLength",
        "config.maxSeqLength",
        "config.kvCacheDtype",
        "config.speculativeType",
        "config.specDraftNMax",
        "config.tensorParallel",
        "config.chatTemplateOverride",
    ):
        assert field in signature, field

    page = " ".join(_read("features/model-picker/components/model-config-page.tsx").split())
    assert "const [initial] = useState(resolveInitial);" in page, "the rule this mirrors"


def test_a_standalone_gguf_has_one_settings_key():
    """The inventory labels a single scanned .gguf from its filename, so the Hub row
    menu keyed its settings to `<path>:Q4_K_M` while the Chat picker, the detail
    card and the backfill all use the bare path: two surfaces, two configs."""
    hub = " ".join(_read("features/hub/hub-page.tsx").split())
    assert "let ggufVariant = settingsGgufVariantForRow(row);" in hub
    assert "row.formatVariant" not in hub, "the row's raw label is not a settings key"

    helper = " ".join(_read("features/hub/inventory/settings-identity.ts").split())
    assert 'row.kind === "local" && row.path.toLowerCase().endsWith(".gguf")' in helper

    common = (
        WORKDIR / "studio" / "backend" / "hub" / "services" / "models" / "common.py"
    ).read_text(encoding = "utf-8")
    # The rule this mirrors: a variant is derived only for a single scanned file.
    assert "extract_quant_label(gguf_files[0].name)" in common
    assert "if scan_path.is_file() and len(gguf_files) == 1" in common


def test_a_standalone_gguf_is_resident_despite_its_derived_quant():
    """A loose .gguf keys its settings by the bare path with no variant, but the
    loader derives one from the filename (llama_cpp sets _hf_variant from
    _extract_quant_label) and /status reports it, so an equality between the two
    could never hold and the settings page withheld the live launch config from
    the very file that was loaded."""
    hub = " ".join(_read("features/hub/hub-page.tsx").split())
    assert "const settingsTargetIsStandaloneFile =" in hub
    assert 'settingsTarget.id.toLowerCase().endsWith(".gguf")' in hub
    assert (
        "(settingsTargetIsStandaloneFile || ggufVariantsMatch(activeGgufVariant, settingsTarget.ggufVariant))"
        in hub
    )
    backend = (WORKDIR / "studio" / "backend" / "core" / "inference" / "llama_cpp.py").read_text(
        encoding = "utf-8"
    )
    assert (
        "self._hf_variant = _extract_quant_label(gguf_path)" in backend
    ), "the derived label this accounts for"


def test_a_standalone_gguf_has_one_settings_identity_everywhere():
    """A loose .gguf has no quant to choose between, but llama_cpp falls back to
    _extract_quant_label(gguf_path) when a load names no variant, and /status
    echoes that as gguf_variant. The sidebar took it straight from the store, so
    an edit there landed under "<path>:Q4_K_M" while the Hub row, the picker and
    the backfill all wrote the bare path. The auto-switch lookup reads the bare
    path BEFORE "<path>:<file_variant>", so the sidebar's entry was never the one
    an API load applied: the user edited settings that could not take effect."""
    sidebar = " ".join(_read("features/model-picker/components/sidebar-model-config.tsx").split())
    # Nulled for the settings identity, and used for every field that keys it.
    assert (
        "const settingsGgufVariant = isStandaloneGgufPath(modelId) ? null : ggufVariant;" in sidebar
    )
    assert "ggufVariant: settingsGgufVariant," in sidebar
    assert "ggufVariant: settingsGgufVariant ?? undefined," in sidebar
    # The label still shows the quant; only the identity drops it.
    assert "displayName: ggufVariant ? `${leaf} · ${ggufVariant}` : leaf," in sidebar

    # One rule, one definition: the Hub row applies the same test.
    identity = _read("features/hub/lib/model-identity.ts")
    assert "export function isStandaloneGgufPath(" in identity
    assert 'modelId.toLowerCase().endsWith(".gguf")' in identity
    row_identity = _read("features/hub/inventory/settings-identity.ts")
    assert 'row.path.toLowerCase().endsWith(".gguf")' in row_identity

    # The precedence that makes the bare path the one that wins.
    route = (WORKDIR / "studio" / "backend" / "routes" / "inference.py").read_text(
        encoding = "utf-8",
    )
    assert 'f"{target_id}:{file_variant}" if file_variant else None,' in route
    bare = route.index("\n                            target_id,\n")
    labelled = route.index('f"{target_id}:{file_variant}"')
    assert bare < labelled, "the bare path must be read before the filename label"
