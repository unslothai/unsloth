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


def _read_backend(rel: str) -> str:
    path = WORKDIR / "studio" / "backend" / rel
    assert path.exists(), f"missing backend source file: {path}"
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
    for field in (
        "gpuMemoryMode",
        "gpuLayers",
        "nCpuMoe",
        "selectedGpuIds",
        "selectedGpuIndexKind",
    ):
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


def test_deferred_gpu_pick_keeps_its_index_namespace():
    """A remembered pick restored before GPU discovery must keep its namespace
    until load-time reconciliation, or Vulkan IDs can be reused as physical IDs."""
    store = _read("features/chat/stores/chat-runtime-store.ts")
    assert "selectedGpuIndexKind: GpuIndexKind | null;" in store

    apply = _read("features/model-picker/model-config/apply-per-model-config.ts")
    assert "selectedGpuIndexKind: s.selectedGpuIndexKind" in apply
    assert "config.selectedGpuIndexKind === undefined" in apply
    assert 'config.selectedGpuIndexKind ?? "physical"' not in apply

    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    assert "stateBeforeUnload.selectedGpuIndexKind," in runtime

    compare = _read("features/chat/shared-composer.tsx")
    assert "store.selectedGpuIndexKind," in compare


def test_gpu_picker_round_trips_requested_pool_not_fitted_subset():
    """A GGUF fit may narrow [0, 1] to [0], but load/status hydration must keep
    [0, 1] as the editable pool so a later reload can grow back onto GPU 1."""
    types = _read("features/chat/types/api.ts")
    assert types.count("requested_gpu_ids?: number[] | null") >= 2

    store = _read("features/chat/stores/chat-runtime-store.ts")
    assert 'hasOwnProperty.call(resp, "requested_gpu_ids")' in store
    assert "const reportedGpuIds = requestedGpuIdsFromResponse(resp)" in store
    assert "loadedGpuIndexKind: GpuIndexKind | null;" in store
    assert "loadedGpuIndexKind: gpuIds == null ? null : (gpuIndexKind ?? null)" in store
    # A cold discovery cache reports gpuIndexKind === undefined (deferred, not
    # rejected); only a warm cache's definitive null should drop the pin.
    assert "reportedGpuIds != null && gpuIndexKind !== null" in store

    status = _read("features/chat/lib/apply-inference-status-to-store.ts")
    assert "const incomingGpuFields = loadedGpuMemoryFields(status)" in status
    assert "const incomingGpuIds = incomingGpuFields.loadedGpuIds" in status
    assert "const incomingGpuIndexKind = incomingGpuFields.loadedGpuIndexKind" in status
    assert status.count("prevState.loadedGpuIndexKind") >= 2
    assert status.count("sameGpuSelection(") >= 2


def test_compare_load_uses_each_models_gpu_config():
    src = _read("features/chat/shared-composer.tsx")
    assert "ownConfig.gpuMemoryMode ?? compareLoadKnobs.gpuMemoryMode" in src
    assert "ownConfig.gpuLayers ?? compareLoadKnobs.gpuLayers" in src
    assert "ownConfig.nCpuMoe ?? compareLoadKnobs.nCpuMoe" in src
    assert "if (ownConfig.selectedGpuIds != null)" in src
    assert "ownConfig.selectedGpuIndexKind," in src
    assert "compareLoadKnobs.selectedGpuIndexKind," in src
    assert src.count("resolvedIsDiffusion === true") >= 2
    for field in (
        "gpu_memory_mode: effectiveGpuMemoryMode",
        "gpu_layers: effectiveGpuLayers",
        "n_cpu_moe: effectiveNCpuMoe",
        "gpu_ids: effectiveSelectedGpuIds ?? undefined",
    ):
        assert field in src

    page = _read("features/chat/chat-page.tsx")
    assert page.count("isDiffusion: meta.isDiffusion") >= 2
    assert "isDiffusion: globalIsDiffusion" in page


def test_diffusion_load_paths_disable_tensor_parallel():
    """Every frontend path that classifies DiffusionGemma must send tensor
    parallelism as false so the ignored setting cannot force repeat reloads."""
    compare = _read("features/chat/shared-composer.tsx")
    compact_compare = " ".join(compare.split())
    assert "const effectiveTensorParallel = resolvedIsDiffusion ? false :" in compact_compare
    assert compare.count("tensor_parallel: effectiveTensorParallel") == 2

    runtime = " ".join(_read("features/chat/hooks/use-chat-model-runtime.ts").split())
    assert "const loadTensorParallel = targetIsDiffusion ? false :" in runtime

    apply = " ".join(_read("features/model-picker/model-config/apply-per-model-config.ts").split())
    assert "tensorParallel: options.isDiffusion ? false :" in apply

    adapter = _read("features/chat/api/chat-adapter.ts")
    autoload = adapter.split("async function loadAutoLoadCandidate", 1)[1]
    autoload = autoload.split("async function autoLoadSmallestModel", 1)[0]
    compact_autoload = " ".join(autoload.split())
    assert "config.selectedGpuIds != null || config.tensorParallel === true" in compact_autoload
    assert "const effectiveTensorParallel = isDiffusion ? false :" in compact_autoload
    assert autoload.count("tensor_parallel: effectiveTensorParallel") == 2


def test_diffusion_load_keeps_the_standing_gpu_memory_mode():
    """A diffusion config is sanitized to gpuMemoryMode "auto" because the mode does
    not apply to it, not because the user picked Auto. Applying that sanitized value
    to the runtime store would strand the session on Auto: persistGpuMemoryModeOnLoad
    deliberately skips diffusion responses, so nothing writes the standing preference
    back, and the next ordinary GGUF loaded without its own config sends the stale
    "auto" and persists it over the user's Manual."""
    apply = " ".join(_read("features/model-picker/model-config/apply-per-model-config.ts").split())
    assert (
        "gpuMemoryMode: options.isDiffusion ? readPersistedGpuMemoryMode() : "
        "(config.gpuMemoryMode ?? readPersistedGpuMemoryMode())," in apply
    )
    # The unconditional form is what leaked the sanitized mode into the store.
    assert "gpuMemoryMode: config.gpuMemoryMode ?? readPersistedGpuMemoryMode()," not in apply

    # The other half of the contract: the diffusion load itself must not persist.
    store = " ".join(_read("features/chat/stores/chat-runtime-store.ts").split())
    assert "if (resp.is_gguf && !resp.is_diffusion) saveGpuMemoryMode(mode);" in store

    # And the sanitizer that produces the "auto" this guards against still runs.
    page = " ".join(_read("features/model-picker/components/model-config-page.tsx").split())
    assert "withoutUnsupportedDiffusionSettings(committedConfig, gpuIndexKind)" in page


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


def test_local_mtp_warning_covers_path_and_native_gguf_sources():
    """The local MTP recovery text must cover direct files, custom folders,
    and native-picker labels instead of classifying only .gguf suffixes."""
    src = _read("features/chat/chat-settings-sheet.tsx")
    local = re.search(r"const isLocalGguf =.*?;", src, re.S)
    assert local
    assert "isGguf &&" in local.group(0)
    assert "activeModelIsLocal" in local.group(0)
    assert "isLocalModelPath" in local.group(0)
    # Two signals must not classify the model here, because both mislabel a
    # remote GGUF as local: a native token, which outlives a switch to a remote
    # model, and a bare .gguf suffix, since a one-slash org/name.gguf is a
    # repository id. activeModelIsLocal is the backend's own answer for both.
    assert "activeNativePathToken" not in local.group(0)
    assert ".gguf" not in local.group(0)

    # Switching models must drop both together: a kept flag would classify the
    # newly selected model by the old one's provenance.
    store = _read("features/chat/stores/chat-runtime-store.ts")
    reset = re.search(r"setCheckpoint: \(modelId, ggufVariant\) =>.*?\}\),", store, re.S)
    assert reset
    assert "activeModelIsLocal: false" in reset.group(0)
    assert "specFallbackReason: null" in reset.group(0)
    assert "isLocalGguf" in src.split('specFallbackReason === "drafter_not_found"', 1)[1]


def test_local_mtp_warning_uses_backend_source_metadata():
    types = _read("features/chat/types/api.ts")
    assert types.count("is_local_model?: boolean") >= 2

    status = _read("features/chat/lib/apply-inference-status-to-store.ts")
    assert "activeModelIsLocal: status.is_local_model ?? false" in status

    runtime = _read("features/chat/stores/chat-runtime-store.ts")
    assert "activeModelIsLocal: boolean" in runtime
    assert runtime.count("activeModelIsLocal: false") >= 2

    load = _read("features/chat/hooks/use-chat-model-runtime.ts")
    assert "activeModelIsLocal: loadResponse.is_local_model ?? false" in load

    models = _read_backend("models/inference.py")
    assert models.count("is_local_model: bool = Field(") >= 2

    route = _read_backend("routes/inference.py")
    assert route.count("is_local_model = config.is_local") >= 2
    # GGUF status reports the provenance the load recorded. Re-deriving it from
    # the filesystem would flip a local model to remote once its directory goes
    # away underneath a running server.
    assert "llama_backend._is_local_model = bool(native_grant_backed or config.is_local)" in route
    # Both GGUF responses report it: the status poll and the already_loaded
    # dedup reply. Either one re-deriving it reintroduces the flip.
    assert route.count("is_local_model = _loaded_is_local_model(") >= 2
    assert "backend.active_model_name and is_local_path(backend.active_model_name)" in route


def test_fixed_layer_gguf_pins_displayed_context():
    """An already-loaded auto-fit GGUF saved with Manual fixed GPU layers must
    pin the shown context, so a later fresh load keeps the fitted placement
    instead of sending native/0 and recreating the OOM."""
    src = _read("features/model-picker/components/model-config-page.tsx")
    assert "const pinFixedLayerContext =" in src
    assert 'loadableConfig.gpuMemoryMode === "manual"' in src
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
    assert "onRun(effectiveLoadConfig, classifiedIsDiffusion)" in src
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
        "(targetIsGguf ? 0 : DEFAULT_MAX_SEQ_LENGTH);" in src
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
    assert "selectedGpuIndexKind: undefined," in src


def test_requested_gpu_pick_survives_fit_narrowing_and_namespace_changes():
    store = _read("features/chat/stores/chat-runtime-store.ts")
    assert "requestedGpuIdsFromResponse(resp)" in store
    gpu_selection = _read("hooks/gpu-selection.ts")
    assert 'savedIndexKind === undefined ? "physical" : savedIndexKind' in gpu_selection
    assert "currentIndexKind !== undefined" in gpu_selection
    assert "expectedIndexKind !== currentIndexKind" in gpu_selection
    assert "A null namespace is only safe while discovery is also unresolved" in gpu_selection
    assert (
        "Namespace knowledge is authoritative even while membership is unavailable" in gpu_selection
    )
    gpu_info = _read("hooks/use-gpu-info.ts")
    assert "cachedPinnableGpuContext" in gpu_info
    assert 'unavailableVulkan ? "vulkan" : undefined' in gpu_info
    assert "cachedPinnableGpuIndexKind" in gpu_info
    config = _read("features/model-picker/model-config/per-model-config.ts")
    assert '"selectedGpuIndexKind"' in config


def test_staged_gpu_pick_reconciles_with_async_namespace_discovery():
    page = _read("features/model-picker/components/model-config-page.tsx")
    assert "reconcileGpuSelection(" in page
    assert "setConfig((current)" in page
    assert "reconcileConfigGpuSelection(" in page
    assert "gpuIndexKind" in page
    assert "pinnableDevices" in page
    assert "reconciled.ids === null ? undefined : reconciled.indexKind" in " ".join(page.split())
    assert "selectsAll ? null : next" not in page
    gpu_info = _read("hooks/use-gpu-info.ts")
    assert 'inferenceGpu?.backend === "vulkan"' in gpu_info
    assert "!inferenceGpu.available" in gpu_info


def test_compare_classifies_gguf_before_reconciling_gpu_ids():
    compare = _read("features/chat/shared-composer.tsx")
    metadata = compare.index("fetchGgufStagedMetadata(")
    reconcile = compare.index("reconcilePersistedGpuIds(", metadata)
    validate = compare.index("const validation = await validateModel(", reconcile)
    load = compare.index("const resp = await loadModel(", validate)
    assert metadata < reconcile < validate < load
    assert compare.count("resolvedIsDiffusion") >= 3
    assert "prepareHfTokenForUse(" in compare
    page = _read("features/model-picker/components/model-config-page.tsx")
    assert "isDiffusion?: boolean" in page
    assert "onRun(effectiveLoadConfig, classifiedIsDiffusion)" in page


def test_model_config_prepares_hf_token_before_gguf_metadata_preflight():
    """Settings classification must use the same stale-token recovery as load."""
    page = _read("features/model-picker/components/model-config-page.tsx")
    assert 'import { prepareHfTokenForUse } from "@/features/hf-auth";' in page
    effect = page.split("// Fetch GGUF header dims", 1)[1]
    effect = effect.split("const stagedDims =", 1)[0]
    prepare = effect.index("prepareHfTokenForUse(hfToken || null)")
    metadata = effect.index("fetchGgufStagedMetadata({", prepare)
    assert prepare < metadata
    assert "if (!preparedToken.proceed)" in effect
    cancel = effect.index("if (!preparedToken.proceed)")
    assert "settleWithoutMetadata();" in effect[cancel:metadata]
    assert effect.count("settleWithoutMetadata();") == 2
    assert "hf_token: preparedToken.token" in effect
    assert "hf_token: hfToken" not in effect


def test_chat_load_prepares_hf_token_before_gguf_metadata_preflight():
    """The single-model load path classifies a GGUF via fetchGgufStagedMetadata
    before validateModel/loadModel run. The Hub rejects an invalid Authorization
    header with 401 even for a PUBLIC repo, so that preflight must prepare the
    token like every other caller; otherwise a stale saved token aborts the whole
    load instead of offering the "continue anonymously / replace token" recovery.
    """
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    prepare = runtime.index("prepareHfTokenForUse(")
    metadata = runtime.index("fetchGgufStagedMetadata({", prepare)
    assert prepare < metadata
    # The raw store token must not be handed to the preflight.
    assert "hf_token: preparedToken.token" in runtime
    assert (
        "hf_token: useChatRuntimeStore.getState().hfToken" not in runtime
    ), "GGUF metadata preflight must not send the unprepared stored token"
    assert 'throw new Error("Model load cancelled.")' in runtime


def test_chat_autoload_prepares_hf_token_before_gguf_metadata_preflight():
    """Background autoload performs the same GGUF classification probe as the
    interactive paths, so a stale saved token must take the same recovery path.
    """
    adapter = _read("features/chat/api/chat-adapter.ts")
    autoload = adapter.split("async function loadAutoLoadCandidate", 1)[1]
    autoload = autoload.split("async function autoLoadSmallestModel", 1)[0]
    prepare = autoload.index("prepareHfTokenForUse(")
    metadata = autoload.index("fetchGgufStagedMetadata({", prepare)
    assert prepare < metadata
    assert "hf_token: preparedToken.token" in autoload
    assert 'throw new Error("Model load cancelled.")' in autoload


def test_cpu_only_llama_build_hides_gpu_picker():
    src = (WORKDIR / "studio" / "backend" / "main.py").read_text(encoding = "utf-8")
    assert "and not LlamaCppBackend._backend_lacks_gpu_lib()" in src


def test_vulkan_inference_devices_do_not_replace_global_gpu_info():
    backend = (WORKDIR / "studio" / "backend" / "main.py").read_text(encoding = "utf-8")
    assert '"gguf_gpu_ids_supported": gpu_ids_supported' in backend
    assert '"inference_gpu": inference_gpu_info' in backend
    frontend = _read("hooks/use-gpu-info.ts")
    assert 'inference?.backend === "vulkan"' in frontend
    assert "data?.gpu?.devices ?? []" in frontend


def test_diffusion_picker_hides_and_clears_unsupported_memory_modes():
    api = _read("features/chat/api/chat-api.ts")
    assert "isDiffusion: res.is_diffusion ?? false" in api

    page = _read("features/model-picker/components/model-config-page.tsx")
    assert 'className={isDiffusion ? "hidden" : ROW_CLASS}' in page
    assert "withoutUnsupportedDiffusionSettings(config, gpuIndexKind)" in page
    assert "reconcileConfigGpuSelection(" in page
    assert "resolvedIsDiffusion" in page
    assert "stagedMetadataPending ||" in page
    assert "config.selectedGpuIds != null" in page
    for field in (
        'gpuMemoryMode: "auto"',
        "gpuLayers: undefined",
        "nCpuMoe: undefined",
        "tensorParallel: false",
        "selectedGpuIds: undefined",
        "selectedGpuIndexKind: undefined",
    ):
        assert field in page


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
    sidebar = _read("features/model-picker/components/sidebar-model-config.tsx")
    # The sidebar form remounts when an external change lands.
    assert 'config.nParallel ?? "",' in sidebar


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
    # Matched on the call prefix, not the whole call: the staged apply also
    # carries the resolved diffusion flag. Only the ordering is the contract.
    assert runtime.index("const previousNParallel") < runtime.index(
        "applyPerModelConfigToRuntime(pendingLoadConfig,"
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
    # The Vulkan inventory is authoritative even while its probe is temporarily
    # empty. Falling through would expose physical CUDA/ROCm IDs in an ordinal
    # picker and make DiffusionGemma offer a selection the route rejects.
    assert "const inference = data?.inference_gpu; " 'if (inference?.backend === "vulkan") {' in src
    # A confirmed-Vulkan backend with no enumerated devices yet must return no
    # devices, not fall through to the torch/CUDA inventory below.
    assert "if (!(inference.devices ?? []).length) return [];" in src
    # Pinnable on the ggml ordinal space, gated on the backend's own support flag.
    assert "const picksAccepted = inference.gguf_gpu_ids_supported !== false;" in src
    assert 'pinnable: picksAccepted && d.index_kind === "vulkan",' in src
    # A Vulkan ordinal is not a CUDA ID, so the torch-side diffusion runner
    # cannot take the pick even when llama-server can.
    assert "diffusionPinnable: false," in src
    # The torch fallback keeps the XPU ban for torch ordinals only: a Vulkan
    # ordinal stays pickable even when this list arrives from an XPU host.
    assert (
        "pinnable: pinnableBackend && "
        '(d.index_kind === "vulkan" || '
        '(data?.device_backend !== "xpu" && d.index_kind === "physical")),' in src
    )
    # Only a physical index is ever handed to the diffusion runner, and ROCm
    # counts: it reuses torch.cuda.* and the same physical-ID path, so excluding
    # it would hide the picker on every multi-GPU ROCm host.
    assert (
        "const diffusionBackend = "
        'data?.device_backend === "cuda" || data?.device_backend === "rocm";' in src
    )
    assert 'diffusionPinnable: diffusionBackend && d.index_kind === "physical",' in src
