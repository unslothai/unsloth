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

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WORKDIR = Path(__file__).resolve().parents[2]
FRONTEND = WORKDIR / "studio" / "frontend" / "src"


def _read(rel: str) -> str:
    path = FRONTEND / rel
    assert path.exists(), f"missing source file: {path}"
    return path.read_text(encoding = "utf-8")


def _read_backend(rel: str) -> str:
    path = WORKDIR / "studio" / "backend" / rel
    assert path.exists(), f"missing backend source file: {path}"
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
    """The active model's config must carry the GPU Memory knobs (GGUF only) so a
    sidebar/hub-gear reload cannot silently reset manual GPU settings, and "Remember
    settings" cannot persist a GPU-less config over a saved one."""
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
    # The GPU knobs are in the editor's instance key, so a reload re-seeds instead of keeping.
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


def test_deferred_gpu_pick_keeps_its_index_namespace():
    """A remembered pick restored before GPU discovery must keep its namespace
    until load-time reconciliation, or Vulkan IDs can be reused as physical IDs."""
    store = _read("features/chat/stores/chat-runtime-store.ts")
    assert "selectedGpuIndexKind: GpuIndexKind | null;" in store

    apply = _read("features/model-picker/model-config/apply-per-model-config.ts")
    assert "selectedGpuIndexKind: s.selectedGpuIndexKind" in apply
    # gpuFieldsSignature moved to config-signature.ts so the editor's instance key can
    # read it without importing the runtime applier; the namespace rule moved with it.
    signature = _read("features/model-picker/model-config/config-signature.ts")
    assert "config.selectedGpuIndexKind === undefined" in signature
    assert 'config.selectedGpuIndexKind ?? "physical"' not in signature

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
    """Local inventory rows can be classified non-chat (canChat false, e.g."""
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
    """A native (picked / drag-drop) GGUF's path lives only in its signed lease, and the
    picker chat-template GET has no lease plumbing, so the default template must be read
    through the lease-aware validate probe: mint a validate-model lease and post
    include_chat_template."""
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
    """A stopped partial safetensors download must keep its options menu (the Delete
    affordance) like the GGUF card does, or partial downloads can only be cleaned up by
    finishing or leaving them."""
    src = _read("features/hub/catalog/safetensors-download-card.tsx")
    assert "(isDownloaded || (isPartial && !downloading))" in src


def test_pinned_validation_uses_cached_local_variant_listing():
    """Pinned-quant validation must use the TTL-cached hub client with preferLocalCache
    (downloaded-ness is local state) instead of one uncached round-trip per pinned repo
    on every picker open."""
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
    """pinFixedLayerContext is computed from the render-time config, before a same-click
    GPU Layers draft is committed."""
    src = _read("features/model-picker/components/model-config-page.tsx")
    assert "const effectivePinFixedLayerContext =" in src
    assert 'effectiveConfig.gpuMemoryMode === "manual"' in src
    assert "effectiveConfig.gpuLayers != null" in src
    assert "effectiveConfig.customContextLength == null" in src
    assert "{ ...effectiveConfig, customContextLength: activeLoadedContext }" in src


def test_blur_cache_cleared_on_every_settled_render():
    """The lastBlurCommittedRef bridge is valid only across the single synchronous
    same-click gesture that set it."""
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
    """The same-click blur bridge must flush every NumericValueInput-backed setting, not
    just Context Length."""
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
    """An explicit customContextLength that equals the native ceiling is still a user
    override, so contextAtDefault must require customContextLength == null."""
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
    """The v1->v2 localStorage migration (unsloth_load_settings -> unsloth_model_configs)
    is invoked on every store read, so it must be idempotent: repeated reads, browser
    reloads, and Studio restarts must never re-migrate, duplicate records, or overwrite
    a newer per-model config."""
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
    # Layer 2: persistent cross-session flag so a completed migration is never redone.
    assert 'const LEGACY_MIGRATION_FLAG = "unsloth_model_configs_migrated";' in src
    assert "if (localStorage.getItem(LEGACY_MIGRATION_FLAG)) {" in src
    assert src.count('localStorage.setItem(LEGACY_MIGRATION_FLAG, "1");') >= 3
    # Layer 3: non-overwriting merge skips an existing (or default) key, so even a
    # forced re-run cannot duplicate or clobber a user's config.
    assert "if (isDefaultConfig(migrated) || Object.hasOwn(map, key)) {" in src


def test_parallel_slots_setting_wired_end_to_end():
    """The per-load Parallel Slots knob (llama-server --parallel) must flow from the
    run-settings form through persistence, every /load builder, the validate preflight
    and the cross-model reset; a lost hop silently reverts the model to the server-wide
    slot default."""
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
    # The sidebar form remounts when an external change lands.
    signature = _read("features/model-picker/model-config/config-signature.ts")
    assert 'config.nParallel ?? "",' in signature
    sidebar = " ".join(_read("features/model-picker/components/sidebar-model-config.tsx").split())
    assert "key={modelConfigInstanceKey(modelId, settingsGgufVariant, loadedConfig)}" in sidebar


def test_parallel_slots_reach_an_api_load_through_the_server_mirror():
    """The server mirror is the hop an OpenAI-compatible auto-switch load reads, and it is
    the browser's only way to express a per-model setting to a load no browser makes."""
    api = " ".join(_read("features/model-picker/api/model-overrides.ts").split())
    assert "n_parallel?: number;" in api
    assert (
        "if (config.nParallel && config.nParallel > 0) { payload.n_parallel = config.nParallel; }"
        in api
    )
    # The monitor lists what a remote load applies, so slots-only must not read as "App defaults".
    monitor = " ".join(_read("features/api-monitor/components/saved-model-settings.tsx").split())
    assert "if (override.n_parallel) {" in monitor

    route = _read_backend("routes/settings.py")
    assert "n_parallel: Optional[int] = Field(" in route
    assert "n_parallel = payload.n_parallel," in route
    store = _read_backend("utils/openai_auto_switch_settings.py")
    assert 'entry["n_parallel"] = n_parallel' in store
    # GGUF-only, like the picker: a safetensors load has no llama-server slots.
    gguf_block = store.split("    if is_gguf:", 1)[1]
    assert 'kwargs["n_parallel"] = override["n_parallel"]' in gguf_block


def test_parallel_slots_control_cleared_when_the_load_never_sent_them():
    """`nParallel` is the editable control ("blank = follow the server default") and
    `loadedNParallel` the rollback baseline."""
    status = " ".join(_read("features/chat/lib/apply-inference-status-to-store.ts").split())
    # A swap under this tab must reset the control, or model A's count follows onto model B.
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
    # ...
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
    """The baseline is what a rollback re-sends and what preset capture reads, so a model
    that cannot have slots must not inherit the previous GGUF's count."""
    src = _read("features/chat/lib/apply-inference-status-to-store.ts")
    assert (
        "(status.is_gguf === false || status.requested_parallel_slots === null) && {" in src
    ), "the slotless clear must key on is_gguf or an explicit null echo"
    clear = src.index("status.is_gguf === false || status.requested_parallel_slots === null")
    assert "loadedNParallel: null," in src[clear : clear + 200]
    # Never `!= null`: that also matches the absent field an older backend sends.
    assert "status.requested_parallel_slots !== null && {" not in src


def test_hydration_keeps_the_slot_control_when_readopting_the_running_model():
    """`hydratingExistingModel` is true whenever the incoming status disagrees with what
    this tab last recorded, which includes RE-ADOPTING a model the tab never lost: the
    resident-adopt branch restores the model's own per-model config and only then
    hydrates, passing the EXTERNAL id as `previousCheckpoint`."""
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
    ``--parallel``, so ``_parallel_slot_echo`` reports null slots for it."""
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
    """The control is never seeded from the status echo, so a model running on a remembered
    override shows a BLANK slot control after a browser reload or a tab move to another
    GGUF."""
    src = _read("features/chat/lib/apply-inference-status-to-store.ts")
    status = " ".join(src.split())
    assert (
        "resolveResidentInitialConfig(checkpointId, status.gguf_variant ?? null)" in status
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


def test_remembered_slots_are_read_through_the_cached_repo_alias():
    """An API auto-switch loads a cached repo by its concrete snapshot path, so
    ``status.model_identifier`` (what ``resolveInferenceCheckpointId`` returns) is that
    path while the settings are keyed by the repo id ``modelConfigIdentity`` writes. Read
    only the raw identifier and the resident model looks unremembered: the slot control
    blanks on the model change and the next Save writes the blank over the saved
    ``n_parallel``, locally and through the server mirror."""
    config = " ".join(_read("features/model-picker/model-config/per-model-config.ts").split())
    # The raw identifier still wins, so a path-keyed record is never shadowed.
    assert (
        "const direct = resolveInitialConfig(modelId, ggufVariant); "
        "if (direct.remembered) {" in config
    )
    assert "const alias = publicModelId(modelId);" in config
    # Only a namespaced collapse, the rule residentModelIdMatches applies: every other
    # path collapses onto a file stem two models can share.
    assert 'if (alias === modelId || !alias.includes("/")) { return direct; }' in config

    status = " ".join(_read("features/chat/lib/apply-inference-status-to-store.ts").split())
    assert "resolveResidentInitialConfig(checkpointId, status.gguf_variant ?? null)" in status
    assert "resolveInitialConfig(checkpointId" not in status

    # The backend applies the same model's override by the same alias, which is why the
    # echo the adoption gate compares against carries the saved count at all.
    route = _read_backend("routes/inference.py")
    overrides = route.split("Apply the saved launch config so an API swap loads as the picker", 1)[
        1
    ].split("load_kwargs = {", 1)[0]
    assert 'f"{override_id}:{variant}" if variant else None,' in overrides
    assert "override_id," in overrides


def test_failed_switch_rollback_restores_the_slot_intent_not_the_resolved_count():
    """`loadedNParallel` holds a RESOLVED count even for a load that sent no slots (the
    echo falls back to the server-wide default), so it is the right value to re-send
    when recreating the previous server and the wrong one to put back in the control: it
    turns "follow the server default" into an explicit override that a later Save or
    preset capture pins."""
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
    """GGUF loads run through llama-server, so on a Vulkan build the picker must offer the
    inference inventory (ggml ordinals, the space `--device Vulkan<i>` pins) rather than
    the torch view, which can miss cards llama-server drives."""
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


def test_training_picker_pagination_preserves_scanned_progress():
    adapter = _read("components/resource-picker/use-picker-hub-pagination.ts")
    assert "const canFetch = enabled && hasMore;" in adapter
    assert "return false;" in adapter
    assert "return fetchMore();" in adapter
    assert "signal: scannedCount" in adapter
    for needle in (
        "enabled: canFetch",
        "isFetching",
        "manualFetchAfterAutoFill: true",
        "maxAutoFillFetches: 5",
        "resetKey",
        "resultCount",
    ):
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
        assert "fetchMoreManually" in src
        assert "onLoadMore={fetchMoreManually}" in src
        assert "showLoadMore={hasMoreHf}" in src
        assert "useLatestRef" not in src

    for rel in (
        "features/model-picker/components/train-model-picker-lists.tsx",
        "features/dataset-picker/components/dataset-selector-lists.tsx",
    ):
        src = _read(rel)
        assert "PickerHubPaginationFooter" in src
        assert src.count("<PickerHubPaginationFooter") == 2
        assert "showLoadMore={showLoadMore}" in src


def test_training_model_picker_only_filters_tasks_for_an_explicit_constraint():
    src = _read("features/model-picker/components/train-model-selector.tsx")
    search_options = src.split("useHubModelSearch(", 1)[1].split("});", 1)[0]
    assert "requiredModelType?: ModelType" in src
    assert "task: trainingModelTaskFilter(requiredModelType)" in search_options
    task_filter = src.split("function trainingModelTaskFilter", 1)[1].split(
        "function commitTrainingModelPick", 1
    )[0]
    assert "MODEL_TYPE_TO_HF_TASKS[requiredModelType]" in task_filter
    assert ": undefined" in task_filter
    assert "eligibleLocalModels" in src
    assert "trainingModelMatchesTypeConstraint(" in src
    assert "commitTrainingModelPick({" in src
    assert 't("studio.modelPicker.reasonTypeMismatch")' in src


def test_onboarding_constrains_model_type_and_recovers_default_loading():
    src = _read("features/onboarding/components/steps/model-selection-step.tsx")

    assert "modelType: state.modelType" in src
    assert "ensureModelDefaultsLoaded: state.ensureModelDefaultsLoaded" in src
    assert "ensureModelDefaultsLoaded();" in src
    assert "[ensureModelDefaultsLoaded, selectedModel]" in src
    assert "requiredModelType={modelType ?? undefined}" in src
    assert "TRAINING_METHOD_ORDER.map((method) =>" in src
    assert "TRAINING_METHOD_META[method]" in src
    for method in ("qlora", "lora", "full", "cpt"):
        assert f'<SelectItem value="{method}">' not in src


def test_picker_tabs_reset_the_shared_scroll_container():
    src = _read("components/resource-picker/picker-shell.tsx")
    tab_change = src.split("function handleTabChange", 1)[1].split(
        "function findMatchingOption", 1
    )[0]

    assert "window.requestAnimationFrame" in tab_change
    assert "scrollRef.current.scrollTop = 0" in tab_change


def test_cached_training_rows_select_canonical_repo_identity():
    src = _read("features/model-picker/components/train-model-picker-view-model.ts")
    cached = src.split("function toCachedTrainModelDeviceItem", 1)[1]
    cached = cached.split("function toLocalTrainModelDeviceItem", 1)[0]
    assert "id: row.repoId" in cached
    assert "localPath: row.cachePath ?? null" in cached

    local = src.split("function toLocalTrainModelDeviceItem", 1)[1]
    local = local.split("function hubTrainingModelCandidate", 1)[0]
    display = _read("features/model-picker/lib/train-model-selection-display.ts")
    candidate = display.split("function toTrainModelDisplayCandidate", 1)[1]
    assert 'row.source === "hf_cache" ? row.repoId?.trim() : null' in candidate
    assert "id: cachedRepoId || row.loadId" in candidate
    assert 'knownCached: row.source === "hf_cache"' in local
    assert "...toTrainModelDisplayCandidate(row)" in local
    assert "localPath: row.path" in candidate

    selector = _read("features/model-picker/components/train-model-selector.tsx")
    assert "toCachedTrainModelDeviceItem(" in selector
    assert "toLocalTrainModelDeviceItem(" in selector
    assert "localModelSourceLabelKey(row.source)" in selector


def test_cached_training_rows_remain_display_candidates_after_inventory_dedupe():
    display = _read("features/model-picker/lib/train-model-selection-display.ts")
    selector = _read("features/model-picker/components/train-model-selector.tsx")
    assert "function toCachedTrainModelDisplayCandidate" in display
    assert "id: row.repoId" in display
    assert "title: row.repoId" in display
    assert "localPath: row.cachePath ?? null" in display
    assert "...cachedRows.map(toCachedTrainModelDisplayCandidate)" in selector
    assert "...localRows.map(toTrainModelDisplayCandidate)" in selector


def test_training_picker_controls_keep_visible_keyboard_focus():
    focus = _read("components/resource-picker/picker-focus.ts")
    shell = _read("components/resource-picker/picker-shell.tsx")
    assert "focus-visible:ring-2" in focus
    assert "focus-visible:ring-ring" in focus
    assert "focus-visible:ring-offset-2" in focus
    assert "focus-visible:ring-offset-background" in focus
    assert "focus-visible:ring-inset" in focus

    model = _read("features/model-picker/components/train-model-selector.tsx")
    dataset = _read("features/dataset-picker/components/dataset-selector.tsx")
    options = _read("components/resource-picker/selectable-picker-item.tsx")
    token = _read("features/hub/components/hf-token-indicator.tsx")
    dataset_controls = _read(
        "features/studio/sections/dataset-advanced-settings.tsx"
    )
    dataset_upload = _read("features/studio/sections/dataset-upload.tsx")
    assert "export const PICKER_TRIGGER_CLASS" in focus
    assert "PICKER_FOCUS_VISIBLE_CLASS" in focus
    assert "PICKER_TRIGGER_CLASS" in model
    assert "PICKER_TRIGGER_CLASS" in dataset
    assert "PICKER_OPTION_FOCUS_VISIBLE_CLASS" in options
    assert "PICKER_FOCUS_VISIBLE_CLASS" in token
    assert "PICKER_FOCUS_VISIBLE_CLASS" in dataset_controls
    assert "PICKER_FOCUS_VISIBLE_CLASS" in dataset_upload
    assert dataset_controls.count("aria-label={t(") >= 4
    assert 't("studio.dataset.streamingInfoAriaLabel")' in dataset_controls
    assert "focus-visible:ring-0" not in model
    assert "focus-visible:ring-0" not in dataset
    assert "focus-visible:ring-0" not in token

    search = shell.split("<Input", 1)[1].split("/>", 1)[0]
    assert "PICKER_FOCUS_VISIBLE_CLASS" not in shell
    assert "border-0" in search
    assert "focus-visible:border-0" in search
    assert "focus-visible:ring-0" in search
    assert "focus-visible:ring-offset-0" in search

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


def test_full_precision_cached_models_do_not_warn_about_a_qlora_download():
    src = _read("features/training/hooks/use-training-resource-notices.ts")
    resolver = src.split("function resolveModelNotice", 1)[1]
    resolver = resolver.split("function resolveDatasetNotice", 1)[0]
    assert "requiresQuantizedCache" not in src
    assert "quantMethod" not in src
    assert "knownCached," in resolver
    assert "localPath," in resolver


def test_training_cache_reconciliation_requires_runnable_weights():
    reconciliation = _read("features/studio/hooks/use-training-cache-reconciliation.ts")
    notices = _read("features/training/hooks/use-training-resource-notices.ts")
    assert reconciliation.count("isTrainableModelFormat(row.model_format)") == 2
    assert "!row.capabilities.canTrain" in notices


def test_dataset_ai_assist_cannot_apply_a_stale_dialog_request():
    dialog = _read("features/studio/sections/dataset-preview-dialog.tsx")
    api = _read("features/training/api/datasets-api.ts")
    assert "aiAssistControllerRef.current?.abort()" in dialog
    assert "aiAssistControllerRef.current !== controller" in dialog
    assert "signal: controller.signal" in dialog
    assert "signal?: AbortSignal" in api
    assert "signal," in api


def test_local_dataset_picker_uses_cross_platform_path_identity():
    selector = _read("features/dataset-picker/components/dataset-selector.tsx")
    display = _read("features/dataset-picker/lib/display.ts")
    assert "cacheLocalPathMatchesSelection(item.path, candidate)" in selector
    assert "cacheLocalPathMatchesSelection(candidate.path, uploadedFile)" in display
    assert "item.path === query" not in selector
    assert "item.path === uploadedFile" not in selector

    lists = _read("features/dataset-picker/components/dataset-selector-lists.tsx")
    assert "export type DatasetDeviceItem" in lists
    assert "export function DatasetDeviceList" in lists
    assert "export function DatasetHubList" in lists
    assert "cacheLocalPathMatchesSelection(selectedLocalPath, item.path)" in lists
    assert "selectedLocalPath === item.path" not in lists
    assert "function DeviceList" not in selector
    assert "function HubList" not in selector

    selection = _read("features/studio/sections/dataset-selection.tsx")
    assert "cacheLocalPathMatchesSelection(item.path, uploadedFile)" in selection
    assert "item.path === uploadedFile" not in selection


def test_studio_local_dataset_inventory_refreshes_when_uploads_may_change():
    inventory = _read("features/studio/sections/use-local-dataset-inventory.ts")
    assert "useDeviceInventorySources(" in inventory
    assert '["localDatasets"]' in inventory
    assert "listLocalDatasets" not in inventory
    assert 'enabled: datasetSource === "upload"' in inventory
    assert "localDatasets.ready" in inventory
    assert "!wasUploadSource.current" in inventory
    assert inventory.count("refresh().catch(() => undefined)") == 2
    assert 'window.addEventListener("focus", refreshWhenVisible)' in inventory
    assert (
        'document.addEventListener("visibilitychange", refreshWhenVisible)'
        in inventory
    )


def test_local_dataset_keyboard_commit_uses_canonical_path_identity():
    shell = _read("components/resource-picker/picker-shell.tsx")
    matcher = _read("components/resource-picker/device-item-match.ts")
    selector = _read("features/dataset-picker/components/dataset-selector.tsx")
    model_selector = _read("features/model-picker/components/train-model-selector.tsx")
    model_view_model = _read("features/model-picker/components/train-model-picker-view-model.ts")

    assert "export type PickerExactQueryCommitResult" in shell
    assert "onExactQueryCommit?: (query: string) => PickerExactQueryCommitResult;" in shell
    assert "isValidHubResourceId(activeQuery)" in selector
    assert "isValidHubResourceId(query)" in model_selector
    assert "query: activeQuery" in model_selector
    assert shell.index('commitResult?.kind === "handled"') < shell.index("if (canUseThis)")
    assert "const canCommitQuery = tab !== PICKER_TAB.hub || online" in shell
    assert "const canUseThis = showUseThis && canCommitQuery" in shell
    assert shell.index("if (!canCommitQuery)") < shell.index(
        "const commitResult = onExactQueryCommit"
    )
    assert "{canUseThis && (" in shell
    assert shell.index("function findMatchingOption") < shell.index(
        'commitResult?.kind === "ambiguous"'
    )
    assert "values.includes(query)" in shell
    assert 'value.trim().normalize("NFC").toLowerCase()' in matcher
    assert matcher.index("const canonicalItem = items.find") < matcher.index(
        "const titleMatches = items.filter"
    )
    assert "titleMatches.length === 1" in matcher
    assert 'kind: "ambiguous", firstItem: titleMatches[0]' in matcher
    assert 'commitResult?.kind === "handled"' in shell
    assert 'commitResult?.kind === "ambiguous"' in shell
    assert "findMatchingOption(commitResult.focusValue)" in shell
    assert "optionMatchesQuery(option, query)" in shell
    ambiguity = shell.split('commitResult?.kind === "ambiguous"', 1)[1].split(
        "if (canUseThis)", 1
    )[0]
    assert "matchingOption?.focus()" in ambiguity
    assert 'matchingOption?.scrollIntoView({ block: "nearest" })' in ambiguity
    assert ".click()" not in ambiguity
    assert "<output" in shell
    assert 'aria-live="polite"' in shell
    assert 'aria-atomic="true"' in shell
    assert 't("picker.multipleMatches", { noun })' in shell

    exact_commit = selector.split("const commitExactQuery = useCallback", 1)[1]
    exact_commit = exact_commit.split("const display =", 1)[0]
    assert "if (tab === PICKER_TAB.hub)" in exact_commit
    assert "hubResourceIdsEqual(candidate.id, query)" in exact_commit
    assert "resolveExactDatasetDeviceItem(query, deviceItems)" in exact_commit
    assert 'resolution.kind === "ambiguous"' in exact_commit
    assert "resolution.firstItem.repoId" in exact_commit
    assert "resolution.firstItem.path" in exact_commit
    assert 'resolution.kind === "none"' in exact_commit
    assert "selectLocalDataset(item.path)" in exact_commit
    assert "onExactQueryCommit={commitExactQuery}" in selector

    model_exact_commit = model_selector.split("function commitExactQuery", 1)[1]
    model_exact_commit = model_exact_commit.split("const display = selectedModel", 1)[0]
    assert "if (tab === PICKER_TAB.hub)" in model_exact_commit
    assert "findCanonicalHubResourceId(query, hubResultIds)" in model_exact_commit
    assert "resolveExactTrainModelDeviceItem(" in model_exact_commit
    assert 'resolution.kind === "ambiguous"' in model_exact_commit
    assert "focusValue: resolution.firstItem.path" in model_exact_commit
    assert 'resolution.kind === "none"' in model_exact_commit
    assert "pickDeviceModel(resolution.item)" in model_exact_commit
    device_picker = model_selector.split("function pickDeviceModel", 1)[1]
    device_picker = device_picker.split("function commitExactQuery", 1)[0]
    assert "knownCached: model.knownCached" in device_picker
    assert "localPath: model.localPath" in device_picker
    assert "modelFormat: model.modelFormat" in device_picker
    assert "onExactQueryCommit={commitExactQuery}" in model_selector
    assert "cacheLocalPathMatchesSelection(item.path, candidate)" in model_view_model


def test_cached_dataset_keyboard_commit_preserves_canonical_hub_identity():
    selector = _read("features/dataset-picker/components/dataset-selector.tsx")
    resolver = selector.split("function resolveExactDatasetDeviceItem", 1)[1]
    resolver = resolver.split("function hasExactDatasetMatch", 1)[0]
    assert "hubResourceIdsEqual(item.repoId, candidate)" in resolver
    assert "cacheLocalPathMatchesSelection(item.path, candidate)" in resolver
    assert "title: (item) => item.title" in resolver

    commit = selector.split("const commitExactQuery = useCallback", 1)[1]
    commit = commit.split("const display =", 1)[0]
    cached_branch = commit.split('if (item.kind === "cached")', 1)[1]
    cached_branch = cached_branch.split("selectLocalDataset(item.path)", 1)[0]
    assert "selectHfDataset(item.repoId," in cached_branch
    assert "knownCached: true" in cached_branch
    assert "localPath: item.cachePath" in cached_branch
    assert "closePicker();" in cached_branch


def test_device_picker_title_resolution_preserves_ambiguity():
    if shutil.which("node") is None:
        pytest.skip("node not available")
    module_uri = (FRONTEND / "components/resource-picker/device-item-match.ts").resolve().as_uri()
    script = (
        f"import {{ resolveDevicePickerItem }} from {json.dumps(module_uri)};\n"
        'const items = [{"id":"a","title":"Same"},'
        '{"id":"b","title":"same"},{"id":"c","title":"Unique"}];\n'
        "const resolve = (query) => resolveDevicePickerItem({\n"
        "query,\n"
        "items,\n"
        "canonicalMatch: (item, candidate) => item.id === candidate,\n"
        "title: (item) => item.title,\n"
        "});\n"
        'process.stdout.write(JSON.stringify(["b","Unique"," same ","missing"].map(resolve)));\n'
    )
    result = subprocess.run(
        [
            "node",
            "--experimental-strip-types",
            "--no-warnings",
            "--input-type=module",
        ],
        input = script,
        text = True,
        capture_output = True,
        check = False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        {"kind": "match", "item": {"id": "b", "title": "same"}},
        {"kind": "match", "item": {"id": "c", "title": "Unique"}},
        {"kind": "ambiguous", "firstItem": {"id": "a", "title": "Same"}},
        {"kind": "none"},
    ]


def test_run_preview_reuses_local_inventory_display_names():
    display_names = _read("features/studio/hooks/use-training-resource-display-names.ts")
    model_display = _read("features/model-picker/lib/train-model-selection-display.ts")
    dataset_display = _read("features/dataset-picker/lib/display.ts")
    preview = _read("features/studio/wizard/run-preview-card.tsx")

    assert 'useDeviceInventorySources(["localModels"]' in display_names
    assert 'useDeviceInventorySources(["localDatasets"]' in display_names
    assert "buildLocalInventoryRows(localModels)" in display_names
    assert ".map(toTrainModelDisplayCandidate)" in display_names
    assert "title: item.label || item.id" in display_names
    assert "trainModelSelectionDisplayName({" in display_names
    assert "datasetSelectionDisplayName({" in display_names
    assert "trainModelDisplayCandidateMatchesSelection({" in model_display
    assert "cacheLocalPathMatchesSelection(candidate.path, uploadedFile)" in dataset_display
    assert "useTrainingResourceDisplayNames({" in preview
    assert "resourceDisplayNames.modelName" in preview
    assert "displayName: resourceDisplayNames.datasetName" in preview


def test_training_model_lookups_preserve_platform_path_identity():
    lookup = _read("features/training/lib/training-picker-lookups.ts")
    assert "map.set(normalizeModelIdentity(value), row);" in lookup
    assert "value.toLowerCase()" not in lookup

    selector = _read("features/model-picker/components/train-model-selector.tsx")
    hub_pick = selector.split("function pickHubModel", 1)[1]
    hub_pick = hub_pick.split("function pickFreeformModel", 1)[0]
    device_pick = selector.split("function pickFreeformModel", 1)[1]
    device_pick = device_pick.split("function pickDeviceModel", 1)[0]
    assert "const key = normalizeModelIdentity(hubId.id);" in hub_pick
    assert "const key = normalizeModelIdentity(localPath);" in device_pick

    identity = _read("features/hub/lib/model-identity.ts")
    normalizer = identity.split("export function normalizeModelIdentity", 1)[1]
    normalizer = normalizer.split("export function normalizeGgufVariantIdentity", 1)[0]
    assert "return trimmed.toLowerCase();" in normalizer
    assert "WINDOWS_DRIVE_PATH_RE.test(trimmed)" in normalizer
    assert 'slashPath.startsWith("//")' in normalizer
    assert "WSL_DRIVE_PATH_RE.test(slashPath)" in normalizer


def test_model_identity_normalizes_cross_platform_trailing_separators():
    if shutil.which("node") is None:
        pytest.skip("node not available")
    module_uri = (FRONTEND / "features/hub/lib/model-identity.ts").resolve().as_uri()
    inputs = [
        "/opt/Models/Foo/",
        "/opt/Models/Foo",
        "/opt/models/foo",
        "C:\\Models\\Foo\\",
        "C:\\",
        "//Server/Share/Foo/",
        "\\\\Server\\Share\\Foo\\",
        "//Server/Share/",
        "/mnt/C/Models/Foo/",
        "/mnt/C/",
        "/",
        "./",
        "../",
        "~/",
        r".\models\demo",
        "./models/demo",
        "..\\models\\demo\\",
    ]
    expected = [
        "/opt/Models/Foo",
        "/opt/Models/Foo",
        "/opt/models/foo",
        "c:/models/foo",
        "c:/",
        "//server/share/foo",
        "//server/share/foo",
        "//server/share",
        "/mnt/c/models/foo",
        "/mnt/c",
        "/",
        ".",
        "..",
        "~",
        "./models/demo",
        "./models/demo",
        "../models/demo",
    ]
    script = (
        f"import {{ normalizeModelIdentity }} from {json.dumps(module_uri)};\n"
        f"const inputs = {json.dumps(inputs)};\n"
        "process.stdout.write(JSON.stringify(inputs.map(normalizeModelIdentity)));\n"
    )
    result = subprocess.run(
        [
            "node",
            "--experimental-strip-types",
            "--no-warnings",
            "--input-type=module",
        ],
        input = script,
        text = True,
        capture_output = True,
        check = False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected


def test_dataset_display_name_handles_cross_platform_trailing_separators():
    source = _read("features/dataset-picker/lib/display.ts")
    display_name = _read("components/resource-picker/dataset-display-name.ts")
    assert "import { datasetDisplayName }" in source
    assert "datasetSelectionDisplayName({" in source
    assert "cacheLocalPathMatchesSelection(candidate.path, uploadedFile)" in source
    assert 'value.replaceAll("\\\\", "/")' in display_name
    assert '.split("/")' in display_name
    assert ".filter(Boolean)" in display_name
    assert "UPLOADED_DATASET_HASH_PREFIX_RE" in display_name
    assert 'parts.lastIndexOf("parquet-files")' in display_name


def test_local_model_trigger_uses_cross_platform_device_display_name():
    selector = _read("features/model-picker/components/train-model-selector.tsx")
    view_model = _read("features/model-picker/components/train-model-picker-view-model.ts")
    display_source = _read("features/model-picker/lib/train-model-selection-display.ts")
    display = display_source.split("export function trainModelSelectionDisplayName", 1)[1].split(
        "export function toTrainModelDisplayCandidate", 1
    )[0]

    assert "isLocalTrainingModelSelection({" in display
    assert "trainModelDisplayCandidateMatchesSelection({" in display
    assert "selectedTitle ??" in display
    assert "pathDisplayName(selectedLocalPath?.trim() || selectedModel)" in display
    assert "return repoOf(selectedModel)" in display
    assert "trainModelSelectionDisplayName({" in selector
    assert "localRows.map(toTrainModelDisplayCandidate)" in selector
    assert "candidates: localModelDisplayCandidates" in selector
    assert "extends TrainModelDisplayCandidate" in view_model
    assert "...toTrainModelDisplayCandidate(row)" in view_model


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
    assert 'runDatasetCheck(datasetId, "train")' in hf_selection

    upload_selection = store.split("const selectLocalDatasetInternal", 1)[1]
    upload_selection = upload_selection.split("const selectS3SourceInternal", 1)[0]
    assert "createUploadBrowseDatasetSelection(uploadedFile)" in upload_selection

    s3_selection = store.split("const selectS3SourceInternal", 1)[1]
    s3_selection = s3_selection.split("const restoreBrowseDatasetSourceInternal", 1)[0]
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

    persistence = _read("features/training/stores/training-config-persistence.ts")
    assert "TRAINING_CONFIG_PERSISTENCE_VERSION = 16" in persistence
    migration = persistence.split("if (version < 14)", 1)[1]
    migration = migration.split("export function migrateTrainingConfig", 1)[0]
    assert "createUploadBrowseDatasetSelection(uploadedFile)" in migration
    assert "createHfBrowseDatasetSelection(dataset" in migration
    assert 'state.isEmbeddingModel = state.modelType === "embeddings";' in migration
    assert "state.datasetUserTemplate = undefined;" in migration
    assert "state.datasetAssistantTemplate = undefined;" in migration

    toggle = _read("features/studio/sections/dataset-source-toggle.tsx")
    assert "restoreBrowseDatasetSource();" in toggle
    assert "selectLocalDataset(uploadedFile)" not in toggle
    assert "selectHfDataset(dataset)" not in toggle


def test_training_picker_segmented_controls_share_full_height_geometry():
    segmented = _read("components/segmented-control.tsx")
    picker_tabs = _read("components/resource-picker/picker-tab-toggle.tsx")

    assert 'role="radiogroup"' in segmented
    assert "aria-label={ariaLabel}" in segmented
    assert "<fieldset" not in segmented
    assert "<legend" not in segmented
    assert "relative z-10 flex h-full min-w-0 flex-1" in segmented
    assert "hub-tab-toggle relative inline-flex h-9 w-full" in picker_tabs
    assert 'className="inset-y-0 start-0"' in picker_tabs
    assert "inline-flex h-full flex-1" in picker_tabs
    assert "text-ui-12p5 font-medium" in picker_tabs
    assert "h-8 w-full" not in picker_tabs
    assert "h-7 flex-1" not in picker_tabs


def test_training_picker_localizes_semantic_inventory_sources():
    inventory_types = _read("features/hub/inventory/types.ts")
    inventory = _read("features/hub/inventory/use-hub-inventory.ts")
    dataset_selector = _read("features/dataset-picker/components/dataset-selector.tsx")
    model_selector = _read("features/model-picker/components/train-model-selector.tsx")
    model_view = _read("features/model-picker/components/train-model-picker-view-model.ts")

    assert 'datasetSource?: "recipe" | "upload";' in inventory_types
    assert "datasetSource:" in inventory
    assert 'ds.source === "recipe" || ds.source === "upload"' in inventory
    assert "d.sourceLabel" not in dataset_selector
    for key in ("sourceRecipe", "sourceUpload", "sourceLocal"):
        assert f"studio.datasetPicker.{key}" in dataset_selector

    assert "function localModelSourceLabelKey(" in model_selector
    assert "localModelSourceLabelKey(row.source)" in model_selector
    assert "sourceLabel: row.sourceLabel" not in model_view
    assert "sourceLabel: string" in model_view


def test_training_validation_returns_translation_keys_and_rechecks_hub_ids():
    validation = _read("features/training/lib/validation.ts")
    start = _read("features/training/lib/start-fresh-training-run.ts")
    readiness = _read("features/training/hooks/use-training-readiness.ts")
    cta = _read("features/studio/wizard/start-training-cta.tsx")

    assert "errorKey: TranslationKey" in validation
    assert "message: string" not in validation
    assert "validateHubResourceId(config.dataset)" in validation
    assert "translate(validation.errorKey)" in start
    assert "configValidation.errorKey" in readiness
    assert "t(configValidation.errorKey)" in cta


def test_legacy_dataset_setter_still_schedules_modality_check():
    store = _read("features/training/stores/training-config-store.ts")
    legacy_setter = store.split("setDataset: (dataset) =>", 1)[1].split("setDatasetSubset:", 1)[0]

    assert "const datasetId = dataset?.trim() || null;" in legacy_setter
    assert 'runDatasetCheck(datasetId, "train")' in legacy_setter


def test_s3_training_payload_excludes_remembered_browse_sources():
    mapper = " ".join(_read("features/training/api/mappers.ts").split())
    assert 'const s3 = config.datasetSource === "s3" ? config.s3Config : null;' in mapper
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


def test_missing_dataset_cache_fallback_clears_remembered_browse_selection():
    source = _read("features/training/lib/start-fresh-training-run.ts")
    helper = source.split("function clearMissingDatasetCacheReference", 1)[1]
    helper = helper.split("function getDatasetName", 1)[0]
    for needle in (
        "datasetKnownCached: false,",
        "datasetLocalPath: null,",
        "browseDatasetSelection:",
        'source: "huggingface",',
        "dataset: datasetName,",
        "knownCached: false,",
        "localPath: null,",
    ):
        assert needle in helper


def test_start_cta_prioritizes_current_blockers_over_previous_start_errors():
    source = _read("features/studio/wizard/start-training-cta.tsx")
    resolver = source.split("function resolveStartTrainingError", 1)[1]
    resolver = resolver.split("function resolveStartTrainingButtonLabel", 1)[0]
    incompatibility = resolver.index("if (isIncompatible)")
    model_error = resolver.index("if (modelError)")
    start_error = resolver.index("if (startError)")
    dataset_warning = resolver.index(
        'return datasetUnverified ? t("studio.training.datasetUnverified") : null;'
    )
    assert incompatibility < start_error < model_error < dataset_warning
    assert "{showsModelWarning && (" in source
    assert "const showsModelWarning =" in source
    assert "!!modelError &&" in source
    assert "!startError &&" in source
    assert "!isIncompatible &&" in source
    assert "configValidation.ok &&" in source


def test_training_config_changes_clear_previous_start_error():
    source = _read("features/training/hooks/use-training-runtime-lifecycle.ts")
    assert "useTrainingConfigStore.subscribe" in source
    assert "state.userEditRevision !== previousState.userEditRevision" in source
    assert "useTrainingConfigStore.subscribe(clearStartError)" not in source
    assert "runtime.startError !== null" in source
    assert "runtime.setStartError(null)" in source


def test_training_auto_method_selection_settles_before_readiness():
    source = _read("features/training/stores/training-config-store.ts")
    loader = source.split("const loadAndApplyModelDefaults = (", 1)[1]
    loader = loader.split("const runDatasetCheck =", 1)[0]

    assert "const autoSelectionPromise =" in loader
    assert "isLoadingModelDefaults: autoSelectionPromise !== null" in loader
    settlement = loader.split("if (autoSelectionPromise)", 1)[1]
    assert "trainingMethod: method" in settlement
    assert "isLoadingModelDefaults: false" in settlement


def test_training_persistence_excludes_actions_and_sanitizes_method():
    source = _read("features/training/stores/training-config-persistence.ts")
    preview = _read("features/studio/wizard/run-preview-card.tsx")
    partialize = source.split("export function partializeTrainingConfig", 1)[1]
    partialize = partialize.split("type PersistedTrainingConfig", 1)[0]

    assert 'typeof value === "function"' in partialize
    assert "isTrainingMethod(persistedState.trainingMethod)" in source
    assert (
        "TRAINING_METHOD_META[trainingMethod] ?? TRAINING_METHOD_META.qlora" in preview
    )


def test_training_setup_changed_error_uses_localization():
    runtime = _read("features/training/lib/training-start-runtime.ts")
    release = runtime.split("export function releaseTrainingStart", 1)[1]
    assert '"studio.training.setupChanged" satisfies TranslationKey' in runtime
    assert "translate(TRAINING_SETUP_CHANGED_ERROR)" in release
    assert "Training setup changed while it was being checked." not in runtime


def test_training_start_legacy_dataset_error_uses_localization():
    source = _read("features/training/lib/training-start-errors.ts")

    assert 'translate("studio.training.legacyDatasetScriptUnsupported")' in source
    assert "This Hub dataset relies on a legacy custom script" not in source


def test_training_start_attempt_is_bound_to_checked_inputs():
    source = _read("features/training/lib/start-fresh-training-run.ts")
    assert "private expectedConfig: TrainingConfigStore;" in source
    assert "private expectedInputs: TrainingStartInputs;" in source
    assert (
        source.count("this.expectedInputs = captureTrainingStartInputs(this.expectedConfig);") == 2
    )
    assert "trainingStartInputsEqual(" in source
    assert "!this.configInputsChanged()" in source
    assert "getHfToken() === this.expectedHfToken" in source
    assert source.count("abortIfInputsChanged()") >= 5
    assert "this.expectedConfig = { ...this.expectedConfig, ...update };" in source
    assert source.count("this.expectedConfig = useTrainingConfigStore.getState();") == 1
    assert "buildTrainingStartPayload(attempt.config, hfToken)" in source
    assert "buildTrainingStartPayload(useTrainingConfigStore.getState()," not in source
    assert "hasIncompatibleTrainingModalities(attempt.config)" in source
    assert "payload.model_known_cached = false;" not in source
    assert "payload.model_local_path = null;" not in source
    assert "payload.dataset_known_cached = false;" not in source
    assert "payload.dataset_local_path = null;" not in source

    readiness = _read("features/training/hooks/use-training-readiness.ts")
    assert "hasIncompatibleTrainingModalities(state)" in readiness


def test_training_start_payload_mapper_receives_token_explicitly():
    mapper = _read("features/training/api/mappers.ts")
    start = _read("features/training/lib/start-fresh-training-run.ts")

    assert "hfToken: string | null," in mapper
    assert "hf_token: hfToken," in mapper
    assert "getHfToken" not in mapper
    snapshot = start.split("function captureTrainingStartInputs", 1)[1].split(
        "type TrainingStartInputs", 1
    )[0]
    assert "buildTrainingStartPayload(config, null)" in snapshot
    assert "payload.hf_token = null" not in snapshot


def test_run_preview_uses_the_app_locale_for_numbers_and_plural_rules():
    source = _read("features/studio/wizard/run-preview-card.tsx")

    assert "const locale = useLocale();" in source
    assert "new Intl.NumberFormat(locale)" in source
    assert "new Intl.PluralRules(locale)" in source
    assert "pluralRules.select(lengthCount)" in source
    assert "numberFormatter.format(lengthCount)" in source
    assert "PREVIEW_COUNT_KEYS[lengthUnit]" in source
    assert 'one: "studio.preview.step"' in source
    assert 'one: "studio.preview.epoch"' in source
    assert 'few: "studio.preview.epochFew"' in source
    assert "numberFormatter.format(contextLength)" in source
    assert ".toLocaleString()" not in source


def test_freeform_device_model_keeps_local_path_intent():
    selector = _read("features/model-picker/components/train-model-selector.tsx")
    assert "return looksLikeLocalPath(trimmed) ? trimmed : `./${trimmed}`;" in selector
    resolver = selector.split("function resolveFreeformTrainingModelPick", 1)[1]
    resolver = resolver.split("export function TrainModelSelector", 1)[0]
    assert "{ knownCached: false, localPath, modelFormat: null }" in resolver
    freeform = selector.split("function pickFreeformModel", 1)[1]
    freeform = freeform.split("function pickDeviceModel", 1)[0]
    assert "const localPath = explicitLocalPath(id);" in freeform
    assert "resolveFreeformTrainingModelPick(" in freeform
    assert "pick(selection.id, selection.options, selection.modelTypeFlags);" in freeform


def test_freeform_model_validation_rejects_binary_peft_artifact():
    validation = _read("features/training/lib/freeform-model-validation.ts")
    assert r"model\.(?:safetensors|bin)" in validation


def test_dataset_picker_restricts_local_selection_to_inventory():
    selector = _read("features/dataset-picker/components/dataset-selector.tsx")
    assert "explicitLocalDatasetPath" not in selector
    assert "looksLikeLocalPath" not in selector
    commit = selector.split("const commitHubQuery", 1)[1]
    commit = commit.split("const commitExactQuery", 1)[0]
    assert "selectHubDataset(next)" in commit
    assert "selectLocalDataset" not in commit
    assert "isValidHubResourceId(activeQuery)" in selector


def test_filtered_device_datasets_render_an_explicit_empty_state():
    source = _read("features/dataset-picker/components/dataset-selector-lists.tsx")
    device_list = source.split("export function DatasetDeviceList", 1)[1]
    device_list = device_list.split("export function DatasetHubList", 1)[0]
    assert 't("studio.datasetPicker.noDatasetsFound")' in device_list
    assert not re.search(r"if \(hasQuery\)\s*\{\s*return null;", device_list)


def test_dataset_hub_list_retains_the_active_hf_selection():
    selector = _read("features/dataset-picker/components/dataset-selector.tsx")
    hub_items = selector.split("const hubItems = useMemo", 1)[1]
    hub_items = hub_items.split("const hubPagination", 1)[0]
    assert 'datasetSource !== "huggingface"' in hub_items
    assert "isValidHubResourceId(item.id)" in hub_items
    assert "isValidHubResourceId(dataset)" in hub_items
    assert "selectableResults.some((item) => hubResourceIdsEqual(item.id, dataset))" in hub_items
    assert "return [...selectableResults, { id: dataset }];" in hub_items
    assert "hasExactDatasetMatch(" in selector
    assert re.search(r"hasExactDatasetMatch\(\s*activeQuery,\s*tab,\s*hubItems,", selector)
    assert "<DatasetHubList" in selector
    assert "items={hubItems}" in selector


def test_filtered_hub_pages_render_empty_state_and_keep_pagination_sentinel():
    for rel, export_name, empty_collection, empty_message in (
        (
            "features/dataset-picker/components/dataset-selector-lists.tsx",
            "DatasetHubList",
            "items",
            't("studio.datasetPicker.noDatasetsFound")',
        ),
        (
            "features/model-picker/components/train-model-picker-lists.tsx",
            "TrainModelHubList",
            "ids",
            't("studio.modelPicker.noModelsFound")',
        ),
    ):
        source = _read(rel)
        hub_list = source.split(f"export function {export_name}", 1)[1]
        assert f"if ({empty_collection}.length === 0)" in hub_list
        assert "if (!hasQuery)" in hub_list
        assert f"{empty_collection}.length === 0 && hasQuery" in hub_list
        assert hub_list.count(empty_message) == 2
        assert "<PickerSearchError" in hub_list
        assert "compact={true}" in hub_list
        assert hub_list.count("<PickerHubPaginationFooter") == 2
        assert not re.search(
            rf"if \({empty_collection}\.length === 0\).*?" r"if \(hasQuery\)\s*\{\s*return null;",
            hub_list,
            re.S,
        )
    footer = _read("components/resource-picker/picker-pagination.tsx")
    assert '<div ref={sentinelRef} className="h-px" />' in footer


def test_infinite_scroll_observes_a_late_mounted_sentinel():
    source = _read("features/hub/hooks/use-hub-infinite-scroll.ts")
    assert re.search(r"const\s+\[sentinelNode,\s*setSentinelNode\]\s*=\s*useState", source)
    assert re.search(
        r"const\s+sentinelRef\s*=\s*useCallback\(.*?setSentinelNode\(",
        source,
        re.S,
    )
    assert re.search(r"observer\.observe\(\s*sentinelNode\s*\)", source)


def test_train_hub_selections_preserve_canonical_identity():
    helper = _read("components/resource-picker/hub-resource-id.ts")
    model_selector = _read("features/model-picker/components/train-model-selector.tsx")
    model_view = _read("features/model-picker/components/train-model-picker-view-model.ts")
    dataset_selector = _read("features/dataset-picker/components/dataset-selector.tsx")

    assert "first?.trim().toLowerCase()" in helper
    assert "second?.trim().toLowerCase()" in helper
    assert "findCanonicalHubResourceId(query, hubIds)" in model_view
    assert "findCanonicalHubResourceId(query, hubResultIds)" in model_selector
    assert ".filter(isValidHubResourceId)" in model_selector
    assert "isValidHubResourceId(selectedModel)" in model_selector
    assert "hubTrainingModelCandidate(canonicalId," in model_selector
    assert re.search(r"\bpick\(\s*canonicalId\s*,", model_selector)
    assert "const canonicalId = cached?.repoId ?? validation.id;" in dataset_selector
    assert "isValidHubResourceId(item.id)" in dataset_selector
    assert "hubResourceIdsEqual(candidate.id, query)" in dataset_selector


def test_train_hub_search_queries_are_not_gated_by_repo_id_validation():
    model_selector = _read("features/model-picker/components/train-model-selector.tsx")
    dataset_selector = _read("features/dataset-picker/components/dataset-selector.tsx")
    model_search = _read("features/hub/hooks/use-hub-model-search.ts")
    dataset_search = _read("features/hub/hooks/use-hub-dataset-search.ts")

    assert "useHubModelSearch(picker.debouncedHubQuery," in model_selector
    assert "useHubDatasetSearch(picker.debouncedHubQuery," in dataset_selector
    assert "validateHubResourceId(picker.debouncedHubQuery)" not in model_selector
    assert "validateHubResourceId(picker.debouncedHubQuery)" not in dataset_selector
    assert "validateHubResourceId" not in model_search
    assert "validateHubResourceId" not in dataset_search


def test_training_dataset_client_uses_the_canonical_hub_router():
    api = _read("features/training/api/datasets-api.ts")

    assert 'authFetch("/api/hub/datasets/check-format"' in api
    assert 'authFetch("/api/hub/datasets/upload"' in api
    assert 'authFetch("/api/hub/datasets/ai-assist-mapping"' in api
    assert 'authFetch("/api/datasets/' not in api


def test_pinned_training_model_retains_size_and_vram_metadata():
    selector = _read("features/model-picker/components/train-model-selector.tsx")
    helper = selector.split("function buildTrainModelVramViews", 1)[1].split(
        "export function TrainModelSelector", 1
    )[0]
    vram_map = selector.split("const vramMap = useMemo", 1)[1].split(
        "const hubPagination", 1
    )[0]

    assert "ids.map((id)" in helper
    assert "parseParamCountB(id)" in helper
    assert "buildModelVramMap(models" in helper
    assert "extractParamLabel(model.id)" in helper
    assert "buildTrainModelVramViews(" in vram_map
    assert "hubResultIds," in vram_map


def test_persisted_invalid_hub_selections_are_blocked_with_picker_errors():
    validation = _read("features/training/lib/validation.ts")
    assert "isLocalTrainingModelSelection({" in validation
    assert "validateHubResourceId(config.selectedModel)" in validation
    assert 'errorKey: "studio.modelPicker.reasonInvalidHubId"' in validation
    assert "validateHubResourceId(config.dataset)" in validation
    assert 'errorKey: "studio.datasetPicker.reasonInvalidHubId"' in validation


def test_new_model_selection_replaces_the_previous_model_type():
    inference = _read("features/training/lib/model-type-inference.ts")
    selector = _read("features/model-picker/components/train-model-selector.tsx")

    assert "export function inferTrainingModelTypeFromFlags" in inference
    assert "resolvePickerInferredModelType" not in inference
    assert "inferTrainingModelTypeFromFlags(inferredFlags)" in selector
    assert "modelType: s.modelType" not in selector


def test_picker_capabilities_survive_model_config_probe_failures():
    selector = _read("features/model-picker/components/train-model-selector.tsx")
    store = _read("features/training/stores/training-config-store.ts")
    persistence = _read("features/training/stores/training-config-persistence.ts")
    api = _read("features/training/api/models-api.ts")
    vision_probe = api.split("export async function checkVisionModel", 1)[1].split(
        "export async function checkEmbeddingModel", 1
    )[0]
    loader = store.split("const loadAndApplyModelDefaults = (", 1)[1]
    failure = loader.split(".catch((error) =>", 1)[1].split("const runDatasetCheck =", 1)[0]

    assert "...options," in selector
    assert "...inferredFlags," in selector
    assert 'options?.isVision ?? effectiveModelType === "vision"' in store
    assert 'options?.isAudio ?? effectiveModelType === "audio"' in store
    assert 'options?.isEmbedding ?? effectiveModelType === "embeddings"' in store
    non_persisted = persistence.split("const NON_PERSISTED_STATE_KEYS", 1)[1].split(
        "export function partializeTrainingConfig", 1
    )[0]
    assert '"isEmbeddingModel"' not in non_persisted
    assert "inferTrainingModelTypeFromFlags({" in failure
    assert "isAudio: state.isAudioModel," in failure
    assert "isEmbedding: state.isEmbeddingModel," in failure
    assert "isAudioModel: false" not in failure
    assert "isEmbeddingModel: false" not in failure
    assert "throw new Error(" in vision_probe
    assert "return false;" not in vision_probe


def test_model_format_follows_the_selected_cache_reference():
    store = _read("features/training/stores/training-config-store.ts")
    selection = store.split("const selectModelInternal = (", 1)[1].split(
        "return {", 1
    )[0]

    adapter_preservation = selection.split("const previousAdapterFormat =", 1)[1].split(
        "const patch:", 1
    )[0]
    assert "!selectionChanged" in adapter_preservation
    assert "selectedModel === previousModel" not in adapter_preservation
    assert "options?.modelFormat ?? previousAdapterFormat" in selection
    assert "previousAdapterFormat ?? options?.modelFormat" not in selection


def test_streaming_dataset_preflight_does_not_read_local_cache():
    source = _read("features/training/lib/start-fresh-training-run.ts")
    store = _read("features/training/stores/training-config-store.ts")
    check = source.split("async function checkSelectedDataset", 1)[1].split(
        "function needsManualMapping", 1
    )[0]
    modality = source.split("function applyDetectedDatasetModality", 1)[1].split(
        "function openManualMapping", 1
    )[0]
    background_check = store.split("const runDatasetCheck =", 1)[1].split(
        "const recheckSelectedDatasetForStreamingMode", 1
    )[0]
    streaming_setter = store.split("setDatasetStreaming: (datasetStreaming) =>", 1)[1].split(
        "setDatasetSliceStart:", 1
    )[0]

    assert "!config.datasetStreaming" in check
    assert "preferLocalCache ? config.datasetLocalPath : null" in check
    assert "requestedPreferLocalCache && !state.datasetStreaming" in background_check
    assert "isHfSelection && preferLocalCache" in background_check
    assert "recheckSelectedDatasetForStreamingMode(false)" in streaming_setter
    assert "recheckSelectedDatasetForStreamingMode(true)" in streaming_setter
    assert "attempt.updateConfig({" in modality
    assert "datasetStreaming: false" in modality
    assert "isDatasetImage: isImage" in modality
    assert "isDatasetAudio: isAudio" in modality
    assert "disabledForDetectedModality" in modality
    assert "return true;" in modality
    assert "attempt.cancel(" not in modality
    assert "TRAINING_SETUP_CHANGED_ERROR" not in modality
    assert "recheckCachedDataset" in source
    assert "return prepareSelectedDataset(attempt, hfToken);" in source
    assert "current.datasetStreaming && (isImage || isAudio)" in background_check
    assert "datasetStreaming: false" in background_check
    assert "disabledForDetectedModality" in background_check
    assert "recheckSelectedDatasetForStreamingMode(false)" in background_check
    cache_setter = store.split("setSelectedDatasetCacheReference: (dataset, localPath) =>", 1)[
        1
    ].split("ensureModelDefaultsLoaded:", 1)[0]
    assert "const cacheReferenceChanged =" in cache_setter
    assert "cacheReferenceChanged && !state.datasetStreaming" in cache_setter
    assert "recheckSelectedDatasetForStreamingMode(false)" in cache_setter


def test_cache_reconciliation_does_not_replace_edited_model_defaults():
    store = _read("features/training/stores/training-config-store.ts")
    loader = store.split("const loadAndApplyModelDefaults = (", 1)[1].split(
        "const runDatasetCheck =", 1
    )[0]
    cache_setters = store.split("setSelectedModelCacheReference:", 1)[1].split(
        "clearSelectedDatasetCacheReference:", 1
    )[0]

    assert "const requestedUserEditRevision = requestState.userEditRevision;" in loader
    assert "const canApplyTrainingDefaults = () =>" in loader
    assert "applyTrainingDefaults &&" in loader
    assert "get().userEditRevision === requestedUserEditRevision" in loader
    assert "shouldApplyTrainingDefaults" in loader
    failure = loader.split(".catch((error) =>", 1)[1]
    assert "...(canApplyTrainingDefaults()" in failure
    assert cache_setters.count("applyTrainingDefaults: canReapplyModelDefaults(") == 2
    assert "function canReapplyModelDefaults(" in store
    assert "_modelDefaultsEditBaseline.userEditRevision" in store
    assert "modelDefaultsAppliedFor: null" not in cache_setters


def test_embedding_payload_survives_legacy_persisted_state():
    mapper = _read("features/training/api/mappers.ts")

    assert 'config.isEmbeddingModel || config.modelType === "embeddings"' in mapper


def test_picker_shell_handles_ime_focus_and_short_viewports():
    shell = _read("components/resource-picker/picker-shell.tsx")

    assert "function isImeCompositionKey(" in shell
    assert "event.nativeEvent.isComposing" in shell
    assert "event.keyCode === 229" in shell
    assert shell.count("isImeCompositionKey(") >= 3
    assert "onCompositionStart" in shell
    assert "onCompositionEnd" in shell
    assert "onOpenAutoFocus" in shell
    assert "autoFocus" not in shell
    assert 'aria-label={t("picker.searchAriaLabel", { noun })}' in shell
    assert "max-h-(--radix-popover-content-available-height)" in shell
    assert "gap-0 overflow-hidden" in shell
    assert '"mt-2.5 flex min-h-0 flex-1 flex-col gap-2"' in shell
    assert '"min-h-0 max-h-[320px] flex-1 overflow-y-auto' in shell
    assert "function switchToDevice()" in shell
    assert "window.requestAnimationFrame" in shell
    assert "onSwitchDevice={switchToDevice}" in shell


def test_train_pickers_prewarm_inventory_and_use_shared_tab_constants():
    model_picker = _read("features/model-picker/components/train-model-selector.tsx")
    dataset_picker = _read("features/dataset-picker/components/dataset-selector.tsx")

    assert 'useHubInventory({ kind: "models" })' in model_picker
    assert 'useHubInventory({ kind: "datasets" })' in dataset_picker
    for picker in (model_picker, dataset_picker):
        assert 'from "@/components/resource-picker/picker-tab-state"' in picker
        assert 'tab === "hub"' not in picker
        assert 'tab === "device"' not in picker


def test_training_controls_expose_context_without_overriding_history_metadata():
    model_picker = _read("features/model-picker/components/train-model-selector.tsx")
    dataset_picker = _read("features/dataset-picker/components/dataset-selector.tsx")
    wizard = _read("features/studio/wizard/training-wizard.tsx")
    history = _read("features/studio/history-card-grid.tsx")

    assert 'aria-label={`${t("studio.wizard.modelLabel")}: ${' in model_picker
    assert 'aria-label={`${t("studio.wizard.datasetLabel")}: ${' in dataset_picker
    assert 'aria-label={`${t("studio.wizard.methodLabel")}: ${activeLabel}`}' in wizard
    assert '<SetupField label={t("studio.wizard.hfTokenLabel")}>' not in wizard
    assert "aria-labelledby={`${cardId}-title`}" in history
    assert "aria-describedby={`${cardId}-status ${cardId}-details ${cardId}-metrics`}" in history
    assert "aria-label={title}" not in history


def test_manual_training_method_wins_over_delayed_auto_selection():
    source = _read("features/training/stores/training-config-store.ts")

    assert "let _trainingMethodEditGeneration = 0;" in source
    loader = source.split("const loadAndApplyModelDefaults = (", 1)[1].split(
        "const runDatasetCheck =", 1
    )[0]
    assert "const trainingMethodEditGeneration = _trainingMethodEditGeneration;" in loader
    assert re.search(
        r"_trainingMethodEditGeneration\s*!==\s*trainingMethodEditGeneration",
        loader,
    )
    setter = source.split("setTrainingMethod: (trainingMethod) =>", 1)[1].split(
        "setDatasetSource:", 1
    )[0]
    assert "_trainingMethodEditGeneration += 1;" in setter


def test_selected_cache_references_flow_into_metadata_requests():
    model_api = _read("features/training/api/models-api.ts")
    store = _read("features/training/stores/training-config-store.ts")
    page = _read("features/studio/studio-page.tsx")
    preview = _read("features/studio/sections/dataset-preview-dialog.tsx")

    assert 'params.set("prefer_local_cache", "true")' in model_api
    assert 'params.set("local_path", options.localPath)' in model_api
    loader = store.split("const loadAndApplyModelDefaults = (", 1)[1].split(
        "const runDatasetCheck =", 1
    )[0]
    assert "requestedKnownCached" in loader
    assert "requestedLocalPath" in loader
    assert "requestMatchesSelection()" in loader
    assert "preferLocalCache," in loader
    assert "localPath: preferLocalCache ? requestedLocalPath : null" in loader

    assert "datasetKnownCached={config.datasetKnownCached}" in page
    assert "datasetLocalPath={config.datasetLocalPath}" in page
    assert "datasetStreaming={config.datasetStreaming}" in page
    assert "!datasetStreaming" in preview
    assert "preferLocalCache: previewRequest.preferLocalCache" in preview
    assert "localPath: previewRequest.localPath" in preview


def test_partial_local_datasets_are_not_selectable():
    selector = _read("features/dataset-picker/components/dataset-selector.tsx")
    device_items = selector.split("const deviceItems = useMemo<DatasetDeviceItem[]>", 1)[1].split(
        "const pickerView =", 1
    )[0]

    assert device_items.count(".filter((d) => !d.partial)") == 2


def test_dataset_panel_requires_a_valid_hub_id_without_filename_heuristics():
    selection = _read("features/training/lib/dataset-selection.ts")
    panel = _read("features/studio/sections/dataset-selection.tsx")
    helpers = _read("features/studio/sections/dataset-panel-helpers.ts")

    assert 'source !== "huggingface"' in selection
    assert "dataset?.trim()" in selection
    assert "isValidHubResourceId" in selection
    assert "isHuggingFaceDatasetSelected(" in panel
    assert "isLikelyLocalDatasetRef" not in panel
    assert "isLikelyLocalDatasetRef" not in helpers


def test_training_transport_failures_reconcile_with_the_backend_before_failing():
    api = _read("features/training/api/train-api.ts")
    runtime = _read("features/training/lib/training-start-runtime.ts")
    fresh = _read("features/training/lib/start-fresh-training-run.ts")
    resume = _read("features/training/lib/resume-training-run.ts")
    backend_models = _read_backend("models/training.py")
    backend_route = _read_backend("routes/training.py")
    backend_runtime = _read_backend("core/training/training.py")

    assert "getTrainingStartRequestStatus(" in runtime
    assert "START_RECONCILIATION_DELAYS_MS" in runtime
    assert "resolveTrainingStartRequestOutcome(" in runtime
    assert ".setStartPending(outcome.jobId, outcome.message)" in runtime
    assert "acknowledgeTrainingStartRequest(lease.startRequestId)" in runtime
    assert "isTrainingStartOutcomeUnknownError(error)" in fresh
    assert "await attempt.recoverTransportFailure()" in fresh
    assert "isTrainingStartOutcomeUnknownError(failure)" in resume
    assert "await reconcileTrainingStartTransportFailure(this.lease)" in resume
    assert "start_request_id: startRequestId" in api
    assert "retryNetworkErrors: false" in api
    assert "start_request_id: Optional[str]" in backend_models
    assert 'state: Literal["pending", "accepted", "rejected"]' in backend_models
    start_route = backend_route.split('async def start_training(', 1)[1]
    assert start_route.index("backend.reserve_start_request(") < start_route.index(
        "_reject_untrainable_model_request,"
    )
    assert 'state = "accepted"' in backend_route
    assert 'state = "rejected"' in backend_route
    backend_start = start_route.split("def _run_backend_start()", 1)[1].split(
        "start_task = asyncio.create_task", 1
    )[0]
    failed_backend_start = backend_start.split("if success:", 1)[1].split(
        "return success", 1
    )[0]
    assert failed_backend_start.count("_reject_start_request(") == 1
    failed_response = start_route.split("if not success:", 1)[1].split(
        "return TrainingJobResponse", 1
    )[0]
    assert "_reject_start_request(" not in failed_response
    assert "start_request_id = request.start_request_id" in backend_route
    assert "start_request_id = start_request_id" in backend_route
    assert "self._pending_start_request_id" in backend_runtime
    assert "self.current_start_request_id = start_request_id" in backend_runtime


def test_multitask_model_search_fans_out_concurrently_and_closes_iterators():
    source = _read("features/hub/hooks/use-hub-model-search.ts")
    merge = _read("features/hub/lib/merge-task-iterators.ts")

    assert "const cursors: TaskCursor<T>[] = [];" in merge
    assert "let pending = pull();" in merge
    assert "for (const task of taskList)" in merge
    assert "cursors.filter((candidate) => candidate.active)" in merge
    assert "const result = await cursor.take(" in merge
    assert "const controller = new AbortController();" in merge
    assert "combineAbortSignals(" in merge
    assert "controller.abort();" in merge
    assert "await Promise.allSettled(" in merge
    assert "iterator.return(undefined)" in merge
    assert "if (!yielded && failures.length > 0)" in merge
    assert "throw next.reason" not in merge
    assert source.count("mergeTaskIterators(") >= 4


def test_page_title_halo_uses_the_configured_accent():
    css = _read("index.css")
    halo = css.split(".page-title-halo {", 1)[1].split("}", 1)[0]

    assert "var(--primary)" in halo
    assert "#e4e3df" not in halo
    assert "#ebeae6" not in halo
    assert "#efeeea" not in halo


def test_only_gguf_configs_are_mirrored_to_the_server():
    """The server override map is read by the OpenAI-compatible auto-switch, and its
    resolver indexes GGUFs only."""
    src = " ".join(_read("features/model-picker/components/model-config-page.tsx").split())
    assert (
        "if ( !saveFailed && (target.apiLoadable ?? target.isGguf) && !nativePathToken ) "
        "{ syncModelOverride(" in src
    )
    # The local save is not behind the same gate.
    assert "if (remember) { saveFailed = !savePerModelConfig(" in src


def test_a_native_leased_gguf_is_not_mirrored_to_the_server():
    """A dropped or file-picked GGUF loads through a signed native-path lease, and
    /api/inference/status reports model_identifier as null for it, so the checkpoint the
    browser keys settings by is the bare file name the backend echoes back."""
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

    inference = _read_backend("routes/inference.py")
    assert (
        "model_identifier = None if _native_grant_backed else _model_id" in inference
    ), "why the checkpoint is only a display name"
    models = _read_backend("routes/models.py")
    assert "display_name = gguf_file.stem," in models, "why the name is never an index key"


def test_evicted_local_configs_drop_their_server_overrides():
    """savePerModelConfig evicts older models when the map exceeds its budget."""
    src = " ".join(_read("features/model-picker/components/model-config-page.tsx").split())
    assert "const evicted: { modelId: string; ggufVariant: string | null }[] = [];" in src
    assert (
        "for (const dropped of evicted) { syncModelOverride(dropped.modelId, "
        "dropped.ggufVariant, null, { keepLaunchFlags: true, }); }" in src
    )
    api = " ".join(_read("features/model-picker/api/model-overrides.ts").split())
    assert "keepLaunchFlags?: boolean;" in api
    # remove=false with no fields: the route re-supplies stored flags and drops an empty row.
    assert "remove: config === null && !options?.keepLaunchFlags," in api
    assert (
        "...(config === null && !options?.keepLaunchFlags ? { llama_extra_args: [] } : {}),"
        in api.replace("// biome-ignore lint/style/useNamingConvention: API schema ", "")
    )
    route = _read_backend("routes/settings.py")
    assert 'requested_extra_args = stored.get("llama_extra_args")' in route, "the rule this mirrors"

    # Eviction reports what it dropped as model id + variant, not the normalized storage key.
    store = " ".join(_read("features/model-picker/model-config/per-model-config.ts").split())
    assert "evicted?: { modelId: string; ggufVariant: string | null }[]" in store
    assert "modelIdFromStorageKey(" in store and "ggufVariantFromStorageKey(" in store


def test_backfill_compares_server_keys_by_normalized_identity():
    """app_settings has no schema version, so an install predating identity normalization
    holds rows keyed by whatever id was typed, e.g."""
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
    """A load, unload or download is recorded as a monitor entry but is not an HTTP call."""
    src = " ".join(_read("features/api-monitor/use-api-monitor.ts").split())
    assert 'if (entry.kind === "lifecycle") { continue; }' in src
    # "Requests" is a request count too, so it cannot stay entries.length.
    assert "total: requests," in src
    assert "total: entries.length" not in src

    backend = _read_backend("core/inference/api_monitor.py")
    assert 'entry.kind != "lifecycle"' in backend, "the rule this mirrors"


def test_api_reach_copy_is_limited_to_gguf_models():
    """The Hub opens this page for every downloaded model, but ModelConfigPage mirrors
    settings to the server only when target.isGguf, because API auto-switch indexes
    GGUFs only."""
    src = " ".join(_read("features/hub/catalog/hub-model-settings-view.tsx").split())
    assert "{(target.apiLoadable ?? target.isGguf)" in src
    assert "Saved settings apply everywhere Studio loads this model." in src


def test_backfill_includes_a_standalone_gguf_with_no_variant():
    """A standalone .gguf picked directly has no quant to choose between, so it is stored
    with a null variant."""
    src = " ".join(_read("features/model-picker/api/migrate-model-overrides.ts").split())
    assert 'entry.modelId.toLowerCase().endsWith(".gguf")' in src
    # Still excluded for safetensors, which auto-switch does not resolve.
    assert "entry.ggufVariant != null ||" in src


def test_monitor_overlay_does_not_pull_in_the_lazy_page():
    """The overlay is mounted from __root.tsx, so a static import of the page for two label
    helpers drags the whole 900-line page and its dependency graph into the eagerly
    loaded bundle and undoes the route's lazyRouteComponent."""
    overlay = _read("features/api-monitor/api-monitor-overlay.tsx")
    assert 'from "./lifecycle"' in overlay
    assert "api-monitor-page" not in overlay, "the overlay must not reach the page"
    # The helpers live in their own module, not re-exported through the page.
    shared = _read("features/api-monitor/lifecycle.ts")
    assert "export function isLifecycleEntry(" in shared
    assert "export function lifecycleLabel(" in shared


def test_override_writes_are_ordered_per_model():
    """Two saves for one model, or a save racing the one-time backfill, started independent
    requests with no sequencing, so the older response could commit last and resurrect
    the entry the newer one meant to replace."""
    src = " ".join(_read("features/model-picker/api/model-overrides.ts").split())
    assert "const writesByKey = new Map<string, Promise<void>>();" in src
    # Keyed by the same override key the server stores under.
    assert (
        "const key = modelOverrideKey( normalizeModelIdentity(modelId), normalizeGgufVariantIdentity(ggufVariant), );"
        in src
    )
    # Chained on the settled tail, so one failed write cannot cancel the next.
    assert "previous .catch(() => {}) .then(() => sendModelOverride(" in src
    # Only the last writer clears the slot, or a queue still building loses order.
    assert "if (writesByKey.get(key) === write) { writesByKey.delete(key); }" in src


def test_backfill_skips_future_schema_local_records():
    """loadPerModelConfig refuses to apply a record written by a newer Studio and eviction
    refuses to drop one, because this client cannot interpret that schema."""
    src = " ".join(_read("features/model-picker/model-config/per-model-config.ts").split())
    listing = src[src.index("export function listPerModelConfigs()") :]
    assert "storedConfigVersion(raw) > STORAGE_SCHEMA_VERSION" in listing[:900]


def test_detail_settings_need_a_resolved_quant():
    """The on-device card passes a null variant while its own lookup is pending or after it
    failed."""
    src = " ".join(_read("features/hub/hub-page.tsx").split())
    # `variant`, not the argument: a resident quant may have replaced it, so judge what is saved.
    guard = "if (!variant && selectedModel.isGguf && selectedModel.requiresVariant) {"
    assert guard in src
    assert src.count("Couldn't determine which quant to configure.") == 2


def test_a_failed_detail_fetch_is_retried():
    """A terminal row's updated_at never advances and selectedIsMissing stays true, so a
    fetch that failed had nothing left to re-run the effect and the payload stayed
    unavailable until another row was selected."""
    src = " ".join(_read("features/api-monitor/api-monitor-page.tsx").split())
    assert "const DETAIL_FETCH_ATTEMPTS = 3;" in src
    assert "const detailInFlight = selectedId_ != null && loadingDetails.has(selectedId_);" in src
    assert "if (attemptsRef.current.count >= DETAIL_FETCH_ATTEMPTS) { return; }" in src


def test_ollama_models_are_not_advertised_as_api_loadable():
    """local_model_resolver skips Ollama's scanner, so an Ollama GGUF is never in the
    auto-switch index and no OpenAI request can resolve it."""
    types_src = " ".join(_read("features/model-picker/components/model-selector/types.ts").split())
    assert "apiLoadable?: boolean;" in types_src
    hub = " ".join(_read("features/hub/hub-page.tsx").split())
    assert "row.source !== LOCAL_MODEL_SOURCE.OLLAMA" in hub
    assert "apiLoadable:" in hub
    backend = _read_backend("core/inference/local_model_resolver.py")
    assert (
        "Ollama's\n    scanner is skipped" in backend or "scanner is skipped" in backend
    ), "the rule this mirrors"


def test_cached_repo_settings_are_keyed_by_the_repo_id():
    """A repo cached outside the active HF cache reports load_id = the snapshot path
    (hub/services/cache_inventory.py), while the chat picker and the auto-switch index
    key it by repo_id."""
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
    assert 'if (kind !== "cache" && resource.source !== "hub_cache") {' in hub
    assert "return resource.repoId ?? resource.runId;" in hub
    # Both openers and the Hub's own load resolve through it.
    assert hub.count("modelConfigIdentity(") == 3
    assert 'const configId = row.kind === "cache" ? row.repoId : id;' in hub
    assert hub.count("configId,") >= 2

    backend = _read_backend("hub/tests/test_model_services.py")
    assert 'fields["load_id"] == str(snapshot)' in backend, "the rule this mirrors"


def test_backfill_splits_a_quant_suffix_the_way_the_backend_does():
    """The backfill compared server keys under an identity taken by splitting on the last
    colon, so a Windows drive letter and an ordinary colon inside a POSIX filename were
    read as quant separators: `/models/foo:Bar.gguf` and `/models/foo:bar.gguf` folded
    to one key, and whichever was already on the server made the other look migrated."""
    identity = " ".join(_read("features/model-picker/model-config/model-identity.ts").split())
    assert "export function splitQuantSuffix(" in identity
    # Two rules keep a path out: no separator in the tail, and a non-.gguf head carries no label.
    assert 'if (tail.includes("/") || tail.includes("\\\\"))' in identity
    assert 'if (!head.toLowerCase().endsWith(".gguf")) { return null; }' in identity
    # A .gguf head is not enough: the suffix must be the label the scanner derives, or a name
    # containing ".gguf:" folds two real POSIX files onto one key.
    assert "tail.toLowerCase() === ggufQuantLabel(filename).toLowerCase()" in identity

    migrate = " ".join(_read("features/model-picker/api/migrate-model-overrides.ts").split())
    assert "const split = splitQuantSuffix(key);" in migrate
    assert 'key.lastIndexOf(":")' not in migrate, "the unconditional split is gone"

    backend = _read_backend("utils/openai_auto_switch_settings.py")
    assert "def split_quant_suffix(" in backend, "the rule this mirrors"
    assert "_BPW_SUFFIX" in backend and "bpw" in identity
    assert "extract_quant_label(filename).casefold()" in backend, "the label rule"
    # Both sides accept the same quant vocabulary; the regex lives with the loader.
    quants = _read_backend("core/inference/llama_cpp.py")
    for token in ("MXFP", "IQ", "TQ", "BF16", "F16", "F32"):
        assert token in quants and token in identity, token
    # The .gguf label helpers are ported too, or the two sides label a filename apart.
    gguf = _read_backend("hub/utils/gguf.py")
    assert "def extract_quant_label(" in gguf and "def _gguf_stem(" in gguf
    assert "function ggufQuantLabel(" in identity and "function ggufStem(" in identity
    assert "_GGUF_SPLIT_SUFFIX_RE" in gguf and "GGUF_SPLIT_SUFFIX" in identity
    assert "_FLOAT_PRECISION_QUANTS" in gguf and "FLOAT_PRECISION_QUANTS" in identity
    # The executable half of this contract, case by case against split_quant_suffix.
    assert (WORKDIR / "studio" / "frontend" / "tests" / "model-identity.test.ts").is_file()


def test_the_detail_card_also_gates_ollama_out_of_the_api_promise():
    """Settings opens from two places in the Hub."""
    hub = " ".join(_read("features/hub/hub-page.tsx").split())
    assert hub.count("LOCAL_MODEL_SOURCE.OLLAMA") == 2
    assert "selectedModel.localSource !== LOCAL_MODEL_SOURCE.OLLAMA" in hub
    assert "row.source !== LOCAL_MODEL_SOURCE.OLLAMA" in hub


def test_the_settings_page_judges_the_config_storage_actually_keeps():
    """savePerModelConfig normalizes before deciding, and the runtime hands this page
    Speculative Decoding "auto", which canonicalizes to null."""
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
    """A settings target opened from the Chat model picker carried no apiLoadable, so the
    `??"""
    picker = " ".join(_read("features/model-picker/components/model-selector.tsx").split())
    assert "apiLoadable: isGguf && !isOllamaLinkPath(id)," in picker
    sidebar = " ".join(_read("features/model-picker/components/sidebar-model-config.tsx").split())
    assert "apiLoadable: isGguf && !isOllamaLinkPath(modelId)," in sidebar
    # The same classification gates the backfill, or an older config still reaches the server.
    backfill = " ".join(_read("features/model-picker/api/migrate-model-overrides.ts").split())
    assert "!isOllamaLinkPath(entry.modelId) &&" in backfill

    identity = _read("features/hub/lib/model-identity.ts")
    assert 'new Set([".studio_links", "ollama_links"])' in identity
    resolver = _read_backend("core/inference/local_model_resolver.py")
    assert 'seg in (".studio_links", "ollama_links")' in resolver, "the rule this mirrors"


def test_the_backfill_fills_in_fields_rather_than_skipping_known_keys():
    """The backfill reads the override map once and then writes each model in turn, so a
    save by another tab during that pass was overwritten by this browser's older
    localStorage copy."""
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

    route = _read_backend("routes/settings.py")
    assert "fill_absent_fields: bool = False" in route, "the rule this mirrors"
    assert "fill_absent_fields = payload.fill_absent_fields," in route
    # A write mode must not leak into saved fields, or model_id-only removal stops working.
    assert '"remove", "fill_absent_fields"' in route

    # The merge is the server's, in the write's transaction: a client-side one reopens the race.
    db = _read_backend("storage/studio_db.py")
    assert "merged = {**entry_value, **stored}" in db
    assert "BEGIN IMMEDIATE" in db


def test_the_hub_settings_page_matches_a_resident_path_loaded_model():
    """A GGUF loaded from an inactive HF cache or straight off disk loads by path, but
    /status reports the clean public id, so comparing it to settingsTarget.id said "not
    loaded" and the page showed saved or default values instead of the live launch
    config."""
    hub = " ".join(_read("features/hub/hub-page.tsx").split())
    assert (
        "residentModelIdMatches( activeCheckpoint, settingsTarget.id, settingsTarget.configId, )"
        in hub
    )
    assert "loadedConfig={settingsTargetIsResident ? activeModelConfig : null}" in hub
    assert "settingsTargetIsResident ? activeGgufContextLength : null" in hub
    # The loadable identifier, as every other status reader records it.
    assert "checkpointId: resolveInferenceCheckpointId(status)," in hub
    assert "setCheckpoint(status.active_model" not in hub
    chat = " ".join(_read("features/chat/lib/apply-inference-status-to-store.ts").split())
    assert "return status.model_identifier ?? status.active_model;" in chat, "the rule this mirrors"
    # The alias is the backend's own public id rule, not a private heuristic.
    identity = _read("features/hub/lib/model-identity.ts")
    assert "export function publicModelId(" in identity
    assert "models--" in identity and "snapshots" in identity
    # Only a namespaced repo id names one model, so a bare stem must never stand in for it.
    assert 'return publicId.includes("/") && modelIdsMatch(active, publicId);' in identity
    backend = _read_backend("core/inference/model_ids.py")
    assert "def public_model_id(" in backend, "the rule this mirrors"


def test_the_hub_hydrates_the_live_settings_before_it_offers_them():
    """The Hub builds activeModelConfig out of the chat runtime store, and landing straight
    on /hub is the one entry point where nothing has applied /api/inference/status yet:
    useChatModelRuntime has no mount sync and the chat page is a different route."""
    hub = " ".join(_read("features/hub/hub-page.tsx").split())
    assert "adoptResidentModelStatus(" in hub
    assert "applyActiveModelStatusToStore(status, {" in hub
    assert "previousCheckpoint: previous.checkpoint ?? undefined," in hub
    assert "previousGgufVariant: previous.ggufVariant," in hub
    assert "modelLoading: store.modelLoading," in hub

    adopt = " ".join(_read("features/hub/lib/adopt-inference-status.ts").split())
    # Unconditional: a persisted checkpoint rehydrates without the fields saying how it launched.
    assert "actions.applyStatus(previous); return true;" in adopt
    # Never fight the owning load, nor describe an external model with the resident's settings.
    assert "if (state.checkpointIsExternal) { return false; }" in adopt
    assert "if (state.modelLoading) { return false; }" in adopt

    # Same call the chat runtime's own refresh makes, which is the rule this mirrors.
    runtime = " ".join(_read("features/chat/hooks/use-chat-model-runtime.ts").split())
    assert "applyActiveModelStatusToStore(statusRes, {" in runtime


def test_the_hub_settings_editor_reseeds_when_the_live_config_lands():
    """ModelConfigPage reads loadedConfig in a useState initializer, so it seeds once per
    mounted instance."""
    view = " ".join(_read("features/hub/catalog/hub-model-settings-view.tsx").split())
    assert "key={modelConfigInstanceKey( target.id, target.ggufVariant, loadedConfig, )}" in view
    # Same key the sidebar entry uses; that parity is the point.
    sidebar = " ".join(_read("features/model-picker/components/sidebar-model-config.tsx").split())
    assert "key={modelConfigInstanceKey(modelId, settingsGgufVariant, loadedConfig)}" in sidebar

    signature = " ".join(_read("features/model-picker/model-config/config-signature.ts").split())
    # "No live config yet" needs its own value: that transition is the one that must remount.
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
    """The inventory labels a single scanned .gguf from its filename, so the Hub row menu
    keyed its settings to `<path>:Q4_K_M` while the Chat picker, the detail card and the
    backfill all use the bare path: two surfaces, two configs."""
    hub = " ".join(_read("features/hub/hub-page.tsx").split())
    assert "let ggufVariant = settingsGgufVariantForRow(row);" in hub
    assert "row.formatVariant" not in hub, "the row's raw label is not a settings key"

    helper = " ".join(_read("features/hub/inventory/settings-identity.ts").split())
    assert 'row.kind === "local" && row.path.toLowerCase().endsWith(".gguf")' in helper

    common = _read_backend("hub/services/models/common.py")
    # The rule this mirrors: a variant is derived only for a single scanned file.
    assert "extract_quant_label(gguf_files[0].name)" in common
    assert "if scan_path.is_file() and len(gguf_files) == 1" in common


def test_a_standalone_gguf_is_resident_despite_its_derived_quant():
    """A loose .gguf keys its settings by the bare path with no variant, but the loader
    derives one from the filename (llama_cpp sets _hf_variant from _extract_quant_label)
    and /status reports it, so an equality between the two could never hold and the
    settings page withheld the live launch config from the very file that was loaded."""
    hub = " ".join(_read("features/hub/hub-page.tsx").split())
    assert "const settingsTargetIsStandaloneFile =" in hub
    assert 'settingsTarget.id.toLowerCase().endsWith(".gguf")' in hub
    assert (
        "(settingsTargetIsStandaloneFile || ggufVariantsMatch(activeGgufVariant, settingsTarget.ggufVariant))"
        in hub
    )
    backend = _read_backend("core/inference/llama_cpp.py")
    assert (
        "self._hf_variant = _extract_quant_label(gguf_path)" in backend
    ), "the derived label this accounts for"


def test_a_standalone_gguf_has_one_settings_identity_everywhere():
    """A loose .gguf has no quant to choose between, but llama_cpp falls back to
    _extract_quant_label(gguf_path) when a load names no variant, and /status echoes
    that as gguf_variant."""
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
    # The suffix, and something that names a file on this machine: see
    # test_a_repo_id_ending_in_gguf_keeps_its_quant for why the suffix is not enough.
    assert "GGUF_SUFFIX_RE.test(modelId)" in identity
    row_identity = _read("features/hub/inventory/settings-identity.ts")
    assert 'row.path.toLowerCase().endsWith(".gguf")' in row_identity

    # The precedence that makes the bare path the one that wins.
    route = _read_backend("routes/inference.py")
    assert 'f"{target_id}:{file_variant}" if file_variant else None,' in route
    bare = route.index("\n                            target_id,\n")
    labelled = route.index('f"{target_id}:{file_variant}"')
    assert bare < labelled, "the bare path must be read before the filename label"


def test_monitor_unload_clears_only_the_model_it_freed():
    """Unload targets the resident local model from /status, but the store may hold either
    spelling: status reports the concrete load path while the store can hold the
    advertised repo id.

    The read/unload/recheck sequence itself is pinned behaviourally by
    studio/frontend/tests/api-monitor-unload-resident.test.ts; this only holds the page to
    delegating it, since a single-pass unload reports success over a model an API
    auto-switch loaded under the click."""
    page = " ".join(_read("features/api-monitor/api-monitor-page.tsx").split())
    assert (
        "aliases: [checkpoint, status.active_model].filter( (alias): alias is string "
        "=> alias != null, )," in page
    )
    assert "!isExternalModelId(selected)" in page
    assert "unloadedAliases.some((alias) => modelIdsMatch(selected, alias))" in page
    assert "store.clearCheckpoint();" in page


def test_settings_open_reads_status_before_resolving_the_quant():
    """A cache row carries no quant, so opening its settings resolves one from the store's
    active variant."""
    page = " ".join(_read("features/hub/hub-page.tsx").split())
    assert "const refreshResidentModelStatus = useCallback((): Promise<void> => {" in page
    assert (
        "await refreshResidentModelStatus(); if (settingsOpenSeq.current !== openSeq) return;"
        in page
    )
    # Three: both handlers read on entry, and openModelSettings reads again after the
    # variant lookup, whose network round trip is its own window for a switch to land.
    assert page.count("await refreshResidentModelStatus();") == 3


def test_a_local_quant_folder_resolves_its_variants_by_path():
    """A local row carries a repo id only inside the HF cache, so a plain folder of
    quants has none while still being marked as needing one, and the row menu's
    Settings could then only reach the error toast."""
    hub = " ".join(_read("features/hub/hub-page.tsx").split())
    assert (
        'const repoId = row.kind === "cache" ? row.repoId : (row.repoId ?? row.path ?? null);'
        in hub
    )
    # The on-device card already lists by path, so both surfaces choose from one set of quants.
    card = " ".join(_read("features/hub/catalog/local-on-device-card.tsx").split())
    assert "repoId: modelId, hfToken, preferLocalCache: true, localPath: localGgufPath," in card
    # The backend scans a path in the repo_id position before the validation that would 400.
    variants = _read_backend("hub/services/models/gguf_variants.py")
    scan = variants.split("if is_local_path(repo_id):", 1)
    assert len(scan) == 2, "the local-path branch this leans on"
    assert "_is_valid_repo_id(repo_id)" in scan[1], "the branch has to come first"


def test_no_settings_target_is_built_on_an_unread_store():
    """Every path out of both handlers seeds the editor from the store, and Apply
    reloads with what it seeded, so a target built before the status read lands can
    persist and launch the settings of the model an API switch displaced."""
    page = " ".join(_read("features/hub/hub-page.tsx").split())
    # Both handlers read first and drop the open if a newer one started meanwhile, and
    # the variant lookup's await gets the same treatment. Every read carries the guard.
    assert page.count("await refreshResidentModelStatus();") == 3
    assert (
        page.count(
            "await refreshResidentModelStatus(); if (settingsOpenSeq.current !== openSeq) return;"
        )
        == 3
    )
    # Nothing may build a target off a concurrent read instead of an awaited one.
    assert "Promise.all([ listGgufVariants(" not in page


def test_an_empty_status_is_read_against_the_idle_unload_setting():
    """/status cannot say whether an empty answer is an idle eviction that reloads
    or a real unload, so the Hub reads the only endpoint that knows and keeps the
    checkpoint only while the loop is armed."""
    hub = " ".join(_read("features/hub/hub-page.tsx").split())
    adopt = " ".join(_read("features/hub/lib/adopt-inference-status.ts").split())
    # Awaited, not raced: the first status read is the one most likely to land on an evicted
    # model, and an unresolved default of false would clear a checkpoint that is coming back.
    assert (
        "Promise.all([getInferenceStatus(), readIdleUnloadArmed()]) "
        ".then(([status, idleUnloadArmed]) => {" in hub
    )
    assert "idleUnloadArmed," in hub
    # Read with every status read, not cached for the life of the page: the idle timeout
    # is editable from Settings while this page stays mounted.
    assert "idleUnloadRead.current ??=" not in hub
    assert "idleUnloadArmed.current = settings.idleUnloadActive;" in hub
    # A failed read keeps the last answer. The default is disarmed, which is the side
    # that clears the checkpoint, so falling back to it would drop a live selection.
    assert ".catch(() => idleUnloadArmed.current)" in hub
    assert "if (state.idleUnloadArmed) { return false; }" in adopt
    assert "actions.clearCheckpoint?.();" in adopt


def test_a_repo_id_ending_in_gguf_keeps_its_quant():
    """Repo ids ending in .gguf are real on the Hub, an iMat repo among them, and
    those hold every quant. A suffix-only test reads one as a single file, drops
    the variant, and saves Q4 and Q8 under one key."""
    identity = " ".join(_read("features/hub/lib/model-identity.ts").split())
    assert "if (modelId == null || !GGUF_SUFFIX_RE.test(modelId)) { return false; }" in identity
    assert "PUBLIC_ID_PATH_PREFIX_RE.test(modelId) ||" in identity
    assert 'modelId.split("/").length - 1 >= 2 ||' in identity
    assert "isNativeFileLabel(modelId)" in identity


def test_cached_repo_settings_key_follows_the_row_not_the_view():
    """A repo in an inactive HF cache loads by snapshot path while its settings are keyed
    by repo id."""
    page = " ".join(_read("features/hub/hub-page.tsx").split())
    assert 'if (kind !== "cache" && resource.source !== "hub_cache") {' in page


def test_detail_settings_defers_a_derived_quant_to_a_fresh_status_read():
    """The on-device card resolves the quant it shows from the store's active variant, and
    nothing re-reads status while the window keeps focus, so an API-driven switch leaves
    that quant naming the model it displaced."""
    page = " ".join(_read("features/hub/hub-page.tsx").split())
    card = " ".join(_read("features/hub/catalog/local-on-device-card.tsx").split())
    assert "if (!quantIsUserPicked) { const settled = useChatRuntimeStore.getState();" in page
    assert "variant = settled.activeGgufVariant;" in page
    assert "onOpenSettings(selectedQuant ?? null, quantIsUserPicked)" in card


def test_only_a_physical_gpu_pin_is_mirrored_to_the_server():
    """The same integers are Vulkan ordinals under Vulkan and device indices elsewhere,
    and the server override carries no namespace, so a backend change would pin the model
    to a different device with ids that validate."""
    mirror = " ".join(_read("features/model-picker/api/model-overrides.ts").split())
    assert 'const gpuIndexKind = config.selectedGpuIndexKind ?? "physical";' in mirror
    assert 'gpuIndexKind === "physical"' in mirror


def test_a_cached_repo_keeps_the_settings_saved_under_its_old_key():
    """A cached repo was keyed by the snapshot path it loads from and is now keyed by its
    repo id; the server backfill only mirrors what is stored, so nothing else moves it."""
    config = " ".join(_read("features/model-picker/model-config/per-model-config.ts").split())
    assert "export function adoptLegacyConfigKey(" in config
    # The key is renamed in one write. Saving the second copy first puts a full map one entry
    # over budget, and savePerModelConfig then evicts the oldest unrelated model silently, with
    # no eviction list handed back here, so that model's server override outlives its settings.
    assert "savePerModelConfig(modelId, ggufVariant, legacy)" not in config
    assert (
        "const alreadySaved = findConfigKeyForModelVariant(map, modelId, ggufVariant) "
        "!== null;" in config
    )
    # A newer save under the current key wins, and the stale record still goes.
    assert "if (!(alreadySaved || isDefaultConfig(legacy)))" in config
    assert "map[key] = toStoredConfig(legacy);" in config
    assert (
        "delete map[legacyKey]; deleteConfigEntriesForModelVariant(map, legacyModelId, "
        "ggufVariant);" in config
    )
    # Both entry points move it before anything reads the new key.
    page = " ".join(_read("features/hub/hub-page.tsx").split())
    assert page.count("adoptLegacyConfigKey(") == 2


def test_clearing_the_log_keeps_a_request_that_is_still_running():
    """Dropping an own row mid-flight loses the request outright: active_count falls to
    zero and the finish or fail that follows has no entry left to land on."""
    monitor = " ".join(_read_backend("core/inference/api_monitor.py").split())
    assert 'if entry.shared or entry.subject != subject or entry.status == "running"' in monitor
