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
    # syncModelCapabilities upserts the summary, matching on the id it is handed.
    assert "syncModelCapabilities(loadedModelId," in autoload


def test_chat_autoload_toast_is_persistent_and_dismissible():
    """Send-triggered autoload stays visible until it settles but remains
    dismissible, matching the explicit model-loading toast's lifetime."""
    src = _read("features/chat/api/chat-adapter.ts")
    auto_load = src.split("async function autoLoadSmallestModel", 1)[1]
    auto_load = auto_load.split("export function createOpenAIStreamAdapter", 1)[0]
    assert "toast.loading(" not in auto_load
    assert "const updateAutoLoadToast =" in auto_load
    assert "if (autoLoadToastDismissed) return;" in auto_load
    # Progress + the terminal "download stopped" notice; success/error use their
    # own toast.success / toast.error helpers.
    assert auto_load.count("toast.message(") == 3
    # The single progress toast is re-titled through every phase: the on-device
    # cascade updates it directly, and the default-model download keeps it live
    # via the same updater handed to ensureDefaultModelDownloaded.
    assert auto_load.count("updateAutoLoadToast(") >= 2
    assert (
        "ensureDefaultModelDownloaded(\n        hfToken,\n        options?.abortSignal,\n        updateAutoLoadToast,\n      )"
        in auto_load
    )
    download_helper = src.split("async function ensureDefaultModelDownloaded", 1)[1]
    download_helper = download_helper.split("async function autoLoadSmallestModel", 1)[0]
    assert download_helper.count("setToast(") >= 2
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
    # The mode/layer rule now lives in lib/gpu-placement.ts (behaviour covered by
    # studio/frontend/tests/gpu-placement.test.ts): assert the delegation here.
    assert "} = resolveComparePlacement(" in src
    assert "shouldPinDiffusionPlacement(" in src
    placement = " ".join(_read("features/chat/lib/gpu-placement.ts").split())
    assert 'own.gpuMemoryMode ?? (treatAsDiffusion ? "auto" : shared.gpuMemoryMode)' in placement
    assert "own.gpuLayers ?? (treatAsDiffusion ? GPU_LAYERS_AUTO : shared.gpuLayers)" in placement
    # An unclassified GGUF is pinned like a confirmed one: /load may still find a
    # diffusion header after the download.
    assert "return isDiffusion === true || diffusionUnknown" in placement
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


def test_a_pinned_cached_row_loads_from_the_id_the_backend_pinned():
    """The cached listing pins a snapshot when the repo's default ref reaches no copy
    that loads, and it is the load that has to follow the pin. The id stays the repo
    id everywhere it is shown, deduped or stored, so only model_path changes."""
    inventory = _read("features/model-picker/inventory/use-chat-picker-inventory.ts")
    for mapper in ("toCachedGgufRepo", "toCachedModelRepo"):
        body = re.search(rf"function {mapper}\(.*?\n}}", inventory, re.S)
        assert body, f"{mapper} not found"
        assert "load_id: row.loadId" in body.group(0), f"{mapper} drops the pinned id"
        # Delete sends this to remove the copy on screen, not the active cache's.
        assert "cache_path: row.cachePath" in body.group(0), f"{mapper} drops the cache path"

    meta = _read("features/model-picker/components/model-selector/types.ts")
    assert "loadId?: string | null;" in meta

    picker = _read("features/model-picker/components/model-selector/pickers.tsx")
    assert "loadId={c.load_id}" in picker, "the quant list cannot pass on a pin it never gets"
    # Every row that can start a load, and every gear beside one, reloads through
    # the same meta and so has to carry the pin. Counted rather than matched
    # loosely, so a new row that forgets it is a failure here rather than a load
    # that silently follows the default ref. #7736 added the third: the collapsed
    # single-quant GGUF row, which is a load site like the other two.
    assert picker.count("loadId: c.load_id") == 3, (
        "a row or gear that can start a load is missing the pin, or a new one was "
        "added and this count needs to follow it"
    )
    block = re.search(r"onConfigure\(repoId, \{.*?\n\s*\}", picker, re.S)
    assert block and "loadId," in block.group(0), "the GGUF gear drops the pin"
    # The variant click withholds it: a quant outside the pinned snapshot lands in a different one.
    block = re.search(r"onSelect\(repoId, \{.*?\n\s*\}", picker, re.S)
    assert block and "loadId: downloaded === true ? loadId : undefined," in block.group(0)
    # localPath alone: preferLocalCache would answer from disk and drop the undownloaded quants.
    # #7767 added the expander's abort signal to this call, so the options are an object
    # literal now rather than the bare localSource ternary.
    call = re.search(r"listGgufVariants\(repoId, hfToken, \{.*?\n\s*\}\)", picker, re.S)
    assert call, "the expander must still list variants for the row's own repo"
    assert "...(localSource ? { localPath: localSource } : {})" in call.group(
        0
    ), "the expander drops the row's own cache directory"
    assert "preferLocalCache" not in call.group(0)
    assert "cachePath={c.cache_path}" in picker

    # A reload rebuilds its target from the checkpoint id, so the resident model remembers the pin.
    page = _read("features/chat/chat-page.tsx")
    assert "loadId: activeLoadId," in page
    assert "activeLoadId: string | null;" in _read("features/chat/stores/chat-runtime-store.ts")

    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    assert "activeLoadId: loadPath === modelId ? null : loadPath," in runtime
    # A failed swap already unloaded the pinned model: reload it from the same place, pin and all.
    assert "model_path: previousActiveLoadId || previousCheckpoint," in runtime
    assert "activeLoadId: previousActiveLoadId ?? null," in runtime
    # A model loaded outside this tab replaces the resident one, and the pin belonged to the old.
    assert "useChatRuntimeStore.setState({ activeLoadId: null });" in runtime

    # Auto-load keys identity off the backend, so the pin is recorded beside it, not in place of it.
    adapter = _read("features/chat/api/chat-adapter.ts")
    assert "activeLoadId: modelPath === candidate.id ? null : modelPath," in adapter
    assert (
        '(typeof selection === "string" ? null : selection.loadId) || modelId' in runtime
    ), "loadPath must fall back to the id, so an unpinned pick is unchanged"
    # Staged metadata, validate and load: all three read the copy that loads.
    assert runtime.count("model_path: loadPath,") == 3
    assert "model_path: modelId," not in runtime
    # A rollback reads the approval under the snapshot path, so store it under both keys.
    assert "rememberApprovedRemoteCode(loadPath, approvedRemoteCodeFingerprint);" in runtime
    assert "approvedRemoteCodeFingerprints.get(previousCheckpoint) ?? null," in runtime


def test_a_local_quant_short_a_shard_is_not_selectable():
    """The variants endpoint now reports a local folder's torn quant as partial. A folder has no
    download to resume, so the row has to say so and refuse the pick rather than send validate and
    load at files that are not on disk."""
    picker = _read("features/model-picker/components/model-selector/pickers.tsx")
    assert "const unusableLocal = isLocalPath && v.partial === true;" in picker
    assert "disabled={unusableLocal}" in picker
    assert "incomplete" in picker


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
    # Both cache-backed sources (Hub GGUF repos and indexed local GGUF folders)
    # scan the exact path they will load from, not the bare repo id.
    sources = src.split("function buildAutoLoadSources", 1)[1]
    sources = sources.split("function isRememberedSource", 1)[0]
    assert sources.count("preferLocalCache: true") == 2
    assert "localPath: repo.cache_path" in sources
    assert "localPath: row.path" in sources

    # #7767 moved the query building out of chat-api into its own module, so the listing
    # imports without the auth barrel. Both halves are pinned: the caller must still hand
    # its options to the builder, and the builder must still forward them.
    chat_api = _read("features/chat/api/chat-api.ts")
    variants_fn = chat_api.split("export async function listGgufVariants", 1)[1]
    variants_fn = variants_fn.split("export interface KvCacheEstimate", 1)[0]
    assert "ggufVariantsQuery(repoId, options, isHuggingFaceOffline())" in variants_fn

    query_src = _read("features/chat/api/gguf-variants-request.ts")
    query_fn = query_src.split("export function ggufVariantsQuery", 1)[1]
    query_fn = query_fn.split("\nexport ", 1)[0]
    assert 'params.set("prefer_local_cache", "true")' in query_fn
    assert 'params.set("local_path", localPath)' in query_fn


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
    reset = store.split("setCheckpoint: (modelId, ggufVariant, options) =>", 1)[1].split(
        "setActiveThreadId:", 1
    )[0]
    assert "activeModelIsLocal: false" in reset
    assert "specFallbackReason: null" in reset
    # The reason chain is a switch, so slice on the case label rather than the
    # ternary comparison it replaced.
    assert "isLocalGguf" in src.split('case "drafter_not_found":', 1)[1]


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
    # The throw carries the cancellation marker, so the sweep stops rather than reopen the dialog.
    assert 'new Error("Model load cancelled.")' in autoload
    assert "unslothUserCancelled: true" in autoload
    assert "recordCandidateFailure(failureLabel, cancelled)" in autoload


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


def test_variant_expander_forwards_the_gguf_filename():
    """A quant pick must carry the exact .gguf filename. The diffusion pages load
    by filename and cannot map a quant label back to one, so without it every hub
    GGUF pick on Images/Video fell through to a silent return and nothing loaded."""
    src = _read("features/model-picker/components/model-selector/pickers.tsx")
    handler = re.search(r"const handleVariantClick = useCallback\(.*?\n  \);", src, re.S)
    assert handler, "handleVariantClick not found"
    assert "ggufFilename: filename," in handler.group(0)
    # The call site must pass it through in the handler's argument order. Matched structurally, since prettier wraps the call once it grows.
    call = re.search(r"handleVariantClick\(([^)]*)\)", src)
    assert call, "handleVariantClick call site not found"
    args = [a.strip() for a in call.group(1).split(",") if a.strip()]
    assert args[:2] == ["v.quant", "v.filename"], args


def test_a_routed_local_single_file_pick_keeps_its_load_kind():
    """A pick routed from the chat picker arrives as ?model=&quant= with no picker metadata,
    so a bare local .gguf / .safetensors has to be recognised from the path. Loading one as a
    pipeline evicts the resident model and then fails on the missing model_index.json, because
    an explicit model_kind wins over the backend's filename sniffing."""
    helper = _read("lib/diffusion-route-pick.ts")
    assert '"gguf"' in helper and '"single_file"' in helper
    assert 'lower.endsWith(".gguf")' in helper
    assert 'lower.endsWith(".safetensors")' in helper
    # A repo id (no recognised extension) still loads as a pipeline, as before.
    assert 'kind: "pipeline"' in helper

    for rel in ("features/images/images-page.tsx", "features/video/video-page.tsx"):
        src = _read(rel)
        assert "diffusionRoutePick(" in src, f"{rel}: route pick not derived"
        # And the derived pick is what gets loaded, not the raw search params.
        assert re.search(r"loadOrStage\(\s*pick\.repoId,\s*pick\.opts", src), rel


def test_a_routed_curated_pick_uses_the_same_load_spec_as_a_direct_one():
    """The chat picker can only forward a GGUF filename (ggufFilename is GGUF-specific), so a
    curated single-file artifact -- an LTX-2.3 checkpoint, an FP8 transformer -- arrives with no
    quant. Classifying it by shape alone made it a pipeline load, which calls from_pretrained on a
    repo that has no model_index.json. The catalog spec the page's own picker consults has to win."""
    helper = _read("lib/diffusion-route-pick.ts")
    assert re.search(r"spec\?:\s*\{\s*kind:", helper), "the helper takes no catalog spec"
    assert (
        "if (spec) return { repoId: model, opts: { kind: spec.kind, filename: spec.filename } };"
        in helper
    )
    for rel, catalog in (
        ("features/images/images-page.tsx", "IMAGE_CATALOG"),
        ("features/video/video-page.tsx", "VIDEO_CATALOG"),
    ):
        src = _read(rel)
        call = re.search(
            r"diffusionRoutePick\(\s*wanted,\s*routeSearch\.quant,\s*(.*?),?\s*\);", src, re.S
        )
        assert call, f"{rel}: the routed pick passes no spec"
        assert f"loadSpecFor(wanted, {catalog})" in call.group(1), rel


def test_a_quantized_load_drops_a_lora_selection_it_cannot_bake():
    """int8/fp8 builds take adapters only at load time. Switching artifact inside one family
    keeps the selection (same family, no clear) while the load did not bake it, so Generate
    would 400 with the picker still showing the adapter as active."""
    src = _read("features/images/images-page.tsx")
    assert "bakedLorasOnLoad.current = bakeLoras.length > 0;" in src
    guard = re.search(
        r"if \(!loraCapable \|\| checkedBuildForBake\.current === residentBuildKey\) return;.*?\n  \}, \[",
        src,
        re.S,
    )
    assert guard, "the bake-only check is missing"
    body = guard.group(0)
    assert '"int8"' in body and '"fp8"' in body
    assert "bakedLorasOnLoad.current" in body, "a baked selection must be kept"
    assert "setLoras([])" in body and "toast.info(" in body, "cleared without telling the user"


def test_diffusion_pages_never_drop_a_gguf_pick_silently():
    """The fallback branch splits a local path; a repo pick reaching it has no
    filename. It must say so instead of returning with no request and no toast."""
    for rel in ("features/images/images-page.tsx", "features/video/video-page.tsx"):
        src = _read(rel)
        branch = re.search(
            r'if \(!filename\.toLowerCase\(\)\.endsWith\("\.gguf"\)\) \{.*?\}', src, re.S
        )
        assert branch, f"{rel}: gguf extension guard not found"
        assert "toast.error(" in branch.group(0), f"{rel}: guard returns silently"


def test_diffusion_pages_stage_downloads_through_the_manager():
    """Images/Video must not download inside the load: an undownloaded hub pick goes to
    the Hub download manager first, so it shares the panel, progress, cancel/resume,
    disk preflight and manifest verification with every other model."""
    for rel in ("features/images/images-page.tsx", "features/video/video-page.tsx"):
        src = _read(rel)
        assert "useStagedDownload" in src, f"{rel}: not wired to the download manager"
        # The plan carries the loader's own file scope, so nothing extra is pulled.
        assert "DownloadPlan(" in src, f"{rel}: does not fetch a download plan"
        stage_fn = re.search(r"const loadOrStage = useCallback\(.*?\n  \);", src, re.S)
        assert stage_fn, f"{rel}: loadOrStage not found"
        body = stage_fn.group(0)
        # Already-downloaded (and local) picks must skip staging and load straight away.
        assert "isDownloaded !== false" in body, f"{rel}: cached picks would re-stage"
        # A missing plan must still load rather than dead-end.
        assert "catch" in body, f"{rel}: no fallback when the plan is unavailable"


def test_a_hidden_diffusion_page_does_not_load_when_its_download_lands():
    """Images and Video stay mounted behind the router, and a load evicts whoever holds the
    GPU. A multi-GB staged download finishing while the user is on another page must not take
    the model out from under them; the pick waits for its page to be visible again."""
    for rel in ("features/images/images-page.tsx", "features/video/video-page.tsx"):
        src = _read(rel)
        ready = re.search(r"onReady: \(\) => \{.*?\n    \},", src, re.S)
        assert ready, f"{rel}: staged-download onReady not found"
        assert "if (!active)" in ready.group(0), f"{rel}: a hidden page still takes the GPU"
        # Deferred, not dropped: something has to fire the held pick when the page returns.
        assert "stagedLoadDeferred" in ready.group(0), f"{rel}: the pick is discarded"
        flush = re.search(
            r"if \(!active \|\| !stagedLoadDeferred\.current\) return;.*?\n  \}, \[active\]\);",
            src,
            re.S,
        )
        assert flush, f"{rel}: nothing flushes the deferred load when the page is shown"
        assert "handleLoadRef.current(" in flush.group(0), f"{rel}: deferred load never runs"


def test_staged_downloads_always_scope_their_files():
    """Every staged entry must go out as a scoped job carrying its file list, GGUF
    checkpoints included. A plain snapshot job drops *.gguf via the Hub's ignore list, so
    it would finish instantly having fetched everything except the weights and leave the
    repo on device unloadable."""
    src = _read("features/hub/download-manager/use-staged-download.ts")
    start = re.search(r"downloadManager\.requestStart\(\{.*?\}\);", src, re.S)
    assert start, "requestStart call not found"
    body = start.group(0)
    # Unconditional: no branch may send a null scope or omit the files.
    assert "scopeId," in body and "files: current.files," in body
    assert "? null" not in body and "? undefined" not in body
    assert "const activeVariant = current ? scopedVariant(scopeId) : null;" in src


def test_local_model_sections_respect_the_task_filter():
    """LM Studio / ./models / custom-folder rows must honour the picker's task filter.
    The backend tags every local model with a task for exactly this; without the gate the
    Images picker listed chat GGUFs (which 400 on a diffusion load) and buried the
    diffusion models the page can actually run."""
    src = _read("features/model-picker/components/model-selector/pickers.tsx")
    for memo in ("sortedLmStudio", "sortedLocalDir", "sortedCustomFolderModels"):
        block = re.search(rf"const {memo} = useMemo\(.*?\n  \);", src, re.S)
        assert block, f"{memo} not found"
        assert "passesTaskGate(m.task" in block.group(0), f"{memo} does not apply the task gate"


def test_chat_picker_routes_diffusion_picks_to_their_page():
    """Chat cannot load a diffusion model. Rather than hiding an on-device one or letting
    it 400, the unfiltered picker routes the pick to the Images/Video page, which loads it."""
    src = _read("features/model-picker/components/model-selector/pickers.tsx")
    gate = re.search(r"function passesTaskGate\(.*?\n\}", src, re.S)
    assert gate, "passesTaskGate not found"
    # The chat branch no longer drops the generation tasks outright.
    assert "UNSUPPORTED_DIFFUSION_TASK" in gate.group(0)
    wrapper = re.search(r"const onSelect = useCallback\(.*?\n  \);", src, re.S)
    assert wrapper, "the routing wrapper around onSelect is missing"
    body = wrapper.group(0)
    assert "diffusionPageForTask" in body and "navigateToPage" in body
    # Task-scoped pickers (already on those pages) must select normally.
    assert "if (!task)" in body


def test_staged_download_callbacks_only_answer_their_own_variant():
    """subscribeJobListeners is per repo, not per job, so a staged entry hears every job on
    that repo (the Models tab fetching a chat quant of the same repo, say). Each callback
    carries the variant it fired for: without comparing it, a sibling job's completion advanced
    the staged queue and started a load whose scoped files were still downloading, and its
    failure wiped a queue that was still running."""
    src = _read("features/hub/download-manager/use-staged-download.ts")
    # The comparison lives in the shared isOurs() guard the three callbacks run (which also binds them to the started file set).
    assert "(variant ?? null) === activeVariant &&" in src
    for callback in ("onComplete", "onError", "onCancelled"):
        handler = re.search(rf"{callback}: \(variant\) => \{{\n(.*?)\n    \}},", src, re.S)
        assert handler, f"{callback} does not take the variant"
        assert "isOurs(variant)" in handler.group(1), callback


def test_video_gallery_fetches_clips_as_their_cards_come_into_view():
    """Each gallery record's src is a blob holding the whole MP4 until the page closes, so
    fetching a full page of them up front pinned hundreds of MB (gigabytes across "load more"
    pages) for cards the user may never scroll to. Fetch on visibility instead, and always
    fetch the selected clip, since that is the one the preview player plays."""
    src = _read("features/video/video-page.tsx")
    assert "new IntersectionObserver(" in src
    assert "ref={stripRef}" in src and "data-clip-id={video.id}" in src
    assert 'root.querySelectorAll("[data-clip-id]")' in src
    # rootMargin applies to the root box only, so the strip (the clipping scroller) must BE the root or the prefetch margin never reaches a clipped card.
    assert '{ root, rootMargin: "0px 600px" }' in src
    # The only surviving whole-page fetches are the no-IntersectionObserver fallbacks.
    eager = list(re.finditer(r"page\.videos\.forEach\(\(video\) => void ensureSrc\(video\)\)", src))
    assert eager, "the jsdom/old-webview fallback fetch is missing"
    for match in eager:
        assert (
            'typeof IntersectionObserver === "undefined"'
            in src[max(0, match.start() - 260) : match.start()]
        )
    assert re.search(
        r"if \(!selected\) return;\s*\n\s*void \(async \(\) => \{\s*\n\s*await ensureSrc\(selected\);",
        src,
    )


def test_on_device_rows_carry_the_task_the_pickers_filter_on():
    """The picker's On Device rows come from the /api/hub inventory, not the models API, and the
    task-scoped pickers drop every row whose task is unset. Without the task threaded through the
    hub inventory and its adapter, the Images and Video pickers listed nothing on device and the
    chat picker never routed a diffusion pick, since diffusionTaskById reads the same field."""
    api = _read("features/hub/inventory/api.ts")
    assert api.count("task?: string | null;") >= 3, "the hub row response types carry no task"
    rows = _read("features/hub/inventory/types.ts")
    assert rows.count("task?: string | null;") >= 2, "the inventory row types carry no task"
    vm = _read("features/hub/inventory/view-models.ts")
    assert "task: row.task ?? null," in vm and "task: model.task ?? null," in vm
    conv = _read("features/model-picker/inventory/use-chat-picker-inventory.ts")
    assert conv.count("task: row.task ?? null,") == 3, "a picker converter drops the task"
    # A generation-task row is not a chat row, so the chat-only guard must not hide it from the pickers that can load it.
    assert "row.capabilities.canChat || studioPageForTask(row.task) !== undefined" in conv


def test_local_diffusion_routing_is_keyed_by_the_id_the_row_selects():
    """A local row's click passes m.id (a filesystem load id), while m.model_id is its HF-style
    name. Keying the routing map on one alone let the lookup miss, so the pick fell through to the
    chat loader instead of navigating to Images or Video."""
    src = _read("features/model-picker/components/model-selector/pickers.tsx")
    block = re.search(r"const diffusionTaskById = useMemo\(.*?\n  \}, \[", src, re.S)
    assert block, "diffusionTaskById not found"
    body = block.group(0)
    assert "put(m.id, m.task);" in body and "put(m.model_id, m.task);" in body


def test_a_staged_download_that_never_starts_clears_the_queue():
    """requestStart can answer "error" (network failure, rejected scoped request, worker refused).
    Nothing completes after that, so leaving the head in place stranded the pick: the effect never
    re-ran and onReady never fired."""
    src = _read("features/hub/download-manager/use-staged-download.ts")
    assert 'if (outcome === "error") {' in src
    branch = src[src.index('if (outcome === "error") {') :][:400]
    assert "setQueue(null)" in branch


def test_staged_download_callbacks_are_bound_to_the_started_file_set():
    """Every scoped pick in a repo shares the "@diffusion" variant, so the variant alone cannot
    tell two file sets in that repo apart: restaging while the first job finishes let its
    completion pass for the new pick and load a checkpoint that had not downloaded."""
    src = _read("features/hub/download-manager/use-staged-download.ts")
    assert "const inFlight = useRef<{ key: string; generation: number } | null>(null);" in src
    assert "inFlight.current.key === entryKey(current)" in src
    assert "inFlight.current.generation === generation.current" in src
    # A fresh plan invalidates the previous job's callbacks.
    stage = re.search(r"const stage = useCallback\(.*?\}, \[\]\);", src, re.S)
    assert stage and "generation.current += 1;" in stage.group(0)
    for callback in ("onComplete", "onError", "onCancelled"):
        handler = re.search(rf"{callback}: \(variant\) => \{{\n(.*?)\n    \}},", src, re.S)
        assert handler and "isOurs(variant)" in handler.group(1), callback


def test_a_lost_generate_post_must_prove_it_reached_the_backend():
    """A rejected fetch does not say whether the POST landed. Treating an immediately idle progress
    read as success made a submission that never reached the server look like a finished image."""
    src = _read("features/images/images-page.tsx")
    fn = src[src.index("async function settleLostGeneration") :]
    fn = fn[: fn.index("\n}\n")]
    assert "knownIds" in fn and "sawActive" in fn
    assert "!knownIds.has(image.id)" in fn
    assert "did not reach the server" in fn
    # And the caller snapshots the ids BEFORE the POST.
    assert "new Set(galleryCache.images.map((image) => image.id))" in src


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
        "} else {", 1
    )
    non_gguf_branch = non_gguf_rest.split("if (!(loadResp.is_lora ?? false)) {", 1)[0]
    # The cached-GGUF branch keeps the remembered override via the gated local...
    assert "nParallel: committedSlots," in gguf_branch
    assert "nParallel: null," not in gguf_branch
    # ...
    assert "nParallel: null," in non_gguf_branch
    assert "loadedNParallel: null," in non_gguf_branch

    fresh_default = adapter.split("// Nothing on the device:", 1)[1].split(
        "showAutoLoadSuccess(\n          `Loaded ${DEFAULT_CHAT_MODEL_LABEL}", 1
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


def test_chat_autoload_records_every_validation_failure():
    """canAutoLoad runs validateModel, which prepares the token, so a dismissed dialog, a dead
    backend or a model-specific rejection throws there rather than from loadModel. The sweep's
    catches are bare, so an unrecorded one reads as an empty device and fetches the Hub default.
    Only a declined dialog ends the sweep."""
    adapter = _read("features/chat/api/chat-adapter.ts")
    recorder = adapter.split("function recordCandidateFailure", 1)[1]
    recorder = recorder.split("async function canAutoLoadRecordingFailures", 1)[0]
    # Unconditional: no tag is consulted before recording, so a plain rejection counts too.
    assert recorder.count("noteLoadFailure(label, error)") == 1
    assert recorder.index("noteLoadFailure(label, error)") < recorder.index(
        "unslothUserCancelled === true"
    ), "recording must not sit behind a marker test"
    # A declined dialog halts the sweep (retrying reopens it); nothing else does.
    assert "unslothUserCancelled === true" in recorder
    assert recorder.count("autoLoadCancelled = true;") == 1
    assert recorder.index("unslothUserCancelled === true") < recorder.index(
        "autoLoadCancelled = true;"
    ), "the halt belongs to the cancellation branch alone"
    # Rethrown by the wrapper, so the candidate still fails and control flow is unchanged.
    wrapper = adapter.split("async function canAutoLoadRecordingFailures", 1)[1]
    wrapper = wrapper.split("async function loadAutoLoadCandidate", 1)[0]
    assert "recordCandidateFailure(label, error)" in wrapper
    assert "throw error;" in wrapper
    autoload = adapter.split("async function loadAutoLoadCandidate", 1)[1]
    autoload = autoload.split("async function autoLoadSmallestModel", 1)[0]
    # The preflight and the GGUF metadata probe both record, and a cancelled sweep skips the rest.
    assert "canAutoLoadRecordingFailures(failureLabel, {" in autoload
    assert "recordCandidateFailure(failureLabel, error)" in autoload
    # The preflight's own cancellation goes through the helper too, or it records without halting.
    assert "recordCandidateFailure(failureLabel, cancelled)" in autoload
    assert "noteLoadFailure(failureLabel, cancelled)" not in autoload
    assert "if (autoLoadCancelled || loadAttempts >= MAX_AUTO_LOAD_ATTEMPTS)" in autoload


def test_auth_retries_tag_transport_failures_like_the_first_attempt():
    """noteLoadFailure keys on the tag, so an untagged TypeError from a retry reads as a
    rejection: retries reissue through retryWithCurrentToken, so it tags like the first attempt."""
    src = (WORKDIR / "studio" / "frontend" / "src" / "features" / "auth" / "api.ts").read_text(
        encoding = "utf-8"
    )
    assert src.count("unslothTransportFailure: true") == 2, "one tag per message, in one helper"
    tagger = src.split("function asTransportFailure", 1)[1].split("\n}\n", 1)[0]
    assert "err instanceof TypeError" in tagger
    assert "navigator.onLine === false" in tagger
    retry = src.split("async function retryWithCurrentToken", 1)[1]
    retry = retry.split("\n}\n", 1)[0]
    assert "fetchWithTauriNetworkRetry" in retry
    assert "throw asTransportFailure(err);" in retry
    first = src.split("export async function authFetch", 1)[1]
    assert "throw asTransportFailure(err);" in first


def test_external_readoption_drops_a_pin_taken_for_another_model():
    """Status polling skips its own pin clearing while an external provider is selected, so the
    re-adoption branch can adopt a resident the pin was never taken for and Apply would reload the
    old model. The branch has to clear it itself."""
    src = _read("features/chat/hooks/use-chat-model-runtime.ts")
    branch = src.split("if (!forceReload && isExternalModelId(selectedCheckpoint))", 1)[1]
    branch = branch.split("const stopDecision = await confirmStopRunningChatsIfNeeded", 1)[0]
    assert "activeLoadId !== modelId" in branch
    assert "setState({ activeLoadId: null })" in branch
    # Clearing must land before the checkpoint moves, so nothing reads the pair half updated.
    assert branch.index("activeLoadId: null") < branch.index(
        ".setCheckpoint(modelId, residentStatus.gguf_variant)"
    ), "the pin must be cleared before the checkpoint is adopted"


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
    # /api/inference/status and the checkpoint helper share _llama_status_model_ids.
    assert (
        "return display_model_id, (None if native_grant_backed else model_id)" in inference
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


# ─────────────────────────────────────────────────────────────────────────────
# Test hooks the Playwright driver depends on.
#
# tests/studio/playwright_model_config.py drives the picker through these exact
# attributes and accessible names. Renaming one is a source-only edit that type
# checks, lints and passes every unit test, and then fails a 25-minute browser job
# with "selector not found" -- which is how the collapsed-row regression (#7736)
# sat on main. Pin them here so that break costs two seconds instead.
# ─────────────────────────────────────────────────────────────────────────────


def test_picker_rows_keep_their_automation_attributes():
    pickers = _read("features/model-picker/components/model-selector/pickers.tsx")
    # On the row button itself, not a wrapper: the driver scopes row text to it.
    assert '"data-model-picker-option": true;' in pickers
    assert '"data-model-picker-option": true,' in pickers
    assert "data-model-picker-search-input" in pickers
    assert "data-model-picker-list" in pickers
    # And that ModelRow still spreads them onto the button. Without this the two
    # above only prove the props are declared and returned: ModelRow reads
    # optionProps?.onKeyDown separately, so the prop stays used and type checks
    # while data-model-picker-option leaves the DOM and every row lookup in the
    # driver fails.
    row_button = re.search(r"const content = \(\s*<button\b(.*?)>", pickers, re.S)
    assert row_button, "ModelRow no longer renders its content as a <button>"
    assert "{...optionProps}" in row_button.group(
        1
    ), "ModelRow stopped spreading optionProps onto the row button"


def test_picker_popover_and_trigger_keep_their_tour_hooks():
    """Both props by name. The trigger's value is a prefix of the popover's, so any
    assertion that falls back to the bare substring is satisfied by the popover alone
    and would pass with the trigger hook deleted -- while the driver's very first
    click, page.locator(TRIGGER), would be the thing that fails."""
    chat = _read("features/chat/chat-page.tsx")
    assert 'triggerDataTour="chat-model-selector"' in chat
    assert 'contentDataTour="chat-model-selector-popover"' in chat


def test_run_settings_gear_label_names_its_model():
    """The driver cannot reach the gear through the row -- it is a sibling at 2 of its
    render sites and an uncle at the rest -- so it matches on this label, and the model
    name in it is what keeps it off another row's gear."""
    pickers = _read("features/model-picker/components/model-selector/pickers.tsx")
    # The expander's own label, not any of them: pickers.tsx interpolates this
    # string at five sites, so a file-wide match stays green while the one the
    # driver depends on drops its repo or its quant. That is the only label with
    # both, and naming the quant is how the driver tells one variant from another.
    assert (
        "ariaLabel={`Inference settings for ${repoId} ${v.quant}`}" in pickers
    ), "the variant gear label no longer names both the repo and the quant"
    action = _read("features/model-picker/components/model-selector/model-load-settings-action.tsx")
    assert "aria-label={ariaLabel}" in action


def test_a_multi_quant_parent_row_still_has_no_gear():
    """The driver reads the absence of a gear as "this row expands rather than loads".
    Give the parent row a gear and that inference silently inverts."""
    pickers = _read("features/model-picker/components/model-selector/pickers.tsx")
    # Inside the parent row, not anywhere in the file. Adding a gear to the parent
    # while leaving the spacer in place satisfies a file-wide search, and that is
    # exactly the change that would invert the driver's inference.
    start = pickers.index("const renderDownloadedGgufRow")
    parent = pickers[start : pickers.index("const renderDownloadedModelRow", start)]
    parent_row = parent[: parent.index("{expanderOpen && (")]
    # The gutter by reference, not by width. Pinning "w-[42px]" pinned two things
    # the driver does not care about: the exact pixel width, and the fact that it
    # was spelled inline. main has since moved to w-[38px] behind ROW_ACTIONS_CLASS,
    # which is the same spacer doing the same job, and this assertion would have
    # failed the moment the two met -- reporting a lost spacer that is still there.
    gutter = "ROW_ACTIONS_CLASS" in parent_row or re.search(r"w-\[\d+px\]", parent_row)
    assert (
        'aria-hidden="true"' in parent_row and gutter
    ), "the multi-quant parent row lost the spacer that stands in for a gear"
    assert "ModelLoadSettingsAction" not in parent_row, (
        "the multi-quant parent row grew a gear, so the driver's 'no gear means "
        "this row expands rather than loads' inference is now wrong"
    )


def test_run_settings_page_keeps_its_identifying_controls():
    page = _read("features/model-picker/components/model-config-page.tsx")
    # How the driver knows run-settings actually opened.
    assert 'aria-label="Back to model list"' in page
    assert 'ariaLabel="Context Length"' in page
    assert 'aria-label="Context Length"' in page
    # The button, not the word: "Reset" also appears in this file's own comments, so a
    # raw source search passes with the control deleted and the Playwright reset gate
    # only finds out 25 minutes later.
    assert re.search(
        r"onClick=\{\(\) => setConfig\(\{ \.\.\.DEFAULT_PER_MODEL_CONFIG \}\)\}\s*>\s*Reset\s*</Button>",
        page,
    ), "the Reset button's JSX is gone or no longer named Reset"


def test_the_primary_action_keeps_its_four_labels():
    """The driver sweeps these names to find the one button the panel shows."""
    page = _read("features/model-picker/components/model-config-page.tsx")
    for label in ('"Save settings"', '"Forget settings"', '"Reload model"', '"Load model"'):
        assert label in page, label
    # And that the label is rendered by a button the driver can press. The four
    # literals above live in the primaryActionLabel calculation, so they survive
    # the button being deleted or turned into a non-button, and get_by_role finds
    # nothing while this stays green.
    # Whole elements, then the one that is both: other props on the opening tag
    # contain braces of their own, so a pattern that tries to reach onClick
    # through a negated class stops at the first `disabled={...}`.
    primary = any(
        "onClick={handleRun}" in el.group(0) and "{primaryActionLabel}" in el.group(0)
        for el in re.finditer(r"<Button\b.*?</Button>", page, re.S)
    )
    assert primary, "primaryActionLabel is no longer rendered inside the handleRun Button"


def test_autoload_sees_every_on_device_inventory_and_fails_closed():
    """#7374: Send downloaded a model over a perfectly loadable local one. The
    cascade only read the two managed-cache lists, and both were wrapped in
    `.catch(() => [])`, so a flaky request also read as "device is empty"."""
    src = _read("features/chat/api/chat-adapter.ts")
    auto_load = src.split("async function autoLoadSmallestModel", 1)[1]
    discovery = auto_load.split("for (const source of sources)", 1)[0]
    # The backend-indexed on-device inventory is consulted, not just the cache.
    assert "listLocalModels()" in discovery
    assert "listCachedGguf(options?.abortSignal)" in discovery
    assert "listCachedModels(hfToken, options?.abortSignal)" in discovery
    # Fail closed: an inventory error must not read as an empty device.
    assert "listCachedGguf().catch(" not in src
    assert "listCachedModels().catch(" not in src
    assert "listLocalModels().catch(" not in src


def test_autoload_local_rows_follow_the_picker_policy():
    """A background pick must be something the user could have picked, minus the
    formats that must stay an explicit action."""
    src = _read("features/chat/api/chat-adapter.ts")
    policy = src.split("function isAutoLoadableLocalRow", 1)[1]
    policy = policy.split("\n}", 1)[0]
    assert "AUTO_LOAD_LOCAL_SOURCES.has(row.source)" in policy
    # Not `=== true`: normalizeCapabilities falls back to the format capability
    # when the backend sends no can_chat, so the picker shows such a row. `===
    # true` skipped it and fell through to downloading the default instead.
    assert "row.capabilities?.can_chat !== false" in policy
    assert "row.partial !== true" in policy
    # An adapter resolves its base model, which for an uncached base is a Hub
    # fetch; a scan-folder checkpoint is a pickle with no Hub security scan.
    assert 'row.model_format !== "adapter"' in policy
    assert 'row.model_format !== "checkpoint"' in policy
    assert "isHiddenModelId(row.model_id, row.id, row.path)" in policy

    sources = src.split("const AUTO_LOAD_LOCAL_SOURCES", 1)[1].split("]", 1)[0]
    assert '"models_dir"' in sources
    assert '"lmstudio"' in sources
    assert '"custom"' in sources
    # hf_cache rows are the cached lists' job; ollama links are not loadable.
    assert '"hf_cache"' not in sources
    assert '"ollama"' not in sources


def test_default_model_download_is_visible_and_cancellable():
    """When the device really is empty Studio still fetches a model, but as a
    managed download: a panel entry with progress and a Cancel that stops it,
    never an inline pull hidden inside /api/inference/load."""
    src = _read("features/chat/api/chat-adapter.ts")
    helper = src.split("async function ensureDefaultModelDownloaded", 1)[1]
    helper = helper.split("async function autoLoadSmallestModel", 1)[0]
    assert "downloadManager.requestStart(request)" in helper
    assert "downloadManager.cancel(" in helper
    assert "subscribeJobListeners(" in helper
    # The toast carries its own Cancel wired to the same job.
    assert "cancelDownload," in helper
    # Live byte progress, so the user is not staring at a static message.
    assert "useDownloadManagerStore.subscribe(" in helper
    # Already on disk from an earlier run: no download at all.
    assert 'if (variant?.downloaded && variant.partial !== true) return "ready";' in helper
    # The load only happens once the bytes are actually here.
    assert "loadModel(" not in helper

    auto_load = src.split("async function autoLoadSmallestModel", 1)[1]
    fallback = auto_load.split("// Nothing on the device:", 1)[1]
    assert 'if (download !== "ready") {' in fallback
    # Cancelling leaves the user with actionable next steps, not a dead end.
    assert "Pick one from the top bar" in fallback

    updater = src.split("const updateAutoLoadToast =", 1)[1].split("\n  };", 1)[0]
    assert 'label: "Cancel"' in updater


def test_default_chat_model_is_gemma_4_e2b():
    """The empty-device default, in one place so it cannot drift."""
    src = _read("features/chat/api/chat-adapter.ts")
    assert 'const DEFAULT_CHAT_MODEL_REPO = "unsloth/gemma-4-E2B-it-GGUF";' in src
    assert 'const DEFAULT_CHAT_MODEL_VARIANT = "UD-Q4_K_XL";' in src
    assert 'const DEFAULT_CHAT_MODEL_LABEL = "Gemma 4 E2B";' in src
    # No hard-coded repo id or quant left anywhere on the send path.
    assert "Qwen3.5-4B-MTP-GGUF" not in src
    assert src.count('"UD-Q4_K_XL"') == 1


def test_first_download_toast_is_informative_not_alarming():
    """The old copy read as a warning about the user's machine ("No downloaded
    models found") and named a raw repo/quant. It now explains what is
    happening and what the user can do about it."""
    src = _read("features/chat/api/chat-adapter.ts")
    assert "No downloaded models found" not in src
    assert "Downloading a small model" not in src
    helper = src.split("async function ensureDefaultModelDownloaded", 1)[1]
    helper = helper.split("async function autoLoadSmallestModel", 1)[0]
    assert "Getting ${DEFAULT_CHAT_MODEL_LABEL} ready" in helper
    assert "Unsloth couldn\u2019t find an existing model." in helper
    assert "You can stop the download or " in helper
    assert "manage models later in the 'Model hub'" in helper


def test_indexed_local_loads_are_remembered_without_bypassing_leases():
    """A local model the user actually ran should be re-picked first, but a
    native file-picker path must not be: reaching it needs a signed lease."""
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    block = runtime.split("const indexedLocalPick =", 1)[1]
    block = block.split("} catch (error) {", 1)[0]
    assert 'selection.source === "local"' in block
    assert "!nativePathToken &&" in block
    assert "(indexedLocalPick || !isLocalModelPath(modelId))" in block

    store = _read("features/chat/utils/last-local-model-load.ts")
    # A cached Hub GGUF repo without a quant names no file; a local .gguf does.
    assert 'input.kind === "gguf" && !ggufVariant && !isPathLikeId(id)' in store
    # Identity only: no tokens, leases, or approvals are persisted.
    assert "token" not in store.lower().replace("isPathLikeId", "")


def test_autoload_skips_image_and_video_rows():
    """The backend only tags an on-device row with a task for image/video
    models, and the chat picker routes those away on click. A background load
    has no routing step."""
    src = _read("features/chat/api/chat-adapter.ts")
    tasks = src.split("const IMAGE_OR_VIDEO_TASKS", 1)[1].split("]", 1)[0]
    assert '"text-to-image"' in tasks
    assert '"text-to-video"' in tasks
    assert '"image-diffusion-unsupported"' in tasks
    policy = src.split("function isAutoLoadableLocalRow", 1)[1].split("\n}", 1)[0]
    assert 'IMAGE_OR_VIDEO_TASKS.has(row.task ?? "")' in policy


def test_remembered_matching_uses_the_same_case_rules_as_the_dedupe_key():
    """Folding a POSIX path would mark two models differing only by case as
    both being the remembered one."""
    src = _read("features/chat/api/chat-adapter.ts")
    remembered = src.split("function isRememberedSource", 1)[1].split("\n}", 1)[0]
    assert "normalizeTarget(remembered.id)" in remembered
    assert "normalizeTarget(source.id)" in remembered
    assert "normalizeTarget(source.loadId)" in remembered
    assert ".toLowerCase()" not in remembered
    key = src.split("function autoLoadSourceKey", 1)[1].split("\n}", 1)[0]
    assert "normalizeTarget(source.loadId)" in key


def test_cached_inventory_lookups_take_the_run_signal():
    """Neither has a timeout of its own, unlike listLocalModels."""
    src = _read("features/chat/api/chat-adapter.ts")
    assert "listCachedGguf(options?.abortSignal)" in src
    assert "listCachedModels(hfToken, options?.abortSignal)" in src


def test_download_cancel_survives_the_pending_start_window():
    """cancel() no-ops on a key with no job yet, so a click landing during
    requestStart's preflights has to be replayed once the job exists, and
    exactly once: cancelling patches the job and wakes the subscription."""
    src = _read("features/chat/api/chat-adapter.ts")
    helper = src.split("async function ensureDefaultModelDownloaded", 1)[1]
    helper = helper.split("async function autoLoadSmallestModel", 1)[0]
    assert "let cancelRequested = false;" in helper
    assert "let cancelInFlight = false;" in helper
    # In flight only: a failed cancel restores the job, so a later click retries.
    assert 'if (cancelInFlight || active.state === "cancelling") return true;' in helper
    assert "if (cancelRequested) {\n          if (!cancelEverIssued) issueCancel();" in helper
    assert 'if (cancelRequested && !issueCancel()) finish("cancelled");' in helper


def test_a_failed_quant_is_marked_tried_so_the_repo_continues():
    """One corrupt quant must not cost a repo that holds a valid one."""
    src = _read("features/chat/api/chat-adapter.ts")
    cascade = src.split("for (const source of sources)", 1)[1]
    cascade = cascade.split("// Nothing on the device:", 1)[0]
    assert "while (!autoLoadCancelled && loadAttempts < MAX_AUTO_LOAD_ATTEMPTS)" in cascade
    assert "skippedAutoLoadCandidates.add(" in cascade


def test_local_safetensors_chat_capability_is_classified_not_assumed():
    """An embedding export is config.json plus model.safetensors, which the
    format-only rule marked chat-capable; those are small, so chat auto-load
    would try them first."""
    src = _read_backend("hub/services/models/common.py")
    assert "def _local_transformers_can_chat(" in src
    assert "can_chat_override" in src
    call = src.split("adapter_type = adapter_type if model_format", 1)[1]
    call = call.split("elif not rows:", 1)[0]
    assert "_local_transformers_can_chat(scan_path)" in call
    assert 'model_format in {"safetensors", "checkpoint"}' in call
    # Fails open: an unfamiliar architecture must not be hidden.
    classifier = src.split("def _local_transformers_can_chat(", 1)[1]
    classifier = classifier.split("\ndef ", 1)[0]
    assert classifier.rstrip().endswith("return None")


def test_sources_dedupe_on_the_load_target_alone():
    """A repo holding both GGUF and safetensors yields a row in each cached
    list, but the backend resolves one target to one model and probes GGUF
    first, so keeping both spends a second attempt on the same files."""
    src = _read("features/chat/api/chat-adapter.ts")
    key = src.split("function autoLoadSourceKey", 1)[1].split("\n}", 1)[0]
    assert "return normalizeTarget(source.loadId);" in key
    assert "source.kind" not in key
    order = src.split("function orderAutoLoadSources", 1)[1].split("\n}\n", 1)[0]
    # Ordered first, so the survivor is the row the cascade would have reached.
    assert order.index("const ordered = [...sources].sort(") < order.index(
        "const seen = new Set<string>()"
    )


def test_variant_scans_take_the_run_signal():
    src = _read("features/chat/api/chat-adapter.ts")
    build = src.split("function buildAutoLoadSources", 1)[1]
    build = build.split("function isRememberedSource", 1)[0]
    assert build.count("signal,") == 2
    assert "options?.abortSignal,\n      )," in src


def test_cached_rows_classify_chat_capability_too():
    """The same encoder gate the scan-folder rows get; cached rows built their
    capabilities from file format alone."""
    src = _read_backend("hub/services/models/cache_inventory.py")
    assert "_local_transformers_can_chat" in src
    fields = src.split("def _cache_inventory_fields", 1)[1].split("\ndef ", 1)[0]
    assert "can_chat_override" in fields
    assert (
        'model_format in {"safetensors", "checkpoint"} and classify_snapshot is not None' in fields
    )


def test_every_load_target_comparison_uses_the_same_case_rules():
    """Source dedupe, remembered matching, and tried-candidate keys all compare
    load targets, so all three must agree on POSIX case or they disagree about
    which model is which."""
    src = _read("features/chat/api/chat-adapter.ts")
    for fn in ("autoLoadSourceKey", "isRememberedSource", "autoLoadCandidateKey"):
        body = src.split(f"function {fn}(", 1)[1].split("\n}", 1)[0]
        assert "normalizeTarget(" in body, fn
        assert "id.toLowerCase()" not in body, fn


def test_the_default_variant_lookup_takes_the_run_signal():
    src = _read("features/chat/api/chat-adapter.ts")
    helper = src.split("async function ensureDefaultModelDownloaded", 1)[1]
    helper = helper.split("async function autoLoadSmallestModel", 1)[0]
    assert (
        "listGgufVariants(DEFAULT_CHAT_MODEL_REPO, undefined, {\n      signal: abortSignal,\n    })"
        in helper
    )
    # The catch below swallows the abort rejection, so the guard after it is
    # what actually stops the download from starting.
    assert helper.index("} catch {") < helper.index("abortSignal?.throwIfAborted();")


def test_cached_rows_get_the_same_image_and_video_gate_as_local_rows():
    """Cached diffusion repos carry their task on the row and report can_chat
    true on file format alone."""
    src = _read("features/chat/api/chat-adapter.ts")
    cached = src.split("function isChattableCachedRepo", 1)[1].split("\n}\n", 1)[0]
    assert 'IMAGE_OR_VIDEO_TASKS.has(repo.task ?? "")' in cached
    local = src.split("function isAutoLoadableLocalRow", 1)[1].split("\n}", 1)[0]
    assert 'IMAGE_OR_VIDEO_TASKS.has(row.task ?? "")' in local
    # A chat GGUF is tagged text-generation, so the gate is a list, not "has a task".
    tasks = src.split("const IMAGE_OR_VIDEO_TASKS", 1)[1].split("]", 1)[0]
    assert '"text-generation"' not in tasks


def test_the_default_download_prepares_the_token_first():
    """startJob sends the stored token raw, with none of the recovery
    validateModel and loadModel get."""
    src = _read("features/chat/api/chat-adapter.ts")
    helper = src.split("async function ensureDefaultModelDownloaded", 1)[1]
    helper = helper.split("async function autoLoadSmallestModel", 1)[0]
    assert "prepareHfTokenForUse(hfToken)" in helper
    assert 'if (!prepared.proceed) return "cancelled";' in helper
    # After the on-disk check, so nothing prompts when there is no download,
    # and before the managed start.
    assert helper.index('return "ready"') < helper.index("prepareHfTokenForUse(hfToken)")
    assert helper.index("prepareHfTokenForUse(hfToken)") < helper.index(
        "downloadManager.requestStart(request)"
    )


def test_a_cancelled_download_is_never_handed_to_a_load():
    """A cancel can fail and the transfer finish anyway."""
    src = _read("features/chat/api/chat-adapter.ts")
    helper = src.split("async function ensureDefaultModelDownloaded", 1)[1]
    helper = helper.split("async function autoLoadSmallestModel", 1)[0]
    assert 'finish(cancelRequested ? "cancelled" : "ready")' in helper


def test_cached_classification_reads_the_snapshot_the_row_loads():
    """_scan_cached_models passes only `identity`, so reading snapshot_path
    alone classified nothing and the gate was a no-op in production."""
    src = _read_backend("hub/services/models/cache_inventory.py")
    fields = src.split("def _cache_inventory_fields", 1)[1].split("\ndef ", 1)[0]
    assert "classify_snapshot = identity.load_snapshot or snapshot_path" in fields
    assert "_local_transformers_can_chat(classify_snapshot)" in fields
    # The resolve must run before the classification reads it.
    assert fields.index("identity = _resolve_load_identity(") < fields.index("classify_snapshot =")


def test_non_chat_conditional_generation_is_excluded_before_the_suffix_check():
    """Whisper ends in ForConditionalGeneration; the order decides the answer."""
    src = _read_backend("hub/services/models/common.py")
    classifier = src.split("def _local_transformers_can_chat(", 1)[1]
    classifier = classifier.split("\ndef ", 1)[0]
    assert classifier.index("_NON_CHAT_GENERATIVE_MODEL_TYPES") < classifier.index(
        "_GENERATIVE_ARCHITECTURE_SUFFIXES"
    )
    assert '"whisper"' in src
    # Real seq2seq and multimodal chat models must not be listed.
    excluded = src.split("_NON_CHAT_GENERATIVE_ARCHITECTURES = frozenset(", 1)[1]
    excluded = excluded.split(")", 1)[0]
    assert "T5ForConditionalGeneration" not in excluded
    assert "Gemma" not in excluded


def test_format_gates_wait_for_the_server_reported_platform():
    """The store's initial chatOnly is a browser guess: a Mac browser on a
    remote Linux Studio would otherwise hide every local safetensors model."""
    src = _read("features/chat/api/chat-adapter.ts")
    gate = src.split("function runsOnThisPlatform", 1)[1].split("\n}", 1)[0]
    assert "if (!platform.fetched || !platform.isChatOnly()) return true;" in gate


def test_the_default_is_preflighted_before_the_managed_download():
    """A refusal from the training or placement guard must not cost several
    gigabytes first."""
    src = _read("features/chat/api/chat-adapter.ts")
    fallback = src.split("// Nothing on the device:", 1)[1]
    fallback = fallback.split("export function createOpenAIStreamAdapter", 1)[0]
    assert fallback.index("canAutoLoad({") < fallback.index("ensureDefaultModelDownloaded(")
    # One GPU snapshot feeds both, so the load sends what was cleared.
    assert fallback.index("const rt = useChatRuntimeStore.getState();") < fallback.index(
        "canAutoLoad({"
    )


def test_cached_non_gguf_rows_get_the_chat_only_platform_gate():
    """The picker hides them outright there (visibleCachedModelRows)."""
    src = _read("features/chat/api/chat-adapter.ts")
    gate = src.split("function cachedModelsRunOnThisPlatform", 1)[1].split("\n}", 1)[0]
    assert "return !platform.fetched || !platform.isChatOnly();" in gate
    assert "cachedModelsRunOnThisPlatform()\n          ? allModelRepos.filter(" in src
    # GGUF runs everywhere, so those rows stay ungated.
    assert "allGgufRepos.filter(isChattableCachedRepo)," in src


def test_bare_vision_and_audio_backbones_are_classified_non_chat():
    """A ViTModel class name has no task suffix, so only the model type
    identifies it."""
    src = _read_backend("hub/services/models/common.py")
    encoders = src.split("_ENCODER_ONLY_MODEL_TYPES = frozenset(", 1)[1]
    encoders = encoders.split(")", 1)[0]
    for model_type in ('"vit"', '"dinov2"', '"swin"', '"wav2vec2"', '"resnet"'):
        assert model_type in encoders, model_type
