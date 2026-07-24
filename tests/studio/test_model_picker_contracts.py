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
    # Load-only substitution of the resolved value.
    assert "maxSeqLength: maxSeqLengthValue" in src
    assert "const loadConfig" in src
    # The persisted record is loaded via onRun(loadConfig), and save uses the
    # untouched runtimeConfig (so a reset/default config stays default).
    assert "onRun(loadConfig)" in src
    assert "savePerModelConfig(" in src


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
    """A model visible in both the cached lists and the local inventory
    (e.g. a custom scan folder pointing into an HF cache) must not be tried
    twice."""
    auto_load = _autoload_section()
    assert "const seenLoadTargets = new Set<string>()" in auto_load
    assert "markSeen(repo.repo_id, repo.load_id, repo.cache_path)" in auto_load
    assert "isSeen(row.load_id, row.id, row.path, row.model_id)" in auto_load


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
    assert 'row.capabilities?.requires_variant === true' in resolve_fn
    assert "if (!isGguf) return null;" in resolve_fn
    assert "listGgufVariants(row.model_id || row.id" in resolve_fn
    assert "localPath: row.path" in resolve_fn
    assert "entry.downloaded && !entry.partial && isAutoLoadableGgufVariant(entry)" in resolve_fn
    # The cascade must keep directory GGUF rows as candidates.
    auto_load = src.split("async function autoLoadOnDeviceModel", 1)[1]
    assert 'row.model_format === "gguf" ||' in auto_load
    assert "await resolveLocalRowCandidate(row)" in auto_load
