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

import ast
import re
from pathlib import Path

WORKDIR = Path(__file__).resolve().parents[2]
FRONTEND = WORKDIR / "studio" / "frontend" / "src"


def _read(rel: str) -> str:
    path = FRONTEND / rel
    assert path.exists(), f"missing source file: {path}"
    return path.read_text(encoding = "utf-8")


def _split_args(captured: str) -> list[str]:
    """Split on top-level commas only, so `loadSpecFor(wanted, CATALOG)` survives."""
    args, depth, current = [], 0, ""
    for char in captured:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            args.append(current)
            current = ""
        else:
            current += char
    args.append(current)
    return args


def _code_only(source: str) -> str:
    """``source`` with comments removed and whitespace collapsed.

    A name discussed in prose is not a call. This file's own comments name the
    functions they explain, so a scan that skipped this would match the sentence.
    """
    source = re.sub(r"/\*.*?\*/", " ", source, flags = re.S)
    source = re.sub(r"//[^\n]*", " ", source)
    return " ".join(source.split())


def _call_arguments(text: str, callee: str) -> list[str]:
    """The argument text of each ``callee(...)`` call in ``text``, parens balanced.

    Per call, so a contract about what every call passes cannot be satisfied by a
    matching property somewhere else in the file, nor broken by one. An optional
    generic parameter list is skipped, since ``f<string>(...)`` is the same call.
    """
    calls = []
    for match in re.finditer(rf"\b{re.escape(callee)}\s*(?:<[^>]*>)?\s*\(", text):
        depth, start = 0, match.end() - 1
        for index in range(start, len(text)):
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
                if depth == 0:
                    calls.append(text[start + 1 : index])
                    break
        else:
            raise AssertionError(f"unbalanced parentheses after {callee} at {start}")
    return calls


def _read_backend(rel: str) -> str:
    path = WORKDIR / "studio" / "backend" / rel
    assert path.exists(), f"missing backend source file: {path}"
    return path.read_text(encoding = "utf-8")


# The frontend greps below have no parser to hand. The backend does, and a rule read out
# of the parse tree survives the edits that are not about it: a formatter that splits a
# set literal one name per line, a reordering, a renamed neighbour. Each helper raises
# rather than returning empty when it cannot find what it was pointed at, so a rule that
# moves reddens here instead of passing vacuously.
def _backend_function(rel: str, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """The definition of ``name`` in backend file ``rel``."""
    tree = ast.parse(_read_backend(rel), filename = rel)
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert len(found) == 1, f"{rel} defines {len(found)} functions named {name}, want 1"
    return found[0]


def _backend_class(rel: str, name: str) -> ast.ClassDef:
    """The definition of class ``name`` in backend file ``rel``."""
    tree = ast.parse(_read_backend(rel), filename = rel)
    found = [
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name
    ]
    assert len(found) == 1, f"{rel} defines {len(found)} classes named {name}, want 1"
    return found[0]


def _model_dump_exclusions(rel: str, function: str) -> set[str]:
    """The names the one ``model_dump(exclude = {...})`` inside ``function`` leaves out.

    Exactly one such call, or the answer would be a union across calls and dropping a
    name from the one that matters could hide behind another.
    """
    excluded: list[set[str]] = []
    for node in ast.walk(_backend_function(rel, function)):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "model_dump"):
            continue
        for keyword in node.keywords:
            if keyword.arg != "exclude":
                continue
            assert isinstance(keyword.value, ast.Set), f"{function}: exclude is not a set literal"
            names = {
                element.value
                for element in keyword.value.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }
            assert len(names) == len(
                keyword.value.elts
            ), f"{function}: exclude holds a name this check cannot read statically"
            excluded.append(names)
    assert (
        len(excluded) == 1
    ), f"{rel}:{function} has {len(excluded)} model_dump(exclude = ...) calls, want 1"
    return excluded[0]


def _annotated_field(rel: str, class_name: str, field: str) -> tuple[str, object]:
    """``(annotation, default)`` for ``field: <annotation> = <default>`` on ``class_name``."""
    for statement in _backend_class(rel, class_name).body:
        if not isinstance(statement, ast.AnnAssign):
            continue
        if not (isinstance(statement.target, ast.Name) and statement.target.id == field):
            continue
        assert isinstance(
            statement.value, ast.Constant
        ), f"{class_name}.{field} no longer defaults to a literal"
        return ast.unparse(statement.annotation), statement.value.value
    raise AssertionError(f"{rel}:{class_name} declares no field named {field}")


def _forwarded_keywords(rel: str, function: str, obj: str) -> set[str]:
    """Keyword arguments passed as ``name = <obj>.name`` anywhere inside ``function``.

    Same name on both sides, so this says the value reached the callee unmodified and
    under its own name, which is the part a caller downstream depends on.
    """
    forwarded = set()
    for node in ast.walk(_backend_function(rel, function)):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            value = keyword.value
            if (
                keyword.arg
                and isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == obj
                and value.attr == keyword.arg
            ):
                forwarded.add(keyword.arg)
    return forwarded


def _brace_matched_body(text: str, declaration: str) -> str:
    """The body of `declaration`, ending at ITS closing brace rather than at end of file.

    Splitting on a declaration and keeping the remainder looks like scoping but is not: the slice
    runs to EOF, so an ordering assertion inside it is still satisfied by code that has been moved
    out of the callback entirely. Match braces from the `{` that opens the body.
    """
    start = text.index(declaration)
    open_brace = text.index("{", text.index("=>", start))
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace : i + 1]
    raise AssertionError(f"unbalanced braces reading {declaration!r}")


def _override_lookup_candidates(*args, **kwargs) -> list[str]:
    """The real override-key ladder, imported rather than grepped out of inference.py: #8702 moved
    it to another module unchanged and took four contract tests red with it. The module is
    import-cheap (stdlib only at import time)."""
    import sys

    backend = str(WORKDIR / "studio" / "backend")
    if backend not in sys.path:
        sys.path.insert(0, backend)
    from utils.openai_auto_switch_settings import override_lookup_candidates

    try:
        return override_lookup_candidates(*args, **kwargs)
    except ModuleNotFoundError as missing:
        # The standalone-.gguf branch lazily imports hub.utils.gguf, which reaches structlog. CI
        # installs studio.txt; a bare `pytest tests/studio/...` does not. Skip on a missing
        # third-party package only -- a missing first-party module is a real break.
        if (missing.name or "").split(".")[0] in {"hub", "loggers", "utils", "core", "models"}:
            raise
        import pytest as _pytest
        _pytest.skip(f"needs the studio backend environment: {missing.name} is not installed")


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


def test_auto_offload_context_matches_picker_custom_seed():
    """Custom's safe seed must track the backend Auto offload context."""
    frontend = _read("features/model-picker/components/model-config-page.tsx")
    backend = _read_backend("core/inference/llama_cpp.py")
    frontend_value = re.search(r"\bAUTO_OFFLOAD_CONTEXT_LENGTH\s*=\s*(\d+)", frontend)
    backend_value = re.search(r"\b_AUTO_OFFLOAD_CTX\s*=\s*(\d+)", backend)
    assert frontend_value is not None
    assert backend_value is not None
    assert frontend_value.group(1) == backend_value.group(1)


def test_ui_safe_zone_anchor_tracks_the_auto_offload_context():
    """The published ceiling and the context Auto runs must come from one constant.

    ``max_context_length`` is the threshold the chat settings sheet warns above.
    When no GPU subset fits, Auto runs at ``_AUTO_OFFLOAD_CTX`` and the ceiling is
    anchored in the same branch. A literal there drifts the moment the constant
    moves, and the symptom is every Auto load warning about a context Auto chose
    for itself, so pin the reference rather than the value.
    """
    backend = _read_backend("core/inference/llama_cpp.py")
    anchor = re.search(
        r"max_available_ctx\s*=\s*min\(\s*([A-Za-z_0-9]+)\s*,\s*native_ctx_for_cap",
        backend,
    )
    assert anchor is not None, "the no-fit UI safe-zone anchor moved or was renamed"
    assert anchor.group(1) == "_AUTO_OFFLOAD_CTX"


def test_auto_offload_context_is_not_below_the_fit_floor():
    """Raising the Auto offload context is placement-neutral only above the fit
    floor: below it, the offload re-check starts awarding GPU residency again."""
    backend = _read_backend("core/inference/llama_cpp.py")
    auto = re.search(r"\b_AUTO_OFFLOAD_CTX\s*=\s*(\d+)", backend)
    floor = re.search(r"\b_FIT_MIN_CTX\s*=\s*(\d+)", backend)
    assert auto is not None and floor is not None
    assert int(auto.group(1)) >= int(floor.group(1))


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
    # One progress toast re-titled through every phase: the cascade updates it
    # directly, and the download keeps it live via the same updater.
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
    # How a degraded load is described belongs to one helper, not to this call site.
    #
    # This assertion has moved three times: it was `description: undefined`, then
    # `description: cpuFallbackReason` when the CPU-fallback branch appeared, then the
    # mmproj branch went in front of that. Each rewrite pinned a fresh spelling of an
    # inline conditional, and the third one shipped a real bug that no spelling-based
    # check could have caught: `mmproj ? ... : cpu ? ...` drops the CPU message when
    # both are set, so a session that lost GPU acceleration AND vision reported only
    # the vision loss.
    #
    # So this no longer describes the conditional at all. It requires the call site to
    # delegate, and the composition itself is tested where it lives, in
    # studio/frontend/tests/mmproj-fallback.test.ts, against both reasons together.
    # Sliced to the end of the helper body, not to the first `};`. That earlier bound
    # stopped at the `options` object literal, so anything declared after it -- the
    # toast severity, which is the half that decides whether a degraded load looks
    # like a plain success -- was silently outside the text being asserted on.
    success_toast = auto_load.split("const showAutoLoadSuccess", 1)[1]
    success_toast = success_toast.split("if (autoLoadToastDismissed)", 1)[0]
    assert "loadFallbackNotice(" in success_toast, (
        "the auto-load success toast no longer builds its notice through "
        "loadFallbackNotice. Inlining the conditional here is how the CPU-fallback "
        "message got dropped when the mmproj branch was added.\n" + success_toast
    )
    assert "description: notice.description" in success_toast, success_toast
    assert "notice.degraded ? toast.warning : toast.success" in success_toast, (
        "a degraded load must not raise a plain success toast:\n" + success_toast
    )
    # And the reasons still reach the helper, or it composes nothing.
    call = success_toast.split("loadFallbackNotice(", 1)[1].split(");", 1)[0]
    for reason in ("cpuFallbackReason", "mmprojFallbackReason"):
        assert reason in call, (
            f"{reason} is no longer passed to loadFallbackNotice, so that half of the "
            f"degradation is invisible:\n{call}"
        )
    assert "icon: undefined" in auto_load
    assert "duration: 5000" in auto_load
    assert "duration: 30000" not in auto_load
    assert auto_load.count("toast.dismiss(toastId)") >= 4

    explicit_load = _read("features/chat/hooks/use-chat-model-runtime.ts")
    assert "duration: Infinity" in explicit_load


def test_a_recipe_restores_the_previous_model_at_the_context_it_asked_for():
    """A recipe pins nothing, so restoring the model it displaced replays what it asked for."""
    src = _read("features/recipe-studio/hooks/use-recipe-executions.ts")
    assert src.count("requestedContextLength: status.requested_context_length ?? null,") == 2, src
    assert (
        "      max_seq_length:\n"
        "        requestedContextLength ??\n"
        "        unpinnedLoadContext(" in src
    ), src


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
    assert "useActiveModelConfig(" in _read("features/chat/chat-page.tsx")
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
    memo = re.search(r"const localModels = useMemo\(.*?\[inventory\.localRows", src, re.S)
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
    # single-quant GGUF row. #7880 added the fourth: the per-quant VRAM bar, which
    # has to price the pinned snapshot rather than the default ref.
    assert picker.count("loadId: c.load_id") == 4, (
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
    # Both cache-backed sources scan the exact path they will load from, not the
    # bare repo id.
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
    # MLX pins via customContextLength, so substituting the shown default would turn
    # "Auto" into a request for that number.
    assert "      : targetIsMlx\n        ? effectiveRuntimeConfig" in src
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
    # Commit returns null unless the field was edited, so an untouched control pins nothing.
    assert "dirtyRef.current" in numeric
    assert "return null;" in numeric
    # P2: blur clears dirtyRef after commit so Reset/slider cannot be
    # overwritten by a stale draft on a later Load.
    assert "dirtyRef.current = false;" in numeric
    assert "draftRef.current = String(final);" in numeric
    # Same-click Load after blur still sees the committed draft.
    assert "lastBlurCommittedRef" in numeric
    # Invalid drafts must not become an explicit pin.
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
    # The committed draft lands in this target's pin field.
    assert (
        "Object.assign(pendingPatch, contextPinPatch(committedMaxSeqLength, targetIsMlx));" in page
    )


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
    # An MLX pin is an override too, so Reset stays enabled, whichever field held it.
    assert (
        "const contextAtDefault = !target.isGguf "
        "? savedContextPin(config) == null "
        ": config.customContextLength == null;" in src
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
    # Same helper as the single-view load, so a pane cannot load at a different size.
    assert (
        "const effectiveMaxSeqLength = savedContextPin(ownConfig) ?? "
        "unpinnedLoadContext( targetIsGguf, "
        "isServedByMlx(targetIsGguf, platform.deviceType, platform.chatOnlyReason), "
        "DEFAULT_MAX_SEQ_LENGTH, );" in src
    )
    # The buggy fallback to the active model's shared runtime value must not return.
    assert "(isGgufLoad ? 0 : maxSeqLength)" not in src
    assert "const maxSeqLength = store.params.maxSeqLength;" not in src


def test_every_load_path_asks_the_backend_before_it_asks_for_a_window():
    """Which backend serves and what window it reported decide the request and the Max
    Tokens ceiling. A literal for the first makes an interactive MLX load ask for the app
    default again; a raw context field for the second raises Max Tokens to meet a length
    nobody measured."""
    load_paths = ("chat/hooks/use-chat-model-runtime.ts", "chat/api/chat-adapter.ts")
    derived_backend = r"isMlx: isServedByMlx\(\s*[\w.=\" ]+,\s*platform\.deviceType,\s*platform\.chatOnlyReason,?\s*\)"
    for name in load_paths:
        src = _read(f"features/{name}")
        calls = src.count("resolveLoadMaxSeqLength({") + src.count("retainedContextPin({")
        derived = len(re.findall(derived_backend, src))
        assert derived >= calls, (name, derived, calls)
    for name in (*load_paths, "chat/lib/apply-inference-status-to-store.ts"):
        src = _read(f"features/{name}")
        # maxTokensCap is exempt: it only lowers a budget, and a transformers load
        # reports there the max_seq_length it was configured with.
        windows = re.sub(r"maxTokensCap:[^,]*,", "", src)
        assert not re.search(r"\w+\.context_length\b", windows), name
    # Including the fallback: on a model change the session length is the outgoing one's.
    policy = _read("features/chat/presets/preset-policy.ts")
    assert (
        "localMaxTokensCeiling(\n    loadedContextLength,\n"
        "    unreportedWindowMaxTokens(response.is_gguf ?? false, current.maxTokens)," in policy
    )
    # No window reported falls back to what this load asked for, not the app default,
    # which halved Max Tokens for a transformers model carrying more.
    adapter = _read("features/chat/api/chat-adapter.ts")
    assert re.search(r"MaxTokensCeiling\(\s+loadedContextFields[^;]+loadedWindow,", adapter)
    assert re.search(r"ContextForParams\(\s+loadedContext[^,]+,\s+effectiveMaxSeqLength,", adapter)


def test_an_mlx_target_is_offered_a_context_length_not_a_sequence_length():
    """MLX sizes its own window, so the control is GGUF's Context Length and states what a
    load would serve. Max Seq Length at 4096 would describe a pin never sent."""
    page = _read("features/model-picker/components/model-config-page.tsx")
    assert 'const label = isMlx ? "Context Length" : "Max Seq Length";' in page
    # A number, not a word: the placeholder is only for a window nobody has read.
    assert 'displayValue={isMlx && windowUnknown ? "—" : undefined}' in page
    assert "savedContextPin(config) == null && mlxServedWindow == null\n" in page
    # The resident model's window, else this model's; request bounds would shorten it.
    assert "(targetIsMlx && isActiveModel ? servedWindow(loadedContextLength) : null) ??" in page
    assert "? servedWindow(modelMaxPosition.maxPositionEmbeddings)" in page
    assert re.search(r"const servedWindow = [^;]*Math\.floor\(value\)\n\s*: null;", page), page
    numeric = _read("features/model-picker/components/numeric-value-input.tsx")
    # Typing the shown number is a choice even where it equals the value beneath it.
    assert "derived={isMlx && !pinned}" in page
    assert "pinned={savedContextPin(config) != null}" in page
    # Committing pins that exact number; one outside the control's range is no commit,
    # which is why the slider stays inside it too.
    assert "const shown = parsed === value;" in numeric
    assert re.search(r"shown && \(\(max[^{]+parsed < min\)+ \{\n\s+return null;", numeric)
    assert "const final = commitDraft(draftRef.current);\n          dirtyRef" in numeric
    assert "value={maxSeqLengthValue}\n              max={maxSeqLengthMax}" in page
    assert "value={[Math.min(Math.max(value, MAX_SEQ_LENGTH_MIN), max)]}" in page
    assert "const final = shown ? parsed : snapToStep(parsed, step, min, max);" in numeric
    assert re.search(
        r"MAX_SEQ_LENGTH_MAX,\s+Math\.max\(native\w+, maxSeqLengthValue\),\s+\);", page
    )
    # A record written before the pin moved fields carries it in maxSeqLength, and wins.
    assert "servedWindow(savedContextPin(config)) ??\n    mlxServedWindow ??" in page
    # A hidden value and a value nobody chose both still count, through one predicate.
    assert "final !== value || displayValue != null || derived;" in numeric
    assert "if (isEdit(final)) {\n      onChange(final);\n    }\n    return final;" in numeric
    assert "lastBlurCommittedRef.current = isEdit(final) ? final : null;" in numeric
    assert "update(contextPinPatch(value, targetIsMlx))" in page
    # The platform answers which backend serves: "anything not GGUF" relabels CUDA.
    assert (
        "isServedByMlx(\n    target.isGguf,\n    platform.deviceType,\n    platform.chatOnlyReason,\n  )"
        in page
    )
    # Both props are optional, so dropping either typechecks and mislabels the control.
    assert "isMlx={targetIsMlx}" in page
    assert "windowUnknown={" in page


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
    pending_gate = page.split("const stagedMetadataPending =", 1)[1].split(";", 1)[0]
    assert "config.nBatch != null" in pending_gate
    assert "config.nUbatch != null" in pending_gate
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
    reloads, and Unsloth restarts must never re-migrate, duplicate records, or overwrite
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
        # The middle argument, the routed filename, has been respelled more than once;
        # only the third is load-bearing, so do not pin the second.
        call = re.search(r"diffusionRoutePick\(\s*wanted,\s*(.*?),?\s*\);", src, re.S)
        assert call, f"{rel}: the routed pick does not go through diffusionRoutePick"
        # By position, not by presence: `diffusionRoutePick(wanted, routedFilename ??
        # loadSpecFor(wanted, IMAGE_CATALOG)?.filename)` type-checks, passes the spec's
        # filename as the quant, drops the spec, and would pass a substring check while
        # curated single-file artifacts load as GGUF.
        args = _split_args(call.group(1))
        assert len(args) >= 2, f"{rel}: the routed pick passes no spec argument"
        assert (
            f"loadSpecFor(wanted, {catalog})" in args[1]
        ), f"{rel}: the routed pick passes no catalog spec"


def test_a_quantized_load_drops_a_lora_selection_it_cannot_bake():
    """int8/fp8 builds take adapters only at load time. Switching artifact inside one family
    keeps the selection (same family, no clear) while the load did not bake it, so Generate
    would 400 with the picker still showing the adapter as active."""
    src = _read("features/images/images-page.tsx")
    assert "bakedLorasOnLoad.current = bakeLoras.length > 0;" in src
    guard = re.search(
        r"if \(\s*!loraCapable \|\| checkedBuildForBake\.current === residentBuildKey\s*\)\s*return;.*?\n  \}, \[",
        src,
        re.S,
    )
    assert guard, "the bake-only check is missing"
    body = guard.group(0)
    assert '"int8"' in body and '"fp8"' in body
    assert "bakedLorasOnLoad.current" in body, "a baked selection must be kept"
    assert "setLoras([])" in body and "toast.info(" in body, "cleared without telling the user"


def test_diffusion_pages_never_drop_a_gguf_pick_silently():
    """The fallback branch splits a local path, so a repo pick reaching it has no
    filename. It used to error; it now resolves the repo's own .gguf. Either way the
    branch must act: returning with no request and no toast drops the pick."""
    for rel in ("features/images/images-page.tsx", "features/video/video-page.tsx"):
        src = _read(rel)
        branch = re.search(
            r'if \(!filename\.toLowerCase\(\)\.endsWith\("\.gguf"\)\) \{.*?\n        \}', src, re.S
        )
        assert branch, f"{rel}: gguf extension guard not found"
        body = branch.group(0)
        assert (
            "toast.error(" in body or "loadGgufRepoPick(" in body
        ), f"{rel}: guard returns silently"


def test_diffusion_pages_stage_downloads_through_the_manager():
    """Images/Video must not download inside the load: a Hub pick with missing files goes
    to the download manager first, so it shares progress, cancel/resume, disk preflight,
    and manifest verification with every other model."""
    for rel in ("features/images/images-page.tsx", "features/video/video-page.tsx"):
        src = _read(rel)
        assert "useStagedDownload" in src, f"{rel}: not wired to the download manager"
        # The plan carries the loader's own file scope, so nothing extra is pulled.
        assert "DownloadPlan(" in src, f"{rel}: does not fetch a download plan"
        stage_fn = re.search(r"const loadOrStage = useCallback\(.*?\n  \);", src, re.S)
        assert stage_fn, f"{rel}: loadOrStage not found"
        body = stage_fn.group(0)
        # A cached GGUF can still be missing a separate text encoder or VAE, and only the plan
        # sees that, so every Hub pick is planned and only local picks bypass it. Safe on both
        # pages because both planners filter against the cache: a fully cached pick returns no
        # entries. Flipping this on a planner that does not filter would re-stage a whole model.
        assert 'source !== "hub"' in body, f"{rel}: local picks would be planned"
        assert (
            "isDownloaded !== false" not in body
        ), f"{rel}: cached checkpoint would hide missing companion assets"
        # A missing plan must still load rather than dead-end.
        assert "catch" in body, f"{rel}: no fallback when the plan is unavailable"

        assert "handleLoadRef.current(repoId, opts, advanced)" in body, rel


def test_every_diffusion_planner_filters_the_cache_before_staging():
    """Planning every Hub pick is only safe on a planner that skips files already on disk;
    without that an unchanged, fully cached model stages its whole footprint again. The two
    move together, so pin them together rather than leaving it to a comment."""
    root = WORKDIR / "studio" / "backend" / "core" / "inference"
    for name in ("diffusion.py", "sd_cpp_backend.py", "video.py"):
        src = (root / name).read_text(encoding = "utf-8")
        plan = re.search(r"def download_plan\(.*?\n    (?=@|def )", src, re.S)
        assert plan, f"{name}: download_plan not found"
        # Either probe: `_hub_file_is_loadable` is the stricter one, adding the stale-live-copy
        # check on top, and a planner may reasonably use it instead.
        assert "_hub_file_is_cached" in plan.group(0) or "_hub_file_is_loadable" in plan.group(
            0
        ), f"{name}: download_plan stages files without checking the cache"


def test_image_load_fallback_names_requirements_instead_of_only_the_model():
    """If planning fails and the backend has to fetch files inside the load, its toast must
    not call companion text encoders and VAEs the selected model. The normal path stages
    them through Downloads; this wording keeps the defensive fallback honest too."""
    src = _read("features/images/images-page.tsx")
    assert '"Downloading model requirements…"' in src
    assert '"Downloading model…"' not in src
    assert '"Downloading the files required to load this model."' in src


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
        # The deps array grew when the load body moved into runStagedLoad, so match the
        # effect by its guard and check the deps separately. Keying the boundary on
        # `[active` instead lets a reordered array run the match on into the next hook,
        # where loadOrStage's own handleLoadRef call satisfies the loader assertion below
        # for a flush that loads nothing.
        flush = re.search(
            r"if \(!active \|\| !stagedLoadDeferred\.current\) return;(.*?)\n  \}, \[([^\]]*)\]\);",
            src,
            re.S,
        )
        assert flush, f"{rel}: nothing flushes the deferred load when the page is shown"
        deps = [dep.strip() for dep in flush.group(2).split(",")]
        assert "active" in deps, f"{rel}: the flush does not re-run when the page is shown"
        assert "runStagedLoad(pending);" in flush.group(0), f"{rel}: deferred load never runs"


def test_a_staged_download_that_ends_rolls_back_the_optimistic_quant():
    """A quant pick sets its label optimistically and hands the rollback to whoever learns the
    load did not take: the `.then` when the load never STARTS, the progress poll when it fails
    after starting. A staged pick has neither -- staging starts no load, so nothing polls, and
    `loadOrStage` returns true when it stages, so the `.then` treats it as started.

    So the label has to come back where the plan dies (cancelled, failed, or never started), or
    the selector goes on describing the still-resident model with a quant nothing ever loaded --
    and images-page writes that label into the gallery cache, so it survives a remount too."""
    for rel in ("features/images/images-page.tsx", "features/video/video-page.tsx"):
        src = _read(rel)
        cancelled = re.search(r"onCancelled: \(\) => \{.*?\n    \},", src, re.S)
        assert cancelled, f"{rel}: staged-download onCancelled not found"
        region = cancelled.group(0)
        # The pick itself still has to go, or a late completion loads a model nobody asked for.
        assert "pendingStagedLoad.current = null" in region, f"{rel}: the dead pick is kept"
        assert "quantRevert.current" in region, f"{rel}: the pending quant rollback is ignored"
        assert re.search(
            r"revertPick\(quantRevert\.current\)", region
        ), f"{rel}: the optimistic quant outlives the download that was supposed to justify it"
        assert "quantRevert.current = null" in region, f"{rel}: the rollback is never consumed"


def test_a_dying_staged_download_only_rolls_back_its_own_pick():
    """Staging leaves `busy` null on purpose, so a second Hub pick can be made while the first
    job is still alive. `quantRevert` is a single ref, so by the time the first job dies it can
    already hold the SECOND pick's entry: rolling back then reverts a label the newer, still-live
    pick owns, and nothing restores it when that pick goes on to stage and load.

    So the rollback has to be bound to the pick that staged the job. `loadOrStage` reads the
    entry BEFORE awaiting its plan (the await is the window in which a newer pick lands) and
    records it when it stages; the cancel path reverts only on an identity match."""
    for rel in ("features/images/images-page.tsx", "features/video/video-page.tsx"):
        src = _read(rel)
        # Captured before the plan await, otherwise it is the newer pick's entry that gets stored.
        own = re.search(r"const ownRevert = quantRevert\.current;\n(.*?)await ", src, re.S)
        assert own, f"{rel}: loadOrStage does not capture its own rollback entry"
        assert "await" not in own.group(
            1
        ), f"{rel}: ownRevert is read after an await, so it can be the newer pick's"
        assert (
            "stagedQuantRevert.current = ownRevert" in src
        ), f"{rel}: the staged job records no owner"

        cancelled = re.search(r"onCancelled: \(\) => \{.*?\n    \},", src, re.S)
        assert cancelled, f"{rel}: staged-download onCancelled not found"
        region = cancelled.group(0)
        assert re.search(
            r"if \(\s*quantRevert\.current &&\s*quantRevert\.current === stagedQuantRevert\.current\s*\)",
            region,
        ), f"{rel}: a dead job can roll back a newer pick's quant label"
        # Cleared either way, or a later job inherits this one's owner and reverts on its behalf.
        assert (
            "stagedQuantRevert.current = null" in region
        ), f"{rel}: the staged owner is never released"


def test_a_plan_that_lands_after_a_newer_pick_is_dropped():
    """Staging never sets `busy`, so a second Hub pick passes handleModelSelect's guard while the
    first plan is still in flight. Plans then resolve in RESPONSE order, not pick order: the older
    one would restage over the newer queue, or fall through and load the model the user left.

    So each pick takes a sequence number and gives up if a newer one has been made since. It must
    report started, not failed: returning false would send this pick's `.then` rollback at a label
    the newer pick now owns. Every exit that acts on the pick is covered, not just the one after a
    successful plan -- a rejected plan falls through to the load, and a pick that never asks for a
    plan at all (local, exported) must still invalidate one already in flight."""
    for rel in ("features/images/images-page.tsx", "features/video/video-page.tsx"):
        src = _read(rel)
        body = re.search(r"const loadOrStage = useCallback\(\n(.*?)\n  \);", src, re.S)
        assert body, f"{rel}: loadOrStage not found"
        text = body.group(1)
        assert "const pick = ++pickSeq.current;" in text, f"{rel}: no pick sequence is taken"
        # Before any real await, or two picks can share a number.
        seq = text.index("const pick = ++pickSeq.current;")
        first_await = min(
            (
                text.index(tok)
                for tok in ("await requestDownloadPlan", "await getVideoDownloadPlan")
                if tok in text
            ),
            default = len(text),
        )
        assert seq < first_await, f"{rel}: the sequence is taken after the plan await"
        # Before the non-hub return, so a local pick invalidates an in-flight hub plan.
        assert seq < text.index(
            'if (source !== "hub")'
        ), f"{rel}: a non-hub pick returns without invalidating an in-flight hub plan"
        guards = re.findall(
            r"if \(pick !== pickSeq\.current(?: \|\| !owns\(\))?\) return (\w+);", text
        )
        assert guards, f"{rel}: a superseded plan is not dropped"
        assert (
            set(guards) == {"true"}
        ), f"{rel}: a superseded pick reports failure, so its rollback fires at the newer pick's label"
        # The fallback load after a rejected plan is guarded too.
        tail = text[text.rindex("} catch {") :]
        assert re.search(
            r"if \(pick !== pickSeq\.current(?: \|\| !owns\(\))?\) return true;.*?return handleLoadRef",
            tail,
            re.S,
        ), f"{rel}: a plan that rejected after a newer pick still reaches the fallback load"


def test_a_pick_that_never_loads_restores_its_generation_recipe():
    """A pick applies its model's step/guidance recipe at the same moment it sets the quant label,
    optimistically. If the load never takes, the previous pipeline stays resident: restoring only
    the label leaves a distilled model's low-step, guidance-0 recipe pointed at a non-distilled
    model, and the next generation silently runs with the wrong settings.

    So the rollback token carries the recipe and every rollback path puts all of it back.

    The token may carry more than the recipe (a preset claim, what the pick applied), so the
    fields are matched inside the declaration rather than against one exact line."""
    for rel in ("features/images/images-page.tsx", "features/video/video-page.tsx"):
        src = _read(rel)
        token = re.search(r"type PickRevert = \{(.*?)\n\};", src, re.S) or re.search(
            r"type PickRevert = \{(.*?)\};", src, re.S
        )
        assert token, f"{rel}: no PickRevert rollback token"
        for field in ("prev: string | null", "steps: number", "guidance: number"):
            assert field in token.group(1), f"{rel}: the rollback token does not carry {field}"
        revert = re.search(
            r"const revertPick = useCallback\(\(r: PickRevert\) => \{(.*?)\}, \[\]\);", src, re.S
        )
        assert revert, f"{rel}: no shared rollback helper"
        body = revert.group(1)
        # The recipe may be restored conditionally (a preset chosen after the pick owns those
        # fields), but the rollback still has to read it off the token.
        for setter in ("setQuant(r.prev)", "setSteps(", "r.steps", "setGuidance(", "r.guidance"):
            assert setter in body, f"{rel}: rollback does not restore {setter}"
        # No rollback path may still put back the label alone.
        assert (
            "setQuant(quantRevert.current.prev)" not in src
        ), f"{rel}: a rollback path restores the label without its recipe"


def test_every_pick_replaces_the_rollback_it_leaves_behind():
    """`quantRevert` is one ref and the staged cancel path reverts on identity. A branch that
    changes the quant or the recipe WITHOUT writing a new entry leaves the previous pick's entry
    in place, so an older staged download cancelling later still matches, and reverts to state
    from before a selection this pick already replaced -- while this pick keeps no rollback of
    its own. Every branch that moves the selection registers its own entry."""
    for rel in ("features/images/images-page.tsx", "features/video/video-page.tsx"):
        src = _read(rel)
        select = re.search(r"const handleModelSelect = useCallback\(\n(.*?)\n  \);", src, re.S)
        assert select, f"{rel}: handleModelSelect not found"
        body = select.group(1)
        # One installed entry per branch that moves the label. Counting is what catches the
        # branches nobody thinks about: the curated non-GGUF one and the generic pipeline
        # fall-through both shipped without an entry at different points.
        moves = body.count("setQuant(")
        installs = body.count("quantRevert.current = revert;")
        assert moves == installs, (
            f"{rel}: {moves} branches move the selection but only {installs} install a rollback, "
            "so an older staged download can revert over a pick that already replaced it"
        )


def test_every_pick_route_invalidates_the_staged_intent():
    """Clearing inside `loadOrStage` is not enough: the direct-local GGUF and safetensors branches
    call `handleLoad` themselves and never go through it, so a staged Hub download kept its intent
    and its `onReady` could load the abandoned Hub model over the local one just picked.

    So the invalidation sits in one helper fired at the top of every pick, before any branch."""
    for rel in ("features/images/images-page.tsx", "features/video/video-page.tsx"):
        src = _read(rel)
        helper = re.search(r"const beginPick = useCallback\(\(\) => \{(.*?)\}, \[\]\);", src, re.S)
        assert helper, f"{rel}: no shared pick-invalidation helper"
        body = helper.group(1)
        for cleared in (
            "pickSeq.current += 1;",
            "pendingStagedLoad.current = null;",
            "stagedLoadDeferred.current = false;",
            "stagedQuantRevert.current = null;",
        ):
            assert cleared in body, f"{rel}: beginPick does not release {cleared}"
        select = re.search(r"const handleModelSelect = useCallback\(\n(.*?)\n  \);", src, re.S)
        assert select, f"{rel}: handleModelSelect not found"
        pick = select.group(1)
        assert "beginPick();" in pick, f"{rel}: a pick can run without invalidating the last one"
        # Before every branch, or the branch that returns first keeps the old intent armed.
        first_branch = min(
            pick.index(tok)
            for tok in ("const spec = loadSpecFor(", "if (meta.ggufVariant")
            if tok in pick
        )
        assert (
            pick.index("beginPick();") < first_branch
        ), f"{rel}: the invalidation runs after a branch that can already have returned"


def test_a_rejected_pick_hands_the_resident_state_back():
    """`beginPick` retires the staged pick before the new row is validated, so a pick that is then
    REJECTED (a bare repo with no quant, a non-unsloth pipeline) loads nothing and has nothing left
    to restore it: the selector would show the abandoned pick's quant and recipe indefinitely.

    Every rejecting early return therefore hands the carried rollback back."""
    for rel in ("features/images/images-page.tsx", "features/video/video-page.tsx"):
        src = _read(rel)
        helper = re.search(r"const abandonPick = useCallback\(\(\) => \{(.*?)\}, \[", src, re.S)
        assert helper, f"{rel}: no rejected-pick restore helper"
        body = helper.group(1)
        assert "revertPick(quantRevert.current)" in body, f"{rel}: the label is not handed back"
        assert "quantRevert.current = null" in body, f"{rel}: the entry is never consumed"

        select = re.search(r"const handleModelSelect = useCallback\(\n(.*?)\n  \);", src, re.S)
        assert select, f"{rel}: handleModelSelect not found"
        pick = select.group(1)
        # Each toast.error that ends the pick must restore before returning.
        rejects = re.findall(r"toast\.error\([^;]*\);\n(\s*)([^\n]*)\n", pick)
        assert rejects, f"{rel}: no rejecting early return found; this guard has gone stale"
        for _indent, following in rejects:
            assert (
                "abandonPick();" in following
            ), f"{rel}: a rejected pick returns without restoring the state it superseded"


def test_a_new_pick_drops_the_previous_staged_intent():
    """A staged download outlives the pick that made it. If the next pick stages nothing of its
    own -- fully cached, local, or no plan at all -- it never calls `stage()`, so the hook's queue
    keeps running the OLDER job and its `onReady` loads the model the user moved away from,
    evicting the one they actually chose. The pick sequence alone does not cover this: the older
    job already staged, so there is no pending response left to invalidate.

    So the intent is dropped at the start of every pick, before any early return, and a pick that
    does stage simply writes a fresh one."""
    for rel in ("features/images/images-page.tsx", "features/video/video-page.tsx"):
        src = _read(rel)
        body = re.search(r"const loadOrStage = useCallback\(\n(.*?)\n  \);", src, re.S)
        assert body, f"{rel}: loadOrStage not found"
        text = body.group(1)
        cleared = text.index("pendingStagedLoad.current = null;")
        assert cleared < text.index(
            'if (source !== "hub")'
        ), f"{rel}: a non-hub pick returns while the previous staged intent is still armed"
        assert cleared < text.index("await "), f"{rel}: the intent survives until the plan resolves"
        # The deferred re-fire and the rollback owner belong to that dead intent too.
        head = text[: text.index('if (source !== "hub")')]
        assert (
            "stagedLoadDeferred.current = false;" in head
        ), f"{rel}: a deferred staged load can still fire for the abandoned pick"
        assert (
            "stagedQuantRevert.current = null;" in head
        ), f"{rel}: the dead intent keeps ownership of the rollback"


def test_a_local_gguf_still_shows_its_remote_companion_footprint():
    """Only the CHECKPOINT is on disk for a local GGUF directory. Its text encoder, VAE, tokenizer
    and configs still come from the remote base, and both diffusion planners size them, so
    suppressing the footprint request understated a local row by the larger half of the download.

    The arithmetic differs though: a local checkpoint is not part of `required_bytes` at all, so
    nothing may be subtracted for it, where a hub pick carries its checkpoint inside that total.
    "On disk" is the listing's verdict, not the spelling of the id, so the gate is
    `checkpointIsLocal` rather than the path prefix test alone."""
    src = _read("features/model-picker/components/model-selector/pickers.tsx")
    effect = re.search(r"setCompanionBytesByKey\(new Map\(\)\);(.*?)\n  \}, \[", src, re.S)
    assert effect, "footprint resolution effect not found"
    body = effect.group(1)
    guard = re.search(r"if \(!resolveDownloadFootprint([^)]*)\) \{", body)
    assert guard, "the footprint effect has no bail-out guard"
    assert "isLocalPath" not in guard.group(
        1
    ), "a local pick skips the footprint request, hiding its remote companion set"
    assert re.search(
        r"const checkpoint = checkpointIsLocal\n?\s*\? 0", body
    ), "a local checkpoint is subtracted from a total it was never part of"


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


def test_staged_downloads_use_one_actionable_download_surface():
    """The Downloads panel owns progress and cancellation. A second informational toast
    duplicates the same state and gives users another X that only dismisses copy."""
    staged = _read("features/hub/download-manager/use-staged-download.ts")
    stage_fn = re.search(
        r"const stage = useCallback\(\(entries: StagedDownloadEntry\[\]\) => \{.*?\n  \}, \[\]\);",
        staged,
        re.S,
    )
    assert stage_fn, "staged-download stage callback not found"
    assert "toast.info" not in stage_fn.group(0)

    panel = _read("features/hub/download-manager/download-manager-panel.tsx")
    assert 'job.variant?.startsWith("@")' in panel
    assert '"Model file" : "Required assets"' in panel


def test_staged_plans_label_the_checkpoint_without_guessing_from_the_extension():
    """The panel's "Model file" vs "Required assets" suffix must come from the plan, not
    from a filename. A checkpoint is not always a GGUF (the curated LTX single-file
    artifact is one ~90GB .safetensors) and companion repos carry .safetensors too, so an
    extension test mislabelled the model itself as "Required assets". The staging page is
    the only place that knows: the entry carrying the picked checkpoint file. Repo identity
    alone is not enough -- a checkpoint sharing its repo with the companions, and already
    cached, leaves an entry of companion files that would still claim to be the model."""
    for page in ("images/images-page.tsx", "video/video-page.tsx"):
        src = _read(f"features/{page}")
        entries = re.search(r"plan\.entries\.map\(\(e\) => \(\{.*?\}\)\)", src, re.S)
        assert entries, f"{page} does not map the plan entries into staged downloads"
        assert "e.files.includes(opts.filename)" in entries.group(
            0
        ), f"{page} does not mark the picked repo's entry as the checkpoint"
        # The plan's own answer wins over both local guesses. A gated pipeline is staged from an
        # ungated MIRROR, so its entry no longer carries the id we picked and the repo-id test
        # reads the whole selected model as "Required assets". Only the planner knows about the
        # swap. `??`, not `||`: a planner that answers false must not fall through to a guess.
        assert "e.checkpoint ??" in entries.group(
            0
        ), f"{page} ignores the checkpoint flag the plan carried"

    staged = _read("features/hub/download-manager/use-staged-download.ts")
    assert "checkpoint?: boolean;" in staged
    start = re.search(r"downloadManager\.requestStart\(\{.*?\}\);", staged, re.S)
    assert start, "requestStart call not found"
    assert "checkpoint: current.checkpoint," in start.group(0)

    # Carried onto the job and back out of persisted state, or a restart loses the label.
    poll = _read("features/hub/download-manager/poll-loop.ts")
    assert "checkpoint: req.checkpoint" in poll
    state = _read("features/hub/download-manager/download-manager-state.ts")
    assert 'typeof value.checkpoint === "boolean"' in state
    assert "{ checkpoint: job.checkpoint }" in state

    panel = _read("features/hub/download-manager/download-manager-panel.tsx")
    suffix = re.search(r"function variantSuffix\(.*?\n\}", panel, re.S)
    assert suffix, "variantSuffix not found"
    body = suffix.group(0)
    assert "job.checkpoint ??" in body, "the label ignores the flag the plan carried"
    # The .gguf guess may only survive as the fallback for jobs persisted before the flag.
    assert body.index("job.checkpoint ??") < body.index(".gguf")


def test_local_model_sections_respect_the_task_filter():
    """LM Studio / ./models / custom-folder rows must honour the picker's task filter.
    The backend tags every local model with a task for exactly this; without the gate the
    Images picker listed chat GGUFs (which 400 on a diffusion load) and buried the
    diffusion models the page can actually run."""
    src = _read("features/model-picker/components/model-selector/pickers.tsx")
    for memo in ("sortedLmStudio", "sortedLocalDir", "sortedCustomFolderModels"):
        block = re.search(rf"const {memo} = useMemo\(.*?\n  \);", src, re.S)
        assert block, f"{memo} not found"
        assert re.search(
            r"passesTaskGate\(\s*m\.task", block.group(0)
        ), f"{memo} does not apply the task gate"


def test_chat_picker_routes_diffusion_picks_to_their_page():
    """Chat cannot load a diffusion or audio model. Rather than hiding an on-device one or
    letting it 400, the unfiltered picker routes the pick to the page that loads it."""
    src = _read("features/model-picker/components/model-selector/pickers.tsx")
    gate = re.search(r"function passesTaskGate\(.*?\n\}", src, re.S)
    assert gate, "passesTaskGate not found"
    # The chat branch no longer drops the generation tasks outright.
    assert "UNSUPPORTED_DIFFUSION_TASK" in gate.group(0)
    wrapper = re.search(r"const onSelect = useCallback\(.*?\n  \);", src, re.S)
    assert wrapper, "the routing wrapper around onSelect is missing"
    body = wrapper.group(0)
    assert "mediaPageForTask" in body and "navigateToPage" in body
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
    assert re.search(
        r"row\.capabilities\.canChat \|\|\s*studioPageForTask\(row\.task\) !== undefined", conv
    )


def test_local_diffusion_routing_is_keyed_by_the_id_the_row_selects():
    """A local row's click passes m.id (a filesystem load id), while m.model_id is its HF-style
    name. Keying the routing map on one alone let the lookup miss, so the pick fell through to the
    chat loader instead of navigating to Images or Video."""
    src = _read("features/model-picker/components/model-selector/pickers.tsx")
    block = re.search(r"const diffusionTaskById = useMemo\(.*?\n  \}, \[", src, re.S)
    assert block, "diffusionTaskById not found"
    body = block.group(0)
    assert re.search(r"put\(m\.id, m\.task[,)]", body) and re.search(
        r"put\(m\.model_id, m\.task[,)]", body
    )


def test_a_staged_download_that_never_starts_clears_the_queue():
    """requestStart can answer "error" (network failure, rejected scoped request, worker refused),
    "conflict" or "busy". Nothing completes after any of them, so leaving the head in place strands
    the pick: the effect never re-runs and onReady never fires. The consumer's pending auto-load
    has to go with it, or a later completion loads a model nobody asked for.

    Asserted over the whole non-started region rather than a fixed window after the first branch,
    so one shared clean-up for all three outcomes passes and three copies would too."""
    src = _read("features/hub/download-manager/use-staged-download.ts")
    assert 'if (outcome === "started") return;' in src
    region = src[src.index('if (outcome === "started") return;') : src.index("return () => {")]
    for outcome in ("error", "conflict", "busy"):
        assert f'outcome === "{outcome}"' in region
    assert "setQueue(null)" in region
    assert "onCancelledRef.current?.()" in region


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
    assert "sawActive" in fn
    assert "did not reach the server" in fn
    # The proof used to be an inline knownIds set; it is now a baseline handed
    # to hasUnknownRecord in lib/gallery-flags.ts. Same proof, named helper, so
    # the assertion follows it rather than pinning the old spelling.
    assert "hasUnknownRecord(" in fn
    assert "hasUnknownRecord(\n        baseline," in fn
    # The caller still snapshots the ids BEFORE the POST and passes THAT SET in.
    # Pin the argument, not the mere presence of the snapshot expression: with a
    # fresh `new Set()` in the knownIds slot the snapshot still exists, is still
    # taken before the POST and `probeBaseline` still reaches the probe, yet every
    # record already on the page reads as unknown and a POST that never landed is
    # reported as a finished image. Whitespace is normalised first, so reformatting
    # the call cannot break the pin.
    snapshot_pattern = (
        r"const (\w+) = new Set\(galleryCache\.images\.map\(\(image\) => image\.id\)\);"
    )
    assert len(re.findall(snapshot_pattern, src)) == 1, "the pre-POST id snapshot is not unique"
    snapshot = re.search(snapshot_pattern, src)
    known_ids = snapshot.group(1)
    flat = " ".join(src.split())
    call = re.search(r"const probeBaseline = newRecordProbeBaseline\(\s*(.*?)\s*\);", flat)
    assert call, "probeBaseline is no longer built by newRecordProbeBaseline"
    args = [arg.strip() for arg in call.group(1).split(",") if arg.strip()]
    assert args == ["galleryCache.images", "galleryCache.hasMore", known_ids], args
    baseline = src.index("const probeBaseline = newRecordProbeBaseline(")
    # Snapshot, then baseline, then the POST -- in that order.
    assert snapshot.start() < baseline
    assert baseline < src.index("await generateDiffusionImage(", baseline)
    assert "settleLostGeneration(() => isMounted.current, probeBaseline)" in src

    flags = _read("lib/gallery-flags.ts")
    assert "export async function hasUnknownRecord" in flags
    # An unknown row is the proof, and only an unpinned one counts.
    assert "return !baseline.knownIds.has(record.id);" in flags
    # Inconclusive listings must refuse to claim proof, else a submission that
    # never landed reads as a finished image, which is the bug this guards.
    assert "if (!baseline.canJudgeUnpinned) return false;" in flags


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


def test_adopting_a_resident_model_reseeds_the_slot_and_batch_controls():
    """The controls in the store belong to the model that just LEFT, so adoption reseeds them.

    This test used to assert the opposite, through a `readoptingSameModel` option that
    suppressed the reseed on re-adoption. #8943 removed the option deliberately and said
    why: the adopt path rolls the outgoing model's config back into the store before it
    hydrates, so the slot and batch controls sitting there describe the model the tab
    just left. Suppressing the reseed left a resident model running 4 slots showing the
    outgoing count, and the next Apply saved that over it.

    So `slotsModelChanged` is `hydratingExistingModel` with nothing subtracted, which is
    how every other load param at this call site already treats a changed checkpoint or
    variant.

    Reseeding is only safe because the same flag gates the remembered lookup: it does
    not blank the control, it re-reads THIS model's own saved config through
    resolveResidentInitialConfig. That is what makes #8943 right rather than merely
    different, so it is asserted here too -- a future change that reseeds without
    re-reading would take the user's saved slot count away for real.
    """
    status = " ".join(_read("features/chat/lib/apply-inference-status-to-store.ts").split())
    assert "const slotsModelChanged = hydratingExistingModel;" in status
    # Nothing may reintroduce a same-model exemption without this test being rewritten.
    assert "readoptingSameModel" not in status
    assert "...(seedLoadParams && slotsModelChanged && { nParallel: null })," in status
    # Never a slot-count proxy for "same model".
    assert "prevState.loadedNParallel === (status.requested_parallel_slots" not in status
    # The baseline seed stays ungated, or a rollback after a tab reload restores
    # the model at the server default slots.
    assert "loadedNParallel: status.requested_parallel_slots," in status
    # The batch pair is told the same thing, from the same local, so the two cannot drift.
    assert "modelChanged: slotsModelChanged," in status
    # The reseed re-reads this model's remembered config rather than blanking. Slot
    # conditions are GGUF's alone; without slot fields it goes by whether the poll is
    # hydrating a model already on screen.
    assert (
        "const remembered = (status.is_gguf "
        "? slotsUnseeded || batchesUnseeded || slotsModelChanged "
        ": hydratingExistingModel) "
        "? resolveResidentInitialConfig(checkpointId, status.gguf_variant ?? null)" in status
    ), "the model-change reseed must feed the remembered lookup, or it discards the saved config"

    # And the rollback that makes the reseed necessary is still ordered before the
    # hydration it protects, in the adopt path.
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    adopt = runtime[runtime.index("const confirmedStatus = await getInferenceStatus()") :]
    adopt = adopt[: adopt.index("void refreshContextUsage(")]
    assert (
        adopt.index("restorePreviousConfig();")
        < adopt.index("applyActiveModelStatusToStore(confirmedStatus, {")
    ), "the rollback must precede the hydration, or the staged snapshot wins over the resident status"


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
        "(status.is_gguf ? slotsUnseeded || batchesUnseeded || slotsModelChanged "
        ": hydratingExistingModel)" in status
    ), "storage is read on a fresh store or a model change, never on a steady poll"
    assert (
        "const rememberedNParallel = status.is_gguf && remembered?.remembered" in status
    ), "slots are a llama.cpp knob; reading MLX's record must not seed one"
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
    # echo the adoption gate compares against carries the saved count at all. Driven through the
    # real ladder rather than grepped, for the reason in
    # test_a_standalone_gguf_has_one_settings_identity_everywhere.
    candidates = _override_lookup_candidates("/models/m.gguf", "org/repo", "Q8_0")
    assert candidates[:2] == [
        "/models/m.gguf:Q8_0",
        "org/repo:Q8_0",
    ], "the variant-qualified keys come first, load path before advertised alias"
    assert "org/repo" in candidates, "the alias is still read, as the cached-alias path relies on"


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
    # Ordering, not adjacency: the concatenated form required the two statements to be neighbours,
    # so #8702 broke it by inserting a line between them without changing the contract.
    # Scoped to selectWithConfig, because the hub auto-load path takes the same snapshot above the
    # only applyModelLoadConfigToRuntime call and would satisfy a whole-file comparison on its own.
    picker = " ".join(_read("features/chat/chat-page.tsx").split())
    handoff = _brace_matched_body(picker, "const selectWithConfig = async (")
    assert (
        "const previousConfig = currentRuntimePerModelConfig({ includeMaxSeqLength: true, });"
        in handoff
    )
    assert handoff.index("const previousConfig = currentRuntimePerModelConfig(") < handoff.index(
        "applyModelLoadConfigToRuntime("
    ), "the snapshot must be taken before the target's config is applied"
    rollback = runtime.split("const rollbackSpeculativeType", 1)[1]
    assert "nParallel: previousNParallel," in rollback
    # Baseline and reload payload keep the resolved count, or the rollback
    # recreates the previous model at a different slot count.
    assert "loadedNParallel: stateBeforeUnload.loadedNParallel ?? null," in rollback
    assert "n_parallel: stateBeforeUnload.loadedNParallel," in runtime


def test_batch_sizes_setting_wired_end_to_end():
    """The per-load Batch / Micro-batch knobs (llama-server --batch-size / --ubatch-size)
    follow the same hops as Parallel Slots; a lost hop silently reverts the model to the
    llama.cpp defaults (2048 / 512)."""
    config = _read("features/model-picker/model-config/per-model-config.ts")
    assert '"nBatch",' in config and '"nUbatch",' in config
    assert "N_BATCH_MAX, Math.round(partial.nBatch)" in config
    assert "N_BATCH_MAX, Math.round(partial.nUbatch)" in config
    assert "config.nBatch == null &&" in config
    assert "config.nUbatch == null &&" in config
    page = _read("features/model-picker/components/model-config-page.tsx")
    assert "Batch Size" in page and "Micro-batch Size" in page
    assert "config.nBatch != null ||" in page
    assert 'aria-label="Prompt batch size"' in page
    assert 'aria-label="Prompt micro-batch size"' in page
    api_types = _read("features/chat/types/api.ts")
    assert "n_batch?: number | null;" in api_types
    assert "n_ubatch?: number | null;" in api_types
    runtime = " ".join(_read("features/chat/hooks/use-chat-model-runtime.ts").split())
    assert "pendingLoadConfig?.nBatch" in runtime
    # omitted when blank: an explicit null reads as set and strips inherited -b / -ub
    assert "...(isGguf && loadNBatch != null ? { n_batch: loadNBatch } : {})," in runtime
    assert "...(isGguf && loadNUbatch != null ? { n_ubatch: loadNUbatch } : {})," in runtime
    assert "...(validateNBatch != null ? { n_batch: validateNBatch } : {})," in runtime
    assert "...(validateNUbatch != null ? { n_ubatch: validateNUbatch } : {})," in runtime
    assert "n_batch: isGguf ? loadNBatch : null," not in runtime
    runtime = _read("features/chat/hooks/use-chat-model-runtime.ts")
    assert "loadNBatch = pendingLoadConfig?.nBatch ?? null;" in runtime
    assert "loadNUbatch = pendingLoadConfig?.nUbatch ?? null;" in runtime
    # rollback re-sends a baseline only when one was asked, for the same reason
    assert "{ n_batch: stateBeforeUnload.loadedNBatch }" in runtime
    assert "{ n_ubatch: stateBeforeUnload.loadedNUbatch }" in runtime
    assert "n_batch: stateBeforeUnload.loadedNBatch," not in runtime
    chat_api = " ".join(_read("features/chat/api/chat-api.ts").split())
    assert "...(payload.n_batch != null ? { n_batch: payload.n_batch } : {})," in chat_api
    assert "...(payload.n_ubatch != null ? { n_ubatch: payload.n_ubatch } : {})," in chat_api
    composer = " ".join(_read("features/chat/shared-composer.tsx").split())
    assert (
        composer.count("...(ownConfig.nBatch != null ? { n_batch: ownConfig.nBatch } : {}),") == 2
    )
    assert (
        composer.count("...(ownConfig.nUbatch != null ? { n_ubatch: ownConfig.nUbatch } : {}),")
        == 2
    )
    adapter = " ".join(_read("features/chat/api/chat-adapter.ts").split())
    assert adapter.count("...(config.nBatch != null ? { n_batch: config.nBatch } : {}),") == 2
    assert adapter.count("...(config.nUbatch != null ? { n_ubatch: config.nUbatch } : {}),") == 2
    assert "loadedNBatch: committedNBatch," in adapter
    assert "loadedNUbatch: committedNUbatch," in adapter
    status = " ".join(_read("features/chat/lib/apply-inference-status-to-store.ts").split())
    # both pairs hydrate through the one shared rule
    assert "incoming: status.requested_n_batch," in status
    assert "incoming: status.requested_n_ubatch," in status
    assert '...("loaded" in nBatchSeed && { loadedNBatch: nBatchSeed.loaded ?? null }),' in status
    assert '...("value" in nBatchSeed && { nBatch: nBatchSeed.value ?? null }),' in status
    assert (
        '...("loaded" in nUbatchSeed && { loadedNUbatch: nUbatchSeed.loaded ?? null, }),' in status
    )
    assert '...("value" in nUbatchSeed && { nUbatch: nUbatchSeed.value ?? null }),' in status
    signature = _read("features/model-picker/model-config/config-signature.ts")
    assert 'config.nBatch ?? "",' in signature
    assert 'config.nUbatch ?? "",' in signature


def test_batch_sizes_reach_an_api_load_through_the_server_mirror():
    """Same server-mirror hop as the slots: an OpenAI auto-switch load happens without a
    browser, so the stored override is its only way to carry the batch sizes."""
    api = " ".join(_read("features/model-picker/api/model-overrides.ts").split())
    assert "n_batch?: number;" in api
    assert "n_ubatch?: number;" in api
    assert "if (config.nBatch && config.nBatch > 0) { payload.n_batch = config.nBatch; }" in api
    assert "if (config.nUbatch && config.nUbatch > 0) { payload.n_ubatch = config.nUbatch; }" in api
    monitor = " ".join(_read("features/api-monitor/components/saved-model-settings.tsx").split())
    assert "if (override.n_batch) {" in monitor
    assert "if (override.n_ubatch) {" in monitor

    route = _read_backend("routes/settings.py")
    assert "n_batch: Optional[int] = Field(" in route
    assert "n_batch = payload.n_batch," in route
    assert "n_ubatch = payload.n_ubatch," in route
    store = _read_backend("utils/openai_auto_switch_settings.py")
    assert '("n_batch", "n_ubatch")' in store
    gguf_block = store.split("    if is_gguf:", 1)[1]
    assert 'kwargs["n_batch"] = override["n_batch"]' in gguf_block
    assert 'kwargs["n_ubatch"] = override["n_ubatch"]' in gguf_block
    # A stored pass-through -b / -ub must not outrank the first-class field.
    assert 'strip_batch = "n_batch" in kwargs,' in store
    assert 'strip_ubatch = "n_ubatch" in kwargs,' in store


def test_hydration_clears_the_batch_baselines_for_a_batchless_model():
    """Like the slot baseline: a model whose load never sent the batch sizes must not
    inherit the previous GGUF's values into the rollback baseline. The null-echo,
    clean-control-follow and pending-edit rules live in resolveBatchSizeSeed and are
    behavior-tested in resolve-batch-size-seed.test.ts; here only the wiring is pinned."""
    seed = " ".join(_read("features/chat/lib/resolve-batch-size-seed.ts").split())
    # a non-gguf clears; an absent field on a gguf is an older backend saying nothing,
    # though a swap still has to drop the pair staged against the model that left.
    # The baseline goes with the control, or a later failed swap rolls back with a
    # batch the backend that reported nothing never ran.
    assert "const effective = isGguf ? incoming : null;" in seed
    assert "if (effective === undefined) {" in seed
    assert "return modelChanged ? { value: null, loaded: null } : {};" in seed
    # a blank control is clean too, or an external load to an explicit size reads as dirty
    assert "const controlIsClean = modelChanged || previous.value === previous.loaded;" in seed
    assert "...(controlIsClean ? { value: effective } : {})," in seed
    # a swap must reach the adopt path, not be swallowed by the steady-echo short-circuit
    assert "if (previous.loaded === effective && !modelChanged) { return {}; }" in seed
    src = _read("features/chat/lib/apply-inference-status-to-store.ts")
    status = " ".join(src.split())
    assert "isGguf: status.is_gguf ?? true," in status
    # A swap under this tab resets the controls too, but through the seed, not a blanket
    # null after it: the batch echo is the REQUESTED size, so clearing it here would also
    # discard the value just adopted from the new model and revert it on the next Reload.
    # EVERY seed is told the same thing from the same local, so none of them can drift
    # apart. Checked per call site rather than as a count: the llama-server tuning knobs
    # deliberately reuse this seed, so a hardcoded number fails the next time the group
    # grows while saying nothing about drift, and a whole-file count of `modelChanged`
    # would both break on an unrelated one elsewhere and let an unrelated one stand in
    # for a seed that omitted the property.
    # Comments stripped first: this file discusses `resolveBatchSizeSeed (modelChanged)`
    # in prose, and a scan that counted that sentence as a call site would fail on a
    # wording change.
    seed_args = _call_arguments(_code_only(src), "resolveBatchSizeSeed")
    assert len(seed_args) >= 2, "the batch pair alone should be two seeds"
    for args in seed_args:
        assert (
            "modelChanged: slotsModelChanged," in args
        ), f"a resolveBatchSizeSeed call is not told modelChanged from slotsModelChanged: {args}"
    assert "{ nBatch: null, nUbatch: null }" not in status
    # The remembered override is re-adopted only when the echo proves it.
    assert (
        "rememberedNBatch != null && rememberedNBatch === "
        "status.requested_n_batch && { nBatch: rememberedNBatch, }" in status
    )
    assert (
        "rememberedNUbatch != null && rememberedNUbatch === "
        "status.requested_n_ubatch && { nUbatch: rememberedNUbatch, }" in status
    )


def test_vulkan_inference_devices_are_the_pickable_set():
    """GGUF loads run through llama-server, so on a Vulkan build the picker must offer the
    inference inventory (ggml ordinals, the space `--device Vulkan<i>` pins) rather than
    the torch view, which can miss cards llama-server drives."""
    src = " ".join(_read("hooks/use-gpu-info.ts").split())
    # The Vulkan inventory is authoritative even while its probe is temporarily
    # empty. Falling through would expose physical CUDA/ROCm IDs in an ordinal
    # picker and make DiffusionGemma offer a selection the route rejects.
    # Scoped to the GGUF picker: an image or video load runs on torch, not llama-server, so it
    # reads the torch inventory even here. A Vulkan chat build says nothing about the CUDA / ROCm
    # devices a diffusion load can be pinned to.
    vulkan_gate = (
        "const inference = data?.inference_gpu; "
        'if (!forDiffusion && inference?.backend === "vulkan") {'
    )
    assert vulkan_gate in src
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


def test_adoption_takes_its_own_pin_before_moving_the_checkpoint():
    """Status polling skips its own pin clearing while an external provider is selected, so the
    adoption branch can adopt a resident the pin was never taken for and Apply would reload the
    old model. The branch has to write the pin itself.

    It used to clear the pin to null. #8943 replaced that with adopting THIS pick's pin by
    the rule a completed load writes it -- the load path, or null where that is just the id
    -- which drops a stale pin the same way and additionally keeps a pinned cached row
    loadable. The ordering requirement is unchanged and is what this still pins.
    """
    src = _read("features/chat/hooks/use-chat-model-runtime.ts")
    branch = src[src.index("const confirmedStatus = await getInferenceStatus()") :]
    branch = branch[: branch.index("void refreshContextUsage(")]
    assert "activeLoadId: loadPath === modelId ? null : loadPath," in branch
    # Landing before the checkpoint moves, so nothing reads the pair half updated.
    assert branch.index("activeLoadId: loadPath === modelId ? null : loadPath,") < branch.index(
        ".setCheckpoint(modelId, confirmedStatus.gguf_variant)"
    ), "the pin must be written before the checkpoint is adopted"


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
    # computeStats lives in stats.ts so it runs under `node --test`; the hook re-exports it.
    src = " ".join(_read("features/api-monitor/stats.ts").split())
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
    assert "Saved settings apply everywhere Unsloth loads this model." in src


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
    """loadPerModelConfig refuses to apply a record written by a newer Unsloth and eviction
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

    # Read out of the route's parse tree, not its text. The rule is which names the route
    # declares, forwards and excludes; a set literal reflowed one name per line by the
    # formatter, or a name added beside them, is not a change to it.
    route = "routes/settings.py"
    handler = "update_openai_auto_switch_override"
    assert _annotated_field(route, "ModelOverridePayload", "fill_absent_fields") == (
        "bool",
        False,
    ), "the rule this mirrors"
    assert "fill_absent_fields" in _forwarded_keywords(route, handler, "payload")
    # A write mode must not leak into saved fields, or model_id-only removal stops working.
    assert {"remove", "fill_absent_fields"} <= _model_dump_exclusions(route, handler)

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
    assert "settingsTargetIsResident ? activeLoadedContextLength : null" in hub
    # The loadable identifier, as every other status reader records it -- except for a
    # speech model, which chat cannot adopt at all. speechOnly rides beside the null so
    # the helper can tell it from the empty slot the idle-unload rule is about.
    assert (
        "checkpointId: isSpeechOnlyStatus(status) ? null "
        ": resolveInferenceCheckpointId(status), "
        "speechOnly: isSpeechOnlyStatus(status)," in hub
    )
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
    # gguf_variant_key, not the hub's extract_quant_label: for a standalone file there is no
    # directory to qualify, so the key IS the quant token -- and it keeps the bpw modifier, which
    # is what makes it agree with the loader's own _extract_quant_label (the equality the sibling
    # test below depends on). The hub label drops that modifier.
    assert "gguf_variant_key(gguf_files[0].name)" in common
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

    # The precedence that makes the bare path win, asserted on the real function rather than on
    # inference.py's text: #8702 moved this ladder to utils/openai_auto_switch_settings.py
    # unchanged, and the grep that used to live here went red for a pure refactor.
    candidates = _override_lookup_candidates("/models/m-Q4_K_M.gguf", "org/repo", None)
    assert candidates == [
        "/models/m-Q4_K_M.gguf",
        "/models/m-Q4_K_M.gguf:Q4_K_M",
        "org/repo",
    ], "the bare path must be read before the filename label, and both before the alias"


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
    scan = variants.split("if is_local_path(repo_id) or probe.exists():", 1)
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
    # An Audio load taking the single slot is not an idle eviction: nothing stashes the
    # chat model, so the exemption must not swallow that case.
    assert "if (state.idleUnloadArmed && !status.speechOnly) { return false; }" in adopt
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
    # Whole elements, like test_the_primary_action_keeps_its_four_labels. The old single-line regex
    # pinned the handler body exactly, so #8702 broke it by reflowing that call across lines while
    # the button itself stayed untouched.
    reset = any(
        "DEFAULT_PER_MODEL_CONFIG" in el.group(0)
        and ">\n          Reset\n        <" in el.group(0)
        or ("DEFAULT_PER_MODEL_CONFIG" in el.group(0) and re.search(r">\s*Reset\s*<", el.group(0)))
        for el in re.finditer(r"<Button\b.*?</Button>", page, re.S)
    )
    assert reset, "the Reset button's JSX is gone or no longer named Reset"


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


def test_the_micro_batch_advisory_compares_against_the_emitted_batch():
    """The loader raises --batch-size to max(slots, 2), so the micro-batch advisory has to
    judge the RAISED value. Against the typed one, batch 4 / slots 8 / ubatch 8 rendered
    two advisories that contradict each other: "will raise it to 8" beside "llama.cpp will
    run at 4", when the launch runs at 8. Both the predicate and the number shown come
    from the emitted batch now."""
    src = _read("features/model-picker/components/model-config-page.tsx")
    page = " ".join(src.split())
    assert "const batchFloor = Math.max(2, config.nParallel ?? 2);" in page
    assert "Math.max(config.nBatch, batchFloor)" in page
    assert "config.nUbatch > effectiveBatch" in page
    # the rendered number too, or the text still names a batch nothing runs at
    assert 'llama.cpp will run at{" "} {effectiveBatch}.' in page
    # and the floor must be declared before the comparison that uses it
    assert page.index("const batchFloor =") < page.index("const effectiveBatch =")


def test_a_blank_batch_still_caps_the_micro_batch_at_the_llama_default():
    """Blank does not mean unbounded: no flag is emitted, so llama.cpp runs its own 2048
    and caps the micro-batch against THAT. Treating blank as null suppressed the advisory
    entirely, so a micro-batch of 4096 with the batch left blank looked usable while the
    server ran 2048. Shared constant rather than a literal, since the backend already
    names the same number."""
    src = _read("features/model-picker/components/model-config-page.tsx")
    page = " ".join(src.split())
    assert "N_BATCH_LLAMA_DEFAULT," in page  # imported, not inlined
    assert ": N_BATCH_LLAMA_DEFAULT;" in page
    # effectiveBatch is now never null, so the predicate must not re-check it for null
    assert "config.nUbatch != null && config.nUbatch > effectiveBatch" in page
    cfg = _read("features/model-picker/model-config/per-model-config.ts")
    assert "export const N_BATCH_LLAMA_DEFAULT = 2048;" in cfg


def test_autoload_sees_every_on_device_inventory_and_fails_closed():
    """#7374: Send downloaded a model over a loadable local one. The cascade read
    only the two managed-cache lists, both wrapped in `.catch(() => [])`, so a
    flaky request also read as "device is empty"."""
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
    # when the backend sends no can_chat, so the picker shows such a row while
    # `=== true` skipped it and fell through to downloading the default.
    assert "row.capabilities?.can_chat !== false" in policy
    assert "row.partial !== true" in policy
    # An adapter resolves its base model, a Hub fetch when that base is uncached;
    # a scan-folder checkpoint is a pickle with no Hub security scan. Stated as
    # an allowlist, which subsumes both: excluding them by name was only as good
    # as the classification, and the backend sends "unknown" when it cannot tell.
    # test_a_local_row_the_picker_would_hide_is_never_auto_loaded covers the
    # behaviour.
    assert '(isGgufLocalRow(row) || row.model_format === "safetensors")' in policy
    assert "isHiddenModelId(row.model_id, row.id, row.path)" in policy

    sources = src.split("const AUTO_LOAD_LOCAL_SOURCES", 1)[1].split("]", 1)[0]
    assert '"models_dir"' in sources
    assert '"lmstudio"' in sources
    assert '"custom"' in sources
    # hf_cache rows are the cached lists' job; ollama links are not loadable.
    assert '"hf_cache"' not in sources
    assert '"ollama"' not in sources


def test_default_model_download_is_visible_and_cancellable():
    """On a genuinely empty device Unsloth still fetches a model, but as a managed
    download with progress and a Cancel, never an inline pull inside /load."""
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
    models found") and named a raw repo/quant."""
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


def test_autoload_skips_rows_chat_cannot_answer():
    """The backend tags a row with a task only for the models that own another page,
    and the picker routes those away on click. A background load has no routing step.
    The audio tasks are here because the chat route answers a turn on a speech model by
    synthesizing the prompt rather than refusing it, so nothing downstream catches it."""
    src = _read("features/chat/api/chat-adapter.ts")
    tasks = src.split("const NON_CHAT_TASKS", 1)[1].split("]", 1)[0]
    assert '"text-to-image"' in tasks
    assert '"text-to-video"' in tasks
    assert '"image-diffusion-unsupported"' in tasks
    assert '"text-to-speech"' in tasks
    assert '"text-to-audio"' in tasks
    assert '"audio-to-audio"' in tasks
    assert '"automatic-speech-recognition"' in tasks
    policy = src.split("function isAutoLoadableLocalRow", 1)[1].split("\n}", 1)[0]
    assert 'NON_CHAT_TASKS.has(row.task ?? "")' in policy


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
    """cancel() no-ops on a key with no job, so a click during requestStart's
    preflights is replayed once the job exists, and exactly once: cancelling
    patches the job and wakes the subscription."""
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
    format-only rule marked chat-capable, and it is small enough to be tried
    first."""
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
    """A repo holding both GGUF and safetensors yields a row in each cached list,
    but the backend resolves one target to one model, so keeping both spends a
    second attempt on the same files. Skipped in the cascade rather than dropped
    while ordering, because dropping the twin lost a loadable safetensors row
    whenever its GGUF twin resolved no quant."""
    src = _read("features/chat/api/chat-adapter.ts")
    key = src.split("function autoLoadSourceKey", 1)[1].split("\n}", 1)[0]
    assert "return normalizeTarget(source.loadId);" in key
    assert "source.kind" not in key
    # Ordering keeps every row; the sort alone decides which twin is reached first.
    order = src.split("function orderAutoLoadSources", 1)[1].split("\n}\n", 1)[0]
    assert "[...sources].sort(" in order
    assert "filter(" not in order
    # The skip is keyed on a candidate having been resolved, not on merely visiting.
    body = src.split("const candidateResolvedFor = new Set<string>();", 1)[1]
    body = body.split("\n    // Cap also gates", 1)[0]
    assert body.index("if (candidateResolvedFor.has(sourceKey)) continue;") < body.index(
        "candidateResolvedFor.add(sourceKey);"
    )
    assert "if (!candidate) break;\n          candidateResolvedFor.add(sourceKey);" in body


def test_variant_scans_take_the_run_signal():
    src = _read("features/chat/api/chat-adapter.ts")
    build = src.split("function buildAutoLoadSources", 1)[1]
    build = build.split("function isRememberedSource", 1)[0]
    assert build.count("signal,") == 2
    assert "options?.abortSignal,\n      )," in src


def test_cached_rows_classify_chat_capability_too():
    """The same encoder gate the scan-folder rows get; cached rows built their
    capabilities from file format alone. Both halves are pinned separately, since
    the gate is now nested rather than one `and`, and the call must read the
    snapshot the load resolves to rather than any other revision."""
    src = _read_backend("hub/services/models/cache_inventory.py")
    assert "_local_transformers_can_chat" in src
    fields = src.split("def _cache_inventory_fields", 1)[1].split("\ndef ", 1)[0]
    assert "can_chat_override" in fields
    assert 'model_format in {"safetensors", "checkpoint"}' in fields
    assert "classify_snapshot is not None" in fields
    assert "_local_transformers_can_chat(classify_snapshot)" in fields


def test_cached_codec_evidence_enriches_an_existing_hub_result():
    """Search results outrank cached metadata, but they must not erase a decoder
    discovered from the local tokenizer files for the same repo."""
    src = _read("features/model-picker/components/model-selector/pickers.tsx")
    evidence = src.split("const hubEvidenceById = useMemo", 1)[1]
    evidence = evidence.split("const capsById = useMemo", 1)[0]
    assert evidence.count("const existing = map.get(c.repo_id);") == 2
    assert evidence.count("audioType: existing.audioType ?? c.audio_type") == 2


def test_every_load_target_comparison_uses_the_same_case_rules():
    """Source dedupe, remembered matching, and tried-candidate keys all compare
    load targets, so they must agree on POSIX case."""
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


def test_cached_rows_get_the_same_non_chat_gate_as_local_rows():
    """Cached diffusion and speech repos carry their task on the row and report can_chat
    true on file format alone. Both inventories, or one of them still offers the row."""
    src = _read("features/chat/api/chat-adapter.ts")
    cached = src.split("function isChattableCachedRepo", 1)[1].split("\n}\n", 1)[0]
    assert 'NON_CHAT_TASKS.has(repo.task ?? "")' in cached
    local = src.split("function isAutoLoadableLocalRow", 1)[1].split("\n}", 1)[0]
    assert 'NON_CHAT_TASKS.has(row.task ?? "")' in local
    # A chat GGUF is tagged text-generation, so the gate is a list, not "has a task".
    tasks = src.split("const NON_CHAT_TASKS", 1)[1].split("]", 1)[0]
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
    """_scan_cached_models passes only `identity`, so snapshot_path alone
    classified nothing and the gate was a no-op in production."""
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
    """The store's initial chatOnly is a browser guess: a Mac browser on a remote
    Linux Unsloth would hide every local safetensors model."""
    src = _read("features/chat/api/chat-adapter.ts")
    gate = src.split("function runsOnThisPlatform", 1)[1].split("\n}", 1)[0]
    assert "if (!platform.fetched || !platform.isChatOnly()) return true;" in gate


def test_the_default_is_preflighted_before_the_managed_download():
    """A refusal from the training or placement guard must not cost gigabytes
    first."""
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
    """A ViTModel name has no task suffix, so only the model type identifies it."""
    src = _read_backend("hub/services/models/common.py")
    encoders = src.split("_ENCODER_ONLY_MODEL_TYPES = frozenset(", 1)[1]
    encoders = encoders.split(")", 1)[0]
    for model_type in ('"vit"', '"dinov2"', '"swin"', '"wav2vec2"', '"resnet"'):
        assert model_type in encoders, model_type


def test_the_gguf_footprint_is_resolved_per_dependency_group_not_per_repo():
    """The companion set a diffusion GGUF needs (text encoder, VAE, tokenizer,
    configs) is not repository-wide: `detect_family_for_pick` falls back to
    `repo_id/filename`, so one neutral repo can hold GGUFs of two families with
    different base repos, and `sd_cpp_text_encoders_for` hands FLUX.2-klein-9B a
    different text encoder than klein-4B in the same repo. Sampling ONE
    representative and pasting its companionBytes onto every row therefore
    advertised a GB-wrong "Full required size" on the rows it did not sample."""
    src = _read("features/model-picker/components/model-selector/pickers.tsx")
    # The old shape: one scalar for the whole listing.
    assert "const [companionBytes, setCompanionBytes] = useState" not in src
    assert (
        "const [companionBytesByKey, setCompanionBytesByKey] = useState<\n    Map<string, number>\n  >"
        in src
    )
    # Representatives are derived per key, not once for the listing.
    group = src.split("const footprintVariants = useMemo(", 1)[1]
    group = group.split("}, [displayVariants, recommendedQuantForVariant]);", 1)[0]
    assert "new Map<string, GgufVariantDetail>()" in group
    assert 'const key = variant.dependency_key ?? "";' in group
    # The recommended quant still wins, and only inside its own group. Main
    # tracked the source through effectiveRecommended becoming a SET and
    # asserted membership in it; but that set is flattened across groups, so it
    # blesses the other group's pick when two families in one repo share quant
    # names.
    #
    # Group-scoping it is right, but this block buckets by the BACKEND's
    # dependency_key ("flux.2-klein:<digest>") while the recommendation map is
    # keyed by PRESENTATION group ("quantizations", "text-frames",
    # "reference-media"). Joining those two key spaces makes every lookup
    # undefined and the recommended-quant branch dead, leaving whichever row
    # tier ordering put first to speak for the group. So ask through the
    # variant, and pin that neither the flattened set nor the crossed-key
    # lookup is what stands here.
    assert "const recommended = recommendedQuantForVariant.get(variant);" in group
    assert "variant.quant === recommended" in group
    assert "current.quant !== recommended" in group
    assert "recommended !== undefined" in group
    assert "effectiveRecommended.has(" not in group
    assert "effectiveRecommendedByGroup.get(key)" not in group

    # And the map it asks is built from the presentation groups, keyed by the
    # variant objects those groups hold.
    builder = src.split("const recommendedQuantForVariant = useMemo(", 1)[1]
    builder = builder.split("}, [variantGroups, effectiveRecommendedByGroup]);", 1)[0]
    assert "new Map<GgufVariantDetail, string>()" in builder
    assert "effectiveRecommendedByGroup.get(group.key)" in builder
    assert "byVariant.set(variant, recommended)" in builder


def test_every_footprint_group_gets_its_own_resolve_call():
    """One request per distinct key. The ordinary repo has exactly one key, so the
    common case stays exactly one request, which is what the representative scheme
    exists to protect."""
    src = _read("features/model-picker/components/model-selector/pickers.tsx")
    effect = src.split("const [companionBytesByKey, setCompanionBytesByKey]", 1)[1]
    effect = effect.split("const variantOptionKeys = useMemo(", 1)[0]
    assert "for (const footprintVariant of footprintVariants) {" in effect
    assert "resolveDownloadFootprint(repoId, {" in effect
    # Each resolution writes only its own key, into a fresh Map: mutating the state
    # Map in place would leave React on the old identity and drop earlier groups.
    assert "next.set(dependencyKey, companion);" in effect
    assert "const next = new Map(previous);" in effect
    # Cleared per listing, so a reopened repo never shows the previous repo's totals.
    assert "setCompanionBytesByKey(new Map());" in effect


def test_the_footprint_asks_the_listing_whether_the_checkpoint_is_on_disk():
    """Whether the checkpoint sits inside `required_bytes` is a question about the
    disk, and the prefix regex cannot answer it: the backend resolves identifiers
    existence-first, so a marker-less relative directory like "models/my-image-model"
    is a local model with no path marker to match. Gating the subtraction on the
    regex alone subtracted a checkpoint the plan had never counted, driving the
    figure to zero and hiding a multi-GB companion set behind the checkpoint size.
    The listing already reports the backend's own verdict as `resolved_locally`."""
    src = _read("features/model-picker/components/model-selector/pickers.tsx")
    # Surfaced by the normalizer, so no caller re-implements the field name.
    assert "resolved_locally?: unknown;" in src
    assert "resolvedLocally: res?.resolved_locally === true," in src
    assert "setResolvedLocally(normalized.resolvedLocally);" in src
    # Reset per listing: the previous row's locality must not decide this row's total.
    assert "setResolvedLocally(false);" in src
    assert "const checkpointIsLocal = isLocalPath || resolvedLocally;" in src
    # The subtraction reads the combined verdict, never the regex on its own.
    effect = src.split("const [companionBytesByKey, setCompanionBytesByKey]", 1)[1]
    effect = effect.split("const variantOptionKeys = useMemo(", 1)[0]
    assert "const checkpoint = checkpointIsLocal" in effect
    assert "const checkpoint = isLocalPath" not in effect


def test_a_refused_load_after_staging_rolls_the_pick_back():
    """Staging reports the pick STARTED as soon as the download is queued, so the caller's own
    `if (!started) revert` has already been skipped. The load only runs minutes later and can
    still be refused: a training run or another load can claim the backend while the download
    is going. Nothing polls for a staged pick, so the poll's rollback never runs either, and the
    selector would keep advertising a quant that was never loaded.

    BOTH deferred paths have to roll back. onReady hands off to the `active` effect when the
    page is off-tab, so leaving the tab during the download otherwise walks straight back into
    the same bug."""
    for page in ("features/images/images-page.tsx", "features/video/video-page.tsx"):
        src = _read(page)
        helper = src.split("const runStagedLoad = useCallback(", 1)[1].split("[revertPick],", 1)[0]
        # Fire-and-forget is precisely the defect: the boolean has to be observed.
        assert ".then((started) => {" in helper, page
        assert "if (started) return;" in helper, page
        # Same identity guard onCancelled uses: a newer pick owns the label from the moment it
        # is made, so a stale completion must not revert it.
        assert "const owned = stagedQuantRevert.current;" in helper, page
        assert "quantRevert.current === owned" in helper, page
        assert "revertPick(quantRevert.current);" in helper, page
        # One implementation, reached from both deferred paths, so neither can drift.
        assert src.count("if (pending) runStagedLoad(pending);") == 2, page
        # Exactly one direct call left, the one inside the helper itself.
        assert len(re.findall(r"void handleLoadRef\s*\.current\(\s*pending\.", src)) == 1, page


def test_each_quant_row_reads_its_own_dependency_key():
    """A row must look up its own group, and keep the plain formatBytes fallback
    when that group has no answer yet (or the backend sends no key at all)."""
    src = _read("features/model-picker/components/model-selector/pickers.tsx")
    row = src.split("{displayVariants.map((v) => {", 1)[1]
    row = row.split("</TooltipContent>", 1)[0]
    assert 'companionBytesByKey.get(v.dependency_key ?? "") ?? null' in row
    # The fallback and the tooltip gate still hang off this row's own value.
    assert (
        "companionBytes === null ? (\n                  <SizeText value={formatBytes(v.size_bytes)} />"
        in row
    )
    assert "companionBytes={companionBytes}" in row


def test_the_dependency_key_survives_the_variant_validator():
    """isValidGgufVariant filters the listing, so a field it rejects never reaches a
    row. An older backend sends none, which must stay valid."""
    src = _read("features/model-picker/components/model-selector/pickers.tsx")
    guard = src.split("function isValidGgufVariant(", 1)[1].split("\n}", 1)[0]
    assert "candidate.dependency_key === undefined" in guard
    assert "candidate.dependency_key === null" in guard
    assert 'typeof candidate.dependency_key === "string"' in guard
    types = _read("features/chat/types/api.ts")
    assert "dependency_key?: string | null;" in types


def test_the_backend_keys_the_footprint_on_family_and_text_encoders():
    """Both sources of variation have to be in the key. Folding in only the family
    would give klein-4B and klein-9B the same key inside one repo, which is the
    exact case the per-row lookup exists for."""
    src = _read_backend("hub/services/models/gguf_variants.py")
    helper = src.split("def _variant_dependency_key(", 1)[1].split("\ndef ", 1)[0]
    assert "detect_family_for_pick(repo_id, filename)" in helper
    assert "sd_cpp_text_encoders_for(" in helper
    assert "inner_dim = inner_dim" in helper
    # No family means no grouping information, not a fabricated key.
    assert "if fam is None:\n            return None" in helper
    # It runs once per row inside the listing, so it must never fail it.
    assert "except Exception as e:" in helper
    assert helper.rstrip().endswith("return None")
    # Every row of the listing carries one, local and remote branches alike.
    answer = src.split("async def get_gguf_variants_answer(", 1)[1]
    assert answer.count("dependency_key = _variant_dependency_key(") >= 4
    schema = _read_backend("hub/schemas/inventory.py")
    assert "dependency_key: Optional[str] = Field(" in schema


def test_the_diffusion_gpu_choices_are_memoized():
    """An unstable array here reaches the GGUF picker as a new footprint resolver on every render.

    ImagesPage feeds the choices into its load-advanced snapshot, that snapshot into
    resolveDownloadFootprint, and the picker's effect depends on the resolver: a fresh identity per
    render clears the companion sizes it had resolved and re-POSTs /images/download-plan for every
    variant, on every status poll, discarding the in-flight answers.
    """
    src = " ".join(_read("hooks/use-gpu-info.ts").split())
    choices = src[src.index("export function useDiffusionGpuChoices") :]
    choices = choices[: choices.index("/** Whether device discovery")]
    assert "return useMemo(() => {" in choices
    # Keyed on the device list, which useGpuDevices only replaces when the inventory changes.
    assert "}, [devices]);" in choices


def test_the_media_gpu_pick_survives_a_reload():
    """Every other Advanced select is reseeded from the loaded build; this one cannot be, because
    the status reports the device a pipeline is on and not which physical card. Without persistence
    a refresh silently reset the pick to Auto while the model stayed put, and the next Reapply
    moved it to the default GPU -- on a mixed box, potentially onto the card that cannot hold it."""
    for page, key in (
        ("features/images/images-page.tsx", "unsloth_image_gpu_choice"),
        ("features/video/video-page.tsx", "unsloth_video_gpu_choice"),
    ):
        src = " ".join(_read(page).split())
        assert f'usePersistedChoice( "{key}", "auto", )' in src, page
        # A stored id is only a hint: a card that has gone falls back to automatic.
        assert "gpuChoices.some((d) => String(d.index) === selectedGpu)" in src or (
            "controls.gpuChoices.some((d) => String(d.index) === controls.selectedGpu)" in src
        ), page
