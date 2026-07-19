"""Static contracts for native GGUF settings reload identity."""

from __future__ import annotations

import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CHAT_PAGE = REPO / "studio/frontend/src/features/chat/chat-page.tsx"
SETTINGS_SHEET = REPO / "studio/frontend/src/features/chat/chat-settings-sheet.tsx"
RUNTIME_HOOK = REPO / "studio/frontend/src/features/chat/hooks/use-chat-model-runtime.ts"
STAGED_PREP = REPO / "studio/frontend/src/features/chat/hooks/use-staged-model-preparation.ts"
RUNTIME_STORE = REPO / "studio/frontend/src/features/chat/stores/chat-runtime-store.ts"
NATIVE_CHIP = REPO / "studio/frontend/src/features/native-intents/components/native-model-chip.tsx"


def _src(path: Path) -> str:
    return path.read_text()


def _minified(src: str) -> str:
    return re.sub(r"\s+", "", src)


def _contract(src: str) -> str:
    return re.sub(r",(?=[})])", "", _minified(src))


_LOCAL_MODEL_PATH_RE = re.compile(r"^(\/|\.{1,2}[\\/]|~[\\/]|[A-Za-z]:[\\/]|\\\\)")


def _is_direct_local_gguf(model_id: str) -> bool:
    return bool(_LOCAL_MODEL_PATH_RE.match(model_id)) and model_id.lower().endswith(".gguf")


def _between(src: str, start_marker: str, end_marker: str) -> str:
    start = src.index(start_marker)
    return src[start : src.index(end_marker, start)]


def _function_body(src: str, signature: str) -> str:
    start = src.index(signature)
    brace_start = src.index("{", src.index(")", start))
    depth = 0
    for idx in range(brace_start, len(src)):
        char = src[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return src[start : idx + 1]
    raise AssertionError(f"Could not find end of {signature}")


def test_native_gguf_reload_tracks_token_expiry_and_reselects_after_expiry():
    store_src = _src(RUNTIME_STORE)
    page_src = _src(CHAT_PAGE)
    hook_src = _src(RUNTIME_HOOK)

    assert "activeNativePathTokenExpiresAtMs: number | null" in store_src
    assert "nativePathTokenExpiresAtMs?: number | null" in store_src
    assert "export function isNativePathTokenExpired" in store_src
    assert "export function hasUsableNativePathToken" in store_src

    assert "nativePathTokenExpiresAtMs: intent.path.expiresAtMs" in page_src
    assert "hasUsableNativePathToken({" in page_src
    assert "Pick or drop the local .gguf file again before applying settings." in page_src
    assert "activeNativePathToken: null" in page_src

    assert "isNativePathTokenExpired(nativePathTokenExpiresAtMs)" in hook_src
    assert "Local model selection expired" in hook_src
    assert "activeNativePathTokenExpiresAtMs: nativePathToken" in hook_src
    assert "previousActiveNativePathTokenExpiresAtMs" in hook_src


def test_loaded_gguf_classifier_is_shared_by_reload_and_settings_sheet():
    store_src = _src(RUNTIME_STORE)
    page_src = _src(CHAT_PAGE)
    sheet_src = _src(SETTINGS_SHEET)

    helper_body = _function_body(store_src, "export function hasLoadedGgufSource")
    helper_contract = _contract(helper_body)
    assert "x.activeGgufVariant != null" in helper_body
    assert "hasUsableNativePathToken" in helper_body
    assert "x.activeNativePathTokenExpiresAtMs != null" in helper_body
    assert (
        '(isLocalModelPath(x.params.checkpoint)&&x.params.checkpoint.toLowerCase().endsWith(".gguf"))'
        in helper_contract
    )
    assert "x.ggufContextLength != null" in helper_body

    assert "const isLoadedGguf = hasLoadedGgufSource(state)" in page_src
    assert "const isLoadedGguf = hasLoadedGgufSource({" in sheet_src


def test_native_token_expiry_flows_from_intent_through_staged_loads():
    page_src = _src(CHAT_PAGE)
    chip_src = _src(NATIVE_CHIP)
    staged_src = _src(STAGED_PREP)

    assert "nativePathTokenExpiresAtMs: selection.nativePathTokenExpiresAtMs" in page_src
    assert "nativePathTokenExpiresAtMs: intent.path.expiresAtMs" in page_src
    assert "nativePathTokenExpiresAtMs: number;" in chip_src
    assert "nativePathTokenExpiresAtMs: intent.path.expiresAtMs" in chip_src
    assert "isNativePathTokenExpired(nativePathTokenExpiresAtMs)" in staged_src


def test_native_display_basename_is_not_a_direct_gguf_path_on_reload():
    page_src = _src(CHAT_PAGE)
    store_src = _src(RUNTIME_STORE)

    assert "isLocalModelPath," in page_src
    assert "const isDirectGguf =" in page_src
    assert (
        'constisDirectGguf=isLocalModelPath(checkpoint)&&checkpoint.toLowerCase().endsWith(".gguf");'
        in _minified(page_src)
    )
    assert (
        '(isLocalModelPath(x.params.checkpoint)&&x.params.checkpoint.toLowerCase().endsWith(".gguf"))'
        in _contract(_function_body(store_src, "export function hasLoadedGgufSource"))
    )


def test_direct_gguf_classifier_truth_table_matches_local_path_contract():
    store_src = _src(RUNTIME_STORE)

    assert r"return/^(\/|\.{1,2}[\\/]|~[\\/]|[A-Za-z]:[\\/]|\\\\)/.test(id);" in _minified(
        store_src
    )

    for model_id in (
        "/models/model.gguf",
        "./model.gguf",
        "../models/model.gguf",
        "~/model.GGUF",
        r"C:\models\model.gguf",
        r"\\server\share\model.gguf",
    ):
        assert _is_direct_local_gguf(model_id)

    for model_id in (
        "model.gguf",
        "org/model.gguf",
        "hf://org/model.gguf",
        "/models/model.safetensors",
        r"C:model.gguf",
    ):
        assert not _is_direct_local_gguf(model_id)


def test_checkpoint_changes_clear_loaded_gguf_metadata():
    store_src = _src(RUNTIME_STORE)

    assert "function clearedLoadedGgufMetadata()" in store_src
    clear_body = _function_body(store_src, "function clearedLoadedGgufMetadata()")
    assert "activeNativePathToken: null" in clear_body
    assert "activeNativePathTokenExpiresAtMs: null" in clear_body
    assert "ggufContextLength: null" in clear_body
    assert "ggufMaxContextLength: null" in clear_body
    assert "ggufNativeContextLength: null" in clear_body
    assert "activeGgufVariant" not in clear_body

    set_params_body = _between(store_src, "setParams: (params)", "setCustomPresets")
    set_params_contract = _contract(set_params_body)
    assert "checkpointChanged" in set_params_body
    assert (
        "checkpointChanged?{contextUsage:null,activeGgufVariant:null,...clearedLoadedGgufMetadata()}:{}"
        in set_params_contract
    )

    set_checkpoint_body = _between(
        store_src, "setCheckpoint: (modelId, ggufVariant)", "setActiveThreadId"
    )
    set_checkpoint_contract = _contract(set_checkpoint_body)
    assert "loadedGgufSourceChanged" in set_checkpoint_body
    assert (
        "constloadedGgufSourceChanged=checkpointChanged||state.activeGgufVariant!==nextGgufVariant;"
        in set_checkpoint_contract
    )
    assert (
        "activeGgufVariant:nextGgufVariant,...(checkpointChanged?{contextUsage:null}:{}),...(loadedGgufSourceChanged?clearedLoadedGgufMetadata():{})"
        in set_checkpoint_contract
    )
