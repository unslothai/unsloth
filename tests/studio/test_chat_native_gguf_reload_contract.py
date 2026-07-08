"""Static contracts for native GGUF settings reload identity."""

from __future__ import annotations

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

    helper_start = store_src.index("export function hasLoadedGgufSource")
    helper_body = store_src[
        helper_start : store_src.index("/** A local-disk model id", helper_start)
    ]
    assert "x.activeGgufVariant != null" in helper_body
    assert "hasUsableNativePathToken" in helper_body
    assert "x.activeNativePathTokenExpiresAtMs != null" in helper_body
    assert "isLocalModelPath(x.params.checkpoint)" in helper_body
    assert 'x.params.checkpoint.toLowerCase().endsWith(".gguf")' in helper_body
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
    assert "isLocalModelPath(checkpoint)" in page_src
    assert 'checkpoint.toLowerCase().endsWith(".gguf")' in page_src
    assert "isLocalModelPath(x.params.checkpoint)" in store_src


def test_checkpoint_changes_clear_loaded_gguf_metadata():
    store_src = _src(RUNTIME_STORE)

    assert "function clearedLoadedGgufMetadata()" in store_src
    clear_start = store_src.index("function clearedLoadedGgufMetadata()")
    clear_body = store_src[clear_start : store_src.index("/** An uncached HF", clear_start)]
    assert "activeNativePathToken: null" in clear_body
    assert "activeNativePathTokenExpiresAtMs: null" in clear_body
    assert "ggufContextLength: null" in clear_body
    assert "ggufMaxContextLength: null" in clear_body
    assert "ggufNativeContextLength: null" in clear_body

    set_params_start = store_src.index("setParams: (params)")
    set_params_body = store_src[
        set_params_start : store_src.index("setCustomPresets", set_params_start)
    ]
    assert "checkpointChanged" in set_params_body
    assert "...clearedLoadedGgufMetadata()" in set_params_body

    set_checkpoint_start = store_src.index("setCheckpoint: (modelId, ggufVariant)")
    set_checkpoint_body = store_src[
        set_checkpoint_start : store_src.index("setActiveThreadId", set_checkpoint_start)
    ]
    assert "loadedGgufSourceChanged" in set_checkpoint_body
    assert "clearedLoadedGgufMetadata()" in set_checkpoint_body
