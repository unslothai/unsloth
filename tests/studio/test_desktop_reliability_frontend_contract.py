# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static contracts for focused packaged-desktop reliability behavior."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "studio/frontend/src"
NATIVE_FILES = FRONTEND / "lib/native-files.ts"
CHAT_EXPORT = FRONTEND / "features/chat/utils/export-chat-history.ts"
CHAT_TAB = FRONTEND / "features/settings/tabs/chat-tab.tsx"
PROMPT_STORAGE = FRONTEND / "features/chat/prompt-storage/prompt-storage-dialog.tsx"

APP_SIDEBAR = FRONTEND / "components/app-sidebar.tsx"
THREAD = FRONTEND / "components/assistant-ui/thread.tsx"
THREAD_SIDEBAR = FRONTEND / "features/chat/thread-sidebar.tsx"
SHARED_COMPOSER = FRONTEND / "features/chat/shared-composer.tsx"
TITLEBAR = FRONTEND / "components/tauri/window-titlebar.tsx"
NATIVE_DIALOGS = REPO / "studio/src-tauri/src/native_file_dialogs.rs"


APP_PROVIDER = FRONTEND / "app/provider.tsx"


def test_file_actions_route_through_native_commands_only_in_tauri():
    helper = NATIVE_FILES.read_text(encoding = "utf-8")
    history = CHAT_EXPORT.read_text(encoding = "utf-8")
    chat_tab = CHAT_TAB.read_text(encoding = "utf-8")
    prompt_storage = PROMPT_STORAGE.read_text(encoding = "utf-8")

    assert 'invoke<string | null>("save_native_file", bytes, {' in helper
    assert '"x-unsloth-default-name"' in helper
    assert "Array.from(new Uint8Array" not in helper
    assert 'invoke<NativeChatImport | null>("pick_native_chat_import")' in helper
    assert "if (isTauri)" in helper
    assert 'document.createElement("a")' in helper
    assert "downloadFile(" in history
    assert "downloadFile(" in prompt_storage
    assert "pickNativeChatImport" in chat_tab
    assert "if (!isTauri)" in chat_tab
    # Browser builds retain the existing hidden-input route.
    assert 'type="file"' in chat_tab
    assert 'accept=".jsonl,.ndjson,.csv"' in chat_tab

    native_dialogs = NATIVE_DIALOGS.read_text(encoding = "utf-8")
    assert 'CHAT_IMPORT_EXTENSIONS: &[&str] = &["jsonl", "ndjson", "csv"]' in native_dialogs
    assert "InvokeBody::Raw" in native_dialogs


def test_chat_exports_await_native_saves_and_markdown_uses_shared_helper():
    app_sidebar = APP_SIDEBAR.read_text(encoding = "utf-8")
    prompt_storage = PROMPT_STORAGE.read_text(encoding = "utf-8")
    thread = THREAD.read_text(encoding = "utf-8")
    thread_sidebar = THREAD_SIDEBAR.read_text(encoding = "utf-8")
    shared_composer = SHARED_COMPOSER.read_text(encoding = "utf-8")

    assert "async function downloadBlob(" in prompt_storage
    assert "const handleExport = useCallback(async () =>" in prompt_storage
    assert prompt_storage.count("await export") >= 12
    assert "await Promise.all(" not in app_sidebar
    assert "for (const id of ids)" in app_sidebar
    assert prompt_storage.count("await downloadBlob(") >= 5
    assert "Promise.all(ids.map((id) => fn(id)))" not in thread_sidebar
    assert "for (const id of ids)" in thread_sidebar
    assert "Promise.all(exportThreadIds.map((id) => fn(id)))" not in shared_composer
    assert "for (const id of exportThreadIds)" in shared_composer
    assert "onExport={exportMessageMarkdown}" in thread
    assert '"text/markdown"' in thread
    assert "downloadFile(" in thread


def test_full_app_layout_uses_its_own_initialized_marker():
    source = APP_PROVIDER.read_text(encoding = "utf-8")

    assert 'invoke<boolean>("has_initialized_app_window_layout")' in source
    assert 'invoke("mark_app_window_layout_initialized")' in source
    assert "hasInitializedAppLayout && hasSavedState" in source


def test_expanded_titlebar_button_and_corner_match_sidebar_edge():
    source = TITLEBAR.read_text(encoding = "utf-8")

    assert 'pinned ? "gap-2 pl-3" : "justify-center"' in source
    assert "gap-2 px-3" not in source
    assert "const contentBorderLeft = pinned" in source
    assert ': "0px";' in source
    # The curved transition and sidebar-colored backing are expanded-only;
    # collapsed content is square and its divider spans the sidebar too.
    assert source.count("{showSidebarSurface && pinned && (") == 2
    assert (
        'className="pointer-events-none absolute top-full size-3 -translate-x-px bg-sidebar"'
        in source
    )
    assert (
        'className="pointer-events-none absolute top-full size-3 -translate-x-px rounded-tl-[12px] border-l border-t border-sidebar-border bg-background"'
        in source
    )
