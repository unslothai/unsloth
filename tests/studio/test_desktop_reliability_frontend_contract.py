# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static contracts for focused packaged-desktop reliability behavior."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "studio/frontend/src"
NATIVE_FILES = FRONTEND / "lib/native-files.ts"
CHAT_EXPORT = FRONTEND / "features/chat/utils/export-chat-history.ts"
DATA_TAB = FRONTEND / "features/settings/tabs/data-tab.tsx"
PROMPT_STORAGE = FRONTEND / "features/chat/prompt-storage/prompt-storage-dialog.tsx"

APP_SIDEBAR = FRONTEND / "components/app-sidebar.tsx"
INDEX_CSS = FRONTEND / "index.css"
THREAD = FRONTEND / "components/assistant-ui/thread.tsx"
THREAD_SIDEBAR = FRONTEND / "features/chat/thread-sidebar.tsx"
SHARED_COMPOSER = FRONTEND / "features/chat/shared-composer.tsx"
TITLEBAR = FRONTEND / "components/tauri/window-titlebar.tsx"
NATIVE_DIALOGS = REPO / "studio/src-tauri/src/native_file_dialogs.rs"


APP_PROVIDER = FRONTEND / "app/provider.tsx"


def test_file_actions_route_through_native_commands_only_in_tauri():
    helper = NATIVE_FILES.read_text(encoding = "utf-8")
    history = CHAT_EXPORT.read_text(encoding = "utf-8")
    data_tab = DATA_TAB.read_text(encoding = "utf-8")
    prompt_storage = PROMPT_STORAGE.read_text(encoding = "utf-8")

    projects = (FRONTEND / "features/chat/projects-page.tsx").read_text(encoding = "utf-8")

    assert 'invoke<string | null>("save_native_file", bytes, {' in helper
    assert '"x-unsloth-default-name"' in helper
    assert "Array.from(new Uint8Array" not in helper
    assert 'invoke<NativeChatImport | null>("pick_native_chat_import")' in helper
    assert "if (isTauri)" in helper
    assert 'document.createElement("a")' in helper
    assert "DownloadCancelledError" in helper
    assert "throw new DownloadCancelledError()" in helper
    assert "return savedPath !== null" not in helper

    assert helper.index("if (isTauri)") < helper.index("  const blob =")
    assert "downloadFile(" in history
    assert "downloadFile(" in prompt_storage
    assert "pickNativeChatImport" in data_tab
    assert "if (!isTauri)" in data_tab

    assert "pickNativeChatImport" in projects
    assert "if (!isTauri)" in projects
    # Browser builds retain the existing hidden-input route.
    assert 'type="file"' in data_tab
    assert 'accept=".jsonl,.ndjson,.csv"' in data_tab

    native_dialogs = NATIVE_DIALOGS.read_text(encoding = "utf-8")
    assert 'CHAT_IMPORT_EXTENSIONS: &[&str] = &["jsonl", "ndjson", "csv"]' in native_dialogs
    assert "InvokeBody::Raw" in native_dialogs

    assert ".tempfile_in(parent)" in native_dialogs
    assert ".persist(&path)" in native_dialogs
    assert "fs::write(&path, content)" not in native_dialogs


def test_chat_exports_await_native_saves_and_markdown_uses_shared_helper():
    app_sidebar = APP_SIDEBAR.read_text(encoding = "utf-8")
    prompt_storage = PROMPT_STORAGE.read_text(encoding = "utf-8")
    thread = THREAD.read_text(encoding = "utf-8")
    thread_sidebar = THREAD_SIDEBAR.read_text(encoding = "utf-8")
    shared_composer = SHARED_COMPOSER.read_text(encoding = "utf-8")

    data_tab = DATA_TAB.read_text(encoding = "utf-8")
    projects = (FRONTEND / "features/chat/projects-page.tsx").read_text(encoding = "utf-8")
    assert "async function downloadBlob(" in prompt_storage
    download_blob = prompt_storage.split("async function downloadBlob(", 1)[1].split("\n}\n", 1)[0]
    assert "return downloadFile(" in download_blob
    assert "catch (error)" not in download_blob
    assert "isDownloadCancelled(error)" in prompt_storage

    for source in (app_sidebar, thread, thread_sidebar, shared_composer, data_tab, projects):
        assert "isDownloadCancelled(error)" in source
    assert "const handleExport = useCallback(async () =>" in prompt_storage
    assert prompt_storage.count("await export") >= 12
    assert "await Promise.all(" not in app_sidebar
    assert "for (const id of ids)" in app_sidebar
    assert prompt_storage.count("await downloadBlob(") >= 5

    assert "await downloadBlob(zipped," in prompt_storage
    assert "new Blob([zipped]" not in prompt_storage
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
    setup_layout = source.split("async function showSetupWindow", 1)[1].split(
        "async function enforceMinimumWindowSize", 1
    )[0]
    reset_call = 'invoke("reset_app_window_layout_initialized")'
    assert reset_call in setup_layout
    assert setup_layout.index(reset_call) < setup_layout.index("win.setSize")
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


def test_chat_sidebar_row_actions_visible_on_coarse_pointers():
    """unslothai/unsloth#7276: Recents chat kebab must be tappable on iPad."""
    sidebar_source = APP_SIDEBAR.read_text(encoding = "utf-8")
    css_source = INDEX_CSS.read_text(encoding = "utf-8")
    assert "renderChatSidebarItem" in sidebar_source
    block = sidebar_source.split("function renderChatSidebarItem", 1)[1].split("\n  function ", 1)[
        0
    ]
    assert "[@media(pointer:coarse)]:pr-10" in block
    assert "sidebar-touch-reveal" in block
    # Coarse-pointer visibility must come after .sidebar-row-action { opacity-0 }.
    coarse_idx = css_source.index("@media (pointer: coarse)")
    base_idx = css_source.index(".sidebar-row-action {")
    assert coarse_idx > base_idx
    coarse_block = css_source[coarse_idx : coarse_idx + 280]
    assert "sidebar-touch-reveal" in coarse_block
    assert "opacity-100" in coarse_block
    assert "pointer-events-auto" in coarse_block
    # Must not reveal every sidebar-row-action (project/run/nav rows lack padding).
    assert ".sidebar-row-action {\n\t\t\t@apply opacity-100" not in coarse_block
    assert ".sidebar-row-action.sidebar-touch-reveal" in coarse_block
