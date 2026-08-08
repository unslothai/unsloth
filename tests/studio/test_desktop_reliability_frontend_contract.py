# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static contracts for focused packaged-desktop reliability behavior."""

import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "studio/frontend/src"
NATIVE_FILES = FRONTEND / "lib/native-files.ts"
CHAT_EXPORT = FRONTEND / "features/chat/utils/export-chat-history.ts"
DATA_TAB = FRONTEND / "features/settings/tabs/data-tab.tsx"
PROMPT_STORAGE = FRONTEND / "features/chat/prompt-storage/prompt-storage-dialog.tsx"

APP_SIDEBAR = FRONTEND / "components/app-sidebar.tsx"
SIDEBAR_PRIMITIVE = FRONTEND / "components/ui/sidebar.tsx"
NAVBAR = FRONTEND / "components/navbar.tsx"
INDEX_CSS = FRONTEND / "index.css"
THREAD = FRONTEND / "components/assistant-ui/thread.tsx"
THREAD_SIDEBAR = FRONTEND / "features/chat/thread-sidebar.tsx"
SHARED_COMPOSER = FRONTEND / "features/chat/shared-composer.tsx"
TITLEBAR = FRONTEND / "components/tauri/window-titlebar.tsx"
SHEET = FRONTEND / "components/ui/sheet.tsx"
RESEARCH_ACTIVITY_PANEL = FRONTEND / "features/chat/components/research-activity-panel.tsx"
RESPONSE_DETAILS_SHEET = FRONTEND / "components/assistant-ui/message-response-details-sheet.tsx"
DOCUMENT_PREVIEW_SHEET = FRONTEND / "features/rag/components/document-preview-sheet.tsx"
NATIVE_DIALOGS = REPO / "studio/src-tauri/src/native_file_dialogs.rs"
NATIVE_CLIPBOARD = REPO / "studio/src-tauri/src/native_clipboard.rs"
TAURI_MAIN = REPO / "studio/src-tauri/src/main.rs"
TAURI_COMMANDS = REPO / "studio/src-tauri/src/commands.rs"
TAURI_UPDATE_CONTEXT = FRONTEND / "hooks/tauri-update-context.ts"
TAURI_UPDATE_HOOK = FRONTEND / "hooks/use-tauri-update.ts"
UPDATE_INSTRUCTIONS = FRONTEND / "features/settings/components/update-studio-instructions.tsx"
DESKTOP_UPDATE_CONTROL = FRONTEND / "features/settings/components/desktop-update-control.tsx"
GENERAL_SETTINGS = FRONTEND / "features/settings/tabs/general-tab.tsx"
DESKTOP_UPDATE_POLICY = REPO / "studio/src-tauri/src/desktop_update_policy.rs"


APP_PROVIDER = FRONTEND / "app/provider.tsx"
ROOT_ROUTE = FRONTEND / "app/routes/__root.tsx"
IMAGES_PAGE = FRONTEND / "features/images/images-page.tsx"
VIDEO_PAGE = FRONTEND / "features/video/video-page.tsx"
VIDEO_API = FRONTEND / "features/video/api.ts"
RAG_API = FRONTEND / "features/rag/api/rag-api.ts"

REMOTE_ACCESS_SECTION = FRONTEND / "features/settings/components/remote-access-section.tsx"
PASSWORD_DIALOG = FRONTEND / "features/settings/components/change-password-dialog.tsx"
GENERAL_TAB = FRONTEND / "features/settings/tabs/general-tab.tsx"

CLIPBOARD_FILES = FRONTEND / "features/chat/utils/clipboard-files.ts"
TAURI_CAPABILITIES = REPO / "studio/src-tauri/capabilities/default.json"
CHAT_PAGE = FRONTEND / "features/chat/chat-page.tsx"
TRAINING_CONFIG_ACTIONS = FRONTEND / "features/studio/wizard/config-actions.tsx"
MARKDOWN_TEXT = FRONTEND / "components/assistant-ui/markdown-text.tsx"
IMAGE = FRONTEND / "components/assistant-ui/image.tsx"
AUDIO_PLAYER = FRONTEND / "components/assistant-ui/audio-player.tsx"


def _chrome_style_blocks(source: str) -> dict[str, dict[str, str]]:
    """Each ``const <NAME>_STYLE = { ... } as CSSProperties`` block as a var -> value map.

    Per block, so a value is only ever compared against the others that ship with it.
    """
    return {
        name: dict(re.findall(r'"(--[\w-]+)":\s*"([^"]+)"', body))
        for name, body in re.findall(
            r"const (\w+_STYLE) = \{(.*?)\} as CSSProperties;", source, re.S
        )
    }


def _titlebar_nav_button_px(source: str) -> int | None:
    """The navigation button's box, read off the class string that sizes it."""
    match = re.search(r"const buttonClass =\s*\n?\s*\"[^\"]*?size-\[(\d+)px\]", source, re.S)
    return int(match.group(1)) if match else None


def _px(value: str | None) -> int | None:
    """*value* as whole pixels, or None if it is not a px literal (rem, calc, absent)."""
    match = re.fullmatch(r"(\d+)px", (value or "").strip())
    return int(match.group(1)) if match else None


def test_desktop_update_offer_remains_actionable_from_settings():
    provider = APP_PROVIDER.read_text(encoding = "utf-8")
    context = TAURI_UPDATE_CONTEXT.read_text(encoding = "utf-8")
    hook = TAURI_UPDATE_HOOK.read_text(encoding = "utf-8")
    settings = DESKTOP_UPDATE_CONTROL.read_text(encoding = "utf-8")

    assert "<TauriUpdateContext.Provider value={update}>" in provider
    context_start = provider.index("<TauriUpdateContext.Provider value={update}>")
    context_end = provider.index("</TauriUpdateContext.Provider>", context_start)
    assert "{appContent}" in provider[context_start:context_end]
    assert "appContent={" in provider
    assert "useContext(TauriUpdateContext)" in context
    # Scope these: bare substrings also match setTimeout(checkForUpdate, 5000) and installUpdate().
    assert "checkForUpdate," in hook.split("  return {", 1)[1]
    manual = hook.split("async function checkForUpdate()", 1)[1]
    assert "checkedRef.current = true;" in manual.split("try {", 1)[0]
    offer = hook.split("function offerUpdate", 1)[1].split("\n  }", 1)[0]
    assert "setDismissed(false);" in offer
    assert "isNewOffer" in offer
    assert "const available = update.info !== null && !checking;" in settings
    assert "void update.installUpdate();" in settings
    assert "void update.checkForUpdate();" in settings


def test_desktop_update_search_has_a_stable_general_tab_destination():
    general = GENERAL_SETTINGS.read_text(encoding = "utf-8")

    assert 'data-settings-label={t("settings.about.updates")}' in general
    assert "<DesktopUpdateControl />" in general


def test_desktop_update_keeps_the_in_app_path_on_a_guessed_policy():
    """resolveUpdatePolicy fails safe to manual_linux_package on every platform.

    Acting on that guess routes macOS, Windows and AppImage into the Linux-only
    command, which returns Ok(None) off Linux, so Settings would claim the app
    was up to date while an update was waiting.
    """
    hook = TAURI_UPDATE_HOOK.read_text(encoding = "utf-8")
    policy = DESKTOP_UPDATE_POLICY.read_text(encoding = "utf-8")

    assert "resolved: boolean" in hook
    assert "resolved: false" in hook

    manual_branch = hook.split("async function checkForUpdate()", 1)[1].split(
        'if (policy.mode === "manual_linux_package") {',
        1,
    )[1]
    give_up = manual_branch.split("checkDesktopUpdate()", 1)[0]
    # Only a resolved policy may end the check without the in-app updater.
    assert "if (resolved) {" in give_up
    assert 'setStatus("idle");' in give_up
    assert "await checkDesktopUpdate();" in manual_branch
    # The Rust command self-gates on the real OS, so it is safe to consult first.
    manual_cmd = policy.split("async fn check_desktop_manual_update", 1)[1]
    assert "ManualLinuxPackage" in manual_cmd.split("{", 1)[1][:400]


def test_settings_update_button_is_inert_while_an_install_runs():
    settings = DESKTOP_UPDATE_CONTROL.read_text(encoding = "utf-8")

    assert 'update.status === "updating-backend"' in settings
    assert 'update.status === "downloading"' in settings
    assert 'update.status === "installing"' in settings
    assert "disabled={busy}" in settings
    assert "aria-busy={busy}" in settings


def test_desktop_update_check_failures_are_retryable():
    hook = TAURI_UPDATE_HOOK.read_text(encoding = "utf-8")
    settings = DESKTOP_UPDATE_CONTROL.read_text(encoding = "utf-8")
    policy = DESKTOP_UPDATE_POLICY.read_text(encoding = "utf-8")

    assert "setCheckError(String(e));" in hook
    assert "update.checkError !== null" in settings
    assert 't("settings.about.update.retryCheck")' in settings
    # The reason must reach the user without guessing that every failure is a network problem.
    assert "description = update.checkError ?? label;" in settings
    assert 't("settings.about.update.desktopCheckFailedDescription")' not in settings
    assert "server returned HTTP {status}" in policy
    request = policy.split("let response = client", 1)[1].split("let metadata", 1)[0]
    assert ".map_err(" in request
    assert "return Ok(None);" not in request


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


def test_generated_download_buttons_use_the_native_save_boundary():
    helper = NATIVE_FILES.read_text(encoding = "utf-8")
    training = TRAINING_CONFIG_ACTIONS.read_text(encoding = "utf-8")
    markdown = MARKDOWN_TEXT.read_text(encoding = "utf-8")
    image = IMAGE.read_text(encoding = "utf-8")
    audio = AUDIO_PLAYER.read_text(encoding = "utf-8")

    assert "downloadFile(bytes, filename" in helper
    assert "browserUrlDownload(url, filename)" in helper
    assert "if (!isTauri)" in helper
    assert "downloadFile(yaml, filename" in training
    assert "downloadFile(text, filename" in markdown
    assert "fallbackExt" in markdown
    assert 'rust: "rs"' in markdown
    assert "downloadUrl(part.image, filename)" in image
    assert "urlToBlob(part.image)" in image
    assert 'downloadUrl(src, "generated-audio.wav")' in audio

    tauri_config = (REPO / "studio/src-tauri/tauri.conf.json").read_text(encoding = "utf-8")
    assert "connect-src 'self' ipc: http://ipc.localhost" in tauri_config

    for source in (training, markdown, image, audio):
        assert 'document.createElement("a")' not in source
        assert "isDownloadCancelled(error)" in source


def test_gallery_video_links_are_absolute_and_saved_natively():
    video_api = VIDEO_API.read_text(encoding = "utf-8")
    video_page = VIDEO_PAGE.read_text(encoding = "utf-8")
    rag_api = RAG_API.read_text(encoding = "utf-8")

    # The backend mints this link relative so a proxy can serve it. Its consumers are
    # <video src> and the download, none of which go through authFetch, so a relative
    # path under Tauri resolves against the webview and yields the SPA shell.
    assert "return apiUrl(body.url);" in video_api
    assert 'from "@/lib/api-base"' in video_api
    # The same fix the RAG document preview already carries.
    assert "return apiUrl(data.url);" in rag_api

    assert "downloadUrl(src, filename)" in video_page
    assert "downloadFile(blob, filename" in video_page
    assert 'document.createElement("a")' not in video_page
    assert "isDownloadCancelled(err)" in video_page

    # media-src, not just connect-src: the signed link is played by an element.
    tauri_config = (REPO / "studio/src-tauri/tauri.conf.json").read_text(encoding = "utf-8")
    assert (
        "media-src 'self' data: blob: https: http://localhost:* http://127.0.0.1:*" in tauri_config
    )

    # The save dialog now offers these to video, not just to the audio player.
    dialogs = NATIVE_DIALOGS.read_text(encoding = "utf-8")
    assert '("MPEG-4 video or audio", filter_extensions(["m4a", "mp4"]))' in dialogs
    assert '("WebM video or audio", filter_extensions(["webm"]))' in dialogs


def test_clipboard_file_paste_is_bounded_and_wired_to_both_composers():
    helper = CLIPBOARD_FILES.read_text(encoding = "utf-8")
    thread = THREAD.read_text(encoding = "utf-8")
    shared_composer = SHARED_COMPOSER.read_text(encoding = "utf-8")
    capabilities = TAURI_CAPABILITIES.read_text(encoding = "utf-8")

    for contract in (
        "clipboardData.files",
        "clipboardData.items",
        "item.getAsFile()",
        "file.size > 0",
        'clipboardData.getData("text/plain")',
        "event.isTrusted",
        "event.defaultPrevented",
        'types.includes("files")',
        'type.includes("uri-list")',
        '"read_native_clipboard_files"',
        "globalThis.atob(file.base64)",
        "new File([bytes], file.name",
        "MAX_CLIPBOARD_BYTES",
        'import("@tauri-apps/plugin-clipboard-manager")',
        "await readImage()",
        "rgba.byteLength !== expectedRgbaBytes",
        "await image.close()",
    ):
        assert contract in helper

    assert "addAttachmentOnPaste={false}" in thread
    assert "onPaste={handleFilePaste}" in thread
    assert "pasteClipboardFiles" in thread
    assert "aui.composer().addAttachment(file)" in thread
    assert "onPaste={handleFilePaste}" in shared_composer
    assert "pasteClipboardFiles" in shared_composer
    assert "addFiles(files)" in shared_composer
    assert capabilities.count('"clipboard-manager:allow-read-image"') == 1
    assert '"clipboard-manager:allow-read-text"' not in capabilities


def test_native_clipboard_bridge_is_bounded_and_registered():
    native_clipboard = NATIVE_CLIPBOARD.read_text(encoding = "utf-8")
    tauri_main = TAURI_MAIN.read_text(encoding = "utf-8")

    for contract in (
        "MAX_CLIPBOARD_FILES",
        "MAX_CLIPBOARD_URI_BYTES",
        "MAX_CLIPBOARD_TOTAL_BYTES",
        "MAX_CLIPBOARD_SOURCE_BYTES",
        "MAX_CLIPBOARD_RGBA_BYTES",
        ".take(limit + 1)",
        ".wait_for_uris()",
        ".wait_for_targets()",
        'contains("copied-files")',
        "open_regular_clipboard_file(&path)",
        '"/proc/self/fd/{}"',
        ".wait_for_image()",
        "glib::filename_from_uri",
        "glib::MainContext::default().invoke",
        "arboard::Clipboard::new()",
        "BASE64.encode(bytes)",
        "tauri::ipc::Response::new(png)",
    ):
        assert contract in native_clipboard

    assert "native_clipboard::read_native_clipboard_files" in tauri_main
    assert "native_clipboard::read_native_clipboard_png" in tauri_main


def test_mac_dock_reopens_hidden_main_window():
    source = TAURI_MAIN.read_text(encoding = "utf-8")
    show_helper = source.split("fn show_main_window", 1)[1].split("\n}\n", 1)[0]
    run_handler = source.split(".run(|app, event|", 1)[1]

    for action in ("window.show()", "window.unminimize()", "window.set_focus()"):
        assert action in show_helper
    assert "tauri::RunEvent::Reopen" in run_handler
    assert "has_visible_windows: false" in run_handler
    reopen_handler = run_handler.split("tauri::RunEvent::Reopen", 1)[1].split("=>", 1)[1]
    assert "show_main_window(app)" in reopen_handler


def test_desktop_manages_the_remote_password_through_the_account_dialog():
    section = REMOTE_ACCESS_SECTION.read_text(encoding = "utf-8")
    dialog = PASSWORD_DIALOG.read_text(encoding = "utf-8")

    row = section.split("function RemotePasswordRow", 1)[1].split(
        "export function RemoteAccessSection", 1
    )[0]
    assert "if (!(isTauri && status)) {" in row
    assert "initial={status.passwordPending}" in row
    assert "<RemotePasswordRow status={status} onDone={refreshStatus} />" in section
    assert "{isTauri ? null : (" in GENERAL_TAB.read_text(encoding = "utf-8")
    # A password change rotates credentials outside the polling requests.
    refresh = section.split("const refreshStatus = useCallback(", 1)[1].split("}, []);", 1)[0]
    assert "mutationEpoch.current += 1;" in refresh
    assert "setPollRevision(" in refresh
    # Initial mode sends no current password; the web flow it serves keeps it.
    body = dialog.split("function changePasswordBody", 1)[1].split("function dialogCopy", 1)[0]
    assert '? [["new_password", nextPassword]]' in body
    assert '["current_password", currentPassword],' in body
    post = dialog.split("function postChangePassword", 1)[1].split(
        "async function requestPasswordChange", 1
    )[0]
    assert '? "/api/auth/desktop-initial-password"' in post
    assert ': "/api/auth/change-password",' in post
    assert "{initial ? null : (" in dialog
    assert "if (!initial && currentPassword.length < MIN_PASSWORD_LENGTH)" in dialog
    submitted = dialog.split("async function submit", 1)[1].split("\n  }", 1)[0]
    assert "storeAuthTokens(accessToken, refreshToken)" in submitted
    assert "onDone?.()" in submitted


def test_desktop_startup_waits_for_auth_without_intermediate_handoff():
    source = APP_PROVIDER.read_text(encoding = "utf-8")

    assert 'const showApp = status === "running" && desktopAuthReady;' in source
    assert "Preparing Unsloth" not in source
    assert "Signing in to desktop session" not in source
    assert "desktopBooting" not in source
    assert "showInteractiveApp" not in source
    assert "<NativeIntentDrain />" in source
    assert "{children}" in source


def test_full_app_layout_uses_its_own_initialized_marker():
    source = APP_PROVIDER.read_text(encoding = "utf-8")

    assert 'invoke<boolean>("has_initialized_app_window_layout")' in source
    setup_layout = source.split("async function showSetupWindow", 1)[1].split(
        "async function enforceWindowSizeBounds", 1
    )[0]
    reset_call = 'invoke("reset_app_window_layout_initialized")'
    assert reset_call in setup_layout
    assert setup_layout.index(reset_call) < setup_layout.index("placeWindow(")
    assert 'invoke("mark_app_window_layout_initialized")' in source
    assert "hasInitializedAppLayout && hasSavedState" in source


def test_first_app_layout_survives_a_stale_setup_window_size():
    source = APP_PROVIDER.read_text(encoding = "utf-8")
    bounds_helper = source.split("async function enforceWindowSizeBounds", 1)[1].split(
        "async function applyAppWindowLayout", 1
    )[0]
    app_layout = source.split("async function applyAppWindowLayout", 1)[1].split(
        "async function showWindowFallback", 1
    )[0]

    assert "requestedSize: LogicalWindowSize = bounds.minimum" in bounds_helper
    assert "constrainWindowSize(currentSize, requestedSize, bounds)" in bounds_helper
    assert "const cssSafeLogicalWidth = measured.monitor" in app_layout
    first_size_call = app_layout.split("requestedSize = calculateFirstAppWindowSize(", 1)[1].split(
        ");", 1
    )[0]
    assert "measured.bounds," in first_size_call
    assert "cssSafeLogicalWidth," in first_size_call
    assert "finalizeAppWindowLayout({" in app_layout
    assert "enforceWindowSizeBounds(" in app_layout
    finalize_call = app_layout.split("finalizeAppWindowLayout({", 1)[1].split("});", 1)[0]
    assert "measured," in finalize_call
    # Limit the check to this call's arguments.
    bounds_call = app_layout.split("enforceWindowSizeBounds(", 1)[1].split(");", 1)[0]
    assert "bounds," in bounds_call
    assert "requestedSize," in bounds_call


def test_expanded_titlebar_button_and_corner_match_sidebar_edge():
    source = TITLEBAR.read_text(encoding = "utf-8")

    assert 'showSidebarSurface && !pinned ? "7rem" : sidebarWidth' in source
    assert "style={{ width: titlebarNavigationWidth }}" in source
    assert "left: titlebarNavigationWidth" in source
    assert "<DesktopTitlebarNavigation" in source
    assert "const contentBorderLeft = pinned" in source
    assert ': "0px";' in source
    # The curved transition and sidebar-colored backing are expanded-only.
    assert source.count("{showSidebarSurface && pinned && (") == 2
    assert (
        'className="pointer-events-none absolute top-full size-3 -translate-x-px bg-sidebar"'
        in source
    )
    assert (
        'className="pointer-events-none absolute top-full size-3 -translate-x-px rounded-tl-[12px] border-l border-t border-sidebar-border bg-background"'
        in source
    )


def test_desktop_titlebar_separates_navigation_from_sidebar_brand():
    titlebar = TITLEBAR.read_text(encoding = "utf-8")
    sidebar = APP_SIDEBAR.read_text(encoding = "utf-8")
    header = sidebar.split("<SidebarHeader", 1)[1].split("</SidebarHeader>", 1)[0]

    # The names, not the whole import list: #8025 added Minus/Square/X to the
    # same line for the window controls and this went red on every open PR.
    lucide = re.search(r"import \{([^}]*)\} from \"lucide-react\";", titlebar)
    assert lucide is not None, "titlebar no longer imports from lucide-react"
    icons = {name.strip() for name in lucide.group(1).split(",")}
    assert {"ArrowLeft", "ArrowRight"} <= icons, icons
    assert "<ArrowLeft" in titlebar
    assert "<ArrowRight" in titlebar
    assert "window.history.back()" in titlebar
    assert "window.history.forward()" in titlebar
    assert 'src="/circle-logo-small.png"' in header
    assert header.index("<DesktopTitlebarNavigation") < header.index('src="/circle-logo-small.png"')


def test_collapsed_tauri_keeps_history_arrows_and_adds_new_chat_by_model_picker():
    titlebar = TITLEBAR.read_text(encoding = "utf-8")
    chat_page = CHAT_PAGE.read_text(encoding = "utf-8")
    navigation = titlebar.split("export function DesktopTitlebarNavigation", 1)[1].split(
        "export function WindowTitlebar", 1
    )[0]

    assert "{expanded && (" not in navigation
    assert navigation.count('aria-label="Go back"') == 1
    assert navigation.count('aria-label="Go forward"') == 1

    assert "inline-flex size-[30px] shrink-0" in navigation

    assert navigation.count("onDoubleClick={stopTitlebarDrag}") == 3
    assert "maximized" not in navigation
    assert "const maximizeRefreshSequence = useRef(0);" in titlebar
    assert "const scheduleMaximizedRefresh = useCallback" in titlebar
    assert "window.setTimeout(() =>" in titlebar
    assert "scheduleMaximizedRefresh();" in titlebar

    assert '"pl-3"' in titlebar
    assert 'isTauri && !isMobile && !pinned && view.mode !== "compare"' in chat_page

    assert "pl-[var(--studio-collapsed-chat-controls-inset,0.75rem)]" in chat_page
    assert '"--studio-collapsed-chat-controls-inset": "188px"' in APP_PROVIDER.read_text(
        encoding = "utf-8"
    )
    assert 'className="!size-[30px] rounded-[10px] text-muted-foreground"' in chat_page
    assert 'aria-label="New chat"' in chat_page
    new_chat_click = chat_page.index("onClick={handleDesktopNewChat}")
    assert new_chat_click < chat_page.index("<ModelSelector", new_chat_click)


def test_tauri_collapse_removes_the_icon_rail_but_web_keeps_it():
    app_sidebar = APP_SIDEBAR.read_text(encoding = "utf-8")
    primitive = SIDEBAR_PRIMITIVE.read_text(encoding = "utf-8")
    navbar = NAVBAR.read_text(encoding = "utf-8")

    assert "collapseToZero={isTauri}" in app_sidebar
    assert "collapseToZero = false" in primitive
    assert 'collapseToZero ? "w-0" : "w-(--sidebar-width-icon)"' in primitive
    assert "usesNativeMacTitlebar && !pinned" in navbar
    assert "<DesktopTitlebarNavigation" in navbar

    assert "top-px z-[60]" in navbar
    assert "z-40 h-[48px]" in navbar

    assert "windowFocused" not in navbar
    assert "bg-[#d0d0d0]" not in navbar
    assert "translate-y-[var(--studio-titlebar-navigation-offset-y,0px)]" in TITLEBAR.read_text(
        encoding = "utf-8"
    )
    # The nudge has to move the navigation without pushing it out of the titlebar it sits
    # in, so the button box travels with it. The container's mt-1 is deliberately not in
    # the sum: translate-y is visual, and the margin already seats the box in the row.
    button = _titlebar_nav_button_px(TITLEBAR.read_text(encoding = "utf-8"))
    assert button is not None, "navigation button size no longer readable from buttonClass"
    blocks = _chrome_style_blocks(APP_PROVIDER.read_text(encoding = "utf-8"))
    nudged = {
        name: values
        for name, values in blocks.items()
        if "--studio-titlebar-navigation-offset-y" in values
    }
    assert nudged, blocks.keys()
    for name, values in nudged.items():
        offset = _px(values["--studio-titlebar-navigation-offset-y"])
        titlebar = _px(values.get("--studio-desktop-titlebar-height"))
        assert offset is not None and offset > 0, (name, values)
        assert titlebar is not None, (name, values)
        assert offset + button <= titlebar, (name, offset, button, titlebar)
    assert "aria-hidden={(hasPinMode && !pinned && collapseToZero) || undefined}" in primitive
    assert "inert={(hasPinMode && !pinned && collapseToZero) || undefined}" in primitive


def test_fixed_sheets_start_below_the_custom_titlebar():
    provider = APP_PROVIDER.read_text(encoding = "utf-8")
    sheet = SHEET.read_text(encoding = "utf-8")

    # Portalled sheets read the height off <html>, so the mirror has to stay.
    assert 'set("--studio-custom-titlebar-height", usesCustomTitlebar ? "34px" : null)' in provider

    # Only viewport-fixed sheets clear the titlebar; the absolute recipe block
    # sheet sits in its own container and keeps a plain top edge.
    assert 'position === "fixed" ? VIEWPORT_TOP_EDGE : CONTAINED_TOP_EDGE' in sheet
    for side in ("left", "right", "top"):
        assert f"data-[side={side}]:top-[var(--studio-custom-titlebar-height,0px)]" in sheet
        assert f"data-[side={side}]:top-0" in sheet

    # Anchor both edges so the inset shrinks the sheet; h-full would instead
    # push its bottom past the viewport.
    for side in ("left", "right"):
        assert f"data-[side={side}]:bottom-0" in sheet
        assert f"data-[side={side}]:h-full" not in sheet

    # The shared class is the only sheet offset; a local one would double up.
    # Dialogs still read --studio-window-chrome-top (DesktopChromeVarsEffect).
    for portalled in (
        RESEARCH_ACTIVITY_PANEL,
        RESPONSE_DETAILS_SHEET,
        DOCUMENT_PREVIEW_SHEET,
    ):
        assert "studio-custom-titlebar-height" not in portalled.read_text(encoding = "utf-8")


def test_visible_mac_sidebar_header_is_a_drag_region():
    source = APP_SIDEBAR.read_text(encoding = "utf-8")
    header = source.split("<SidebarHeader", 1)[1].split("</SidebarHeader>", 1)[0]
    drag_region = "data-tauri-drag-region={usesNativeMacTitlebar || undefined}"

    assert drag_region in header
    assert header.index(drag_region) < header.index('"relative z-10 flex items-center')


def test_mac_chat_header_controls_share_the_titlebar_row():
    source = CHAT_PAGE.read_text(encoding = "utf-8")
    provider = APP_PROVIDER.read_text(encoding = "utf-8")

    assert "shouldUseNativeMacWindowTitlebar" not in source
    assert "[--studio-content-top-inset:var(--studio-mac-titlebar-height" not in source
    assert source.count("var(--studio-mac-traffic-light-inset") == 2
    # Sharing the row is the contract: the padding must leave the control room inside the
    # header, so a retune to a large value fails here rather than shipping a clipped row.
    blocks = _chrome_style_blocks(provider)
    padded = {
        name: values
        for name, values in blocks.items()
        if "--studio-chat-header-padding-top" in values
    }
    assert padded, blocks.keys()
    for name, values in padded.items():
        padding = _px(values["--studio-chat-header-padding-top"])
        header = _px(values.get("--studio-chat-header-height"))
        control = _px(values.get("--studio-chat-control-height"))
        assert padding is not None and padding > 0, (name, values)
        assert header is not None and control is not None, (name, values)
        assert padding + control <= header, (name, padding, control, header)
    assert "pt-[var(--studio-content-top-inset,0px)] md:flex-row" in source
    assert "absolute top-[var(--studio-content-top-inset,0px)]" in source


def test_collapsed_mac_sidebar_hides_divider():
    source = APP_SIDEBAR.read_text(encoding = "utf-8")

    assert "group-data-[collapsible=icon]:[&_[data-sidebar=sidebar]]:border-r-0" in source
    assert "top-[var(--studio-mac-titlebar-height,34px)]" not in source


def test_chat_sidebar_rows_are_compact_without_vertical_padding():
    sidebar_source = APP_SIDEBAR.read_text(encoding = "utf-8")
    block = sidebar_source.split("function renderChatSidebarItem", 1)[1]

    assert (
        '"sidebar-nav-btn h-[30px] cursor-pointer rounded-full py-0 pr-4 '
        'text-ui-14p5 leading-ui-19 tracking-nav font-medium"'
    ) in block
    assert (
        '"text-foreground h-[30px] w-full border-0 bg-transparent py-0 pr-4 '
        'text-ui-14p5 leading-ui-19 font-medium tracking-nav outline-none"'
    ) in block
    assert 'isPinned && variant !== "project" && "gap-[8.5px]"' in block
    assert 'variant === "project" ? "pl-[39px]" : "pl-3"' in block


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


def test_media_pages_clear_the_custom_titlebar():
    """The chat-style layout gives the media pages no outer inset, so each applies its own."""
    root = ROOT_ROUTE.read_text(encoding = "utf-8")

    assert "const isChatLike = isChatRoute || isImagesRoute || isVideoRoute;" in root
    for page in (IMAGES_PAGE, VIDEO_PAGE):
        shell = page.read_text(encoding = "utf-8").split('"diffusion-surface', 1)[1].split(">", 1)[0]
        assert "pt-[var(--studio-content-top-inset,0px)]" in shell, page.name


def test_media_page_headers_out_stack_the_mac_drag_region():
    """macOS insets the media pages 0px, so their 48px header overlaps the navbar's 34px drag
    strip: the band must out-stack it yet stay click-through (controls click, gaps drag)."""
    navbar = NAVBAR.read_text(encoding = "utf-8")

    # The strip to beat: same z-40, but earlier in DOM order.
    assert "pointer-events-none absolute inset-x-0 top-0 z-40 h-[48px]" in navbar
    assert "data-tauri-drag-region" in navbar

    for page in (IMAGES_PAGE, VIDEO_PAGE):
        source = page.read_text(encoding = "utf-8")
        before, marker, band = source.partition("h-[48px] shrink-0 items-start justify-between")
        assert marker, page.name
        opening = before.rsplit('<div className="', 1)[1]
        for token in ("pointer-events-none", "relative", "z-40"):
            assert token in opening, (page.name, token)

        band = band.split("MediaPageLink", 1)[0]
        groups = re.findall(r'<div className="([^"]*flex items-center gap-[^"]*)"', band)
        assert len(groups) >= 2, (page.name, groups)
        for group in groups:
            assert "pointer-events-auto" in group, (page.name, group)


def test_a_stopped_repair_update_is_recorded_as_canceled_not_failed():
    """unslothai/unsloth#7793: the support report prints final_status verbatim, so a
    user quitting mid-update must not read as a failed repair."""
    source = TAURI_COMMANDS.read_text(encoding = "utf-8")
    stopped_arm = source.split("if msg == update::UPDATE_STOPPED", 1)[1].split(
        "return Err(msg);", 1
    )[0]
    # The status argument of the call, so the surrounding comment cannot satisfy this.
    call = stopped_arm.split("finish_repair_group(", 1)[1].split(");", 1)[0]
    assert '"canceled"' in call
    assert '"failed"' not in call
