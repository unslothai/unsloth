# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static contracts for focused packaged-desktop reliability behavior."""

import functools
import re
from pathlib import Path
from tests.studio._js_source import (
    attribute_expressions,
    binding_joining,
    boolean_table,
    expand_bindings,
    gates_the_markup,
)


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
INTERFACE_SCALE_RUNTIME = FRONTEND / "features/settings/lib/interface-scale-runtime.ts"
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
AUDIO_PAGE = FRONTEND / "features/audio/audio-page.tsx"

DIFFUSION_TRAIN_PANEL = FRONTEND / "features/images/train/diffusion-train-panel.tsx"
MEDIA_PAGE_LINK = FRONTEND / "components/media-page-link.tsx"
VIDEO_PAGE = FRONTEND / "features/video/video-page.tsx"
VIDEO_API = FRONTEND / "features/video/api.ts"
RAG_API = FRONTEND / "features/rag/api/rag-api.ts"

REMOTE_ACCESS_SECTION = FRONTEND / "features/settings/components/remote-access-section.tsx"
PASSWORD_DIALOG = FRONTEND / "features/settings/components/change-password-dialog.tsx"
GENERAL_TAB = FRONTEND / "features/settings/tabs/general-tab.tsx"

CLIPBOARD_FILES = FRONTEND / "features/chat/utils/clipboard-files.ts"
# The DataTransfer reading half moved here when long pastes became attachments
# (#8472). Both halves are still one contract, so read them as one.
CLIPBOARD_PAYLOAD = FRONTEND / "features/chat/utils/clipboard-payload.ts"
TAURI_CAPABILITIES = REPO / "studio/src-tauri/capabilities/default.json"
CHAT_PAGE = FRONTEND / "features/chat/chat-page.tsx"
TRAINING_CONFIG_ACTIONS = FRONTEND / "features/studio/wizard/config-actions.tsx"
MARKDOWN_TEXT = FRONTEND / "components/assistant-ui/markdown-text.tsx"
IMAGE = FRONTEND / "components/assistant-ui/image.tsx"
AUDIO_PLAYER = FRONTEND / "components/assistant-ui/audio-player.tsx"


def _scale_constants() -> dict[str, str]:
    """The mac chrome constants, resolved to the strings provider.tsx puts in a style block.

    ``NATIVE_MAC_TITLEBAR_HEIGHT_VAR`` is ``var(--studio-native-titlebar-height, 34px)``
    built from ``NATIVE_MAC_TITLEBAR_HEIGHT_PX``. The runtime divides by the interface zoom
    and provider.tsx uses the same constant as the CSS fallback, which is what keeps a
    single 34 in the codebase, so read it from there rather than repeating it here.
    """
    source = _ui_source(INTERFACE_SCALE_RUNTIME)
    numbers = dict(re.findall(r"export const (\w+_PX) = (\d+);", source))
    return {
        name: re.sub(r"\$\{(\w+)\}", lambda m: numbers.get(m.group(1), m.group(0)), body)
        for name, body in re.findall(r"export const (\w+_VAR) = `([^`]+)`;", source)
    }


def _chrome_style_blocks(source: str) -> dict[str, dict[str, str]]:
    """Each ``const <NAME>_STYLE = { ... } as CSSProperties`` block as a var -> value map.

    Per block, so a value is only ever compared against the others that ship with it.

    A value may be a string, a template, or one of the imported constants above. Before the
    interface-scale setting they were all plain px strings; resolving the other two is what
    keeps these contracts checking the same arithmetic instead of silently reading absent.
    """
    constants = _scale_constants()

    def resolve(raw: str) -> str:
        raw = raw.strip()
        if raw[:1] in ('"', "`"):
            raw = raw[1:-1]
        return re.sub(
            r"\$\{(\w+)\}", lambda m: constants.get(m.group(1), m.group(0)), constants.get(raw, raw)
        )

    return {
        name: {
            var: resolve(value)
            for var, value in re.findall(r'"(--[\w-]+)":\s*(`[^`]*`|"[^"]*"|\w+)', body)
        }
        for name, body in re.findall(
            r"const (\w+_STYLE) = \{(.*?)\} as CSSProperties;", source, re.S
        )
    }


def _titlebar_nav_button_px(source: str) -> int | None:
    """The navigation button's box, read off the class string that sizes it."""
    match = re.search(r"const buttonClass =\s*\n?\s*\"[^\"]*?size-\[(\d+)px\]", source, re.S)
    return int(match.group(1)) if match else None


def _px(value: str | None) -> int | None:
    """*value* as whole pixels at 100% interface scale, or None if it is not pixel-valued.

    **At 100%, and only there.** The mac chrome vars carry their own px fallback and the
    runtime divides that number by the webview zoom, because macOS draws the titlebar and
    traffic lights at a size zoom does not touch. So every sum below is the arithmetic as
    it ships at 100%, which is the scale these contracts were written against and the only
    one a static read of the source can see.
    """
    text = (value or "").strip()
    for pattern in (
        r"(\d+)px",
        r"var\(--[\w-]+,\s*(\d+)px\)",
    ):
        match = re.fullmatch(pattern, text)
        if match:
            return int(match.group(1))
    match = re.fullmatch(r"calc\((\d+)px\s*\+\s*var\(--[\w-]+,\s*(\d+)px\)\)", text)
    return int(match.group(1)) + int(match.group(2)) if match else None


def test_desktop_update_offer_remains_actionable_from_settings():
    provider = _ui_source(APP_PROVIDER)
    context = _ui_source(TAURI_UPDATE_CONTEXT)
    hook = _ui_source(TAURI_UPDATE_HOOK)
    settings = _ui_source(DESKTOP_UPDATE_CONTROL)

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
    general = _ui_source(GENERAL_SETTINGS)

    assert 'data-settings-label={t("settings.about.updates")}' in general
    assert "<DesktopUpdateControl />" in general


def test_desktop_update_keeps_the_in_app_path_on_a_guessed_policy():
    """resolveUpdatePolicy fails safe to manual_linux_package on every platform.

    Acting on that guess routes macOS, Windows and AppImage into the Linux-only
    command, which returns Ok(None) off Linux, so Settings would claim the app
    was up to date while an update was waiting.
    """
    hook = _ui_source(TAURI_UPDATE_HOOK)
    policy = _ui_source(DESKTOP_UPDATE_POLICY)

    assert "resolved: boolean" in hook
    assert "resolved: false" in hook

    manual_branch = hook.split("async function checkForUpdate()", 1)[1].split(
        'if (policy.mode === "manual_linux_package") {',
        1,
    )[1]
    give_up = manual_branch.split("checkDesktopUpdate()", 1)[0]
    # Only a resolved policy may end the check without the in-app updater.
    assert "if (resolved) {" in give_up
    assert 'updateStatus("idle");' in give_up
    assert "await checkDesktopUpdate();" in manual_branch
    # The Rust command self-gates on the real OS, so it is safe to consult first.
    manual_cmd = policy.split("async fn check_desktop_manual_update", 1)[1]
    assert "ManualLinuxPackage" in manual_cmd.split("{", 1)[1][:400]


def test_settings_update_button_is_inert_while_an_install_runs():
    settings = _ui_source(DESKTOP_UPDATE_CONTROL)

    assert 'update.status === "updating-backend"' in settings
    assert 'update.status === "downloading"' in settings
    assert 'update.status === "installing"' in settings
    assert "disabled={busy}" in settings
    assert "aria-busy={busy}" in settings


def test_desktop_update_check_failures_are_retryable():
    hook = _ui_source(TAURI_UPDATE_HOOK)
    settings = _ui_source(DESKTOP_UPDATE_CONTROL)
    policy = _ui_source(DESKTOP_UPDATE_POLICY)

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
    helper = _ui_source(NATIVE_FILES)
    history = _ui_source(CHAT_EXPORT)
    data_tab = _ui_source(DATA_TAB)
    prompt_storage = _ui_source(PROMPT_STORAGE)

    projects = _ui_source(FRONTEND / "features/chat/projects-page.tsx")

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
    # Open WebUI exports are .json arrays, so the picker takes that too.
    assert 'accept=".json,.jsonl,.ndjson,.csv"' in data_tab

    native_dialogs = _ui_source(NATIVE_DIALOGS)
    assert 'CHAT_IMPORT_EXTENSIONS: &[&str] = &["json", "jsonl", "ndjson", "csv"]' in native_dialogs
    assert "InvokeBody::Raw" in native_dialogs

    assert ".tempfile_in(parent)" in native_dialogs
    assert ".persist(&path)" in native_dialogs
    assert "fs::write(&path, content)" not in native_dialogs


def test_media_galleries_save_natively_with_feedback():
    images_page = _ui_source(IMAGES_PAGE)
    video_page = _ui_source(VIDEO_PAGE)
    reencode = images_page.split("async function reencodeImage(", 1)[1].split(
        "\n}\n\nasync function downloadImage", 1
    )[0]
    download = images_page.split("async function downloadImage(", 1)[1].split(
        "\n}\n\nfunction formatTimestamp", 1
    )[0]
    video_download = video_page.split("const handleDownload = useCallback(", 1)[1].split(
        "\n\n  const handleDelete", 1
    )[0]

    assert "await downloadUrl(src, filename);" in download
    assert "await downloadFile(outputBlob, filename, outputBlob.type);" in download
    assert "const originalBlob = await fetchGalleryBlob(image.url);" in download
    assert "await downloadFile(originalBlob, filename, originalBlob.type);" in download
    assert "blob.type !== `image/${format}`" in reencode
    assert "isDownloadCancelled(error)" in download
    assert "if (isTauri)" in download
    assert 'toast.success("Image saved", { description: filename });' in download
    assert 'document.createElement("a")' not in download

    assert "await downloadFile(blob, exportFilename(video, format), blob.type);" in video_page
    assert "if (isTauri)" in video_download
    assert 'toast.success("Video saved"' in video_download
    assert "function saveLink(" not in video_page


def test_chat_exports_await_native_saves_and_markdown_uses_shared_helper():
    app_sidebar = _ui_source(APP_SIDEBAR)
    prompt_storage = _ui_source(PROMPT_STORAGE)
    thread = _ui_source(THREAD)
    thread_sidebar = _ui_source(THREAD_SIDEBAR)
    shared_composer = _ui_source(SHARED_COMPOSER)

    data_tab = _ui_source(DATA_TAB)
    projects = _ui_source(FRONTEND / "features/chat/projects-page.tsx")
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
    helper = _ui_source(NATIVE_FILES)
    training = _ui_source(TRAINING_CONFIG_ACTIONS)
    markdown = _ui_source(MARKDOWN_TEXT)
    image = _ui_source(IMAGE)
    audio = _ui_source(AUDIO_PLAYER)

    assert "downloadFile(bytes, filename" in helper
    assert "browserUrlDownload(url, filename)" in helper
    assert "if (!isTauri)" in helper
    assert "downloadFile(yaml, filename" in training
    assert "downloadFile(text, filename" in markdown
    assert "fallbackExt" in markdown
    assert 'rust: "rs"' in markdown
    assert "downloadUrl(part.image, filename)" in image
    assert "urlToBlob(part.image)" in image
    assert "downloadUrl(src, filename)" in audio
    assert 'filename = "generated-audio.wav"' in audio

    tauri_config = (REPO / "studio/src-tauri/tauri.conf.json").read_text(encoding = "utf-8")
    assert "connect-src 'self' ipc: http://ipc.localhost" in tauri_config

    for source in (training, markdown, image, audio):
        assert 'document.createElement("a")' not in source
        assert "isDownloadCancelled(error)" in source


def test_gallery_video_links_are_absolute_and_saved_natively():
    video_api = _ui_source(VIDEO_API)
    video_page = _ui_source(VIDEO_PAGE)
    rag_api = _ui_source(RAG_API)

    # The backend mints this link relative so a proxy can serve it. Its consumers are
    # <video src> and the download, none of which go through authFetch, so a relative
    # path under Tauri resolves against the webview and yields the SPA shell.
    assert "return apiUrl(body.url);" in video_api
    assert 'from "@/lib/api-base"' in video_api
    # The same fix the RAG document preview already carries.
    assert "return apiUrl(data.url);" in rag_api

    # An absolute link is cross-origin, where the download attribute stops saving, so the
    # MP4 goes native. Streaming, not downloadUrl: a clip is capped at 2048x2048 x 1024
    # frames, too big to buffer for IPC, and the chooser must not wait on the body.
    helper = _ui_source(NATIVE_FILES)
    assert "downloadUrlStreaming(src, exportFilename(video, format))" in video_page
    assert '"save_native_file_from_url"' in helper
    assert "isDownloadCancelled(err)" in video_page
    # Converted exports cross the same native boundary after the backend returns their blob.
    assert "await downloadFile(blob, exportFilename(video, format), blob.type);" in video_page
    assert "URL.createObjectURL(blob)" not in video_page

    # media-src, not just connect-src: the signed link is played by an element.
    tauri_config = (REPO / "studio/src-tauri/tauri.conf.json").read_text(encoding = "utf-8")
    assert (
        "media-src 'self' data: blob: https: http://localhost:* http://127.0.0.1:*" in tauri_config
    )

    # The save dialog now offers these to video, not just to the audio player, and the
    # streaming command is registered and pinned to the local backend.
    dialogs = _ui_source(NATIVE_DIALOGS)
    assert '("MPEG-4 video or audio", filter_extensions(["m4a", "mp4"]))' in dialogs
    assert '("WebM video or audio", filter_extensions(["webm"]))' in dialogs
    assert "async fn stream_url_to_path" in dialogs
    # Parsed, not sliced: in http://127.0.0.1:8888@evil.test the loopback part is userinfo.
    assert "reqwest::Url::parse(url)" in dialogs
    assert "parsed.username().is_empty()" in dialogs
    assert "parsed.password().is_some()" in dialogs
    # The chooser has to come first, or the user waits on the body before being asked where.
    streaming = dialogs[dialogs.index("pub async fn save_native_file_from_url") :]
    assert streaming.index(".save_file(") < streaming.index("stream_url_to_path(&url")
    # No proxy (the signed URL must not reach one) and no redirects (they would leave loopback
    # after the check). read_timeout, not timeout: it bounds each chunk, so a backend that goes
    # quiet cannot hang the save while a legitimately large clip still finishes.
    loopback = (REPO / "studio/src-tauri/src/loopback_http.rs").read_text(encoding = "utf-8")
    assert "fn streaming_client" in loopback
    assert "redirect(reqwest::redirect::Policy::none())" in loopback
    assert ".read_timeout(read_timeout)" in loopback
    assert ".timeout(" not in loopback.split("fn streaming_client")[1]
    assert loopback.count(".no_proxy()") == 2
    assert "loopback_http::streaming_client" in dialogs
    main_rs = (REPO / "studio/src-tauri/src/main.rs").read_text(encoding = "utf-8")
    assert "native_file_dialogs::save_native_file_from_url," in main_rs


def test_clipboard_file_paste_is_bounded_and_wired_to_both_composers():
    helper = _ui_source(CLIPBOARD_FILES) + _ui_source(CLIPBOARD_PAYLOAD)
    thread = _ui_source(THREAD)
    shared_composer = _ui_source(SHARED_COMPOSER)
    capabilities = _ui_source(TAURI_CAPABILITIES)

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
    native_clipboard = _ui_source(NATIVE_CLIPBOARD)
    tauri_main = _ui_source(TAURI_MAIN)

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
    source = _ui_source(TAURI_MAIN)
    show_helper = source.split("fn show_main_window", 1)[1].split("\n}\n", 1)[0]
    run_handler = source.split(".run(|app, event|", 1)[1]

    for action in ("window.show()", "window.unminimize()", "window.set_focus()"):
        assert action in show_helper
    assert "tauri::RunEvent::Reopen" in run_handler
    assert "has_visible_windows: false" in run_handler
    reopen_handler = run_handler.split("tauri::RunEvent::Reopen", 1)[1].split("=>", 1)[1]
    assert "show_main_window(app)" in reopen_handler


def test_windows_browser_guard_runs_only_in_release_builds():
    # WebView2 is not reachable from Python, so pin the release-only call that
    # keeps refresh controls available during development.
    source = _ui_source(TAURI_MAIN)

    assert "fn setup_windows_browser_guards" in source
    before_call = source.split("setup_windows_browser_guards(app)?;", 1)[0]
    assert before_call.rstrip().endswith("#[cfg(all(windows, not(debug_assertions)))]")


def test_desktop_manages_the_remote_password_through_the_account_dialog():
    section = _ui_source(REMOTE_ACCESS_SECTION)
    dialog = _ui_source(PASSWORD_DIALOG)

    row = section.split("function RemotePasswordRow", 1)[1].split(
        "export function RemoteAccessSection", 1
    )[0]
    assert "if (!(isTauri && status)) {" in row
    assert "initial={status.passwordPending}" in row
    assert "<RemotePasswordRow status={status} onDone={refreshStatus} />" in section
    assert "{isTauri && isOwner ? null : (" in _ui_source(GENERAL_TAB)
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
    source = _ui_source(APP_PROVIDER)

    # The gate has been renamed once already (showApp -> canMountApp) and gained a second
    # clause, so pin the CONDITION that makes the app wait for auth, not the name in front
    # of it. A rename or a rewrap is a refactor; dropping desktopAuthReady is the regression.
    gate = binding_joining(source, "&&", {'status === "running"', "desktopAuthReady"})
    assert gate, "no binding requires both a running status and desktopAuthReady"
    assert gates_the_markup(
        source, gate
    ), f"{gate} is computed but does not condition the mount in the markup"
    assert "Preparing Unsloth" not in source
    assert "Signing in to desktop session" not in source
    assert "desktopBooting" not in source
    assert "showInteractiveApp" not in source
    assert "<NativeIntentDrain />" in source
    assert "{children}" in source


def test_full_app_layout_uses_its_own_initialized_marker():
    source = _ui_source(APP_PROVIDER)

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
    source = _ui_source(APP_PROVIDER)
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
    source = _ui_source(TITLEBAR)

    # Matched across whitespace, and against the floor rather than one spelling: #11458 made
    # the slot `max(7rem, calc(7rem * var(--ui-space-scale, 1)))` so it grows with the UI font
    # and never drops under the three 30px buttons, and prettier then wrapped the ternary over
    # three lines. Both are the same 7rem at the default scale. A slot that stopped being
    # 7rem-based, or stopped keying off this branch, still fails.
    assert re.search(
        r"showSidebarSurface && !pinned\s*\?\s*"
        r'"(?:7rem|max\(\s*7rem\s*,\s*7rem\s*\))"'
        r"\s*:\s*sidebarWidth",
        source,
    ), "the unpinned sidebar surface no longer sizes the titlebar navigation slot from 7rem"
    assert "style={{ width: titlebarNavigationWidth }}" in source
    assert "left: titlebarNavigationWidth" in source
    assert "<DesktopTitlebarNavigation" in source
    assert "const contentBorderLeft = pinned" in source
    assert ': "0px";' in source

    # Keep the decoration below z-50 modals and outside the z-[70] header.
    assert 'data-slot="window-titlebar-decoration"' in source
    decoration = source.split('data-slot="window-titlebar-decoration"', 1)[1].split("<header", 1)[0]
    assert (
        'className="pointer-events-none absolute inset-x-0 '
        'top-[var(--studio-custom-titlebar-height)] z-[45] h-3"' in decoration
    )
    # The border is always visible.
    assert 'className="absolute top-0 h-px bg-sidebar-border"' in decoration
    # The backing and corner only appear when pinned.
    assert decoration.count("{pinned && (") == 2
    assert 'className="absolute top-0 size-3 -translate-x-px bg-sidebar"' in decoration
    assert (
        'className="absolute top-0 size-3 -translate-x-px rounded-tl-[12px] border-l border-t border-sidebar-border bg-background"'
        in decoration
    )


def test_desktop_titlebar_separates_navigation_from_sidebar_brand():
    titlebar = _ui_source(TITLEBAR)
    sidebar = _ui_source(APP_SIDEBAR)
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
    titlebar = _ui_source(TITLEBAR)
    chat_page = _ui_source(CHAT_PAGE)
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

    # The navigation box's left inset is deliberately not asserted here. Whether that
    # element ends up with one is a computed style: it depends on the tailwind-merge
    # cascade, the important modifier, whether an arbitrary value is valid CSS, whether
    # the class is hoisted into a const or interpolated into a template hole, and
    # whether DesktopTitlebarNavigation applies it from its own className prop. None of
    # that is decidable from this file, and the exact-value form this replaces failed
    # #10321 for retuning 12px to 16px, which is what an alignment pass is for. A
    # computed-style check belongs in a driver that renders the titlebar.
    assert 'isTauri && !isMobile && !pinned && view.mode !== "compare"' in chat_page

    assert "pl-[var(--studio-collapsed-chat-controls-inset,0.75rem)]" in chat_page
    # 188 is the number, not the spelling. It ships as `calc(110px + var(...78px))` so the
    # traffic-light half can be divided by the interface zoom while the content half is
    # not, and asserting the literal string is what broke when that landed. The custom
    # titlebar sets the same var to its own much smaller inset, hence the mac-only filter.
    insets = {
        name: _px(values["--studio-collapsed-chat-controls-inset"])
        for name, values in _chrome_style_blocks(_ui_source(APP_PROVIDER)).items()
        if "--studio-mac-traffic-light-inset" in values
    }
    assert insets, "no style block sets both the traffic-light and collapsed-controls insets"
    assert set(insets.values()) == {188}, insets
    assert 'className="!size-[30px] rounded-[10px] text-muted-foreground"' in chat_page
    assert 'aria-label="New chat"' in chat_page
    new_chat_click = chat_page.index("onClick={handleDesktopNewChat}")
    assert new_chat_click < chat_page.index("<ModelSelector", new_chat_click)


def test_tauri_collapse_removes_the_icon_rail_but_web_keeps_it():
    titlebar = _ui_source(TITLEBAR)
    app_sidebar = _ui_source(APP_SIDEBAR)
    primitive = _ui_source(SIDEBAR_PRIMITIVE)
    navbar = _ui_source(NAVBAR)

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
    # in, so the button box travels with it. The mac-only margin is deliberately not in the
    # sum: translate-y is visual, and the margin already seats the box in the native row.
    navigation = titlebar.split("export function DesktopTitlebarNavigation", 1)[1].split(
        "export function WindowTitlebar", 1
    )[0]
    assert "mt-1" not in navigation
    assert "mt-[var(--studio-titlebar-navigation-margin-top,0px)]" in navigation
    button = _titlebar_nav_button_px(titlebar)
    assert button is not None, "navigation button size no longer readable from buttonClass"
    blocks = _chrome_style_blocks(_ui_source(APP_PROVIDER))
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
    # Read the CONDITION, not the text that spells it. The exact-string form this replaces
    # pinned the inlined expression, so #10706 broke it by hoisting that expression into a
    # named const and giving it a peek exception: a refactor that changed nothing this
    # contract protects, and it left main and every open PR red for a day. What must hold is
    # that a sidebar collapsing to nothing leaves the accessibility tree, and that it goes
    # inert on exactly the same condition, since hidden-but-focusable is the actual bug.
    hidden = attribute_expressions(primitive, "aria-hidden")
    inert = attribute_expressions(primitive, "inert")
    assert len(hidden) == 1 and len(inert) == 1, (hidden, inert)
    assert hidden == inert, (hidden, inert)
    # Asking only that the held-out condition still appears would accept dropping the peek
    # exception with it, and a peeked sidebar is on screen: aria-hidden on a visible panel
    # is the same defect this guards, pointing the other way. So state WHEN the panel leaves
    # the accessibility tree, over every combination of the four inputs, and let any
    # spelling that admits exactly those states pass.
    inputs = ("hasPinMode", "pinned", "collapseToZero", "peeking")
    table = boolean_table(expand_bindings(primitive, hidden[0], stop = inputs), inputs)
    for combination, removed in table.items():
        has_pin_mode, is_pinned, collapses_to_zero, is_peeking = combination
        assert removed == (
            has_pin_mode and not is_pinned and collapses_to_zero and not is_peeking
        ), (combination, hidden[0])


def test_fixed_sheets_start_below_the_custom_titlebar():
    provider = _window_chrome_source(APP_PROVIDER)
    sheet = _ui_source(SHEET)

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
        assert "studio-custom-titlebar-height" not in _ui_source(portalled)


def test_visible_mac_sidebar_header_is_a_drag_region():
    source = _ui_source(APP_SIDEBAR)
    header = source.split("<SidebarHeader", 1)[1].split("</SidebarHeader>", 1)[0]
    drag_region = "data-tauri-drag-region={usesNativeMacTitlebar || undefined}"

    assert drag_region in header
    assert header.index(drag_region) < header.index('"relative z-10 flex items-center')


def test_mac_chat_header_controls_share_the_titlebar_row():
    source = _ui_source(CHAT_PAGE)
    provider = _ui_source(APP_PROVIDER)

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
    source = _ui_source(APP_SIDEBAR)

    assert "group-data-[collapsible=icon]:[&_[data-sidebar=sidebar]]:border-r-0" in source
    assert "top-[var(--studio-mac-titlebar-height,34px)]" not in source


def test_chat_sidebar_rows_are_compact_without_vertical_padding():
    sidebar_source = _ui_source(APP_SIDEBAR)
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


def _resolve_classes(source: str, expression: str, variant: str) -> str | None:
    """`expression` as the class string it yields for `variant`, or None if unreadable.

    Every argument has to resolve. Collecting the quoted literals and ignoring the rest is
    what let the pin's own `cn(..., REVEAL_WITH_OPEN_MENU_PROJECT_CHAT)` read as complete:
    a constant supplying `sidebar-row-action`, an `is-*` modifier or a positioning utility
    would have been invisible, and the button skipped or its reach understated with the
    contract still green.

    Four forms are read, and anything else is None: a literal, `cn(...)` over readable
    arguments, a ternary on `variant === "..."`, and an identifier defined as a const in this
    file, which is resolved recursively. `undefined` and a plain conditional's short-circuit
    contribute nothing, which is what they do.
    """
    expression = expression.strip().rstrip(",").strip()
    if not expression or expression == "undefined":
        return ""
    literal = re.fullmatch(r'"([^"]*)"', expression)
    if literal:
        return literal.group(1)
    call = re.fullmatch(r"cn\((.*)\)", expression, re.S)
    if call:
        parts = []
        for argument in _cn_arguments(call.group(1)):
            piece = _resolve_classes(source, argument, variant)
            if piece is None:
                return None
            parts.append(piece)
        return " ".join(part for part in parts if part)
    branched = re.match(r'\s*variant\s*===\s*"(\w+)"\s*\?', expression, re.S)
    if branched:
        marks = _operators(expression[branched.end() :])
        pairing = next((at for at, token in marks if token == ":"), None)
        if pairing is None:
            return None
        rest = expression[branched.end() :]
        taken = rest[:pairing] if branched.group(1) == variant else rest[pairing + 1 :]
        return _resolve_classes(source, taken, variant)
    identifier = re.fullmatch(r"[A-Za-z_$][\w$]*", expression)
    if identifier:
        definition = re.search(rf"const {re.escape(expression)} =(.*?);\n", source, re.S)
        if not definition:
            return None
        return _resolve_classes(source, definition.group(1), variant)
    return None


def _spread_may_supply(tag: str, attribute: str = "className") -> bool:
    """True when a top-level JSX spread could be supplying or replacing *attribute*.

    A spread's contents are not resolvable here, and both ways it can matter are silent. With
    no explicit `className`, a spread may be the only thing supplying one, and this reader
    returned "" for that tag so `_labelled_actions` skipped the action entirely: a pin handed
    `{...{ className: "sidebar-row-action sidebar-touch-reveal right-40" }}` left the reach
    calculation altogether while the shared options button kept every later assertion
    satisfied. With an explicit `className`, only a spread written AFTER it can override, since
    JSX applies attributes left to right and the last write wins.
    """
    spreads, depth = [], 0
    for index, char in enumerate(tag):
        if char == "{":
            if depth == 0 and re.match(r"\{\s*\.\.\.", tag[index:]):
                spreads.append(index)
            depth += 1
        elif char == "}":
            depth -= 1
    if not spreads:
        return False
    explicit = re.search(rf"(?:^|[\s{{]){re.escape(attribute)}=", tag)
    return explicit is None or max(spreads) > explicit.start()


def _button_classes(source: str, tag: str, variant: str) -> str | None:
    """The classes a `<button>` tag ends up with for `variant`, or None if unreadable."""
    if _spread_may_supply(tag):
        return None
    # A suffixed lookalike is refused, not read as absent. `data-className={cn(...)}` is not a
    # className, so the boundary above correctly declines to read it, but returning "" then
    # said "this button carries no classes" and `_labelled_actions` skipped it as not a row
    # action. The pin can vanish from the reach calculation that way while still rendering,
    # now with none of the classes that position or reveal it, and the shared options button
    # keeps every later assertion satisfied.
    lookalike = re.search(r"[\w-]className=", tag)
    assert not lookalike, (
        f"a button in renderChatSidebarItem carries {lookalike.group(0)!r} rather than a "
        f"className. It renders with none of the classes that string holds, and this guard "
        f"would otherwise read it as a button that simply has no classes: {tag!r}"
    )
    match = re.search(r"(?:^|[\s{])className=(\{.*?\}|\"[^\"]*\")", tag, re.S)
    if not match:
        return ""
    value = match.group(1).strip()
    if value.startswith('"'):
        return value.strip('"')
    return _resolve_classes(source, value[1:-1], variant)


# #11458 made spacing follow the UI font size: every fixed length in the stylesheet and in
# the page markup became `calc(<length> * var(--ui-space-scale, 1))`, and #11459 did the same
# for the dark wash with `--contrast-wash-gain`. Both scales default to 1, so the rendered
# value is unchanged; only the spelling moved. These contracts are written against the
# lengths, so resolve the wrapper back to the length it scales. A real change to the length
# still fails, which is the whole point of reading the value rather than the spelling.
#
# Only `--ui-space-scale` reads back as the length it wraps, and only it. `index.css:316`
# declares `--ui-font-scale: 0.9375` and derives `--ui-space-scale` as the ratio of the two,
# so the spacing scale is 1 by default while the font scale is not: a length respelled onto
# `calc(48px * var(--ui-font-scale, 1))` renders at 45px. Reading the font scale, or
# `var(--anything, 1)` at large, would hand a contract back the literal it asks about while
# the pane had quietly stopped following the interface font size, or moved to a typo such as
# `--ui-spcae-scale` that resolves to nothing and takes the fallback.
_SCALED_LENGTH = re.compile(
    r"calc\(\s*(-?[\d.]+(?:px|rem|em))\s*\*\s*var\(\s*--ui-space-scale\s*,\s*1\s*\)\s*\)"
)

# The gains are not declared at all: `appearance-custom-store.ts` sets them only away from
# the default, so both take the `1` fallback here and a colour reads back as its own amount.
# Which gain, though, is load-bearing. The store gives the edge and the wash different spans,
# so a border scaled by the wash renders a colour its contract does not name as soon as the
# contrast setting moves off default. Surfaces take the wash; borders, rings and outlines
# take the edge; each is resolved only under the gain its own role is entitled to.
_SCALED_AMOUNT = re.compile(
    r"calc\(\s*([\d.]+%?)\s*\*\s*var\(\s*--contrast-(edge|wash)-gain\s*,\s*1\s*\)\s*\)"
)
_COLOUR_ROLE = re.compile(
    r"bg-|background(?:-color)?\s*:|border(?:-color)?\s*:"
    r"|border-|ring-|outline-|--tw-ring-color\s*:"
)


def _gain_owed_to(preceding: str) -> str | None:
    """The gain the colour written after *preceding* is entitled to, if that role is readable."""
    roles = _COLOUR_ROLE.findall(preceding[-200:])
    if not roles:
        return None
    return "wash" if roles[-1].startswith(("bg-", "background")) else "edge"


def _resolve_amount(source: str, match: "re.Match[str]") -> str:
    """A gain-scaled colour amount, read back only where the gain matches the colour's role."""
    if _gain_owed_to(source[: match.start()]) != match.group(2):
        return match.group(0)
    return match.group(1)


# The same move in colour. #11459 respelled the opacity shorthands as a colour function
# whose amount scales with a gain, again defaulting to 1: `foreground/10` became a
# `color-mix` of `--foreground` at 10% with transparent, and `white/[0.06]` an `rgb()` at
# 0.06. Those are the shorthands, written out. Reading them back keeps a contract about the
# colour asking about the colour.
#
# The gain has to be there. A colour that keeps the amount but drops the gain renders the
# same thing at the default setting and nothing like it anywhere else, so the two are not
# interchangeable and only the gain-bearing spelling is read back as the shorthand.
_MIXED_TOKEN = re.compile(
    r"\[color-mix\(in_oklab,\s*var\(--([\w-]+)\)_"
    r"calc\(([\d.]+)%\*var\(--contrast-(edge|wash)-gain,\s*1\)\)\s*,\s*transparent\)\]"
)
_WHITE_ALPHA = re.compile(
    r"\[rgb\(255_255_255_/_calc\(([\d.]+)\*var\(--contrast-(edge|wash)-gain,\s*1\)\)\)\]"
)


def _resolve_mix(source: str, match: "re.Match[str]") -> str:
    """A gain-scaled `color-mix`, read back as its shorthand under its own gain."""
    if _gain_owed_to(source[: match.start()]) != match.group(3):
        return match.group(0)
    amount = match.group(2)
    return f"{match.group(1)}/{amount[:-2] if amount.endswith('.0') else amount}"


def _resolve_white_alpha(source: str, match: "re.Match[str]") -> str:
    """A gain-scaled white lift, read back as its shorthand under its own gain."""
    if _gain_owed_to(source[: match.start()]) != match.group(2):
        return match.group(0)
    return f"white/[{match.group(1)}]"


def _at_default_scale(source: str) -> str:
    """*source* with every scale-wrapped length and gain-wrapped colour read back as itself."""
    resolved = _SCALED_LENGTH.sub(r"\1", source)
    resolved = _MIXED_TOKEN.sub(lambda m: _resolve_mix(resolved, m), resolved)
    resolved = _WHITE_ALPHA.sub(lambda m: _resolve_white_alpha(resolved, m), resolved)
    return _SCALED_AMOUNT.sub(lambda m: _resolve_amount(resolved, m), resolved)


def _ui_source(path) -> str:
    """A checked-in source, read at the default UI scale.

    Every contract in this file is written against fixed lengths. #11458 wrapped those
    lengths in `calc(... * var(--ui-space-scale, 1))` so they follow the UI font size, which
    renders identically at the default scale of 1. Reading through `_at_default_scale` keeps
    each contract asking about the length it was written for instead of the spelling.
    """
    return _at_default_scale(path.read_text(encoding = "utf-8"))


def _window_chrome_source(path) -> str:
    """A checked-in source, read exactly as written.

    Window chrome does not scale with the interface font size. The custom titlebar is 34px
    because the Tauri window decoration is 34px, and `DesktopChromeVarsEffect` only mirrors
    that number onto `<html>` for the portalled sheets to sit below. Reading that mirror
    through `_ui_source` would accept `calc(34px * var(--ui-space-scale, 1))` as 34px while
    the real titlebar stayed put and every fixed sheet slid off it at a non-default UI size,
    so the chrome is read raw and a scale wrapper fails the contract.
    """
    return path.read_text(encoding = "utf-8")


def _spacing_rem(live_css: str) -> float | None:
    """The rem one Tailwind spacing unit is worth, read from the theme's `--spacing`.

    Every `pr-N` here is N of these, while the pin's offset and padding are stated in the
    stylesheet as fixed rem. Assuming 0.25 made the two comparable only by coincidence: set
    `--spacing: 0.20rem` and `pr-14` buys 2.8rem where the pin still needs 3.5, so the action
    overlaps the title while this arithmetic, done in assumed units, says it does not.

    Every declaration has to agree. More than one value means the answer depends on which
    theme block is in force, which this guard does not model.
    """
    stated = {match.group(1) for match in re.finditer(r"--spacing:\s*([\d.]+)rem\s*;", live_css)}
    return float(stated.pop()) / 1 if len(stated) == 1 else None


def _as_spacing_units(cls: str, spacing: float) -> str:
    """`pr-[78px]` as `pr-19.5`, so an arbitrary gutter is compared rather than refused.

    The checks below compare `pr-N`, where N counts Tailwind spacing units of 0.25rem. A row
    that states its touch gutter as an exact pixel value is saying the same thing in another
    spelling, and refusing it made a correct row (#11408's spinner column, `pr-[78px]`) fail a
    guard about a defect it does not have.

    Only px and rem convert, and only on `pr-`: those are the two the arithmetic here is
    defined in. Anything else (`%`, `calc()`, `var()`) resolves against something this cannot
    see, so it is left alone for the refusal below to catch. Left alone rather than dropped,
    which is the point: an unconvertible value must still reach a check that says so.
    """
    match = re.fullmatch(r"((?:\S*:)?pr)-\[(\d+(?:\.\d+)?)(px|rem)\]", cls)
    if not match:
        return cls
    prefix, amount, unit = match.group(1), float(match.group(2)), match.group(3)
    units = amount / 16 / spacing if unit == "px" else amount / spacing
    return f"{prefix}-{units:g}"


def _own_declarations(live_css: str, selector: str) -> str | None:
    """One rule's own body, with any rule nested inside it removed.

    Brace-matched rather than read as `[^}]*`, which stops at the first `}` and so takes in a
    nested rule's selector and declarations while cutting the outer rule short.
    `.sidebar-row-action` has such a nested rule, `.sidebar-touch-reveal`, so the lazy form was
    reading part of a different rule as if it belonged to this one.
    """
    start = re.search(rf"{re.escape(selector)}\s*\{{", live_css)
    return None if not start else _declarations_at(live_css, start.end() - 1)


def _declarations_at(live_css: str, brace: str | int) -> str | None:
    """The body of the rule whose opening brace is at *brace*, minus any nested rule."""
    depth = 0
    for index in range(brace, len(live_css)):
        if live_css[index] == "{":
            depth += 1
        elif live_css[index] == "}":
            depth -= 1
            if depth == 0:
                return re.sub(r"[^{}]*\{[^{}]*\}", " ", live_css[brace + 1 : index])
    return None


def _modifier_rules(live_css: str) -> dict[str, str]:
    """Every `.sidebar-row-action.is-*` rule the stylesheet defines, as its own declarations."""
    rules: dict[str, str] = {}
    for match in re.finditer(r"\.sidebar-row-action\.(is-[\w-]+)\s*\{", live_css):
        body = _declarations_at(live_css, match.end() - 1)
        if body is None:
            continue
        # Joined in source order, not replaced. A selector may appear more than once, and CSS
        # keeps an earlier declaration for any property the later rule does not restate: one
        # rule setting `right: 5rem` followed by another setting only `color` still renders at
        # 5rem. Overwriting recorded the second rule alone, so the offset fell back to the
        # base and the action overlapped the title with this green. Joining also means a
        # property genuinely restated appears twice, which `_sole_measure` then refuses rather
        # than resolving, as it does within a single rule.
        rules[match.group(1)] = f"{rules.get(match.group(1), '')}\n{body}"
    return rules


# What else in the same rule can set the edge this guard measures. A shorthand overrides the
# longhand it contains, so `padding: 0 5rem` beats a parsed `pr-1.5` and `inset: 0 5rem` beats
# a parsed `right`. Reading the longhand and ignoring these reported a reach that does not
# render. Tailwind's own shorthands are listed beside the CSS ones because `@apply p-20` is the
# same statement written another way.
_SHORTHANDS = {
    "pr": (("padding",), ("p", "px", "pe")),
    "pl": (("padding",), ("p", "px", "ps")),
    "right": (("inset",), ("inset", "inset-x")),
}


def _sole_measure(body: str, utility: str, prop: str, spacing: float) -> float | None | str:
    """One rule's value for a measure: the number, None if unreadable, "" if it states none.

    Every declaration is collected, because CSS resolves a repeat to the last and a
    first-match read reports the first. Two readable values are refused rather than resolved:
    which one wins also depends on specificity and on where Tailwind emits the utility, and
    this guard models neither.
    """
    properties, utilities = _SHORTHANDS[utility]
    shorthand = [
        name for name in properties if re.search(rf"(?<![\w-]){re.escape(name)}:", body)
    ] + [name for name in utilities if re.search(rf"(?<![\w-])@?{re.escape(name)}-\S", body)]
    if shorthand:
        # Unreadable rather than absent: the shorthand renders, and falling back to the base
        # rule or reporting the longhand would both describe something that does not.
        return None
    values = _stated_units(body, utility, prop, spacing)
    if not values:
        return ""
    if len(values) > 1 or values[0] is None:
        return None
    return values[0]


def _stated_units(body: str, utility: str, prop: str, spacing: float) -> list[float | None]:
    """Every value this rule states for one measure, in spacing units.

    EVERY one, because CSS resolves a repeated declaration to the last, and a first-match
    search reports the first. `.sidebar-row-action-glyph` gaining a `size-20` after its
    `size-6`, or the base action rule gaining an `@apply pr-20` after its `pr-1.5`, changes
    what renders and left every floor here unmoved. The caller refuses anything but a single
    readable answer rather than picking one, since which wins also depends on specificity and
    on where Tailwind emits the utility, and this guard does not model either.

    None marks a value it cannot read, which must not collapse into "not stated".
    """
    found: list[float | None] = []
    for match in re.finditer(rf"(?<![\w-]){re.escape(utility)}-(\S+?)(?=[\s;]|$)", body):
        raw = match.group(1)
        found.append(float(raw) if re.fullmatch(r"\d+(?:\.\d+)?", raw) else None)
    for match in re.finditer(rf"(?<![\w-]){re.escape(prop)}:\s*([^;]+);", body):
        value = match.group(1).strip()
        rem = re.fullmatch(r"([\d.]+)rem", value)
        px = re.fullmatch(r"([\d.]+)px", value)
        if rem:
            found.append(float(rem.group(1)) / spacing)
        elif px:
            found.append(float(px.group(1)) / 16 / spacing)
        elif value == "0":
            found.append(0.0)
        else:
            found.append(None)
    return found


def _base_row_action_offset(live_css: str, spacing: float) -> float | None:
    """The right edge `.sidebar-row-action` itself sets, in units, or None if unreadable.

    Every action's position is measured from this. It is `right-0` today, so an assumed zero
    was right by luck; and returning on the first `@apply right-*` ignored a later
    `right: 5rem` in the same rule, which is what CSS would render, so the reader went on
    saying zero while every action extended into the title.
    """
    body = _own_declarations(live_css, ".sidebar-row-action")
    if body is None:
        return None
    measure = _sole_measure(body, "right", "right", spacing)
    return None if measure == "" or not isinstance(measure, float) else measure


def _row_action_offsets(live_css: str, base: float, spacing: float) -> dict[str, float | None]:
    """Each `.sidebar-row-action.is-*` modifier, and the right edge it renders with.

    Read from CSS rather than named here, so an action positioned by a modifier this file has
    never heard of is refused instead of being recorded as flush right. A modifier that states
    no edge leaves the base rule's in force, which is what CSS does, so it gets *base* rather
    than zero. None means the rule states one this cannot resolve, and the caller refuses it.
    """
    return {
        name: base
        if (measure := _sole_measure(body, "right", "right", spacing)) == ""
        else (measure if isinstance(measure, float) else None)
        for name, body in _modifier_rules(live_css).items()
    }


def _row_action_left_paddings(
    live_css: str, base: float, spacing: float
) -> dict[str, float | None]:
    """Each modifier, and the LEFT padding it renders with, in units.

    The left padding is inside the button, so it is part of what a tap hits even though it
    shows nothing, and `.sidebar-row-action.sidebar-touch-reveal` makes the button clickable
    on a coarse pointer. The pin sets it to zero for that reason; the shared options button
    keeps the base, which faces the pin rather than the title.
    """
    return {
        name: base
        if (measure := _sole_measure(body, "pl", "padding-left", spacing)) == ""
        else (measure if isinstance(measure, float) else None)
        for name, body in _modifier_rules(live_css).items()
    }


def _row_action_paddings(live_css: str, base: float, spacing: float) -> dict[str, float | None]:
    """Each modifier, and the right padding it renders with, in units.

    The base `pr-1.5` is not what every action gets: the container is justify-end, so its
    right padding decides where the glyph sits, and `.is-unpin-action` overrides it to
    `0.125rem` to close the gap to the options button. Applying the base to every action put
    the pin's reach at 15 when it is 14, and that false floor rejected a sufficient `pr-14`.
    """
    return {
        name: base
        if (measure := _sole_measure(body, "pr", "padding-right", spacing)) == ""
        else (measure if isinstance(measure, float) else None)
        for name, body in _modifier_rules(live_css).items()
    }


def _labelled_actions(
    source: str,
    block: str,
    variant: str,
    offsets: dict[str, float | None],
    paddings: dict[str, float | None],
    left_paddings: dict[str, float | None],
    base_padding: float,
    base_left: float,
    base_offset: float,
    live_css: str,
) -> dict[int, tuple[str, float]]:
    """The row actions one variant renders: label -> whether it is the offset one.

    A row's actions are not all shared: the pin sits inside `{variant === "recent" && (` or
    its project counterpart, while the options button is outside both and renders on every
    row. Counting them together and applying the total to both rows would make an action
    added to one of them require room on the other, failing a change that is correct.

    A gate is read as everything between `{variant === "x" && (` and the parenthesis that
    closes it; a button outside every gate belongs to both rows.
    """
    gates = []
    for match in re.finditer(r'\{\s*variant\s*===\s*"(\w+)"\s*&&\s*\(', block):
        depth = 0
        for index in range(match.end() - 1, len(block)):
            if block[index] == "(":
                depth += 1
            elif block[index] == ")":
                depth -= 1
                if depth == 0:
                    gates.append((match.group(1), match.start(), index))
                    break
    found = {}
    shared = []
    cursor = 0
    for tag in _opening_jsx_tags(block, "<button"):
        at = block.find(tag, cursor)
        cursor = at + 1 if at != -1 else cursor
        classes = _button_classes(source, tag, variant)
        assert classes is not None, (
            f"a button in renderChatSidebarItem carries classes this guard cannot read, so it "
            f"cannot tell whether it is a row action or how far it reaches: {tag!r}"
        )
        # Whole tokens, both here and below. CSS matches a class name exactly, so a substring
        # test answers a different question than the stylesheet does: `sidebar-row-action-glyph`
        # contains `sidebar-row-action` and would enrol a button that no rule positions.
        worn = classes.split()
        if "sidebar-row-action" not in worn:
            continue
        owners = [name for name, start, stop in gates if start <= at <= stop]
        if owners and variant not in owners:
            continue
        # Keyed by where the element sits, not by its label. Keying on aria-label dropped any
        # action that names itself another way, `aria-labelledby` being the ordinary one, and
        # dropping the offset pin took the row's reach down to a single glyph while the pin
        # went on rendering. The label is identity for the message only.
        labels = re.findall(r"aria-label=\{?([^\n]{0,60})", tag)
        name = labels[0] if labels else f"<unlabelled at {at}>"
        # Visible on touch, which is the whole affordance. An action that keeps its row-action
        # class but loses `sidebar-touch-reveal` stays counted, so the gutter still looks
        # right while the button is transparent and inert on a coarse pointer.
        # A whole token for the same reason, and it matters more in this direction. A typo or
        # a rename to something longer, `sidebar-touch-reveal-disabled` being the obvious one,
        # satisfies a substring test while matching no rule: the CSS that reveals the action
        # still exists, so every other check here stays green, and the button goes on being
        # transparent and inert on a coarse pointer.
        assert "sidebar-touch-reveal" in worn, (
            f"the {name} action on the {variant} row no longer carries sidebar-touch-reveal, "
            f"so it is invisible and inert on a coarse pointer however much room the row "
            f"reserves for it (#7276). It carries {worn}"
        )
        # Nothing inline. Every number below comes from the class list and index.css, and an
        # inline style outranks both: `style={{ right: "5rem" }}` or
        # `style={{ paddingRight: "5rem" }}` on the action moves it into the title while the
        # reach here is still computed from the rules it no longer obeys. The row carrier is
        # already refused this for the same reason; the actions needed it too.
        escape = _escapes_the_model(tag)
        assert escape is None, (
            f"the {name} action {escape}, which outranks the classes and the CSS rules this "
            f"guard measures it by, so the reach it computes is not the reach that renders: "
            f"{tag!r}"
        )
        # Where it sits, read from the stylesheet. A `right-*` utility or a modifier the CSS
        # does not define is positioning this guard has not modelled, and recording it as
        # flush right would understate the reach of an action that renders further in.
        # Under any variant, not bare. `[@media(pointer:coarse)]:right-40` positions the action
        # on exactly the path this test is about, and matching only the unqualified spelling
        # let it through: the reach would then be taken from the CSS modifier, or from zero,
        # while the action rendered far into the title on every touch device.
        # `right-*` was the only positioning refused, and it is not the only one that moves
        # the action. A transform is the clearest case: `-translate-x-20` slides the pin five
        # rem toward the title with its `right` untouched, so the computed reach did not
        # change and the contract passed. Anything else that shifts the box horizontally, or
        # sets the same edge by another route, belongs here for the same reason: the reach is
        # derived from `right` and `padding-right` alone, and a utility outside that model
        # renders something this arithmetic does not describe.
        # Padding utilities too. The reach below substitutes the padding read from index.css
        # for every action, so `!pr-20` on the element replaces the number being used while
        # the calculation goes on with the stylesheet's, and the glyph clears the gutter.
        utility = [
            token
            for token in worn
            # The importance marker around `right-*` too. `!right-40` and `md:!right-40`
            # render further in than the stylesheet offset this guard substitutes, and the
            # bare form was the only one matched, so they went through untouched.
            if re.fullmatch(rf"(?:\S*:)?!?right-\S+?!?|{_MOVES_HORIZONTALLY}", token)
            or re.fullmatch(r"(?:\S*:)?!?(?:p|px|pe|pr)-\S+", token)
        ]
        assert not utility, (
            f"the {name} action is positioned with {utility}, which this guard does not model: "
            f"it reads the row's action offsets from index.css, so state the offset there"
        )
        # Plain classes on the action, looked up the same way the row's are. Only `is-*` is
        # resolved through the stylesheet below, and `_escapes_the_model` reads inline styles
        # and utility tokens, so an ordinary class with `.pushed-action { right: 5rem }`
        # behind it moved the pin with nothing here the wiser.
        named = {
            token.rpartition(":")[2].strip("!")
            for token in worn
            if re.fullmatch(r"(?:\S*:)?!?[a-z][\w-]*!?", token) and not token.startswith("is-")
        }
        moved_by = _classes_setting(
            live_css, named - _MODELLED_ACTION_CLASSES, _DECLARES_RIGHT_EDGE
        )
        assert not moved_by, (
            f"the {name} action carries {moved_by}, whose rules in index.css move its right "
            f"edge or change its padding. The reach below is computed from the base rule and "
            f"the is-* modifiers alone, so it would not describe where this action renders"
        )
        modifiers = [token for token in worn if token.startswith("is-")]
        unknown = [token for token in modifiers if token not in offsets]
        assert not unknown, (
            f"the {name} action carries {unknown}, which index.css does not define for "
            f".sidebar-row-action, so this guard cannot tell how far that action reaches"
        )
        # A modifier whose rule sets `right` in a spelling this cannot read is not the same
        # as one that leaves it alone, and recording it as zero would lower the floor while
        # the action rendered further in.
        unreadable = [token for token in modifiers if offsets[token] is None]
        assert not unreadable, (
            f"the {name} action carries {unreadable}, whose right edge index.css states in a "
            f"spelling this guard cannot read. It reads a bare rem value, so state the offset "
            f"that way or teach this guard the other one"
        )
        if not owners:
            shared.append(name)
        # The padding this action actually gets, not the base rule's. A modifier may override
        # it, and `is-unpin-action` does, so crediting every action with `pr-1.5` overstated
        # the pin's reach by a whole spacing unit.
        stated = [token for token in modifiers if token in paddings]
        unreadable_padding = [token for token in stated if paddings[token] is None]
        assert not unreadable_padding, (
            f"the {name} action carries {unreadable_padding}, whose right padding index.css "
            f"states in a spelling this guard cannot read. It reads a bare rem value or an "
            f"@apply pr-N, so state it that way or teach this guard the other one"
        )
        # Last modifier wins is not assumed: more than one stating a padding is an order
        # question this does not adjudicate, so it is refused rather than guessed.
        assert len(stated) <= 1, (
            f"the {name} action carries {stated}, more than one of which sets a right "
            f"padding. Which one applies is a question of source order in index.css, and "
            f"this guard does not adjudicate it: state the padding on one modifier"
        )
        # No modifier at all means the base rule is the whole answer.
        padding = paddings[stated[0]] if stated else base_padding
        # The left padding too, because the button's whole box is what a tap hits and
        # `sidebar-touch-reveal` makes it clickable on a coarse pointer. Measuring only as far
        # as the glyph left 8px of the project row's title under the pin, where a tap pinned
        # the chat instead of opening it.
        stated_left = [token for token in modifiers if token in left_paddings]
        unreadable_left = [token for token in stated_left if left_paddings[token] is None]
        assert not unreadable_left, (
            f"the {name} action carries {unreadable_left}, whose left padding index.css states "
            f"in a spelling this guard cannot read, so how far the button reaches is unknown"
        )
        assert len(stated_left) <= 1, (
            f"the {name} action carries {stated_left}, more than one of which sets a left "
            f"padding, and which applies is a source-order question this does not adjudicate"
        )
        left = left_paddings[stated_left[0]] if stated_left else base_left
        # A modifier that states no edge leaves the base rule's in force, so that is the
        # fallback, not zero.
        shift = max((offsets[token] for token in modifiers), default = base_offset)
        found[at] = (name, shift + padding + left)
    # Both rows carry an action that no `variant === "..."` gate guards, and every assertion
    # below is written about a row that has one. Without this the per-variant pins alone keep
    # both maps non-empty, so deleting the shared options button, or letting it lose
    # `sidebar-row-action`, leaves the floor and the reserve agreeing with each other while
    # the action the gutter was widened for is no longer rendered at all. Structural on
    # purpose: an earlier form of this file keyed actions on `aria-label`, and matching the
    # English "Chat options" would fail the moment the row is translated.
    assert shared, (
        f"the {variant} row renders no ungated .sidebar-row-action: every action it has sits "
        f'inside a `variant === "..."` gate. The shared action is what the gutter on both '
        f"rows is sized for, so if it moved behind a gate say so here, and if it was removed "
        f"the reserved room should shrink with it (#7276)"
    )
    return found


def _opening_jsx_tags(source: str, marker: str) -> list[str]:
    """Every `marker ... >` opening tag in `source`, braces balanced.

    A `>` inside an attribute expression does not end the tag, so depth is tracked rather
    than scanning to the first one.
    """
    tags, start = [], source.find(marker)
    while start != -1:
        depth = 0
        for index in range(start, len(source)):
            char = source[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            elif char == ">" and depth == 0:
                tags.append(source[start : index + 1])
                break
        start = source.find(marker, start + 1)
    return tags


def _operators(text: str) -> list[tuple[int, str]]:
    """Positions of `?`, `:`, `&&` and `||` that are not inside a string or brackets.

    Tailwind's own colons all sit inside a quoted class list or inside `[...]`, so depth and
    quoting are the whole of it. `?.` and `??` are skipped: they are not this grammar.
    """
    found, depth, quoted, index = [], 0, False, 0
    while index < len(text):
        char = text[index]
        if quoted:
            if char == '"':
                quoted = False
            index += 1
            continue
        if char == '"':
            quoted = True
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and (text.startswith("&&", index) or text.startswith("||", index)):
            found.append((index, text[index : index + 2]))
            index += 2
            continue
        elif depth == 0 and char == "?" and text[index + 1 : index + 2] not in (".", "?"):
            found.append((index, "?"))
        elif depth == 0 and char == ":":
            found.append((index, ":"))
        index += 1
    return found


def _branches(argument: str) -> list[tuple[tuple[tuple[str, bool], ...], str]]:
    """Every value one cn() argument can evaluate to, with the conditions that select it.

    The conditions come back as (text, truth) so that two arguments branching on the SAME
    expression cannot be combined into a state neither can be in. That matters here: the
    row's touch gutters live in `showWorkSpinner ? undefined : "...pair..."` while the room
    for the spinner case is stated in a different argument, also on `showWorkSpinner`. A
    product that ignored the correlation would invent a row with neither.
    """
    text = argument.strip()
    marks = _operators(text)
    opened = next((index for index, (_, token) in enumerate(marks) if token == "?"), None)
    if opened is not None:
        nested, closed = 0, None
        for index in range(opened + 1, len(marks)):
            token = marks[index][1]
            if token == "?":
                nested += 1
            elif token == ":":
                if nested == 0:
                    closed = marks[index][0]
                    break
                nested -= 1
        assert closed is not None, f"unbalanced ternary in {argument!r}"
        condition = " ".join(text[: marks[opened][0]].split())
        taken = []
        for part, truth in (
            (text[marks[opened][0] + 1 : closed], True),
            (text[closed + 1 :], False),
        ):
            taken += [
                (((condition, truth),) + constraints, value)
                for constraints, value in _branches(part)
            ]
        return taken
    # `||` binds looser than `&&`, so it splits first: `a || b && "x"` is `a || (b && "x")`,
    # and taking the last operator textually would read it as `(a || b) && "x"` and hand the
    # classes to a row that renders none of them. Within one operator they are
    # left-associative, so the last of those is the outermost: `a && b && "x"` is one
    # condition `a && b` carrying one value. Keeping the head whole rather than splitting it
    # is what lets an identical head elsewhere correlate with this one.
    short = [mark for mark in marks if mark[1] == "||"] or [
        mark for mark in marks if mark[1] == "&&"
    ]
    if short:
        cut, token = short[-1]
        head = " ".join(text[:cut].split())
        tail = text[cut + 2 :].strip()
        if token == "&&":
            return [(((head, True),), tail), (((head, False),), "undefined")]
        return [(((head, True),), head), (((head, False),), tail)]
    return [((), text)]


def _values_are_readable(argument: str) -> bool:
    """True when every value `argument` can evaluate to is a string literal or `undefined`.

    Conditions are not values: in `a && "x"` and `c ? "x" : undefined` only the operands that
    can BECOME the class string matter, so `a` and `c` may be anything.
    """
    return all(
        value.strip() in ("", "undefined") or re.fullmatch(r'"[^"]*"', value.strip())
        for _, value in _branches(argument)
    )


def _tokens(value: str) -> list[str]:
    """The classes a branch value contributes, in order. `undefined` contributes none."""
    value = value.strip()
    return value.strip('"').split() if re.fullmatch(r'"[^"]*"', value) else []


# Utilities that move an element horizontally by a route this file does not model. The reach
# arithmetic is `right` plus `padding-right` and nothing else, so any of these renders
# something it does not describe. `translate-y` is deliberately absent: it moves the element
# vertically and the production spinner uses it, so refusing it would fail a correct row.
_MOVES_HORIZONTALLY = (
    r"(?:\S*:)?(?:-?translate-x|-?translate-(?!y)|inset-x|inset|left|-?mr|-?me)-\S+"
    r"|(?:\S*:)?transform"
)


# The two the reach model already reads in full, so they are not "unmodelled".
_MODELLED_ACTION_CLASSES = {"sidebar-row-action", "sidebar-row-action-glyph"}

_DECLARES_RIGHT_PADDING = (
    r"(?<![\w-])padding(?:-right|-inline-end)?:|@apply[^;]*(?<![\w-])!?(?:p|px|pe|pr)-"
)
# What moves an action's right edge, as a stylesheet declaration rather than a utility token.
_DECLARES_RIGHT_EDGE = (
    _DECLARES_RIGHT_PADDING
    + r"|(?<![\w-])(?:right|inset(?:-inline)?|left|transform|translate|margin(?:-right)?):"
    + r"|@apply[^;]*(?<![\w-])!?(?:right|inset|inset-x|left|-?mr|-?me|-?translate-x)-"
)


@functools.lru_cache(maxsize = 4)
def _css_rules(live_css: str) -> tuple[tuple[str, str], ...]:
    """Every rule in the stylesheet once, as (selector list, own declarations).

    Cached and shared. The per-class scan below re-read the whole file for every class it was
    asked about, five times over, which took about six seconds on its own: more than the other
    thirty-six tests in this file put together.
    """
    rules = []
    for match in re.finditer(r"([^{}]*)\{", live_css):
        body = _declarations_at(live_css, match.end() - 1)
        if body is not None:
            rules.append((match.group(1), body))
    return tuple(rules)


def _classes_setting(live_css: str, classes: set[str], declares: str) -> list[str]:
    """Which of *classes* index.css gives a declaration matching *declares*.

    The gutter and reach comparisons read utilities off the elements, so an ordinary project
    class whose rule sets the same property replaces the number that renders while the
    comparison carries on with the utility's. A coarse-pointer rule doing it with `!important`
    is the worst case: the row still says `pr-16` and the rendered gutter is zero.
    """
    offenders = []
    for name in sorted(classes):
        mentions = re.compile(rf"\.{re.escape(name)}(?![\w-])")
        for selectors, body in _css_rules(live_css):
            if not mentions.search(selectors):
                continue
            # On the element that carries the class, not on a descendant of it.
            # `.sidebar-nav-btn .decorative-child { padding-right: 0 }` styles the child and
            # leaves the row's gutter alone; reading it as the row's own padding would refuse
            # harmless descendant styling. The class has to appear in the LAST compound of
            # some selector in the list, which is the element the rule targets.
            if not any(
                mentions.search(re.split(r"[\s>+~]+", one.strip())[-1])
                for one in selectors.split(",")
                if one.strip()
            ):
                continue
            if re.search(declares, body):
                offenders.append(name)
                break
    return offenders


def _escapes_the_model(tag: str) -> str | None:
    """Why *tag*'s position cannot be read from its classes and index.css, or None.

    Every number this file computes comes from a class list and a stylesheet rule. Three
    things beat both: an inline `style`, a spread that could supply one, and a utility that
    moves the element by a route the `right` plus `padding-right` sum does not model.

    One function because the drift was the actual defect. The carrier, the row actions, the
    spinner's wrapper and the spinner itself each grew these checks separately and each ended
    up with a different subset, so a rule added to one path kept being missing from the next.
    """
    if re.search(r"(?:^|[\s{])style=", tag):
        return "sets an inline style"
    if any(_spread_may_supply(tag, name) for name in ("className", "style")):
        return "takes a spread that may supply a className or a style"
    moved = re.search(_MOVES_HORIZONTALLY, tag)
    return f"is moved by {moved.group(0)!r}" if moved else None


def _touch_spinner_reach(block: str, spacing: float) -> tuple[str, float] | None:
    """How far the working-row spinner reaches into the row on a coarse pointer, in units.

    The row actions are not the only thing the title has to clear. A working row also renders
    a spinner, anchored right and pushed clear of the actions on touch, and the gutter has to
    hold both. Measuring only the actions let the touch gutter drop from the 78px the spinner
    needs to the 64px the pin needs, with the spinner then over the title.

    Read the way everything else here is: the offset off the element that carries it, the
    width off the glyph inside it, and None for anything this cannot resolve.
    """
    # The ternary that mounts a <span>, whatever its condition is called. Matching the name
    # here and correlating on the same literal elsewhere would let the two drift apart.
    gate = next(
        (
            found
            for found in re.finditer(r"\{\s*([A-Za-z_$][\w$]*)\s*\?\s*\(", block)
            if re.match(r"\s*<span", block[found.end() :])
        ),
        None,
    )
    if not gate:
        return None
    depth, end = 0, None
    for index in range(gate.end() - 1, len(block)):
        if block[index] == "(":
            depth += 1
        elif block[index] == ")":
            depth -= 1
            if depth == 0:
                end = index
                break
    if end is None:
        return None
    rendered = block[gate.end() : end]
    # Nothing inline or unmodelled on the element that positions it, for the same reason the
    # actions and the carrier refuse both: `style={{ right: "8rem" }}` outranks the coarse
    # `right-16` this reads, and a transform moves the spinner without touching `right` at
    # all, so the reach reported here would not be the reach that renders.
    positioning = [tag for tag in _opening_jsx_tags(rendered, "<span") if "absolute" in tag]
    # The glyph as well as the wrapper. A transform on the <Spinner> moves what the reader
    # sees while the wrapper's `right-16` and the glyph's `size-3.5`, which is all this
    # measures, stay exactly as they were.
    for tag in positioning + _opening_jsx_tags(rendered, "<Spinner"):
        if _escapes_the_model(tag):
            return None
    offsets = [
        float(match.group(1))
        for match in re.finditer(
            r"\[@media\(pointer:coarse\)\]:right-(\d+(?:\.\d+)?)(?![\w.-])", rendered
        )
    ]
    # The spinner's own width, by the one route this reads. Another width utility on the same
    # element renders wider than `size-3.5` while the regex below goes on reporting 3.5, so
    # any of them makes the measurement unreadable rather than merely different.
    for tag in _opening_jsx_tags(rendered, "<Spinner"):
        if re.search(r"(?<![\w-])!?(?:w|min-w|max-w|basis)-\S", tag):
            return None
        if len(re.findall(r"(?<![\w-])size-\d", tag)) > 1:
            return None
    widths = [
        float(match.group(1))
        for match in re.finditer(
            r"<Spinner[^>]*?(?<![\w-])size-(\d+(?:\.\d+)?)(?![\w.-])", rendered, re.S
        )
    ]
    if len(offsets) != 1 or len(widths) != 1:
        return None
    # The gate's own condition is returned with the measurement, because the caller has to
    # correlate it with the builder's branches. Naming `showWorkSpinner` there and reading it
    # here would let the two drift: spell the gate `Boolean(showWorkSpinner)` in both places
    # and a caller keyed on the literal name finds no spinner rendering and skips them all.
    return gate.group(1), offsets[0] + widths[0]


def _rendered_class_lists(arguments: list[str]) -> list[list[str]]:
    """Every class list the builder can produce, in builder order, one per live branch.

    Combinations that would need one condition to hold two truths at once are dropped, not
    checked: they are not rows anyone can render.
    """
    return [classes for _, classes in _rendered_with_conditions(arguments)]


def _rendered_with_conditions(arguments: list[str]) -> list[tuple[dict[str, bool], list[str]]]:
    """As above, but keeping which conditions each rendering needed.

    The conditions are what tie a rendering to the rest of the row. The spinner is gated on
    the same flag as the row's widest gutter, and without them a check can only ask what SOME
    rendering reserves, not what the rendering that shows the spinner reserves.
    """
    lists: list[tuple[dict[str, bool], list[str]]] = [({}, [])]
    for argument in arguments:
        grown = []
        for constraints, classes in lists:
            for extra, value in _branches(argument):
                merged = dict(constraints)
                if any(merged.setdefault(name, truth) != truth for name, truth in extra):
                    continue
                grown.append((merged, classes + _tokens(value)))
        lists = grown
    return lists


def _cn_arguments(body: str) -> list[str]:
    """`body` split on top-level commas, ignoring those inside brackets or strings."""
    parts, depth, quoted, current = [], 0, False, []
    for char in body:
        if quoted:
            current.append(char)
            if char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    if current:
        parts.append("".join(current))
    return [part for part in parts if part.strip()]


def test_chat_sidebar_row_actions_visible_on_coarse_pointers():
    """unslothai/unsloth#7276: Recents chat kebab must be tappable on iPad."""
    sidebar_source = _ui_source(APP_SIDEBAR)
    css_source = _ui_source(INDEX_CSS)
    assert "renderChatSidebarItem" in sidebar_source
    block = sidebar_source.split("function renderChatSidebarItem", 1)[1].split("\n  function ", 1)[
        0
    ]
    # The split above bounds the block at the next top-level function, and a nested one inside
    # renderChatSidebarItem ends it early. When #11373 added one, the block shrank to the
    # signature and a comment, and every assertion below started reading an empty room. A
    # truncated block must fail as a stale guard, not as a missing affordance.
    assert len(block) > 2000, (
        f"renderChatSidebarItem now yields only {len(block)} characters, so this guard is "
        f"reading a fragment rather than the row. Widen the bound before trusting anything "
        f"it says about the row's classes"
    )
    # Reserved room for the action on touch, where there is no hover to make it appear. The
    # width is not a constant to pin: it was pr-10, and became pr-14 for project rows and
    # pr-16 for recents when the row gained an action. What does not move is that the row
    # already states how much room that action needs, in the padding it applies on HOVER. So
    # the claim is that a coarse pointer gets at least the same gutter, whatever it is.
    #
    # Per VARIANT, not once for the block. The two rows carry their own paddings, so a single
    # search over the whole function is satisfied by the project row on its own and would stay
    # green while recents lost theirs, which is the half of #7276 that was actually reported.
    #
    # An earlier version of this worked out which padding WINS: last in cn order, across
    # arguments, with conditional arguments applying only to their own branch and an
    # important utility beating an ordinary one written after it. Every rule it gained was
    # right and the next one was still missing, because that question is tailwind-merge plus
    # the cascade and a test file should not hold a second copy of either.
    #
    # So it does not decide. Each variant's row states one hover gutter and one coarse
    # padding, both written plainly, and the coarse one must be at least the hover one.
    # Anything else, a second coarse padding anywhere in the row's own cn(), a variant
    # qualifier, an importance marker, is refused as something this guard will not
    # adjudicate. That is stricter than the framework and it is stricter LOUDLY, which is
    # the half that matters: it cannot quietly approve a gutter nobody checked.
    # The row's own builder, by name, and the classes are read only from inside it. Searching
    # the whole function let the same verified pairs be moved to any other cn() call and
    # still satisfy this, while the buttons that render carried no gutter at all.
    # Comments out of the way BEFORE anything is located, not after. An old builder left
    # inside `/* ... */` sits earlier in the function than the live one, so a search over the
    # raw text selected the dead declaration and the gutter analysis then described classes
    # nothing renders, while the carrier check below saw the live `buttonClass` and agreed.
    # `{/* ... */}` is how a prop is commented out in JSX, and it is the spelling that would
    # be used here, so a stripper that only knew `//` left a disabled className reading as a
    # live one. Block form first, then line form.
    applied = re.sub(r"\{?\s*/\*.*?\*/\s*\}?", " ", block, flags = re.S)
    applied = "\n".join(re.sub(r"(?<!:)//.*$", "", line) for line in applied.splitlines())
    builder = re.search(r"const buttonClass = cn\(", applied)
    assert builder, (
        "renderChatSidebarItem no longer builds its row classes in a `const buttonClass = "
        "cn(...)`, so this guard cannot tell which classes reach the row button"
    )
    depth, builder_end = 0, None
    for index in range(builder.end() - 1, len(applied)):
        if applied[index] == "(":
            depth += 1
        elif applied[index] == ")":
            depth -= 1
            if depth == 0:
                builder_end = index
                break
    assert builder_end is not None, "unbalanced `const buttonClass = cn(` in the sidebar"
    # On the row button itself, not merely somewhere in the function. The row also renders an
    # inline rename input and a pin, and handing buttonClass to one of those while the button
    # went without would leave every padding check below describing classes that reach nothing
    # the gutters are measured against.
    carriers = [
        tag
        for tag in _opening_jsx_tags(applied, "<SidebarMenuButton")
        if re.search(r"(?:^|[\s{])className=\{buttonClass\}", tag)
    ]
    # An inline style outranks every Tailwind utility, and none of them is read here.
    # `style={{ paddingRight: 0 }}` on the carrier puts the always-visible touch actions
    # straight back over the title while every gutter assertion below stays green, because
    # only buttonClass is analysed.
    # A spread counts wherever it sits. Ordering only settles `className`, where the explicit
    # attribute beats a spread written before it; it settles nothing about `style`, which no
    # attribute here declares, so `{...rowProps}` with a `style.paddingRight` of 0 overrides
    # every pr-N gutter below from either side of the className.
    escaped = [(tag, _escapes_the_model(tag)) for tag in carriers]
    offending = [(tag, why) for tag, why in escaped if why]
    assert not offending, (
        f"a row carrier {offending[0][1] if offending else ''}, which outranks the pr-N "
        f"gutters this guard compares, so the room it computes is not the room that renders: "
        f"{[tag for tag, _ in offending]!r}"
    )
    assert carriers, (
        "no <SidebarMenuButton> in renderChatSidebarItem receives className={buttonClass}, so "
        "the classes checked below do not reach the row button and say nothing about the row "
        "that renders"
    )
    # Only a spread written AFTER the class, because JSX applies attributes in order and the
    # last write wins: `<SidebarMenuButton {...rowProps} className={buttonClass}>` ends with
    # the explicit one whatever the spread holds. Refusing that shape would fail a refactor
    # that forwards unrelated props, which stops correct work rather than catching anything.
    spreading = [
        tag
        for tag in carriers
        if any(
            match.start() > re.search(r"(?:^|[\s{])className=\{buttonClass\}", tag).start()
            for match in re.finditer(r"\{\s*\.\.\.", tag)
        )
    ]
    assert not spreading, (
        f"the row button spreads props after className={{buttonClass}}, so whether those "
        f"classes survive depends on what the spread holds, which this guard cannot resolve: "
        f"{spreading!r}"
    )
    # Comments first: they hold commas and prose, and splitting arguments around them turns
    # a sentence into an unreadable "value".
    row_classes = "\n".join(
        re.sub(r"(?<!:)//.*$", "", line)
        for line in applied[builder.end() : builder_end].splitlines()
    )
    # Every value the builder contributes has to be readable. An identifier holding a class
    # string is invisible to a scan over quoted literals, so `cn(..., coarseOverride)` would
    # make pr-0 effective while this guard went on reporting the gutter above it. Conditions
    # may be anything; it is the VALUES that have to be literals or undefined.
    unresolved = [
        argument for argument in _cn_arguments(row_classes) if not _values_are_readable(argument)
    ]
    assert not unresolved, (
        f"buttonClass is built from values this guard cannot read: {unresolved}. A class "
        f"string held in an identifier can override the row's gutters without appearing "
        f"here, so keep the row's classes as literals"
    )

    # Every class the builder can contribute, whichever way its conditions fall. Padding is
    # read off these branches rather than off the literals, because a literal found anywhere
    # in the builder says nothing about the rows that do not take it: moving the verified
    # pair into `showWorkSpinner ? "...pair..." : undefined` leaves every ordinary row with
    # no gutter at all while a scan over literals still finds it.
    live_css = _at_default_scale(re.sub(r"/\*.*?\*/", " ", css_source, flags = re.S))
    spacing = _spacing_rem(live_css)
    assert spacing is not None, (
        "index.css does not state one readable --spacing, so this guard cannot convert the "
        "pin's fixed rem offset and padding into the pr-N units the row's gutters are in, and "
        "the two sides of every comparison below would be in different scales"
    )
    rendered = [
        (constraints, [_as_spacing_units(cls, spacing) for cls in rendering])
        for constraints, rendering in _rendered_with_conditions(_cn_arguments(row_classes))
    ]
    renderings = [classes for _, classes in rendered]
    every_class = [cls for rendering in renderings for cls in rendering]
    # Refused across the WHOLE builder, not just the row's own literal. What follows compares
    # pr-N numbers and takes the last one to win, which is tailwind-merge's answer only while
    # nothing here changes the same edge by another route or jumps the queue with `!`. A
    # single `"!pr-0"` argument beats every coarse gutter below it and carries no marker that
    # a search for coarse-pointer strings would find.
    marked = [
        cls for cls in every_class if re.fullmatch(r"(?:\S*:)?!p\w*-\S+|(?:\S*:)?p\w*-\S+!", cls)
    ]
    assert not marked, (
        f"buttonClass sets padding with an importance marker: {marked}. `!` beats an "
        f"ordinary utility written after it, so which gutter the row ends up with stops "
        f"being a question of order, and this guard will not adjudicate it"
    )
    shorthand = [cls for cls in every_class if re.fullmatch(r"(?:\S*:)?(?:p|px|pe)-\S+", cls)]
    assert not shorthand, (
        f"buttonClass sets padding with a shorthand that also moves the right edge: "
        f"{shorthand}. It overrides the pr-N gutters this guard compares, so state the "
        f"row's padding with pr-N alone"
    )
    arbitrary = [
        cls
        for cls in every_class
        if re.search(r"\[padding(?:-right)?:[^\]]*\]|(?<![\w-])p[rxe]?-\[[^\]]*\]", cls)
    ]
    assert not arbitrary, (
        f"buttonClass sets its right padding through an arbitrary value: {arbitrary}. This "
        f"guard compares pr-N gutters and will not work out how that interacts with them: "
        f"state the touch padding as a pr-N utility"
    )

    # Everything below compares pr-N numbers, and tailwind-merge drops an earlier `pr-` for a
    # later one whatever the later one's value is. `pr-px` is a real utility worth one pixel:
    # it replaces the checked gutter and, being unnumbered, was read by nothing here. So any
    # right padding the builder can contribute has to be a number this guard can compare.
    unnumbered = [
        cls
        for cls in every_class
        if re.fullmatch(r"(?:\S*:)?pr-\S+", cls)
        and not re.fullmatch(r"(?:\S*:)?pr-\d+(?:\.\d+)?", cls)
    ]
    assert not unnumbered, (
        f"buttonClass states a right padding this guard cannot compare: {unnumbered}. It "
        f"still replaces the numeric gutters through tailwind-merge, so the comparison below "
        f"would go on reporting a value that no longer renders"
    )

    # And under a qualifier it reads. Being numeric is not enough: the three shapes below are
    # the ones the comparison looks at, and a padding under any other variant was matched by
    # none of them and so left out of the answer entirely while still rendering.
    # `focus:[@media(pointer:coarse)]:pr-0` is the case, effective on a focused touch row and
    # invisible here. Refusing it is the same rule already applied to importance markers and
    # arbitrary values: this guard does not adjudicate what it cannot read.
    readable_padding = re.compile(
        r"pr-\d+(?:\.\d+)?"
        r"|\[@media\(pointer:coarse\)\]:pr-\d+(?:\.\d+)?"
        r"|[\w:\[\]().,=^-]*/(?:project-chat-item|recent-item):pr-\d+(?:\.\d+)?"
    )
    unqualified = sorted(
        {
            cls
            for cls in every_class
            if re.search(r"(?:^|:)pr-\d", cls) and not readable_padding.fullmatch(cls)
        }
    )
    assert not unqualified, (
        f"buttonClass states a right padding under a qualifier this guard does not read: "
        f"{unqualified}. It renders in the state that qualifier names, and the comparison "
        f"below reports the gutters it does read, so the row would be credited with padding "
        f"that state does not have"
    )

    coarse_prefix = r"\[@media\(pointer:coarse\)\]:"
    # And no plain class on the row quietly sets the same edge. Everything below compares
    # `pr-N` utilities; a project class whose rule states a right padding replaces what
    # renders without appearing in that comparison at all.
    named = {
        token.rpartition(":")[2].strip("!")
        for token in every_class
        if re.fullmatch(r"(?:\S*:)?!?[a-z][\w-]*!?", token)
    }
    gutter_classes = _classes_setting(live_css, named, _DECLARES_RIGHT_PADDING)
    assert not gutter_classes, (
        f"the row carries {gutter_classes}, whose rules in index.css set a right padding of "
        f"their own. That is the gutter that renders, and the comparison below reads the pr-N "
        f"utilities instead, so it would credit the row with room it does not have"
    )

    # The spinner too, on the renderings that show it. Everything above measures the row's
    # actions, and a working row also renders a spinner that the title has to clear: it sits
    # further in than the pin on touch, which is why the row reserves 78px there and not the
    # 64 the actions alone would need. Without this the touch gutter could drop to the
    # actions' floor with the spinner left over the title, and every check above would pass.
    spinner = _touch_spinner_reach(applied, spacing)
    assert spinner is not None, (
        "renderChatSidebarItem no longer states the working-row spinner's coarse offset and "
        "size in a form this guard can read, so it cannot tell how much room a working row "
        "has to reserve beyond its actions"
    )
    spinner_gate, spinner_reach = spinner
    # Correlated, or refused. Skipping renderings whose constraints do not mention the gate
    # treats "this is not a working row" and "this guard could not tell" as the same answer,
    # and the second one is how the whole check quietly stops running.
    showing = [
        (constraints, rendering)
        for constraints, rendering in rendered
        if constraints.get(spinner_gate)
    ]
    assert showing, (
        f"no rendering of the row's classes is conditioned on {spinner_gate!r}, the same "
        f"condition that gates the spinner, so this guard cannot tell which rows show one "
        f"and would check the spinner's gutter on none of them"
    )
    for constraints, rendering in showing:
        touch = [
            float(match.group(1))
            for match in (
                re.fullmatch(coarse_prefix + r"pr-(\d+(?:\.\d+)?)", cls) for cls in rendering
            )
            if match
        ]
        plain = [
            float(match.group(1))
            for match in (re.fullmatch(r"pr-(\d+(?:\.\d+)?)", cls) for cls in rendering)
            if match
        ]
        reserved = touch[-1] if touch else (plain[-1] if plain else None)
        assert reserved is not None and reserved >= spinner_reach, (
            f"a working row reserves {reserved} on a coarse pointer, under the "
            f"{spinner_reach} its spinner reaches. The spinner is always visible there, so "
            f"it would sit over the title (#7276)"
        )

    variants = ("project-chat-item", "recent-item")
    # Every rendering is some row, and a rendering that claims no gutter at all was being
    # skipped as "not this variant" by each variant's loop in turn, so wrapping both variants'
    # gutters in the same condition left ordinary rows with no touch padding and nothing
    # checking them. A row is identified by the gutter it claims, so a row that claims none
    # cannot be identified, and that is the thing to refuse rather than to skip.
    for rendering in renderings:
        if not any(
            re.fullmatch(rf"\S*/{re.escape(name)}:pr-\d+(?:\.\d+)?", cls)
            for name in variants
            for cls in rendering
        ):
            raise AssertionError(
                f"buttonClass can render a row that states no action gutter for either "
                f"variant: {rendering}. The row's actions are always visible on touch, so a "
                f"row that reserves nothing puts them over the title (#7276). If this branch "
                f"really renders no action, give it a gutter of its own rather than leaving "
                f"it unidentifiable"
            )

    # A floor under both sides, because the comparison below is relative and reducing the two
    # together satisfies it while reserving nothing usable: pr-0.5 against pr-0.5 passes. The
    # floor is what the row's actions occupy, so it has to count them rather than assume one.
    # A row carries a pin and an options button, and reserving a single glyph for the two of
    # them puts the pin back over the title, which is the regression this exists to stop.
    # Over CSS with its comments removed, for the same reason the TSX reads are: an old rule
    # left inside `/* ... */` sits before the live one and is the one a search finds, so the
    # floor would be measured from a glyph nothing renders.
    glyph_rule = _own_declarations(live_css, ".sidebar-row-action-glyph")
    assert glyph_rule is not None, (
        "index.css no longer has a .sidebar-row-action-glyph rule, so this guard cannot tell "
        "how much room one action needs"
    )
    # Any other route to the glyph's width is refused. `size-*` and `width` are the two this
    # reads; a `min-width`, `max-width`, `inline-size` or an `@apply w-*` in the same rule
    # widens the box while this goes on reporting the original size.
    other_width = re.search(
        r"(?<![\w-])(?:min-width|max-width|inline-size|block-size):"
        r"|@apply[^;]*(?<![\w-])(?:w|min-w|max-w|basis)-",
        glyph_rule,
    )
    assert not other_width, (
        f"the glyph's rule sets its width through {other_width.group(0)!r} as well as the "
        f"size this guard reads, so the box that renders is wider than the one it measures"
    )
    # And at each glyph that renders, not only in the shared rule. The floor is one number
    # taken from `.sidebar-row-action-glyph`, so an instance carrying `min-w-20`, or an inline
    # width, is wider than every action is credited with while this goes on reporting size-6.
    for tag in _opening_jsx_tags(applied, "<span"):
        if not re.search(r"(?<![\w-])sidebar-row-action-glyph(?![\w-])", tag):
            continue
        escape = _escapes_the_model(tag)
        assert escape is None, (
            f"a row action's glyph {escape}, so the box that renders is not the one the floor "
            f"below is measured from: {tag!r}"
        )
        widened = re.findall(r"(?<![\w-])!?(?:w|min-w|max-w|basis|size)-\S+", tag)
        assert not widened, (
            f"a row action's glyph sets its own width with {widened}, overriding the shared "
            f"rule this guard measures, so that action reaches further than the floor says"
        )

    sizes = _stated_units(glyph_rule, "size", "width", spacing)
    assert len(sizes) == 1 and sizes[0] is not None, (
        f"the glyph's size is not one value this guard can read: {sizes}. A second one later "
        f"in the rule is what renders, and every floor below would go on being measured from "
        f"the first, so state it once in a spelling this reads"
    )
    # Read per variant, because each variant's own actions decide its floor: an action added to
    # one row only would otherwise require the other to reserve room for something it does not
    # render, and this guard would fail a correct change.
    glyph_size = sizes[0]
    # The actions do not sit side by side and counting them assumed they did. They are
    # absolutely positioned and one is pushed clear of the other, so what the row has to
    # reserve is how far the furthest one reaches, not how many there are. Each action's
    # offset comes from `_row_action_offsets`, read out of the stylesheet.
    # To the far edge of the GLYPH, which is where the ink stops, but through the padding
    # that positions it. The container is justify-end with its own pr, so the glyph's right
    # edge sits that far inside the container's, and its left edge is offset + pr + size.
    # Leaving the pr out under-measured every row by 1.5, which is how the project row's
    # pr-14 passed while its pin reached 15.
    base_rule = _own_declarations(live_css, ".sidebar-row-action")
    assert base_rule is not None, (
        "index.css no longer has a .sidebar-row-action rule, so this guard cannot tell where "
        "inside its container the glyph sits"
    )
    # Through _sole_measure, so the shorthand refusal reaches this rule too. Reading the
    # longhand directly here was how `padding: 0 5rem` in the base rule went unnoticed.
    base_padding = _sole_measure(base_rule, "pr", "padding-right", spacing)
    assert isinstance(base_padding, float), (
        f"the base action's right padding is not one value this guard can read "
        f"({base_padding!r}). A later declaration, or a shorthand that contains it, is what "
        f"renders, and the shared action would reach further into the title while the floor "
        f"stayed put"
    )
    inner_padding = base_padding
    # The container's `pl` IS part of the reach, and for a while this said otherwise. The
    # argument then was that `pr` decides where the glyph sits while `pl` only extends a
    # transparent box, so only the ink counted. That is right about what is SEEN and wrong
    # about what is HIT: `.sidebar-row-action.sidebar-touch-reveal` is `pointer-events-auto`,
    # so the whole button takes taps, padding included, and #7276 is about the action
    # intercepting the title. The pin's own `padding-left: 0` is what keeps the two answers
    # the same size now, so measuring the full box costs the rows nothing.
    base_left_measure = _sole_measure(base_rule, "pl", "padding-left", spacing)
    assert isinstance(base_left_measure, float), (
        f"the base action's left padding is not one value this guard can read "
        f"({base_left_measure!r}). It sits inside the button, so a tap lands on it, and the "
        f"reach below cannot be computed without it"
    )
    base_left = base_left_measure
    base_offset = _base_row_action_offset(live_css, spacing)
    assert base_offset is not None, (
        "index.css no longer states a right edge for .sidebar-row-action in a spelling this "
        "guard can read. Every action's position is measured from it, so reading it as flush "
        "would understate every reach below by however far the base rule moves them"
    )
    offsets = _row_action_offsets(live_css, base_offset, spacing)
    actions = {
        name: _labelled_actions(
            sidebar_source,
            applied,
            name,
            offsets,
            _row_action_paddings(live_css, inner_padding, spacing),
            _row_action_left_paddings(live_css, base_left, spacing),
            inner_padding,
            base_left,
            base_offset,
            live_css,
        )
        for name in ("project", "recent")
    }
    assert all(actions.values()), (
        f"no labelled row action left for one of the variants, so this guard cannot tell how "
        f"much room that row has to reserve: "
        f"{ {name: len(found) for name, found in actions.items()} }"
    )
    reach = {
        name: max(edge + glyph_size for _, edge in found.values())
        for name, found in actions.items()
    }
    floors = {"project-chat-item": reach["project"], "recent-item": reach["recent"]}

    for variant in variants:
        # Any qualified gutter for this variant, not the hover one alone: hover, an open menu
        # and keyboard focus each state how much room the row's action needs, and each is a
        # state a coarse pointer is permanently in, because there the action is always shown.
        qualified = re.compile(rf"\S*/{re.escape(variant)}:pr-(\d+(?:\.\d+)?)$")
        assert any(qualified.fullmatch(cls) for cls in every_class), (
            f"no {variant} row left that widens its padding to make room for the action, so "
            f"this guard can no longer tell whether the touch case is covered"
        )
        for rendering in renderings:
            claimed = [
                float(match.group(1))
                for match in (qualified.fullmatch(cls) for cls in rendering)
                if match
            ]
            # A rendering that states no gutter for this variant is not this variant's row.
            if not claimed:
                continue
            touch = [
                float(match.group(1))
                for match in (
                    re.fullmatch(coarse_prefix + r"pr-(\d+(?:\.\d+)?)", cls) for cls in rendering
                )
                if match
            ]
            plain = [
                float(match.group(1))
                for match in (re.fullmatch(r"pr-(\d+(?:\.\d+)?)", cls) for cls in rendering)
                if match
            ]
            # Last one wins: same variant, so tailwind-merge keeps the later utility, and a
            # coarse-pointer utility outranks an unqualified one whatever the order, since
            # Tailwind emits variants after base and a media query adds no specificity.
            reserved = touch[-1] if touch else (plain[-1] if plain else None)
            needed = max(claimed)
            # Both sides reduced together satisfies the comparison below and reserves nothing
            # usable, which is the state this test was written against: the action keeps its
            # width whatever the row says, so a row that claims less than one action's worth
            # has not been fixed, it has stopped claiming. The gutter's exact size is still
            # not pinned here, only that it holds at least one action.
            # EVERY state that claims a gutter, not the largest of them. Each qualified class
            # names a state in which the action is visible, and in that state the row reserves
            # exactly that gutter, so one of them dropping to pr-0 removes the reservation
            # there while `max` goes on reporting a sibling's 16. An open menu on a coarse
            # pointer is such a state.
            assert min(claimed) >= floors[variant], (
                f"a {variant} row states an action gutter of {min(claimed)} among {claimed}, "
                f"under the {floors[variant]} its furthest action reaches. The state that "
                f"claims it reserves nothing, whatever the others claim, and the action is "
                f"visible in it (#7276)"
            )
            assert needed >= floors[variant], (
                f"a {variant} row states its action gutter as {claimed}, under the "
                f"{floors[variant]} that its furthest action reaches "
                f"at .sidebar-row-action-glyph's size. Nothing then reserves room for actions "
                f"that still have their width, and the comparison below is satisfied by two "
                f"equally small numbers, which is the overlap this test exists to catch "
                f"rather than a row that has been fixed (#7276)"
            )
            assert reserved is not None and reserved >= needed, (
                f"a {variant} row reserves less room on a coarse pointer than it says its "
                f"action needs: {reserved} against {needed}, on the row that renders as "
                f"{rendering}. The action is always visible on touch, so it needs at least "
                f"the room the hover, open-menu and focus cases already say it needs, or it "
                f"sits over the title (#7276)"
            )
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
    root = _ui_source(ROOT_ROUTE)

    assert re.search(
        r"const isChatLike =\s*isChatRoute \|\| isImagesRoute \|\| isVideoRoute \|\| isAudioRoute;",
        root,
    )
    for page in (IMAGES_PAGE, VIDEO_PAGE):
        shell = _ui_source(page).split('"diffusion-surface', 1)[1].split(">", 1)[0]
        assert "pt-[var(--studio-content-top-inset,0px)]" in shell, page.name


def test_image_page_structural_panes_share_the_container_breakpoint():
    source = _ui_source(IMAGES_PAGE)
    shell = source.split('className="diffusion-surface', 1)[1].split(">", 1)[0]
    section = source.split("Settings column + preview canvas", 1)[1]

    assert "@container" in shell
    assert "@[50rem]:flex-row @[50rem]:overflow-hidden" in section
    assert "@[50rem]:w-[408px]" in section
    assert "md:flex-row" not in section
    # pb-6, not the old pb-20: the action is an in-flow footer now, so the rail no longer
    # reserves 80px for an overlay to sit in. The crossfade into that footer is the
    # -action mask, which is why the two are asserted together -- the small padding is
    # only correct while the fade is there to dissolve the last control into the footer.
    assert "panel-scroll-fade-action" in section
    assert "gap-4 px-10 pt-9 pb-6 @[50rem]:overflow-y-auto" in section
    assert "p-6 px-10 @[50rem]:pt-[60px]" in section
    assert "border-t border-foreground/10 px-10 py-3" in section


def test_audio_page_matches_the_image_rail_header_and_action_footer():
    source = _ui_source(AUDIO_PAGE)
    before, marker, after = source.partition("h-[48px] shrink-0")
    assert marker
    header_opening = before.rsplit('<div className="', 1)[1] + marker + after.split(">", 1)[0]
    header = header_opening + after.split("Below 50rem", 1)[0]
    layout = source.split("Below 50rem", 1)[1]

    assert "grid-cols-[minmax(0,408px)_minmax(13rem,1fr)]" in header_opening
    assert "pointer-events-none" in header_opening
    assert "relative" in header_opening
    assert "z-40" in header_opening
    assert "@[50rem]:border-r" in header
    assert (
        'className="!h-[34px] max-w-full gap-1 overflow-hidden pl-3 pr-1 '
        '@[68rem]:gap-2 @[68rem]:pl-4 @[68rem]:pr-2"' in header
    )
    assert 'triggerLabelClassName="text-ui-14 @[68rem]:text-ui-16"' in header
    assert "grid h-full min-w-0 grid-cols-[1fr_auto]" in header
    assert "@[50rem]:grid-cols-[1fr_auto_1fr]" in header
    assert "col-start-2 justify-self-end pr-3" in header
    assert "@[50rem]:justify-self-center @[50rem]:pr-0" in header
    assert "absolute" not in header.split("<PillTabs", 1)[0]

    assert "@[50rem]:flex-row @[50rem]:overflow-hidden" in layout
    assert "@[50rem]:w-[408px]" in layout
    assert "@[50rem]:border-r @[50rem]:border-b-0" in layout
    assert "gap-4 px-10 pt-9 pb-6 @[50rem]:overflow-y-auto" in layout
    assert 'mode === "speak"' in layout
    assert '"panel-scroll-fade-action"' in layout
    assert '"panel-scroll-fade"' in layout
    assert "relative z-10 flex shrink-0 justify-center px-10 pt-0.5 pb-4" in layout
    assert "btn-float-action" not in layout
    assert "absolute inset-x-0 bottom-0" not in layout
    assert layout.count("p-6 px-10 @[50rem]:pt-[60px]") == 2


def test_image_train_rail_matches_create_and_header():
    source = _ui_source(DIFFUSION_TRAIN_PANEL)
    layout = source.split("overflow-x-hidden: an unset overflow-x", 1)[1]

    assert "@[50rem]:flex-row @[50rem]:overflow-hidden" in layout
    assert "pl-10 @[50rem]:w-[408px]" in layout
    assert "@[50rem]:border-r @[50rem]:border-b-0" in layout
    assert "@container hover-scrollbar" in layout
    assert "@[50rem]:pt-[42px]" in layout
    assert "md:w-[416px]" not in layout


def test_compact_media_link_keeps_accessible_name_and_truncation():
    source = _ui_source(MEDIA_PAGE_LINK)
    button = source.split("<button", 1)[1].split("</button>", 1)[0]

    assert "aria-label={label}" in button
    assert 'cn("min-w-0 truncate", labelClassName)' in button
    assert "arrowClassName" in button


def test_media_page_link_tooltip_drops_below_titlebar_controls():
    """unslothai/unsloth#10226: Images/Video park this link in the top-right header beside
    Windows controls; a top tooltip blocks minimize/maximize/close."""
    source = _ui_source(MEDIA_PAGE_LINK)
    tooltip = source.split("<TooltipContent", 1)[1].split("</TooltipContent>", 1)[0]

    assert 'side="bottom"' in tooltip
    assert "sideOffset={6}" in tooltip


def test_media_page_headers_out_stack_the_mac_drag_region():
    """macOS insets the media pages 0px, so their 48px header overlaps the navbar's 34px drag
    strip: the band must out-stack it yet stay click-through (controls click, gaps drag)."""
    navbar = _ui_source(NAVBAR)

    # The strip to beat: same z-40, but earlier in DOM order.
    assert "pointer-events-none absolute inset-x-0 top-0 z-40 h-[48px]" in navbar
    assert "data-tauri-drag-region" in navbar

    # (page, end of the header band, clickable control groups expected inside it)
    for page, band_end, min_groups in (
        (IMAGES_PAGE, "MediaPageLink", 3),
        (VIDEO_PAGE, "MediaPageLink", 2),
        (AUDIO_PAGE, "PillTabs", 2),
    ):
        source = _ui_source(page)
        # matched on the band's size alone: Images lays its header out as a grid and Video as a
        # flex row, so the stacking contract below is what this pins, not one layout's utilities.
        before, marker, band = source.partition("h-[48px] shrink-0")
        assert marker, page.name
        opening = before.rsplit('<div className="', 1)[1]
        for token in ("pointer-events-none", "relative", "z-40"):
            assert token in opening, (page.name, token)

        band = band.split(band_end, 1)[0]
        # every control group in the band has to opt back in, whatever utilities lay it out:
        # Audio and Images seat their mode pills in a grid cell, Video in a flex row, so
        # matching on the opt-in alone is what keeps this honest across all three.
        groups = re.findall(r'"([^"]*pointer-events-auto[^"]*)"', band)
        assert len(groups) >= min_groups, (page.name, groups)


def test_images_header_tracks_preview_and_preserves_titlebar_controls():
    source = _ui_source(IMAGES_PAGE)
    before, marker, after = source.partition("h-[48px] shrink-0")
    assert marker
    opening = before.rsplit("<div", 1)[1] + marker + after.split(">", 1)[0]
    header = (
        opening + after.split('      {pageMode === "train" ? (\n        <DiffusionTrainPanel', 1)[0]
    )

    assert "const { isMobile, pinned } = useSidebar();" in source
    assert "grid-cols-[minmax(0,408px)_minmax(13rem,1fr)]" in opening
    assert "@[50rem]:border-r" in header
    assert "isMobile" in header and "pl-12" in header
    assert "!pinned && isTauri" in header
    assert "pl-[var(--studio-collapsed-chat-controls-inset,0.75rem)]" in header
    assert (
        'className="!h-[34px] max-w-full gap-1 overflow-hidden pl-3 pr-1 '
        '@[68rem]:gap-2 @[68rem]:pl-4 @[68rem]:pr-2"' in header
    )
    assert 'triggerLabelClassName="text-ui-14 @[68rem]:text-ui-16"' in header
    assert "grid h-full min-w-0 grid-cols-[1fr_auto_auto] gap-2" in header
    assert "@[50rem]:grid-cols-[1fr_auto_1fr] @[50rem]:gap-0" in header
    assert "col-start-2" in header
    assert "col-start-3" in header
    assert 'labelClassName="hidden @[50rem]:inline"' in header
    assert 'arrowClassName="hidden @[50rem]:block"' in header
    assert "absolute" not in header.split("<PillTabs", 1)[0]


def test_a_stopped_repair_update_is_recorded_as_canceled_not_failed():
    """unslothai/unsloth#7793: the support report prints final_status verbatim, so a
    user quitting mid-update must not read as a failed repair."""
    source = _ui_source(TAURI_COMMANDS)
    stopped_arm = source.split("if msg == update::UPDATE_STOPPED", 1)[1].split(
        "return Err(msg);", 1
    )[0]
    # The status argument of the call, so the surrounding comment cannot satisfy this.
    call = stopped_arm.split("finish_repair_group(", 1)[1].split(");", 1)[0]
    assert '"canceled"' in call
    assert '"failed"' not in call


# Every geometry contract above reads its lengths through `_ui_source`, which answers with
# the length at the default scale. That is the same 48px whether the source still scales the
# band or has gone back to a bare `h-[48px]`, so no contract that measures the band can tell
# those apart, and none of them should have to: the scaling is a separate claim and it is
# stated here. Each row is a length one of those contracts measures, with how many times the
# file states it. Unwrap one and this fails, rather than every contract downstream of it
# passing while the layout has stopped following the interface font size.
_LENGTHS_THAT_MUST_KEEP_THE_SCALE = (
    (NAVBAR, "h", "48px", 2),
    (IMAGES_PAGE, "h", "48px", 1),
    (AUDIO_PAGE, "h", "48px", 1),
    (VIDEO_PAGE, "h", "48px", 1),
    (CHAT_PAGE, "h", "48px", 1),
    (IMAGES_PAGE, "pt", "60px", 1),
    (AUDIO_PAGE, "pt", "60px", 2),
    (DIFFUSION_TRAIN_PANEL, "pt", "42px", 2),
)


def test_the_lengths_these_contracts_measure_still_follow_the_ui_scale():
    # The utility has to be the whole utility. `h` is a substring of `min-h`, and a band
    # respelled as `min-h-[calc(48px*...)]` is no longer a fixed band at all: it may grow
    # past the titlebar geometry these contracts measure, while a count of the substring
    # says nothing changed.
    for path, utility, length, expected in _LENGTHS_THAT_MUST_KEEP_THE_SCALE:
        source = path.read_text(encoding = "utf-8")
        boundary = r"(?<![\w-])"
        scaled = len(
            re.findall(
                boundary + re.escape(f"{utility}-[calc({length}*var(--ui-space-scale,1))]"),
                source,
            )
        )
        assert (
            scaled == expected
        ), f"{path.name} states {scaled} scaled {utility}-{length}, not {expected}"
        assert not re.search(
            boundary + re.escape(f"{utility}-[{length}]"), source
        ), f"{path.name} has a bare {utility}-[{length}], which stays put while its text grows"
