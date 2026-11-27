// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! Microphone consent for the desktop webview.
//!
//! WebView2 raises its own permission dialog, which looks nothing like Unsloth and
//! remembers a deny with no site-settings UI (#9001). The app asks with its own
//! dialog instead and keeps the answer here, so the engine is only ever told what
//! the user chose in the app. A `PermissionRequested` gate answers every microphone
//! request from that stored choice; Edge's prompt never appears.
//!
//! The gate is Windows-only. WKWebView defers to the system permission, which macOS
//! exposes in System Settings, and webkit2gtk asks per session, so both keep their
//! native flow.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use tauri::{AppHandle, Manager};

/// The reset is one profile write on the UI thread. Long enough to outlast a busy thread,
/// short enough that a stuck WebView cannot hang the settings button.
const RESET_TIMEOUT: Duration = Duration::from_secs(10);
const STATE_FILE: &str = "microphone-permission";

const STATE_UNKNOWN: u8 = 0;
const STATE_GRANTED: u8 = 1;
const STATE_DENIED: u8 = 2;

static MICROPHONE_STATE: AtomicU8 = AtomicU8::new(STATE_UNKNOWN);

type Reply = mpsc::Sender<Result<(), String>>;

fn state_name(state: u8) -> &'static str {
    match state {
        STATE_GRANTED => "granted",
        STATE_DENIED => "denied",
        _ => "unknown",
    }
}

fn state_value(name: &str) -> Option<u8> {
    match name {
        "granted" => Some(STATE_GRANTED),
        "denied" => Some(STATE_DENIED),
        "unknown" => Some(STATE_UNKNOWN),
        _ => None,
    }
}

fn state_path(app: &AppHandle) -> Option<PathBuf> {
    app.path()
        .app_config_dir()
        .ok()
        .map(|dir| dir.join(STATE_FILE))
}

fn read_saved_state(app: &AppHandle) -> u8 {
    let Some(path) = state_path(app) else {
        return STATE_UNKNOWN;
    };
    let Ok(text) = std::fs::read_to_string(path) else {
        return STATE_UNKNOWN;
    };
    state_value(text.trim()).unwrap_or(STATE_UNKNOWN)
}

fn write_saved_state(app: &AppHandle, state: u8) -> Result<(), String> {
    let Some(path) = state_path(app) else {
        return Err("Could not determine the app configuration directory".to_string());
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    // Write beside the target and rename so a concurrent reader never sees a
    // half-written file (or an empty one after a crash).
    let temporary = path.with_extension("state.tmp");
    std::fs::write(&temporary, format!("{}\n", state_name(state)))
        .map_err(|error| error.to_string())?;
    std::fs::rename(&temporary, &path).map_err(|error| {
        let _ = std::fs::remove_file(&temporary);
        error.to_string()
    })
}

/// Load the saved answer and wire the gate. Call once, after the main window exists.
pub fn init(app: &AppHandle) {
    MICROPHONE_STATE.store(read_saved_state(app), Ordering::SeqCst);
    install_gate(app);
    // An older build may have left a deny in the WebView2 profile. Clear it once so
    // this build's gate is the only thing that decides.
    clear_profile_answer(app);
}

fn install_gate(app: &AppHandle) {
    let Some(window) = app.get_webview_window("main") else {
        log::warn!("Microphone permission gate skipped: main window is not ready");
        return;
    };
    let _ = window.with_webview(|platform| {
        #[cfg(windows)]
        {
            use webview2_com::Microsoft::Web::WebView2::Win32::{
                ICoreWebView2PermissionRequestedEventArgs3,
                COREWEBVIEW2_PERMISSION_KIND_MICROPHONE,
                COREWEBVIEW2_PERMISSION_KIND_UNKNOWN_PERMISSION,
                COREWEBVIEW2_PERMISSION_STATE_ALLOW, COREWEBVIEW2_PERMISSION_STATE_DENY,
            };
            use webview2_com::PermissionRequestedEventHandler;
            use windows_core::Interface;

            let result = (|| -> windows_core::Result<()> {
                let view = unsafe { platform.controller().CoreWebView2()? };
                let mut token = 0;
                unsafe {
                    view.add_PermissionRequested(
                        &PermissionRequestedEventHandler::create(Box::new(|_, args| {
                            let Some(args) = args else { return Ok(()) };
                            let mut kind = COREWEBVIEW2_PERMISSION_KIND_UNKNOWN_PERMISSION;
                            args.PermissionKind(&mut kind)?;
                            if kind != COREWEBVIEW2_PERMISSION_KIND_MICROPHONE {
                                return Ok(());
                            }
                            // Our dialog is the only prompt the user sees. Keeping the
                            // answer out of the profile means the gate stays in charge.
                            if let Ok(args3) =
                                args.cast::<ICoreWebView2PermissionRequestedEventArgs3>()
                            {
                                let _ = args3.SetSavesInProfile(false);
                            }
                            let allowed = MICROPHONE_STATE.load(Ordering::SeqCst) == STATE_GRANTED;
                            args.SetState(if allowed {
                                COREWEBVIEW2_PERMISSION_STATE_ALLOW
                            } else {
                                COREWEBVIEW2_PERMISSION_STATE_DENY
                            })?;
                            Ok(())
                        })),
                        &mut token,
                    )?;
                }
                Ok(())
            })();
            if let Err(error) = result {
                log::warn!("Microphone permission gate could not be installed: {error}");
            }
        }
        #[cfg(not(windows))]
        let _ = platform;
    });
}

fn clear_profile_answer(app: &AppHandle) {
    let Some(window) = app.get_webview_window("main") else {
        return;
    };
    let Ok(url) = window.url() else { return };
    let Some(origin) = permission_origin(&url) else {
        return;
    };
    let _ = window.with_webview(move |platform| {
        let (tx, _rx) = mpsc::channel();
        if let Err(error) = clear_saved_answer(&platform, &origin, tx) {
            log::warn!("Could not clear the saved microphone answer: {error}");
        }
    });
}

/// What the app currently allows. `unknown` and `denied` both keep the gate closed.
#[tauri::command]
pub fn microphone_permission_state() -> String {
    state_name(MICROPHONE_STATE.load(Ordering::SeqCst)).to_string()
}

/// Persist the answer from the in-app dialog. The next microphone request is gated
/// with it, so the engine never prompts on its own.
#[tauri::command]
pub fn set_microphone_permission(app: AppHandle, state: String) -> Result<(), String> {
    let Some(value) = state_value(&state) else {
        return Err(format!("Unknown microphone permission state: {state}"));
    };
    MICROPHONE_STATE.store(value, Ordering::SeqCst);
    write_saved_state(&app, value)
}

/// Forget the answer and any decision WebView2 saved, so the app can ask again.
#[tauri::command]
pub async fn reset_microphone_permission(
    app: AppHandle,
    webview: tauri::Webview,
) -> Result<(), String> {
    MICROPHONE_STATE.store(STATE_UNKNOWN, Ordering::SeqCst);
    let saved = write_saved_state(&app, STATE_UNKNOWN);
    let url = webview.url().map_err(|error| error.to_string())?;
    let Some(origin) = permission_origin(&url) else {
        log::warn!("Microphone permission reset skipped: window URL {url} has no origin");
        return saved;
    };

    // with_webview hops to the UI thread and the platform call answers later still, so both
    // report back through this channel.
    let (tx, rx) = mpsc::channel();
    let failed = tx.clone();
    webview
        .with_webview(move |platform| {
            if let Err(error) = clear_saved_answer(&platform, &origin, tx) {
                let _ = failed.send(Err(error));
            }
        })
        .map_err(|error| error.to_string())?;

    let profile = tokio::task::spawn_blocking(move || rx.recv_timeout(RESET_TIMEOUT))
        .await
        .map_err(|error| error.to_string())?
        .map_err(|_| "the WebView did not answer the microphone permission reset".to_string())?;
    match (saved, profile) {
        (Err(error), _) => Err(error),
        (Ok(()), Err(error)) => {
            // The gate is authoritative now, so a profile that refuses to clear does
            // not block the user; it is worth a log rather than a failed reset.
            log::warn!("Saved WebView2 microphone answer was not cleared: {error}");
            Ok(())
        }
        (Ok(()), Ok(())) => Ok(()),
    }
}

/// Origin to key the permission on. None when the URL cannot have one, which
/// SetPermissionState rejects with E_INVALIDARG.
pub(crate) fn permission_origin(url: &tauri::Url) -> Option<String> {
    if !matches!(url.scheme(), "http" | "https") {
        return None;
    }
    url.host_str()?;
    let origin = url.origin().ascii_serialization();
    (origin != "null").then_some(origin)
}

#[cfg(windows)]
fn clear_saved_answer(
    webview: &tauri::webview::PlatformWebview,
    origin: &str,
    tx: Reply,
) -> Result<(), String> {
    write_profile_answer(
        webview,
        origin,
        webview2_com::Microsoft::Web::WebView2::Win32::COREWEBVIEW2_PERMISSION_STATE_DEFAULT,
        tx,
    )
}

#[cfg(not(windows))]
fn clear_saved_answer(
    _webview: &tauri::webview::PlatformWebview,
    _origin: &str,
    tx: Reply,
) -> Result<(), String> {
    // Nothing to forget: neither WKWebView nor webkit2gtk keeps a saved answer of its own.
    let _ = tx.send(Ok(()));
    Ok(())
}

#[cfg(windows)]
fn write_profile_answer(
    webview: &tauri::webview::PlatformWebview,
    origin: &str,
    state: webview2_com::Microsoft::Web::WebView2::Win32::COREWEBVIEW2_PERMISSION_STATE,
    tx: Reply,
) -> Result<(), String> {
    use webview2_com::Microsoft::Web::WebView2::Win32::{
        ICoreWebView2Profile4, ICoreWebView2_13, COREWEBVIEW2_PERMISSION_KIND_MICROPHONE,
    };
    use webview2_com::SetPermissionStateCompletedHandler;
    use windows_core::{Interface, HSTRING};

    unsafe {
        let core = webview
            .controller()
            .CoreWebView2()
            .map_err(|error| error.to_string())?;
        // Profile4 carries the permission APIs. A runtime without it also has no way to
        // undo the deny, so say so rather than fail silently.
        let profile = core
            .cast::<ICoreWebView2_13>()
            .and_then(|core| core.Profile())
            .and_then(|profile| profile.cast::<ICoreWebView2Profile4>())
            .map_err(|error| format!("this WebView2 runtime cannot reset permissions: {error}"))?;

        let handler = SetPermissionStateCompletedHandler::create(Box::new(move |result| {
            let _ = tx.send(result.map_err(|error| error.to_string()));
            Ok(())
        }));
        // Bound, not inlined: SetPermissionState reads the string during the call.
        let origin = HSTRING::from(origin);
        profile
            .SetPermissionState(
                COREWEBVIEW2_PERMISSION_KIND_MICROPHONE,
                &origin,
                state,
                &handler,
            )
            .map_err(|error| error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::{permission_origin, state_name, state_value};
    use super::{STATE_DENIED, STATE_GRANTED, STATE_UNKNOWN};

    fn origin_of(url: &str) -> Option<String> {
        permission_origin(&tauri::Url::parse(url).expect("test URL"))
    }

    #[test]
    fn the_packaged_and_dev_pages_both_yield_an_origin() {
        // What the window actually loads: tauri.localhost when bundled, the vite port in
        // dev. The path is dropped, which is what SetPermissionState keys on.
        assert_eq!(
            origin_of("http://tauri.localhost/index.html"),
            Some("http://tauri.localhost".to_string())
        );
        assert_eq!(
            origin_of("https://tauri.localhost/"),
            Some("https://tauri.localhost".to_string())
        );
        assert_eq!(
            origin_of("http://localhost:5173/settings"),
            Some("http://localhost:5173".to_string())
        );
    }

    #[test]
    fn a_url_without_a_usable_origin_is_refused_rather_than_sent() {
        // SetPermissionState rejects these with E_INVALIDARG, so catching them here keeps
        // the failure readable instead of surfacing a bare HRESULT.
        assert_eq!(origin_of("file:///C:/index.html"), None);
        assert_eq!(origin_of("data:text/html,hi"), None);
        assert_eq!(origin_of("about:blank"), None);
    }

    #[test]
    fn permission_states_round_trip() {
        assert_eq!(state_name(STATE_GRANTED), "granted");
        assert_eq!(state_name(STATE_DENIED), "denied");
        assert_eq!(state_name(STATE_UNKNOWN), "unknown");
        assert_eq!(state_value("granted"), Some(STATE_GRANTED));
        assert_eq!(state_value("denied"), Some(STATE_DENIED));
        assert_eq!(state_value("unknown"), Some(STATE_UNKNOWN));
        assert_eq!(state_value("sometimes"), None);
    }
}
