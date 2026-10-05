// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! Clearing a saved microphone decision so the next request can ask again.
//!
//! WebView2 saves "Don't allow" in its profile and ships no site-settings UI, so on Windows
//! one accidental deny blocked dictation for good (#9001): the profile survives reinstalls,
//! and the request never reaches the OS, so Windows privacy settings never list the app to
//! toggle either. Resetting to DEFAULT erases the saved answer and nothing more, so the next
//! getUserMedia prompts exactly as the first one did.
//!
//! Only Windows has this trap. WKWebView defers to the system permission, which macOS
//! exposes in System Settings, and webkit2gtk asks per session, so both are no-ops here.

use std::sync::mpsc;
use std::time::Duration;

/// The reset is one profile write on the UI thread. Long enough to outlast a busy thread,
/// short enough that a stuck WebView cannot hang the settings button.
const RESET_TIMEOUT: Duration = Duration::from_secs(10);

type Reply = mpsc::Sender<Result<(), String>>;

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

/// Forget a saved microphone decision for this window's origin.
///
/// Never grants access: DEFAULT restores the unanswered state, so consent still comes from
/// the prompt the next request raises.
#[tauri::command]
pub async fn reset_microphone_permission(window: tauri::WebviewWindow) -> Result<(), String> {
    let url = window.url().map_err(|error| error.to_string())?;
    let origin =
        permission_origin(&url).ok_or_else(|| format!("no origin to reset in window URL {url}"))?;

    // with_webview hops to the UI thread and the platform call answers later still, so both
    // report back through this channel.
    let (tx, rx) = mpsc::channel();
    let failed = tx.clone();
    window
        .with_webview(move |webview| {
            if let Err(error) = clear_saved_answer(&webview, &origin, tx) {
                let _ = failed.send(Err(error));
            }
        })
        .map_err(|error| error.to_string())?;

    tokio::task::spawn_blocking(move || rx.recv_timeout(RESET_TIMEOUT))
        .await
        .map_err(|error| error.to_string())?
        .map_err(|_| "the WebView did not answer the microphone permission reset".to_string())?
}

#[cfg(windows)]
fn clear_saved_answer(
    webview: &tauri::webview::PlatformWebview,
    origin: &str,
    tx: Reply,
) -> Result<(), String> {
    use webview2_com::Microsoft::Web::WebView2::Win32::{
        ICoreWebView2Profile4, ICoreWebView2_13, COREWEBVIEW2_PERMISSION_KIND_MICROPHONE,
        COREWEBVIEW2_PERMISSION_STATE_DEFAULT,
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
                COREWEBVIEW2_PERMISSION_STATE_DEFAULT,
                &handler,
            )
            .map_err(|error| error.to_string())
    }
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

#[cfg(test)]
mod tests {
    use super::permission_origin;

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
}
