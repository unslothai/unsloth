// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

//! Desktop-only browser controller. Remote documents never receive Studio IPC,
//! tokens, scripts, or the main webview's profile. GTK allocation is app-owned;
//! the other engines use Tauri child bounds. Native work is asynchronous.
use serde::{Deserialize, Serialize};
use std::net::IpAddr;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};
use std::time::Duration;
use tauri::{Manager, Webview, WebviewUrl};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BrowserRect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}
impl BrowserRect {
    fn validate(self) -> Result<Self, String> {
        if [self.x, self.y, self.width, self.height]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0 || *v > 32768.0)
        {
            return Err("Invalid browser rectangle".into());
        }
        Ok(self)
    }
    fn clamp(self, width: f64, height: f64) -> Self {
        let x = self.x.min(width);
        let y = self.y.min(height);
        Self {
            x,
            y,
            width: self.width.min((width - x).max(0.0)),
            height: self.height.min((height - y).max(0.0)),
        }
    }
}

#[derive(Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserSnapshot {
    pub session_id: String,
    pub url: String,
    pub title: String,
    pub loading: bool,
    pub can_go_back: bool,
    pub can_go_forward: bool,
    pub visible: bool,
    pub bounds: BrowserRect,
    pub actual_bounds: Option<BrowserRect>,
    pub popup_url: Option<String>,
    pub error: Option<String>,
    // counts popups so an agent click can tell a new one from a repeated URL.
    #[serde(skip)]
    pub popups: u64,
    // engines queue evaluated scripts without replying until the first commit.
    #[serde(skip)]
    pub committed: bool,
}
#[derive(Clone)]
struct Session {
    snapshot: BrowserSnapshot,
    revision: u64,
}
pub struct BrowserState {
    // Serialize creation/close/native actions without ever blocking GTK's thread.
    pub(crate) operations: tokio::sync::Mutex<()>,
    session: Mutex<Option<Session>>,
    exiting: AtomicBool,
    // one agent request at a time, so marks and clicks never interleave.
    pub(crate) agent: tokio::sync::Mutex<()>,
    // random per run, so pages cannot pre-claim the runtime's global.
    pub(crate) agent_name: String,
}
impl Default for BrowserState {
    fn default() -> Self {
        Self {
            operations: Default::default(),
            session: Default::default(),
            exiting: Default::default(),
            agent: Default::default(),
            agent_name: crate::browser_agent::runtime_name(),
        }
    }
}

/// Window-state 2.x enumerates WebviewWindows, excluding windows with a child.
/// Remove the child before the final exit so its save sees the real main window.
/// Only intercept once, even if the native engine fails to close during shutdown.
pub fn has_open_session(app: &tauri::AppHandle) -> bool {
    app.state::<BrowserState>()
        .session
        .lock()
        .map(|s| s.is_some())
        .unwrap_or(false)
}
pub fn begin_exit(app: &tauri::AppHandle) -> bool {
    let state = app.state::<BrowserState>();
    let open = has_open_session(app);
    open && !state.exiting.swap(true, Ordering::SeqCst)
}
pub async fn prepare_exit(app: &tauri::AppHandle) -> Result<(), String> {
    let state = app.state::<BrowserState>();
    let _operation = state.operations.lock().await;
    close_existing(app).await
}
fn label(session_id: &str) -> Result<String, String> {
    if session_id.is_empty()
        || session_id.len() > 64
        || !session_id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-')
    {
        return Err("Invalid browser session id".into());
    }
    Ok(format!("desktop-browser-{session_id}"))
}

pub(crate) fn is_private_host(host: &str) -> bool {
    let host = host.trim_matches(['[', ']']).to_ascii_lowercase();
    if host == "localhost" || host.ends_with(".localhost") || host.ends_with(".local") {
        return true;
    }
    match host.parse::<IpAddr>() {
        Ok(IpAddr::V4(ip)) => {
            ip.is_loopback()
                || ip.is_private()
                || ip.is_link_local()
                || ip.is_unspecified()
                || ip.is_multicast()
                || ip.is_broadcast()
                || ip.octets()[0] == 0
        }
        Ok(IpAddr::V6(ip)) => {
            ip.is_loopback()
                || ip.is_unspecified()
                || ip.is_unique_local()
                || ip.is_unicast_link_local()
                || ip.is_multicast()
                || ip
                    .to_ipv4_mapped()
                    .is_some_and(|ip| is_private_host(&ip.to_string()))
        }
        _ => false,
    }
}
fn validate_url(raw: &str, test_origin: Option<&str>) -> Result<tauri::Url, String> {
    if raw.len() > 8192 || raw.chars().any(char::is_control) {
        return Err("Invalid browser URL".into());
    }
    let url = tauri::Url::parse(raw).map_err(|_| "Invalid browser URL")?;
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return Err("Only HTTP(S) websites without embedded credentials can open here".into());
    }
    let host = url.host_str().ok_or("Website has no host")?;
    if is_private_host(host) {
        // This exact-origin exception is compiled out of release builds. It is
        // used by the isolated native smoke, never a wildcard for local services.
        let allowed = test_origin
            .and_then(|raw| tauri::Url::parse(raw).ok())
            .is_some_and(|fixture| {
                fixture.host_str() == Some("127.0.0.1")
                    && fixture.port().is_some()
                    && fixture.origin() == url.origin()
            });
        if !allowed {
            return Err("Local services cannot open in the browser pane".into());
        }
    }
    Ok(url)
}
pub(crate) fn website(raw: &str) -> Result<tauri::Url, String> {
    #[cfg(debug_assertions)]
    let fixture = std::env::var("UNSLOTH_BROWSER_TEST_ORIGIN").ok();
    #[cfg(not(debug_assertions))]
    let fixture: Option<String> = None;
    validate_url(raw, fixture.as_deref())
}
fn update(app: &tauri::AppHandle, id: &str, f: impl FnOnce(&mut BrowserSnapshot)) {
    let state = app.state::<BrowserState>();
    if let Ok(mut current) = state.session.lock() {
        if let Some(session) = current.as_mut().filter(|s| s.snapshot.session_id == id) {
            f(&mut session.snapshot);
        }
    };
}
pub(crate) fn current(
    app: &tauri::AppHandle,
    id: &str,
) -> Result<(Webview, BrowserSnapshot), String> {
    let state = app.state::<BrowserState>();
    let snapshot = state
        .session
        .lock()
        .map_err(|_| "Browser state unavailable")?
        .as_ref()
        .filter(|s| s.snapshot.session_id == id)
        .map(|s| s.snapshot.clone())
        .ok_or("Browser session is closed")?;
    let view = app
        .get_webview(&label(id)?)
        .ok_or("Browser webview is closed")?;
    Ok((view, snapshot))
}
async fn on_platform<T: Send + 'static>(
    view: &Webview,
    work: impl FnOnce(&tauri::webview::PlatformWebview) -> Result<T, String> + Send + 'static,
) -> Result<T, String> {
    let (send, receive) = tokio::sync::oneshot::channel();
    view.with_webview(move |platform| {
        let _ = send.send(work(&platform));
    })
    .map_err(|e| e.to_string())?;
    tokio::time::timeout(Duration::from_secs(5), receive)
        .await
        .map_err(|_| "Browser engine did not answer")?
        .map_err(|_| "Browser engine closed")?
}
async fn snapshot(app: &tauri::AppHandle, id: &str) -> Result<BrowserSnapshot, String> {
    let (view, mut snapshot) = current(app, id)?;
    snapshot.url = view.url().map_err(|e| e.to_string())?.to_string();
    let native = on_platform(&view, crate::browser_platform::read).await?;
    snapshot.can_go_back = native.can_go_back;
    snapshot.can_go_forward = native.can_go_forward;
    snapshot.actual_bounds = native.actual_bounds;
    Ok(snapshot)
}
fn bounded_rect(window: &tauri::Window, rect: BrowserRect) -> Result<BrowserRect, String> {
    let rect = rect.validate()?;
    let scale = window.scale_factor().map_err(|e| e.to_string())?;
    let size = window
        .inner_size()
        .map_err(|e| e.to_string())?
        .to_logical::<f64>(scale);
    Ok(rect.clamp(size.width, size.height))
}
async fn bounds(view: &Webview, rect: BrowserRect, visible: bool) -> Result<(), String> {
    #[cfg(target_os = "linux")]
    on_platform(view, move |platform| {
        crate::browser_linux::set_bounds(&platform.inner(), rect, visible)
    })
    .await?;
    #[cfg(not(target_os = "linux"))]
    {
        view.set_bounds(tauri::Rect {
            position: tauri::LogicalPosition::new(rect.x, rect.y).into(),
            size: tauri::LogicalSize::new(rect.width.max(1.0), rect.height.max(1.0)).into(),
        })
        .map_err(|e| e.to_string())?;
        if visible && rect.width >= 1.0 && rect.height >= 1.0 {
            view.show()
        } else {
            view.hide()
        }
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}
async fn close_existing(app: &tauri::AppHandle) -> Result<(), String> {
    let old = app
        .state::<BrowserState>()
        .session
        .lock()
        .map_err(|_| "Browser state unavailable")?
        .clone();
    if let Some(old) = old {
        if let Some(view) = app.get_webview(&label(&old.snapshot.session_id)?) {
            #[cfg(target_os = "linux")]
            on_platform(&view, |platform| {
                crate::browser_linux::detach(&platform.inner());
                Ok(())
            })
            .await?;
            view.close().map_err(|e| e.to_string())?;
        }
        *app.state::<BrowserState>()
            .session
            .lock()
            .map_err(|_| "Browser state unavailable")? = None;
    }
    Ok(())
}

#[tauri::command]
pub async fn desktop_browser_open(
    webview: Webview,
    app: tauri::AppHandle,
    session_id: String,
    url: String,
    rect: BrowserRect,
) -> Result<BrowserSnapshot, String> {
    crate::native_intents::ensure_main_window(&webview)?;
    #[cfg(target_os = "macos")]
    {
        use objc2_foundation::{NSOperatingSystemVersion, NSProcessInfo};
        if !NSProcessInfo::processInfo().isOperatingSystemAtLeastVersion(NSOperatingSystemVersion {
            majorVersion: 14,
            minorVersion: 0,
            patchVersion: 0,
        }) {
            return Err(
                "The isolated browser pane requires macOS 14 or newer; use your external browser"
                    .into(),
            );
        }
    }
    let name = label(&session_id)?;
    let url = website(&url)?;
    let window = webview.window();
    let rect = bounded_rect(&window, rect)?;
    let state = app.state::<BrowserState>();
    let _operation = state.operations.lock().await;
    close_existing(&app).await?;
    let profile = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("desktop-browser/profile-v1");
    std::fs::create_dir_all(&profile).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&profile, std::fs::Permissions::from_mode(0o700))
            .map_err(|e| e.to_string())?;
    }
    *state
        .session
        .lock()
        .map_err(|_| "Browser state unavailable")? = Some(Session {
        snapshot: BrowserSnapshot {
            session_id: session_id.clone(),
            bounds: rect,
            loading: true,
            ..BrowserSnapshot::default()
        },
        revision: 0,
    });
    let nav_app = app.clone();
    let nav_id = session_id.clone();
    let title_app = app.clone();
    let title_id = session_id.clone();
    let load_app = app.clone();
    let load_id = session_id.clone();
    let popup_app = app.clone();
    let popup_id = session_id.clone();
    let download_app = app.clone();
    let download_id = session_id.clone();
    let builder = tauri::webview::WebviewBuilder::new(
        &name,
        WebviewUrl::External("about:blank".parse().unwrap()),
    )
    .data_directory(profile)
    .initialization_script_for_all_frames(crate::browser_agent::init_script(&state.agent_name))
    .on_navigation(move |url| {
        // WebKit calls this for subframes too. Inline documents are ordinary
        // browser content (and required by Turnstile), not Studio origins.
        let allowed =
            matches!(url.as_str(), "about:blank" | "about:srcdoc") || website(url.as_str()).is_ok();
        update(&nav_app, &nav_id, |s| {
            s.error = (!allowed).then(|| "This destination cannot open in the browser pane".into());
        });
        // Only on_page_load owns the main page's URL/loading state: an iframe
        // navigation must not leave the toolbar permanently "loading".
        allowed
    })
    .on_document_title_changed(move |_, title| {
        update(&title_app, &title_id, |s| {
            s.title = title
                .chars()
                .filter(|c| !c.is_control())
                .take(200)
                .collect()
        });
    })
    .on_page_load(move |_, payload| {
        update(&load_app, &load_id, |s| {
            s.url = payload.url().to_string();
            s.loading = payload.event() == tauri::webview::PageLoadEvent::Started;
            s.committed = true;
        });
    })
    .on_new_window(move |url, _| {
        if website(url.as_str()).is_ok() {
            update(&popup_app, &popup_id, |s| {
                s.popup_url = Some(url.to_string());
                s.popups += 1;
            });
        }
        tauri::webview::NewWindowResponse::Deny
    })
    .on_download(move |_, _| {
        update(&download_app, &download_id, |s| {
            s.error = Some("Use Open externally for downloads in this demo".into())
        });
        false
    });
    #[cfg(target_os = "macos")]
    let builder = builder.data_store_identifier(*b"unsloth-browser1");
    // add_child waits for the event loop. Never call it from a sync/main-thread command.
    let created = tokio::task::spawn_blocking(move || {
        window.add_child(
            builder,
            tauri::LogicalPosition::new(rect.x, rect.y),
            tauri::LogicalSize::new(rect.width.max(1.0), rect.height.max(1.0)),
        )
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string());
    let view = match created {
        Ok(view) => view,
        Err(error) => {
            *state
                .session
                .lock()
                .map_err(|_| "Browser state unavailable")? = None;
            return Err(error);
        }
    };
    #[cfg(target_os = "linux")]
    if let Err(error) = on_platform(&view, move |platform| {
        // Keep WebKit's normal permission defaults. Blanket denial also blocks
        // harmless browser APIs such as user-initiated pointer lock; do not
        // inherit Studio's microphone auto-allow handler into external pages.
        crate::browser_linux::attach(platform.inner(), rect)
    })
    .await
    {
        let _ = close_existing(&app).await;
        return Err(error);
    }
    bounds(&view, rect, rect.width >= 1.0 && rect.height >= 1.0).await?;
    update(&app, &session_id, |s| {
        s.visible = rect.width >= 1.0 && rect.height >= 1.0
    });
    view.navigate(url).map_err(|e| e.to_string())?;
    snapshot(&app, &session_id).await
}

#[tauri::command]
pub async fn desktop_browser_close(
    webview: Webview,
    app: tauri::AppHandle,
    session_id: String,
) -> Result<(), String> {
    crate::native_intents::ensure_main_window(&webview)?;
    let state = app.state::<BrowserState>();
    let _operation = state.operations.lock().await;
    let matches = state
        .session
        .lock()
        .map_err(|_| "Browser state unavailable")?
        .as_ref()
        .is_some_and(|s| s.snapshot.session_id == session_id);
    if matches {
        close_existing(&app).await?;
    }
    Ok(())
}
#[tauri::command]
pub async fn desktop_browser_set_bounds(
    webview: Webview,
    app: tauri::AppHandle,
    session_id: String,
    revision: u64,
    rect: BrowserRect,
    visible: bool,
) -> Result<(), String> {
    crate::native_intents::ensure_main_window(&webview)?;
    let rect = bounded_rect(&webview.window(), rect)?;
    let state = app.state::<BrowserState>();
    let _operation = state.operations.lock().await;
    let (view, _) = current(&app, &session_id)?;
    {
        let mut current = state
            .session
            .lock()
            .map_err(|_| "Browser state unavailable")?;
        let session = current.as_mut().ok_or("Browser closed")?;
        if revision <= session.revision {
            return Ok(());
        }
        session.revision = revision;
        session.snapshot.bounds = rect;
        session.snapshot.visible = visible && rect.width >= 1.0 && rect.height >= 1.0;
    }
    bounds(&view, rect, visible).await
}
#[tauri::command]
pub async fn desktop_browser_navigate(
    webview: Webview,
    app: tauri::AppHandle,
    session_id: String,
    url: String,
) -> Result<(), String> {
    crate::native_intents::ensure_main_window(&webview)?;
    let url = website(&url)?;
    let state = app.state::<BrowserState>();
    let _operation = state.operations.lock().await;
    let (view, _) = current(&app, &session_id)?;
    update(&app, &session_id, |s| {
        s.popup_url = None;
        s.error = None;
        s.loading = true;
    });
    view.navigate(url).map_err(|e| e.to_string())
}
#[tauri::command]
pub async fn desktop_browser_action(
    webview: Webview,
    app: tauri::AppHandle,
    session_id: String,
    action: String,
) -> Result<(), String> {
    crate::native_intents::ensure_main_window(&webview)?;
    let state = app.state::<BrowserState>();
    let _operation = state.operations.lock().await;
    let (view, _) = current(&app, &session_id)?;
    match action.as_str() {
        "back" | "forward" | "stop" => {
            on_platform(&view, move |p| crate::browser_platform::action(p, &action)).await
        }
        "reload" => view.reload().map_err(|e| e.to_string()),
        "focus" => view.set_focus().map_err(|e| e.to_string()),
        "dismiss-popup" => {
            update(&app, &session_id, |s| s.popup_url = None);
            Ok(())
        }
        _ => Err("Unknown browser action".into()),
    }
}
#[tauri::command]
pub async fn desktop_browser_snapshot(
    webview: Webview,
    app: tauri::AppHandle,
    session_id: String,
) -> Result<BrowserSnapshot, String> {
    crate::native_intents::ensure_main_window(&webview)?;
    let state = app.state::<BrowserState>();
    let _operation = state.operations.lock().await;
    snapshot(&app, &session_id).await
}
#[tauri::command]
pub async fn desktop_browser_clear_data(
    webview: Webview,
    app: tauri::AppHandle,
    session_id: String,
) -> Result<(), String> {
    crate::native_intents::ensure_main_window(&webview)?;
    let state = app.state::<BrowserState>();
    let _operation = state.operations.lock().await;
    let (view, _) = current(&app, &session_id)?;
    view.clear_all_browsing_data().map_err(|e| e.to_string())?;
    view.navigate(website("https://example.com")?)
        .map_err(|e| e.to_string())
}
