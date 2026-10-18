//! The small always-on-top overlay that shows dictation state while the user is
//! typing into another app. It is created lazily, positioned on the monitor under
//! the cursor, and never takes focus (Windows' `WS_EX_NOACTIVATE` via
//! `focusable(false)`), so the caret stays in the app being dictated into.

use std::sync::atomic::Ordering;
use std::sync::Mutex;
use std::time::Duration;

use log::warn;
use serde::Serialize;
use tauri::{
    AppHandle, Emitter, Manager, PhysicalPosition, Position, WebviewUrl, WebviewWindow,
    WebviewWindowBuilder,
};

const HUD_LABEL: &str = "voice-hud";
const HUD_WIDTH: f64 = 224.0;
const HUD_HEIGHT: f64 = 56.0;
/// Distance from the bottom of the monitor to the overlay's bottom edge.
const BOTTOM_MARGIN: f64 = 88.0;

#[derive(Clone, Serialize)]
struct HudEvent {
    #[serde(skip_serializing_if = "Option::is_none")]
    state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    level: Option<f32>,
}

/// Last announced phase and mic level, replayed when the overlay page loads late.
static SNAPSHOT: Mutex<(String, f32)> = Mutex::new((String::new(), 0.0));

fn existing(app: &AppHandle) -> Option<WebviewWindow> {
    if let Some(window) = app.get_webview_window(HUD_LABEL) {
        return Some(window);
    }
    let built = WebviewWindowBuilder::new(app, HUD_LABEL, WebviewUrl::App("voice-hud.html".into()))
        .title("Unsloth voice typing")
        .inner_size(HUD_WIDTH, HUD_HEIGHT)
        .resizable(false)
        .decorations(false)
        .transparent(true)
        .shadow(false)
        .always_on_top(true)
        .skip_taskbar(true)
        .focused(false)
        .focusable(false)
        .visible(false)
        .build();
    match built {
        Ok(window) => {
            // Passive indicator: clicks pass through to whatever is underneath.
            let _ = window.set_ignore_cursor_events(true);
            Some(window)
        }
        Err(error) => {
            warn!("Voice typing overlay could not be created: {error}");
            None
        }
    }
}

/// Show the overlay without ever taking focus from the app being dictated into.
pub fn show(app: &AppHandle) {
    let handle = app.clone();
    let _ = app.clone().run_on_main_thread(move || {
        // The closure is queued: by the time it runs the session may already have
        // ended (e.g. cancelled before the window existed) — then stay hidden.
        if !super::SESSION_ACTIVE.load(Ordering::SeqCst) {
            return;
        }
        let Some(window) = existing(&handle) else {
            return;
        };
        position(&window);
        let _ = window.show();
    });

    // The first show of a session can race the overlay page's load, because the
    // window is created lazily; re-announce the snapshot once the page is up.
    let replay = app.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(700));
        emit_snapshot(&replay);
    });
}

pub fn hide(app: &AppHandle) {
    if let Some(window) = app.get_webview_window(HUD_LABEL) {
        let _ = window.hide();
    }
}

pub fn emit_state(app: &AppHandle, state: &str) {
    {
        let mut snapshot = SNAPSHOT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        snapshot.0 = state.to_string();
    }
    if let Some(window) = app.get_webview_window(HUD_LABEL) {
        let _ = window.emit(
            "voice-typing:hud",
            HudEvent {
                state: Some(state.to_string()),
                level: None,
            },
        );
    }
}

pub fn emit_level(app: &AppHandle, level: f32) {
    {
        let mut snapshot = SNAPSHOT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        snapshot.1 = level;
    }
    if let Some(window) = app.get_webview_window(HUD_LABEL) {
        let _ = window.emit(
            "voice-typing:hud",
            HudEvent {
                state: None,
                level: Some(level),
            },
        );
    }
}

fn emit_snapshot(app: &AppHandle) {
    let Some(window) = app.get_webview_window(HUD_LABEL) else {
        return;
    };
    let (state, level) = {
        let snapshot = SNAPSHOT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        (snapshot.0.clone(), snapshot.1)
    };
    if state.is_empty() {
        return;
    }
    let _ = window.emit(
        "voice-typing:hud",
        HudEvent {
            state: Some(state),
            level: Some(level),
        },
    );
}

fn position(window: &WebviewWindow) {
    let monitor = cursor_position()
        .and_then(|(x, y)| window.app_handle().monitor_from_point(x, y).ok().flatten())
        .or_else(|| window.primary_monitor().ok().flatten());
    let Some(monitor) = monitor else {
        return;
    };
    let scale = monitor.scale_factor();
    let width = (HUD_WIDTH * scale).round() as i32;
    let height = (HUD_HEIGHT * scale).round() as i32;
    let margin = (BOTTOM_MARGIN * scale).round() as i32;
    let area = monitor.position();
    let size = monitor.size();
    let x = area.x + (size.width as i32 - width) / 2;
    let y = area.y + size.height as i32 - height - margin;
    let _ = window.set_position(Position::Physical(PhysicalPosition::new(x, y)));
}

/// Physical point of the mouse cursor, so the overlay follows the active monitor.
#[cfg(target_os = "windows")]
fn cursor_position() -> Option<(f64, f64)> {
    use windows_sys::Win32::Foundation::POINT;
    use windows_sys::Win32::UI::WindowsAndMessaging::GetCursorPos;

    let mut point = POINT { x: 0, y: 0 };
    if unsafe { GetCursorPos(&mut point) } != 0 {
        Some((point.x as f64, point.y as f64))
    } else {
        None
    }
}

#[cfg(not(target_os = "windows"))]
fn cursor_position() -> Option<(f64, f64)> {
    None
}
