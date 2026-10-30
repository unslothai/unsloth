//! The small always-on-top overlay that shows dictation state while the user is
//! typing into another app. It is created lazily, positioned on the monitor under
//! the cursor, and never takes focus (Windows' `WS_EX_NOACTIVATE` via
//! `focusable(false)`), so the caret stays in the app being dictated into.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use log::warn;
use serde::Serialize;
use tauri::{
    AppHandle, Emitter, Manager, PhysicalPosition, Position, WebviewUrl, WebviewWindow,
    WebviewWindowBuilder,
};

const HUD_LABEL: &str = "voice-hud";
/// The pill itself is compact, while the transparent window gives CSS shadows and
/// live transcript ticker text breathing room without clipping.
const HUD_WIDTH: f64 = 360.0;
const HUD_HEIGHT: f64 = 72.0;
/// Distance from the top or bottom of the monitor work area to the window edge.
const WORK_AREA_MARGIN: f64 = 16.0;

#[derive(Clone, Default, Serialize)]
pub struct HudEvent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sublabel: Option<String>,
}

#[derive(Clone, Default)]
struct HudSnapshot {
    state: String,
    level: f32,
    label: Option<String>,
    sublabel: Option<String>,
}

/// Last announced phase, mic level, and localized text, replayed when the overlay loads late.
static SNAPSHOT: LazyLock<Mutex<HudSnapshot>> = LazyLock::new(|| Mutex::new(HudSnapshot::default()));
/// Whether the overlay window is currently shown for an active session. Prevents
/// repositioning the window across monitors on every phase transition.
static HUD_VISIBLE: AtomicBool = AtomicBool::new(false);
/// Whether the initial webview load replay has already been scheduled.
static HUD_BOOTSTRAPPED: AtomicBool = AtomicBool::new(false);
/// `true` places the overlay near the top of the active monitor's work area; `false` near the bottom.
static HUD_TOP: AtomicBool = AtomicBool::new(false);

pub fn set_position_mode(app: &AppHandle, position_mode: Option<&str>) {
    let top = matches!(position_mode, Some("top"));
    let changed = HUD_TOP.swap(top, Ordering::SeqCst) != top;
    if changed && HUD_VISIBLE.load(Ordering::SeqCst) {
        let handle = app.clone();
        let _ = app.clone().run_on_main_thread(move || {
            if let Some(window) = handle.get_webview_window(HUD_LABEL) {
                position(&window);
            }
        });
    }
}

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
            // Focusable is false (WS_EX_NOACTIVATE), so clicking the cancel button on
            // the HUD pill never steals focus from the application being dictated into.
            let _ = window.set_ignore_cursor_events(false);
            Some(window)
        }
        Err(error) => {
            warn!("Voice typing overlay could not be created: {error}");
            None
        }
    }
}

/// Build the overlay window ahead of the first press.
///
/// Creating a webview is the expensive part of the first show, and that cost
/// currently lands between the shortcut and the recording indicator. Built hidden,
/// so nothing changes on screen; a failure is the same as never having been
/// prewarmed, and `show` reports it if it still cannot build the window.
pub fn prewarm(app: &AppHandle) {
    let handle = app.clone();
    let _ = app.clone().run_on_main_thread(move || {
        let _ = existing(&handle);
    });
}

/// Show the overlay without ever taking focus from the app being dictated into.
/// Only positions and shows the window on transition from hidden -> visible so
/// mid-session phase transitions (`preparing -> recording -> transcribing`) never
/// jump across monitors or spawn redundant replay threads.
pub fn show(app: &AppHandle) {
    if HUD_VISIBLE.swap(true, Ordering::SeqCst) {
        return;
    }
    let handle = app.clone();
    let _ = app.clone().run_on_main_thread(move || {
        // The closure is queued: by the time it runs the session may already have
        // ended (e.g. cancelled before the window existed) — then stay hidden.
        if !HUD_VISIBLE.load(Ordering::SeqCst) {
            return;
        }
        let Some(window) = existing(&handle) else {
            return;
        };
        position(&window);
        let _ = window.show();
    });

    // The very first show can race the overlay page's load if prewarm hasn't finished;
    // replay the snapshot once after 250ms and 650ms.
    if !HUD_BOOTSTRAPPED.swap(true, Ordering::SeqCst) {
        let replay = app.clone();
        std::thread::spawn(move || {
            for delay_ms in [250, 400] {
                std::thread::sleep(Duration::from_millis(delay_ms));
                emit_snapshot(&replay);
            }
        });
    }
}

pub fn hide(app: &AppHandle) {
    HUD_VISIBLE.store(false, Ordering::SeqCst);
    let handle = app.clone();
    let _ = app.clone().run_on_main_thread(move || {
        if let Some(window) = handle.get_webview_window(HUD_LABEL) {
            let _ = window.hide();
        }
    });
}

pub fn emit_state(
    app: &AppHandle,
    state: &str,
    label: Option<String>,
    sublabel: Option<String>,
) {
    {
        let mut snapshot = SNAPSHOT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        snapshot.state = state.to_string();
        if label.is_some() {
            snapshot.label = label.clone();
        }
        if sublabel.is_some() || state == "preparing" || state == "idle" {
            snapshot.sublabel = sublabel.clone();
        }
        // A late-loading overlay replays the level too; a stale level from the
        // previous session would show full bars before the first frame arrives.
        if state == "preparing" || state == "idle" {
            snapshot.level = 0.0;
        }
    }
    if let Some(window) = app.get_webview_window(HUD_LABEL) {
        let _ = window.emit(
            "voice-typing:hud",
            HudEvent {
                state: Some(state.to_string()),
                level: None,
                label,
                sublabel,
            },
        );
    }
}

pub fn emit_level(app: &AppHandle, level: f32) {
    {
        let mut snapshot = SNAPSHOT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        snapshot.level = level;
    }
    if let Some(window) = app.get_webview_window(HUD_LABEL) {
        let _ = window.emit(
            "voice-typing:hud",
            HudEvent {
                state: None,
                level: Some(level),
                label: None,
                sublabel: None,
            },
        );
    }
}

fn emit_snapshot(app: &AppHandle) {
    let Some(window) = app.get_webview_window(HUD_LABEL) else {
        return;
    };
    let snap = {
        let snapshot = SNAPSHOT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        snapshot.clone()
    };
    if snap.state.is_empty() {
        return;
    }
    let _ = window.emit(
        "voice-typing:hud",
        HudEvent {
            state: Some(snap.state),
            level: Some(snap.level),
            label: snap.label,
            sublabel: snap.sublabel,
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
    let margin = (WORK_AREA_MARGIN * scale).round() as i32;
    let work_area = monitor.work_area();
    let area = &work_area.position;
    let size = &work_area.size;
    let x = area.x + (size.width as i32 - width) / 2;
    let y = if HUD_TOP.load(Ordering::SeqCst) {
        area.y + margin
    } else {
        area.y + size.height as i32 - height - margin
    };
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
