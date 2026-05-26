use std::path::PathBuf;
use std::sync::Mutex;

use log::warn;
use serde::{Deserialize, Serialize};
use tauri::Manager;

// Fallback size used only when the live window size can't be read. Keep in sync
// with MIN_WINDOW_WIDTH / MIN_WINDOW_HEIGHT in frontend/src/app/provider.tsx.
const MIN_WIDTH: f64 = 900.0;
const MIN_HEIGHT: f64 = 600.0;

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct WindowState {
    pub width: f64,
    pub height: f64,
    pub maximized: bool,
}

/// Last known non-maximized size plus the current maximized flag, refreshed on
/// every resize. Tracking the normal size separately lets us persist a sensible
/// restore size even when the app exits while maximized (querying the window at
/// that point would return the maximized bounds).
#[derive(Default)]
pub struct WindowStateCache(Mutex<Option<WindowState>>);

fn state_file() -> PathBuf {
    crate::diagnostics::studio_dir().join("window-state.json")
}

fn logical_size(window: &tauri::Window) -> Option<(f64, f64)> {
    let scale = window.scale_factor().ok()?;
    let size = window.inner_size().ok()?.to_logical::<f64>(scale);
    Some((size.width, size.height))
}

pub fn track_resize(window: &tauri::Window) {
    if window.is_minimized().unwrap_or(false) {
        // A minimized window reports 0x0 / tiny bounds on some platforms; keep
        // the last known size instead of clobbering it.
        return;
    }
    let Some(cache) = window.try_state::<WindowStateCache>() else {
        return;
    };
    let mut guard = cache.0.lock().unwrap();
    if window.is_maximized().unwrap_or(false) {
        // Keep the existing normal size; only record that we're maximized.
        guard
            .get_or_insert(WindowState {
                width: MIN_WIDTH,
                height: MIN_HEIGHT,
                maximized: false,
            })
            .maximized = true;
    } else if let Some((width, height)) = logical_size(window) {
        *guard = Some(WindowState {
            width,
            height,
            maximized: false,
        });
    }
    // If the size can't be read, leave the cached value untouched.
}

pub fn save(app: &tauri::AppHandle) {
    let Some(cache) = app.try_state::<WindowStateCache>() else {
        return;
    };
    let Some(state) = *cache.0.lock().unwrap() else {
        return;
    };
    let path = state_file();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(bytes) = serde_json::to_vec_pretty(&state) {
        if let Err(err) = std::fs::write(&path, bytes) {
            warn!("failed to persist window state: {err}");
        }
    }
}

#[tauri::command]
pub fn load_window_state() -> Option<WindowState> {
    let bytes = std::fs::read(state_file()).ok()?;
    serde_json::from_slice(&bytes).ok()
}
