use super::{config::PillConfig, PillState, ASK_WINDOW_LABEL};
use tauri::{AppHandle, State, WebviewWindow};

fn ensure_main_window(window: &WebviewWindow) -> Result<(), String> {
    if window.label() == "main" {
        Ok(())
    } else {
        Err("Config commands are only available to the main window.".to_string())
    }
}

fn ensure_ask_window(window: &WebviewWindow) -> Result<(), String> {
    if window.label() == ASK_WINDOW_LABEL {
        Ok(())
    } else {
        Err("Ask commands are only available to the ask window.".to_string())
    }
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PillStatus {
    pub supported: bool,
    pub enabled: bool,
    pub hotkey: String,
    pub excluded_apps: Vec<String>,
}

fn make_status(config: PillConfig) -> PillStatus {
    PillStatus {
        supported: cfg!(target_os = "macos"),
        enabled: config.enabled,
        hotkey: config.ask_hotkey,
        excluded_apps: config.excluded_apps,
    }
}

#[tauri::command]
pub fn pill_status(
    window: WebviewWindow,
    state: State<'_, PillState>,
) -> Result<PillStatus, String> {
    ensure_main_window(&window)?;
    Ok(make_status(state.config.lock().unwrap().clone()))
}

#[tauri::command]
pub fn pill_set_config(
    app: AppHandle,
    window: WebviewWindow,
    state: State<'_, PillState>,
    mut config: PillConfig,
) -> Result<PillStatus, String> {
    ensure_main_window(&window)?;
    // The UI never edits hotkeys; preserve the stored values.
    {
        let current = state.config.lock().unwrap();
        config.hotkey = current.hotkey.clone();
        config.ask_hotkey = current.ask_hotkey.clone();
    }
    *state.config.lock().unwrap() = config.clone();
    persist_and_apply(&app, &config)?;
    Ok(make_status(config))
}

// The server-port event is a one-shot broadcast the hidden webview can miss
// while it is still loading; this lets it pull the current port instead.
#[tauri::command]
pub fn pill_server_port(
    window: WebviewWindow,
    backend: State<'_, crate::process::BackendState>,
) -> Result<Option<u16>, String> {
    ensure_ask_window(&window)?;
    Ok(backend.lock().unwrap().owned_backend_port())
}

#[tauri::command]
pub fn ask_hide(app: AppHandle, window: WebviewWindow) -> Result<(), String> {
    ensure_ask_window(&window)?;
    hide_ask(&app, &window);
    Ok(())
}

#[tauri::command]
pub fn ask_resize(window: WebviewWindow, width: f64, height: f64) -> Result<(), String> {
    ensure_ask_window(&window)?;
    window
        .set_size(tauri::LogicalSize::new(width.max(320.0), height.max(48.0)))
        .map_err(|e| format!("Failed to resize ask window: {e}"))
}

#[cfg(target_os = "macos")]
fn persist_and_apply(app: &AppHandle, config: &PillConfig) -> Result<(), String> {
    super::engine::apply_config(app, config)
}

#[cfg(not(target_os = "macos"))]
fn persist_and_apply(app: &AppHandle, config: &PillConfig) -> Result<(), String> {
    super::config::save_for_app(app, config)
}

#[cfg(target_os = "macos")]
fn hide_ask(app: &AppHandle, window: &WebviewWindow) {
    super::monitor::hide_ask(app, window);
}

#[cfg(not(target_os = "macos"))]
fn hide_ask(_app: &AppHandle, window: &WebviewWindow) {
    let _ = window.hide();
}
