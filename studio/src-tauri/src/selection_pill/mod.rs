pub mod config;
pub mod geometry;

#[cfg(target_os = "macos")]
mod engine;
#[cfg(target_os = "macos")]
mod monitor;
#[cfg(target_os = "macos")]
mod panel;

pub mod commands;

use config::PillConfig;
use std::sync::Mutex;

pub const ASK_WINDOW_LABEL: &str = "ask";
pub const EVENT_ASK_SHOW: &str = "ask://show";
pub const EVENT_ASK_HIDE: &str = "ask://hide";

pub struct PillState {
    pub config: Mutex<PillConfig>,
}

pub fn new_pill_state() -> PillState {
    PillState {
        config: Mutex::new(PillConfig::default()),
    }
}

#[cfg(target_os = "macos")]
pub fn init(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    engine::init(app)
}

/// Hide the ask panel. main.rs routes a close request on this window here so a
/// transient panel never runs the main window's close policy. Takes the
/// `Window` the window-event handler is given, not a `WebviewWindow`.
#[cfg(target_os = "macos")]
pub fn hide_ask_window(window: &tauri::Window) {
    use tauri::Manager;
    let app = window.app_handle().clone();
    if let Some(ask) = app.get_webview_window(ASK_WINDOW_LABEL) {
        monitor::hide_ask(&app, &ask);
    }
}

#[cfg(not(target_os = "macos"))]
pub fn init(_app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}
