use super::config::{self, PillConfig};
use super::{geometry, monitor, panel, PillState, ASK_WINDOW_LABEL, EVENT_ASK_SHOW};
use core_graphics::display::CGDisplay;
use log::{info, warn};
use tauri::{
    AppHandle, Emitter, LogicalPosition, Manager, WebviewUrl, WebviewWindowBuilder,
};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

const ASK_SIZE: (f64, f64) = (640.0, 72.0);

pub fn init(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let handle = app.handle().clone();
    let dir = handle.path().app_config_dir()?;
    let loaded = config::load_config(&dir);
    {
        let state = app.state::<PillState>();
        *state.config.lock().unwrap() = loaded.clone();
    }

    let ask_window = WebviewWindowBuilder::new(
        &handle,
        ASK_WINDOW_LABEL,
        WebviewUrl::App("ask.html".into()),
    )
    .title("Unsloth")
    .inner_size(ASK_SIZE.0, ASK_SIZE.1)
    .visible(false)
    .decorations(false)
    .transparent(true)
    .resizable(false)
    .skip_taskbar(true)
    .always_on_top(true)
    .shadow(false)
    .focused(false)
    .build()?;
    panel::convert_to_key_panel(&ask_window).map_err(std::io::Error::other)?;

    // Native frosted glass behind the transparent webview; radius must match
    // the CSS corner radius (rounded-2xl = 16).
    {
        use window_vibrancy::{apply_vibrancy, NSVisualEffectMaterial, NSVisualEffectState};
        if let Err(e) = apply_vibrancy(
            &ask_window,
            NSVisualEffectMaterial::Popover,
            Some(NSVisualEffectState::Active),
            Some(16.0),
        ) {
            warn!("ask: vibrancy failed: {e}");
        }
    }

    monitor::install_dismiss_monitors(handle.clone());
    if let Err(e) = apply_hotkey(&handle, &loaded) {
        warn!("ask: hotkey registration failed: {e}");
    }
    info!(
        "ask: initialized (enabled: {}, hotkey: {})",
        loaded.enabled, loaded.ask_hotkey
    );
    Ok(())
}

pub fn apply_config(app: &AppHandle, config: &PillConfig) -> Result<(), String> {
    config::save_for_app(app, config)?;
    // Registration is best-effort: a taken hotkey (another launcher, a stale
    // instance) must not fail the save the user just made.
    if let Err(e) = apply_hotkey(app, config) {
        warn!("ask: hotkey registration failed: {e}");
    }
    Ok(())
}

fn apply_hotkey(app: &AppHandle, config: &PillConfig) -> Result<(), String> {
    let shortcuts = app.global_shortcut();
    shortcuts
        .unregister_all()
        .map_err(|e| format!("Failed to clear shortcuts: {e}"))?;
    if config.enabled {
        shortcuts
            .on_shortcut(config.ask_hotkey.as_str(), move |app, _shortcut, event| {
                if event.state == ShortcutState::Pressed {
                    toggle_ask(app);
                }
            })
            .map_err(|e| {
                format!("Failed to register ask hotkey '{}': {e}", config.ask_hotkey)
            })?;
    }
    Ok(())
}

/// Raycast-style toggle: hide when visible, else center on the mouse screen
/// and take keyboard focus without activating the app.
pub fn toggle_ask(app: &AppHandle) {
    let Some(window) = app.get_webview_window(ASK_WINDOW_LABEL) else {
        return;
    };
    if panel::is_panel_visible(&window) {
        monitor::hide_ask(app, &window);
        return;
    }
    show_ask(app, None);
}

/// Show the ask bar, optionally seeded with context text.
pub fn show_ask(app: &AppHandle, context: Option<String>) {
    let Some(window) = app.get_webview_window(ASK_WINDOW_LABEL) else {
        return;
    };
    let screen = screen_containing(mouse_anchor());
    let size = panel::panel_frame(&window)
        .map(|frame| (frame.width, frame.height))
        .unwrap_or(ASK_SIZE);
    let x = screen.x + (screen.width - size.0) / 2.0;
    let y = screen.y + screen.height * 0.22;
    let _ = window.set_position(LogicalPosition::new(x, y));
    let _ = app.emit_to(ASK_WINDOW_LABEL, EVENT_ASK_SHOW, context);
    panel::show_key_panel(&window);
}

fn mouse_anchor() -> geometry::Rect {
    use core_graphics::event::CGEvent;
    use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
    let location = CGEventSource::new(CGEventSourceStateID::CombinedSessionState)
        .ok()
        .and_then(|source| CGEvent::new(source).ok())
        .map(|event| event.location());
    match location {
        Some(point) => geometry::Rect::new(point.x, point.y, 1.0, 1.0),
        None => geometry::Rect::new(0.0, 0.0, 1.0, 1.0),
    }
}

fn screen_containing(anchor: geometry::Rect) -> geometry::Rect {
    let center_x = anchor.x + anchor.width / 2.0;
    let center_y = anchor.y + anchor.height / 2.0;
    let displays = CGDisplay::active_displays().unwrap_or_default();
    for id in displays {
        let bounds = CGDisplay::new(id).bounds();
        if center_x >= bounds.origin.x
            && center_x < bounds.origin.x + bounds.size.width
            && center_y >= bounds.origin.y
            && center_y < bounds.origin.y + bounds.size.height
        {
            return geometry::Rect::new(
                bounds.origin.x,
                bounds.origin.y,
                bounds.size.width,
                bounds.size.height,
            );
        }
    }
    let main = CGDisplay::main().bounds();
    geometry::Rect::new(main.origin.x, main.origin.y, main.size.width, main.size.height)
}
