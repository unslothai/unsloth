//! The macOS Ask bar: a global shortcut opens a floating panel that asks the local model.
//! Off by default; its window and shortcut only exist while it is enabled.

use log::warn;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

pub const ASK_WINDOW_LABEL: &str = "ask";
const PREFERENCE_FILE: &str = "ask-bar-enabled";

#[derive(Default)]
pub struct AskBarState(AtomicBool);

fn preference_path(config_dir: &Path) -> PathBuf {
    config_dir.join(PREFERENCE_FILE)
}

fn read_preference(config_dir: &Path) -> bool {
    fs::read_to_string(preference_path(config_dir))
        .map(|value| value.trim() == "true")
        .unwrap_or(false)
}

fn write_preference(config_dir: &Path, enabled: bool) -> Result<(), String> {
    fs::create_dir_all(config_dir).map_err(|error| format!("Failed to create config dir: {error}"))?;
    fs::write(preference_path(config_dir), format!("{enabled}\n"))
        .map_err(|error| format!("Failed to save the Ask bar setting: {error}"))
}

fn config_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    use tauri::Manager;
    app.path()
        .app_config_dir()
        .map_err(|error| format!("Could not determine app configuration directory: {error}"))
}

/// Registers the shortcut when the saved preference is on. A clash must not fail startup.
pub fn init(app: &tauri::AppHandle) {
    use tauri::Manager;
    let enabled = config_dir(app).map(|dir| read_preference(&dir)).unwrap_or(false);
    if !enabled || !cfg!(target_os = "macos") {
        return;
    }
    match platform::apply(app, true) {
        Ok(()) => app.state::<AskBarState>().0.store(true, Ordering::SeqCst),
        Err(error) => warn!("Ask bar: {error}"),
    }
}

#[tauri::command]
pub fn get_ask_bar(state: tauri::State<'_, AskBarState>) -> Option<bool> {
    cfg!(target_os = "macos").then(|| state.0.load(Ordering::SeqCst))
}

#[tauri::command]
pub fn set_ask_bar(
    app: tauri::AppHandle,
    state: tauri::State<'_, AskBarState>,
    enabled: bool,
) -> Result<bool, String> {
    if !cfg!(target_os = "macos") {
        return Err("The Ask bar is only available on macOS".to_string());
    }
    let dir = config_dir(&app)?;
    // Apply first so a shortcut that failed to register is never saved as on.
    platform::apply(&app, enabled)?;
    if let Err(error) = write_preference(&dir, enabled) {
        let _ = platform::apply(&app, !enabled);
        return Err(error);
    }
    state.0.store(enabled, Ordering::SeqCst);
    Ok(enabled)
}

#[tauri::command]
pub fn ask_hide(window: tauri::WebviewWindow) -> Result<(), String> {
    ensure_ask_window(&window)?;
    platform::hide(&window);
    Ok(())
}

#[tauri::command]
pub fn ask_resize(window: tauri::WebviewWindow, width: f64, height: f64) -> Result<(), String> {
    ensure_ask_window(&window)?;
    window
        .set_size(tauri::LogicalSize::new(width.max(320.0), height.max(48.0)))
        .map_err(|error| format!("Failed to resize the Ask bar: {error}"))
}

fn ensure_ask_window(window: &tauri::WebviewWindow) -> Result<(), String> {
    if window.label() == ASK_WINDOW_LABEL {
        Ok(())
    } else {
        Err("Only the Ask bar window can do this".to_string())
    }
}

/// Cmd+W on the panel hides it; the main window's close policy would quit the app under it.
pub fn hide_window(window: &tauri::Window) {
    use tauri::Manager;
    if let Some(ask) = window.app_handle().get_webview_window(ASK_WINDOW_LABEL) {
        platform::hide(&ask);
    }
}

#[cfg(target_os = "macos")]
use macos as platform;

#[cfg(not(target_os = "macos"))]
mod platform {
    pub fn apply(_app: &tauri::AppHandle, _enabled: bool) -> Result<(), String> {
        Ok(())
    }

    pub fn hide(window: &tauri::WebviewWindow) {
        let _ = window.hide();
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use super::ASK_WINDOW_LABEL;
    use block2::RcBlock;
    use core_graphics::display::CGDisplay;
    use core_graphics::event::CGEvent;
    use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
    use log::{info, warn};
    use objc2::{define_class, msg_send, ClassType};
    use objc2_app_kit::{
        NSEvent, NSEventMask, NSPanel, NSWindow, NSWindowCollectionBehavior, NSWindowStyleMask,
        NSWorkspace,
    };
    use objc2_foundation::NSNotification;
    use std::ptr::NonNull;
    use std::sync::atomic::{AtomicBool, Ordering};
    use tauri::{AppHandle, Emitter, LogicalPosition, Manager, WebviewUrl, WebviewWindow};
    use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

    const HOTKEY: &str = "alt+space";
    const SIZE: (f64, f64) = (640.0, 72.0);
    // NSPopUpMenuWindowLevel: above full-screen apps' windows.
    const PANEL_LEVEL: isize = 101;

    // Read by the system-wide click monitor on every click, so it avoids a window lookup.
    static VISIBLE: AtomicBool = AtomicBool::new(false);

    define_class!(
        // A non-activating panel that can still become key, so typing works without
        // pulling Unsloth in front of the app the user is in.
        #[unsafe(super(NSPanel))]
        #[name = "UnslothAskPanel"]
        struct AskPanel;

        impl AskPanel {
            #[unsafe(method(canBecomeKeyWindow))]
            fn can_become_key_window(&self) -> bool {
                true
            }

            #[unsafe(method(canBecomeMainWindow))]
            fn can_become_main_window(&self) -> bool {
                false
            }
        }
    );

    pub fn apply(app: &AppHandle, enabled: bool) -> Result<(), String> {
        let shortcuts = app.global_shortcut();
        if !enabled {
            if shortcuts.is_registered(HOTKEY) {
                shortcuts
                    .unregister(HOTKEY)
                    .map_err(|error| format!("Failed to release {HOTKEY}: {error}"))?;
            }
            if let Some(window) = app.get_webview_window(ASK_WINDOW_LABEL) {
                hide(&window);
            }
            return Ok(());
        }
        ensure_window(app)?;
        if !shortcuts.is_registered(HOTKEY) {
            shortcuts
                .on_shortcut(HOTKEY, |app, _shortcut, event| {
                    if event.state == ShortcutState::Pressed {
                        toggle(app);
                    }
                })
                .map_err(|error| format!("Failed to register {HOTKEY}: {error}"))?;
        }
        info!("Ask bar: enabled on {HOTKEY}");
        Ok(())
    }

    // Built on first enable, not at launch: most installs never turn the bar on.
    fn ensure_window(app: &AppHandle) -> Result<(), String> {
        if app.get_webview_window(ASK_WINDOW_LABEL).is_some() {
            return Ok(());
        }
        let window = tauri::WebviewWindowBuilder::new(
            app,
            ASK_WINDOW_LABEL,
            WebviewUrl::App("ask.html".into()),
        )
        .title("Unsloth")
        .inner_size(SIZE.0, SIZE.1)
        .visible(false)
        .decorations(false)
        .transparent(true)
        .resizable(false)
        .skip_taskbar(true)
        .always_on_top(true)
        .shadow(false)
        .focused(false)
        .build()
        .map_err(|error| format!("Failed to create the Ask bar window: {error}"))?;
        convert_to_panel(&window)?;
        // Radius matches the panel's rounded-2xl.
        if let Err(error) = window_vibrancy::apply_vibrancy(
            &window,
            window_vibrancy::NSVisualEffectMaterial::Popover,
            Some(window_vibrancy::NSVisualEffectState::Active),
            Some(16.0),
        ) {
            warn!("Ask bar: vibrancy failed: {error}");
        }
        install_dismiss_monitors(app.clone());
        Ok(())
    }

    fn convert_to_panel(window: &WebviewWindow) -> Result<(), String> {
        let ns_window = window
            .ns_window()
            .map_err(|error| format!("No NSWindow for the Ask bar: {error}"))?
            as *mut NSWindow;
        if ns_window.is_null() {
            return Err("No NSWindow for the Ask bar".to_string());
        }
        unsafe {
            objc2::ffi::object_setClass(
                ns_window as *mut objc2::runtime::AnyObject,
                AskPanel::class() as *const _ as *mut _,
            );
            let panel = &*(ns_window as *mut NSPanel);
            panel.setStyleMask(panel.styleMask() | NSWindowStyleMask::NonactivatingPanel);
            panel.setLevel(PANEL_LEVEL);
            panel.setCollectionBehavior(
                NSWindowCollectionBehavior::CanJoinAllSpaces
                    | NSWindowCollectionBehavior::FullScreenAuxiliary,
            );
            panel.setHidesOnDeactivate(false);
            panel.setBecomesKeyOnlyIfNeeded(true);
            let _: () = msg_send![panel, setFloatingPanel: true];
        }
        Ok(())
    }

    fn toggle(app: &AppHandle) {
        let Some(window) = app.get_webview_window(ASK_WINDOW_LABEL) else {
            return;
        };
        if VISIBLE.load(Ordering::SeqCst) {
            hide(&window);
        } else {
            show(app, &window);
        }
    }

    fn show(app: &AppHandle, window: &WebviewWindow) {
        let screen = screen_under_mouse();
        let width = window
            .outer_size()
            .ok()
            .zip(window.scale_factor().ok())
            .map(|(size, scale)| size.width as f64 / scale)
            .unwrap_or(SIZE.0);
        let x = screen.0 + (screen.2 - width) / 2.0;
        let y = screen.1 + screen.3 * 0.22;
        let _ = window.set_position(LogicalPosition::new(x, y));
        // Resets the conversation before the panel is ordered front.
        let _ = app.emit_to(ASK_WINDOW_LABEL, "ask://show", ());
        VISIBLE.store(true, Ordering::SeqCst);
        let window = window.clone();
        let _ = window.clone().run_on_main_thread(move || {
            if let Ok(ns_window) = window.ns_window() {
                let panel = ns_window as *mut NSPanel;
                if !panel.is_null() {
                    unsafe {
                        (*panel).orderFrontRegardless();
                        (*panel).makeKeyWindow();
                    }
                }
            }
        });
        info!("Ask bar: shown");
    }

    pub fn hide(window: &WebviewWindow) {
        VISIBLE.store(false, Ordering::SeqCst);
        let _ = window.emit_to(ASK_WINDOW_LABEL, "ask://hide", ());
        let window = window.clone();
        let _ = window.clone().run_on_main_thread(move || {
            if let Ok(ns_window) = window.ns_window() {
                let panel = ns_window as *mut NSPanel;
                if !panel.is_null() {
                    unsafe { (*panel).orderOut(None) };
                }
            }
        });
    }

    fn dismiss(app: &AppHandle) {
        if !VISIBLE.load(Ordering::SeqCst) {
            return;
        }
        if let Some(window) = app.get_webview_window(ASK_WINDOW_LABEL) {
            hide(&window);
        }
    }

    // Global monitors never see our own panel's events and need no Accessibility grant.
    // They live as long as the app, so their tokens are leaked.
    fn install_dismiss_monitors(app: AppHandle) {
        let click_app = app.clone();
        let on_click = RcBlock::new(move |_event: NonNull<NSEvent>| dismiss(&click_app));
        let mask = NSEventMask::LeftMouseDown
            | NSEventMask::RightMouseDown
            | NSEventMask::OtherMouseDown
            | NSEventMask::ScrollWheel;
        if let Some(token) = NSEvent::addGlobalMonitorForEventsMatchingMask_handler(mask, &on_click)
        {
            std::mem::forget(token);
        }
        let on_switch = RcBlock::new(move |_note: NonNull<NSNotification>| dismiss(&app));
        unsafe {
            let token = NSWorkspace::sharedWorkspace()
                .notificationCenter()
                .addObserverForName_object_queue_usingBlock(
                    Some(objc2_app_kit::NSWorkspaceDidActivateApplicationNotification),
                    None,
                    None,
                    &on_switch,
                );
            std::mem::forget(token);
        }
    }

    /// (x, y, width, height) of the display under the mouse, in global top-left points.
    fn screen_under_mouse() -> (f64, f64, f64, f64) {
        let mouse = CGEventSource::new(CGEventSourceStateID::CombinedSessionState)
            .ok()
            .and_then(|source| CGEvent::new(source).ok())
            .map(|event| event.location());
        let displays = CGDisplay::active_displays().unwrap_or_default();
        let bounds = mouse
            .and_then(|point| {
                displays.into_iter().map(|id| CGDisplay::new(id).bounds()).find(|b| {
                    point.x >= b.origin.x
                        && point.x < b.origin.x + b.size.width
                        && point.y >= b.origin.y
                        && point.y < b.origin.y + b.size.height
                })
            })
            .unwrap_or_else(|| CGDisplay::main().bounds());
        (bounds.origin.x, bounds.origin.y, bounds.size.width, bounds.size.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preference_defaults_off_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!read_preference(dir.path()));
        write_preference(dir.path(), true).unwrap();
        assert!(read_preference(dir.path()));
        write_preference(dir.path(), false).unwrap();
        assert!(!read_preference(dir.path()));
        fs::write(preference_path(dir.path()), "garbage").unwrap();
        assert!(!read_preference(dir.path()));
    }
}
