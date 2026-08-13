// Global NSEvent mouse monitors are TCC-free and never see our own panel's
// events, so no hit-testing is needed.

use super::{ASK_WINDOW_LABEL, EVENT_ASK_HIDE};
use block2::RcBlock;
use objc2_app_kit::{NSEvent, NSEventMask, NSWorkspace};
use objc2_foundation::NSNotification;
use std::ptr::NonNull;
use tauri::{AppHandle, Emitter, Manager};

fn dismiss(app: &AppHandle) {
    if let Some(window) = app.get_webview_window(ASK_WINDOW_LABEL) {
        if super::panel::is_panel_visible(&window) {
            hide_ask(app, &window);
        }
    }
}

pub(crate) fn hide_ask(app: &AppHandle, window: &tauri::WebviewWindow) {
    super::panel::hide_panel(window);
    let _ = app.emit_to(ASK_WINDOW_LABEL, EVENT_ASK_HIDE, ());
}

/// Must run on the main thread. Monitors live for the app's lifetime; their
/// tokens are intentionally leaked.
pub fn install_dismiss_monitors(app: AppHandle) {
    let mouse_app = app.clone();
    let mouse_handler = RcBlock::new(move |_event: NonNull<NSEvent>| {
        dismiss(&mouse_app);
    });
    let mask = NSEventMask::LeftMouseDown
        | NSEventMask::RightMouseDown
        | NSEventMask::OtherMouseDown
        | NSEventMask::ScrollWheel;
    let token = NSEvent::addGlobalMonitorForEventsMatchingMask_handler(mask, &mouse_handler);
    if let Some(token) = token {
        std::mem::forget(token);
    }

    let switch_app = app;
    let switch_handler = RcBlock::new(move |_note: NonNull<NSNotification>| {
        dismiss(&switch_app);
    });
    unsafe {
        let center = NSWorkspace::sharedWorkspace().notificationCenter();
        let token = center.addObserverForName_object_queue_usingBlock(
            Some(objc2_app_kit::NSWorkspaceDidActivateApplicationNotification),
            None,
            None,
            &switch_handler,
        );
        std::mem::forget(token);
    }
}
