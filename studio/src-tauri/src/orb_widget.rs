// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! The floating voice orb's handoff. The voice loop runs in the main webview, and WebKit
//! defers getUserMedia (and mutes capture) while a view is not visible, so on macOS the main
//! window is kept ordered in but transparent and click-through instead of hidden, with the
//! webview told to ignore occlusion.

use std::sync::atomic::{AtomicBool, Ordering};

use tauri::Manager;

pub(crate) const LABEL: &str = "orb-widget";

static FLOATING: AtomicBool = AtomicBool::new(false);

pub(crate) fn is_floating() -> bool {
    FLOATING.load(Ordering::SeqCst)
}

/// Place and show the warm widget at `x`, `y` (logical, top left), then fade the main window.
#[tauri::command]
pub(crate) fn enter_orb_widget_mode(
    app: tauri::AppHandle,
    x: Option<f64>,
    y: Option<f64>,
) -> Result<(), String> {
    let widget = app.get_webview_window(LABEL).ok_or("The orb widget is not ready")?;
    show_widget(&widget, x.zip(y))?;
    FLOATING.store(true, Ordering::SeqCst);
    set_main_floating(&app, true)
}

/// Undo `enter_orb_widget_mode` and hide the widget, kept warm for the next time; the caller
/// shows the main window.
pub(crate) fn leave(app: &tauri::AppHandle) {
    if !FLOATING.swap(false, Ordering::SeqCst) {
        return;
    }
    let _ = set_main_floating(app, false);
    if let Some(widget) = app.get_window(LABEL) {
        let _ = widget.hide();
    }
}

/// A status-bar level window: above the menu bar, and AppKit stops constraining its frame
/// to the area below it (a floating window is clamped to the visible frame).
#[cfg(target_os = "macos")]
const STATUS_WINDOW_LEVEL: isize = 25;

#[cfg(target_os = "macos")]
fn show_widget(widget: &tauri::WebviewWindow, at: Option<(f64, f64)>) -> Result<(), String> {
    widget
        .with_webview(move |platform| {
            use objc2::runtime::AnyObject;
            use objc2::{class, msg_send};
            use objc2_app_kit::NSWindow;
            use objc2_foundation::{NSPoint, NSRect};
            // Safety: wry's live window, on the main thread.
            unsafe {
                let Some(window) = (platform.ns_window() as *const NSWindow).as_ref() else {
                    return;
                };
                window.setLevel(STATUS_WINDOW_LEVEL);
                if let Some((x, y)) = at {
                    // Cocoa counts up from the primary screen's bottom edge.
                    let screens: *mut AnyObject = msg_send![class!(NSScreen), screens];
                    let primary: *mut AnyObject = msg_send![screens, firstObject];
                    if let Some(primary) = primary.as_ref() {
                        let frame: NSRect = msg_send![primary, frame];
                        window.setFrameTopLeftPoint(NSPoint::new(x, frame.size.height - y));
                    }
                }
                window.makeKeyAndOrderFront(None);
            }
        })
        .map_err(|error| error.to_string())
}

#[cfg(not(target_os = "macos"))]
fn show_widget(widget: &tauri::WebviewWindow, at: Option<(f64, f64)>) -> Result<(), String> {
    if let Some((x, y)) = at {
        let _ = widget.set_position(tauri::LogicalPosition::new(x, y));
    }
    widget.show().map_err(|error| error.to_string())?;
    let _ = widget.set_focus();
    Ok(())
}

#[cfg(target_os = "macos")]
fn set_main_floating(app: &tauri::AppHandle, floating: bool) -> Result<(), String> {
    let webview = app.get_webview("main").ok_or("The main webview is missing")?;
    webview
        .with_webview(move |platform| {
            use objc2::runtime::{AnyObject, Bool};
            use objc2::{msg_send, sel};
            use objc2_app_kit::NSWindow;
            // Safety: wry's live view and its window, on the main thread.
            unsafe {
                if let Some(view) = (platform.inner() as *const AnyObject).as_ref() {
                    let occlusion = sel!(_setWindowOcclusionDetectionEnabled:);
                    if view.class().responds_to(occlusion) {
                        let _: () =
                            msg_send![view, _setWindowOcclusionDetectionEnabled: Bool::new(!floating)];
                    }
                }
                if let Some(window) = (platform.ns_window() as *const NSWindow).as_ref() {
                    window.setIgnoresMouseEvents(floating);
                    window.setAlphaValue(if floating { 0.0 } else { 1.0 });
                    if floating {
                        window.orderBack(None);
                    }
                }
            }
        })
        .map_err(|error| error.to_string())
}

#[cfg(not(target_os = "macos"))]
fn set_main_floating(app: &tauri::AppHandle, floating: bool) -> Result<(), String> {
    if floating {
        if let Some(window) = app.get_window("main") {
            window.hide().map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}
