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

#[tauri::command]
pub(crate) fn enter_orb_widget_mode(app: tauri::AppHandle) -> Result<(), String> {
    FLOATING.store(true, Ordering::SeqCst);
    set_main_floating(&app, true)
}

/// Undo `enter_orb_widget_mode` and close the widget; the caller shows the main window.
pub(crate) fn leave(app: &tauri::AppHandle) {
    if !FLOATING.swap(false, Ordering::SeqCst) {
        return;
    }
    let _ = set_main_floating(app, false);
    if let Some(widget) = app.get_window(LABEL) {
        let _ = widget.destroy();
    }
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
