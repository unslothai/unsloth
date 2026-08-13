// The ask bar must accept typing without activating the app: a
// NonactivatingPanel that can become key, via an Objective-C class swap.

use objc2::{define_class, msg_send, ClassType};
use objc2_app_kit::{NSPanel, NSWindow, NSWindowCollectionBehavior, NSWindowStyleMask};
use tauri::{Runtime, WebviewWindow};

define_class!(
    #[unsafe(super(NSPanel))]
    #[name = "UnslothAskPanel"]
    pub struct AskPanel;

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

const NS_POPUP_MENU_WINDOW_LEVEL: isize = 101;

/// Must run on the main thread.
pub fn convert_to_key_panel<R: Runtime>(window: &WebviewWindow<R>) -> Result<(), String> {
    convert_with_class(window, AskPanel::class())
}

fn convert_with_class<R: Runtime>(
    window: &WebviewWindow<R>,
    class: &objc2::runtime::AnyClass,
) -> Result<(), String> {
    let ns_window = window
        .ns_window()
        .map_err(|e| format!("No NSWindow for pill: {e}"))? as *mut NSWindow;
    if ns_window.is_null() {
        return Err("Null NSWindow for pill".to_string());
    }
    unsafe {
        objc2::ffi::object_setClass(
            ns_window as *mut objc2::runtime::AnyObject,
            class as *const _ as *mut _,
        );
        let panel = &*(ns_window as *mut NSPanel);
        panel.setStyleMask(
            panel.styleMask() | NSWindowStyleMask::NonactivatingPanel,
        );
        panel.setLevel(NS_POPUP_MENU_WINDOW_LEVEL);
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

/// Show and take keyboard focus without activating the app.
pub fn show_key_panel<R: Runtime>(window: &WebviewWindow<R>) {
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
}

pub fn hide_panel<R: Runtime>(window: &WebviewWindow<R>) {
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

pub fn is_panel_visible<R: Runtime>(window: &WebviewWindow<R>) -> bool {
    window.is_visible().unwrap_or(false)
}

/// Frame in logical global top-left coordinates.
pub fn panel_frame<R: Runtime>(
    window: &WebviewWindow<R>,
) -> Option<crate::selection_pill::geometry::Rect> {
    let position = window.outer_position().ok()?;
    let size = window.outer_size().ok()?;
    let scale = window.scale_factor().unwrap_or(1.0);
    Some(crate::selection_pill::geometry::Rect::new(
        position.x as f64 / scale,
        position.y as f64 / scale,
        size.width as f64 / scale,
        size.height as f64 / scale,
    ))
}
