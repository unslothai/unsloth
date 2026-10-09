// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! macOS: browser pages sit under the app's webview, which is transparent over them, so panel
//! overlays draw on top of the live page. The app's webview returns nil from `hitTest:` over the
//! page unless panel UI covers that point (`browser_view_input`).

use crate::browser_webview::ViewBounds;
use objc2::runtime::{AnyClass, AnyObject, Imp, Sel};
use objc2::sel;
use objc2_app_kit::{NSEvent, NSView, NSWindowOrderingMode};
use objc2_foundation::{ns_string, NSNumber, NSObjectNSKeyValueCoding, NSPoint, NSRect, NSSize};
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::{Mutex, OnceLock};
use tauri::{Runtime, Webview};

type HitTest = unsafe extern "C-unwind" fn(*mut AnyObject, Sel, NSPoint) -> *mut AnyObject;
type MouseMoved = unsafe extern "C-unwind" fn(*mut AnyObject, Sel, *mut AnyObject);

/// The app's webview, set once on the main thread; it lives as long as the app.
static MAIN: AtomicPtr<AnyObject> = AtomicPtr::new(null_mut());
static INSTALLING: AtomicBool = AtomicBool::new(false);
static HIT_TEST: OnceLock<HitTest> = OnceLock::new();
static MOUSE_MOVED: OnceLock<MouseMoved> = OnceLock::new();
/// Whether the app's webview last saw the pointer over the page.
static MAIN_OVER_PAGE: AtomicBool = AtomicBool::new(false);
static INPUT: Mutex<Input> = Mutex::new(Input {
    blocked: false,
    exclude: Vec::new(),
});

/// Panel UI over the page, in the app's CSS pixels.
struct Input {
    /// A menu or dialog is open, so the panel takes all page input.
    blocked: bool,
    /// `(x, y, width, height, viewport_width)` of clickable UI over the page (toasts, find bar).
    exclude: Vec<(f64, f64, f64, f64, f64)>,
}

/// Sets which parts of the page the panel's UI covers.
pub fn set_input(blocked: bool, exclude: &[ViewBounds]) {
    let mut input = INPUT.lock().unwrap();
    input.blocked = blocked;
    input.exclude = exclude.iter().map(ViewBounds::parts).collect();
}

/// Makes the app's webview transparent and routes input. Runs once, before any page exists.
pub fn install<R: Runtime>(main: &Webview<R>) {
    if INSTALLING.swap(true, Ordering::SeqCst) {
        return;
    }
    let _ = main.with_webview(|platform| {
        let view = platform.inner() as *mut AnyObject;
        // Safety: wry's live WKWebView, on the main thread.
        unsafe { install_on(view) };
    });
}

/// Moves a new page under the app's webview.
pub fn place<R: Runtime>(page: &Webview<R>) {
    let _ = page.with_webview(|platform| {
        let main = MAIN.load(Ordering::Acquire);
        if main.is_null() {
            return;
        }
        // Safety: both are live NSViews, used on the main thread.
        unsafe {
            let main = &*(main as *const NSView);
            let page = &*(platform.inner() as *const NSView);
            if let Some(parent) = main.superview() {
                if page.superview().as_deref() == Some(&*parent) {
                    parent.addSubview_positioned_relativeTo(
                        page,
                        NSWindowOrderingMode::Below,
                        Some(main),
                    );
                }
            }
        }
    });
}

unsafe fn install_on(view: *mut AnyObject) {
    let Some(class) = (unsafe { view.as_ref() }).map(AnyObject::class) else {
        return;
    };
    // Pages share wry's class, so the overrides check for the main view by pointer.
    let (Some(hit_test), Some(mouse_moved)) = (
        class.instance_method(sel!(hitTest:)),
        class.instance_method(sel!(mouseMoved:)),
    ) else {
        log::warn!("browser pages stay over the panel: the webview class can't be routed");
        return;
    };
    // Safety: same signatures; the originals are stored before the swap can call them.
    unsafe {
        let _ = HIT_TEST.set(std::mem::transmute::<Imp, HitTest>(
            hit_test.implementation(),
        ));
        let _ = MOUSE_MOVED.set(std::mem::transmute::<Imp, MouseMoved>(
            mouse_moved.implementation(),
        ));
        let class = (class as *const AnyClass).cast_mut();
        objc2::ffi::class_replaceMethod(
            class,
            sel!(hitTest:),
            std::mem::transmute::<HitTest, Imp>(routed_hit_test),
            objc2::ffi::method_getTypeEncoding(hit_test),
        );
        objc2::ffi::class_replaceMethod(
            class,
            sel!(mouseMoved:),
            std::mem::transmute::<MouseMoved, Imp>(routed_mouse_moved),
            objc2::ffi::method_getTypeEncoding(mouse_moved),
        );
        // Private KVC key, as used by wry's `transparent` option.
        let main = &*(view as *const NSView);
        main.setValue_forKey(
            Some(&NSNumber::numberWithBool(false)),
            ns_string!("drawsBackground"),
        );
    }
    MAIN.store(view, Ordering::Release);
}

/// Whether `point` (main view coordinates) belongs to the page.
fn over_page(main: &NSView, point: NSPoint) -> bool {
    let input = INPUT.lock().unwrap();
    if input.blocked {
        return false;
    }
    // Safety: the main thread's view tree.
    let Some(parent) = (unsafe { main.superview() }) else {
        return false;
    };
    let class = main.class();
    let page = parent.subviews().iter().any(|view| {
        !std::ptr::eq(&*view, main)
            && std::ptr::eq(view.class(), class)
            && !view.isHidden()
            && point_in(
                main.convertRect_fromView(view.frame(), Some(&parent)),
                point,
            )
    });
    if !page {
        return false;
    }
    // The main view is flipped: points are CSS px times the app zoom.
    let width = main.frame().size.width;
    !input.exclude.iter().any(|&(x, y, w, h, viewport)| {
        let scale = if viewport > 0.0 {
            width / viewport
        } else {
            1.0
        };
        point_in(
            NSRect::new(
                NSPoint::new(x * scale, y * scale),
                NSSize::new(w * scale, h * scale),
            ),
            point,
        )
    })
}

fn point_in(rect: NSRect, point: NSPoint) -> bool {
    point.x >= rect.origin.x
        && point.y >= rect.origin.y
        && point.x < rect.origin.x + rect.size.width
        && point.y < rect.origin.y + rect.size.height
}

fn main_view() -> Option<&'static NSView> {
    // Safety: set once to the app's webview, which outlives every event.
    unsafe { (MAIN.load(Ordering::Acquire) as *const NSView).as_ref() }
}

unsafe extern "C-unwind" fn routed_hit_test(
    this: *mut AnyObject,
    cmd: Sel,
    point: NSPoint,
) -> *mut AnyObject {
    if let Some(main) = main_view().filter(|main| std::ptr::eq(*main as *const NSView, this.cast()))
    {
        // `point` is in the superview's coordinates.
        // Safety: the main thread's view tree.
        let local = main.convertPoint_fromView(point, unsafe { main.superview() }.as_deref());
        if over_page(main, local) {
            return null_mut();
        }
    }
    let original = HIT_TEST.get().expect("installed before routing");
    // Safety: AppKit's arguments, forwarded to the original implementation.
    unsafe { original(this, cmd, point) }
}

/// Drops moves meant for the other view, so page hover and cursor don't fight panel menus.
unsafe extern "C-unwind" fn routed_mouse_moved(
    this: *mut AnyObject,
    cmd: Sel,
    event: *mut AnyObject,
) {
    // Safety: AppKit passes a live NSEvent.
    if let (Some(main), Some(ns_event)) =
        (main_view(), unsafe { (event as *const NSEvent).as_ref() })
    {
        let is_main = std::ptr::eq(main as *const NSView, this.cast());
        // Safety: `this` is a live NSView (wry's class).
        let this_view = unsafe { &*(this as *const NSView) };
        if is_main || this_view.window() == main.window() {
            let point = main.convertPoint_fromView(ns_event.locationInWindow(), None);
            let page = over_page(main, point);
            if is_main {
                // Pass the first move onto the page through to end the panel's hover.
                if page && MAIN_OVER_PAGE.swap(true, Ordering::Relaxed) {
                    return;
                }
                if !page {
                    MAIN_OVER_PAGE.store(false, Ordering::Relaxed);
                }
            } else if !page {
                return;
            }
        }
    }
    let original = MOUSE_MOVED.get().expect("installed before routing");
    // Safety: AppKit's arguments, forwarded to the original implementation.
    unsafe { original(this, cmd, event) }
}
