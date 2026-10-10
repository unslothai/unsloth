// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! macOS: browser pages sit under the app's webview, which is transparent over them, so panel
//! overlays draw on top of the live page. The app's webview returns nil from `hitTest:` and
//! `_hitTest:dragTypes:` over the page unless panel UI covers that point (`browser_view_input`).

use crate::browser_webview::ViewBounds;
use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::{AnyClass, AnyObject, Bool, Imp, Sel};
use objc2::{class, msg_send, sel};
use objc2_app_kit::{NSEvent, NSView, NSWindowOrderingMode};
use objc2_foundation::{ns_string, NSNumber, NSObjectNSKeyValueCoding, NSPoint, NSRect, NSSize};
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};
use tauri::{Runtime, Webview};

type HitTest = unsafe extern "C-unwind" fn(*mut AnyObject, Sel, NSPoint) -> *mut AnyObject;
/// AppKit's private drop target lookup; WKWebView claims its whole frame without `hitTest:`.
type DragHitTest = unsafe extern "C-unwind" fn(
    *mut AnyObject,
    Sel,
    *mut NSPoint,
    *mut AnyObject,
) -> *mut AnyObject;

/// The app's webview, set once on the main thread; it lives as long as the app.
static MAIN: AtomicPtr<AnyObject> = AtomicPtr::new(null_mut());
static INSTALLING: AtomicBool = AtomicBool::new(false);
static HIT_TEST: OnceLock<HitTest> = OnceLock::new();
static DRAG_HIT_TEST: OnceLock<DragHitTest> = OnceLock::new();
/// Which webview takes mouse moves (`APP` or `PAGE`, 0 before the first). WebKit's tracking
/// areas hand every move in the window to both views whatever is on top, so both would set the
/// cursor and hover in turn, a flicker over links, text and annotate.
static MOVES_TO: AtomicU8 = AtomicU8::new(0);
/// The view that just lost moves, cut off on the next one so it sees the pointer leave first.
static LAGGING: AtomicU8 = AtomicU8::new(0);
const APP: u8 = 1;
const PAGE: u8 = 2;
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

/// Sets which parts of the page the panel's UI covers; true when `blocked` changed.
pub fn set_input(blocked: bool, exclude: &[ViewBounds]) -> bool {
    let mut input = INPUT.lock().unwrap();
    let changed = input.blocked != blocked;
    input.blocked = blocked;
    input.exclude = exclude.iter().map(ViewBounds::parts).collect();
    changed
}

/// Gives mouse moves to whichever view owns the pointer now, e.g. after a menu opened. Main thread.
pub fn reroute_moves() {
    let Some(main) = main_view() else {
        return;
    };
    let Some(window) = main.window() else {
        return;
    };
    // Safety: AppKit class and window methods, on the main thread.
    let point: NSPoint = unsafe {
        let screen: NSPoint = msg_send![class!(NSEvent), mouseLocation];
        msg_send![&*window, convertPointFromScreen: screen]
    };
    route_moves(main, main.convertPoint_fromView(point, None));
}

fn route_moves(main: &NSView, point: NSPoint) {
    let to = if over_page(main, point) { PAGE } else { APP };
    let lagging = LAGGING.swap(0, Ordering::Relaxed);
    if lagging != 0 && lagging != to {
        ignore_moves(main, lagging, true);
    }
    let from = MOVES_TO.swap(to, Ordering::Relaxed);
    if from == to {
        return;
    }
    ignore_moves(main, to, false);
    match from {
        0 => ignore_moves(main, APP + PAGE - to, true),
        _ => LAGGING.store(from, Ordering::Relaxed),
    }
}

/// `_setIgnoresMouseMoveEvents:` (private, macOS 13+, checked first) on the app or every page.
fn ignore_moves(main: &NSView, which: u8, ignore: bool) {
    let set = |view: &NSView| {
        // Safety: a live view of wry's class, on the main thread.
        unsafe {
            let responds: bool =
                msg_send![view, respondsToSelector: sel!(_setIgnoresMouseMoveEvents:)];
            if responds {
                let _: () = msg_send![view, _setIgnoresMouseMoveEvents: Bool::new(ignore)];
            }
        }
    };
    if which == APP {
        set(main);
        return;
    }
    // Safety: the main thread's view tree.
    let Some(parent) = (unsafe { main.superview() }) else {
        return;
    };
    for view in parent.subviews().iter() {
        if !std::ptr::eq(&*view, main) && std::ptr::eq(view.class(), main.class()) {
            set(&view);
        }
    }
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
                    MOVES_TO.store(0, Ordering::Relaxed);
                    reroute_moves();
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
    let (Some(hit_test), Some(drag_hit_test)) = (
        class.instance_method(sel!(hitTest:)),
        class.instance_method(sel!(_hitTest:dragTypes:)),
    ) else {
        log::warn!("browser pages stay over the panel: the webview class can't be routed");
        return;
    };
    // Safety: same signatures; the originals are stored before the swap can call them.
    unsafe {
        let _ = HIT_TEST.set(std::mem::transmute::<Imp, HitTest>(
            hit_test.implementation(),
        ));
        let _ = DRAG_HIT_TEST.set(std::mem::transmute::<Imp, DragHitTest>(
            drag_hit_test.implementation(),
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
            sel!(_hitTest:dragTypes:),
            std::mem::transmute::<DragHitTest, Imp>(routed_drag_hit_test),
            objc2::ffi::method_getTypeEncoding(drag_hit_test),
        );
        // Private KVC key, as used by wry's `transparent` option.
        let main = &*(view as *const NSView);
        main.setValue_forKey(
            Some(&NSNumber::numberWithBool(false)),
            ns_string!("drawsBackground"),
        );
    }
    MAIN.store(view, Ordering::Release);
    // Before dispatch, so the view losing moves is cut off at the move that leaves it.
    let route = RcBlock::new(|event: *mut AnyObject| -> *mut AnyObject {
        // Safety: AppKit passes a live NSEvent to the monitor, on the main thread.
        if let (Some(main), Some(moved)) =
            (main_view(), unsafe { (event as *const NSEvent).as_ref() })
        {
            if main.window().map(|window| window.windowNumber()) == Some(moved.windowNumber()) {
                route_moves(
                    main,
                    main.convertPoint_fromView(moved.locationInWindow(), None),
                );
            }
        }
        event
    });
    // NSEventMaskMouseMoved. Kept for the app's lifetime, like the main webview.
    let mask: u64 = 1 << 5;
    // Safety: a class method taking a mask and an `NSEvent *(^)(NSEvent *)` block.
    let monitor: Option<Retained<AnyObject>> = unsafe {
        msg_send![class!(NSEvent), addLocalMonitorForEventsMatchingMask: mask, handler: &*route]
    };
    std::mem::forget(monitor);
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

/// Files dragged over the page drop on it, as before it sat under the app's webview.
unsafe extern "C-unwind" fn routed_drag_hit_test(
    this: *mut AnyObject,
    cmd: Sel,
    point: *mut NSPoint,
    types: *mut AnyObject,
) -> *mut AnyObject {
    if let Some(main) = main_view().filter(|main| std::ptr::eq(*main as *const NSView, this.cast()))
    {
        // Safety: AppKit passes a live point in the superview's coordinates, on the main thread.
        if let Some(at) = unsafe { point.as_ref() } {
            let local = main.convertPoint_fromView(*at, unsafe { main.superview() }.as_deref());
            if over_page(main, local) {
                return null_mut();
            }
        }
    }
    let original = DRAG_HIT_TEST.get().expect("installed before routing");
    // Safety: AppKit's arguments, forwarded to the original implementation.
    unsafe { original(this, cmd, point, types) }
}
