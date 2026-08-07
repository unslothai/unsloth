// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

//! Turn on Xlib's internal locking before anything opens an X display.
//!
//! GTK3 never calls `XInitThreads()`: `libgdk-3.so` does not so much as
//! reference the symbol. Without it every Xlib lock is a no-op, so the request
//! sequence numbers Xlib keeps for its XCB transport are maintained with no
//! mutual exclusion at all. This process is unavoidably multi-threaded around
//! the X connection -- WebKitGTK's UI-process compositor and GLib worker
//! threads, tao's event loop and the clipboard all live here -- so the
//! sequence counter can be advanced by one thread between another thread
//! writing its request and recording it.
//!
//! When that happens libX11 either aborts on its own assertion:
//!
//! ```text
//! [xcb] Unknown request in queue while dequeuing
//! [xcb] Most likely this is a multi-threaded client and XInitThreads has not been called
//! [xcb] Aborting, sorry about that.
//! unsloth-studio: ../../src/xcb_io.c:175: dequeue_pending_request:
//!     Assertion `!xcb_xlib_unknown_req_in_deq' failed.
//! ```
//!
//! or, more often, `_XReply` cannot match a reply to its request and calls
//! `_XIOError` even though `xcb_connection_has_error()` is 0. GDK installs
//! `gdk_x_io_error()` as the Xlib IO error handler, and that reports through
//! `g_debug()` -- dropped unless `G_MESSAGES_DEBUG` names the domain -- and
//! then calls `_exit(1)`. The result is the app vanishing with a bare rc=1 and
//! no output at all, which is issue #8062.
//!
//! `XInitThreads()` has to run before any other Xlib call, so this is the
//! first thing `main` does, ahead of the GTK initialisation Tauri performs
//! while it builds the app.

#[cfg(target_os = "linux")]
mod imp {
    use std::os::raw::{c_char, c_int, c_void};

    extern "C" {
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    const RTLD_DEFAULT: *mut c_void = std::ptr::null_mut();

    /// Resolved at run time rather than linked: the crate does not otherwise
    /// link libX11, and on a Wayland-only or headless system the symbol may
    /// legitimately be absent, where there is no X connection to protect.
    pub fn init() -> bool {
        let symbol = unsafe { dlsym(RTLD_DEFAULT, c"XInitThreads".as_ptr()) };
        if symbol.is_null() {
            return false;
        }
        let init_threads: extern "C" fn() -> c_int = unsafe { std::mem::transmute(symbol) };
        init_threads() != 0
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    /// Only Xlib needs this. Windows and macOS marshal their UI calls through
    /// their own main-thread requirements instead.
    pub fn init() -> bool {
        false
    }
}

/// Returns whether Xlib locking is now on. False on non-Linux, and on Linux
/// when libX11 is not loaded at all.
pub fn init_x11_threads() -> bool {
    imp::init()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_is_safe_to_call_and_idempotent() {
        // Never panics, whatever the platform, and repeat calls are harmless:
        // XInitThreads is documented as safe to call more than once.
        let first = init_x11_threads();
        assert_eq!(first, init_x11_threads());
    }
}
