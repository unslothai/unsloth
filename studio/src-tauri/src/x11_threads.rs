// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

//! Turn on Xlib's internal locking before anything opens an X display.
//!
//! GTK3 never calls `XInitThreads()` (`libgdk-3.so` does not even reference the
//! symbol), so every Xlib lock is a no-op. This process drives X from several
//! threads regardless -- WebKitGTK's compositor, GLib workers, tao's event loop,
//! the clipboard -- so the request/reply sequence Xlib keeps for its XCB
//! transport gets corrupted.
//!
//! That surfaces either as libX11 aborting on its own assertion:
//!
//! ```text
//! [xcb] Most likely this is a multi-threaded client and XInitThreads has not been called
//! unsloth-studio: ../../src/xcb_io.c:175: dequeue_pending_request:
//!     Assertion `!xcb_xlib_unknown_req_in_deq' failed.
//! ```
//!
//! or, more often, `_XReply` failing to match a reply and calling `_XIOError`
//! while `xcb_connection_has_error()` is 0. GDK's IO error handler reports that
//! through `g_debug()` (dropped unless `G_MESSAGES_DEBUG` names the domain) and
//! then `_exit(1)`s, so the app vanishes with a bare rc=1 and no output at all:
//! issue #8062.
//!
//! `XInitThreads()` must precede every other Xlib call, hence `main`'s first
//! statement, ahead of the GTK init Tauri performs while building the app.

#[cfg(target_os = "linux")]
mod imp {
    use std::os::raw::{c_char, c_int, c_void};

    extern "C" {
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    const RTLD_DEFAULT: *mut c_void = std::ptr::null_mut();

    /// Resolved at run time rather than linked: the crate does not otherwise
    /// link libX11, and on Wayland-only or headless systems the symbol is
    /// legitimately absent -- there is no X connection to protect.
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
    /// Only Xlib needs this; Windows and macOS have their own main-thread rules.
    pub fn init() -> bool {
        false
    }
}

/// Returns whether Xlib locking is now on: false on non-Linux, and on Linux
/// when libX11 is not loaded.
pub fn init_x11_threads() -> bool {
    imp::init()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_is_safe_to_call_and_idempotent() {
        // Never panics on any platform; XInitThreads is safe to call repeatedly.
        let first = init_x11_threads();
        assert_eq!(first, init_x11_threads());
    }
}
