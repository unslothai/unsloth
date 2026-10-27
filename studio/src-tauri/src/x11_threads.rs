// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

//! Turn on Xlib's internal locking before anything opens an X display. GTK3 never calls
//! `XInitThreads()`, yet several threads drive X, which corrupts Xlib's XCB sequence and makes
//! GDK `_exit(1)` silently. It must precede every other Xlib call, hence `main`'s first statement.

#[cfg(target_os = "linux")]
mod imp {
    use std::os::raw::{c_char, c_int, c_void};

    extern "C" {
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    const RTLD_DEFAULT: *mut c_void = std::ptr::null_mut();

    /// Resolved at run time: the crate does not link libX11, and Wayland/headless hosts lack it.
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
    pub fn init() -> bool {
        false
    }
}

/// False on non-Linux, and on Linux when libX11 is not loaded.
pub fn init_x11_threads() -> bool {
    imp::init()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_is_safe_to_call_and_idempotent() {
        let first = init_x11_threads();
        assert_eq!(first, init_x11_threads());
    }
}
