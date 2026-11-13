// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! Holds off idle system sleep while training runs; the display may still sleep. Both requests
//! belong to the thread that makes them (Windows' execution state is per thread, and macOS'
//! activity token is not `Send`), so a dedicated thread takes the request and releases it.

use log::warn;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Mutex;
use std::thread::JoinHandle;

static HOLDER: Mutex<Option<(Sender<()>, JoinHandle<()>)>> = Mutex::new(None);

/// Idempotent: the frontend republishes training activity on every mount.
pub fn set(enabled: bool) {
    let Ok(mut holder) = HOLDER.lock() else {
        return;
    };
    if enabled == holder.is_some() {
        return;
    }
    if let Some((release, thread)) = holder.take() {
        // Dropping the sender wakes the thread, which releases the request before it exits.
        drop(release);
        let _ = thread.join();
        return;
    }
    let (release, released) = channel();
    match std::thread::Builder::new()
        .name("keep-awake".to_string())
        .spawn(move || hold_until(released))
    {
        Ok(thread) => *holder = Some((release, thread)),
        Err(error) => warn!("Could not start the keep-awake thread: {error}"),
    }
}

#[cfg(windows)]
fn hold_until(released: Receiver<()>) {
    use log::info;
    use windows_sys::Win32::System::Power::{
        SetThreadExecutionState, ES_CONTINUOUS, ES_SYSTEM_REQUIRED,
    };

    // SAFETY: only changes this thread's execution state.
    if unsafe { SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) } == 0 {
        warn!("Could not keep the system awake during training");
        return;
    }
    info!("Idle system sleep blocked while training");
    let _ = released.recv();
    // SAFETY: as above; clears the flags this thread set.
    unsafe { SetThreadExecutionState(ES_CONTINUOUS) };
    info!("Idle system sleep allowed again");
}

#[cfg(target_os = "macos")]
fn hold_until(released: Receiver<()>) {
    use log::info;
    use objc2::rc::autoreleasepool;
    use objc2_foundation::{NSActivityOptions, NSProcessInfo, NSString};

    // A spawned thread has no autorelease pool of its own.
    autoreleasepool(|_| {
        let process = NSProcessInfo::processInfo();
        let activity = process.beginActivityWithOptions_reason(
            NSActivityOptions::IdleSystemSleepDisabled,
            &NSString::from_str("Training a model in Unsloth Studio"),
        );
        info!("Idle system sleep blocked while training");
        let _ = released.recv();
        // SAFETY: `activity` is the token beginActivityWithOptions:reason: returned.
        unsafe { process.endActivity(&activity) };
        info!("Idle system sleep allowed again");
    });
}

/// No sleep inhibitor on Linux yet, so nothing is logged as blocked.
#[cfg(not(any(windows, target_os = "macos")))]
fn hold_until(released: Receiver<()>) {
    let _ = released.recv();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn held() -> bool {
        HOLDER.lock().unwrap().is_some()
    }

    #[test]
    fn repeated_calls_hold_one_request_and_release_it() {
        set(true);
        set(true);
        assert!(held());
        set(false);
        set(false);
        assert!(!held());
    }
}
