// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! The "Keep computer awake" desktop setting. While it is on, the app holds the OS's block on
//! idle system sleep for as long as it runs; the display may still turn off. The OS drops the
//! block when the process exits, so a crash cannot leave the computer unable to sleep.

use log::{info, warn};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};
use tauri::Manager;

const PREFERENCE_FILE: &str = "keep-awake-v1";

const REASON: &str = "Unsloth Studio is set to keep this computer awake";

const SUPPORTED: bool = cfg!(any(
    target_os = "windows",
    target_os = "macos",
    target_os = "linux"
));

static HELD: Mutex<Option<platform::SleepBlock>> = Mutex::new(None);

fn preference_path(config_dir: &Path) -> PathBuf {
    config_dir.join(PREFERENCE_FILE)
}

fn read_preference(config_dir: &Path) -> bool {
    let path = preference_path(config_dir);
    match fs::read_to_string(&path) {
        Ok(value) => value.trim().parse::<bool>().unwrap_or_else(|error| {
            warn!(
                "Ignoring invalid keep-awake preference {}: {error}",
                path.display()
            );
            false
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => {
            warn!(
                "Could not read keep-awake preference {}: {error}",
                path.display()
            );
            false
        }
    }
}

fn write_preference(config_dir: &Path, enabled: bool) -> Result<(), String> {
    fs::create_dir_all(config_dir).map_err(|error| {
        format!(
            "Failed to create app configuration directory {}: {error}",
            config_dir.display()
        )
    })?;
    let path = preference_path(config_dir);
    fs::write(&path, format!("{enabled}\n")).map_err(|error| {
        format!(
            "Failed to save keep-awake preference {}: {error}",
            path.display()
        )
    })
}

fn config_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_config_dir()
        .map_err(|error| format!("Could not determine app configuration directory: {error}"))
}

fn is_held() -> bool {
    HELD.lock()
        .unwrap_or_else(PoisonError::into_inner)
        .is_some()
}

/// Takes or drops the sleep block. Returns whether it was held before.
fn apply(enabled: bool) -> Result<bool, String> {
    let mut held = HELD.lock().unwrap_or_else(PoisonError::into_inner);
    let was_held = held.is_some();
    if enabled != was_held {
        // Assigning None drops the old block, which releases it.
        *held = if enabled {
            Some(platform::SleepBlock::acquire()?)
        } else {
            None
        };
    }
    Ok(was_held)
}

/// Re-takes the block at launch when the setting is on. Off the setup thread: on Linux this is
/// a D-Bus round trip.
pub fn initialize(app: &tauri::AppHandle) {
    if !SUPPORTED {
        return;
    }
    let enabled = match config_dir(app) {
        Ok(dir) => read_preference(&dir),
        Err(error) => {
            warn!("{error}");
            false
        }
    };
    if !enabled {
        return;
    }
    std::thread::spawn(|| match apply(true) {
        Ok(_) => info!("Keeping the computer awake (setting is on)"),
        Err(error) => warn!("Could not keep the computer awake: {error}"),
    });
}

/// Whether the block is held right now, so a setting that failed to apply reads as off.
/// Async so the main thread never waits on the lock while a launch-time acquire is in flight.
#[tauri::command]
pub async fn get_keep_awake() -> Option<bool> {
    SUPPORTED.then(is_held)
}

#[tauri::command]
pub async fn set_keep_awake(app: tauri::AppHandle, enabled: bool) -> Result<bool, String> {
    if !SUPPORTED {
        return Err("Keep computer awake is not supported on this system".to_string());
    }
    let dir = config_dir(&app)?;
    tauri::async_runtime::spawn_blocking(move || {
        let was_held = apply(enabled)?;
        if let Err(error) = write_preference(&dir, enabled) {
            if let Err(revert) = apply(was_held) {
                warn!("Could not restore the previous keep-awake state: {revert}");
            }
            return Err(error);
        }
        info!("Keep computer awake set to {enabled}");
        Ok(enabled)
    })
    .await
    .map_err(|error| format!("Keep computer awake task failed: {error}"))?
}

#[cfg(windows)]
mod platform {
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Power::{
        PowerClearRequest, PowerCreateRequest, PowerRequestSystemRequired, PowerSetRequest,
    };
    use windows_sys::Win32::System::Threading::{
        POWER_REQUEST_CONTEXT_SIMPLE_STRING, REASON_CONTEXT, REASON_CONTEXT_0,
    };

    /// `POWER_REQUEST_CONTEXT_VERSION`, which lives behind another windows-sys feature.
    const CONTEXT_VERSION: u32 = 0;

    /// A power request, listed with its reason by `powercfg /requests`.
    pub struct SleepBlock(HANDLE);

    // SAFETY: a power request is a kernel object handle, usable and closable from any thread.
    unsafe impl Send for SleepBlock {}

    impl SleepBlock {
        pub fn acquire() -> Result<Self, String> {
            let mut reason: Vec<u16> = super::REASON.encode_utf16().chain([0]).collect();
            let context = REASON_CONTEXT {
                Version: CONTEXT_VERSION,
                Flags: POWER_REQUEST_CONTEXT_SIMPLE_STRING,
                Reason: REASON_CONTEXT_0 {
                    SimpleReasonString: reason.as_mut_ptr(),
                },
            };
            // SAFETY: `context` and the reason string it points to outlive the call.
            let handle = unsafe { PowerCreateRequest(&context) };
            if handle == INVALID_HANDLE_VALUE {
                return Err(format!(
                    "PowerCreateRequest failed: {}",
                    std::io::Error::last_os_error()
                ));
            }
            // SAFETY: `handle` is the power request created above.
            if unsafe { PowerSetRequest(handle, PowerRequestSystemRequired) } == 0 {
                let error = std::io::Error::last_os_error();
                // SAFETY: closing the request this function owns.
                unsafe { CloseHandle(handle) };
                return Err(format!("PowerSetRequest failed: {error}"));
            }
            Ok(Self(handle))
        }
    }

    impl Drop for SleepBlock {
        fn drop(&mut self) {
            // SAFETY: the handle is a live power request owned by this value.
            unsafe {
                PowerClearRequest(self.0, PowerRequestSystemRequired);
                CloseHandle(self.0);
            }
        }
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use objc2::rc::Retained;
    use objc2_foundation::NSString;
    use std::ffi::c_void;

    #[link(name = "IOKit", kind = "framework")]
    extern "C" {
        fn IOPMAssertionCreateWithName(
            assertion_type: *const c_void,
            level: u32,
            name: *const c_void,
            id: *mut u32,
        ) -> i32;
        fn IOPMAssertionRelease(id: u32) -> i32;
    }

    /// `kIOPMAssertionLevelOn`.
    const LEVEL_ON: u32 = 255;

    /// An IOKit power assertion, listed with its reason by `pmset -g assertions`.
    pub struct SleepBlock(u32);

    impl SleepBlock {
        pub fn acquire() -> Result<Self, String> {
            // NSString is toll-free bridged to the CFStringRef these take.
            let kind = NSString::from_str("PreventUserIdleSystemSleep");
            let name = NSString::from_str(super::REASON);
            let mut id = 0u32;
            // SAFETY: both strings are live for the call and `id` is a valid out pointer.
            let status = unsafe {
                IOPMAssertionCreateWithName(
                    Retained::as_ptr(&kind).cast(),
                    LEVEL_ON,
                    Retained::as_ptr(&name).cast(),
                    &mut id,
                )
            };
            if status != 0 {
                return Err(format!("IOPMAssertionCreateWithName failed: {status:#x}"));
            }
            Ok(Self(id))
        }
    }

    impl Drop for SleepBlock {
        fn drop(&mut self) {
            // SAFETY: the id is an assertion this value created and has not released.
            unsafe { IOPMAssertionRelease(self.0) };
        }
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use gio::prelude::*;
    use std::os::fd::{FromRawFd, OwnedFd};

    const CALL_TIMEOUT_MS: i32 = 5_000;
    const APP_NAME: &str = "Unsloth Studio";

    /// A desktop's session-bus inhibit service. Each drops a cookie when its holder leaves the
    /// bus. The xdg portal's Inhibit is not used: on Plasma 6 its inhibitions outlive both
    /// `Request.Close` and the caller's exit.
    pub struct SessionService {
        bus_name: &'static str,
        path: &'static str,
        interface: &'static str,
        uninhibit: &'static str,
        inhibit_args: fn() -> glib::Variant,
    }

    const SESSION_SERVICES: [SessionService; 2] = [
        // KDE Plasma (PowerDevil), Xfce, LXQt.
        SessionService {
            bus_name: "org.freedesktop.PowerManagement",
            path: "/org/freedesktop/PowerManagement/Inhibit",
            interface: "org.freedesktop.PowerManagement.Inhibit",
            uninhibit: "UnInhibit",
            inhibit_args: || (APP_NAME, super::REASON).to_variant(),
        },
        // GNOME and Cinnamon. Flag 4 blocks suspend only; the screen can still blank and lock.
        SessionService {
            bus_name: "org.gnome.SessionManager",
            path: "/org/gnome/SessionManager",
            interface: "org.gnome.SessionManager",
            uninhibit: "Uninhibit",
            inhibit_args: || (APP_NAME, 0u32, super::REASON, 4u32).to_variant(),
        },
    ];

    pub enum SleepBlock {
        /// Owns its connection: the service drops the cookie once the connection closes, and a
        /// shared session bus connection closes when its last reference goes.
        Session {
            bus: gio::DBusConnection,
            service: &'static SessionService,
            cookie: u32,
        },
        /// A logind idle inhibitor, for sessions with neither service. Held until the fd closes.
        Logind { _inhibitor: OwnedFd },
    }

    impl SleepBlock {
        pub fn acquire() -> Result<Self, String> {
            let mut errors = Vec::new();
            match gio::bus_get_sync(gio::BusType::Session, gio::Cancellable::NONE) {
                Ok(bus) => {
                    for service in &SESSION_SERVICES {
                        match session_inhibit(&bus, service) {
                            Ok(cookie) => {
                                return Ok(Self::Session {
                                    bus: bus.clone(),
                                    service,
                                    cookie,
                                })
                            }
                            Err(error) => errors.push(format!("{}: {error}", service.bus_name)),
                        }
                    }
                }
                Err(error) => errors.push(format!("session bus: {error}")),
            }
            match logind_inhibit() {
                Ok(fd) => Ok(Self::Logind { _inhibitor: fd }),
                Err(error) => {
                    errors.push(format!("logind: {error}"));
                    Err(format!(
                        "No sleep inhibitor is available ({})",
                        errors.join("; ")
                    ))
                }
            }
        }
    }

    impl Drop for SleepBlock {
        fn drop(&mut self) {
            let Self::Session {
                bus,
                service,
                cookie,
            } = self
            else {
                return;
            };
            let released = bus.call_sync(
                Some(service.bus_name),
                service.path,
                service.interface,
                service.uninhibit,
                Some(&(*cookie,).to_variant()),
                None,
                gio::DBusCallFlags::NO_AUTO_START,
                CALL_TIMEOUT_MS,
                gio::Cancellable::NONE,
            );
            if let Err(error) = released {
                log::warn!(
                    "Could not release the {} sleep inhibitor: {error}",
                    service.bus_name
                );
            }
        }
    }

    fn session_inhibit(bus: &gio::DBusConnection, service: &SessionService) -> Result<u32, String> {
        let reply = bus
            .call_sync(
                Some(service.bus_name),
                service.path,
                service.interface,
                "Inhibit",
                Some(&(service.inhibit_args)()),
                Some(glib::VariantTy::new("(u)").expect("valid variant type")),
                gio::DBusCallFlags::NO_AUTO_START,
                CALL_TIMEOUT_MS,
                gio::Cancellable::NONE,
            )
            .map_err(|error| error.to_string())?;
        reply
            .child_value(0)
            .get::<u32>()
            .ok_or_else(|| "no inhibit cookie in the reply".to_string())
    }

    fn logind_inhibit() -> Result<OwnedFd, String> {
        let bus = gio::bus_get_sync(gio::BusType::System, gio::Cancellable::NONE)
            .map_err(|error| error.to_string())?;
        let (reply, fds) = bus
            .call_with_unix_fd_list_sync(
                Some("org.freedesktop.login1"),
                "/org/freedesktop/login1",
                "org.freedesktop.login1.Manager",
                "Inhibit",
                Some(&("idle", APP_NAME, super::REASON, "block").to_variant()),
                Some(glib::VariantTy::new("(h)").expect("valid variant type")),
                gio::DBusCallFlags::NONE,
                CALL_TIMEOUT_MS,
                None::<&gio::UnixFDList>,
                gio::Cancellable::NONE,
            )
            .map_err(|error| error.to_string())?;
        let index = reply
            .child_value(0)
            .get::<glib::variant::Handle>()
            .ok_or_else(|| "logind returned no inhibitor handle".to_string())?
            .0;
        // `get` duplicates the fd, so the copy is ours to own and the list closes its own.
        let fd = fds.get(index).map_err(|error| error.to_string())?;
        // SAFETY: `fd` is a fresh duplicate nothing else owns.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
mod platform {
    pub struct SleepBlock;

    impl SleepBlock {
        pub fn acquire() -> Result<Self, String> {
            Err("Keep computer awake is not supported on this system".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preference_defaults_off_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();

        assert!(!read_preference(dir.path()));
        write_preference(dir.path(), true).unwrap();
        assert!(read_preference(dir.path()));
        write_preference(dir.path(), false).unwrap();
        assert!(!read_preference(dir.path()));
    }

    #[test]
    fn invalid_preference_falls_back_to_off() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(preference_path(dir.path()), b"maybe\n").unwrap();

        assert!(!read_preference(dir.path()));
    }
}
