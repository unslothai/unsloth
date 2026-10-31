// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! Which WebKitGTK an AppImage runs on. `linux/appimage-apprun.sh` probes the host's
//! GTK3/webkit2gtk-4.1 stack with [`PROBE_FLAG`] and, when it is complete and new enough,
//! starts the app on it instead of the bundled copy, exporting [`MODE_ENV`]`=host`.

use std::ffi::OsString;

/// Run by AppRun on the unbundled copy of this binary: report the WebKitGTK that loaded.
pub const PROBE_FLAG: &str = "--unsloth-webkit-probe";
/// Set by AppRun to `host` or `bundled`.
pub const MODE_ENV: &str = "UNSLOTH_APPIMAGE_WEBKIT_MODE";
const APPIMAGE: &str = "APPIMAGE";
/// No tray library: setup_tray would panic (libappindicator-sys dlopens one at startup).
#[cfg(target_os = "linux")]
const EXIT_NO_APPINDICATOR: i32 = 3;
/// libappindicator-sys's own order.
#[cfg(target_os = "linux")]
const APPINDICATOR_LIBRARIES: [&[u8]; 2] = [
    b"libayatana-appindicator3.so.1\0",
    b"libappindicator3.so.1\0",
];

/// True when this AppImage runs on its own bundled libraries. A host-mode AppImage loads the
/// system's stack, so renderer decisions treat it like a native install.
pub fn bundled_runtime(env: impl Fn(&str) -> Option<OsString>) -> bool {
    env(APPIMAGE).is_some() && env(MODE_ENV).as_deref() != Some("host".as_ref())
}

/// For diagnostics: never echoes the variable, so only these three values can appear.
pub fn mode_label(env: impl Fn(&str) -> Option<OsString>) -> &'static str {
    if env(APPIMAGE).is_none() {
        "none"
    } else if bundled_runtime(env) {
        "bundled"
    } else {
        "host"
    }
}

/// The probe: `Some(exit code)` when argv is exactly `[exe, PROBE_FLAG]`. Runs before any
/// GTK or logging. The process only starts when every linked library and symbol resolved
/// (the binary is linked BIND_NOW), so reaching this code already proves the stack loads.
#[cfg(target_os = "linux")]
pub fn run_probe() -> Option<i32> {
    let mut args = std::env::args_os().skip(1);
    if args.next().as_deref() != Some(PROBE_FLAG.as_ref()) || args.next().is_some() {
        return None;
    }
    if !APPINDICATOR_LIBRARIES
        .iter()
        .any(|name| library_loads(name))
    {
        return Some(EXIT_NO_APPINDICATOR);
    }
    let (major, minor, micro) = linked_webkit_version();
    println!("webkit={major}.{minor}.{micro}");
    Some(0)
}

#[cfg(target_os = "linux")]
fn library_loads(name: &[u8]) -> bool {
    // SAFETY: a NUL-terminated name; the handle is closed straight away.
    unsafe {
        let handle = libc::dlopen(name.as_ptr().cast(), libc::RTLD_LAZY | libc::RTLD_LOCAL);
        if handle.is_null() {
            return false;
        }
        libc::dlclose(handle);
        true
    }
}

/// The version of the libwebkit2gtk that actually loaded.
#[cfg(target_os = "linux")]
pub fn linked_webkit_version() -> (u32, u32, u32) {
    // SAFETY: plain getters with no preconditions; no GTK initialization needed.
    unsafe {
        (
            webkit2gtk_sys::webkit_get_major_version(),
            webkit2gtk_sys::webkit_get_minor_version(),
            webkit2gtk_sys::webkit_get_micro_version(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env<'a>(vars: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<OsString> + 'a {
        move |name| {
            vars.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| OsString::from(value))
        }
    }

    #[test]
    fn only_an_exact_host_mode_leaves_the_bundled_runtime() {
        assert!(!bundled_runtime(env(&[])));
        assert!(!bundled_runtime(env(&[(MODE_ENV, "host")])));
        for mode in [
            None,
            Some("bundled"),
            Some("HOST"),
            Some("host "),
            Some(""),
            Some("x"),
        ] {
            let mut vars = vec![(APPIMAGE, "/tmp/Unsloth.AppImage")];
            if let Some(mode) = mode {
                vars.push((MODE_ENV, mode));
            }
            assert!(bundled_runtime(env(&vars)), "{mode:?}");
        }
        assert!(!bundled_runtime(env(&[
            (APPIMAGE, "/tmp/Unsloth.AppImage"),
            (MODE_ENV, "host")
        ])));
    }

    #[test]
    fn the_diagnostics_label_is_one_of_three_values() {
        assert_eq!(mode_label(env(&[])), "none");
        assert_eq!(
            mode_label(env(&[(APPIMAGE, "/a"), (MODE_ENV, "host")])),
            "host"
        );
        for mode in ["bundled", "<script>", "host\nx"] {
            assert_eq!(
                mode_label(env(&[(APPIMAGE, "/a"), (MODE_ENV, mode)])),
                "bundled"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_linked_webkit_reports_a_plausible_version() {
        let (major, minor, _) = linked_webkit_version();
        assert_eq!(major, 2);
        assert!(minor >= 36, "{minor}");
    }
}
