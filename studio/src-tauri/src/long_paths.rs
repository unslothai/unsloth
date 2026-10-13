// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! Windows Long Paths, explained on the startup screen and elevated from here before the install,
//! instead of setup.ps1 raising an unannounced reg.exe UAC minutes into it.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LongPathsStatus {
    Enabled,
    Disabled,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EnableLongPathsOutcome {
    Enabled,
    Declined,
    Failed,
}

#[tauri::command]
pub fn windows_long_paths_status() -> LongPathsStatus {
    status()
}

#[tauri::command]
pub async fn enable_windows_long_paths() -> EnableLongPathsOutcome {
    tokio::task::spawn_blocking(enable)
        .await
        .unwrap_or(EnableLongPathsOutcome::Failed)
}

#[cfg(windows)]
fn status() -> LongPathsStatus {
    use winreg::enums::{HKEY_LOCAL_MACHINE, KEY_QUERY_VALUE};
    use winreg::RegKey;

    let enabled = RegKey::predef(HKEY_LOCAL_MACHINE)
        .open_subkey_with_flags(
            r"SYSTEM\CurrentControlSet\Control\FileSystem",
            KEY_QUERY_VALUE,
        )
        .and_then(|key| key.get_value::<u32, _>("LongPathsEnabled"))
        .is_ok_and(|value| value == 1);
    if enabled {
        LongPathsStatus::Enabled
    } else {
        LongPathsStatus::Disabled
    }
}

#[cfg(not(windows))]
fn status() -> LongPathsStatus {
    LongPathsStatus::Unsupported
}

#[cfg(windows)]
fn enable() -> EnableLongPathsOutcome {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, ERROR_CANCELLED};
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, WaitForSingleObject, INFINITE,
    };
    use windows_sys::Win32::UI::Shell::{
        ShellExecuteExW, SEE_MASK_NOASYNC, SEE_MASK_NOCLOSEPROCESS, SHELLEXECUTEINFOW,
    };

    fn wide(s: &OsStr) -> Vec<u16> {
        s.encode_wide().chain(std::iter::once(0)).collect()
    }

    // Full path: nothing earlier on PATH gets elevated.
    let system_root = std::env::var_os("SystemRoot").unwrap_or_else(|| r"C:\Windows".into());
    let reg_exe = std::path::Path::new(&system_root).join(r"System32\reg.exe");
    let verb = wide(OsStr::new("runas"));
    let file = wide(reg_exe.as_os_str());
    let params = wide(OsStr::new(
        r#"add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f"#,
    ));

    let mut info: SHELLEXECUTEINFOW = unsafe { std::mem::zeroed() };
    info.cbSize = std::mem::size_of::<SHELLEXECUTEINFOW>() as u32;
    info.fMask = SEE_MASK_NOCLOSEPROCESS | SEE_MASK_NOASYNC;
    info.lpVerb = verb.as_ptr();
    info.lpFile = file.as_ptr();
    info.lpParameters = params.as_ptr();
    info.nShow = 0; // SW_HIDE

    if unsafe { ShellExecuteExW(&mut info) } == 0 {
        let error = unsafe { GetLastError() };
        if error == ERROR_CANCELLED {
            log::info!("long paths: UAC declined");
            return EnableLongPathsOutcome::Declined;
        }
        log::warn!("long paths: ShellExecuteExW failed ({error})");
        return EnableLongPathsOutcome::Failed;
    }
    if info.hProcess.is_null() {
        return outcome_from_registry();
    }
    let mut code: u32 = 1;
    unsafe {
        WaitForSingleObject(info.hProcess, INFINITE);
        GetExitCodeProcess(info.hProcess, &mut code);
        let _ = CloseHandle(info.hProcess);
    }
    if code != 0 {
        log::warn!("long paths: reg.exe exited {code}");
    }
    outcome_from_registry()
}

#[cfg(windows)]
fn outcome_from_registry() -> EnableLongPathsOutcome {
    if status() == LongPathsStatus::Enabled {
        log::info!("long paths: enabled via UAC");
        EnableLongPathsOutcome::Enabled
    } else {
        EnableLongPathsOutcome::Failed
    }
}

#[cfg(not(windows))]
fn enable() -> EnableLongPathsOutcome {
    EnableLongPathsOutcome::Failed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_as_lowercase_strings_for_the_frontend() {
        assert_eq!(
            serde_json::to_string(&LongPathsStatus::Disabled).unwrap(),
            "\"disabled\""
        );
        assert_eq!(
            serde_json::to_string(&LongPathsStatus::Unsupported).unwrap(),
            "\"unsupported\""
        );
        assert_eq!(
            serde_json::to_string(&EnableLongPathsOutcome::Declined).unwrap(),
            "\"declined\""
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn other_platforms_never_ask() {
        assert_eq!(status(), LongPathsStatus::Unsupported);
    }
}
