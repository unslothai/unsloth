// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! Windows Long Paths, elevated from the startup screen instead of mid-install by setup.ps1.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
#[cfg_attr(not(windows), allow(dead_code))]
pub enum LongPathsStatus {
    Enabled,
    Disabled,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
#[cfg_attr(not(windows), allow(dead_code))]
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
pub async fn enable_windows_long_paths(webview: tauri::Webview) -> EnableLongPathsOutcome {
    let owner = owner_window(&webview);
    tokio::task::spawn_blocking(move || enable(owner))
        .await
        .unwrap_or(EnableLongPathsOutcome::Failed)
}

/// Owning the consent prompt keeps it in front of the app instead of flashing in the taskbar.
#[cfg(windows)]
fn owner_window(webview: &tauri::Webview) -> isize {
    webview.window().hwnd().map_or(0, |hwnd| hwnd.0 as isize)
}

#[cfg(not(windows))]
fn owner_window(_webview: &tauri::Webview) -> isize {
    0
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
fn enable(owner: isize) -> EnableLongPathsOutcome {
    use std::ffi::OsStr;
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::{
        CloseHandle, GetLastError, ERROR_CANCELLED, HWND, WAIT_TIMEOUT,
    };
    use windows_sys::Win32::System::Com::{
        CoInitializeEx, CoUninitialize, COINIT_APARTMENTTHREADED, COINIT_DISABLE_OLE1DDE,
    };
    use windows_sys::Win32::System::Threading::{GetExitCodeProcess, WaitForSingleObject};
    use windows_sys::Win32::UI::Shell::{
        ShellExecuteExW, SEE_MASK_NOASYNC, SEE_MASK_NOCLOSEPROCESS, SHELLEXECUTEINFOW,
    };

    fn wide(s: &OsStr) -> Vec<u16> {
        s.encode_wide().chain(std::iter::once(0)).collect()
    }

    // Turned on since the explainer opened (another installer, an admin): no prompt.
    if status() == LongPathsStatus::Enabled {
        return EnableLongPathsOutcome::Enabled;
    }
    let Some(reg_exe) = system_reg_exe() else {
        log::warn!("long paths: no system directory");
        return EnableLongPathsOutcome::Failed;
    };
    let verb = wide(OsStr::new("runas"));
    let file = wide(reg_exe.as_os_str());
    let params = wide(OsStr::new(
        r#"add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f"#,
    ));

    let mut info: SHELLEXECUTEINFOW = unsafe { std::mem::zeroed() };
    info.cbSize = std::mem::size_of::<SHELLEXECUTEINFOW>() as u32;
    info.fMask = SEE_MASK_NOCLOSEPROCESS | SEE_MASK_NOASYNC;
    info.hwnd = owner as HWND;
    info.lpVerb = verb.as_ptr();
    info.lpFile = file.as_ptr();
    info.lpParameters = params.as_ptr();
    info.nShow = 0; // SW_HIDE

    // ShellExecuteEx can route a verb through shell extensions, which expect COM on this thread.
    let com = unsafe {
        CoInitializeEx(
            std::ptr::null(),
            (COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE) as u32,
        )
    };
    let launched = unsafe { ShellExecuteExW(&mut info) } != 0;
    let error = if launched {
        0
    } else {
        unsafe { GetLastError() }
    };
    if com >= 0 {
        unsafe { CoUninitialize() };
    }
    if !launched {
        if error == ERROR_CANCELLED {
            log::info!("long paths: UAC declined");
            return EnableLongPathsOutcome::Declined;
        }
        log::warn!("long paths: ShellExecuteExW failed ({error})");
        return EnableLongPathsOutcome::Failed;
    }
    if !info.hProcess.is_null() {
        // Consent is already given here; reg.exe itself takes milliseconds.
        let mut code: u32 = 1;
        unsafe {
            if WaitForSingleObject(info.hProcess, 60_000) == WAIT_TIMEOUT {
                log::warn!("long paths: reg.exe still running after 60 s");
            } else if GetExitCodeProcess(info.hProcess, &mut code) != 0 && code != 0 {
                log::warn!("long paths: reg.exe exited {code}");
            }
            CloseHandle(info.hProcess);
        }
    }
    // The registry, not reg.exe's exit code, decides.
    outcome_from_registry()
}

/// From the system directory itself: %SystemRoot% comes from the user's environment.
#[cfg(windows)]
fn system_reg_exe() -> Option<std::path::PathBuf> {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::System::SystemInformation::GetSystemDirectoryW;

    let mut buffer = [0u16; 260];
    let len = unsafe { GetSystemDirectoryW(buffer.as_mut_ptr(), buffer.len() as u32) } as usize;
    if len == 0 || len >= buffer.len() {
        return None;
    }
    let dir = std::ffi::OsString::from_wide(&buffer[..len]);
    Some(std::path::Path::new(&dir).join("reg.exe"))
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
fn enable(_owner: isize) -> EnableLongPathsOutcome {
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

    #[cfg(windows)]
    #[test]
    fn reg_exe_comes_from_the_system_directory() {
        let reg_exe = system_reg_exe().expect("system directory");
        assert!(reg_exe.is_file(), "{}", reg_exe.display());
        let parent = reg_exe.parent().unwrap().file_name().unwrap();
        assert!(
            parent.eq_ignore_ascii_case("System32"),
            "{}",
            reg_exe.display()
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn other_platforms_never_ask() {
        assert_eq!(status(), LongPathsStatus::Unsupported);
    }
}
