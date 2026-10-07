use crate::diagnostics::{self, BackendLog, DiagnosticsState};
use log::{debug, error, info, warn};
use process_wrap::std::*;
use regex::Regex;
use std::collections::VecDeque;
use std::io::BufRead;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager};

const MAX_LOG_LINES: usize = 1000;

// Where the AppRun parks the host LD_LIBRARY_PATH it must keep away from the bundle.
#[cfg(target_os = "linux")]
const APPIMAGE_HOST_LIBRARY_PATH: &str = "UNSLOTH_HOST_LD_LIBRARY_PATH";

// Keep AppImage GUI paths and Python overrides out of managed host Python.
#[cfg(target_os = "linux")]
fn scrub_appimage_library_path() -> Option<std::ffi::OsString> {
    let appdir = std::env::var_os("APPDIR")?;
    let library_path = std::env::var_os("LD_LIBRARY_PATH")
        .or_else(|| std::env::var_os(APPIMAGE_HOST_LIBRARY_PATH))?;
    let host_paths: Vec<_> = std::env::split_paths(&library_path)
        .filter(|path| !path.starts_with(&appdir))
        .collect();
    if host_paths.is_empty() {
        None
    } else {
        std::env::join_paths(host_paths).ok()
    }
}

#[cfg(target_os = "linux")]
fn apply_scrubbed_appimage_library_path(cmd: &mut Command) {
    cmd.env_remove(APPIMAGE_HOST_LIBRARY_PATH);
    match scrub_appimage_library_path() {
        Some(library_path) => {
            cmd.env("LD_LIBRARY_PATH", library_path);
        }
        None => {
            cmd.env_remove("LD_LIBRARY_PATH");
        }
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn scrub_appimage_python_env(cmd: &mut Command) {
    if std::env::var_os("APPIMAGE").is_some() {
        apply_scrubbed_appimage_library_path(cmd);

        if let Some(path) = appdir_entries_demoted("PATH") {
            cmd.env("PATH", path);
        }
        for name in APPIMAGE_GUI_ONLY_VARS {
            cmd.env_remove(name);
        }
        cmd.env_remove("PYTHONHOME");
        cmd.env_remove("PYTHONPATH");
    }
}

// Restore a host-safe environment before launching browsers and file managers.
#[cfg(target_os = "linux")]
const APPIMAGE_GUI_ONLY_VARS: &[&str] = &[
    "FONTCONFIG_FILE",
    "GIO_MODULE_DIR",
    "GIO_EXTRA_MODULES",
    "GTK_PATH",
    "GTK_EXE_PREFIX",
    "GTK_DATA_PREFIX",
    "GTK_IM_MODULE_FILE",
    "GDK_PIXBUF_MODULE_FILE",
    "GSETTINGS_SCHEMA_DIR",
    "GST_PLUGIN_SYSTEM_PATH",
    "GST_PLUGIN_SYSTEM_PATH_1_0",
    "GST_PLUGIN_PATH",
    "GST_PLUGIN_PATH_1_0",
    "GST_PLUGIN_SCANNER",
    "GST_PLUGIN_SCANNER_1_0",
    "GST_PTP_HELPER_1_0",
    "GST_REGISTRY_REUSE_PLUGIN_SCANNER",
    "QT_PLUGIN_PATH",
    "PERLLIB",
];

#[cfg(target_os = "linux")]
fn appdir_entries_demoted(name: &str) -> Option<std::ffi::OsString> {
    let appdir = std::env::var_os("APPDIR")?;
    let value = std::env::var_os(name)?;
    let (bundled, host): (Vec<_>, Vec<_>) =
        std::env::split_paths(&value).partition(|path| path.starts_with(&appdir));
    if bundled.is_empty() {
        return None;
    }
    std::env::join_paths(host.into_iter().chain(bundled)).ok()
}

#[cfg(target_os = "linux")]
fn scrub_appimage_launcher_env(cmd: &mut Command) {
    if std::env::var_os("APPIMAGE").is_none() {
        return;
    }
    apply_scrubbed_appimage_library_path(cmd);
    if let Some(path) = appdir_entries_demoted("PATH") {
        cmd.env("PATH", path);
    }
    for name in APPIMAGE_GUI_ONLY_VARS {
        cmd.env_remove(name);
    }
    cmd.env_remove("PYTHONHOME");
    cmd.env_remove("PYTHONPATH");
}

/// Open a target detached with a host-safe environment.
#[cfg(target_os = "linux")]
pub(crate) fn open_detached(target: impl AsRef<std::ffi::OsStr>) -> std::io::Result<()> {
    let mut last_error = None;
    for mut cmd in open::commands(target.as_ref()) {
        scrub_appimage_launcher_env(&mut cmd);
        cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
        // Same double fork + setsid as the `open` crate, so the launcher outlives us; waiting
        // reaps the intermediate child.
        unsafe {
            use std::os::unix::process::CommandExt;
            cmd.pre_exec(|| {
                match libc::fork() {
                    -1 => return Err(std::io::Error::last_os_error()),
                    0 => (),
                    _ => libc::_exit(0),
                }
                if libc::setsid() == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        match cmd.spawn() {
            Ok(mut child) => {
                let _ = child.wait();
                return Ok(());
            }
            Err(error) => last_error = Some(error),
        }
    }
    Err(last_error.unwrap_or_else(|| std::io::Error::other("no launcher is available")))
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn open_detached(target: impl AsRef<std::ffi::OsStr>) -> std::io::Result<()> {
    open::that_detached(target)
}

#[cfg(target_os = "linux")]
pub(crate) fn scrub_appimage_python_env_tokio(cmd: &mut tokio::process::Command) {
    if std::env::var_os("APPIMAGE").is_some() {
        cmd.env_remove(APPIMAGE_HOST_LIBRARY_PATH);
        match scrub_appimage_library_path() {
            Some(library_path) => {
                cmd.env("LD_LIBRARY_PATH", library_path);
            }
            None => {
                cmd.env_remove("LD_LIBRARY_PATH");
            }
        }
        if let Some(path) = appdir_entries_demoted("PATH") {
            cmd.env("PATH", path);
        }
        for name in APPIMAGE_GUI_ONLY_VARS {
            cmd.env_remove(name);
        }

        cmd.env_remove("PYTHONHOME");
        cmd.env_remove("PYTHONPATH");
    }
}

#[cfg(all(test, target_os = "linux"))]
mod appimage_environment_tests {
    use super::*;
    use std::ffi::OsStr;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    fn appimage_isolated_path() -> &'static OsStr {
        OsStr::new("/tmp/.mount_Unsloth/usr/lib")
    }

    fn env_contains_name(env: &str, name: &str) -> bool {
        let prefix = format!("{name}=");
        env.lines().any(|line| line.starts_with(&prefix))
    }

    #[test]
    fn std_managed_child_drops_appimage_gui_and_python_paths() {
        let _guard = env_lock();
        let old_appimage = std::env::var_os("APPIMAGE");
        let old_appdir = std::env::var_os("APPDIR");
        let old_library_path = std::env::var_os("LD_LIBRARY_PATH");
        let old_path = std::env::var_os("PATH");
        std::env::set_var("APPIMAGE", "/tmp/Unsloth.AppImage");
        std::env::set_var("APPDIR", "/tmp/.mount_Unsloth");
        std::env::set_var(
            "LD_LIBRARY_PATH",
            "/tmp/.mount_Unsloth/usr/lib:/opt/cuda/lib64:/private/runtime",
        );
        std::env::set_var("PATH", "/tmp/.mount_Unsloth/usr/bin:/usr/bin:/bin");
        let mut cmd = Command::new("/usr/bin/env");
        cmd.env("PYTHONHOME", "/activated/python")
            .env("PYTHONPATH", "/activated/modules");
        for name in APPIMAGE_GUI_ONLY_VARS {
            cmd.env(name, format!("/tmp/.mount_Unsloth/gui-runtime/{name}"));
        }
        scrub_appimage_python_env(&mut cmd);
        let output = cmd.output().expect("run isolated child");
        let env = String::from_utf8(output.stdout).unwrap();
        assert!(env.contains("LD_LIBRARY_PATH=/opt/cuda/lib64:/private/runtime"));
        assert!(env.contains("PATH=/usr/bin:/bin:/tmp/.mount_Unsloth/usr/bin"));
        assert!(!env.contains(appimage_isolated_path().to_string_lossy().as_ref()));
        assert!(!env.contains("PYTHONHOME="));
        assert!(!env.contains("PYTHONPATH="));
        for name in APPIMAGE_GUI_ONLY_VARS {
            assert!(
                !env_contains_name(&env, name),
                "managed AppImage child inherited {name}: {env}"
            );
        }
        for (key, old_value) in [
            ("APPIMAGE", old_appimage),
            ("APPDIR", old_appdir),
            ("LD_LIBRARY_PATH", old_library_path),
            ("PATH", old_path),
        ] {
            match old_value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }

    #[test]
    fn managed_child_recovers_the_library_path_the_apprun_parked() {
        let _guard = env_lock();
        let old = ["APPIMAGE", "APPDIR", "LD_LIBRARY_PATH", APPIMAGE_HOST_LIBRARY_PATH]
            .map(|key| (key, std::env::var_os(key)));
        std::env::set_var("APPIMAGE", "/tmp/Unsloth.AppImage");
        std::env::set_var("APPDIR", "/tmp/.mount_Unsloth");
        std::env::remove_var("LD_LIBRARY_PATH");
        std::env::set_var(
            APPIMAGE_HOST_LIBRARY_PATH,
            "/tmp/.mount_Unsloth/usr/lib:/opt/rocm/lib",
        );
        let mut cmd = Command::new("/usr/bin/env");
        scrub_appimage_python_env(&mut cmd);
        let output = cmd.output().expect("run managed child");
        let env = String::from_utf8(output.stdout).unwrap();
        assert!(env.lines().any(|line| line == "LD_LIBRARY_PATH=/opt/rocm/lib"), "{env}");
        assert!(!env_contains_name(&env, APPIMAGE_HOST_LIBRARY_PATH), "{env}");
        for (key, value) in old {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }

    #[test]
    fn native_package_child_keeps_caller_environment() {
        let _guard = env_lock();
        let old_appimage = std::env::var_os("APPIMAGE");
        std::env::remove_var("APPIMAGE");
        let mut cmd = Command::new("/usr/bin/env");
        cmd.env("LD_LIBRARY_PATH", appimage_isolated_path())
            .env("PYTHONHOME", "/activated/python")
            .env("PYTHONPATH", "/activated/modules");
        for name in APPIMAGE_GUI_ONLY_VARS {
            cmd.env(name, format!("/native/gui-runtime/{name}"));
        }
        scrub_appimage_python_env(&mut cmd);
        let output = cmd.output().expect("run native-package child");
        let env = String::from_utf8(output.stdout).unwrap();
        assert!(env.contains("LD_LIBRARY_PATH=/tmp/.mount_Unsloth/usr/lib"));
        assert!(env.contains("PYTHONHOME=/activated/python"));
        assert!(env.contains("PYTHONPATH=/activated/modules"));
        for name in APPIMAGE_GUI_ONLY_VARS {
            assert!(
                env_contains_name(&env, name),
                "native-package child unexpectedly lost {name}: {env}"
            );
        }
        if let Some(value) = old_appimage {
            std::env::set_var("APPIMAGE", value);
        }
    }

    #[test]
    fn host_launcher_child_keeps_the_bundle_off_the_host_runtime() {
        let _guard = env_lock();
        let old = ["APPIMAGE", "APPDIR", "LD_LIBRARY_PATH", "PATH"]
            .map(|key| (key, std::env::var_os(key)));
        std::env::set_var("APPIMAGE", "/tmp/Unsloth.AppImage");
        std::env::set_var("APPDIR", "/tmp/.mount_Unsloth");
        std::env::set_var(
            "LD_LIBRARY_PATH",
            "/tmp/.mount_Unsloth/usr/lib:/opt/cuda/lib64",
        );
        std::env::set_var("PATH", "/tmp/.mount_Unsloth/usr/bin:/usr/bin:/bin");
        let mut cmd = Command::new("/usr/bin/env");
        cmd.env("FONTCONFIG_FILE", "/tmp/.mount_Unsloth/usr/etc/fonts/fonts.conf")
            .env("GTK_PATH", "/tmp/.mount_Unsloth/usr/lib/gtk-3.0")
            .env("QT_PLUGIN_PATH", "/tmp/.mount_Unsloth/usr/lib/qt5/plugins")
            .env("PYTHONPATH", "/tmp/.mount_Unsloth/usr/share/pyshared");
        scrub_appimage_launcher_env(&mut cmd);
        let output = cmd.output().expect("run host launcher");
        let env = String::from_utf8(output.stdout).unwrap();
        assert!(env.contains("LD_LIBRARY_PATH=/opt/cuda/lib64"));
        assert!(env.contains("PATH=/usr/bin:/bin:/tmp/.mount_Unsloth/usr/bin"));
        assert!(!env.contains("FONTCONFIG_FILE="));
        assert!(!env.contains("GTK_PATH="));
        assert!(!env.contains("QT_PLUGIN_PATH="));
        assert!(!env.contains("PYTHONPATH="));
        for (key, value) in old {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
    }
}

#[cfg(windows)]
const STUDIO_MANAGED_RUNTIME_MUTEX_PREFIX: &str = "Global\\UnslothStudioManagedEnvironment-";

pub(crate) const STUDIO_RUNTIME_GATE_HANDOFF_ENV: &str = "_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF";
pub(crate) const STUDIO_RUNTIME_GATE_BUSY: &str = "Unsloth installation is modifying the managed environment. Wait for it to finish, then start the backend again.";
const STUDIO_RUNTIME_GATE_ACQUIRE_ENV: &str = "_UNSLOTH_STUDIO_RUNTIME_GATE_ACQUIRE";

#[cfg(windows)]
#[derive(Debug)]
struct StudioManagedRuntimeLaunchGuard {
    handle: windows_sys::Win32::Foundation::HANDLE,
}

#[cfg(windows)]
impl Drop for StudioManagedRuntimeLaunchGuard {
    fn drop(&mut self) {
        unsafe {
            let _ = windows_sys::Win32::System::Threading::ReleaseMutex(self.handle);
            let _ = windows_sys::Win32::Foundation::CloseHandle(self.handle);
        }
    }
}

#[cfg(unix)]
#[derive(Debug)]
struct StudioManagedRuntimeLaunchGuard {
    file: std::fs::File,
}

#[cfg(unix)]
impl Drop for StudioManagedRuntimeLaunchGuard {
    fn drop(&mut self) {
        use std::os::fd::AsRawFd;

        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

#[cfg(windows)]
fn acquire_named_studio_runtime_launch_guard(
    name: &str,
) -> Result<StudioManagedRuntimeLaunchGuard, String> {
    const WAIT_OBJECT_0: u32 = 0x0000_0000;
    const WAIT_ABANDONED: u32 = 0x0000_0080;
    const WAIT_TIMEOUT: u32 = 0x0000_0102;

    let wide_name: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
    let handle = unsafe {
        windows_sys::Win32::System::Threading::CreateMutexW(std::ptr::null(), 0, wide_name.as_ptr())
    };
    if handle.is_null() {
        return Err(format!(
            "Could not create the Unsloth runtime lock: {}",
            std::io::Error::last_os_error()
        ));
    }

    let wait = unsafe { windows_sys::Win32::System::Threading::WaitForSingleObject(handle, 0) };
    match wait {
        WAIT_OBJECT_0 | WAIT_ABANDONED => Ok(StudioManagedRuntimeLaunchGuard { handle }),
        WAIT_TIMEOUT => {
            unsafe {
                let _ = windows_sys::Win32::Foundation::CloseHandle(handle);
            }
            Err(STUDIO_RUNTIME_GATE_BUSY.to_string())
        }
        _ => {
            let error = std::io::Error::last_os_error();
            unsafe {
                let _ = windows_sys::Win32::Foundation::CloseHandle(handle);
            }
            Err(format!(
                "Could not acquire the Unsloth runtime lock: {error}"
            ))
        }
    }
}

#[cfg(windows)]
fn studio_runtime_mutex_name_for_sid(sid: &str) -> String {
    format!("{STUDIO_MANAGED_RUNTIME_MUTEX_PREFIX}{sid}")
}

#[cfg(windows)]
fn current_windows_user_sid() -> Result<String, String> {
    use windows_sys::Win32::Security::{
        GetSidIdentifierAuthority, GetSidSubAuthority, GetSidSubAuthorityCount,
        GetTokenInformation, IsValidSid, TokenUser, TOKEN_QUERY, TOKEN_USER,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    let mut token = std::ptr::null_mut();
    if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) } == 0 {
        return Err(format!(
            "Could not open the Windows user token for the Unsloth runtime lock: {}",
            std::io::Error::last_os_error()
        ));
    }

    let result = (|| -> Result<String, String> {
        let mut required = 0_u32;
        unsafe {
            GetTokenInformation(token, TokenUser, std::ptr::null_mut(), 0, &mut required);
        }
        if required == 0 {
            return Err(format!(
                "Could not size the Windows user SID for the Unsloth runtime lock: {}",
                std::io::Error::last_os_error()
            ));
        }

        let word_size = std::mem::size_of::<usize>();
        let mut buffer = vec![0_usize; (required as usize).div_ceil(word_size)];
        if unsafe {
            GetTokenInformation(
                token,
                TokenUser,
                buffer.as_mut_ptr().cast(),
                required,
                &mut required,
            )
        } == 0
        {
            return Err(format!(
                "Could not read the Windows user SID for the Unsloth runtime lock: {}",
                std::io::Error::last_os_error()
            ));
        }

        let token_user = unsafe { &*(buffer.as_ptr().cast::<TOKEN_USER>()) };
        let sid = token_user.User.Sid;
        if sid.is_null() || unsafe { IsValidSid(sid) } == 0 {
            return Err("Windows returned an invalid user SID for the Unsloth runtime lock".into());
        }

        let authority_ptr = unsafe { GetSidIdentifierAuthority(sid) };
        let count_ptr = unsafe { GetSidSubAuthorityCount(sid) };
        if authority_ptr.is_null() || count_ptr.is_null() {
            return Err(
                "Could not inspect the Windows user SID for the Unsloth runtime lock".into(),
            );
        }
        let authority = unsafe { (*authority_ptr).Value }
            .iter()
            .fold(0_u64, |value, byte| (value << 8) | u64::from(*byte));
        let revision = unsafe { *sid.cast::<u8>() };
        let count = unsafe { *count_ptr };
        let mut sid_text = format!("S-{revision}-{authority}");
        for index in 0..u32::from(count) {
            let sub_authority = unsafe { GetSidSubAuthority(sid, index) };
            if sub_authority.is_null() {
                return Err(
                    "Could not inspect the Windows user SID for the Unsloth runtime lock".into(),
                );
            }
            sid_text.push_str(&format!("-{}", unsafe { *sub_authority }));
        }
        Ok(sid_text)
    })();

    unsafe {
        let _ = windows_sys::Win32::Foundation::CloseHandle(token);
    }
    result
}

#[cfg(windows)]
fn acquire_studio_runtime_launch_guard() -> Result<StudioManagedRuntimeLaunchGuard, String> {
    let name = studio_runtime_mutex_name_for_sid(&current_windows_user_sid()?);
    acquire_named_studio_runtime_launch_guard(&name)
}

#[cfg(unix)]
fn acquire_file_studio_runtime_launch_guard(
    home: &std::path::Path,
) -> Result<StudioManagedRuntimeLaunchGuard, String> {
    use std::os::fd::AsRawFd;

    std::fs::create_dir_all(home)
        .map_err(|error| format!("Could not create the Unsloth runtime lock directory: {error}"))?;
    let path = home.join(".studio-runtime.lock");
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&path)
        .map_err(|error| format!("Could not open the Unsloth runtime lock: {error}"))?;
    let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if result == 0 {
        return Ok(StudioManagedRuntimeLaunchGuard { file });
    }
    let error = std::io::Error::last_os_error();
    if error.kind() == std::io::ErrorKind::WouldBlock {
        return Err(STUDIO_RUNTIME_GATE_BUSY.to_string());
    }
    Err(format!("Could not acquire the Unsloth runtime lock: {error}"))
}

#[cfg(unix)]
fn acquire_studio_runtime_launch_guard() -> Result<StudioManagedRuntimeLaunchGuard, String> {
    acquire_file_studio_runtime_launch_guard(&crate::diagnostics::studio_dir())
}

/// serialize managed-environment child creation with install and repair.
#[cfg(windows)]
fn with_named_studio_runtime_launch_guard<T>(
    name: &str,
    operation: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    let _runtime_launch_guard = acquire_named_studio_runtime_launch_guard(name)?;
    operation()
}

pub(crate) fn with_studio_runtime_launch_guard<T>(
    operation: impl FnOnce() -> Result<T, String>,
) -> Result<T, String> {
    #[cfg(windows)]
    {
        let name = studio_runtime_mutex_name_for_sid(&current_windows_user_sid()?);
        return with_named_studio_runtime_launch_guard(&name, operation);
    }
    #[cfg(unix)]
    {
        let _runtime_launch_guard = acquire_studio_runtime_launch_guard()?;
        return operation();
    }
    #[cfg(not(any(windows, unix)))]
    {
        operation()
    }
}

#[cfg(all(test, unix))]
mod posix_studio_runtime_launch_guard_tests {
    use super::*;

    #[test]
    fn blocks_a_second_launcher_until_the_first_releases_the_file_lock() {
        let home = tempfile::tempdir().unwrap();
        let first = acquire_file_studio_runtime_launch_guard(home.path()).unwrap();
        let path = home.path().to_path_buf();
        let error = std::thread::spawn(move || {
            acquire_file_studio_runtime_launch_guard(&path)
                .err()
                .expect("second launcher unexpectedly acquired the gate")
        })
        .join()
        .unwrap();
        assert!(error.contains("installation is modifying"));

        drop(first);
        acquire_file_studio_runtime_launch_guard(home.path()).unwrap();
    }
}

#[cfg(windows)]
fn normalized_existing_windows_path(path: &std::path::Path) -> Result<String, String> {
    let resolved = std::fs::canonicalize(path)
        .map_err(|error| format!("Could not resolve managed Unsloth path {:?}: {error}", path))?;
    Ok(resolved
        .to_string_lossy()
        .trim_end_matches(['\\', '/'])
        .replace('/', "\\"))
}

#[cfg(windows)]
fn windows_ordinal_ignore_case_equal(left: &[u16], right: &[u16]) -> Result<bool, String> {
    use windows_sys::Win32::Globalization::{CompareStringOrdinal, CSTR_EQUAL};

    let left_length = i32::try_from(left.len())
        .map_err(|_| "Normalized Unsloth path exceeds Win32 comparison limits".to_string())?;
    let right_length = i32::try_from(right.len())
        .map_err(|_| "Normalized Unsloth path exceeds Win32 comparison limits".to_string())?;
    let comparison = unsafe {
        CompareStringOrdinal(left.as_ptr(), left_length, right.as_ptr(), right_length, 1)
    };
    if comparison == 0 {
        return Err(format!(
            "Could not compare normalized Unsloth paths: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(comparison == CSTR_EQUAL)
}

#[cfg(windows)]
fn windows_paths_are_equal(left: &str, right: &str) -> Result<bool, String> {
    let left_wide: Vec<u16> = left.encode_utf16().collect();
    let right_wide: Vec<u16> = right.encode_utf16().collect();
    windows_ordinal_ignore_case_equal(&left_wide, &right_wide)
}

#[cfg(windows)]
fn windows_path_is_within(candidate: &str, root: &str) -> Result<bool, String> {
    let candidate_wide: Vec<u16> = candidate.encode_utf16().collect();
    let root_wide: Vec<u16> = root.encode_utf16().collect();
    if candidate_wide.len() < root_wide.len() {
        return Ok(false);
    }

    let same_root =
        windows_ordinal_ignore_case_equal(&candidate_wide[..root_wide.len()], &root_wide)?;
    Ok(same_root
        && (candidate_wide.len() == root_wide.len()
            || candidate_wide[root_wide.len()] == u16::from(b'\\')))
}

#[cfg(windows)]
fn process_image_path(process_id: u32) -> Option<std::path::PathBuf> {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::System::Threading::{
        OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32,
        PROCESS_QUERY_LIMITED_INFORMATION,
    };

    let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, process_id) };
    if process.is_null() {
        return None;
    }
    let mut buffer = vec![0_u16; 32_768];
    let mut length = buffer.len() as u32;
    let ok = unsafe {
        QueryFullProcessImageNameW(
            process,
            PROCESS_NAME_WIN32,
            buffer.as_mut_ptr(),
            &mut length,
        )
    };
    unsafe {
        let _ = windows_sys::Win32::Foundation::CloseHandle(process);
    }
    if ok == 0 {
        return None;
    }
    Some(std::path::PathBuf::from(std::ffi::OsString::from_wide(
        &buffer[..length as usize],
    )))
}

/// Reject an update while a process runs from the target venv or shim. Callers must hold the
/// runtime launch mutex across the mutation: this scan finds existing users, the gate blocks new ones.
pub(crate) fn ensure_managed_environment_is_idle(
    managed_binary: &std::path::Path,
) -> Result<(), String> {
    #[cfg(not(windows))]
    {
        let _ = managed_binary;
        return Ok(());
    }

    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::{
            CloseHandle, GetLastError, ERROR_NO_MORE_FILES, INVALID_HANDLE_VALUE,
        };
        use windows_sys::Win32::System::Diagnostics::ToolHelp::{
            CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W,
            TH32CS_SNAPPROCESS,
        };

        let venv = managed_binary
            .parent()
            .and_then(std::path::Path::parent)
            .ok_or_else(|| {
                format!(
                    "Could not determine the managed Unsloth environment for {:?}",
                    managed_binary
                )
            })?;
        let studio_home = venv.parent().ok_or_else(|| {
            format!(
                "Could not determine the managed Unsloth root for {:?}",
                managed_binary
            )
        })?;
        let canonical_root = normalized_existing_windows_path(venv)?;
        let shim = studio_home.join("bin").join("unsloth.exe");
        let canonical_shim = shim
            .exists()
            .then(|| normalized_existing_windows_path(&shim))
            .transpose()?;

        let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) };
        if snapshot == INVALID_HANDLE_VALUE {
            return Err(format!(
                "Could not inspect running processes before Unsloth update: {}",
                std::io::Error::last_os_error()
            ));
        }

        let result = (|| {
            let mut entry = PROCESSENTRY32W {
                dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32,
                ..Default::default()
            };
            let mut has_entry = unsafe { Process32FirstW(snapshot, &mut entry) };
            if has_entry == 0 {
                let error = unsafe { GetLastError() };
                if error == ERROR_NO_MORE_FILES {
                    return Ok(());
                }
                return Err(format!(
                    "Could not enumerate running processes before Unsloth update: {}",
                    std::io::Error::from_raw_os_error(error as i32)
                ));
            }

            loop {
                if let Some(image) = process_image_path(entry.th32ProcessID) {
                    if let Ok(image_key) = normalized_existing_windows_path(&image) {
                        let image_is_shim = canonical_shim
                            .as_ref()
                            .map(|shim| windows_paths_are_equal(&image_key, shim))
                            .transpose()?
                            .unwrap_or(false);
                        if windows_path_is_within(&image_key, &canonical_root)? || image_is_shim {
                            let name_length = entry
                                .szExeFile
                                .iter()
                                .position(|character| *character == 0)
                                .unwrap_or(entry.szExeFile.len());
                            let name = String::from_utf16_lossy(&entry.szExeFile[..name_length]);
                            return Err(format!(
                                "The managed Unsloth environment is in use by {} (PID {}). Stop that process, then retry the update.",
                                name, entry.th32ProcessID
                            ));
                        }
                    }
                }

                has_entry = unsafe { Process32NextW(snapshot, &mut entry) };
                if has_entry == 0 {
                    let error = unsafe { GetLastError() };
                    if error == ERROR_NO_MORE_FILES {
                        break;
                    }
                    return Err(format!(
                        "Could not finish enumerating running processes before Unsloth update: {}",
                        std::io::Error::from_raw_os_error(error as i32)
                    ));
                }
            }
            Ok(())
        })();

        unsafe {
            let _ = CloseHandle(snapshot);
        }
        result
    }
}
#[cfg(all(test, windows))]
mod studio_runtime_launch_guard_tests {
    use super::*;

    #[test]
    fn blocks_a_second_launcher_until_the_first_releases_the_gate() {
        let name = format!(
            "Local\\UnslothStudioRuntimeGateTest-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let first = acquire_named_studio_runtime_launch_guard(&name).unwrap();
        let contender_name = name.clone();
        let error = std::thread::spawn(move || {
            acquire_named_studio_runtime_launch_guard(&contender_name)
                .err()
                .expect("second launcher unexpectedly acquired the gate")
        })
        .join()
        .unwrap();
        assert!(error.contains("installation is modifying"));
        drop(first);
        acquire_named_studio_runtime_launch_guard(&name).unwrap();
    }

    #[test]
    fn guarded_operation_is_skipped_while_busy_and_runs_after_release() {
        let name = format!(
            "Local\\UnslothStudioRuntimeGateOperationTest-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let first = acquire_named_studio_runtime_launch_guard(&name).unwrap();
        let invoked = Arc::new(AtomicBool::new(false));
        let contender_invoked = invoked.clone();
        let contender_name = name.clone();
        let error = std::thread::spawn(move || {
            with_named_studio_runtime_launch_guard(&contender_name, || {
                contender_invoked.store(true, Ordering::SeqCst);
                Ok(())
            })
            .unwrap_err()
        })
        .join()
        .unwrap();
        assert!(error.contains("installation is modifying"));
        assert!(!invoked.load(Ordering::SeqCst));

        drop(first);
        with_named_studio_runtime_launch_guard(&name, || {
            invoked.store(true, Ordering::SeqCst);
            Ok(())
        })
        .unwrap();
        assert!(invoked.load(Ordering::SeqCst));
    }

    #[test]
    fn guarded_operation_releases_the_gate_after_an_operation_error() {
        let name = format!(
            "Local\\UnslothStudioRuntimeGateErrorTest-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let error = with_named_studio_runtime_launch_guard(&name, || {
            Err::<(), _>("synthetic spawn failure".to_string())
        })
        .unwrap_err();
        assert_eq!(error, "synthetic spawn failure");

        with_named_studio_runtime_launch_guard(&name, || Ok(())).unwrap();
    }

    // The long-lived image is Scripts\python.exe, not unsloth.exe; both must count as in use.
    #[test]
    fn idle_scan_still_covers_a_python_hosted_studio() {
        let venv = std::env::temp_dir().join(format!(
            "unsloth-idle-scan-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let scripts = venv.join("Scripts");
        std::fs::create_dir_all(&scripts).unwrap();
        let python = scripts.join("python.exe");
        let stub = scripts.join("unsloth.exe");
        std::fs::write(&python, "").unwrap();
        std::fs::write(&stub, "").unwrap();

        let root = normalized_existing_windows_path(&venv).unwrap();
        for image in [&python, &stub] {
            let key = normalized_existing_windows_path(image).unwrap();
            assert!(
                windows_path_is_within(&key, &root).unwrap(),
                "{key} escaped the managed venv root {root}"
            );
        }

        let sibling = venv.with_file_name(format!(
            "{}-other",
            venv.file_name().unwrap().to_string_lossy()
        ));
        std::fs::create_dir_all(&sibling).unwrap();
        let sibling_key = normalized_existing_windows_path(&sibling).unwrap();
        assert!(!windows_path_is_within(&sibling_key, &root).unwrap());

        std::fs::remove_dir_all(&venv).unwrap();
        std::fs::remove_dir_all(&sibling).unwrap();
    }

    #[test]
    fn managed_environment_scan_finds_a_process_inside_the_target_root() {
        let current_exe = std::env::current_exe().unwrap();
        let target_root = current_exe.parent().unwrap();
        let managed_binary = target_root.join("Scripts").join("unsloth.exe");

        let error = ensure_managed_environment_is_idle(&managed_binary).unwrap_err();
        assert!(error.contains("managed Unsloth environment is in use"));
    }

    #[test]
    fn windows_path_containment_requires_a_component_boundary() {
        assert!(windows_path_is_within(
            r"c:\\users\\pc\\.unsloth\\studio\\unsloth_studio\\scripts\\python.exe",
            r"c:\\users\\pc\\.unsloth\\studio\\unsloth_studio"
        )
        .unwrap());
        assert!(!windows_path_is_within(
            r"c:\\users\\pc\\.unsloth\\studio\\unsloth_studio_old\\scripts\\python.exe",
            r"c:\\users\\pc\\.unsloth\\studio\\unsloth_studio"
        )
        .unwrap());
    }

    #[test]
    fn windows_path_comparison_uses_ordinal_case_insensitive_semantics() {
        assert!(windows_paths_are_equal(
            r"C:\\Users\\PC\\.Unsloth\\Studio",
            r"c:\\users\\pc\\.unsloth\\studio"
        )
        .unwrap());

        let dotted_capital_root = r"C:\\Users\\İ\\.unsloth\\studio";
        let expanded_lowercase_root = "C:\\\\Users\\\\i\u{307}\\\\.unsloth\\\\studio";
        assert_eq!(
            dotted_capital_root.to_lowercase(),
            expanded_lowercase_root.to_lowercase()
        );
        assert!(!windows_paths_are_equal(dotted_capital_root, expanded_lowercase_root).unwrap());

        let unrelated_image = format!("{expanded_lowercase_root}\\\\Scripts\\\\python.exe");
        assert!(!windows_path_is_within(&unrelated_image, dotted_capital_root).unwrap());
    }

    #[test]
    fn runtime_mutex_name_is_global_and_user_scoped() {
        let first = studio_runtime_mutex_name_for_sid("S-1-5-21-111-222-333-1001");
        let second = studio_runtime_mutex_name_for_sid("S-1-5-21-111-222-333-1002");
        assert_eq!(
            first,
            "Global\\UnslothStudioManagedEnvironment-S-1-5-21-111-222-333-1001"
        );
        assert_ne!(first, second);
        assert!(current_windows_user_sid().unwrap().starts_with("S-1-"));
    }
}

#[allow(dead_code)]
pub(crate) enum OwnedBackendHandle {
    Spawned {
        child: Box<dyn ChildWrapper + Send>,
        owner: Option<crate::desktop_backend_owner::BackendOwnerState>,
        reported_port: Option<u16>,
        pid: u32,
        generation: u64,
    },
    Adopted {
        owner: crate::desktop_backend_owner::BackendOwnerState,
        port: u16,
        pid: u32,
        generation: u64,
    },
}

#[allow(dead_code)]
impl OwnedBackendHandle {
    pub(crate) fn spawned(
        child: Box<dyn ChildWrapper + Send>,
        owner: Option<crate::desktop_backend_owner::BackendOwnerState>,
        pid: u32,
        generation: u64,
    ) -> Self {
        Self::Spawned {
            child,
            owner,
            reported_port: None,
            pid,
            generation,
        }
    }

    pub(crate) fn adopted(
        owner: crate::desktop_backend_owner::BackendOwnerState,
        port: u16,
        pid: u32,
        generation: u64,
    ) -> Self {
        Self::Adopted {
            owner,
            port,
            pid,
            generation,
        }
    }

    pub(crate) fn port(&self) -> Option<u16> {
        match self {
            Self::Spawned { reported_port, .. } => *reported_port,
            Self::Adopted { port, .. } => Some(*port),
        }
    }

    fn set_reported_port(&mut self, port: u16) {
        if let Self::Spawned {
            reported_port,
            owner,
            ..
        } = self
        {
            *reported_port = Some(port);
            if let Some(owner) = owner.as_mut() {
                if let Err(error) = owner.update_port(port) {
                    warn!("Could not update desktop backend owner metadata: {}", error);
                }
            }
        }
    }

    fn spawned_child_mut(&mut self) -> Option<&mut Box<dyn ChildWrapper + Send>> {
        match self {
            Self::Spawned { child, .. } => Some(child),
            Self::Adopted { .. } => None,
        }
    }

    fn remove_owner_metadata(self) {
        match self {
            Self::Spawned {
                owner: Some(owner), ..
            }
            | Self::Adopted { owner, .. } => owner.remove(),
            Self::Spawned { owner: None, .. } => {}
        }
    }
}

pub struct BackendProcess {
    pub owned: Option<OwnedBackendHandle>,
    pub port: Option<u16>,
    pub logs: VecDeque<String>,
    pub intentional_stop: bool,
    pub generation: u64,
    pub diagnostics_session: Option<BackendLog>,
    pub adopted_watchdog_generation: Option<u64>,
    /// Set by the start watchdog, under this mutex, once it commits to server-start-timeout; port
    /// validation then refuses to claim.
    pub start_timed_out: bool,
}

impl BackendProcess {
    pub(crate) fn has_owned_backend(&self) -> bool {
        self.owned.is_some()
    }

    pub(crate) fn has_adopted_backend(&self) -> bool {
        matches!(self.owned, Some(OwnedBackendHandle::Adopted { .. }))
    }

    pub(crate) fn owned_backend_port(&self) -> Option<u16> {
        self.owned.as_ref().and_then(OwnedBackendHandle::port)
    }
}

#[derive(Clone)]
pub(crate) struct OwnedBackendSnapshot {
    pub(crate) owner: Option<crate::desktop_backend_owner::BackendOwnerState>,
    pub(crate) port: Option<u16>,
    pub(crate) generation: u64,
    pub(crate) is_adopted: bool,
}

pub(crate) struct AdoptedBackendState {
    pub(crate) generation: u64,
    pub(crate) newly_adopted: bool,
}

pub(crate) fn adopt_verified_backend(
    state: &BackendState,
    verified: crate::desktop_backend_owner::VerifiedOwnedBackend,
) -> Result<AdoptedBackendState, String> {
    let mut proc = state.lock().map_err(|e| e.to_string())?;
    if proc.has_owned_backend() {
        if proc.owned_backend_port() == Some(verified.port) {
            proc.port = Some(verified.port);
            return Ok(AdoptedBackendState {
                generation: proc.generation,
                newly_adopted: false,
            });
        }
        return Err("Backend is already running.".to_string());
    }

    proc.generation = proc.generation.wrapping_add(1);
    proc.port = Some(verified.port);
    proc.logs.clear();
    proc.intentional_stop = false;
    proc.diagnostics_session = None;
    proc.adopted_watchdog_generation = None;
    proc.start_timed_out = false;
    proc.owned = Some(OwnedBackendHandle::adopted(
        verified.owner,
        verified.port,
        verified.backend_pid,
        verified.generation,
    ));
    Ok(AdoptedBackendState {
        generation: proc.generation,
        newly_adopted: true,
    })
}

pub(crate) fn owned_backend_snapshot(
    state: &BackendState,
) -> Result<Option<OwnedBackendSnapshot>, String> {
    let proc = state.lock().map_err(|e| e.to_string())?;
    let snapshot = match proc.owned.as_ref() {
        Some(OwnedBackendHandle::Spawned {
            owner,
            reported_port,
            ..
        }) => Some(OwnedBackendSnapshot {
            owner: owner.clone(),
            port: *reported_port,
            generation: proc.generation,
            is_adopted: false,
        }),
        Some(OwnedBackendHandle::Adopted { owner, port, .. }) => Some(OwnedBackendSnapshot {
            owner: Some(owner.clone()),
            port: Some(*port),
            generation: proc.generation,
            is_adopted: true,
        }),
        None => None,
    };
    Ok(snapshot)
}

/// Whether the handle naming *port* refers to a process that still exists; unreadable state trusts it.
pub(crate) fn owned_backend_on_port_is_running(state: &BackendState, port: u16) -> bool {
    let mut proc = match state.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let handle = match proc.owned.as_mut() {
        Some(handle) => handle,
        None => return false,
    };
    if handle.port() != Some(port) {
        return false;
    }
    match handle {
        OwnedBackendHandle::Spawned { child, .. } => !matches!(child.try_wait(), Ok(Some(_))),
        OwnedBackendHandle::Adopted { pid, .. } => backend_pid_is_running(*pid),
    }
}

/// Whether anything of ours could bind *port*: a spawned handle names no port until validated.
pub(crate) fn owned_backend_could_bind_port(state: &BackendState, port: u16) -> bool {
    let mut proc = match state.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let handle = match proc.owned.as_mut() {
        Some(handle) => handle,
        None => return false,
    };
    match handle {
        // A port it has not claimed yet is a port it may still claim.
        OwnedBackendHandle::Spawned {
            child,
            reported_port,
            ..
        } => {
            reported_port.is_none_or(|bound| bound == port)
                && !matches!(child.try_wait(), Ok(Some(_)))
        }
        OwnedBackendHandle::Adopted {
            port: owned_port,
            pid,
            ..
        } => *owned_port == port && backend_pid_is_running(*pid),
    }
}

/// False only when *pid* is provably gone, zombies included (kill(pid, 0) cannot see those).
pub(crate) fn backend_pid_is_running(pid: u32) -> bool {
    crate::desktop_backend_owner::pid_is_not_dead(pid)
        && !crate::process_identity::is_zombie(pid)
}

pub(crate) fn record_owned_backend_port_if_current(
    state: &BackendState,
    generation: u64,
    port: u16,
) -> bool {
    let mut proc = match state.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    if proc.generation != generation {
        return false;
    }
    match proc.owned.as_mut() {
        Some(OwnedBackendHandle::Spawned { .. }) => {
            proc.port = Some(port);
            if let Some(owned) = proc.owned.as_mut() {
                owned.set_reported_port(port);
            }
            true
        }
        Some(OwnedBackendHandle::Adopted {
            port: current_port, ..
        }) if *current_port == port => {
            proc.port = Some(port);
            true
        }
        _ => false,
    }
}

pub(crate) fn clear_adopted_backend_if_current(
    state: &BackendState,
    generation: u64,
    port: Option<u16>,
    reason: &str,
) -> bool {
    let mut proc = match state.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    if proc.generation != generation {
        return false;
    }
    let matches_adopted = matches!(
        proc.owned.as_ref(),
        Some(OwnedBackendHandle::Adopted { port: current_port, .. })
            if port.is_none_or(|port| port == *current_port)
    );
    if !matches_adopted {
        return false;
    }

    warn!("Clearing adopted backend state after {reason}");
    proc.owned = None;
    proc.port = None;
    proc.diagnostics_session = None;
    proc.adopted_watchdog_generation = None;
    true
}

/// Drop a spawned handle whose child provably exited (e.g. died without stdout EOF), so a launch
/// can spawn. Removes the owner file too.
pub(crate) fn clear_spawned_backend_if_exited(
    state: &BackendState,
    generation: u64,
    reason: &str,
) -> bool {
    let mut proc = match state.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    if proc.generation != generation {
        return false;
    }
    let status = match proc
        .owned
        .as_mut()
        .and_then(OwnedBackendHandle::spawned_child_mut)
    {
        // One try_wait, not a polling loop: this runs on preflight and must not block.
        Some(child) => match child.try_wait() {
            Ok(Some(status)) => status.to_string(),
            Ok(None) => return false,
            Err(error) => {
                warn!("Could not read the spawned backend's exit status: {error}");
                return false;
            }
        },
        None => return false,
    };

    warn!("Clearing spawned backend state after {reason}; the child had exited: {status}");
    if let Some(owned) = proc.owned.take() {
        owned.remove_owner_metadata();
    }
    proc.port = None;
    proc.diagnostics_session = None;
    proc.adopted_watchdog_generation = None;
    true
}

pub(crate) fn claim_adopted_watchdog_if_current(state: &BackendState, generation: u64) -> bool {
    let mut proc = match state.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    if proc.generation != generation || !proc.has_adopted_backend() {
        return false;
    }
    if proc.adopted_watchdog_generation == Some(generation) {
        return false;
    }
    proc.adopted_watchdog_generation = Some(generation);
    true
}

pub(crate) fn clear_adopted_watchdog_if_current(state: &BackendState, generation: u64) {
    let mut proc = match state.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    if proc.generation == generation && proc.adopted_watchdog_generation == Some(generation) {
        proc.adopted_watchdog_generation = None;
    }
}

impl Default for BackendProcess {
    fn default() -> Self {
        Self {
            owned: None,
            port: None,
            logs: VecDeque::with_capacity(MAX_LOG_LINES),
            intentional_stop: false,
            generation: 0,
            diagnostics_session: None,
            adopted_watchdog_generation: None,
            start_timed_out: false,
        }
    }
}

pub type BackendState = Arc<Mutex<BackendProcess>>;
pub type ShutdownFlag = Arc<AtomicBool>;

pub fn new_backend_state() -> BackendState {
    Arc::new(Mutex::new(BackendProcess::default()))
}

pub fn new_shutdown_flag() -> ShutdownFlag {
    Arc::new(AtomicBool::new(false))
}

pub(crate) fn trim_line_endings(bytes: &[u8]) -> &[u8] {
    let mut end = bytes.len();
    while end > 0 && matches!(bytes[end - 1], b'\n' | b'\r') {
        end -= 1;
    }
    &bytes[..end]
}

/// Longest line handed to `tauri.log`; matches the phase log's cap.
pub(crate) const MAX_BACKEND_LOG_LINE_BYTES: usize = 16 * 1024;

/// Keep only the last non-empty frame of a `\r` progress redraw.
/// Same rule as `_TeeStream._last_frame` in studio/backend/run.py.
pub(crate) fn collapse_progress_frames(text: &str) -> &str {
    if !text.contains('\r') {
        return text;
    }
    text.rsplit('\r')
        .find(|frame| !frame.trim().is_empty())
        // Return a frame, not the whole text, which still holds the `\r`.
        .unwrap_or_else(|| text.rsplit('\r').next().unwrap_or(""))
}

/// True for the backend's own 2xx access-log records, already in the phase and session logs.
/// Non-2xx and unparseable records stay at INFO so failures reach tauri.log.
fn is_backend_access_log_line(text: &str) -> bool {
    let trimmed = text.trim_start();
    if !trimmed.starts_with('{') {
        return false;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        return false;
    };
    if value.get("event").and_then(|event| event.as_str()) != Some("request_completed") {
        return false;
    }
    matches!(
        value.get("status_code").and_then(|code| code.as_u64()),
        Some(200..=299)
    )
}

/// Windows `CREATE_NO_WINDOW` flag: suppresses console windows for children.
#[cfg(windows)]
pub(crate) const CREATE_NO_WINDOW: u32 = 0x08000000;

/// Force-kill a process tree via hidden `taskkill /T /F`, falling back to `child.kill()`; reaps after.
#[cfg(windows)]
pub(crate) fn force_kill_process_tree(
    pid: u32,
    child: &mut Box<dyn ChildWrapper + Send>,
    label: &str,
) {
    use std::os::windows::process::CommandExt;

    let taskkill_status = Command::new("taskkill.exe")
        .creation_flags(CREATE_NO_WINDOW)
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    match taskkill_status {
        Ok(status) if status.success() => {}
        Ok(status) => {
            warn!(
                "taskkill returned non-zero status for {} pid {}: {}",
                label, pid, status
            );
            let _ = child.kill();
        }
        Err(e) => {
            warn!("taskkill failed for {} pid {}: {}", label, pid, e);
            let _ = child.kill();
        }
    }

    let _ = child.wait();
    info!("{} process tree force stopped", label);
}

/// Whether site-packages holds something the trampoline could import (package dir, or dist-info for
/// editable installs). Filesystem-only, since this runs on the launch path.
#[cfg(windows)]
fn windows_site_packages_carries_the_cli(site_packages: &std::path::Path) -> bool {
    if site_packages.join("unsloth_cli").exists() {
        return true;
    }
    let Ok(entries) = std::fs::read_dir(site_packages) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        name.starts_with("unsloth-") && name.ends_with(".dist-info")
    })
}

/// Managed venv unsloth binary: new layout (unsloth_studio/) first, then legacy .venv/.
pub(crate) fn find_unsloth_binary_in_studio_dir(
    studio: &std::path::Path,
) -> Option<std::path::PathBuf> {
    let new_base = studio.join("unsloth_studio");
    let old_base = studio.join(".venv");

    let bases = [new_base, old_base];

    // Three passes, since an interrupted migration can leave half of either layout:
    // 1) launcher+interpreter, 2) (Windows) interpreter only, package-holding first, 3) launcher only.
    for base in &bases {
        #[cfg(unix)]
        let bin = base.join("bin").join("unsloth");
        #[cfg(windows)]
        let bin = base.join("Scripts").join("unsloth.exe");

        #[cfg(unix)]
        let complete = bin.exists();
        #[cfg(windows)]
        let complete = bin.exists() && base.join("Scripts").join("python.exe").exists();

        if complete {
            return Some(bin);
        }
    }

    // Package-holding interpreter first; a stat, not an import probe, since this runs on the launch path.
    #[cfg(windows)]
    for base in &bases {
        if base.join("Scripts").join("python.exe").exists()
            && windows_site_packages_carries_the_cli(&base.join("Lib").join("site-packages"))
        {
            return Some(base.join("Scripts").join("unsloth.exe"));
        }
    }

    #[cfg(windows)]
    for base in &bases {
        if base.join("Scripts").join("python.exe").exists() {
            return Some(base.join("Scripts").join("unsloth.exe"));
        }
    }

    #[cfg(windows)]
    for base in &bases {
        let bin = base.join("Scripts").join("unsloth.exe");
        if bin.exists() {
            return Some(bin);
        }
    }

    None
}

pub fn find_unsloth_binary() -> Option<std::path::PathBuf> {
    let home = dirs::home_dir()?;
    let studio = home.join(".unsloth").join("studio");

    find_unsloth_binary_in_studio_dir(&studio)
}

/// Run the CLI via python.exe: Application Control blocks the unsigned unsloth.exe wrapper.
/// No -I (it implies -E); instead drop the implicit cwd sys.path[0], except under safe_path
/// (3.11+, hence getattr).
/// -X utf8 deliberately diverges from the console script. sys.argv[0] is set before the import:
/// unsloth_cli checks it at import time.
#[cfg(windows)]
pub(crate) const WINDOWS_CLI_ENTRYPOINT: &str =
    "import sys, os; sys.path[:1] = [x for x in sys.path[:1] if getattr(sys.flags, 'safe_path', False) or x not in ('', os.getcwd())]; sys.argv[0] = 'unsloth'; from unsloth_cli import app; sys.exit(app())";

/// Program and argv that run the managed CLI without executing `bin` (POSIX keeps `bin`).
#[derive(Debug)]
pub(crate) struct ManagedCliInvocation {
    pub program: std::path::PathBuf,
    pub args: Vec<std::ffi::OsString>,
}

impl ManagedCliInvocation {
    /// The single place an invocation becomes a process.
    pub(crate) fn to_command(&self) -> Command {
        let mut cmd = Command::new(&self.program);
        cmd.args(&self.args);
        cmd
    }
}

/// Fails closed on Windows without the interpreter rather than falling back to the blocked stub.
pub(crate) fn resolve_managed_cli_invocation(
    bin: &std::path::Path,
    args: &[&str],
) -> Result<ManagedCliInvocation, String> {
    resolve_managed_cli_invocation_with(bin, args, Isolation::Inherit)
}

/// Only the updater runs isolated (-I): it rewrites the environment it runs in. Everything else
/// inherits the ambient Python env like the console script.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Isolation {
    Inherit,
    Isolated,
}

pub(crate) fn resolve_managed_cli_invocation_with(
    bin: &std::path::Path,
    args: &[&str],
    isolation: Isolation,
) -> Result<ManagedCliInvocation, String> {
    #[cfg(windows)]
    {
        let python = bin
            .parent()
            .ok_or_else(|| "Managed Unsloth executable has no parent directory.".to_string())?
            .join("python.exe");
        if !python.is_file() {
            return Err(format!(
                "Managed Python interpreter not found beside Unsloth: {}",
                python.display()
            ));
        }
        // No -I (see WINDOWS_CLI_ENTRYPOINT). -X utf8 goes before -I: -I implies -E, which drops
        // PYTHONUTF8.
        let mut argv: Vec<std::ffi::OsString> = match isolation {
            Isolation::Inherit => vec!["-X", "utf8", "-c", WINDOWS_CLI_ENTRYPOINT],
            Isolation::Isolated => vec!["-X", "utf8", "-I", "-c", WINDOWS_CLI_ENTRYPOINT],
        }
        .into_iter()
        .map(std::ffi::OsString::from)
        .collect();
        argv.extend(args.iter().copied().map(std::ffi::OsString::from));
        Ok(ManagedCliInvocation {
            program: python,
            args: argv,
        })
    }

    #[cfg(not(windows))]
    {
        let _ = isolation;
        Ok(ManagedCliInvocation {
            program: bin.to_path_buf(),
            args: args.iter().copied().map(std::ffi::OsString::from).collect(),
        })
    }
}

/// Blocking flavour of [`resolve_managed_cli_invocation`].
pub(crate) fn build_managed_cli_command(
    bin: &std::path::Path,
    args: &[&str],
) -> Result<Command, String> {
    build_managed_cli_command_with(bin, args, Isolation::Inherit)
}

pub(crate) fn build_managed_cli_command_with(
    bin: &std::path::Path,
    args: &[&str],
    isolation: Isolation,
) -> Result<Command, String> {
    let cmd = resolve_managed_cli_invocation_with(bin, args, isolation)?.to_command();
    // PYTHONHOME/PYTHONPATH deliberately kept: the console script honours both.
    Ok(cmd)
}

/// Async flavour of [`resolve_managed_cli_invocation`], for the probe and
/// provisioning call sites that already drive tokio children.
pub(crate) fn build_managed_cli_command_tokio(
    bin: &std::path::Path,
    args: &[&str],
) -> Result<tokio::process::Command, String> {
    let invocation = resolve_managed_cli_invocation(bin, args)?;
    let mut cmd = tokio::process::Command::new(&invocation.program);
    cmd.args(&invocation.args);
    // PYTHONHOME / PYTHONPATH left alone, for the reason in the blocking flavour.
    Ok(cmd)
}

/// Whether the user's profile is reachable at all: the managed install lives
/// under it, so an unmounted roaming profile looks like no install.
pub(crate) fn home_dir_available() -> Result<(), String> {
    usable_home_dir(dirs::home_dir(), &windows_roots(), true).map(|_| ())
}

/// One policy for both callers, so a SYSTEM account is never sent to an install it cannot start.
fn usable_home_dir(
    home: Option<std::path::PathBuf>,
    windirs: &[std::path::PathBuf],
    require_existing: bool,
) -> Result<std::path::PathBuf, String> {
    let home = home.ok_or_else(|| "Could not determine the home directory".to_string())?;

    // A SYSTEM account's home is under the Windows tree
    // (C:\Windows\System32\config\systemprofile), the folder the CLI rejects.
    if is_inside_windows_dir(&home, windirs) {
        return Err(format!(
            "Home directory {} is inside the Windows directory",
            home.display()
        ));
    }

    // Only the installer may create the profile; elsewhere a missing home is an unmounted roaming profile.
    if require_existing && !home.is_dir() {
        return Err(format!(
            "Home directory {} is not reachable",
            home.display()
        ));
    }
    Ok(home)
}

/// Set on every desktop-owned CLI child; the Python CLI reads it.
pub(crate) const DESKTOP_MANAGED_ENV: &str = "UNSLOTH_DESKTOP_MANAGED";

/// Login start runs from C:\Windows\system32, which the CLI refuses, so pick the directory explicitly.
pub(crate) fn managed_cli_working_dir_from(
    home: Option<std::path::PathBuf>,
    windirs: &[std::path::PathBuf],
) -> Result<std::path::PathBuf, String> {
    working_dir_under(home, windirs, true)
}

/// Installer variant: may create the profile it runs under.
pub(crate) fn install_working_dir(
    home: Option<std::path::PathBuf>,
) -> Result<std::path::PathBuf, String> {
    working_dir_under(home, &[], false)
}

fn working_dir_under(
    home: Option<std::path::PathBuf>,
    windirs: &[std::path::PathBuf],
    require_existing_home: bool,
) -> Result<std::path::PathBuf, String> {
    let home = usable_home_dir(home, windirs, require_existing_home)?;

    let work_dir = home.join(".unsloth");
    if !work_dir.exists() {
        std::fs::create_dir_all(&work_dir)
            .map_err(|e| format!("Failed to create {}: {}", work_dir.display(), e))?;
    }
    if !work_dir.is_dir() {
        return Err(format!("{} is not a directory", work_dir.display()));
    }
    Ok(work_dir)
}

// Case-insensitive, either separator, no trailing one, no \\?\ prefix. Not
// cfg-gated, so the check stays unit-testable from Linux CI.
fn normalize_windows_path(path: &std::path::Path) -> String {
    // Lowercased first: \\?\unc\ is accepted too, and must not read as relative.
    let text = path.to_string_lossy().replace('/', "\\").to_lowercase();
    let text = match text.strip_prefix("\\\\?\\unc\\") {
        Some(rest) => format!("\\\\{rest}"),
        None => text.strip_prefix("\\\\?\\").unwrap_or(&text).to_string(),
    };
    text.trim_end_matches('\\').to_string()
}

fn is_inside_windows_dir(path: &std::path::Path, windirs: &[std::path::PathBuf]) -> bool {
    let normalized = normalize_windows_path(path);
    windirs.iter().any(|windir| {
        let root = normalize_windows_path(windir);
        // "c:" (a WINDIR of "C:\") would otherwise match the whole drive.
        root.len() > 2 && (normalized == root || normalized.starts_with(&(root.clone() + "\\")))
    })
}

/// The inherited cwd unless unusable (a system folder on Windows). Resolved per call; never a temp dir.
pub(crate) fn managed_cli_working_dir() -> Result<std::path::PathBuf, String> {
    let windirs = windows_roots();
    let appdir = std::env::var_os("APPDIR").map(std::path::PathBuf::from);
    if let Ok(cwd) = std::env::current_dir() {
        if !is_unusable_cwd(&cwd, &windirs, appdir.as_deref()) {
            return Ok(cwd);
        }
    }
    managed_cli_working_dir_from(dirs::home_dir(), &windirs)
}

/// Only the directories the CLI guard refuses (C:\Windows\Temp stays allowed), plus the AppImage's
/// read-only `$APPDIR/usr`.
fn is_unusable_cwd(
    path: &std::path::Path,
    windirs: &[std::path::PathBuf],
    appdir: Option<&std::path::Path>,
) -> bool {
    if appdir.is_some_and(|appdir| path.starts_with(appdir)) {
        return true;
    }
    windirs.iter().any(|windir| {
        ["System32", "SysWOW64"]
            .iter()
            .any(|name| is_inside_windows_dir(path, &[windir.join(name)]))
    })
}

/// Windows roots, checked to hold System32 since WINDIR is settable. Empty off Windows.
fn windows_roots() -> Vec<std::path::PathBuf> {
    if !cfg!(windows) {
        return Vec::new();
    }
    let system_root = std::env::var("SystemRoot").ok();
    let fallback = system_root
        .clone()
        .unwrap_or_else(|| r"C:\Windows".to_string());
    windows_roots_from(
        [
            system_root,
            std::env::var("WINDIR").ok(),
            Some(r"C:\Windows".to_string()),
        ]
        .into_iter()
        .flatten()
        .map(std::path::PathBuf::from)
        .collect(),
        std::path::PathBuf::from(fallback),
        |root| root.join("System32").is_dir(),
    )
}

fn windows_roots_from(
    candidates: Vec<std::path::PathBuf>,
    fallback: std::path::PathBuf,
    is_windows_dir: impl Fn(&std::path::Path) -> bool,
) -> Vec<std::path::PathBuf> {
    let mut roots: Vec<std::path::PathBuf> = Vec::new();
    for root in candidates {
        if is_windows_dir(&root) && !roots.contains(&root) {
            roots.push(root);
        }
    }
    if roots.is_empty() {
        // No Windows install found: keep the check alive on a non-settable value.
        roots.push(fallback);
    }
    roots
}

/// Single-path overrides that a relative value makes cwd-dependent.
/// Must stay identical to `_RELATIVE_PATH_ENV` in unsloth_cli/_system_dir_guard.py (parity test).
pub(crate) const RELATIVE_PATH_ENV: &[&str] = &[
    "UNSLOTH_HOME",
    "UNSLOTH_STUDIO_HOME",
    "STUDIO_HOME",
    "UNSLOTH_STUDIO_DOCUMENTS_HOME",
    "UNSLOTH_STUDIO_PROJECTS_HOME",
    "UNSLOTH_STUDIO_SANDBOX_HOME",
    "STUDIO_LOCAL_REPO",
    "UNSLOTH_LLAMA_CPP_PATH",
    "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR",
    "UNSLOTH_SD_CPP_PATH",
    "UNSLOTH_WHISPER_CPP_PATH",
    "UNSLOTH_AUDIO_CPP_PATH",
    "LLAMA_SERVER_PATH",
    "WHISPER_SERVER_PATH",
    "AUDIOCPP_SERVER_PATH",
    "SD_CLI_PATH",
    "SD_SERVER_PATH",
    "LLAMA_ARG_MODEL",
    "LLAMA_ARG_MMPROJ",
    "LLAMA_ARG_MODEL_DRAFT",
    "LLAMA_ARG_SPEC_DRAFT_MODEL",
    "AMDGPU_ASIC_ID_TABLE_PATH",
    "VLLM_CACHE_ROOT",
    "GGML_BACKEND_PATH",
    "CUDA_PATH",
    "HIP_PATH",
    "HIP_PATH_57",
    "ROCM_PATH",
    "MLX_HOSTFILE",
    // Read exactly like MLX_HOSTFILE: either inline JSON or a filename.
    "MLX_IBV_DEVICES",
    "OLLAMA_MODELS",
    "DG_VISUAL_BIN",
    "UNSLOTH_DG_SHIM",
    "UNSLOTH_COMPILE_LOCATION",
    "TORCHINDUCTOR_CACHE_DIR",
    // Filled only when blank, so a relative user value is kept as written.
    "TORCH_EXTENSIONS_DIR",
    "TORCH_HOME",
    "TRITON_HOME",
    "TRITON_CACHE_DIR",
    "TRITON_DUMP_DIR",
    "CUDA_CACHE_PATH",
    "MPLCONFIGDIR",
    "NUMBA_CACHE_DIR",
    "DATA_DESIGNER_HOME",
    "DATA_DESIGNER_MANAGED_ASSETS_PATH",
    "UNSLOTH_DIFFUSION_COMPILE_CACHE_DIR",
    "UNSLOTH_DIFFUSION_COND_CACHE_DIR",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_XET_CACHE",
    "HF_DATASETS_CACHE",
    "HF_ASSETS_CACHE",
    // transformers appends this to sys.path.
    "HF_MODULES_CACHE",
    "HF_TOKEN_PATH",
    "UV_CACHE_DIR",
    "TRANSFORMERS_CACHE",
    "SENTENCE_TRANSFORMERS_HOME",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "UNSLOTH_STUDIO_CHILD_RECORD",
    "UNSLOTH_LLAMA_INSTALLER",
    "CUDA_HOME",
    "CUDA_ROOT",
];

/// Path lists: one relative entry changes the whole list. PATH is deliberately absent.
pub(crate) const PATH_LIST_ENV: &[&str] = &[
    "UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH",
    "CUDA_RUNTIME_DLL_DIR",
    "PYTHONPATH",
];

/// Windows rules on every platform; matches `_is_fully_qualified` in the CLI guard.
fn is_fully_qualified(value: &str) -> bool {
    let lowered = value.to_lowercase();
    if lowered.starts_with("\\\\?\\unc\\") {
        return true;
    }
    let value = if lowered.starts_with("\\\\?\\") {
        &value[4..]
    } else {
        value
    };
    if value.starts_with("\\\\") || value.starts_with("//") {
        return true;
    }
    // Spelled out (not Path::is_absolute) so Linux CI gives the Windows answer.
    matches!(value.as_bytes(), [drive, b':', sep, ..]
        if drive.is_ascii_alphabetic() && (*sep == b'\\' || *sep == b'/'))
}

/// posixpath.expandvars semantics: unset names and stray `$` are left as written.
fn expand_posix_vars(value: &str, lookup: &impl Fn(&str) -> Option<String>) -> String {
    let bytes = value.as_bytes();
    let mut out = String::with_capacity(value.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'$' {
            let start = index;
            while index < bytes.len() && bytes[index] != b'$' {
                index += 1;
            }
            out.push_str(&value[start..index]);
            continue;
        }
        let rest = &value[index + 1..];
        let (name, consumed) = if let Some(rest) = rest.strip_prefix('{') {
            match rest.find('}') {
                Some(end) => (&rest[..end], end + 2),
                None => {
                    out.push_str(&value[index..]);
                    break;
                }
            }
        } else {
            let end = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(rest.len());
            (&rest[..end], end)
        };
        if name.is_empty() {
            out.push('$');
            index += 1;
            continue;
        }
        match lookup(name) {
            Some(value) => out.push_str(&value),
            None => out.push_str(&value[index..index + consumed + 1]),
        }
        index += consumed + 1;
    }
    out
}

/// Windows rules on Windows, POSIX off it; a Windows-shaped value stays Windows-judged.
fn is_cwd_independent(value: &str, windows: bool) -> bool {
    if windows {
        is_fully_qualified(value)
    } else {
        value.starts_with('/')
    }
}

/// The most a Windows environment variable holds, terminator included.
const WINDOWS_ENV_VALUE_LIMIT: usize = 32_767;

/// What separates the entries of a path list, as os.pathsep spells it.
fn path_list_separator(windows: bool) -> char {
    if windows {
        ';'
    } else {
        ':'
    }
}

/// Whether the value depends on process state `join` cannot see: the current
/// directory of a drive ("D:cache") or of the current drive ("\\cache").
fn needs_os_resolution(value: &str) -> bool {
    if value.starts_with('\\') || value.starts_with('/') {
        return true;
    }
    matches!(value.as_bytes(), [drive, b':', ..] if drive.is_ascii_alphabetic())
}

/// MLX_HOSTFILE holds either a filename or the host list itself, as JSON.
const INLINE_JSON_ENV: &[&str] = &["MLX_HOSTFILE", "MLX_IBV_DEVICES"];

/// Names whose readers disagree about %VAR% (huggingface_hub expands, hf_cache_settings does not);
/// expanding first makes both see one path. Scoped since "%data%" is a legal folder name.
const EXPANDED_ENV: &[&str] = &[
    "HF_HOME",
    "HF_TOKEN_PATH",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_ASSETS_CACHE",
    "XDG_CACHE_HOME",
    "SENTENCE_TRANSFORMERS_HOME",
];

/// The pre-quant allowlist skips a bare on/off token so there is no allow-all
/// mode; anchoring one would turn it into a real allowlisted directory.
const TOGGLE_ENV: &[&str] = &["UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH"];

/// Mirrors ntpath.expandvars, pattern included: `'[^']*'?|%(%|[^%]*%?)|\$(\$|[-\w]+|\{[^}]*\}?)`.
fn expand_windows_vars(value: &str, lookup: &impl Fn(&str) -> Option<String>) -> String {
    let bytes = value.as_bytes();
    let mut out = String::with_capacity(value.len());
    let mut index = 0;
    while index < bytes.len() {
        // Only taken for the three ASCII markers below, where index + 1 is
        // always a character boundary.
        match bytes[index] {
            b'\'' => {
                // A quoted run is copied through unexpanded.
                let rest = &value[index + 1..];
                let end = rest
                    .find('\'')
                    .map_or(value.len(), |offset| index + 1 + offset + 1);
                out.push_str(&value[index..end]);
                index = end;
            }
            b'%' => {
                let rest = &value[index + 1..];
                if rest.starts_with('%') {
                    out.push('%');
                    index += 2;
                    continue;
                }
                match rest.find('%') {
                    Some(offset) => {
                        let end = index + 1 + offset + 1;
                        match lookup(&rest[..offset]) {
                            Some(expanded) => out.push_str(&expanded),
                            None => out.push_str(&value[index..end]),
                        }
                        index = end;
                    }
                    // No closing %, so nothing here is a reference.
                    None => {
                        out.push_str(&value[index..]);
                        index = value.len();
                    }
                }
            }
            b'$' => {
                let rest = &value[index + 1..];
                if rest.starts_with('$') {
                    out.push('$');
                    index += 2;
                    continue;
                }
                if rest.starts_with('{') {
                    match rest.find('}') {
                        Some(offset) => {
                            let end = index + 1 + offset + 1;
                            match lookup(&rest[1..offset]) {
                                Some(expanded) => out.push_str(&expanded),
                                None => out.push_str(&value[index..end]),
                            }
                            index = end;
                        }
                        None => {
                            out.push_str(&value[index..]);
                            index = value.len();
                        }
                    }
                    continue;
                }
                // \w under re.ASCII, plus the hyphen ntpath allows.
                let end = rest
                    .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '-'))
                    .unwrap_or(rest.len());
                if end == 0 {
                    out.push('$');
                    index += 1;
                    continue;
                }
                match lookup(&rest[..end]) {
                    Some(expanded) => out.push_str(&expanded),
                    None => out.push_str(&value[index..index + 1 + end]),
                }
                index += 1 + end;
            }
            _ => {
                let step = value[index..].chars().next().map_or(1, char::len_utf8);
                out.push_str(&value[index..index + step]);
                index += step;
            }
        }
    }
    out
}

/// `value` expanded once, or None if a second pass would still change it (nested, escaped, self-ref).
/// The twin of `_expand_settled` in the CLI guard.
fn expand_settled(
    value: &str,
    lookup: &impl Fn(&str) -> Option<String>,
    windows: bool,
) -> Option<String> {
    let expand = |value: &str| {
        if windows {
            expand_windows_vars(value, lookup)
        } else {
            expand_posix_vars(value, lookup)
        }
    };
    let expanded = expand(value);
    (expand(&expanded) == expanded).then_some(expanded)
}

fn names_a_path(name: &str, value: &str) -> bool {
    if INLINE_JSON_ENV.contains(&name) && (value.starts_with('[') || value.starts_with('{')) {
        return false;
    }
    if TOGGLE_ENV.contains(&name)
        && matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on" | "0" | "false" | "no" | "off"
        )
    {
        return false;
    }
    true
}

/// Removed before every managed spawn: Tauri uses the legacy Unsloth root regardless.
/// UNSLOTH_HOME and UNSLOTH_PORTABLE also move roots and caches.
pub(crate) const MANAGED_CHILD_SCRUBBED_ENV: &[&str] = &[
    "UNSLOTH_HOME",
    "UNSLOTH_STUDIO_HOME",
    "STUDIO_HOME",
    "UNSLOTH_PORTABLE",
];

/// Only read by the update/installer path, so a stale value must not fail other spawns.
const UPDATE_ONLY_ENV: &[&str] = &["STUDIO_LOCAL_REPO"];

/// `~`/`~name` as ntpath.expanduser resolves them; written out since llama_cpp.py does not expand.
fn expand_windows_user(
    value: &str,
    home: &std::path::Path,
    username: Option<&str>,
) -> String {
    if !value.starts_with('~') {
        return value.to_string();
    }
    let end = value[1..]
        .find(['\\', '/'])
        .map_or(value.len(), |offset| offset + 1);
    let (name, rest) = (&value[1..end], &value[end..]);
    let home = home.to_string_lossy();
    let base = if name.is_empty() {
        home.into_owned()
    } else {
        // ntpath only guesses a sibling when this profile is named after the current user. Split on the
        // string: these are Windows paths on any platform.
        let cut = match home.rfind(['\\', '/']) {
            Some(cut) => cut,
            None => return value.to_string(),
        };
        let this_profile = &home[cut + 1..];
        match username {
            Some(user) if user == name => home.clone().into_owned(),
            Some(user) if user == this_profile => format!("{}{}", &home[..cut + 1], name),
            _ => return value.to_string(),
        }
    };
    format!("{}{}", base, rest)
}

/// `~`, `~/rest`, `~name/rest` off Windows via getpwnam_r, the lookup preflight::managed uses,
/// so child pinning and the fingerprint agree. Unknown names are left as is.
fn expand_posix_user(value: &str, home: Option<&std::path::Path>) -> String {
    if !value.starts_with('~') {
        return value.to_string();
    }
    let end = value[1..].find('/').map_or(value.len(), |offset| offset + 1);
    let (name, rest) = (&value[1..end], &value[end..]);
    if name.is_empty() {
        return match home {
            Some(home) => format!("{}{}", home.to_string_lossy(), rest),
            None => value.to_string(),
        };
    }
    match crate::preflight::managed::named_user_home(value, home) {
        Some(resolved) => resolved.to_string_lossy().into_owned(),
        None => value.to_string(),
    }
}

/// Windows rules on Windows, POSIX off it.
fn expand_user(
    value: &str,
    home: Option<&std::path::Path>,
    username: Option<&str>,
    windows: bool,
) -> String {
    if windows {
        match home {
            Some(home) => expand_windows_user(value, home, username),
            None => value.to_string(),
        }
    } else {
        expand_posix_user(value, home)
    }
}

fn relative_override_pins_from(
    cwd: Option<std::path::PathBuf>,
    work_dir: &std::path::Path,
    lookup: impl Fn(&str) -> Option<String>,
    absolute: impl Fn(&str) -> Option<std::path::PathBuf>,
    home: Option<&std::path::Path>,
    skipped: &[&str],
    windows: bool,
) -> Result<Vec<(&'static str, std::path::PathBuf)>, String> {
    // Written out so readers that do and do not expand %VAR% land in the same folder.
    let username = lookup("USERNAME");
    // USERPROFILE, matching ntpath.expanduser; dirs::home_dir() reads the known folder.
    let tilde_home = lookup("USERPROFILE")
        .map(std::path::PathBuf::from)
        .or_else(|| home.map(|home| home.to_path_buf()));
    let home = tilde_home.as_deref();
    // Readers expand once; if that does not settle the value the move is refused.
    let expanded = |name: &str, value: &str| -> Result<Option<String>, String> {
        if !EXPANDED_ENV.contains(&name) {
            return Ok(Some(value.to_string()));
        }
        match expand_settled(value, &lookup, windows) {
            Some(settled) => Ok(Some(settled)),
            None => {
                let once = if windows {
                    expand_windows_vars(value, &lookup)
                } else {
                    expand_posix_vars(value, &lookup)
                };
                if is_cwd_independent(&once, windows) {
                    Ok(None)
                } else {
                    Err(format!("{name} does not expand to one folder"))
                }
            }
        }
    };
    let Some(cwd) = cwd else {
        // Lost cwd: nothing can be anchored, so this only survives with no relative values.
        for name in RELATIVE_PATH_ENV.iter().chain(PATH_LIST_ENV) {
            if skipped.contains(name) {
                continue;
            }
            let Some(value) = lookup(name) else { continue };
            // A list is judged entry by entry: "C:\\vendor;plugins" starts with a
            // drive and still carries something the lost directory decided.
            let raw = value.trim();
            let entries: Vec<&str> = if PATH_LIST_ENV.contains(name) {
                raw.split(path_list_separator(windows)).collect()
            } else {
                vec![raw]
            };
            for entry in entries {
                let entry = entry.trim();
                if entry.is_empty() {
                    // An empty PYTHONPATH component is the directory itself.
                    if *name == "PYTHONPATH" && !raw.is_empty() {
                        return Err(format!(
                            "{name} names the directory it was written against, which is gone"
                        ));
                    }
                    continue;
                }
                // Judged like the moving path, or expanded/JSON/toggle values would refuse every spawn.
                let entry = expand_user(entry, home, username.as_deref(), windows);
                let Some(entry) = expanded(name, &entry)? else {
                    continue;
                };
                if is_cwd_independent(&entry, windows) {
                    continue;
                }
                if !names_a_path(name, &entry) {
                    continue;
                }
                return Err(format!(
                    "{name} is relative and the directory it was written against is gone"
                ));
            }
        }
        return Ok(Vec::new());
    };
    // The usual case: the child keeps the directory it inherited, so every
    // relative value still means what it did and nothing is rewritten.
    if cwd == work_dir {
        return Ok(Vec::new());
    }
    // A value the OS cannot resolve refuses the move rather than silently retargeting (as the CLI guard).
    let anchor = |name: &str, value: &str| -> Result<std::path::PathBuf, String> {
        if needs_os_resolution(value) {
            // "D:cache" is drive D's own current directory and "\\cache" the
            // root of the current drive, neither of which join() knows.
            absolute(value)
                .ok_or_else(|| format!("{name} names a path this machine cannot resolve"))
        } else {
            Ok(cwd.join(value))
        }
    };
    let mut pins = Vec::new();
    for name in RELATIVE_PATH_ENV {
        if skipped.contains(name) {
            continue;
        }
        let Some(value) = lookup(name) else { continue };
        let original = value.trim().to_string();
        // Tilde first, then variables: the CLI guard's order.
        let value = expand_user(&original, home, username.as_deref(), windows);
        let Some(value) = expanded(name, &value)? else {
            continue;
        };
        if value.is_empty() {
            continue;
        }
        if is_cwd_independent(&value, windows) {
            // Written back if expansion is what made it absolute, for readers that do not expand.
            if value != original {
                pins.push((*name, std::path::PathBuf::from(value)));
            }
            continue;
        }
        if !names_a_path(name, &value) {
            continue;
        }
        match anchor(name, &value) {
            Ok(pinned) => {
                if windows && pinned.as_os_str().len() >= WINDOWS_ENV_VALUE_LIMIT {
                    return Err(format!(
                        "{name} does not fit in an environment variable once it names its folder in full"
                    ));
                }
                pins.push((*name, pinned));
            }
            Err(error) => {
                if !BEST_EFFORT_ENV.contains(name) {
                    return Err(error);
                }
            }
        }
    }
    for name in PATH_LIST_ENV {
        let Some(raw) = lookup(name) else { continue };
        if raw.trim().is_empty() {
            continue;
        }
        let separator = path_list_separator(windows);
        let mut entries: Vec<String> = Vec::new();
        for entry in raw.split(separator) {
            let original = entry.trim().to_string();
            // Python never expands `~` in PYTHONPATH.
            let entry = match home {
                Some(home) if *name != "PYTHONPATH" => {
                    expand_windows_user(&original, home, username.as_deref())
                }
                _ => original.clone(),
            };
            let Some(entry) = expanded(name, &entry)? else {
                entries.push(original);
                continue;
            };
            // An empty PYTHONPATH component means the working directory itself.
            if *name == "PYTHONPATH" && entry.is_empty() {
                entries.push(cwd.to_string_lossy().into_owned());
                continue;
            }
            if entry.is_empty() || is_cwd_independent(&entry, windows) {
                entries.push(entry);
                continue;
            }
            if !names_a_path(name, &entry) {
                entries.push(entry);
                continue;
            }
            entries.push(anchor(name, &entry)?.to_string_lossy().into_owned());
        }
        let joined = entries.join(&separator.to_string());
        // Values over the Windows env limit are reported here, not left to fail in CreateProcess.
        if windows && joined.len() >= WINDOWS_ENV_VALUE_LIMIT {
            return Err(format!(
                "{name} does not fit in an environment variable once each entry names its folder in full"
            ));
        }
        if joined == raw {
            continue;
        }
        pins.push((*name, std::path::PathBuf::from(joined)));
    }
    Ok(pins)
}

/// Relative overrides, anchored to the directory the child is being moved out of.
fn relative_override_pins(
    work_dir: &std::path::Path,
    skipped: &[&str],
) -> Result<Vec<(&'static str, std::path::PathBuf)>, String> {
    relative_override_pins_from(
        std::env::current_dir().ok(),
        work_dir,
        |name| std::env::var(name).ok(),
        // GetFullPathNameW on Windows knows each drive's own cwd.
        |value| std::path::absolute(value).ok(),
        // preflight::managed's reader, so the child pin and the fingerprint use the same home.
        crate::preflight::managed::tilde_home().as_deref(),
        skipped,
        cfg!(windows),
    )
}

/// None when the move must not happen: an unnamed cwd means stay put rather than refuse the spawn.
fn pins_for_move(
    work_dir: &std::path::Path,
    skipped: &[&str],
) -> Result<Option<Vec<(&'static str, std::path::PathBuf)>>, String> {
    stay_put_on_lost_cwd(
        relative_override_pins(work_dir, skipped),
        std::env::current_dir().is_ok(),
    )
}

/// Split out so the decision is testable without moving this process.
fn stay_put_on_lost_cwd(
    pins: Result<Vec<(&'static str, std::path::PathBuf)>, String>,
    cwd_is_known: bool,
) -> Result<Option<Vec<(&'static str, std::path::PathBuf)>>, String> {
    match pins {
        Ok(pins) => Ok(Some(pins)),
        Err(error) if cwd_is_known => Err(error),
        Err(_) => Ok(None),
    }
}

/// The update child reads STUDIO_LOCAL_REPO and gets no PYTHONPATH on Windows (-I covers one interpreter).
fn update_child_skipped_env() -> Vec<&'static str> {
    MANAGED_CHILD_SCRUBBED_ENV
        .iter()
        .copied()
        .chain(cfg!(windows).then_some("PYTHONPATH"))
        .collect()
}

/// Best effort: a stale value must not block `studio update`. Twin of `_BEST_EFFORT_ENV` in the CLI guard.
const BEST_EFFORT_ENV: &[&str] = &["STUDIO_LOCAL_REPO"];

/// What an ordinary managed child neither receives nor needs.
fn child_skipped_env() -> Vec<&'static str> {
    MANAGED_CHILD_SCRUBBED_ENV
        .iter()
        .chain(UPDATE_ONLY_ENV)
        .copied()
        .collect()
}

/// Preflight asks first: an unbuildable context is not a broken CLI and must not trigger repair.
pub(crate) fn managed_cli_context_error() -> Option<ManagedContextError> {
    let work_dir = match managed_cli_working_dir() {
        Ok(work_dir) => work_dir,
        Err(error) => return Some(ManagedContextError::WorkingDirectory(error)),
    };
    pins_for_move(&work_dir, &child_skipped_env())
        .err()
        .map(ManagedContextError::PathSetting)
}

/// Why a managed spawn cannot be configured: unreachable profile vs unresolvable path setting.
#[derive(Debug, Clone)]
pub(crate) enum ManagedContextError {
    WorkingDirectory(String),
    PathSetting(String),
}

impl std::fmt::Display for ManagedContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WorkingDirectory(error) | Self::PathSetting(error) => f.write_str(error),
        }
    }
}

/// No when cwd already matches: reopening by name can fail where inheriting the handle works.
fn needs_explicit_cwd(work_dir: &std::path::Path) -> bool {
    std::env::current_dir()
        .map(|cwd| cwd != work_dir)
        .unwrap_or(true)
}

/// Pin the working directory and mark the child desktop-managed; env scrubbing stays with the caller.
/// For the update, which is the one child that reads STUDIO_LOCAL_REPO.
pub(crate) fn apply_managed_cli_context(cmd: &mut Command) -> Result<(), String> {
    apply_managed_cli_context_inner(cmd, &managed_cli_working_dir()?, &update_child_skipped_env())
}

pub(crate) fn apply_managed_cli_context_at(
    cmd: &mut Command,
    work_dir: &std::path::Path,
) -> Result<(), String> {
    apply_managed_cli_context_inner(cmd, work_dir, &child_skipped_env())
}

fn apply_managed_cli_context_inner(
    cmd: &mut Command,
    work_dir: &std::path::Path,
    skipped: &[&str],
) -> Result<(), String> {
    if let Some(pins) = pins_for_move(work_dir, skipped)? {
        for (name, pinned) in pins {
            cmd.env(name, pinned);
        }
        if needs_explicit_cwd(work_dir) {
            cmd.current_dir(work_dir);
        }
    }
    // Removed here too, so the skip holds whatever the caller does.
    for name in skipped {
        cmd.env_remove(name);
    }
    cmd.env(DESKTOP_MANAGED_ENV, "1");
    Ok(())
}

pub(crate) fn apply_managed_cli_context_tokio(
    cmd: &mut tokio::process::Command,
) -> Result<(), String> {
    let work_dir = managed_cli_working_dir()?;
    let skipped = child_skipped_env();
    if let Some(pins) = pins_for_move(&work_dir, &skipped)? {
        for (name, pinned) in pins {
            cmd.env(name, pinned);
        }
        if needs_explicit_cwd(&work_dir) {
            cmd.current_dir(&work_dir);
        }
    }
    for name in &skipped {
        cmd.env_remove(name);
    }
    cmd.env(DESKTOP_MANAGED_ENV, "1");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::path::PathBuf;
    use std::sync::mpsc;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_studio_dir(test_name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "unsloth-{test_name}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn finds_new_layout_before_legacy_layout_and_falls_back() {
        let temp = temp_studio_dir("layout-preference");

        #[cfg(unix)]
        let new_bin = temp.join("unsloth_studio/bin/unsloth");
        #[cfg(unix)]
        let old_bin = temp.join(".venv/bin/unsloth");
        #[cfg(windows)]
        let new_bin = temp.join("unsloth_studio/Scripts/unsloth.exe");
        #[cfg(windows)]
        let old_bin = temp.join(".venv/Scripts/unsloth.exe");

        fs::create_dir_all(new_bin.parent().unwrap()).unwrap();
        fs::create_dir_all(old_bin.parent().unwrap()).unwrap();
        fs::write(&new_bin, "").unwrap();
        fs::write(&old_bin, "").unwrap();

        assert_eq!(
            find_unsloth_binary_in_studio_dir(&temp),
            Some(new_bin.clone())
        );
        fs::remove_file(&new_bin).unwrap();
        assert_eq!(find_unsloth_binary_in_studio_dir(&temp), Some(old_bin));
        fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn backend_args_always_enable_api_only() {
        assert_eq!(
            backend_args(8888),
            vec!["studio", "--api-only", "-H", "127.0.0.1", "-p", "8888"]
        );
    }

    // Application Control denies unsloth.exe; every invocation must go through the interpreter.
    #[cfg(windows)]
    fn managed_venv(test_name: &str) -> (PathBuf, PathBuf, PathBuf) {
        let dir = temp_studio_dir(test_name);
        let python = dir.join("python.exe");
        let bin = dir.join("unsloth.exe");
        fs::write(&python, "").unwrap();
        fs::write(&bin, "").unwrap();
        (dir, python, bin)
    }

    // Quarantine removes the unsigned stub but the environment still runs.
    #[cfg(windows)]
    #[test]
    fn a_quarantined_stub_is_still_a_managed_install() {
        let studio = temp_studio_dir("quarantined-stub");
        let scripts = studio.join("unsloth_studio").join("Scripts");
        fs::create_dir_all(&scripts).unwrap();

        assert_eq!(find_unsloth_binary_in_studio_dir(&studio), None);

        fs::write(scripts.join("python.exe"), "").unwrap();
        assert_eq!(
            find_unsloth_binary_in_studio_dir(&studio),
            Some(scripts.join("unsloth.exe")),
            "a stub-less environment with an interpreter is still an install"
        );

        let invocation =
            resolve_managed_cli_invocation(&scripts.join("unsloth.exe"), &["studio"]).unwrap();
        assert_eq!(invocation.program, scripts.join("python.exe"));

        fs::remove_dir_all(studio).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn a_legacy_launcher_outranks_a_stubless_new_environment() {
        let studio = temp_studio_dir("interrupted-migration");
        let new_scripts = studio.join("unsloth_studio").join("Scripts");
        let old_scripts = studio.join(".venv").join("Scripts");
        fs::create_dir_all(&new_scripts).unwrap();
        fs::create_dir_all(&old_scripts).unwrap();
        fs::write(new_scripts.join("python.exe"), "").unwrap();
        fs::write(old_scripts.join("python.exe"), "").unwrap();
        fs::write(old_scripts.join("unsloth.exe"), "").unwrap();

        assert_eq!(
            find_unsloth_binary_in_studio_dir(&studio),
            Some(old_scripts.join("unsloth.exe")),
            "a working legacy install must win over a partial new one"
        );

        fs::write(new_scripts.join("unsloth.exe"), "").unwrap();
        assert_eq!(
            find_unsloth_binary_in_studio_dir(&studio),
            Some(new_scripts.join("unsloth.exe"))
        );

        fs::remove_dir_all(studio).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn a_complete_legacy_environment_beats_a_launcher_with_no_interpreter() {
        let studio = temp_studio_dir("split-migration");
        let new_scripts = studio.join("unsloth_studio").join("Scripts");
        let old_scripts = studio.join(".venv").join("Scripts");
        fs::create_dir_all(&new_scripts).unwrap();
        fs::create_dir_all(&old_scripts).unwrap();
        fs::write(new_scripts.join("unsloth.exe"), "").unwrap();
        fs::write(old_scripts.join("python.exe"), "").unwrap();
        fs::write(old_scripts.join("unsloth.exe"), "").unwrap();

        assert_eq!(
            find_unsloth_binary_in_studio_dir(&studio),
            Some(old_scripts.join("unsloth.exe")),
            "a complete environment must win over a launcher that cannot start"
        );

        fs::remove_file(old_scripts.join("python.exe")).unwrap();
        fs::remove_file(old_scripts.join("unsloth.exe")).unwrap();
        assert_eq!(
            find_unsloth_binary_in_studio_dir(&studio),
            Some(new_scripts.join("unsloth.exe"))
        );

        fs::remove_dir_all(studio).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn an_interpreter_that_still_has_the_package_outranks_one_that_does_not() {
        let studio = temp_studio_dir("stubless-both-halves");
        let new_base = studio.join("unsloth_studio");
        let old_base = studio.join(".venv");
        for base in [&new_base, &old_base] {
            fs::create_dir_all(base.join("Scripts")).unwrap();
            fs::write(base.join("Scripts").join("python.exe"), "").unwrap();
        }

        assert_eq!(
            find_unsloth_binary_in_studio_dir(&studio),
            Some(new_base.join("Scripts").join("unsloth.exe")),
            "with nothing to choose between them the new layout still wins"
        );

        fs::create_dir_all(old_base.join("Lib").join("site-packages").join("unsloth_cli")).unwrap();
        assert_eq!(
            find_unsloth_binary_in_studio_dir(&studio),
            Some(old_base.join("Scripts").join("unsloth.exe")),
            "the only base with a package to import must win"
        );

        fs::create_dir_all(new_base.join("Lib").join("site-packages").join("unsloth_cli")).unwrap();
        assert_eq!(
            find_unsloth_binary_in_studio_dir(&studio),
            Some(new_base.join("Scripts").join("unsloth.exe")),
            "package on both sides is not a reason to prefer the legacy layout"
        );

        fs::write(old_base.join("Scripts").join("unsloth.exe"), "").unwrap();
        assert_eq!(
            find_unsloth_binary_in_studio_dir(&studio),
            Some(old_base.join("Scripts").join("unsloth.exe"))
        );

        fs::remove_dir_all(studio).unwrap();
    }

    // Editable installs leave a .pth and dist-info but no unsloth_cli/ directory.
    #[cfg(windows)]
    #[test]
    fn an_editable_install_counts_as_carrying_the_package() {
        let studio = temp_studio_dir("stubless-editable");
        let new_base = studio.join("unsloth_studio");
        let old_base = studio.join(".venv");
        for base in [&new_base, &old_base] {
            fs::create_dir_all(base.join("Scripts")).unwrap();
            fs::write(base.join("Scripts").join("python.exe"), "").unwrap();
            fs::create_dir_all(base.join("Lib").join("site-packages")).unwrap();
        }

        let legacy_site_packages = old_base.join("Lib").join("site-packages");
        fs::create_dir_all(legacy_site_packages.join("unsloth-2026.8.1.dist-info")).unwrap();
        fs::write(legacy_site_packages.join("__editable__.unsloth.pth"), "").unwrap();

        assert_eq!(
            find_unsloth_binary_in_studio_dir(&studio),
            Some(old_base.join("Scripts").join("unsloth.exe")),
            "an editable install has a package to import and must outrank an empty venv"
        );

        fs::create_dir_all(
            new_base
                .join("Lib")
                .join("site-packages")
                .join("unsloth_zoo-2026.8.1.dist-info"),
        )
        .unwrap();
        assert_eq!(
            find_unsloth_binary_in_studio_dir(&studio),
            Some(old_base.join("Scripts").join("unsloth.exe")),
            "a dist-info for another distribution must not count"
        );

        fs::remove_dir_all(studio).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn managed_invocation_runs_python_with_the_trampoline_and_caller_args() {
        use std::ffi::OsString;

        let (dir, python, bin) = managed_venv("managed-cli-invocation");
        let invocation =
            resolve_managed_cli_invocation(&bin, &["studio", "--api-only", "-p", "8888"]).unwrap();

        assert_eq!(invocation.program, python);
        assert_ne!(invocation.program, bin);
        assert_eq!(
            invocation.args,
            vec![
                // No -I; see WINDOWS_CLI_ENTRYPOINT.
                OsString::from("-X"),
                OsString::from("utf8"),
                OsString::from("-c"),
                OsString::from(WINDOWS_CLI_ENTRYPOINT),
                OsString::from("studio"),
                OsString::from("--api-only"),
                OsString::from("-p"),
                OsString::from("8888"),
            ]
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn managed_invocation_does_not_isolate_the_interpreter() {
        let (dir, _python, bin) = managed_venv("managed-cli-no-isolation");
        let invocation = resolve_managed_cli_invocation(&bin, &["-h"]).unwrap();

        assert!(
            !invocation.args.iter().any(|arg| arg == "-I"),
            "{:?}",
            invocation.args
        );
        assert!(
            WINDOWS_CLI_ENTRYPOINT.contains("sys.path[:1]"),
            "{WINDOWS_CLI_ENTRYPOINT}"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    // The updater stays isolated; asserted here and in update.rs.
    #[cfg(windows)]
    #[test]
    fn only_the_isolated_flavour_carries_the_isolation_flag() {
        let (dir, _python, bin) = managed_venv("managed-cli-isolated");

        let inherit =
            resolve_managed_cli_invocation_with(&bin, &["studio"], Isolation::Inherit).unwrap();
        let isolated =
            resolve_managed_cli_invocation_with(&bin, &["studio"], Isolation::Isolated).unwrap();

        assert!(!inherit.args.iter().any(|arg| arg == "-I"), "{:?}", inherit.args);
        assert!(isolated.args.iter().any(|arg| arg == "-I"), "{:?}", isolated.args);
        assert_eq!(isolated.args[0], std::ffi::OsString::from("-X"));
        assert_eq!(isolated.args[1], std::ffi::OsString::from("utf8"));
        assert_eq!(isolated.args[2], std::ffi::OsString::from("-I"));
        assert_eq!(inherit.program, isolated.program);
        assert_eq!(inherit.args.last(), isolated.args.last());
        assert_eq!(
            resolve_managed_cli_invocation(&bin, &["studio"]).unwrap().args,
            inherit.args
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn managed_trampoline_assigns_argv0_before_importing_the_cli() {
        // unsloth_cli checks sys.argv[0] at import time, so it must be set first.
        let strip = WINDOWS_CLI_ENTRYPOINT.find("sys.path[:1]");
        let assignment = WINDOWS_CLI_ENTRYPOINT.find("sys.argv[0] = 'unsloth'");
        let import = WINDOWS_CLI_ENTRYPOINT.find("from unsloth_cli import app");
        assert!(strip.is_some() && assignment.is_some() && import.is_some());
        assert!(assignment < import, "{WINDOWS_CLI_ENTRYPOINT}");
        assert!(strip < import, "{WINDOWS_CLI_ENTRYPOINT}");
    }

    #[cfg(windows)]
    #[test]
    fn managed_commands_leave_the_python_environment_alone() {
        use std::ffi::OsStr;

        let (dir, _python, bin) = managed_venv("managed-cli-env");
        let cmd = build_managed_cli_command(&bin, &["-h"]).unwrap();
        for name in ["PYTHONHOME", "PYTHONPATH"] {
            assert!(
                !cmd.get_envs().any(|(key, _)| key == OsStr::new(name)),
                "{name} must be inherited, not overridden"
            );
        }
        let tokio_cmd = build_managed_cli_command_tokio(&bin, &["-h"]).unwrap();
        let std_cmd = tokio_cmd.as_std();
        for name in ["PYTHONHOME", "PYTHONPATH"] {
            assert!(
                !std_cmd.get_envs().any(|(key, _)| key == OsStr::new(name)),
                "{name} must be inherited, not overridden"
            );
        }
        assert_eq!(std_cmd.get_program(), cmd.get_program());
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn managed_invocation_fails_closed_without_the_interpreter() {
        let bin = temp_studio_dir("managed-cli-no-python").join("unsloth.exe");
        let error = resolve_managed_cli_invocation(&bin, &["-h"]).unwrap_err();
        assert!(error.contains("python.exe"), "{error}");
    }

    #[cfg(not(windows))]
    #[test]
    fn posix_managed_invocation_still_execs_the_console_script() {
        use std::ffi::OsString;

        let bin = std::path::Path::new("/opt/unsloth/bin/unsloth");
        let invocation = resolve_managed_cli_invocation(bin, &["studio", "--api-only"]).unwrap();

        assert_eq!(invocation.program, bin);
        assert_eq!(
            invocation.args,
            vec![OsString::from("studio"), OsString::from("--api-only")]
        );

        let cmd = build_managed_cli_command(bin, &["studio", "--api-only"]).unwrap();
        assert_eq!(cmd.get_program(), bin.as_os_str());
        assert_eq!(
            cmd.get_args().map(OsString::from).collect::<Vec<_>>(),
            vec![OsString::from("studio"), OsString::from("--api-only")]
        );
        assert!(cmd.get_envs().next().is_none());
    }

    #[cfg(not(windows))]
    #[test]
    fn posix_managed_invocation_needs_no_interpreter_beside_the_script() {
        let bin = std::path::Path::new("/definitely/not/here/bin/unsloth");
        assert!(resolve_managed_cli_invocation(bin, &["-h"]).is_ok());
    }

    fn listening_non_studio_port() -> (u16, mpsc::Sender<()>, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let (tx, rx) = mpsc::channel::<()>();
        let handle = std::thread::spawn(move || loop {
            if rx.try_recv().is_ok() {
                break;
            }
            match listener.accept() {
                Ok((mut stream, _)) => {
                    let mut buf = [0_u8; 512];
                    let _ = stream.read(&mut buf);
                    let _ = stream.write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(_) => break,
            }
        });
        (port, tx, handle)
    }

    // A spawned handle whose child exited without stdout EOF must not block the next launch.

    #[cfg(unix)]
    const ALREADY_EXITED: [&str; 3] = ["/bin/sh", "-c", "exit 3"];
    #[cfg(unix)]
    const STILL_RUNNING: [&str; 3] = ["/bin/sh", "-c", "exec sleep 30"];
    #[cfg(windows)]
    const ALREADY_EXITED: [&str; 3] = ["cmd", "/C", "exit 3"];
    #[cfg(windows)]
    const STILL_RUNNING: [&str; 3] = ["cmd", "/C", "ping -n 31 127.0.0.1"];

    // Shaped like start_backend's spawn: a process group on Unix, a bare Child on Windows.
    fn spawn_test_child(args: &[&str]) -> Box<dyn ChildWrapper + Send> {
        let mut cmd = Command::new(args[0]);
        cmd.args(&args[1..])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        {
            let mut wrap = CommandWrap::from(cmd);
            wrap.wrap(ProcessGroup::leader());
            wrap.spawn().expect("spawn test child")
        }
        #[cfg(windows)]
        {
            Box::new(cmd.spawn().expect("spawn test child"))
        }
    }

    fn spawn_and_reap() -> Box<dyn ChildWrapper + Send> {
        let mut child = spawn_test_child(&ALREADY_EXITED);
        for _ in 0..100 {
            match child.try_wait() {
                Ok(Some(_)) => return child,
                Ok(None) => std::thread::sleep(Duration::from_millis(50)),
                Err(error) => panic!("could not poll the test child: {error}"),
            }
        }
        panic!("the test child never exited");
    }

    fn state_with_spawned(child: Box<dyn ChildWrapper + Send>, generation: u64) -> BackendState {
        let state = new_backend_state();
        {
            let mut proc = state.lock().unwrap();
            proc.generation = generation;
            proc.port = Some(8888);
            proc.owned = Some(OwnedBackendHandle::spawned(child, None, 4242, generation));
        }
        state
    }

    #[test]
    fn a_dead_spawned_backend_is_cleared_and_stops_blocking_a_launch() {
        let state = state_with_spawned(spawn_and_reap(), 5);
        assert!(
            state.lock().unwrap().has_owned_backend(),
            "precondition: this is what makes start_backend answer Backend is already running."
        );
        assert!(clear_spawned_backend_if_exited(&state, 5, "test"));
        let proc = state.lock().unwrap();
        assert!(!proc.has_owned_backend(), "the dead handle still blocks a launch");
        assert!(proc.port.is_none());
        assert!(proc.diagnostics_session.is_none());
    }

    #[test]
    fn a_live_spawned_backend_is_left_alone() {
        // A handle with no validated port is usually a cold start; it must not be cleared.
        let state = state_with_spawned(spawn_test_child(&STILL_RUNNING), 5);
        assert!(!clear_spawned_backend_if_exited(&state, 5, "test"));
        {
            let mut proc = state.lock().unwrap();
            assert!(proc.has_owned_backend(), "cleared a backend that was still running");
            assert_eq!(proc.port, Some(8888));
            if let Some(child) = proc
                .owned
                .as_mut()
                .and_then(OwnedBackendHandle::spawned_child_mut)
            {
                let _ = child.start_kill();
            }
        }
    }

    #[test]
    fn a_dead_spawned_backend_from_an_older_generation_is_left_alone() {
        let state = state_with_spawned(spawn_and_reap(), 5);
        assert!(!clear_spawned_backend_if_exited(&state, 4, "test"));
        assert!(state.lock().unwrap().has_owned_backend());
    }

    #[test]
    fn an_adopted_backend_is_not_this_functions_business() {
        let state = new_backend_state();
        let owner = crate::desktop_backend_owner::test_owner_state(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "desktop-owner-token",
            8888,
        );
        {
            let mut proc = state.lock().unwrap();
            proc.generation = 5;
            proc.owned = Some(OwnedBackendHandle::adopted(owner, 8888, 4242, 5));
        }
        assert!(!clear_spawned_backend_if_exited(&state, 5, "test"));
        assert!(state.lock().unwrap().has_owned_backend());
    }

    #[test]
    fn nothing_owned_is_not_an_error() {
        let state = new_backend_state();
        state.lock().unwrap().generation = 5;
        assert!(!clear_spawned_backend_if_exited(&state, 5, "test"));
    }

    #[test]
    fn stop_backend_rolls_back_shutdown_flag_when_adopted_stop_fails() {
        let (port, stop_listener, listener_thread) = listening_non_studio_port();
        let state = new_backend_state();
        let shutdown = new_shutdown_flag();
        let owner = crate::desktop_backend_owner::test_owner_state(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "desktop-owner-token",
            port,
        );

        {
            let mut proc = state.lock().unwrap();
            proc.generation = 7;
            proc.port = Some(port);
            proc.owned = Some(OwnedBackendHandle::adopted(owner, port, 1234, 3));
        }

        let error = stop_backend(&state, &shutdown, None)
            .expect_err("adopted backend should refuse unsafe stop fallback");

        assert!(error.contains("Refusing to stop adopted backend"));
        assert!(!shutdown.load(Ordering::SeqCst));
        assert!(state.lock().unwrap().has_adopted_backend());

        let _ = stop_listener.send(());
        let _ = std::net::TcpStream::connect(("127.0.0.1", port));
        listener_thread.join().unwrap();
    }
}

/// Find the unsloth binary; debug builds prefer the repo's local .venv.
pub(crate) fn resolve_backend_binary() -> Result<std::path::PathBuf, String> {
    #[cfg(debug_assertions)]
    {
        // CARGO_MANIFEST_DIR is studio/src-tauri; the repo root is two levels up.
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let repo_root = std::path::Path::new(manifest_dir)
            .parent()
            .and_then(|p| p.parent());

        if let Some(root) = repo_root {
            #[cfg(unix)]
            let dev_bin = root.join(".venv/bin/unsloth");
            #[cfg(windows)]
            let dev_bin = root.join(".venv/Scripts/unsloth.exe");

            if dev_bin.exists() {
                info!("Dev mode: using local repo backend at {:?}", dev_bin);
                return Ok(dev_bin.to_path_buf());
            }
        }
        info!("Dev mode: no local .venv found, falling back to installed backend");
    }

    find_unsloth_binary()
        .ok_or_else(|| "Unsloth binary not found. Please install Unsloth first.".to_string())
}

fn backend_args(port: u16) -> Vec<String> {
    [
        "studio",
        "--api-only",
        "-H",
        "127.0.0.1",
        "-p",
        &port.to_string(),
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

/// Spawn the backend process and wire up stdout/stderr reader threads.
pub fn start_backend(
    app: &AppHandle,
    state: &BackendState,
    port: u16,
    shutdown: &ShutdownFlag,
    diagnostics_state: &DiagnosticsState,
) -> Result<u64, String> {
    let _runtime_launch_guard = acquire_studio_runtime_launch_guard()?;

    // Checked on the spawn path: a webview remount can start a backend while the job is disarmed.
    #[cfg(windows)]
    if !crate::windows_job::kill_on_close_armed().unwrap_or(false) {
        crate::windows_job::resume_after_update_installer().map_err(|error| {
            format!("Refusing to start the backend with crash cleanup disarmed: {error}")
        })?;
        // Reset like the UI's resume, or the next pre-exit hook suspends kill-on-close without
        // stopping this backend.
        crate::reset_termination_cleanup();
    }

    let bin = match resolve_backend_binary() {
        Ok(bin) => bin,
        Err(msg) => {
            diagnostics::record_backend_start_failure(
                diagnostics_state,
                Some(port),
                None,
                "resolve_backend_binary",
                &msg,
            );
            return Err(msg);
        }
    };

    let work_dir = match managed_cli_working_dir() {
        Ok(work_dir) => work_dir,
        Err(error) => {
            let msg = format!(
                "Failed to pick a working directory for the backend: {}",
                error
            );
            diagnostics::record_backend_start_failure(
                diagnostics_state,
                Some(port),
                None,
                "resolve_working_directory",
                &msg,
            );
            return Err(msg);
        }
    };

    let args = backend_args(port);
    // Before claiming ownership, so a missing interpreter leaves no pending owner.
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let invocation = match resolve_managed_cli_invocation(&bin, &arg_refs) {
        Ok(invocation) => invocation,
        Err(msg) => {
            diagnostics::record_backend_start_failure(
                diagnostics_state,
                Some(port),
                None,
                "build_backend_command",
                &msg,
            );
            return Err(msg);
        }
    };
    // Log the program actually spawned (python.exe on Windows), not the stub, for support reports.
    let start_line = format!(
        "Starting backend: {:?} {}",
        invocation.program,
        invocation
            .args
            .iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join(" ")
    );
    let mut cmd = invocation.to_command();
    let pending_owner = match crate::desktop_backend_owner::new_pending_owner() {
        Ok(pending_owner) => pending_owner,
        Err(error) => {
            let msg = format!("Failed to claim ownership of the backend: {}", error);
            diagnostics::record_backend_start_failure(
                diagnostics_state,
                Some(port),
                None,
                "claim_backend_ownership",
                &msg,
            );
            return Err(msg);
        }
    };
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    if let Err(error) = apply_managed_cli_context_at(&mut cmd, &work_dir) {
        // The override's drive can vanish after preflight; a panic here takes the desktop down.
        let msg = format!("Failed to prepare the Unsloth backend command: {}", error);
        diagnostics::record_backend_start_failure(
            diagnostics_state,
            Some(port),
            None,
            "managed_cli_context",
            &msg,
        );
        return Err(msg);
    }

    cmd.env_remove(STUDIO_RUNTIME_GATE_HANDOFF_ENV);
    cmd.env(STUDIO_RUNTIME_GATE_ACQUIRE_ENV, "1");

    if let Some(native_state) = app.try_state::<crate::native_intents::NativeIntakeState>() {
        cmd.env(
            crate::native_backend_lease::LEASE_SECRET_ENV,
            native_state.lease_secret_env(),
        );
    }

    crate::desktop_backend_owner::apply_owner_env(&mut cmd, &pending_owner);

    #[cfg(target_os = "linux")]
    scrub_appimage_python_env(&mut cmd);

    // Scrub via the shared list so the backend cannot diverge from Tauri's legacy root.
    // UNSLOTH_LLAMA_CPP_PATH is a user override; keep it.
    for name in MANAGED_CHILD_SCRUBBED_ENV {
        cmd.env_remove(name);
    }

    // read_output_stream decodes UTF-8; without these Python uses the locale code page.
    #[cfg(windows)]
    {
        cmd.env("PYTHONUTF8", "1");
        cmd.env("PYTHONIOENCODING", "utf-8");
    }

    // Reset, spawn and store under one lock so a concurrent start/stop cannot interleave.
    let (generation, backend_log, stdout, stderr) = {
        let mut proc = state.lock().map_err(|e| e.to_string())?;
        if proc.has_owned_backend() {
            return Err("Backend is already running.".to_string());
        }

        shutdown.store(false, Ordering::SeqCst);
        proc.generation = proc.generation.wrapping_add(1);
        proc.port = None;
        proc.logs.clear();
        proc.intentional_stop = false;
        proc.diagnostics_session = None;
        proc.adopted_watchdog_generation = None;
        proc.start_timed_out = false;
        proc.owned = None;
        let generation = proc.generation;

        let backend_log = diagnostics::begin_backend_session(diagnostics_state, port, generation);

        // The app is in a KILL_ON_JOB_CLOSE job (main.rs), so children are cleaned up without a
        // per-child job.
        #[cfg(windows)]
        let mut child: Box<dyn ChildWrapper + Send> = {
            use std::os::windows::process::CommandExt;

            const CREATE_NEW_PROCESS_GROUP: u32 = 0x00000200;
            cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
            let child = cmd.spawn().map_err(|e| {
                let msg = format!("Failed to spawn backend: {}", e);
                diagnostics::record_backend_start_failure(
                    diagnostics_state,
                    Some(port),
                    Some(generation),
                    "spawn_backend",
                    &msg,
                );
                msg
            })?;
            Box::new(child)
        };

        #[cfg(unix)]
        let mut child: Box<dyn ChildWrapper + Send> = {
            // Keep the backend tree in a process group on Unix for cleanup.
            let mut wrap = CommandWrap::from(cmd);
            wrap.wrap(ProcessGroup::leader());
            wrap.spawn().map_err(|e| {
                let msg = format!("Failed to spawn backend: {}", e);
                diagnostics::record_backend_start_failure(
                    diagnostics_state,
                    Some(port),
                    Some(generation),
                    "spawn_backend",
                    &msg,
                );
                msg
            })?
        };

        let backend_pid = child.id();
        let stdout = child.stdout().take();
        let stderr = child.stderr().take();
        let owner = match crate::desktop_backend_owner::activate_owner(
            pending_owner,
            port,
            generation,
            backend_pid,
        ) {
            Ok(owner) => owner,
            Err(error) => {
                // No handle owns this live child yet, so stop it before returning (mutex still held).
                if let Err(stop_error) = stop_spawned_backend(child, None, None, backend_pid) {
                    warn!(
                        "Could not stop the unclaimed backend (pid {}): {}",
                        backend_pid, stop_error
                    );
                }
                let msg = format!(
                    "Failed to claim ownership of the backend, so it was stopped: {}",
                    error
                );
                diagnostics::record_backend_start_failure(
                    diagnostics_state,
                    Some(port),
                    Some(generation),
                    "activate_backend_ownership",
                    &msg,
                );
                return Err(msg);
            }
        };

        proc.owned = Some(OwnedBackendHandle::spawned(
            child,
            Some(owner),
            backend_pid,
            generation,
        ));
        proc.diagnostics_session = Some(backend_log.clone());
        (generation, backend_log, stdout, stderr)
    };

    info!("{}", start_line);
    diagnostics::append_phase_line(&backend_log.handle, "meta", &start_line);
    // Shared with the watchdog: port validation must not outlive server-start-timeout.
    let start_deadline = std::time::Instant::now() + BACKEND_START_DEADLINE;
    start_watchdog(app, state, shutdown, generation, &backend_log);

    if let Some(stdout) = stdout {
        let app_handle = app.clone();
        let state_clone = Arc::clone(state);
        let diagnostics_clone = diagnostics_state.clone();
        let backend_log_clone = backend_log.clone();
        std::thread::spawn(move || {
            read_output_stream(
                stdout,
                &app_handle,
                &state_clone,
                &diagnostics_clone,
                &backend_log_clone,
                false,
                generation,
                start_deadline,
            );
        });
    }

    if let Some(stderr) = stderr {
        let app_handle = app.clone();
        let state_clone = Arc::clone(state);
        let diagnostics_clone = diagnostics_state.clone();
        let backend_log_clone = backend_log.clone();
        std::thread::spawn(move || {
            read_output_stream(
                stderr,
                &app_handle,
                &state_clone,
                &diagnostics_clone,
                &backend_log_clone,
                true,
                generation,
                start_deadline,
            );
        });
    }

    Ok(generation)
}

async fn generic_backend_health_ok(port: u16) -> bool {
    let started = std::time::Instant::now();
    let client = match crate::loopback_http::client(Duration::from_secs(2)) {
        Ok(client) => client,
        Err(error) => {
            warn!("Could not build backend validation client: {}", error);
            return false;
        }
    };
    let mut last_status = None;
    let mut json = None;
    for path in ["/api/liveness", "/api/health"] {
        let response = match client
            .get(format!("http://127.0.0.1:{port}{path}"))
            .send()
            .await
        {
            Ok(response) => response,
            Err(error) => {
                warn!(
                    "Backend port candidate {} failed health request: {}",
                    port, error
                );
                return false;
            }
        };
        if response.status() == reqwest::StatusCode::NOT_FOUND && path == "/api/liveness" {
            last_status = Some(response.status());
            continue;
        }
        if !response.status().is_success() {
            warn!(
                "Backend port candidate {} returned HTTP {} from health",
                port,
                response.status()
            );
            return false;
        }
        json = match response.json::<serde_json::Value>().await {
            Ok(json) => Some(json),
            Err(error) => {
                warn!(
                    "Backend port candidate {} returned invalid health JSON: {}",
                    port, error
                );
                return false;
            }
        };
        break;
    }
    let Some(json) = json else {
        warn!(
            "Backend port candidate {} returned HTTP {} from health",
            port,
            last_status
                .map(|status| status.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        );
        return false;
    };
    let live = json
        .get("status")
        .and_then(|v| v.as_str())
        .map(|s| s == "alive" || s == "healthy")
        .unwrap_or(false);
    let service = json
        .get("service")
        .and_then(|v| v.as_str())
        .map(|s| s == "Unsloth UI Backend")
        .unwrap_or(false);
    info!(
        "Backend port candidate {} liveness live={} service={} in {}ms",
        port,
        live,
        service,
        started.elapsed().as_millis()
    );
    live && service
}

/// Doubling backoff: each probe costs two requests against a busy backend.
const PORT_VALIDATION_RETRY_MIN: Duration = Duration::from_millis(250);
const PORT_VALIDATION_RETRY_MAX: Duration = Duration::from_secs(5);

async fn validate_candidate_port(
    app: AppHandle,
    state: BackendState,
    diagnostics_state: DiagnosticsState,
    session_id: String,
    generation: u64,
    port: u16,
    deadline: std::time::Instant,
) {
    let started = std::time::Instant::now();
    let owner = {
        let proc = match state.lock() {
            Ok(proc) => proc,
            Err(error) => {
                warn!("Backend state unavailable for port validation: {}", error);
                return;
            }
        };
        if proc.generation != generation || proc.port.is_some() {
            return;
        }
        match proc.owned.as_ref() {
            Some(OwnedBackendHandle::Spawned { owner, .. }) => owner.clone(),
            _ => return,
        }
    };

    // The port is announced once, possibly before the backend can answer, so retry until the deadline.
    let mut delay = PORT_VALIDATION_RETRY_MIN;
    let mut attempts = 0u32;
    let mut verified_late = false;
    let valid = loop {
        // Checked before the probe too: the announcement itself can arrive late.
        if std::time::Instant::now() >= deadline {
            break false;
        }
        attempts += 1;
        let ok = if let Some(owner) = owner.clone() {
            matches!(
                crate::desktop_backend_owner::probe_owned_backend_state(owner, Some(port), false)
                    .await,
                crate::desktop_backend_owner::OwnedBackendProbe::Verified(
                    crate::desktop_backend_owner::VerifiedOwnedBackend { port: verified_port, .. }
                ) if verified_port == port
            )
        } else {
            generic_backend_health_ok(port).await
        };
        if ok {
            // A late success must not emit server-port after server-start-timeout.
            if std::time::Instant::now() < deadline {
                break true;
            }
            verified_late = true;
            break false;
        }
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            break false;
        }
        tokio::time::sleep(delay.min(remaining)).await;
        delay = (delay * 2).min(PORT_VALIDATION_RETRY_MAX);
        // Guard scoped to this statement so the future stays Send across the await.
        let still_current = match state.lock() {
            Ok(proc) => proc.generation == generation && proc.port.is_none(),
            Err(_) => false,
        };
        if !still_current {
            return;
        }
    };

    if !valid {
        if verified_late {
            warn!(
                "Backend port {} verified after the start deadline; not emitting",
                port
            );
        } else if attempts == 0 {
            warn!(
                "TAURI_PORT candidate {} arrived after the start deadline",
                port
            );
        } else {
            warn!(
                "Ignoring unverified TAURI_PORT candidate {} after {} attempts",
                port, attempts
            );
        }
        return;
    }

    let should_emit = {
        let mut proc = match state.lock() {
            Ok(proc) => proc,
            Err(error) => {
                warn!("Backend state unavailable after port validation: {}", error);
                return;
            }
        };
        // start_timed_out is claimed under this lock, so exactly one outcome reaches the window.
        if proc.generation != generation || proc.port.is_some() || proc.start_timed_out {
            false
        } else if matches!(proc.owned, Some(OwnedBackendHandle::Spawned { .. })) {
            proc.port = Some(port);
            if let Some(owned) = proc.owned.as_mut() {
                owned.set_reported_port(port);
            }
            true
        } else {
            false
        }
    };

    info!(
        "Validated backend port candidate {} valid={} emit={} in {}ms",
        port,
        valid,
        should_emit,
        started.elapsed().as_millis()
    );

    if should_emit {
        diagnostics::record_backend_port(&diagnostics_state, &session_id, port);
        info!("Validated backend port: {}", port);
        let _ = app.emit("server-port", port);
    }
}

/// Deliberately loose (~12s typical): ends an unbounded wait without failing slow machines.
const BACKEND_START_DEADLINE: Duration = Duration::from_secs(300);

/// Report a backend that never becomes reachable, with its last log lines, without killing it;
/// the health watchdog owns the kill decision.
fn start_watchdog(
    app: &AppHandle,
    state: &BackendState,
    shutdown: &ShutdownFlag,
    generation: u64,
    backend_log: &BackendLog,
) {
    let app = app.clone();
    let state = Arc::clone(state);
    let shutdown = Arc::clone(shutdown);
    let backend_log = backend_log.clone();
    std::thread::spawn(move || {
        let started = std::time::Instant::now();
        while started.elapsed() < BACKEND_START_DEADLINE {
            std::thread::sleep(Duration::from_secs(1));
            if shutdown.load(Ordering::SeqCst) {
                return;
            }
            match state.lock() {
                Ok(proc) => {
                    if proc.generation != generation || !proc.has_owned_backend() {
                        return;
                    }
                    // The port is recorded only after validation, so this means reachable.
                    if proc.port.is_some() {
                        return;
                    }
                }
                Err(_) => {
                    warn!("Backend start watchdog giving up: state mutex poisoned");
                    return;
                }
            }
        }

        let (still_ours, tail) = match state.lock() {
            Ok(mut proc) => {
                // Same three conditions as the loop, or a late crash could be overwritten.
                if proc.generation != generation || proc.port.is_some() || !proc.has_owned_backend()
                {
                    (false, String::new())
                } else {
                    // Claim under the lock so a concurrent port validation cannot also emit.
                    proc.start_timed_out = true;
                    let skip = proc.logs.len().saturating_sub(20);
                    let tail: Vec<String> = proc.logs.iter().skip(skip).cloned().collect();
                    (true, tail.join("\n"))
                }
            }
            Err(_) => (false, String::new()),
        };
        if !still_ours || shutdown.load(Ordering::SeqCst) {
            return;
        }

        let secs = BACKEND_START_DEADLINE.as_secs();
        let msg = if tail.trim().is_empty() {
            format!(
                "The Unsloth backend did not start within {secs} seconds and produced no \
                 output at all. It is still running but is not responding."
            )
        } else {
            format!(
                "The Unsloth backend did not start within {secs} seconds. Its last output \
                 was:\n{tail}"
            )
        };
        error!("Backend start deadline exceeded after {}s", secs);
        diagnostics::append_phase_line(&backend_log.handle, "error", &msg);
        let _ = app.emit("server-start-timeout", msg);
    });
}

/// Read a child stream; stdout parses TAURI_PORT candidates and emits server-crashed on unexpected close.
fn read_output_stream<R: std::io::Read>(
    stream: R,
    app: &AppHandle,
    state: &BackendState,
    diagnostics_state: &DiagnosticsState,
    backend_log: &BackendLog,
    is_stderr: bool,
    generation: u64,
    start_deadline: std::time::Instant,
) {
    let mut reader = std::io::BufReader::new(stream);
    let port_re = Regex::new(r"TAURI_PORT=(\d+)").unwrap();
    let mut buf = Vec::new();
    let mut saw_eof = false;

    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) => {
                saw_eof = true;
                break;
            }
            Ok(_) => {
                let raw = String::from_utf8_lossy(trim_line_endings(&buf));
                let text = collapse_progress_frames(&raw).to_owned();
                let log_line = if is_stderr {
                    format!("[stderr] {}", text)
                } else {
                    text.clone()
                };

                diagnostics::append_phase_line(
                    &backend_log.handle,
                    if is_stderr { "stderr" } else { "stdout" },
                    &text,
                );

                let detected_port = if !is_stderr {
                    port_re
                        .captures(&text)
                        .and_then(|caps| caps.get(1))
                        .and_then(|port_str| port_str.as_str().parse::<u16>().ok())
                } else {
                    None
                };

                // Old reader threads can outlive a restart; only the current generation may record
                // port or logs.
                let mut candidate_port = None;
                let current_generation = if let Ok(mut proc) = state.lock() {
                    if proc.generation != generation {
                        false
                    } else {
                        candidate_port = detected_port;
                        if proc.logs.len() >= MAX_LOG_LINES {
                            proc.logs.pop_front();
                        }
                        proc.logs.push_back(log_line.clone());
                        true
                    }
                } else {
                    false
                };

                if !current_generation {
                    break;
                }

                if let Some(port) = candidate_port {
                    let app_handle = app.clone();
                    let state_clone = Arc::clone(state);
                    let diagnostics_clone = diagnostics_state.clone();
                    let session_id = backend_log.session_id.clone();
                    tauri::async_runtime::spawn(async move {
                        validate_candidate_port(
                            app_handle,
                            state_clone,
                            diagnostics_clone,
                            session_id,
                            generation,
                            port,
                            start_deadline,
                        )
                        .await;
                    });
                }

                if is_backend_access_log_line(&text) {
                    debug!("[backend] {}", log_line);
                } else if log_line.len() > MAX_BACKEND_LOG_LINE_BYTES {
                    let mut end = MAX_BACKEND_LOG_LINE_BYTES;
                    while end > 0 && !log_line.is_char_boundary(end) {
                        end -= 1;
                    }
                    info!("[backend] {} [line truncated]", &log_line[..end]);
                } else {
                    info!("[backend] {}", log_line);
                }

                let _ = app.emit("server-log", &log_line);
            }
            Err(e) => {
                warn!(
                    "Error reading backend {}: {}",
                    if is_stderr { "stderr" } else { "stdout" },
                    e
                );
                break;
            }
        }
    }

    // Leaving early must not drop the read end of a live child: its next write gets EPIPE and it dies.
    // Keep draining to EOF.
    if !saw_eof {
        warn!(
            "Backend {} reader stopped parsing without eof (generation {}); draining so \
             the child keeps a reader",
            if is_stderr { "stderr" } else { "stdout" },
            generation
        );
        use std::io::Read;
        let mut sink = [0u8; 8192];
        loop {
            match reader.read(&mut sink) {
                Ok(0) => break,
                Ok(_) => continue,
                // Raw reads do not retry EINTR; giving up would drop the read end (EPIPE above).
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
    }

    // Stream closed. Only the stdout reader checks for crashes.
    if !is_stderr {
        let mut exit_record: Option<(String, bool)> = None;
        let mut emit_crash = false;
        if let Ok(mut proc) = state.lock() {
            if proc.generation != generation {
                return;
            }
            let intentional = proc.intentional_stop;
            let exited = if let Some(child) = proc
                .owned
                .as_mut()
                .and_then(OwnedBackendHandle::spawned_child_mut)
            {
                match exit_status_after_stdout_closed(child) {
                    Some(status) => {
                        info!("Backend stdout stream ended with status: {}", status);
                        exit_record = Some((status, intentional));
                        true
                    }
                    None => {
                        warn!(
                            "Backend stdout stream ended and the process has still not \
                             reported an exit status; leaving it marked as running"
                        );
                        false
                    }
                }
            } else {
                false
            };

            if exited {
                if let Some(owned) = proc.owned.take() {
                    owned.remove_owner_metadata();
                }
                proc.port = None;
                proc.diagnostics_session = None;
                emit_crash = !intentional;
            }
        }
        if let Some((status, intentional)) = exit_record {
            diagnostics::record_backend_exit(
                diagnostics_state,
                &backend_log.session_id,
                Some(status),
                intentional,
                None,
            );
        }
        if emit_crash {
            error!("Backend process stdout closed unexpectedly (crash detected)");
            let _ = app.emit("server-crashed", ());
        }
    }
}

/// Exit status of a child whose stdout closed, or None if still alive. Polls briefly since EOF can
/// precede exit visibility; stdout EOF alone is not death (the backend may close stdout).
fn exit_status_after_stdout_closed(child: &mut Box<dyn ChildWrapper + Send>) -> Option<String> {
    for attempt in 0..30 {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status.to_string()),
            Ok(None) => {
                if attempt > 0 {
                    std::thread::sleep(Duration::from_millis(100));
                }
            }
            Err(e) => {
                warn!("Failed to query backend status after stdout closed: {}", e);
                return None;
            }
        }
    }
    None
}

fn wait_for_child_exit(child: &mut Box<dyn ChildWrapper + Send>, label: &str) -> bool {
    for _ in 0..50 {
        match child.try_wait() {
            Ok(Some(status)) => {
                info!("{} exited with status: {}", label, status);
                return true;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(100)),
            Err(e) => {
                warn!("Error polling {} process: {}", label, e);
                return false;
            }
        }
    }
    false
}

fn wait_for_port_disconnect(port: u16, timeout: Duration) -> bool {
    let started = std::time::Instant::now();
    while started.elapsed() < timeout {
        if !crate::desktop_backend_owner::port_is_listening_blocking(
            port,
            Duration::from_millis(150),
        ) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    false
}

fn try_exact_port_http_shutdown(port: u16, label: &str) -> bool {
    match crate::desktop_backend_owner::exact_port_http_shutdown_blocking(port) {
        Ok(()) => {
            info!(
                "{} exact-port HTTP shutdown requested on port {}",
                label, port
            );
            true
        }
        Err(error) => {
            warn!(
                "{} exact-port HTTP shutdown failed on port {}: {}",
                label, port, error
            );
            false
        }
    }
}

fn remove_optional_owner(owner: Option<crate::desktop_backend_owner::BackendOwnerState>) {
    if let Some(owner) = owner {
        owner.remove();
    }
}

fn stop_spawned_backend(
    mut child: Box<dyn ChildWrapper + Send>,
    owner: Option<crate::desktop_backend_owner::BackendOwnerState>,
    reported_port: Option<u16>,
    pid: u32,
) -> Result<(), String> {
    #[cfg(not(windows))]
    let _ = reported_port;
    info!("Stopping spawned backend process group (pid {})", pid);

    #[cfg(windows)]
    if let Some(port) = reported_port {
        let verified = owner
            .as_ref()
            .map(|owner| owner.verifies_exact_port_blocking(port))
            .unwrap_or(false);
        if verified
            && try_exact_port_http_shutdown(port, "Spawned backend")
            && wait_for_child_exit(&mut child, "Backend")
        {
            remove_optional_owner(owner);
            return Ok(());
        }
    }

    #[cfg(unix)]
    {
        if pid > i32::MAX as u32 {
            warn!("PID {} exceeds i32 range, using direct kill", pid);
            let _ = child.kill();
            let _ = child.wait();
            remove_optional_owner(owner);
            return Ok(());
        }
        unsafe {
            libc::kill(-(pid as i32), libc::SIGTERM);
        }
    }

    #[cfg(windows)]
    {
        unsafe {
            windows_sys::Win32::System::Console::GenerateConsoleCtrlEvent(
                windows_sys::Win32::System::Console::CTRL_BREAK_EVENT,
                pid,
            );
        }
    }

    if wait_for_child_exit(&mut child, "Backend") {
        remove_optional_owner(owner);
        return Ok(());
    }

    #[cfg(windows)]
    {
        warn!(
            "Backend did not exit gracefully, force killing process tree (pid {})",
            pid
        );
        force_kill_process_tree(pid, &mut child, "Backend");
        remove_optional_owner(owner);
        return Ok(());
    }

    #[cfg(unix)]
    {
        warn!(
            "Backend did not exit gracefully, force killing group (pid {})",
            pid
        );
        let _ = child.kill();
        let _ = child.wait();
        remove_optional_owner(owner);
        info!("Backend process group forcefully stopped");
        Ok(())
    }
}

fn stop_adopted_backend(
    owner: crate::desktop_backend_owner::BackendOwnerState,
    port: u16,
    pid: u32,
) -> Result<(), String> {
    info!(
        "Stopping adopted desktop-owned backend on exact port {} (pid {})",
        port, pid
    );

    if !owner.verifies_exact_port_blocking(port) {
        return Err(
            "Refusing to stop adopted backend because ownership could not be verified".to_string(),
        );
    }

    if try_exact_port_http_shutdown(port, "Adopted backend")
        && wait_for_port_disconnect(port, Duration::from_secs(5))
    {
        owner.remove();
        return Ok(());
    }

    Err(
        "Adopted backend did not stop via exact-port HTTP shutdown; refusing PID fallback without verified port-to-PID binding"
            .to_string(),
    )
}

/// Graceful shutdown of owned backend handles.
/// Unix spawned: SIGTERM to process group -> wait -> SIGKILL.
/// Windows spawned: exact-port HTTP shutdown -> CTRL_BREAK_EVENT -> taskkill.
/// Adopted handles: exact-port HTTP shutdown only; PID fallback is refused
/// until the backend process identity can be bound to the verified port.
pub fn stop_backend(
    state: &BackendState,
    shutdown: &ShutdownFlag,
    diagnostics_state: Option<&DiagnosticsState>,
) -> Result<(), String> {
    stop_backend_inner(state, shutdown, diagnostics_state, true)
}

/// Stop before update/repair mutations without letting the watchdog exit unless the stop succeeds.
pub fn stop_backend_for_mutation(
    state: &BackendState,
    shutdown: &ShutdownFlag,
    diagnostics_state: Option<&DiagnosticsState>,
) -> Result<(), String> {
    stop_backend_inner(state, shutdown, diagnostics_state, false)
}

fn stop_backend_inner(
    state: &BackendState,
    shutdown: &ShutdownFlag,
    diagnostics_state: Option<&DiagnosticsState>,
    signal_shutdown_before_stop: bool,
) -> Result<(), String> {
    let previous_shutdown = shutdown.load(Ordering::SeqCst);
    if signal_shutdown_before_stop {
        shutdown.store(true, Ordering::SeqCst);
    }
    if let Some(diagnostics_state) = diagnostics_state {
        diagnostics::record_backend_intentional_stop(diagnostics_state);
    }

    enum StopTarget {
        Spawned(OwnedBackendHandle),
        Adopted {
            owner: crate::desktop_backend_owner::BackendOwnerState,
            port: u16,
            pid: u32,
            generation: u64,
            local_generation: u64,
        },
    }

    let target = {
        let mut proc = match state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!("Backend state mutex poisoned, recovering for cleanup");
                poisoned.into_inner()
            }
        };
        proc.intentional_stop = true;
        match proc.owned.as_ref() {
            Some(OwnedBackendHandle::Spawned { .. }) => {
                proc.port = None;
                proc.diagnostics_session = None;
                proc.adopted_watchdog_generation = None;
                proc.owned.take().map(StopTarget::Spawned)
            }
            Some(OwnedBackendHandle::Adopted {
                owner,
                port,
                pid,
                generation,
            }) => Some(StopTarget::Adopted {
                owner: owner.clone(),
                port: *port,
                pid: *pid,
                generation: *generation,
                local_generation: proc.generation,
            }),
            None => None,
        }
    };

    let result = match target {
        Some(StopTarget::Spawned(OwnedBackendHandle::Spawned {
            child,
            owner,
            reported_port,
            pid,
            ..
        })) => stop_spawned_backend(child, owner, reported_port, pid),
        Some(StopTarget::Adopted {
            owner,
            port,
            pid,
            generation,
            local_generation,
        }) => {
            if let Err(error) = stop_adopted_backend(owner, port, pid) {
                if !crate::desktop_backend_owner::port_is_listening_blocking(
                    port,
                    Duration::from_millis(150),
                ) {
                    clear_adopted_backend_if_current(
                        state,
                        local_generation,
                        Some(port),
                        "adopted port disappeared during stop",
                    );
                    Ok(())
                } else {
                    Err(error)
                }
            } else {
                let mut proc = match state.lock() {
                    Ok(guard) => guard,
                    Err(poisoned) => {
                        warn!("Backend state mutex poisoned, recovering after adopted stop");
                        poisoned.into_inner()
                    }
                };
                if matches!(
                    proc.owned.as_ref(),
                    Some(OwnedBackendHandle::Adopted {
                        port: current_port,
                        pid: current_pid,
                        generation: current_generation,
                        ..
                    }) if *current_port == port && *current_pid == pid && *current_generation == generation
                ) {
                    proc.owned = None;
                    proc.port = None;
                    proc.diagnostics_session = None;
                    proc.adopted_watchdog_generation = None;
                }
                Ok(())
            }
        }
        Some(StopTarget::Spawned(OwnedBackendHandle::Adopted { .. })) => unreachable!(),
        None => Ok(()),
    };

    if result.is_ok() && !signal_shutdown_before_stop {
        shutdown.store(true, Ordering::SeqCst);
    } else if result.is_err() && signal_shutdown_before_stop {
        shutdown.store(previous_shutdown, Ordering::SeqCst);
    }

    result
}

#[cfg(test)]
mod backend_log_line_tests {
    use super::*;

    #[test]
    fn plain_line_is_untouched() {
        assert_eq!(collapse_progress_frames("Hardware detected: ROCm"), "Hardware detected: ROCm");
        assert_eq!(collapse_progress_frames(""), "");
    }

    #[test]
    fn progress_bar_keeps_only_the_final_frame() {
        let bar = "Loading weights:   0%| | 0/617\rLoading weights:  47%| | 288/617\rLoading weights: 100%|#| 617/617";
        assert_eq!(collapse_progress_frames(bar), "Loading weights: 100%|#| 617/617");
    }

    #[test]
    fn trailing_blank_frame_falls_back_to_the_last_real_one() {
        assert_eq!(collapse_progress_frames("Map:  50%\rMap: 100%\r   "), "Map: 100%");
    }

    #[test]
    fn an_all_blank_line_never_keeps_its_carriage_returns() {
        // The log handle appends its own terminator, so a surviving `\r` becomes "\r\r\n" on Windows.
        for line in ["\r", "\r\r\r", "   \r   "] {
            assert!(
                !collapse_progress_frames(line).contains('\r'),
                "collapse left a carriage return in {line:?}"
            );
        }
    }

    #[test]
    fn a_crlf_line_keeps_its_payload() {
        let raw = b"Hardware detected: NVIDIA GeForce RTX 4090\r\n";
        let trimmed = String::from_utf8_lossy(trim_line_endings(raw)).into_owned();
        assert_eq!(
            collapse_progress_frames(&trimmed),
            "Hardware detected: NVIDIA GeForce RTX 4090"
        );
    }

    #[test]
    fn a_crlf_terminated_bar_still_collapses() {
        let raw = b"Map:  50%\rMap: 100%\r\n";
        let trimmed = String::from_utf8_lossy(trim_line_endings(raw)).into_owned();
        assert_eq!(collapse_progress_frames(&trimmed), "Map: 100%");
    }

    #[test]
    fn a_crlf_json_record_stays_parseable() {
        let raw = b"{\"event\": \"request_completed\", \"status_code\": 200}\r\n";
        let trimmed = String::from_utf8_lossy(trim_line_endings(raw)).into_owned();
        let line = collapse_progress_frames(&trimmed);
        assert!(serde_json::from_str::<serde_json::Value>(line).is_ok(), "{line:?}");
        assert!(is_backend_access_log_line(line));
    }

    #[test]
    fn a_marker_line_survives_the_collapse() {
        for raw in [
            &b"TAURI_PORT=8888\n"[..],
            &b"TAURI_PORT=8888\r\n"[..],
            &b"Loading weights:  47%\rTAURI_PORT=8888\r\n"[..],
        ] {
            let trimmed = String::from_utf8_lossy(trim_line_endings(raw)).into_owned();
            assert_eq!(collapse_progress_frames(&trimmed), "TAURI_PORT=8888", "{raw:?}");
        }
    }

    #[test]
    fn truncation_lands_on_a_character_boundary() {
        let line = "\u{1f680}".repeat(MAX_BACKEND_LOG_LINE_BYTES);
        let mut end = MAX_BACKEND_LOG_LINE_BYTES;
        while end > 0 && !line.is_char_boundary(end) {
            end -= 1;
        }
        assert!(end <= MAX_BACKEND_LOG_LINE_BYTES);
        assert_eq!(line[..end].len(), end);
    }

    #[test]
    fn access_log_records_are_recognised() {
        assert!(is_backend_access_log_line(
            r#"{"timestamp": "2026-08-13T14:22:11Z", "level": "info", "event": "request_completed", "path": "/api/liveness", "status_code": 200}"#
        ));
        assert!(is_backend_access_log_line(
            r#"{"event": "request_completed", "path": "/api/models/local", "status_code": 204}"#
        ));
    }

    #[test]
    fn failed_access_records_keep_their_info_line() {
        assert!(!is_backend_access_log_line(
            r#"{"event": "request_completed", "path": "/api/liveness", "status_code": 503}"#
        ));
        assert!(!is_backend_access_log_line(
            r#"{"event": "request_completed", "path": "/api/train/start", "status_code": 401}"#
        ));
        assert!(!is_backend_access_log_line(
            r#"{"event": "request_completed", "path": "/api/liveness"}"#
        ));
    }

    #[test]
    fn other_structured_events_and_plain_text_are_not() {
        assert!(!is_backend_access_log_line(
            r#"{"level": "info", "event": "engine_stats", "gen_tok_s": 64.9}"#
        ));
        assert!(!is_backend_access_log_line("TAURI_PORT=8888"));
        assert!(!is_backend_access_log_line("saw request_completed in the trace"));
        assert!(!is_backend_access_log_line(
            r#"{"event": "request_completed", "status_code": 200"#
        ));
    }
}

/// Shared corpus with the backend so both implementations of the rule cannot drift.
#[cfg(test)]
mod shared_access_log_fixture_tests {
    use super::*;

    #[test]
    fn every_shared_case_agrees_with_the_desktop_filter() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../backend/tests/fixtures/access_log_records.json");
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read the shared fixture at {path:?}: {e}"));
        let fixture: serde_json::Value =
            serde_json::from_str(&raw).expect("the shared fixture is not valid JSON");
        let cases = fixture["cases"]
            .as_array()
            .expect("the shared fixture has no `cases` array");
        assert!(!cases.is_empty(), "the shared fixture is empty, so this test proves nothing");

        for case in cases {
            let name = case["name"].as_str().unwrap_or("<unnamed>");
            let line = case["line"].as_str().unwrap_or_else(|| {
                panic!("case {name:?} has no `line`");
            });
            let keep = case["keep"]
                .as_bool()
                .unwrap_or_else(|| panic!("case {name:?} has no boolean `keep`"));
            assert_eq!(
                is_backend_access_log_line(line),
                !keep,
                "case {name:?}: {}",
                case["why"].as_str().unwrap_or("no rationale recorded")
            );
        }
    }
}

#[cfg(test)]
mod managed_cli_working_dir_tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn scratch(test_name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "unsloth-{test_name}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_normal_home_yields_the_unsloth_directory_and_creates_it() {
        let home = scratch("cwd-normal-home");
        let resolved = managed_cli_working_dir_from(Some(home.clone()), &[])
            .expect("a normal home must resolve");
        assert_eq!(resolved, home.join(".unsloth"));
        assert!(resolved.is_dir(), "the working directory must exist");
        fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn an_existing_working_directory_is_reused() {
        let home = scratch("cwd-existing");
        fs::create_dir_all(home.join(".unsloth")).unwrap();
        let resolved = managed_cli_working_dir_from(Some(home.clone()), &[]).unwrap();
        assert_eq!(resolved, home.join(".unsloth"));
        fs::remove_dir_all(&home).ok();
    }

    #[test]
    fn a_home_with_spaces_and_non_ascii_survives_intact() {
        let base = scratch("cwd-unicode");
        let home = base.join("Jane O'Brien ünïcode");
        fs::create_dir_all(&home).unwrap();
        let resolved = managed_cli_working_dir_from(Some(home.clone()), &[]).unwrap();
        assert_eq!(resolved, home.join(".unsloth"));
        fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn a_home_inside_the_windows_directory_is_not_reported_as_available() {
        let windirs = [PathBuf::from("C:\\Windows")];
        let home = PathBuf::from("C:\\Windows\\System32\\config\\systemprofile");
        let error = usable_home_dir(Some(home), &windirs, true).unwrap_err();
        assert!(
            error.contains("inside the Windows directory"),
            "unexpected error: {error}"
        );
        let real_home = scratch("usable-home");
        fs::create_dir_all(&real_home).unwrap();
        assert_eq!(
            usable_home_dir(Some(real_home.clone()), &windirs, true).unwrap(),
            real_home
        );
        fs::remove_dir_all(&real_home).ok();
    }

    #[test]
    fn relative_path_overrides_are_pinned_only_when_the_child_moves() {
        let cwd = PathBuf::from("C:\\Windows\\System32");
        let work_dir = PathBuf::from("C:\\Users\\me\\.unsloth");
        let env = |name: &str| match name {
            "HF_HOME" => Some("cache".to_string()),
            "UNSLOTH_COMPILE_LOCATION" => Some("  studio  ".to_string()),
            "OLLAMA_MODELS" => Some("D:\\models".to_string()),
            "HF_HUB_CACHE" => Some("C:\\hub".to_string()),
            "XDG_CACHE_HOME" => Some("   ".to_string()),
            "LLAMA_SERVER_PATH" => Some("\\srv\\llama-server".to_string()),
            "HF_DATASETS_CACHE" => Some("D:datasets".to_string()),
            _ => None,
        };
        let absolute = |value: &str| match value {
            "D:datasets" => Some(PathBuf::from("D:\\work\\datasets")),
            "\\srv\\llama-server" => Some(PathBuf::from("C:\\srv\\llama-server")),
            other => panic!("unexpected value needing the OS: {other}"),
        };

        let pins =
            relative_override_pins_from(Some(cwd.clone()), &work_dir, env, absolute, Some(std::path::Path::new("C:\\Users\\me")), MANAGED_CHILD_SCRUBBED_ENV, true).unwrap();
        assert_eq!(
            pins,
            vec![
                (
                    "LLAMA_SERVER_PATH",
                    PathBuf::from("C:\\srv\\llama-server")
                ),
                ("UNSLOTH_COMPILE_LOCATION", cwd.join("studio")),
                ("HF_HOME", cwd.join("cache")),
                (
                    "HF_DATASETS_CACHE",
                    PathBuf::from("D:\\work\\datasets")
                ),
            ],
            "only values that name no directory on their own are rewritten"
        );

        assert!(relative_override_pins_from(Some(cwd.clone()), &work_dir, env, |_| None, Some(std::path::Path::new("C:\\Users\\me")), MANAGED_CHILD_SCRUBBED_ENV, true).is_err());

        assert!(
            relative_override_pins_from(Some(work_dir.clone()), &work_dir, env, absolute, Some(std::path::Path::new("C:\\Users\\me")), MANAGED_CHILD_SCRUBBED_ENV, true)
                .unwrap()
                .is_empty()
        );
        assert!(relative_override_pins_from(None, &work_dir, env, absolute, Some(std::path::Path::new("C:\\Users\\me")), MANAGED_CHILD_SCRUBBED_ENV, true).is_err());
        let absolute_only = |name: &str| match name {
            "HF_HOME" => Some("D:\\cache".to_string()),
            "HF_HUB_CACHE" => Some("C:\\hub".to_string()),
            "UNSLOTH_STUDIO_HOME" => Some("studio".to_string()),
            _ => None,
        };
        assert!(
            relative_override_pins_from(None, &work_dir, absolute_only, absolute, Some(std::path::Path::new("C:\\Users\\me")), MANAGED_CHILD_SCRUBBED_ENV, true)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn a_profile_that_is_not_there_yet_stops_a_managed_child_but_not_the_installer() {
        let missing = scratch("absent-profile").join("someone");
        let error = managed_cli_working_dir_from(Some(missing.clone()), &[]).unwrap_err();
        assert!(error.contains("not reachable"), "unexpected error: {error}");
        assert_eq!(
            install_working_dir(Some(missing.clone())).unwrap(),
            missing.join(".unsloth")
        );
        assert!(missing.join(".unsloth").is_dir());
        fs::remove_dir_all(missing.parent().unwrap()).ok();
    }

    #[test]
    fn the_update_child_skips_the_value_the_windows_update_drops_anyway() {
        let skipped = update_child_skipped_env();
        assert_eq!(
            skipped.contains(&"PYTHONPATH"),
            cfg!(windows),
            "PYTHONPATH is skipped exactly where the update drops it"
        );
        assert!(!skipped.contains(&"STUDIO_LOCAL_REPO"));
        assert!(child_skipped_env().contains(&"STUDIO_LOCAL_REPO"));
    }

    #[cfg(unix)]
    #[test]
    fn a_named_user_override_is_pinned_where_the_fingerprint_watches() {
        // root exists on every unix box and matches preflight::managed's own test.
        let cwd = PathBuf::from("/mnt/work/session");
        let work_dir = PathBuf::from("/home/me/.unsloth");
        let pins = relative_override_pins_from(
            Some(cwd.clone()),
            &work_dir,
            |name: &str| {
                (name == "UNSLOTH_LLAMA_CPP_PATH").then(|| "~root/llama.cpp".to_string())
            },
            |value: &str| panic!("unexpected value needing the OS: {value}"),
            Some(std::path::Path::new("/home/me")),
            MANAGED_CHILD_SCRUBBED_ENV,
            false,
        )
        .unwrap();
        let pinned = pins
            .iter()
            .find(|(name, _)| *name == "UNSLOTH_LLAMA_CPP_PATH")
            .map(|(_, path)| path.clone())
            .expect("the override has to be pinned, not dropped");
        assert!(
            !pinned.starts_with(&cwd),
            "a named-user override was anchored under the directory being left: {}",
            pinned.display()
        );
        let watched = crate::preflight::managed::named_user_home(
            "~root/llama.cpp",
            Some(std::path::Path::new("/home/me")),
        )
        .expect("root must resolve, or the lookup is not answering at all");
        assert_eq!(pinned, watched, "the child and the cache must grade one tree");
    }

    #[cfg(unix)]
    #[test]
    fn an_unknown_named_user_is_left_alone_and_anchored_by_both_halves() {
        let cwd = PathBuf::from("/mnt/work/session");
        let value = "~no-such-account-anywhere/llama.cpp";
        let pins = relative_override_pins_from(
            Some(cwd.clone()),
            std::path::Path::new("/home/me/.unsloth"),
            |name: &str| (name == "UNSLOTH_LLAMA_CPP_PATH").then(|| value.to_string()),
            |value: &str| panic!("unexpected value needing the OS: {value}"),
            Some(std::path::Path::new("/home/me")),
            MANAGED_CHILD_SCRUBBED_ENV,
            false,
        )
        .unwrap();
        assert_eq!(
            pins,
            vec![("UNSLOTH_LLAMA_CPP_PATH", cwd.join(value))],
            "an unresolvable name is anchored, the way the fingerprint anchors it"
        );
    }

    #[test]
    fn the_child_and_the_fingerprint_expand_a_bare_tilde_to_one_home() {
        let home = crate::preflight::managed::tilde_home();
        let pinned = expand_user("~/llama.cpp", home.as_deref(), None, cfg!(windows));
        let watched = crate::preflight::managed::llama_runtime_override_from(
            Some("~/llama.cpp"),
            home.as_deref(),
            Some(std::path::Path::new("/nowhere-relative")),
        );
        assert_eq!(
            Some(std::path::PathBuf::from(pinned)),
            watched,
            "the tree the child is pinned to and the tree the cache watches must be one"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_bare_tilde_override_still_reaches_home_off_windows() {
        let pins = relative_override_pins_from(
            Some(PathBuf::from("/mnt/work/session")),
            std::path::Path::new("/home/me/.unsloth"),
            |name: &str| (name == "UNSLOTH_LLAMA_CPP_PATH").then(|| "~/llama.cpp".to_string()),
            |value: &str| panic!("unexpected value needing the OS: {value}"),
            Some(std::path::Path::new("/home/me")),
            MANAGED_CHILD_SCRUBBED_ENV,
            false,
        )
        .unwrap();
        assert_eq!(
            pins,
            vec![("UNSLOTH_LLAMA_CPP_PATH", PathBuf::from("/home/me/llama.cpp"))]
        );
    }

    #[test]
    fn a_posix_list_is_split_and_joined_with_its_own_separator() {
        let cwd = PathBuf::from("/mnt/work/session");
        let work_dir = PathBuf::from("/home/me/.unsloth");
        let pins = relative_override_pins_from(
            Some(cwd.clone()),
            &work_dir,
            |name: &str| (name == "PYTHONPATH").then(|| "plugins:/opt/vendor".to_string()),
            |value: &str| panic!("unexpected value needing the OS: {value}"),
            Some(std::path::Path::new("/home/me")),
            MANAGED_CHILD_SCRUBBED_ENV,
            false,
        )
        .unwrap();
        assert_eq!(
            pins,
            vec![(
                "PYTHONPATH",
                PathBuf::from(format!("{}:/opt/vendor", cwd.join("plugins").display()))
            )]
        );
    }

    #[test]
    fn a_scalar_that_would_not_fit_is_reported_too() {
        let cwd = PathBuf::from("C:\\Windows\\System32");
        let long = "x".repeat(WINDOWS_ENV_VALUE_LIMIT);
        let error = relative_override_pins_from(
            Some(cwd),
            std::path::Path::new("C:\\Users\\me\\.unsloth"),
            |name: &str| (name == "HF_HOME").then(|| long.clone()),
            |value: &str| panic!("unexpected value needing the OS: {value}"),
            Some(std::path::Path::new("C:\\Users\\me")),
            MANAGED_CHILD_SCRUBBED_ENV,
            true,
        )
        .unwrap_err();
        assert!(error.starts_with("HF_HOME does not fit"), "{error}");
    }

    #[test]
    fn a_posix_reader_expands_its_own_spelling_and_no_other() {
        let lookup = |name: &str| match name {
            "HOME" => Some("/home/me".to_string()),
            _ => None,
        };
        assert_eq!(expand_posix_vars("$HOME/hf", &lookup), "/home/me/hf");
        assert_eq!(expand_posix_vars("${HOME}/hf", &lookup), "/home/me/hf");
        assert_eq!(expand_posix_vars("%HOME%/hf", &lookup), "%HOME%/hf");
        assert_eq!(expand_posix_vars("$UNSET/hf", &lookup), "$UNSET/hf");
        assert_eq!(expand_posix_vars("${UNTERMINATED/hf", &lookup), "${UNTERMINATED/hf");
        assert_eq!(expand_posix_vars("cost: $5", &lookup), "cost: $5");
        assert_eq!(expand_posix_vars("plain/path", &lookup), "plain/path");

        let error = relative_override_pins_from(
            None,
            std::path::Path::new("/home/me/.unsloth"),
            |name: &str| match name {
                "HOME" => Some("/home/me".to_string()),
                "HF_HOME" => Some("%HOME%/hf".to_string()),
                _ => None,
            },
            |value: &str| panic!("unexpected value needing the OS: {value}"),
            Some(std::path::Path::new("/home/me")),
            MANAGED_CHILD_SCRUBBED_ENV,
            false,
        )
        .unwrap_err();
        assert!(error.contains("HF_HOME"), "{error}");
        assert_eq!(
            relative_override_pins_from(
                None,
                std::path::Path::new("/home/me/.unsloth"),
                |name: &str| match name {
                    "HOME" => Some("/home/me".to_string()),
                    "HF_HOME" => Some("$HOME/hf".to_string()),
                    _ => None,
                },
                |value: &str| panic!("unexpected value needing the OS: {value}"),
                Some(std::path::Path::new("/home/me")),
                MANAGED_CHILD_SCRUBBED_ENV,
                false,
            )
            .unwrap(),
            Vec::new()
        );
    }

    #[test]
    fn a_list_that_would_not_fit_is_reported_rather_than_spawned() {
        let cwd = PathBuf::from("C:\\Windows\\System32");
        let work_dir = PathBuf::from("C:\\Users\\me\\.unsloth");
        let long = std::iter::repeat("entry")
            .take(WINDOWS_ENV_VALUE_LIMIT / 5)
            .collect::<Vec<_>>()
            .join(";");
        let error = relative_override_pins_from(
            Some(cwd),
            &work_dir,
            |name: &str| (name == "PYTHONPATH").then(|| long.clone()),
            |value: &str| panic!("unexpected value needing the OS: {value}"),
            Some(std::path::Path::new("C:\\Users\\me")),
            MANAGED_CHILD_SCRUBBED_ENV,
            true,
        )
        .unwrap_err();
        assert!(error.starts_with("PYTHONPATH does not fit"), "{error}");
    }

    #[test]
    fn the_tilde_follows_the_profile_the_cli_guard_reads() {
        let cwd = PathBuf::from("C:\\Windows\\System32");
        let work_dir = PathBuf::from("C:\\Users\\me\\.unsloth");
        let pins = relative_override_pins_from(
            Some(cwd),
            &work_dir,
            |name: &str| match name {
                "USERPROFILE" => Some("D:\\portable\\me".to_string()),
                "UNSLOTH_LLAMA_CPP_PATH" => Some("~\\llama.cpp".to_string()),
                _ => None,
            },
            |value: &str| panic!("unexpected value needing the OS: {value}"),
            Some(std::path::Path::new("C:\\Users\\me")),
            MANAGED_CHILD_SCRUBBED_ENV,
            true,
        )
        .unwrap();
        assert_eq!(
            pins,
            vec![(
                "UNSLOTH_LLAMA_CPP_PATH",
                PathBuf::from("D:\\portable\\me\\llama.cpp")
            )]
        );
    }

    #[test]
    fn the_pin_decision_is_a_table_with_no_other_outcomes() {
        let home = std::path::Path::new("C:\\Users\\me");
        let work_dir = PathBuf::from("C:\\Users\\me\\.unsloth");
        let cwds = [
            (Some(PathBuf::from("C:\\Users\\me\\project")), "elsewhere"),
            (Some(work_dir.clone()), "already-there"),
            (Some(PathBuf::from("C:\\Windows\\System32")), "system"),
            (None, "unknown"),
        ];
        let envs: [(&dyn Fn(&str) -> Option<String>, &str); 4] = [
            (&|_: &str| None, "clean"),
            (
                &|name: &str| (name == "HF_HOME").then(|| "cache".to_string()),
                "relative",
            ),
            (
                &|name: &str| (name == "HF_HOME").then(|| "D:\\cache".to_string()),
                "absolute",
            ),
            (
                &|name: &str| {
                    (name == "UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH").then(|| "1".to_string())
                },
                "toggle",
            ),
        ];
        let absolute = |value: &str| panic!("unexpected value needing the OS: {value}");
        let mut table = Vec::new();
        for (cwd, cwd_kind) in &cwds {
            for (lookup, env_kind) in &envs {
                let outcome = relative_override_pins_from(
                    cwd.clone(),
                    &work_dir,
                    lookup,
                    absolute,
                    Some(home),
                    MANAGED_CHILD_SCRUBBED_ENV,
                    true,
                );
                let cell = match &outcome {
                    Ok(pins) if pins.is_empty() => "nothing rewritten",
                    Ok(_) => "anchored to the directory being left",
                    Err(_) => "reported as unpreservable",
                };
                table.push((*cwd_kind, *env_kind, cell));
            }
        }
        assert_eq!(
            table,
            vec![
                ("elsewhere", "clean", "nothing rewritten"),
                ("elsewhere", "relative", "anchored to the directory being left"),
                ("elsewhere", "absolute", "nothing rewritten"),
                ("elsewhere", "toggle", "nothing rewritten"),
                ("already-there", "clean", "nothing rewritten"),
                ("already-there", "relative", "nothing rewritten"),
                ("already-there", "absolute", "nothing rewritten"),
                ("already-there", "toggle", "nothing rewritten"),
                ("system", "clean", "nothing rewritten"),
                ("system", "relative", "anchored to the directory being left"),
                ("system", "absolute", "nothing rewritten"),
                ("system", "toggle", "nothing rewritten"),
                ("unknown", "clean", "nothing rewritten"),
                ("unknown", "relative", "reported as unpreservable"),
                ("unknown", "absolute", "nothing rewritten"),
                ("unknown", "toggle", "nothing rewritten"),
            ]
        );
    }

    #[test]
    fn a_directory_that_cannot_be_named_leaves_the_child_where_it_is() {
        let work_dir = PathBuf::from("C:\\Users\\me\\.unsloth");
        let relative = |name: &str| match name {
            "DG_VISUAL_BIN" => Some("visual".to_string()),
            _ => None,
        };
        let absolute = |value: &str| panic!("unexpected value needing the OS: {value}");
        assert!(
            relative_override_pins_from(
                None,
                &work_dir,
                relative,
                absolute,
                Some(std::path::Path::new("C:\\Users\\me")),
                MANAGED_CHILD_SCRUBBED_ENV,
                true,
            )
            .is_err(),
            "the pins still report what a move would lose"
        );
        let unpinnable: Result<Vec<(&'static str, PathBuf)>, String> =
            Err("DG_VISUAL_BIN is relative and the directory it was written against is gone"
                .to_string());
        assert_eq!(stay_put_on_lost_cwd(unpinnable.clone(), false), Ok(None));
        assert!(stay_put_on_lost_cwd(unpinnable, true).is_err());
    }

    #[test]
    fn an_expansion_that_stays_relative_refuses_the_move() {
        fn pins(
            lookup: impl Fn(&str) -> Option<String>,
        ) -> Result<Vec<(&'static str, PathBuf)>, String> {
            relative_override_pins_from(
                Some(PathBuf::from("C:\\Windows\\System32")),
                std::path::Path::new("C:\\Users\\me\\.unsloth"),
                lookup,
                |value: &str| panic!("unexpected value needing the OS: {value}"),
                Some(std::path::Path::new("C:\\Users\\me")),
                MANAGED_CHILD_SCRUBBED_ENV,
                true,
            )
        }
        let refused = |error: String| {
            assert!(
                error.contains("does not expand to one folder"),
                "unexpected error: {error}"
            );
        };
        refused(
            pins(|name: &str| match name {
                "USERPROFILE" => Some("C:\\Users\\me".to_string()),
                "NESTED" => Some("%USERPROFILE%\\AppData\\Local".to_string()),
                "HF_ASSETS_CACHE" => Some("%NESTED%\\assets".to_string()),
                _ => None,
            })
            .unwrap_err(),
        );
        refused(
            pins(|name: &str| match name {
                "USERPROFILE" => Some("C:\\Users\\me".to_string()),
                "XDG_CACHE_HOME" => Some("%%USERPROFILE%%\\xdg".to_string()),
                _ => None,
            })
            .unwrap_err(),
        );
        refused(
            pins(|name: &str| (name == "HF_HOME").then(|| "%HF_HOME%\\cache".to_string()))
                .unwrap_err(),
        );
        assert_eq!(
            pins(|name: &str| (name == "HF_ASSETS_CACHE")
                .then(|| "C:\\cache\\%UNSET%\\assets".to_string()))
            .unwrap(),
            Vec::new()
        );
    }

    #[test]
    fn the_model_paths_llama_server_reads_for_itself_are_pinned() {
        let cwd = PathBuf::from("C:\\Windows\\System32");
        let work_dir = PathBuf::from("C:\\Users\\me\\.unsloth");
        let env = |name: &str| match name {
            "LLAMA_ARG_MODEL" => Some("models\\qwen.gguf".to_string()),
            "LLAMA_ARG_MMPROJ" => Some(".\\mmproj.gguf".to_string()),
            "LLAMA_ARG_MODEL_DRAFT" => Some("draft.gguf".to_string()),
            "LLAMA_ARG_SPEC_DRAFT_MODEL" => Some("D:\\drafts\\small.gguf".to_string()),
            "LLAMA_ARG_MMPROJ_URL" => Some("https://example.invalid/proj.gguf".to_string()),
            _ => None,
        };
        let absolute = |value: &str| panic!("unexpected value needing the OS: {value}");
        let pins = relative_override_pins_from(
            Some(cwd.clone()),
            &work_dir,
            env,
            absolute,
            Some(std::path::Path::new("C:\\Users\\me")),
            MANAGED_CHILD_SCRUBBED_ENV,
            true,
        )
        .unwrap();
        assert_eq!(
            pins,
            vec![
                ("LLAMA_ARG_MODEL", cwd.join("models\\qwen.gguf")),
                ("LLAMA_ARG_MMPROJ", cwd.join(".\\mmproj.gguf")),
                ("LLAMA_ARG_MODEL_DRAFT", cwd.join("draft.gguf")),
            ]
        );
    }

    #[test]
    fn a_lost_directory_still_reads_a_setting_that_never_depended_on_it() {
        let work_dir = std::path::PathBuf::from("C:\\Users\\me\\.unsloth");
        let env = |name: &str| match name {
            "LOCALAPPDATA" => Some("C:\\Users\\me\\AppData\\Local".to_string()),
            "HF_HOME" => Some("%LOCALAPPDATA%\\hf".to_string()),
            "MLX_HOSTFILE" => Some("[\"127.0.0.1\"]".to_string()),
            "UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH" => Some("1".to_string()),
            _ => None,
        };
        let absolute = |_: &str| None;
        assert!(
            relative_override_pins_from(
                None,
                &work_dir,
                env,
                absolute,
                Some(std::path::Path::new("C:\\Users\\me")),
                MANAGED_CHILD_SCRUBBED_ENV,
                true
            )
            .unwrap()
            .is_empty()
        );
        let with_relative = |name: &str| match name {
            "UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH" => Some("1;models".to_string()),
            _ => None,
        };
        assert!(
            relative_override_pins_from(
                None,
                &work_dir,
                with_relative,
                absolute,
                Some(std::path::Path::new("C:\\Users\\me")),
                MANAGED_CHILD_SCRUBBED_ENV,
                true
            )
            .is_err()
        );
    }

    #[test]
    fn the_pinned_override_list_matches_the_cli_guard() {
        // Must stay in sync with the CLI guard's list.
        let guard = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../unsloth_cli/_system_dir_guard.py");
        let source = fs::read_to_string(&guard).unwrap();
        let start = source.find("_RELATIVE_PATH_ENV = (").unwrap();
        let block = &source[start..start + source[start..].find("\n)").unwrap()];
        for name in RELATIVE_PATH_ENV {
            assert!(
                block.contains(&format!("\"{name}\"")),
                "{name} is pinned by the desktop but not by the CLI guard"
            );
        }
        let names = block.matches('"').count() / 2;
        assert_eq!(
            names,
            RELATIVE_PATH_ENV.len(),
            "the CLI guard pins names the desktop does not"
        );
    }

    #[test]
    fn a_missing_home_is_an_error_rather_than_a_fallback() {
        let error = managed_cli_working_dir_from(None, &[]).unwrap_err();
        assert!(
            error.contains("home directory"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn a_home_that_does_not_exist_is_rejected() {
        let home = scratch("cwd-absent").join("gone");
        let error = managed_cli_working_dir_from(Some(home), &[]).unwrap_err();
        assert!(error.contains("not reachable"), "unexpected error: {error}");
    }

    #[test]
    fn a_file_masquerading_as_a_home_is_rejected() {
        let base = scratch("cwd-file-home");
        let home = base.join("home-is-a-file");
        fs::write(&home, b"not a directory").unwrap();
        let error = managed_cli_working_dir_from(Some(home), &[]).unwrap_err();
        assert!(error.contains("not reachable"), "unexpected error: {error}");
        fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn a_home_inside_the_windows_directory_is_rejected() {
        let windir = PathBuf::from("C:\\Windows");
        for home in [
            "C:\\Windows",
            "C:\\Windows\\System32\\config\\systemprofile",
            "C:\\Windows\\",
        ] {
            let error = managed_cli_working_dir_from(
                Some(PathBuf::from(home)),
                std::slice::from_ref(&windir),
            )
            .unwrap_err();
            assert!(
                error.contains("inside the Windows directory"),
                "{home} must be rejected, got: {error}"
            );
        }
    }

    #[test]
    fn a_home_that_merely_shares_a_prefix_with_the_windows_directory_is_allowed() {
        let error = managed_cli_working_dir_from(
            Some(PathBuf::from(r"C:\Windows2\Users\jane")),
            &[PathBuf::from(r"C:\Windows")],
        )
        .unwrap_err();
        assert!(
            !error.contains("inside the Windows directory"),
            "C:\\Windows2 is a normal folder, got: {error}"
        );
    }

    #[test]
    fn a_drive_root_windows_directory_does_not_reject_every_home() {
        let error = managed_cli_working_dir_from(
            Some(PathBuf::from(r"C:\Users\me")),
            &[PathBuf::from("C:\\")],
        )
        .unwrap_err();
        assert!(
            !error.contains("inside the Windows directory"),
            "unexpected rejection: {error}"
        );
    }

    #[test]
    fn forward_slashes_and_case_do_not_hide_the_windows_directory() {
        let error = managed_cli_working_dir_from(
            Some(PathBuf::from("c:/WINDOWS/System32/config/systemprofile")),
            &[PathBuf::from(r"C:\Windows")],
        )
        .unwrap_err();
        assert!(
            error.contains("inside the Windows directory"),
            "got: {error}"
        );
    }

    #[test]
    fn an_extended_length_path_does_not_hide_the_windows_directory() {
        let error = managed_cli_working_dir_from(
            Some(PathBuf::from(
                r"\\?\C:\Windows\System32\config\systemprofile",
            )),
            &[PathBuf::from(r"C:\Windows")],
        )
        .unwrap_err();
        assert!(
            error.contains("inside the Windows directory"),
            "got: {error}"
        );
    }

    #[test]
    fn a_usable_inherited_directory_is_kept() {
        assert_eq!(
            managed_cli_working_dir().expect("the test's own directory is usable"),
            std::env::current_dir().unwrap()
        );
    }

    #[test]
    fn only_a_windows_directory_counts_as_unusable() {
        let windirs = [PathBuf::from(r"C:\Windows")];
        assert!(is_inside_windows_dir(
            std::path::Path::new(r"C:\Windows\System32"),
            &windirs
        ));
        assert!(!is_inside_windows_dir(
            std::path::Path::new(r"D:\projects\llm"),
            &windirs
        ));
        assert!(!is_inside_windows_dir(
            std::path::Path::new("/home/me"),
            &windirs
        ));
    }

    #[test]
    fn a_candidate_that_holds_no_system32_is_not_a_windows_directory() {
        let roots = windows_roots_from(
            vec![
                PathBuf::from(r"C:\Windows"),
                PathBuf::from(r"C:\Users\me"),
                PathBuf::from(r"C:\Windows"),
            ],
            PathBuf::from(r"C:\Windows"),
            |root| root == std::path::Path::new(r"C:\Windows"),
        );
        assert_eq!(roots, vec![PathBuf::from(r"C:\Windows")]);
    }

    #[test]
    fn a_shadowed_windir_does_not_hide_the_real_windows_directory() {
        let roots = windows_roots_from(
            vec![PathBuf::from(r"D:\Windows"), PathBuf::from(r"C:\Users\me")],
            PathBuf::from(r"D:\Windows"),
            |root| root == std::path::Path::new(r"D:\Windows"),
        );
        assert_eq!(roots, vec![PathBuf::from(r"D:\Windows")]);
    }

    #[test]
    fn nothing_that_looks_like_windows_falls_back_to_the_authoritative_value() {
        let roots = windows_roots_from(
            vec![PathBuf::from(r"E:\Windows"), PathBuf::from(r"C:\Users\me")],
            PathBuf::from(r"E:\Windows"),
            |_root| false,
        );
        assert_eq!(
            roots,
            vec![PathBuf::from(r"E:\Windows")],
            "the guard must stay alive, and never on the settable value"
        );
    }

    #[test]
    fn a_configured_command_carries_the_directory_and_the_marker() {
        // Takes the crate-wide env lock: other tests may swap the environment.
        let _env = crate::native_path_policy::PROCESS_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let work_dir = scratch("cwd-command-shape");

        let mut cmd = Command::new("unsloth");
        apply_managed_cli_context_at(&mut cmd, &work_dir).unwrap();
        assert_eq!(cmd.get_current_dir(), Some(work_dir.as_path()));
        assert!(
            cmd.get_envs()
                .any(|(key, value)| key == DESKTOP_MANAGED_ENV && value == Some("1".as_ref())),
            "the desktop marker must be set"
        );
        fs::remove_dir_all(&work_dir).ok();
    }

    #[test]
    fn a_mounted_appimage_is_never_a_managed_child_working_directory() {
        let appdir = PathBuf::from("/tmp/.mount_Unsloth1a2b3c");
        for unusable in ["/tmp/.mount_Unsloth1a2b3c", "/tmp/.mount_Unsloth1a2b3c/usr"] {
            assert!(
                is_unusable_cwd(std::path::Path::new(unusable), &[], Some(&appdir)),
                "{unusable} must be replaced"
            );
        }
        for usable in ["/home/me/projects", "/tmp/.mount_Other/usr"] {
            assert!(
                !is_unusable_cwd(std::path::Path::new(usable), &[], Some(&appdir)),
                "{usable} must be kept"
            );
        }
        assert!(!is_unusable_cwd(
            std::path::Path::new("/tmp/.mount_Unsloth1a2b3c/usr"),
            &[],
            None
        ));
    }

    #[test]
    fn only_the_folders_the_cli_refuses_count_as_unusable() {
        let windirs = [PathBuf::from("C:\\Windows")];
        for unusable in [
            "C:\\Windows\\System32",
            "c:\\windows\\system32\\config\\systemprofile",
            "C:\\Windows\\SysWOW64",
            "\\\\?\\C:\\Windows\\System32",
        ] {
            assert!(
                is_unusable_cwd(std::path::Path::new(unusable), &windirs, None),
                "{unusable} must be replaced"
            );
        }
        for usable in [
            "C:\\Windows\\Temp\\project",
            "C:\\Windows",
            "C:\\Windows2\\System32",
            "C:\\Users\\me\\projects",
        ] {
            assert!(
                !is_unusable_cwd(std::path::Path::new(usable), &windirs, None),
                "{usable} must be kept"
            );
        }
    }

    #[test]
    fn an_extended_unc_path_compares_the_same_in_either_case() {
        assert_eq!(
            normalize_windows_path(std::path::Path::new("\\\\?\\UNC\\server\\profiles\\me")),
            normalize_windows_path(std::path::Path::new("\\\\?\\unc\\server\\profiles\\me"))
        );
        assert!(is_fully_qualified("\\\\?\\unc\\server\\profiles\\me"));
        assert!(is_fully_qualified("\\\\?\\UNC\\server\\profiles\\me"));
    }

    #[test]
    fn a_lost_directory_is_judged_entry_by_entry() {
        let work_dir = PathBuf::from("C:\\Users\\me\\.unsloth");
        let home = Some(std::path::Path::new("C:\\Users\\me"));
        let mixed = |name: &str| match name {
            "PYTHONPATH" => Some("C:\\vendor;plugins".to_string()),
            _ => None,
        };
        assert!(relative_override_pins_from(
            None,
            &work_dir,
            mixed,
            |_| None,
            home,
            MANAGED_CHILD_SCRUBBED_ENV,
            true
        )
        .is_err());
        let qualified = |name: &str| match name {
            "PYTHONPATH" => Some("C:\\vendor;D:\\plugins".to_string()),
            _ => None,
        };
        assert!(relative_override_pins_from(
            None,
            &work_dir,
            qualified,
            |_| None,
            home,
            MANAGED_CHILD_SCRUBBED_ENV,
            true
        )
        .unwrap()
        .is_empty());
    }

    #[test]
    fn a_lost_directory_reads_a_posix_environment_by_posix_rules() {
        let work_dir = PathBuf::from("/home/me/.unsloth");
        let home = Some(std::path::Path::new("/home/me"));
        let posix = |name: &str| match name {
            "XDG_CACHE_HOME" => Some("/var/cache/unsloth".to_string()),
            _ => None,
        };
        assert!(
            relative_override_pins_from(
                None,
                &work_dir,
                posix,
                |_| None,
                home,
                MANAGED_CHILD_SCRUBBED_ENV,
                false
            )
            .unwrap()
            .is_empty()
        );
        let relative = |name: &str| match name {
            "XDG_CACHE_HOME" => Some("cache".to_string()),
            _ => None,
        };
        assert!(relative_override_pins_from(
            None,
            &work_dir,
            relative,
            |_| None,
            home,
            MANAGED_CHILD_SCRUBBED_ENV,
            false
        )
        .is_err());
        let mixed = |name: &str| match name {
            "PYTHONPATH" => Some("/opt/vendor:plugins".to_string()),
            _ => None,
        };
        assert!(relative_override_pins_from(
            None,
            &work_dir,
            mixed,
            |_| None,
            home,
            MANAGED_CHILD_SCRUBBED_ENV,
            false
        )
        .is_err());
        assert!(relative_override_pins_from(
            None,
            &PathBuf::from("C:\\Users\\me\\.unsloth"),
            posix,
            |_| None,
            Some(std::path::Path::new("C:\\Users\\me")),
            MANAGED_CHILD_SCRUBBED_ENV,
            true
        )
        .is_err());
    }

    #[test]
    fn only_the_update_child_carries_the_local_checkout() {
        let cwd = PathBuf::from("C:\\Windows\\System32");
        let work_dir = PathBuf::from("C:\\Users\\me\\.unsloth");
        let home = Some(std::path::Path::new("C:\\Users\\me"));
        let env = |name: &str| match name {
            "STUDIO_LOCAL_REPO" => Some("..\\src\\unsloth".to_string()),
            _ => None,
        };
        let for_update = relative_override_pins_from(
            Some(cwd.clone()),
            &work_dir,
            env,
            |_| None,
            home,
            MANAGED_CHILD_SCRUBBED_ENV,
            true,
        )
        .unwrap();
        assert_eq!(
            for_update,
            vec![("STUDIO_LOCAL_REPO", cwd.join("..\\src\\unsloth"))]
        );
        let for_child = relative_override_pins_from(
            Some(cwd),
            &work_dir,
            env,
            |_| None,
            home,
            &child_skipped_env(),
            true,
        )
        .unwrap();
        assert!(for_child.is_empty(), "a child that ignores it must not pin it");
    }

    #[test]
    fn a_tilde_is_written_out_the_way_expanduser_writes_it() {
        let home = std::path::Path::new("C:\\Users\\me");
        let me = Some("me");
        assert_eq!(expand_windows_user("~", home, me), "C:\\Users\\me");
        assert_eq!(
            expand_windows_user("~\\llama.cpp", home, me),
            "C:\\Users\\me\\llama.cpp"
        );
        assert_eq!(
            expand_windows_user("~/llama.cpp", home, me),
            "C:\\Users\\me/llama.cpp"
        );
        assert_eq!(expand_windows_user("~other\\x", home, me), "C:\\Users\\other\\x");
        let domain = std::path::Path::new("C:\\Users\\me.DOMAIN");
        assert_eq!(expand_windows_user("~me\\x", domain, me), "C:\\Users\\me.DOMAIN\\x");
        assert_eq!(expand_windows_user("~other\\x", domain, me), "~other\\x");
        assert_eq!(expand_windows_user("~other\\x", home, None), "~other\\x");
        for value in ["cache", "C:\\cache", "a~b"] {
            assert_eq!(expand_windows_user(value, home, me), value);
        }
    }

    #[test]
    fn every_spelling_expandvars_takes_is_expanded_here_too() {
        let lookup = |name: &str| match name {
            "LOCALAPPDATA" => Some("C:\\Users\\me\\AppData\\Local".to_string()),
            "CACHE-ROOT" => Some("C:\\right".to_string()),
            "CACHE" => Some("C:\\wrong".to_string()),
            "TWO WORDS" => Some("C:\\spaced".to_string()),
            _ => None,
        };
        for value in [
            "%LOCALAPPDATA%\\hf",
            "$LOCALAPPDATA\\hf",
            "${LOCALAPPDATA}\\hf",
        ] {
            assert_eq!(
                expand_windows_vars(value, &lookup),
                "C:\\Users\\me\\AppData\\Local\\hf",
                "{value} did not expand the way the CLI guard expands it"
            );
        }
        assert_eq!(expand_windows_vars("$CACHE-ROOT\\hf", &lookup), "C:\\right\\hf");
        assert_eq!(expand_windows_vars("%TWO WORDS%\\x", &lookup), "C:\\spaced\\x");
        assert_eq!(expand_windows_vars("$CACHE.d", &lookup), "C:\\wrong.d");
        assert_eq!(expand_windows_vars("100%%", &lookup), "100%");
        assert_eq!(expand_windows_vars("$$HOME", &lookup), "$HOME");
        assert_eq!(
            expand_windows_vars("'%LOCALAPPDATA%'\\hf", &lookup),
            "'%LOCALAPPDATA%'\\hf"
        );
        for value in [
            "%NOT_SET%\\hub",
            "$NOT_SET\\hub",
            "${LOCALAPPDATA\\hf",
            "%LOCALAPPDATA\\hf",
            "a$b",
            "$",
            "50% off",
            "caché\\modèles",
        ] {
            assert_eq!(
                expand_windows_vars(value, &lookup),
                value,
                "{value} was rewritten and should not have been"
            );
        }
    }

    #[test]
    fn the_pythonpath_spellings_that_follow_the_process_are_anchored() {
        let cwd = PathBuf::from("C:\\Windows\\System32");
        let work_dir = PathBuf::from("C:\\Users\\me\\.unsloth");
        let pins = relative_override_pins_from(
            Some(cwd.clone()),
            &work_dir,
            |name| match name {
                "PYTHONPATH" => Some(";~\\plugins;C:\\shared\\lib".to_string()),
                _ => None,
            },
            |_| None,
            Some(std::path::Path::new("C:\\Users\\me")),
            MANAGED_CHILD_SCRUBBED_ENV,
            true,
        )
        .unwrap();
        let expected = format!(
            "{};{};C:\\shared\\lib",
            cwd.to_string_lossy(),
            cwd.join("~\\plugins").to_string_lossy()
        );
        assert_eq!(pins, vec![("PYTHONPATH", PathBuf::from(expected))]);
    }

    #[test]
    fn a_cache_override_is_expanded_before_it_is_judged() {
        let cwd = PathBuf::from("C:\\Windows\\System32");
        let work_dir = PathBuf::from("C:\\Users\\me\\.unsloth");
        let pins = relative_override_pins_from(
            Some(cwd.clone()),
            &work_dir,
            |name| match name {
                "LOCALAPPDATA" => Some("C:\\Users\\me\\AppData\\Local".to_string()),
                "HF_HOME" => Some("%LOCALAPPDATA%\\hf".to_string()),
                "HF_HUB_CACHE" => Some("%NOT_SET%\\hub".to_string()),
                _ => None,
            },
            |_| None,
            Some(std::path::Path::new("C:\\Users\\me")),
            MANAGED_CHILD_SCRUBBED_ENV,
            true,
        )
        .unwrap();
        assert_eq!(
            pins,
            vec![
                (
                    "HF_HOME",
                    PathBuf::from("C:\\Users\\me\\AppData\\Local\\hf")
                ),
                ("HF_HUB_CACHE", cwd.join("%NOT_SET%\\hub")),
            ],
            "an expanded cache override must name one folder for both readers"
        );
    }

    #[test]
    fn an_exemption_only_applies_to_the_variable_that_supports_it() {
        let cwd = PathBuf::from("C:\\Windows\\System32");
        let work_dir = PathBuf::from("C:\\Users\\me\\.unsloth");
        let pins = relative_override_pins_from(
            Some(cwd.clone()),
            &work_dir,
            |name| match name {
                "UNSLOTH_LLAMA_CPP_PATH" => Some("[llama]".to_string()),
                "UNSLOTH_COMPILE_LOCATION" => Some("%data%".to_string()),
                _ => None,
            },
            |_| None,
            Some(std::path::Path::new("C:\\Users\\me")),
            MANAGED_CHILD_SCRUBBED_ENV,
            true,
        )
        .unwrap();
        assert_eq!(
            pins,
            vec![
                ("UNSLOTH_LLAMA_CPP_PATH", cwd.join("[llama]")),
                ("UNSLOTH_COMPILE_LOCATION", cwd.join("%data%")),
            ],
            "a legal directory name was mistaken for JSON or a placeholder"
        );
    }

    #[test]
    fn a_value_the_working_directory_does_not_resolve_is_left_alone() {
        let cwd = PathBuf::from("C:\\Windows\\System32");
        let work_dir = PathBuf::from("C:\\Users\\me\\.unsloth");
        let pins = relative_override_pins_from(
            Some(cwd),
            &work_dir,
            |name| match name {
                "MLX_HOSTFILE" => Some("[{\"ssh\": \"node0\"}]".to_string()),
                "UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH" => Some("1".to_string()),
                _ => None,
            },
            |_| None,
            Some(std::path::Path::new("C:\\Users\\me")),
            MANAGED_CHILD_SCRUBBED_ENV,
            true,
        )
        .unwrap();
        assert!(pins.is_empty(), "a non-path value was rewritten: {pins:?}");
    }

    #[test]
    fn each_entry_of_a_path_list_is_anchored_on_its_own() {
        let cwd = PathBuf::from("C:\\Windows\\System32");
        let work_dir = PathBuf::from("C:\\Users\\me\\.unsloth");
        let pins = relative_override_pins_from(
            Some(cwd.clone()),
            &work_dir,
            |name| match name {
                "UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH" => {
                    Some("trusted;D:\\shared;~\\mine".to_string())
                }
                _ => None,
            },
            |_| None,
            Some(std::path::Path::new("C:\\Users\\me")),
            MANAGED_CHILD_SCRUBBED_ENV,
            true,
        )
        .unwrap();
        let expected = format!(
            "{};D:\\shared;C:\\Users\\me\\mine",
            cwd.join("trusted").to_string_lossy()
        );
        assert_eq!(
            pins,
            vec![(
                "UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH",
                PathBuf::from(expected)
            )],
            "a relative entry must not authorise a different directory after the move"
        );
    }

    #[test]
    fn configuring_a_command_twice_changes_nothing() {
        // Takes the crate-wide env lock: other tests may swap the environment.
        let _env = crate::native_path_policy::PROCESS_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let work_dir = scratch("cwd-command-twice");

        let mut once = Command::new("unsloth");
        apply_managed_cli_context_at(&mut once, &work_dir).unwrap();
        let mut twice = Command::new("unsloth");
        apply_managed_cli_context_at(&mut twice, &work_dir).unwrap();
        apply_managed_cli_context_at(&mut twice, &work_dir).unwrap();

        let envs = |cmd: &Command| {
            let mut pairs: Vec<(String, Option<String>)> = cmd
                .get_envs()
                .map(|(key, value)| {
                    (
                        key.to_string_lossy().into_owned(),
                        value.map(|v| v.to_string_lossy().into_owned()),
                    )
                })
                .collect();
            pairs.sort();
            pairs.dedup();
            pairs
        };
        assert_eq!(envs(&once), envs(&twice));
        assert_eq!(twice.get_current_dir(), Some(work_dir.as_path()));
        fs::remove_dir_all(&work_dir).ok();
    }

    #[test]
    fn pinning_an_already_pinned_value_is_a_no_op() {
        let cwd = PathBuf::from("C:\\Windows\\System32");
        let work_dir = PathBuf::from("C:\\Users\\me\\.unsloth");
        let absolute = |_: &str| Some(PathBuf::from("D:\\work\\datasets"));

        let mut values = vec![
            ("HF_HOME", "cache".to_string()),
            ("HF_DATASETS_CACHE", "D:datasets".to_string()),
        ];
        for round in 0..3 {
            let pins = relative_override_pins_from(
                Some(cwd.clone()),
                &work_dir,
                |name| {
                    values
                        .iter()
                        .find(|(key, _)| *key == name)
                        .map(|(_, value)| value.clone())
                },
                absolute,
                Some(std::path::Path::new("C:\\Users\\me")),
                MANAGED_CHILD_SCRUBBED_ENV,
                true,
            )
            .unwrap();
            if round == 0 {
                assert_eq!(pins.len(), 2, "the first pass rewrites both values");
            } else {
                assert!(pins.is_empty(), "pass {round} rewrote an anchored value");
            }
            for (name, pinned) in pins {
                let slot = values.iter_mut().find(|(key, _)| *key == name).unwrap();
                slot.1 = pinned.to_string_lossy().into_owned();
            }
        }
        assert_eq!(values[0].1, cwd.join("cache").to_string_lossy());
        assert_eq!(values[1].1, "D:\\work\\datasets");
        assert!(is_fully_qualified(&values[0].1) && is_fully_qualified(&values[1].1));
    }

    #[test]
    fn a_configured_tokio_command_carries_the_directory_and_the_marker() {
        // Takes the crate-wide env lock: other tests may swap the environment.
        let _env = crate::native_path_policy::PROCESS_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let expected = managed_cli_working_dir().expect("home must resolve");
        let mut tokio_cmd = tokio::process::Command::new("unsloth");
        apply_managed_cli_context_tokio(&mut tokio_cmd).expect("context must apply");
        let configured = tokio_cmd.as_std().get_current_dir();
        match std::env::current_dir() {
            Ok(cwd) if cwd == expected => assert_eq!(configured, None),
            _ => assert_eq!(configured, Some(expected.as_path())),
        }
        assert!(
            tokio_cmd
                .as_std()
                .get_envs()
                .any(|(key, value)| key == DESKTOP_MANAGED_ENV && value == Some("1".as_ref())),
            "the desktop marker must be set on tokio commands too"
        );
    }

    // The Python side must read the same name; a rename degrades silently to argv matching.
    #[test]
    fn the_marker_name_matches_the_python_guard() {
        let guard = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../unsloth_cli/_system_dir_guard.py");
        let source = fs::read_to_string(&guard).expect("the Python guard must be readable");
        assert!(
            source.contains(&format!("DESKTOP_MANAGED_ENV = \"{DESKTOP_MANAGED_ENV}\"")),
            "{} must define the same marker name",
            guard.display()
        );
    }

    #[test]
    fn configuring_a_child_leaves_the_parent_directory_alone() {
        let before = std::env::current_dir().unwrap();
        let work_dir = scratch("cwd-parent-untouched");
        let mut cmd = Command::new("unsloth");
        apply_managed_cli_context_at(&mut cmd, &work_dir).unwrap();
        assert_eq!(std::env::current_dir().unwrap(), before);
        fs::remove_dir_all(&work_dir).ok();
    }

    #[test]
    fn backend_args_are_unchanged_by_the_working_directory_fix() {
        assert_eq!(
            backend_args(8888),
            vec!["studio", "--api-only", "-H", "127.0.0.1", "-p", "8888"]
        );
    }

    #[cfg(windows)]
    #[test]
    fn a_spawned_child_runs_from_the_resolved_directory_on_windows() {
        let expected = scratch("cwd-spawned-child-win");

        let mut cmd = Command::new("cmd.exe");
        cmd.args(["/C", "cd"]).stdout(Stdio::piped());
        apply_managed_cli_context_at(&mut cmd, &expected).unwrap();
        let output = cmd.output().expect("spawn test child");
        let reported = String::from_utf8_lossy(&output.stdout).trim().to_string();

        assert_eq!(
            normalize_windows_path(std::path::Path::new(&reported)),
            normalize_windows_path(&expected)
        );
        fs::remove_dir_all(&expected).ok();
    }

    #[cfg(unix)]
    #[test]
    fn a_spawned_child_runs_from_the_resolved_directory() {
        let expected = scratch("cwd-spawned-child");

        let mut cmd = Command::new("/bin/sh");
        cmd.args(["-c", "pwd -P"]).stdout(Stdio::piped());
        apply_managed_cli_context_at(&mut cmd, &expected).unwrap();
        let mut wrap = CommandWrap::from(cmd);
        wrap.wrap(ProcessGroup::leader());
        let mut child = wrap.spawn().expect("spawn test child");
        let mut out = String::new();
        std::io::Read::read_to_string(child.stdout().as_mut().unwrap(), &mut out).unwrap();
        let _ = child.wait();

        assert_eq!(
            std::fs::canonicalize(out.trim()).unwrap(),
            std::fs::canonicalize(&expected).unwrap()
        );
    }
}

// Real processes: stdout can EOF before the exit is visible to try_wait.
#[cfg(test)]
#[cfg(unix)]
mod exit_status_after_stdout_closed_tests {
    use super::*;

    fn spawn(args: &[&str]) -> Box<dyn ChildWrapper + Send> {
        let mut cmd = Command::new(args[0]);
        cmd.args(&args[1..])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut wrap = CommandWrap::from(cmd);
        wrap.wrap(ProcessGroup::leader());
        wrap.spawn().expect("spawn test child")
    }

    #[test]
    fn reports_the_status_of_a_child_that_has_already_exited() {
        let mut child = spawn(&["/bin/sh", "-c", "exit 3"]);
        let status = exit_status_after_stdout_closed(&mut child)
            .expect("a child that exited must be reported, not read as alive");
        assert!(status.contains('3'), "expected the real exit code in {status:?}");
    }

    #[test]
    fn a_child_that_is_still_running_is_not_reported_as_dead() {
        let mut child = spawn(&["/bin/sh", "-c", "exec sleep 30"]);
        let status = exit_status_after_stdout_closed(&mut child);
        let _ = child.start_kill();
        assert!(
            status.is_none(),
            "a live child must not be reported as exited (got {status:?})"
        );
    }

    #[test]
    fn wins_the_race_against_a_child_exiting_mid_check() {
        for delay_ms in [0, 5, 25, 120, 400] {
            let mut child = spawn(&[
                "/bin/sh",
                "-c",
                &format!("sleep {}; exit 7", delay_ms as f64 / 1000.0),
            ]);
            assert!(
                exit_status_after_stdout_closed(&mut child).is_some(),
                "child exiting after {delay_ms}ms was read as still alive"
            );
        }
    }
}

#[cfg(test)]
mod owned_backend_liveness_tests {
    use super::*;

    #[cfg(unix)]
    fn spawn_owned(args: &[&str]) -> Box<dyn ChildWrapper + Send> {
        let mut cmd = Command::new(args[0]);
        cmd.args(&args[1..])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut wrap = CommandWrap::from(cmd);
        wrap.wrap(ProcessGroup::leader());
        wrap.spawn().expect("spawn test child")
    }

    #[cfg(windows)]
    fn spawn_owned(args: &[&str]) -> Box<dyn ChildWrapper + Send> {
        let mut cmd = Command::new(args[0]);
        cmd.args(&args[1..])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        CommandWrap::from(cmd).spawn().expect("spawn test child")
    }

    #[cfg(unix)]
    const LIVE_CHILD: [&str; 3] = ["/bin/sh", "-c", "exec sleep 30"];
    #[cfg(unix)]
    const DEAD_CHILD: [&str; 3] = ["/bin/sh", "-c", "exit 0"];
    #[cfg(windows)]
    const LIVE_CHILD: [&str; 3] = ["cmd.exe", "/C", "ping -n 30 127.0.0.1"];
    #[cfg(windows)]
    const DEAD_CHILD: [&str; 3] = ["cmd.exe", "/C", "exit 0"];

    fn state_owning(child: Box<dyn ChildWrapper + Send>, port: u16) -> BackendState {
        let state = new_backend_state();
        {
            let mut proc = state.lock().unwrap();
            let pid = 0;
            proc.owned = Some(OwnedBackendHandle::spawned(child, None, pid, 1));
            if let Some(handle) = proc.owned.as_mut() {
                handle.set_reported_port(port);
            }
            proc.port = Some(port);
        }
        state
    }

    #[test]
    fn a_child_that_is_still_running_is_ours() {
        let state = state_owning(spawn_owned(&LIVE_CHILD), 8765);
        assert!(owned_backend_on_port_is_running(&state, 8765));
        let mut proc = state.lock().unwrap();
        if let Some(handle) = proc.owned.as_mut() {
            if let Some(child) = handle.spawned_child_mut() {
                let _ = child.start_kill();
            }
        }
    }

    #[test]
    fn a_child_that_has_exited_leaves_the_port_to_strangers() {
        let mut child = spawn_owned(&DEAD_CHILD);
        let _ = child.wait();
        let state = state_owning(child, 8765);
        assert!(
            !owned_backend_on_port_is_running(&state, 8765),
            "an exited child still counted as the managed backend, so a foreign service on \
             its port would be reported to the user as Unsloth still running"
        );
    }

    #[test]
    fn a_handle_for_another_port_is_not_this_port() {
        let state = state_owning(spawn_owned(&LIVE_CHILD), 8765);
        assert!(!owned_backend_on_port_is_running(&state, 8766));
        let mut proc = state.lock().unwrap();
        if let Some(handle) = proc.owned.as_mut() {
            if let Some(child) = handle.spawned_child_mut() {
                let _ = child.start_kill();
            }
        }
    }

    #[test]
    fn no_handle_at_all_is_not_a_managed_backend() {
        let state = new_backend_state();
        assert!(!owned_backend_on_port_is_running(&state, 8765));
        assert!(!owned_backend_could_bind_port(&state, 8765));
    }

    #[test]
    fn a_child_that_has_not_reported_a_port_could_still_bind_the_one_asked_about() {
        let state = new_backend_state();
        {
            let mut proc = state.lock().unwrap();
            proc.owned = Some(OwnedBackendHandle::spawned(
                spawn_owned(&LIVE_CHILD),
                None,
                0,
                1,
            ));
        }
        assert!(
            !owned_backend_on_port_is_running(&state, 8765),
            "presence is unchanged: a handle with no port names no port"
        );
        assert!(
            owned_backend_could_bind_port(&state, 8765),
            "a live backend of ours that has not bound a port yet was ruled out as the owner \
             of the port it is starting on, which is the slow start #10520 exists to survive"
        );
        let mut proc = state.lock().unwrap();
        if let Some(handle) = proc.owned.as_mut() {
            if let Some(child) = handle.spawned_child_mut() {
                let _ = child.start_kill();
            }
        }
    }

    #[test]
    fn a_child_that_died_before_reporting_a_port_cannot_bind_anything() {
        let mut child = spawn_owned(&DEAD_CHILD);
        let _ = child.wait();
        let state = new_backend_state();
        {
            let mut proc = state.lock().unwrap();
            proc.owned = Some(OwnedBackendHandle::spawned(child, None, 0, 1));
        }
        assert!(
            !owned_backend_could_bind_port(&state, 8765),
            "an exited child kept the fast path switched off for every port"
        );
    }

    #[test]
    fn a_handle_that_reported_another_port_does_not_cover_this_one() {
        let state = state_owning(spawn_owned(&LIVE_CHILD), 8765);
        assert!(
            !owned_backend_could_bind_port(&state, 8766),
            "a backend that has told us its port was still treated as a candidate for others"
        );
        assert!(owned_backend_could_bind_port(&state, 8765));
        let mut proc = state.lock().unwrap();
        if let Some(handle) = proc.owned.as_mut() {
            if let Some(child) = handle.spawned_child_mut() {
                let _ = child.start_kill();
            }
        }
    }

    #[test]
    fn an_adopted_pid_that_is_gone_is_not_running() {
        assert!(backend_pid_is_running(std::process::id()));
        let mut child = spawn_owned(&DEAD_CHILD);
        let pid = child.id();
        let _ = child.wait();
        assert!(
            !backend_pid_is_running(pid),
            "an adopted backend that has exited still read as running"
        );
    }
}
