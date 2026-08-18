use crate::diagnostics::{self, BackendLog, DiagnosticsState};
use log::{error, info, warn};
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

// An AppImage can be launched from an activated Python environment. Keep the
// host library path the thin bundle needs, but do not let PYTHONHOME/PYTHONPATH
// shadow the managed Studio environment.
#[cfg(target_os = "linux")]
pub(crate) fn scrub_appimage_python_env(cmd: &mut Command) {
    if std::env::var_os("APPIMAGE").is_some() {
        cmd.env_remove("PYTHONHOME");
        cmd.env_remove("PYTHONPATH");
    }
}

#[cfg(target_os = "linux")]
pub(crate) fn scrub_appimage_python_env_tokio(cmd: &mut tokio::process::Command) {
    if std::env::var_os("APPIMAGE").is_some() {
        cmd.env_remove("PYTHONHOME");
        cmd.env_remove("PYTHONPATH");
    }
}

#[cfg(windows)]
const STUDIO_MANAGED_RUNTIME_MUTEX_PREFIX: &str = "Global\\UnslothStudioManagedEnvironment-";

#[cfg(windows)]
pub(crate) const STUDIO_RUNTIME_GATE_HANDOFF_ENV: &str = "_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF";

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
            "Could not create the Studio runtime lock: {}",
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
            Err(
                "Unsloth installation is modifying the managed environment. Wait for it to finish, then start the backend again."
                    .to_string(),
            )
        }
        _ => {
            let error = std::io::Error::last_os_error();
            unsafe {
                let _ = windows_sys::Win32::Foundation::CloseHandle(handle);
            }
            Err(format!(
                "Could not acquire the Studio runtime lock: {error}"
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
            "Could not open the Windows user token for the Studio runtime lock: {}",
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
                "Could not size the Windows user SID for the Studio runtime lock: {}",
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
                "Could not read the Windows user SID for the Studio runtime lock: {}",
                std::io::Error::last_os_error()
            ));
        }

        let token_user = unsafe { &*(buffer.as_ptr().cast::<TOKEN_USER>()) };
        let sid = token_user.User.Sid;
        if sid.is_null() || unsafe { IsValidSid(sid) } == 0 {
            return Err("Windows returned an invalid user SID for the Studio runtime lock".into());
        }

        let authority_ptr = unsafe { GetSidIdentifierAuthority(sid) };
        let count_ptr = unsafe { GetSidSubAuthorityCount(sid) };
        if authority_ptr.is_null() || count_ptr.is_null() {
            return Err(
                "Could not inspect the Windows user SID for the Studio runtime lock".into(),
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
                    "Could not inspect the Windows user SID for the Studio runtime lock".into(),
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

/// Serialize creation of managed-environment children with install/repair.
///
/// The guard ends when the sync operation returns. The installer takes the same
/// mutex then scans for managed processes, so holding it through child creation
/// closes the race without carrying a thread-owned Win32 mutex across an await.
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
    #[cfg(not(windows))]
    operation()
}

#[cfg(windows)]
fn normalized_existing_windows_path(path: &std::path::Path) -> Result<String, String> {
    let resolved = std::fs::canonicalize(path)
        .map_err(|error| format!("Could not resolve managed Studio path {:?}: {error}", path))?;
    Ok(resolved
        .to_string_lossy()
        .trim_end_matches(['\\', '/'])
        .replace('/', "\\"))
}

#[cfg(windows)]
fn windows_ordinal_ignore_case_equal(left: &[u16], right: &[u16]) -> Result<bool, String> {
    use windows_sys::Win32::Globalization::{CompareStringOrdinal, CSTR_EQUAL};

    let left_length = i32::try_from(left.len())
        .map_err(|_| "Normalized Studio path exceeds Win32 comparison limits".to_string())?;
    let right_length = i32::try_from(right.len())
        .map_err(|_| "Normalized Studio path exceeds Win32 comparison limits".to_string())?;
    let comparison = unsafe {
        CompareStringOrdinal(left.as_ptr(), left_length, right.as_ptr(), right_length, 1)
    };
    if comparison == 0 {
        return Err(format!(
            "Could not compare normalized Studio paths: {}",
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

/// Reject an update when a process image runs from the target venv or the exact
/// supported Studio shim.
///
/// Callers must hold the runtime launch mutex across the whole mutation: this
/// scan finds older consumers, and the gate blocks new launches after it.
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
                    "Could not determine the managed Studio environment for {:?}",
                    managed_binary
                )
            })?;
        let studio_home = venv.parent().ok_or_else(|| {
            format!(
                "Could not determine the managed Studio root for {:?}",
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
                "Could not inspect running processes before Studio update: {}",
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
                    "Could not enumerate running processes before Studio update: {}",
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
                                "The managed Studio environment is in use by {} (PID {}). Stop that process, then retry the update.",
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
                        "Could not finish enumerating running processes before Studio update: {}",
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

    // Since issue #8490 the long-lived Studio image is Scripts\python.exe, not
    // Scripts\unsloth.exe. ensure_managed_environment_is_idle matches by venv
    // root, so both must still register as "the environment is in use" -- a
    // miss here would let an update mutate a venv somebody is running.
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

        // A sibling directory sharing the root's prefix must still be outside.
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
        assert!(error.contains("managed Studio environment is in use"));
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
    /// Set by the start watchdog, under this mutex, once it has committed to
    /// emitting server-start-timeout for the current generation. Port
    /// validation refuses to claim afterwards, so the window never receives
    /// server-port behind an error it has no handler to clear.
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

/// Windows `CREATE_NO_WINDOW` flag — suppresses console windows for child processes.
#[cfg(windows)]
pub(crate) const CREATE_NO_WINDOW: u32 = 0x08000000;

/// Force-kill a Windows process tree via hidden `taskkill /T /F`, falling
/// back to `child.kill()` if taskkill itself fails.  Reaps the child afterward.
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

/// Whether a Windows venv's site-packages still holds something the CLI trampoline
/// could import.
///
/// The dist-info is accepted alongside the package directory, and for the same reason
/// `_managed_cli_site_packages_layout` in unsloth_cli/commands/studio.py accepts it: a
/// PEP 660 editable install of the checkout leaves a .pth and a `unsloth-*.dist-info`
/// here and no `unsloth_cli/` at all. Ranking that below an empty new layout would send
/// every capability probe at the interpreter with nothing to import.
///
/// It cannot prove the package imports, and does not try to. This runs on the launch
/// path, so it stays filesystem-only; the two Python-side probes that DO run an import
/// are the ones allowed to spawn an interpreter.
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

/// Returns the path to the unsloth binary inside the managed venv, if it exists.
/// Checks the new layout (~/.unsloth/studio/unsloth_studio/) first,
/// then falls back to the old layout (~/.unsloth/studio/.venv/) for compat.
fn find_unsloth_binary_in_studio_dir(studio: &std::path::Path) -> Option<std::path::PathBuf> {
    // New layout (upstream scripts >= March 2026)
    let new_base = studio.join("unsloth_studio");
    // Old layout (bundled scripts, older upstream)
    let old_base = studio.join(".venv");

    let bases = [new_base, old_base];

    // Three passes rather than one, because a migration interrupted by an open
    // handle can leave HALF of either layout behind (install.ps1 says so where it
    // renames the tree), and layout order alone then picks the half that cannot
    // run. Preference is by usefulness first, layout second:
    //
    //   1. a launcher with the interpreter beside it -- a complete environment;
    //   2. Windows only, an interpreter with no launcher -- quarantine takes the
    //      unsigned stub and leaves a working install, and nothing executes the
    //      stub any more, so its absence no longer means "not installed". Within
    //      this pass, a layout that still holds the CLI package outranks one
    //      that holds only an interpreter, whichever layout it is;
    //   3. a launcher with no interpreter -- useless to every caller here, but it
    //      is what this function answered before, and its error message names the
    //      missing interpreter, which is more use than "not installed".
    //
    // The returned path is the canonical handle whether or not the file exists:
    // every Windows caller reaches the CLI through its parent directory.
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

    // Pass 2 twice over, package first. When an interrupted migration leaves an
    // interpreter in BOTH layouts, layout order alone would hand back the new
    // one even if its site-packages is empty and the old one still carries the
    // package the trampoline imports, and every capability probe would then
    // report an install that cannot start. A directory test, not an import
    // probe: this is called on the launch path and from the capability checks,
    // so it must stay a stat rather than an interpreter spawn. It cannot prove
    // the package is importable, only that one candidate has something to
    // import and the other has nothing, which is the whole difference here.
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

/// The Windows console script is a generated, unsigned PE wrapper. Application
/// Control (AppLocker, WDAC, Smart App Control) denies it while the managed
/// python.exe beside it - a copy of the signed CPython binary - still runs, so
/// every managed CLI invocation goes through the interpreter instead.
///
/// The leading sys.path edit is what `-I` used to buy, without the rest of it.
/// `python -c` puts the working directory on sys.path[0] and the console script
/// never does, so a stray unsloth_cli beside the caller would shadow the managed
/// package; stripping that one entry closes it and leaves alone everything the
/// console script honours. `-I` implies `-E`, which discarded PYTHONPATH,
/// PYTHONWARNINGS, PYTHONHASHSEED, PYTHONPROFILEIMPORTTIME and user
/// site-packages, an observable difference on machines with no policy at all.
///
/// The safe_path guard is why the comprehension is not simply `x not in (...)`.
/// Under -P or PYTHONSAFEPATH there is no implicit `-c` entry to remove, so
/// whatever sits at sys.path[0] is an explicit PYTHONPATH entry the console
/// script would have honoured; a PYTHONPATH that starts at the working
/// directory was measured selecting a different package before and after.
/// getattr, not sys.flags.safe_path: the attribute is 3.11+ and this repo
/// supports 3.9.
///
/// -X utf8 is the one deliberate divergence from the console script, and it
/// predates this work: the shipped updater already spelled it this way, and
/// mangled setup output on a non-UTF-8 console code page is what it exists to
/// prevent (see tests/python/test_windows_setup_output_encoding.py). Under an
/// explicit PYTHONUTF8=0 the child therefore runs in UTF-8 mode where the stub
/// would not have.
///
/// sys.argv[0] is assigned before the import because unsloth_cli decides at
/// import time whether it is the console script, which gates the Windows UTF-8
/// stream reconfigure and the -np<N> argv rewrite. It also sets Typer's
/// prog_name to "unsloth"; the stub prints "unsloth.exe", so usage text reads
/// slightly cleaner here rather than matching byte for byte.
#[cfg(windows)]
pub(crate) const WINDOWS_CLI_ENTRYPOINT: &str =
    "import sys, os; sys.path[:1] = [x for x in sys.path[:1] if getattr(sys.flags, 'safe_path', False) or x not in ('', os.getcwd())]; sys.argv[0] = 'unsloth'; from unsloth_cli import app; sys.exit(app())";

/// The program and argument vector that run the managed CLI without executing
/// `bin` itself. On non-Windows platforms `bin` is a plain script with a
/// shebang and stays the program.
#[derive(Debug)]
pub(crate) struct ManagedCliInvocation {
    pub program: std::path::PathBuf,
    pub args: Vec<std::ffi::OsString>,
}

impl ManagedCliInvocation {
    /// The single place an invocation becomes a process. Callers that need the
    /// resolved program before spawning, to log what they are about to start,
    /// resolve first and come back through here rather than rebuilding argv.
    pub(crate) fn to_command(&self) -> Command {
        let mut cmd = Command::new(&self.program);
        cmd.args(&self.args);
        cmd
    }
}

/// Resolve how to run `bin <args>` for the managed install. Fails closed on
/// Windows when the interpreter is missing rather than falling back to the
/// stub, which is exactly what a policy-blocked machine cannot run.
pub(crate) fn resolve_managed_cli_invocation(
    bin: &std::path::Path,
    args: &[&str],
) -> Result<ManagedCliInvocation, String> {
    resolve_managed_cli_invocation_with(bin, args, Isolation::Inherit)
}

/// Whether a managed invocation runs isolated from the ambient Python environment.
///
/// Everything the user could equally have typed themselves inherits, because the
/// console script does and the swap has to be invisible. The desktop updater does
/// not: it shipped with `-I` before any of this, nobody types it, and it rewrites
/// the very environment it runs in, so a `pip install --user unsloth_cli` deciding
/// which package gets updated is a real hazard rather than a parity question.
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
        // No -I, for the reason on WINDOWS_CLI_ENTRYPOINT. -X utf8 stays, so this
        // process writes UTF-8 into read_lossy_lines whatever the locale, and a
        // caller's PYTHONIOENCODING overrides it exactly as it overrides the
        // console script's.
        // -X utf8 before -I: -I implies -E, which would discard PYTHONUTF8, and the
        // flag form survives it.
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
        // Isolation is a Windows-only concept: POSIX executes the console script
        // itself, so there is no interpreter command line to isolate.
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
    // PYTHONHOME and PYTHONPATH are deliberately left alone. Removing them was
    // belt and braces under -I, which dropped them anyway. Without -I it would
    // bite: the console script honours both, so scrubbing them here would make
    // the swap observable.
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

/// The profile a managed install may live under, or why it cannot. One policy for
/// both callers: a home reported available but then rejected by the resolver sends
/// a SYSTEM account to an install flow that cannot start.
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

    // The installer is the one caller that may build the profile as it goes, the
    // way it did before it shared this resolver. For everything else a home that
    // is not there yet is an unmounted roaming profile, and creating it would
    // leave an empty folder shadowing the real one when it arrives.
    if require_existing && !home.is_dir() {
        return Err(format!(
            "Home directory {} is not reachable",
            home.display()
        ));
    }
    Ok(home)
}

/// Marker the desktop sets on every CLI child it owns. The Python CLI reads it to
/// tell a desktop-managed launch from a user typing the same command in a shell.
pub(crate) const DESKTOP_MANAGED_ENV: &str = "UNSLOTH_DESKTOP_MANAGED";

/// Windows registers "run at login" as an HKCU Run value, which carries no working
/// directory, so the app starts in C:\Windows\system32 and every child inherits it.
/// The CLI refuses to run there, so pick the directory explicitly.
pub(crate) fn managed_cli_working_dir_from(
    home: Option<std::path::PathBuf>,
    windirs: &[std::path::PathBuf],
) -> Result<std::path::PathBuf, String> {
    working_dir_under(home, windirs, true)
}

/// The same directory for the installer, which may create the profile it runs
/// under: install.ps1 and install.sh detect a SYSTEM profile themselves, and
/// before they shared this resolver a home that did not exist yet was simply
/// created along with ~/.unsloth.
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

    // Where the installer already runs, so ~/.unsloth stays the one working root.
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

/// Directory a desktop-spawned `unsloth` CLI child should run from: the inherited
/// one, unless unusable (on Windows, a system folder), so cwd-relative defaults
/// keep resolving where they did and only the login start moves. Resolved per call
/// so a late-mounting profile recovers, and never falls back to a temp dir.
pub(crate) fn managed_cli_working_dir() -> Result<std::path::PathBuf, String> {
    let windirs = windows_roots();
    if let Ok(cwd) = std::env::current_dir() {
        if !is_unusable_cwd(&cwd, &windirs) {
            return Ok(cwd);
        }
    }
    managed_cli_working_dir_from(dirs::home_dir(), &windirs)
}

/// The directories the CLI guard refuses to run from, and only those. The rest of
/// the Windows tree still disqualifies a *home*, but a child already running from
/// C:\Windows\Temp keeps doing so: the guard allowed that before this change.
fn is_unusable_cwd(path: &std::path::Path, windirs: &[std::path::PathBuf]) -> bool {
    windirs.iter().any(|windir| {
        ["System32", "SysWOW64"]
            .iter()
            .any(|name| is_inside_windows_dir(path, &[windir.join(name)]))
    })
}

/// Every real Windows directory, empty off Windows. Candidates are checked, not
/// trusted: a settable variable could aim the check somewhere harmless, and a
/// WINDIR aimed at the profile would reject it. So a root must hold System32.
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

/// Path overrides a relative value makes cwd-dependent, so moving the child
/// without rewriting them would point them somewhere else. Search lists such as
/// PATH are absent: they are not single paths. Mirrors `_RELATIVE_PATH_ENV` in
/// unsloth_cli/_system_dir_guard.py, held identical by a parity test.
pub(crate) const RELATIVE_PATH_ENV: &[&str] = &[
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
    "LLAMA_SERVER_PATH",
    "WHISPER_SERVER_PATH",
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
    "UNSLOTH_DIFFUSION_COMPILE_CACHE_DIR",
    "UNSLOTH_DIFFUSION_COND_CACHE_DIR",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_XET_CACHE",
    "HF_DATASETS_CACHE",
    "HF_ASSETS_CACHE",
    // huggingface_hub resolves the credential file from here; a relative value
    // would follow the child and silently lose access to gated repos.
    "HF_TOKEN_PATH",
    // uv reads this as written and Studio treats a non-blank value as
    // authoritative, so an update would install from a different cache.
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

/// The same, for values holding several separated directories: one relative
/// entry changes what the whole list allows or searches. PYTHONPATH is here
/// because a relative entry in it is resolved at import time, so a moved child
/// would let whatever sits in the new directory shadow a managed import. PATH is
/// deliberately absent: it is mostly other people's absolute entries, and
/// refusing a launch over one unresolvable entry there would cost more than it
/// protects.
pub(crate) const PATH_LIST_ENV: &[&str] = &[
    "UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH",
    "CUDA_RUNTIME_DLL_DIR",
    "PYTHONPATH",
];

/// Whether a value names one directory whatever the process does next.
///
/// Windows rules on every platform: a Windows value is what reaches this code,
/// and hard-coding them keeps the check testable from Linux CI. Matches
/// `_is_fully_qualified` in the CLI guard, "C:sub" and "\\cache" included, both
/// of which name a directory only in combination with process state.
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
    // Spelled out rather than deferred to Path::is_absolute, so the answer is
    // the same on the Linux runner that tests it as on the Windows machine that
    // runs it, and the same as _is_fully_qualified in the CLI guard.
    matches!(value.as_bytes(), [drive, b':', sep, ..]
        if drive.is_ascii_alphabetic() && (*sep == b'\\' || *sep == b'/'))
}

/// `$NAME` and `${NAME}` against the process environment, as posixpath.expandvars
/// reads them: an unset name and a `$` that starts no name are left exactly as
/// written, and `%NAME%` is an ordinary filename character off Windows.
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

/// Whether the value names the same place from any working directory.
///
/// Windows rules on Windows, the native ones off it. The lost-directory
/// fallback below is the one branch a Linux or macOS process reaches, and what
/// it reads there is a POSIX environment: `/var/cache/unsloth` is no more
/// cwd-dependent than `C:\cache`, and judging it by the Windows rules refused
/// every managed spawn over a setting that was never relative. A Windows-shaped
/// value stays Windows-judged, so "/cache" is still the root of whichever drive
/// the process is on.
fn is_cwd_independent(value: &str, windows: bool) -> bool {
    if windows {
        is_fully_qualified(value)
    } else {
        // The whole of absoluteness off Windows.
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

/// Names whose readers disagree about %VAR%: huggingface_hub expands HF_HOME
/// (and the XDG_CACHE_HOME it defaults from), HF_HUB_CACHE and HF_ASSETS_CACHE,
/// and Studio expands SENTENCE_TRANSFORMERS_HOME, but Studio's own
/// hf_cache_settings does not, so it reads %LOCALAPPDATA%\hf as a relative
/// folder. Expanding before deciding settles it: both readers then see one
/// absolute path. Scoped to these names because a directory really called
/// "%data%" is legal and every other name is read as one.
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

/// %NAME%, $NAME and ${NAME} against the process environment.
///
/// Mirrors ntpath.expandvars, which is what the CLI guard uses, down to its
/// pattern: `'[^']*'?|%(%|[^%]*%?)|\$(\$|[-\w]+|\{[^}]*\}?)`. The details matter
/// because the two layers have to name the same folder: a single-quoted run is
/// copied through unexpanded, `%%` and `$$` stand for one character, a `$` name
/// may contain a hyphen, and anything unterminated is left exactly as written,
/// as is a name this machine does not set.
fn expand_windows_vars(value: &str, lookup: &impl Fn(&str) -> Option<String>) -> String {
    let bytes = value.as_bytes();
    let mut out = String::with_capacity(value.len());
    let mut index = 0;
    while index < bytes.len() {
        // Only taken for the three ASCII markers below, where index + 1 is
        // always a character boundary.
        match bytes[index] {
            b'\'' => {
                // A quoted run is copied through, terminator included when it
                // has one, so a reference inside it is not expanded.
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

/// Whether the working directory is what resolves this variable's value.
///
/// The twin of `_names_a_path` in `unsloth_cli/_system_dir_guard.py`. Each
/// exemption is scoped to the variables whose reader proves it, because the
/// syntax is only special there: a directory really called "[llama]" is legal on
/// Windows, and UNSLOTH_LLAMA_CPP_PATH is read as one.
/// `value` expanded once, or None if one pass does not settle it.
///
/// One pass is what every reader does, so one pass is what this does. The result
/// is only usable if expanding it again would change nothing, because the reader
/// expands whatever gets written back: a nested %LOCALAPPDATA% that itself holds
/// %USERPROFILE%, an escaped %%NAME%%, or a self-reference would be expanded a
/// second time and read as a folder with another drive in the middle of it. The
/// twin of `_expand_settled` in the CLI guard.
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

/// Names every managed spawn removes before starting the child: Tauri uses the
/// legacy Studio root whatever the environment says. Resolving one can only
/// invent a failure for a value the child is never going to see.
const MANAGED_CHILD_SCRUBBED_ENV: &[&str] = &["UNSLOTH_STUDIO_HOME", "STUDIO_HOME"];

/// Read only by the update and installer path (install_python_stack.py), so a
/// stale value must not be able to fail a probe, a backend start or an auth
/// provision that would never have looked at it.
const UPDATE_ONLY_ENV: &[&str] = &["STUDIO_LOCAL_REPO"];

/// `~` and `~name`, resolved the way ntpath.expanduser resolves them.
///
/// Written out rather than skipped, because only some readers of these names
/// expand it themselves: llama_cpp.py hands UNSLOTH_LLAMA_CPP_PATH straight to
/// Path(), so a moved child would look for a folder called "~" beside its new
/// working directory.
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
        // ~someone-else is the sibling of this profile, but only where ntpath
        // agrees: it declines to guess unless this profile is named after the
        // current user, since C:\Users\alice.DOMAIN is not alice's sibling.
        // Split on the string, not with Path::parent: these are Windows paths
        // whichever platform the code is running on.
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

fn relative_override_pins_from(
    cwd: Option<std::path::PathBuf>,
    work_dir: &std::path::Path,
    lookup: impl Fn(&str) -> Option<String>,
    absolute: impl Fn(&str) -> Option<std::path::PathBuf>,
    home: Option<&std::path::Path>,
    skipped: &[&str],
    windows: bool,
) -> Result<Vec<(&'static str, std::path::PathBuf)>, String> {
    // Written out, so the reader that expands %VAR% and the reader that does not
    // land in the same folder. Only for the names that have both.
    let username = lookup("USERNAME");
    // ntpath.expanduser answers USERPROFILE, and the CLI guard uses it, so the
    // tilde has to resolve to the same folder here. dirs::home_dir() reads the
    // known folder instead, which a portable or overridden environment moves.
    let tilde_home = lookup("USERPROFILE")
        .map(std::path::PathBuf::from)
        .or_else(|| home.map(|home| home.to_path_buf()));
    let home = tilde_home.as_deref();
    // One pass does not settle every value. What the reader sees is one pass: if
    // that names a folder on its own the value is safe to leave alone, and if it
    // does not, the folder it names depends on where the process is standing, so
    // the move has to be refused rather than taken with the value following it.
    let expanded = |name: &str, value: &str| -> Result<Option<String>, String> {
        if !EXPANDED_ENV.contains(&name) {
            return Ok(Some(value.to_string()));
        }
        // Windows rules on Windows, the native ones off it: a POSIX reader leaves
        // %HOME% as an ordinary name and expands $HOME instead, so reading this
        // the Windows way would call a relative value absolute.
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
        // The directory being left is unknown, so nothing can be anchored to it.
        // Moving anyway would quietly retarget every relative value at the new
        // directory, so this is only survivable with nothing left to preserve.
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
                // Judged the same way the moving path judges it, or a setting
                // that decides nothing here (an expanded %LOCALAPPDATA%, inline
                // JSON, a 0/1 toggle) would be read as relative and refuse
                // every spawn over a value no directory ever decided.
                let entry = match home {
                    Some(home) => expand_windows_user(entry, home, username.as_deref()),
                    None => entry.to_string(),
                };
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
    // A value the OS declines to resolve is refused rather than dropped: moving
    // the child with that override still relative would retarget it silently,
    // which is what the pinning exists to prevent. The CLI guard refuses the
    // same case.
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
        // The tilde first and the variables second, the order the CLI guard uses,
        // so a value that names a folder through both reaches the same one.
        let value = match home {
            Some(home) => expand_windows_user(&original, home, username.as_deref()),
            None => original.clone(),
        };
        let Some(value) = expanded(name, &value)? else {
            continue;
        };
        if value.is_empty() {
            continue;
        }
        // Windows rules on Windows, the native ones off it, exactly as the
        // lost-directory branch reads them: /opt/vendor is no more cwd-dependent
        // there than C:\\cache is here.
        if is_cwd_independent(&value, windows) {
            // Already names one folder. Still worth writing back if expanding is
            // what made it name one: the reader that does not expand cannot see
            // that on its own.
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
            // Python never expands `~` in PYTHONPATH, so `~\plugins` is an
            // ordinary relative folder there, and expanding it would point the
            // import at a profile folder the interpreter was never reading.
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
            // PYTHONPATH has two spellings that follow the process rather than
            // the caller: an empty component means the working directory itself,
            // and a leading `~` is never expanded there, so Python reads
            // `~\plugins` as an ordinary relative folder.
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
        // 2. A value the OS will not accept is a failure to report here, not one
        // to discover in CreateProcess: the caller turns this into the same
        // "path setting" message as an unresolvable one, and a repair that would
        // hit the same wall is not offered.
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
        // GetFullPathNameW on Windows, which is what knows each drive's own
        // current directory.
        |value| std::path::absolute(value).ok(),
        dirs::home_dir().as_deref(),
        skipped,
        cfg!(windows),
    )
}

/// The pins a move needs, or None when the move must not happen.
///
/// A directory that cannot be named is one nothing can be anchored to. Staying is
/// what the child did before this file learned about working directories, and it
/// keeps every relative setting meaning exactly what it does now; only a
/// directory we can name is worth moving out of. Refusing the whole spawn instead
/// would take a probe, a backend, an auth provision or an update down over a
/// setting the command may never read.
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

/// What the update child neither receives nor needs. It is the one child that
/// reads STUDIO_LOCAL_REPO, and the one that wants no PYTHONPATH at all on
/// Windows, where -I covers only the first interpreter and the update starts
/// more. Skipping it here rather than removing it afterwards means an
/// unresolvable entry cannot refuse an update that was never going to read it.
fn update_child_skipped_env() -> Vec<&'static str> {
    MANAGED_CHILD_SCRUBBED_ENV
        .iter()
        .copied()
        .chain(cfg!(windows).then_some("PYTHONPATH"))
        .collect()
}

/// Pinned when it can be, but never at the cost of the spawn. The update is the
/// one child that receives STUDIO_LOCAL_REPO, and a bare `unsloth studio update`
/// drops it before anything reads it (commands/studio.py), so refusing to start
/// over a stale drive-relative value would defeat the fallback this exists for.
/// The twin of `_BEST_EFFORT_ENV` in the CLI guard.
const BEST_EFFORT_ENV: &[&str] = &["STUDIO_LOCAL_REPO"];

/// What an ordinary managed child neither receives nor needs.
fn child_skipped_env() -> Vec<&'static str> {
    MANAGED_CHILD_SCRUBBED_ENV
        .iter()
        .chain(UPDATE_ONLY_ENV)
        .copied()
        .collect()
}

/// The error a managed spawn would fail with, without spawning anything.
///
/// Preflight asks first: a context that cannot be built is not a broken CLI, and
/// reporting it as one starts an automatic repair that needs the same context
/// and fails the same way.
pub(crate) fn managed_cli_context_error() -> Option<ManagedContextError> {
    let work_dir = match managed_cli_working_dir() {
        Ok(work_dir) => work_dir,
        Err(error) => return Some(ManagedContextError::WorkingDirectory(error)),
    };
    pins_for_move(&work_dir, &child_skipped_env())
        .err()
        .map(ManagedContextError::PathSetting)
}

/// Why a managed spawn cannot be configured. The two are told apart because the
/// fixes are: a profile that has to come back, and a path setting the user wrote
/// that the OS cannot resolve.
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

/// Whether the child has to be told where to run.
///
/// It does not when that is already where the parent is: reopening the inherited
/// directory by name can fail (an ancestor turned unreadable after launch) where
/// simply inheriting the open handle would have worked, so the no-move case is
/// left exactly as it was before this file learned about working directories.
fn needs_explicit_cwd(work_dir: &std::path::Path) -> bool {
    std::env::current_dir()
        .map(|cwd| cwd != work_dir)
        .unwrap_or(true)
}

/// Pin the working directory and mark the child as desktop-managed. Env
/// scrubbing, creation flags and the ownership handshake stay with the caller.
/// For the update, which is the one child that reads STUDIO_LOCAL_REPO.
pub(crate) fn apply_managed_cli_context(cmd: &mut Command) -> Result<(), String> {
    // The update is the one child that reads STUDIO_LOCAL_REPO, and the one that
    // wants no PYTHONPATH at all on Windows: -I covers only the first
    // interpreter, and the update starts more. Skipping it here rather than
    // removing it afterwards means an unresolvable entry cannot refuse an update
    // that was never going to read it.
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
    // Removed here as well as at the call sites, so the skip above is a fact
    // about the child rather than an assumption about every caller.
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

    // issue #8490: Application Control denies the generated unsloth.exe. Every
    // managed invocation must reach the CLI through the interpreter beside it.
    #[cfg(windows)]
    fn managed_venv(test_name: &str) -> (PathBuf, PathBuf, PathBuf) {
        let dir = temp_studio_dir(test_name);
        let python = dir.join("python.exe");
        let bin = dir.join("unsloth.exe");
        fs::write(&python, "").unwrap();
        fs::write(&bin, "").unwrap();
        (dir, python, bin)
    }

    // Quarantine takes the unsigned stub and leaves the environment intact. The
    // finder gates the backend, the updater and the install-status probe, so a None
    // here reports "not installed" for a Studio that still runs.
    #[cfg(windows)]
    #[test]
    fn a_quarantined_stub_is_still_a_managed_install() {
        let studio = temp_studio_dir("quarantined-stub");
        let scripts = studio.join("unsloth_studio").join("Scripts");
        fs::create_dir_all(&scripts).unwrap();

        // No interpreter yet: there is genuinely nothing to run.
        assert_eq!(find_unsloth_binary_in_studio_dir(&studio), None);

        fs::write(scripts.join("python.exe"), "").unwrap();
        assert_eq!(
            find_unsloth_binary_in_studio_dir(&studio),
            Some(scripts.join("unsloth.exe")),
            "a stub-less environment with an interpreter is still an install"
        );

        // And the handle it hands back drives the interpreter as usual.
        let invocation =
            resolve_managed_cli_invocation(&scripts.join("unsloth.exe"), &["studio"]).unwrap();
        assert_eq!(invocation.program, scripts.join("python.exe"));

        fs::remove_dir_all(studio).unwrap();
    }

    // An interrupted migration leaves a half-built new environment beside a legacy
    // one that still works. A launcher anywhere outranks a bare interpreter, or the
    // desktop would drive the broken half and report an install it cannot start.
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

        // Once the new layout has its own launcher, layout order takes over again.
        fs::write(new_scripts.join("unsloth.exe"), "").unwrap();
        assert_eq!(
            find_unsloth_binary_in_studio_dir(&studio),
            Some(new_scripts.join("unsloth.exe"))
        );

        fs::remove_dir_all(studio).unwrap();
    }

    // The other half of the same interruption: the new layout kept the launcher and
    // the legacy one kept the interpreter. Returning the new launcher there fails
    // later with "Managed Python interpreter not found beside Unsloth" while a
    // usable environment sits in the other base.
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

        // With nothing usable anywhere, still answer what this function always
        // answered: the caller's error then names the missing interpreter.
        fs::remove_file(old_scripts.join("python.exe")).unwrap();
        fs::remove_file(old_scripts.join("unsloth.exe")).unwrap();
        assert_eq!(
            find_unsloth_binary_in_studio_dir(&studio),
            Some(new_scripts.join("unsloth.exe"))
        );

        fs::remove_dir_all(studio).unwrap();
    }

    // Both halves of an interrupted migration kept an interpreter and neither kept
    // a launcher, so pass 1 cannot decide and layout order alone would hand back
    // the new base whether or not anything is installed in it. The base that still
    // holds the package the trampoline imports has to win, in either direction, or
    // every capability probe drives an interpreter with nothing to run.
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

        // Neither carries the package: layout order decides, exactly as before.
        assert_eq!(
            find_unsloth_binary_in_studio_dir(&studio),
            Some(new_base.join("Scripts").join("unsloth.exe")),
            "with nothing to choose between them the new layout still wins"
        );

        // The legacy base has the package and the new one does not.
        fs::create_dir_all(old_base.join("Lib").join("site-packages").join("unsloth_cli")).unwrap();
        assert_eq!(
            find_unsloth_binary_in_studio_dir(&studio),
            Some(old_base.join("Scripts").join("unsloth.exe")),
            "the only base with a package to import must win"
        );

        // Once the new base has it too, layout order takes over again.
        fs::create_dir_all(new_base.join("Lib").join("site-packages").join("unsloth_cli")).unwrap();
        assert_eq!(
            find_unsloth_binary_in_studio_dir(&studio),
            Some(new_base.join("Scripts").join("unsloth.exe")),
            "package on both sides is not a reason to prefer the legacy layout"
        );

        // A complete environment anywhere still outranks this whole pass.
        fs::write(old_base.join("Scripts").join("unsloth.exe"), "").unwrap();
        assert_eq!(
            find_unsloth_binary_in_studio_dir(&studio),
            Some(old_base.join("Scripts").join("unsloth.exe"))
        );

        fs::remove_dir_all(studio).unwrap();
    }

    // An editable install of the checkout leaves a .pth and a dist-info in
    // site-packages and no unsloth_cli/ directory at all, so a package-directory test
    // alone would rank a working legacy venv below an empty new one. The Python side
    // accepts the dist-info for this exact shape; this side has to agree.
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

        // Unrelated metadata is not this package. A dist-info for something else must
        // not make an empty venv look installed.
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
                // Caller arguments follow the script, in order.
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
        // Measured on a machine with no policy: PYTHONPROFILEIMPORTTIME=1 gave
        // ~24 KB of stderr from the console script and nothing from a -I
        // trampoline, and PYTHONPATH stopped shadowing.
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

    // The exception, and the only one: the updater rewrites the environment it
    // runs in, so it stays isolated exactly as it shipped. Asserted here as well
    // as in update.rs so the two halves of the rule are stated together.
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
        // -X utf8 comes first either way: -I implies -E, which would drop
        // PYTHONUTF8, and the flag form survives it.
        assert_eq!(isolated.args[0], std::ffi::OsString::from("-X"));
        assert_eq!(isolated.args[1], std::ffi::OsString::from("utf8"));
        assert_eq!(isolated.args[2], std::ffi::OsString::from("-I"));
        // Same program, same trailing arguments; only the flag differs.
        assert_eq!(inherit.program, isolated.program);
        assert_eq!(inherit.args.last(), isolated.args.last());
        // And the default entry point is the inheriting one.
        assert_eq!(
            resolve_managed_cli_invocation(&bin, &["studio"]).unwrap().args,
            inherit.args
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn managed_trampoline_assigns_argv0_before_importing_the_cli() {
        // The order is the whole point: unsloth_cli decides at import time
        // whether it is the console script, which gates the UTF-8 stream
        // reconfigure, the -np<N> rewrite and Typer's prog_name.
        let strip = WINDOWS_CLI_ENTRYPOINT.find("sys.path[:1]");
        let assignment = WINDOWS_CLI_ENTRYPOINT.find("sys.argv[0] = 'unsloth'");
        let import = WINDOWS_CLI_ENTRYPOINT.find("from unsloth_cli import app");
        assert!(strip.is_some() && assignment.is_some() && import.is_some());
        assert!(assignment < import, "{WINDOWS_CLI_ENTRYPOINT}");
        // The cwd must leave sys.path before the import too, or the entry it
        // guards against is still live for that one import.
        assert!(strip < import, "{WINDOWS_CLI_ENTRYPOINT}");
    }

    #[cfg(windows)]
    #[test]
    fn managed_commands_leave_the_python_environment_alone() {
        use std::ffi::OsStr;

        // Harmless under -I, which discarded them anyway; without it, a change
        // the console script does not make.
        let (dir, _python, bin) = managed_venv("managed-cli-env");
        let cmd = build_managed_cli_command(&bin, &["-h"]).unwrap();
        for name in ["PYTHONHOME", "PYTHONPATH"] {
            assert!(
                !cmd.get_envs().any(|(key, _)| key == OsStr::new(name)),
                "{name} must be inherited, not overridden"
            );
        }
        // The async flavour must not drift from the blocking one.
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
        // Falling back to the stub is exactly what a policy-blocked machine
        // cannot run, so a missing interpreter is an error, not a downgrade.
        let bin = temp_studio_dir("managed-cli-no-python").join("unsloth.exe");
        let error = resolve_managed_cli_invocation(&bin, &["-h"]).unwrap_err();
        assert!(error.contains("python.exe"), "{error}");
    }

    // Parity guard: the change is Windows-only. macOS and Linux keep execing
    // the console script with the caller's arguments and nothing else.
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
        // The Windows arm fails closed on a missing python.exe; the POSIX arm
        // must not acquire that failure mode for a path that never had it.
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

/// Find the unsloth binary, preferring the dev repo if available.
/// In dev mode (debug builds), checks for a local .venv in the repo first.
/// Falls back to find_unsloth_binary() which checks ~/.unsloth/studio/unsloth_studio/
/// (new layout) then ~/.unsloth/studio/.venv/ (old layout).
pub(crate) fn resolve_backend_binary() -> Result<std::path::PathBuf, String> {
    // In dev mode, check for local repo venv first
    #[cfg(debug_assertions)]
    {
        // CARGO_MANIFEST_DIR is set at compile time to studio/src-tauri/
        // Repo root is 2 levels up: studio/src-tauri -> studio -> repo_root
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let repo_root = std::path::Path::new(manifest_dir)
            .parent() // studio/
            .and_then(|p| p.parent()); // repo_root/

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
    #[cfg(windows)]
    let _runtime_launch_guard = acquire_studio_runtime_launch_guard()?;

    // A backend started while the job is disarmed is the orphan this guards
    // against. The UI gate is per update action, and a webview remount starts
    // one on its own, so the check belongs on the path that actually spawns.
    #[cfg(windows)]
    if !crate::windows_job::kill_on_close_armed().unwrap_or(false) {
        crate::windows_job::resume_after_update_installer().map_err(|error| {
            format!("Refusing to start the backend with crash cleanup disarmed: {error}")
        })?;
        // The same pair the UI's resume does: the pre-exit hook has already run
        // its cleanup, and leaving that guard set means the next attempt's hook
        // suspends kill-on-close without stopping this backend first.
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

    // A precondition like resolve_backend_binary above, so it runs before any
    // ownership or job state is touched.
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
    // Built before ownership is claimed: a missing managed interpreter must not
    // leave a pending owner behind for a backend that was never spawned.
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
    // The program actually spawned, not the console script it resolved from. This
    // line is what a user attaches to an issue, and naming the stub would point
    // every Application Control report at a binary we never start.
    // Same shape as before, so a support log reads the same on every platform; only
    // the program differs, and only where the interpreter is what actually starts.
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
        // The drive holding an override can go between the preflight check and
        // this call, and a panic in the spawn path takes the desktop with it.
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

    #[cfg(windows)]
    cmd.env(STUDIO_RUNTIME_GATE_HANDOFF_ENV, "1");

    if let Some(native_state) = app.try_state::<crate::native_intents::NativeIntakeState>() {
        cmd.env(
            crate::native_backend_lease::LEASE_SECRET_ENV,
            native_state.lease_secret_env(),
        );
    }

    crate::desktop_backend_owner::apply_owner_env(&mut cmd, &pending_owner);

    #[cfg(target_os = "linux")]
    scrub_appimage_python_env(&mut cmd);

    // Tauri uses the legacy root regardless of UNSLOTH_STUDIO_HOME / STUDIO_HOME;
    // scrub so the spawned Python backend can't diverge. UNSLOTH_LLAMA_CPP_PATH
    // is a pre-existing user-controlled llama.cpp dir override; keep it.
    cmd.env_remove("UNSLOTH_STUDIO_HOME");
    cmd.env_remove("STUDIO_HOME");

    // read_output_stream decodes as UTF-8; without these, Python encodes its
    // redirected streams with the locale code page and non-ASCII lands as U+FFFD.
    // The backend gets UTF-8 from -X utf8 in its argv; these reach it too now
    // that -I is gone, and carry on down to whatever it spawns.
    #[cfg(windows)]
    {
        cmd.env("PYTHONUTF8", "1");
        cmd.env("PYTHONIOENCODING", "utf-8");
    }

    // Reset state, spawn, and store the child while holding the backend mutex.
    // This keeps the no-child check atomic: a concurrent start/stop cannot slip
    // into the old window between generation reset and child storage.
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

        // On Windows, launch the backend directly with hidden-window flags.
        // The app process is assigned to a KILL_ON_JOB_CLOSE job in main.rs, so
        // children inherit crash-safe cleanup without the buggy per-child JobObject wrapper.
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
                // No handle owns this live child yet, so stop it before returning.
                // The backend mutex stays held until cleanup finishes.
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
    // One deadline for this start, shared with the watchdog below. Port
    // validation must not outlive it: the watchdog's server-start-timeout puts
    // the window in an error state that a later server-port does not clear.
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

/// Backoff between port-verification probes, doubling from min to max.
///
/// A probe is not free: it costs a liveness request plus a desktop-login
/// request, each up to LOCAL_HTTP_TIMEOUT, against a backend that is by
/// definition busy. Polling at a fixed short interval would add load to the
/// slow start it is waiting on. Starting small still wins the common race,
/// where the backend is a few hundred ms from ready, while the cap keeps a
/// long torch import down to a handful of probes rather than dozens.
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

    // The backend announces its port once. That line arrives while it is still
    // importing torch, so a single probe races a backend that cannot answer
    // inside LOCAL_HTTP_TIMEOUT yet: on a cold CPU-only machine /api/liveness
    // has been seen taking 2.1 s against a 2 s budget. Discarding the only
    // announcement left the window waiting out the start deadline on the port
    // the backend had already reported it could not bind. Keep probing until
    // the backend answers or that deadline, shared with the watchdog, passes.
    let mut delay = PORT_VALIDATION_RETRY_MIN;
    let mut attempts = 0u32;
    let mut verified_late = false;
    let valid = loop {
        // Before the probe, not just after a failed one: the announcement
        // itself can arrive past the deadline on a very slow start, and the
        // watchdog does not kill the backend when it times out.
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
            // A probe that started in time can still finish late. Emitting
            // server-port after the watchdog's server-start-timeout strands the
            // window in an error state it has no handler to leave.
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
        // Stop once this generation is gone or another path claimed the port,
        // so a restarted backend does not keep an old probe alive. Bound the
        // guard to this statement: the future must stay Send across the await.
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
        // start_timed_out is the watchdog's claim, taken under this same lock,
        // so exactly one of the two outcomes reaches the window.
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

/// How long a backend may take to become reachable before the window stops waiting.
///
/// A healthy managed backend imports torch and serves /api/health in roughly 12 s on a
/// CI runner. This is deliberately an order of magnitude looser: the deadline exists to
/// end an unbounded wait, not to police slow machines, and a false "failed to start" on
/// a cold laptop would be worse than the bug it fixes.
const BACKEND_START_DEADLINE: Duration = Duration::from_secs(300);

/// Report an unresponsive backend early, with its own output.
///
/// A backend that hangs (alive, silent, never binds its port) is not unreported today:
/// `commands.rs`'s health watchdog kills it and emits `server-crashed` once three
/// 15 s probes fail past `BACKEND_STARTUP_GRACE_PERIOD`, so at roughly t+330 s. What
/// that path cannot do is say why: `server-crashed` carries no payload, so the user
/// gets "Server stopped unexpectedly" and nothing to act on.
///
/// This fires ~30 s earlier, carries the backend's last log lines, and deliberately
/// does NOT kill it, leaving the kill policy with the watchdog that has the health
/// evidence to justify it.
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
                    // A newer start superseded this one, or the backend is gone: either
                    // way this watchdog is watching something that no longer exists.
                    if proc.generation != generation || !proc.has_owned_backend() {
                        return;
                    }
                    // The port is only recorded after validation, so this is
                    // "reachable", not merely "printed a number".
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
                // Same three conditions as the loop. Dropping has_owned_backend here
                // would let a crash in the last second be overwritten by a message
                // claiming the backend is still running.
                if proc.generation != generation || proc.port.is_some() || !proc.has_owned_backend()
                {
                    (false, String::new())
                } else {
                    // Claim the outcome while still holding the lock. Deciding
                    // here and emitting after the unlock would otherwise let a
                    // port validation that succeeded in between emit
                    // server-port on top of this timeout.
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

/// Read lines from a child process stream (stdout or stderr).
/// For stdout, parse TAURI_PORT=(\d+) candidates for async validation.
/// When stdout closes and the stop was not intentional, emit server-crashed.
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
    // Did we leave the loop because the child closed the stream, or for a reason of our
    // own? That decides whether dropping the read end here is safe. See below.
    let mut saw_eof = false;

    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) => {
                saw_eof = true;
                break;
            }
            Ok(_) => {
                let text = String::from_utf8_lossy(trim_line_endings(&buf)).into_owned();
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

                // Buffer the log line only for the current backend generation.
                // Old reader threads can briefly outlive a stop/start cycle;
                // they must not overwrite the new backend's port or logs.
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

                info!("[backend] {}", log_line);

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

    // Every other way out of that loop (a generation mismatch, a poisoned mutex, a read
    // error) leaves the child ALIVE while this function is about to drop the read end.
    // That closes the pipe underneath it, and the backend's next write takes EPIPE and
    // dies. Seen under strace on a failing run: three good writes, then
    //     write(1, "Session log: ...", 88) = -1 EPIPE
    //     --- SIGPIPE ---  exit_group(1)
    // with that same line reaching the session log on disk a moment later. The server
    // had everything it needed and stopped only because we stopped listening.
    //
    // So never hand a live child a reader-less pipe: if we are giving up on parsing,
    // keep draining to EOF. Discarding costs one blocked thread; closing costs the
    // backend.
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
                // read_until retries this internally; a raw read does not, and giving
                // up on EINTR would drop the read end and re-create the EPIPE above.
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

/// Exit status of a child whose stdout just closed, or None if it really is still alive.
///
/// A single non-blocking `try_wait()` at the instant stdout EOFs is a race: the pipe
/// closes when the process drops its handles, observably before the process object
/// signals, so a backend that HAS died reports `Ok(None)`.
///
/// Losing that race was not cosmetic. `exited` stayed false, the owner metadata was
/// never cleared, `proc.port` kept pointing at a dead server and no `server-crashed`
/// was emitted, so the window sat on the startup screen with nothing reported. Seen on
/// Windows CI, which logged "process is still running" for a PID that was already gone.
///
/// Returning None still means "genuinely alive", which matters because Studio may close
/// its own stdout once logging moves to the session log, so stdout EOF alone must not
/// be read as death. Same poll shape as `wait_for_child_exit` below.
fn exit_status_after_stdout_closed(child: &mut Box<dyn ChildWrapper + Send>) -> Option<String> {
    for attempt in 0..30 {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status.to_string()),
            Ok(None) => {
                // Cheap first look before paying for any sleep: a clean shutdown has
                // usually reaped by the time we get here.
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

// A login-started desktop on Windows inherits C:\Windows\system32 (issue #8510),
// and so did every CLI child, where the Python CLI refuses to run. These pin the
// replacement directory and that each command carries it.
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
        // Reporting it available sends a SYSTEM account to an install flow that
        // the working directory resolver then refuses to start.
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
            // A name the child keeps: the two Studio roots are removed for
            // every managed spawn, so they are never pinned.
            "UNSLOTH_COMPILE_LOCATION" => Some("  studio  ".to_string()),
            "OLLAMA_MODELS" => Some("D:\\models".to_string()),
            "HF_HUB_CACHE" => Some("C:\\hub".to_string()),
            "XDG_CACHE_HOME" => Some("   ".to_string()),
            "LLAMA_SERVER_PATH" => Some("\\srv\\llama-server".to_string()),
            // Relative to drive D's own current directory, not to the cwd.
            "HF_DATASETS_CACHE" => Some("D:datasets".to_string()),
            _ => None,
        };
        // What GetFullPathNameW would answer: drive D's own current directory,
        // and the current drive for a root-relative value.
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
                // Root-relative: the drive is the current one, which the move
                // can change, so the OS resolves it before that happens.
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

        // An unresolvable drive-relative value refuses the whole move rather
        // than being dropped: a child moved with that override still relative
        // would read and write somewhere else than the caller named.
        assert!(relative_override_pins_from(Some(cwd.clone()), &work_dir, env, |_| None, Some(std::path::Path::new("C:\\Users\\me")), MANAGED_CHILD_SCRUBBED_ENV, true).is_err());

        // Staying put is the common case: nothing is rewritten, so a desktop
        // started from a project folder keeps every override as it was.
        assert!(
            relative_override_pins_from(Some(work_dir.clone()), &work_dir, env, absolute, Some(std::path::Path::new("C:\\Users\\me")), MANAGED_CHILD_SCRUBBED_ENV, true)
                .unwrap()
                .is_empty()
        );
        // The directory being left is unknown, so a relative override cannot be
        // anchored to it and the move is refused rather than retargeting it.
        assert!(relative_override_pins_from(None, &work_dir, env, absolute, Some(std::path::Path::new("C:\\Users\\me")), MANAGED_CHILD_SCRUBBED_ENV, true).is_err());
        // With nothing relative left to preserve, an unknown directory is fine.
        let absolute_only = |name: &str| match name {
            "HF_HOME" => Some("D:\\cache".to_string()),
            "HF_HUB_CACHE" => Some("C:\\hub".to_string()),
            // Removed for every managed child, so never in the way.
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
        // The installer built the profile as it went before it shared this
        // resolver; a managed child that created one would leave an empty folder
        // shadowing the roaming profile when it finally arrives.
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
        // build_update_command removes PYTHONPATH on Windows, so pinning it there
        // could only refuse an update over a value the child never receives.
        let skipped = update_child_skipped_env();
        assert_eq!(
            skipped.contains(&"PYTHONPATH"),
            cfg!(windows),
            "PYTHONPATH is skipped exactly where the update drops it"
        );
        // STUDIO_LOCAL_REPO stays: the update is the one child that reads it.
        assert!(!skipped.contains(&"STUDIO_LOCAL_REPO"));
        assert!(child_skipped_env().contains(&"STUDIO_LOCAL_REPO"));
    }

    #[test]
    fn a_posix_list_is_split_and_joined_with_its_own_separator() {
        // The lost-directory branch already reads POSIX rules; the moving path
        // has to as well, or "plugins:vendor" is one entry and both import roots
        // leave with it.
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
        // Anchoring a value that was already near the limit can cross it, and
        // CreateProcess is too late to say which setting did.
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
        // posixpath.expandvars reads $NAME and ${NAME}; %NAME% is an ordinary
        // filename there, so reading it the Windows way would call a relative
        // value absolute and let the child move out from under it.
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

        // And the lost-directory branch reads it that way: %HOME%/hf depends on
        // where the process is standing, so the move is refused rather than
        // taken with the value following it.
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
        // The POSIX spelling of the same setting names one folder, so it passes.
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
        // Windows refuses the variable, and CreateProcess is too late to say
        // which setting did it: the caller turns this into the same message as
        // an unresolvable one, and offers no repair that would hit the same wall.
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
        // ntpath.expanduser answers USERPROFILE, so a portable or overridden
        // environment must not send the two layers to different folders.
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
            // What dirs::home_dir() would answer, which is not where the tilde
            // points once USERPROFILE says otherwise.
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
        // Every combination of the three things the pinning looks at, so "nothing
        // happens unless the directory is one the CLI refuses" is a table rather
        // than a claim. The child either stays with nothing rewritten, moves with
        // every relative setting anchored to the directory it left, or the caller
        // is told exactly which setting could not be preserved. There is no
        // fourth outcome.
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
                // Staying put rewrites nothing whatever the environment holds.
                ("already-there", "clean", "nothing rewritten"),
                ("already-there", "relative", "nothing rewritten"),
                ("already-there", "absolute", "nothing rewritten"),
                ("already-there", "toggle", "nothing rewritten"),
                ("system", "clean", "nothing rewritten"),
                ("system", "relative", "anchored to the directory being left"),
                ("system", "absolute", "nothing rewritten"),
                ("system", "toggle", "nothing rewritten"),
                // Nothing can be anchored to a directory with no name, so a
                // relative setting is reported and the caller stays put.
                ("unknown", "clean", "nothing rewritten"),
                ("unknown", "relative", "reported as unpreservable"),
                ("unknown", "absolute", "nothing rewritten"),
                ("unknown", "toggle", "nothing rewritten"),
            ]
        );
    }

    #[test]
    fn a_directory_that_cannot_be_named_leaves_the_child_where_it_is() {
        // Nothing can be anchored to a directory this process cannot name, so the
        // child stays in it rather than the spawn failing over a setting the
        // command may never read. What it must not do is move and take the
        // setting with it.
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
        // And the spawn path turns that report into staying put, without
        // moving this process to prove it: an unpinnable environment plus a
        // directory that cannot be named means the child stays where it is.
        let unpinnable: Result<Vec<(&'static str, PathBuf)>, String> =
            Err("DG_VISUAL_BIN is relative and the directory it was written against is gone"
                .to_string());
        assert_eq!(stay_put_on_lost_cwd(unpinnable.clone(), false), Ok(None));
        assert!(stay_put_on_lost_cwd(unpinnable, true).is_err());
    }

    #[test]
    fn an_expansion_that_stays_relative_refuses_the_move() {
        // The reader expands once. When that still holds a reference the folder
        // depends on where the process is standing, so the move is refused; when
        // it already names a drive the value is left exactly as written.
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
        // Nested: one pass leaves the reference NESTED itself holds.
        refused(
            pins(|name: &str| match name {
                "USERPROFILE" => Some("C:\\Users\\me".to_string()),
                "NESTED" => Some("%USERPROFILE%\\AppData\\Local".to_string()),
                "HF_ASSETS_CACHE" => Some("%NESTED%\\assets".to_string()),
                _ => None,
            })
            .unwrap_err(),
        );
        // Escaped: one pass turns %% into the reference it was protecting.
        refused(
            pins(|name: &str| match name {
                "USERPROFILE" => Some("C:\\Users\\me".to_string()),
                "XDG_CACHE_HOME" => Some("%%USERPROFILE%%\\xdg".to_string()),
                _ => None,
            })
            .unwrap_err(),
        );
        // Self-referencing: no number of passes settles it.
        refused(
            pins(|name: &str| (name == "HF_HOME").then(|| "%HF_HOME%\\cache".to_string()))
                .unwrap_err(),
        );
        // Already names a drive after one pass, so it means the same folder from
        // anywhere and nothing is rewritten.
        assert_eq!(
            pins(|name: &str| (name == "HF_ASSETS_CACHE")
                .then(|| "C:\\cache\\%UNSET%\\assets".to_string()))
            .unwrap(),
            Vec::new()
        );
    }

    #[test]
    fn the_model_paths_llama_server_reads_for_itself_are_pinned() {
        // llama-server resolves these against its own working directory, so a
        // relative one has to move with the child. The URL spelling names no
        // local file.
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
        // %LOCALAPPDATA%\hf, inline JSON and a toggle are the same three shapes
        // the moving path exempts. Judging them raw here refused every managed
        // spawn over values no directory ever decided.
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
        // A genuinely relative folder in the same list is still refused.
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
        // A name in one list and not the other means the same install places
        // state in two folders, depending on which layer moved the child.
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

    // The SYSTEM account's profile is C:\Windows\System32\config\systemprofile, so
    // trusting the home API here would hand the child the folder it just rejected.
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

    // The separator keeps the match on a path boundary, so a profile that merely
    // starts with the same letters as the Windows directory is not rejected.
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

    // A WINDIR of "C:\" would otherwise swallow the whole drive.
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

    // \\?\C:\Windows\... is the same folder spelled the long way; the Python
    // guard strips the prefix too.
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

    // The pin exists to replace an unusable directory, not to relocate every
    // launch: a desktop started from a project folder keeps resolving ./models
    // and other cwd-relative defaults there, on every platform.
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

    // A WINDIR aimed at the user's profile would otherwise make this reject that
    // profile, so the backend would never start anywhere on that machine.
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
        // relative_override_pins reads the ambient environment, which another
        // test may be swapping, so this takes the crate-wide env lock.
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
    fn only_the_folders_the_cli_refuses_count_as_unusable() {
        let windirs = [PathBuf::from("C:\\Windows")];
        for unusable in [
            "C:\\Windows\\System32",
            "c:\\windows\\system32\\config\\systemprofile",
            "C:\\Windows\\SysWOW64",
            "\\\\?\\C:\\Windows\\System32",
        ] {
            assert!(
                is_unusable_cwd(std::path::Path::new(unusable), &windirs),
                "{unusable} must be replaced"
            );
        }
        for usable in [
            // The guard has always allowed these, so a child that was running
            // from one keeps running from it.
            "C:\\Windows\\Temp\\project",
            "C:\\Windows",
            "C:\\Windows2\\System32",
            "C:\\Users\\me\\projects",
        ] {
            assert!(
                !is_unusable_cwd(std::path::Path::new(usable), &windirs),
                "{usable} must be kept"
            );
        }
    }

    #[test]
    fn an_extended_unc_path_compares_the_same_in_either_case() {
        // The object manager accepts \\?\unc\, so a profile spelled that way
        // must not be read as a relative name.
        assert_eq!(
            normalize_windows_path(std::path::Path::new("\\\\?\\UNC\\server\\profiles\\me")),
            normalize_windows_path(std::path::Path::new("\\\\?\\unc\\server\\profiles\\me"))
        );
        assert!(is_fully_qualified("\\\\?\\unc\\server\\profiles\\me"));
        assert!(is_fully_qualified("\\\\?\\UNC\\server\\profiles\\me"));
    }

    #[test]
    fn a_lost_directory_is_judged_entry_by_entry() {
        // "C:\\vendor;plugins" starts with a drive and still carries something
        // the directory that is gone decided.
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
        // Every entry qualified: nothing is left depending on it.
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
        // Off Windows the same fallback runs, and what it reads there is a POSIX
        // environment. Judging /var/cache/unsloth by the Windows rules called it
        // relative and failed preflight and every managed spawn over a setting
        // that was never relative.
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
        // Something genuinely relative is still refused.
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
        // A POSIX list is separated by ':', so a relative entry behind an
        // absolute one is still caught.
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
        // And on Windows the same value stays Windows-judged: "/var/cache" is
        // the root of whichever drive the process is on.
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
        // STUDIO_LOCAL_REPO is read by the update and installer path alone, so a
        // stale one must not fail a probe or a backend start that ignores it.
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
        // ~name is the sibling profile, as ntpath resolves it.
        assert_eq!(expand_windows_user("~other\\x", home, me), "C:\\Users\\other\\x");
        // ~me is this profile whatever the folder is called.
        let domain = std::path::Path::new("C:\\Users\\me.DOMAIN");
        assert_eq!(expand_windows_user("~me\\x", domain, me), "C:\\Users\\me.DOMAIN\\x");
        // And ntpath declines to guess a sibling when this profile is not named
        // after the current user, so neither does this.
        assert_eq!(expand_windows_user("~other\\x", domain, me), "~other\\x");
        assert_eq!(expand_windows_user("~other\\x", home, None), "~other\\x");
        // Nothing else is touched.
        for value in ["cache", "C:\\cache", "a~b"] {
            assert_eq!(expand_windows_user(value, home, me), value);
        }
    }

    #[test]
    fn every_spelling_expandvars_takes_is_expanded_here_too() {
        let lookup = |name: &str| match name {
            "LOCALAPPDATA" => Some("C:\\Users\\me\\AppData\\Local".to_string()),
            // ntpath counts the hyphen as part of a $ name, so this one has to
            // win over a lookup of "CACHE".
            "CACHE-ROOT" => Some("C:\\right".to_string()),
            "CACHE" => Some("C:\\wrong".to_string()),
            "TWO WORDS" => Some("C:\\spaced".to_string()),
            _ => None,
        };
        // The three forms os.path.expandvars takes on a Windows path.
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
        // A percent name may hold spaces; a dollar name stops at the dot.
        assert_eq!(expand_windows_vars("%TWO WORDS%\\x", &lookup), "C:\\spaced\\x");
        assert_eq!(expand_windows_vars("$CACHE.d", &lookup), "C:\\wrong.d");
        // Doubled markers stand for one character.
        assert_eq!(expand_windows_vars("100%%", &lookup), "100%");
        assert_eq!(expand_windows_vars("$$HOME", &lookup), "$HOME");
        // A quoted run is copied through, so what is inside it is not expanded.
        assert_eq!(
            expand_windows_vars("'%LOCALAPPDATA%'\\hf", &lookup),
            "'%LOCALAPPDATA%'\\hf"
        );
        // Unset names, unterminated references and a lone marker stay as written.
        for value in [
            "%NOT_SET%\\hub",
            "$NOT_SET\\hub",
            "${LOCALAPPDATA\\hf",
            "%LOCALAPPDATA\\hf",
            "a$b",
            "$",
            "50% off",
            // Non-ASCII, which \w under re.ASCII does not match: the byte walk
            // must step over it whole rather than split the character.
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
        // An empty component is the working directory, and `~` is never expanded
        // in PYTHONPATH, so Python reads it as an ordinary relative folder.
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
        // `~\plugins` is anchored as the literal relative folder Python reads,
        // not turned into the profile the interpreter was never looking at.
        let expected = format!(
            "{};{};C:\\shared\\lib",
            cwd.to_string_lossy(),
            // join, like the anchoring does, so the separator is the host's.
            cwd.join("~\\plugins").to_string_lossy()
        );
        assert_eq!(pins, vec![("PYTHONPATH", PathBuf::from(expected))]);
    }

    #[test]
    fn a_cache_override_is_expanded_before_it_is_judged() {
        // One reader expands %LOCALAPPDATA% and one does not, so the value is
        // written out here and both then see the same folder.
        let cwd = PathBuf::from("C:\\Windows\\System32");
        let work_dir = PathBuf::from("C:\\Users\\me\\.unsloth");
        let pins = relative_override_pins_from(
            Some(cwd.clone()),
            &work_dir,
            |name| match name {
                "LOCALAPPDATA" => Some("C:\\Users\\me\\AppData\\Local".to_string()),
                "HF_HOME" => Some("%LOCALAPPDATA%\\hf".to_string()),
                // A name this machine does not set stays as written, and is
                // anchored like any other relative value.
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
        // A directory really called "[llama]" or "%data%" is legal on Windows,
        // and the readers of these two names take it as exactly that.
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
        // Three readers in the tree treat these as something other than a path,
        // so anchoring one would change its meaning rather than preserve it.
        let cwd = PathBuf::from("C:\\Windows\\System32");
        let work_dir = PathBuf::from("C:\\Users\\me\\.unsloth");
        let pins = relative_override_pins_from(
            Some(cwd),
            &work_dir,
            |name| match name {
                // MLX_HOSTFILE holds either a filename or the host list itself.
                "MLX_HOSTFILE" => Some("[{\"ssh\": \"node0\"}]".to_string()),
                // A bare toggle is ignored on purpose: there is no "allow all".
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
        // `~` is written out: only some readers of these names expand it.
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
        // relative_override_pins reads the ambient environment, which another
        // test may be swapping, so this takes the crate-wide env lock.
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
        // The child's environment reaches grandchildren, and the desktop
        // resolves the directory again on every spawn, so the rewrite has to
        // land on the same value however many times it runs.
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
        // The separator is whatever this platform's join produces; the point is
        // that the value stopped moving after the first pass.
        assert_eq!(values[0].1, cwd.join("cache").to_string_lossy());
        assert_eq!(values[1].1, "D:\\work\\datasets");
        assert!(is_fully_qualified(&values[0].1) && is_fully_qualified(&values[1].1));
    }

    #[test]
    fn a_configured_tokio_command_carries_the_directory_and_the_marker() {
        // relative_override_pins reads the ambient environment, which another
        // test may be swapping, so this takes the crate-wide env lock.
        let _env = crate::native_path_policy::PROCESS_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let expected = managed_cli_working_dir().expect("home must resolve");
        let mut tokio_cmd = tokio::process::Command::new("unsloth");
        apply_managed_cli_context_tokio(&mut tokio_cmd).expect("context must apply");
        // Under test the resolver keeps this process's own directory, and the
        // child inherits it rather than reopening it by name.
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

    // The marker only helps if the Python side reads the same name; a rename on
    // either side degrades silently to argv matching, which is the fallback path.
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

    // Configuring a child must not move the desktop process itself: a global chdir
    // would change relative path resolution for dialogs, the updater and cleanup.
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

    // The platform the bug was reported on, on the Windows leg of studio-tauri-smoke:
    // a child must observe the chosen directory rather than the launcher's.
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

    // The end of the chain the bug actually broke: the child must observe the chosen
    // directory, including through the unix process-group wrapper used by start_backend.
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

// The race this pins is not visible from the type system, so it uses real processes
// rather than a mock: a child's stdout pipe can EOF a moment before the kernel reports
// the exit, and a single non-blocking try_wait() at that instant returns Ok(None). The
// old code read that as "still running", never emitted server-crashed, and left the
// window waiting on a backend that was already gone.
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

    // The half of the contract a naive "retry until you get something" would break: a
    // backend may legitimately close its own stdout after moving logging to its session
    // log, and calling that a crash would kill a healthy backend.
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

    // The regression itself: exit and observation race, so the check has to survive the
    // child dying at an arbitrary point rather than only before or only after.
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
