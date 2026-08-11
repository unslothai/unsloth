//! Which install tree a live process is running out of.

use std::path::{Path, PathBuf};

/// Absolute path of the executable behind `pid`, when the OS will say.
///
/// None means unknown, never "no such process": a pid belonging to another user
/// is refused on every platform here, and callers must not read that as absence.
pub(crate) fn executable_path(pid: u32) -> Option<PathBuf> {
    executable_path_impl(pid)
}

/// Whether `pid` is running out of `tree`.
///
/// None when the executable could not be read, which every caller has to decide
/// about for itself: it is the answer both for a process we may not query and
/// for a platform with no implementation here.
pub(crate) fn runs_from(pid: u32, tree: &Path) -> Option<bool> {
    let exe = executable_path(pid)?;
    Some(path_is_within(&exe, tree))
}

fn path_is_within(path: &Path, tree: &Path) -> bool {
    // Compared case-insensitively throughout: Windows paths differ in case for
    // the same file, and a false negative here silently drops a real blocker.
    // Component-wise, so a sibling tree with a shared name prefix cannot match.
    let mut tree_parts = tree.components();
    let mut path_parts = path.components();
    loop {
        let Some(expected) = tree_parts.next() else {
            return true;
        };
        let Some(actual) = path_parts.next() else {
            return false;
        };
        let expected = expected.as_os_str().to_string_lossy().to_lowercase();
        let actual = actual.as_os_str().to_string_lossy().to_lowercase();
        if expected != actual {
            return false;
        }
    }
}

#[cfg(target_os = "linux")]
fn executable_path_impl(pid: u32) -> Option<PathBuf> {
    std::fs::read_link(format!("/proc/{pid}/exe")).ok()
}

#[cfg(target_os = "macos")]
fn executable_path_impl(pid: u32) -> Option<PathBuf> {
    if pid > i32::MAX as u32 {
        return None;
    }
    // PROC_PIDPATHINFO_MAXSIZE, which libc does not re-export.
    let mut buffer = vec![0u8; 4 * libc::MAXPATHLEN as usize];
    let written = unsafe {
        libc::proc_pidpath(
            pid as i32,
            buffer.as_mut_ptr() as *mut libc::c_void,
            buffer.len() as u32,
        )
    };
    if written <= 0 {
        return None;
    }
    buffer.truncate(written as usize);
    Some(PathBuf::from(String::from_utf8(buffer).ok()?))
}

#[cfg(windows)]
fn executable_path_impl(pid: u32) -> Option<PathBuf> {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::Threading::{
        OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32,
        PROCESS_QUERY_LIMITED_INFORMATION,
    };

    unsafe {
        // The limited right, not PROCESS_QUERY_INFORMATION: it is granted for
        // processes at a higher integrity level, which a Studio started from an
        // elevated terminal is.
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return None;
        }
        let mut buffer = vec![0u16; 32768];
        let mut length = buffer.len() as u32;
        let ok = QueryFullProcessImageNameW(
            handle,
            PROCESS_NAME_WIN32,
            buffer.as_mut_ptr(),
            &mut length,
        );
        let _ = CloseHandle(handle);
        if ok == 0 {
            return None;
        }
        buffer.truncate(length as usize);
        Some(PathBuf::from(std::ffi::OsString::from_wide(&buffer)))
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn executable_path_impl(_pid: u32) -> Option<PathBuf> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn our_own_executable_is_resolvable() {
        let pid = std::process::id();
        let exe = executable_path(pid).expect("this platform should resolve its own path");

        assert_eq!(exe, std::env::current_exe().unwrap());
    }

    #[test]
    fn a_process_is_inside_the_tree_that_contains_it() {
        let exe = std::env::current_exe().unwrap();
        let tree = exe.parent().unwrap().to_path_buf();

        assert_eq!(runs_from(std::process::id(), &tree), Some(true));
        assert_eq!(
            runs_from(std::process::id(), Path::new("/definitely/elsewhere")),
            Some(false)
        );
    }

    #[test]
    fn a_shared_name_prefix_is_not_containment() {
        assert!(!path_is_within(
            Path::new("/home/u/.unsloth/studio-old/bin/python"),
            Path::new("/home/u/.unsloth/studio"),
        ));
        assert!(path_is_within(
            Path::new("/home/u/.unsloth/studio/unsloth_studio/bin/python"),
            Path::new("/home/u/.unsloth/studio"),
        ));
    }

    /// Windows reports the same file under different casings, and a false
    /// negative here silently drops a real blocker.
    #[test]
    fn containment_ignores_case() {
        assert!(path_is_within(
            Path::new("/Users/U/.Unsloth/Studio/unsloth_studio/python"),
            Path::new("/users/u/.unsloth/studio"),
        ));
    }

    #[test]
    fn a_tree_contains_itself_but_not_its_parent() {
        let tree = Path::new("/home/u/.unsloth/studio");

        assert!(path_is_within(tree, tree));
        assert!(!path_is_within(Path::new("/home/u/.unsloth"), tree));
    }
}
