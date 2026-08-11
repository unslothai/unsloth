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
/// None is "cannot tell", which every caller has to decide about for itself. It
/// covers a process we may not query, a platform with no implementation here,
/// and the venv case below.
///
/// `shared_interpreters` are executables that a process inside the tree may be
/// running without the tree appearing anywhere in its image path. On unix uv
/// symlinks `bin/python` at the base interpreter, which `install.sh` relies on,
/// and both `/proc/{pid}/exe` and `proc_pidpath` report the resolved target. On
/// Windows uv's `Scripts/python.exe` is a trampoline that spawns the base
/// interpreter as a child, so the image is the base one there too. Seeing that
/// binary is therefore no proof either way, so it answers None rather than
/// false: guessing false would let a repair rewrite the venv underneath a live
/// backend, which is the failure this guards.
pub(crate) fn runs_from(pid: u32, tree: &Path, shared_interpreters: &[PathBuf]) -> Option<bool> {
    // argv[0] keeps the path as invoked, so it still names the venv where the
    // image path no longer does. Positive evidence only: a process can be given
    // any argv, and there is no platform where its absence means anything.
    if let Some(argv0) = first_argument(pid) {
        if path_is_within(&argv0, tree) {
            return Some(true);
        }
    }
    let exe = executable_path(pid)?;
    if path_is_within(&exe, tree) {
        return Some(true);
    }
    if shared_interpreters.iter().any(|shared| is_same_path(shared, &exe)) {
        return None;
    }
    Some(false)
}

fn is_same_path(left: &Path, right: &Path) -> bool {
    // Both sides are canonicalized by the caller where the filesystem allows
    // it, so this is the last-resort textual comparison.
    left.as_os_str().to_string_lossy().to_lowercase()
        == right.as_os_str().to_string_lossy().to_lowercase()
}

/// The command line's argv[0] for `pid`.
///
/// Linux only. macOS needs a KERN_PROCARGS2 sysctl and Windows reads the
/// remote PEB, neither of which is implemented. On those two a backend started
/// through a venv is therefore indeterminate rather than positively ours, which
/// still blocks on a record naming the port.
#[cfg(target_os = "linux")]
fn first_argument(pid: u32) -> Option<PathBuf> {
    let cmdline = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
    let argv0 = cmdline.split(|byte| *byte == 0).next()?;
    if argv0.is_empty() {
        return None;
    }
    Some(PathBuf::from(String::from_utf8_lossy(argv0).into_owned()))
}

#[cfg(not(target_os = "linux"))]
fn first_argument(_pid: u32) -> Option<PathBuf> {
    None
}

/// Interpreters a backend of this install may be running through a symlink.
///
/// Both venv layouts, since `~/.unsloth/studio` is shared with the CLI
/// installer and an older install still has the `.venv` one. Canonicalized so
/// the comparison above meets the same resolved path the OS reports; an entry
/// that does not resolve is dropped, since nothing can be running it.
pub(crate) fn shared_interpreters_in(tree: &Path) -> Vec<PathBuf> {
    let mut resolved = Vec::new();
    let mut add = |candidate: PathBuf| {
        if let Ok(target) = std::fs::canonicalize(&candidate) {
            if !resolved.contains(&target) {
                resolved.push(target);
            }
        }
    };
    for venv in ["unsloth_studio", ".venv"] {
        let venv = tree.join(venv);
        for (dir, name) in [("bin", "python"), ("Scripts", "python.exe")] {
            add(venv.join(dir).join(name));
        }
        for base in base_interpreters_of(&venv) {
            add(base);
        }
    }
    resolved
}

/// The base interpreters a venv defers to, from its `pyvenv.cfg`.
///
/// Needed for more than tidiness on Windows: uv puts a trampoline exe at
/// `Scripts/python.exe` which spawns the base interpreter as a CHILD process,
/// so the backend's image path is the base one and the venv path appears
/// nowhere in it. Canonicalizing the trampoline yields the trampoline, so
/// without this the running backend would look like a foreign process on the
/// one platform where argv[0] is not read.
fn base_interpreters_of(venv: &Path) -> Vec<PathBuf> {
    let Ok(config) = std::fs::read_to_string(venv.join("pyvenv.cfg")) else {
        return Vec::new();
    };
    let mut home = None;
    let mut version = None;
    for line in config.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        match key.trim() {
            // The documented key: the directory holding the interpreter this
            // venv was built from.
            "home" => home = Some(PathBuf::from(value.trim())),
            "version_info" => {
                let mut parts = value.trim().split('.');
                if let (Some(major), Some(minor)) = (parts.next(), parts.next()) {
                    version = Some(format!("{major}.{minor}"));
                }
            }
            _ => {}
        }
    }
    let Some(home) = home else {
        return Vec::new();
    };
    let mut names = vec![
        "python.exe".to_string(),
        "python3.exe".to_string(),
        "python".to_string(),
        "python3".to_string(),
    ];
    if let Some(version) = version {
        names.push(format!("python{version}"));
        names.push(format!("python{version}.exe"));
    }
    names.into_iter().map(|name| home.join(name)).collect()
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

        assert_eq!(runs_from(std::process::id(), &tree, &[]), Some(true));
        assert_eq!(
            runs_from(std::process::id(), Path::new("/definitely/elsewhere"), &[]),
            Some(false)
        );
    }

    /// The reported venv case: uv symlinks bin/python at the base interpreter,
    /// so the image path leaves the tree entirely. Answering "not ours" there
    /// would rewrite the venv underneath a live backend.
    #[test]
    fn a_process_running_a_shared_interpreter_is_indeterminate() {
        let exe = std::env::current_exe().unwrap();

        assert_eq!(
            runs_from(
                std::process::id(),
                Path::new("/definitely/elsewhere"),
                std::slice::from_ref(&exe)
            ),
            None
        );
        // Some other program on the same pid stays a clear no.
        assert_eq!(
            runs_from(
                std::process::id(),
                Path::new("/definitely/elsewhere"),
                &[PathBuf::from("/usr/bin/some-other-python")]
            ),
            Some(false)
        );
    }

    /// uv's Windows trampoline spawns the base interpreter as a child, so the
    /// venv path is absent from the running image and pyvenv.cfg is the only
    /// thing tying the two together.
    #[test]
    fn the_base_interpreter_from_pyvenv_cfg_is_listed() {
        let tree = tempfile::tempdir().unwrap();
        let venv = tree.path().join("unsloth_studio");
        let base = tree.path().join("base");
        std::fs::create_dir_all(&venv).unwrap();
        std::fs::create_dir_all(&base).unwrap();
        let interpreter = base.join(if cfg!(windows) { "python.exe" } else { "python3.13" });
        std::fs::write(&interpreter, "").unwrap();
        std::fs::write(
            venv.join("pyvenv.cfg"),
            format!(
                "home = {}\nimplementation = CPython\nversion_info = 3.13.12\n",
                base.display()
            ),
        )
        .unwrap();

        assert_eq!(
            shared_interpreters_in(tree.path()),
            vec![std::fs::canonicalize(&interpreter).unwrap()]
        );
    }

    #[test]
    fn a_venv_without_a_config_lists_nothing_extra() {
        let tree = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tree.path().join("unsloth_studio")).unwrap();

        assert!(shared_interpreters_in(tree.path()).is_empty());
    }

    #[test]
    fn only_a_resolvable_interpreter_is_listed() {
        let tree = tempfile::tempdir().unwrap();
        let bin = tree.path().join("unsloth_studio").join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        let target = tree.path().join("base-python");
        std::fs::write(&target, "").unwrap();

        assert!(shared_interpreters_in(tree.path()).is_empty());

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&target, bin.join("python")).unwrap();

            assert_eq!(
                shared_interpreters_in(tree.path()),
                vec![std::fs::canonicalize(&target).unwrap()]
            );
        }
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
