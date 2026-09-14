//! Which install tree a live process is running out of.

use std::path::{Path, PathBuf};

/// Absolute path of the executable behind `pid`, when the OS will say.
///
/// None means unknown, never "no such process": a pid belonging to another user
/// is refused on every platform here, and callers must not read that as absence.
pub(crate) fn executable_path(pid: u32) -> Option<PathBuf> {
    executable_path_impl(pid)
}

/// Where a live process is running from, relative to an install tree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProcessOrigin {
    /// Positively this install.
    InsideTree,
    /// Running an interpreter this install's venv defers to, so it may be ours
    /// and may be any other program using the same base interpreter.
    SharedInterpreter,
    /// Positively not this install.
    Elsewhere,
    /// The OS would not say.
    Unknown,
}

/// The interpreters a backend of an install may appear to be running.
pub(crate) struct TreeInterpreters {
    shared: Vec<PathBuf>,
    /// A venv is there but its `pyvenv.cfg` could not be read, so the base
    /// interpreter behind it is unknown. A damaged install is exactly the state
    /// that triggers a repair, so this must not read as "definitely not ours".
    base_unknown: bool,
}

/// Where `pid` is running from.
///
/// `interpreters` covers the executables a process inside the tree may be
/// running without the tree appearing anywhere in its image path. On unix uv
/// symlinks `bin/python` at the base interpreter, which `install.sh` documents,
/// and both `/proc/{pid}/exe` and `proc_pidpath` report the resolved target. On
/// Windows uv's `Scripts/python.exe` is a trampoline that spawns the base
/// interpreter as a child, so the image is the base one there too.
pub(crate) fn origin_of(pid: u32, tree: &Path, interpreters: &TreeInterpreters) -> ProcessOrigin {
    // argv[0] keeps the path as invoked, so it still names the venv where the
    // image path no longer does. Positive evidence only: a process can be given
    // any argv, and there is no platform where its absence means anything.
    if let Some(argv0) = first_argument(pid) {
        if path_is_within(&argv0, tree) {
            return ProcessOrigin::InsideTree;
        }
    }
    let Some(exe) = executable_path(pid) else {
        return ProcessOrigin::Unknown;
    };
    if path_is_within(&exe, tree) {
        return ProcessOrigin::InsideTree;
    }
    if interpreters.base_unknown
        || interpreters
            .shared
            .iter()
            .any(|shared| is_same_path(shared, &exe))
    {
        return ProcessOrigin::SharedInterpreter;
    }
    ProcessOrigin::Elsewhere
}

fn is_same_path(left: &Path, right: &Path) -> bool {
    // Simplified on both sides: canonicalize hands back an extended-length
    // \\?\C:\... path on Windows while QueryFullProcessImageNameW hands back a
    // plain one, and comparing those two forms never matches.
    simplified(left).to_string_lossy().to_lowercase()
        == simplified(right).to_string_lossy().to_lowercase()
}

/// A Windows extended-length path in its ordinary form. A no-op elsewhere.
fn simplified(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{rest}"));
    }
    match text.strip_prefix(r"\\?\") {
        Some(rest) => PathBuf::from(rest.to_string()),
        None => path.to_path_buf(),
    }
}

/// The command line's argv[0] for `pid`.
///
/// Linux only. macOS needs a KERN_PROCARGS2 sysctl and Windows reads the
/// remote PEB, neither of which is implemented. On those two a backend started
/// through a venv is therefore a shared-interpreter answer rather than a
/// positive one, which still blocks on a record naming the port.
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

/// Interpreters a backend of this install may be running through.
///
/// Both venv layouts, since `~/.unsloth/studio` is shared with the CLI
/// installer and an older install still has the `.venv` one. Canonicalized so
/// the comparison meets the same resolved path the OS reports; an entry that
/// does not resolve is dropped, since nothing can be running it.
pub(crate) fn interpreters_of(tree: &Path) -> TreeInterpreters {
    let mut shared: Vec<PathBuf> = Vec::new();
    let mut base_unknown = false;
    for name in ["unsloth_studio", ".venv"] {
        let venv = tree.join(name);
        if !venv.is_dir() {
            continue;
        }
        for (dir, exe) in [("bin", "python"), ("Scripts", "python.exe")] {
            if let Ok(target) = std::fs::canonicalize(venv.join(dir).join(exe)) {
                if !shared.contains(&target) {
                    shared.push(target);
                }
            }
        }
        // Unknown covers both "the config would not say" and "it named a base
        // interpreter that is not there": a uv venv whose base interpreter was
        // deleted or moved is a damaged install, which is the state a repair
        // runs in, and its backend is still running out of the old one.
        let bases = base_interpreters_of(&venv);
        let mut resolved_a_base = false;
        for base in bases {
            if let Ok(target) = std::fs::canonicalize(base) {
                resolved_a_base = true;
                if !shared.contains(&target) {
                    shared.push(target);
                }
            }
        }
        if !resolved_a_base {
            base_unknown = true;
        }
    }
    TreeInterpreters {
        shared,
        base_unknown,
    }
}

/// The base interpreters a venv defers to, from its `pyvenv.cfg`.
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

/// Whether `pid` has exited but has not been reaped.
///
/// A zombie holds its pid and answers `kill(pid, 0)`, so a liveness check alone
/// reads a crashed backend as running. It matters here because the desktop app
/// spawns the backend and does not wait on it, so a crash leaves a zombie for
/// as long as the app lives, and the record would block every repair until the
/// app exited. The socket is already released by then.
///
/// Windows has no equivalent: an exited process is signalled, which the
/// liveness check already reads as dead.
pub(crate) fn is_zombie(pid: u32) -> bool {
    is_zombie_impl(pid)
}

#[cfg(target_os = "linux")]
fn is_zombie_impl(pid: u32) -> bool {
    // Field 3 of /proc/{pid}/stat, read from the last ')' so a comm containing
    // spaces or parentheses cannot shift the offset.
    let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
        return false;
    };
    let Some(after_comm) = stat.rfind(')').map(|at| &stat[at + 1..]) else {
        return false;
    };
    after_comm.split_whitespace().next() == Some("Z")
}

/// Offsets into `struct kinfo_proc`, whose `kp_proc` is a `struct extern_proc`
/// starting at 0: `p_stat` follows the two pointers, the `p_un` union and
/// `p_flag`, and `p_pid` follows `p_stat`. libc does not declare `kinfo_proc`
/// for Apple, so the record is read as bytes. Measured on macos-14 (arm64):
/// `offsetof(kinfo_proc, kp_proc.p_stat) == 36` and the whole record is 648
/// bytes. `p_pid` is read back as a check that the layout still holds.
#[cfg(target_os = "macos")]
const KINFO_PROC_P_STAT_OFFSET: usize = 36;
#[cfg(target_os = "macos")]
const KINFO_PROC_P_PID_OFFSET: usize = 40;

/// The `p_stat` value for a process that has exited and not been reaped,
/// from sys/proc.h, which libc does not re-export.
#[cfg(target_os = "macos")]
const SZOMB: u8 = 5;

#[cfg(target_os = "macos")]
fn is_zombie_impl(pid: u32) -> bool {
    if pid > i32::MAX as u32 {
        return false;
    }
    // sysctl, not proc_pidinfo. Measured on macos-14: for a killed but unreaped
    // child BOTH proc_pidinfo flavors fail with ESRCH, the short one included,
    // so neither can ever answer yes. KERN_PROC_PID reported p_stat = SZOMB for
    // that same pid, SRUN while it was running, and no record at all once it
    // had been reaped.
    match kern_proc_record(pid) {
        Some(record) => record[KINFO_PROC_P_STAT_OFFSET] == SZOMB,
        // The kernel answered in a shape this does not recognise. Rather than
        // report every process alive, which is the bug this replaces, fall back
        // to the behaviour the same measurement showed: a pid that still exists
        // but that proc_pidinfo reports as ESRCH is a zombie.
        None => pid_exists(pid) && !proc_pidinfo_sees(pid),
    }
}

/// The KERN_PROC_PID record for `pid`, once it is long enough to read `p_stat`
/// out of and its own `p_pid` field agrees with the pid asked for.
#[cfg(target_os = "macos")]
fn kern_proc_record(pid: u32) -> Option<Vec<u8>> {
    let mut mib: [libc::c_int; 4] = [
        libc::CTL_KERN,
        libc::KERN_PROC,
        libc::KERN_PROC_PID,
        pid as libc::c_int,
    ];
    // The kernel's own size for the record, so nothing here has to hardcode it.
    // A short buffer is not filled in part: it fails with ENOMEM.
    let mut size: libc::size_t = 0;
    let sized = unsafe {
        libc::sysctl(
            mib.as_mut_ptr(),
            mib.len() as libc::c_uint,
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if sized != 0 || size <= KINFO_PROC_P_PID_OFFSET + 4 {
        return None;
    }
    let mut record = vec![0u8; size];
    let read = unsafe {
        libc::sysctl(
            mib.as_mut_ptr(),
            mib.len() as libc::c_uint,
            record.as_mut_ptr() as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    // A reaped pid answers with no error and a zero-length record.
    if read != 0 || size <= KINFO_PROC_P_PID_OFFSET + 4 {
        return None;
    }
    record.truncate(size);
    let recorded_pid = u32::from_ne_bytes(
        record[KINFO_PROC_P_PID_OFFSET..KINFO_PROC_P_PID_OFFSET + 4]
            .try_into()
            .ok()?,
    );
    (recorded_pid == pid).then_some(record)
}

/// Whether the OS still has this pid, zombie or not. EPERM is somebody else's
/// live process, which is an answer too.
#[cfg(target_os = "macos")]
fn pid_exists(pid: u32) -> bool {
    if unsafe { libc::kill(pid as i32, 0) } == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// Whether proc_pidinfo will describe this pid. The short flavor is the one
/// that does not require the same uid, so a no here is about the process being
/// gone rather than about who owns it.
#[cfg(target_os = "macos")]
fn proc_pidinfo_sees(pid: u32) -> bool {
    let mut info: libc::proc_bsdshortinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_bsdshortinfo>() as libc::c_int;
    let written = unsafe {
        libc::proc_pidinfo(
            pid as i32,
            libc::PROC_PIDT_SHORTBSDINFO,
            0,
            &mut info as *mut _ as *mut libc::c_void,
            size,
        )
    };
    written == size
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn is_zombie_impl(_pid: u32) -> bool {
    false
}

/// Unix epoch seconds at which `pid` started, when the OS will say.
///
/// Compared against the start time the server recorded next to its pid, which
/// is what tells a live backend apart from an unrelated process that inherited
/// its pid after a crash. `psutil.Process.create_time()` writes the same clock
/// on the Python side.
pub(crate) fn process_start_time_secs(pid: u32) -> Option<f64> {
    process_start_time_impl(pid)
}

/// The Python side uses the same one-second window in `_pid_is_studio_backend`.
pub(crate) const PID_START_TIME_TOLERANCE_SECS: f64 = 1.0;

/// Line two of a `studio-{port}-{pid}.pid` record, written by the server as
/// `psutil` epoch seconds. A blank or absent line means it could not say.
pub(crate) fn recorded_pid_start_time(path: &Path) -> Option<f64> {
    std::fs::read_to_string(path)
        .ok()?
        .lines()
        .nth(1)?
        .trim()
        .parse()
        .ok()
}

/// Whether the process wearing `pid` is the one the record described.
///
/// A recorded start time that disagrees is proof the pid was reused, which no
/// amount of executable evidence can override. Records outlive crashes, so
/// without this an unrelated process inheriting the pid reads as ours.
pub(crate) fn pid_start_time_matches(pid: u32, recorded: Option<f64>) -> bool {
    let (Some(recorded), Some(actual)) = (recorded, process_start_time_secs(pid)) else {
        return true;
    };
    (actual - recorded).abs() <= PID_START_TIME_TOLERANCE_SECS
}

#[cfg(target_os = "linux")]
fn process_start_time_impl(pid: u32) -> Option<f64> {
    // Field 22 of /proc/{pid}/stat, in clock ticks since boot. Parsed from the
    // last ')' onwards because field 2 is the comm, which may itself contain
    // spaces and parentheses.
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let after_comm = &stat[stat.rfind(')')? + 1..];
    let ticks: f64 = after_comm.split_whitespace().nth(19)?.parse().ok()?;
    let hz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    if hz <= 0 {
        return None;
    }
    Some(boot_time_secs()? + ticks / hz as f64)
}

#[cfg(target_os = "linux")]
fn boot_time_secs() -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/stat").ok()?;
    stat.lines()
        .find_map(|line| line.strip_prefix("btime "))
        .and_then(|value| value.trim().parse::<f64>().ok())
}

#[cfg(target_os = "macos")]
fn process_start_time_impl(pid: u32) -> Option<f64> {
    if pid > i32::MAX as u32 {
        return None;
    }
    let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_bsdinfo>() as libc::c_int;
    let written = unsafe {
        libc::proc_pidinfo(
            pid as i32,
            libc::PROC_PIDTBSDINFO,
            0,
            &mut info as *mut _ as *mut libc::c_void,
            size,
        )
    };
    if written != size {
        return None;
    }
    Some(info.pbi_start_tvsec as f64 + info.pbi_start_tvusec as f64 / 1_000_000.0)
}

#[cfg(windows)]
fn process_start_time_impl(pid: u32) -> Option<f64> {
    use windows_sys::Win32::Foundation::{CloseHandle, FILETIME};
    use windows_sys::Win32::System::Threading::{
        GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    };

    // 100ns intervals between 1601-01-01 and 1970-01-01.
    const EPOCH_OFFSET: u64 = 116_444_736_000_000_000;
    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if handle.is_null() {
            return None;
        }
        let mut created = FILETIME {
            dwLowDateTime: 0,
            dwHighDateTime: 0,
        };
        let mut ignored = created;
        let ok = GetProcessTimes(
            handle,
            &mut created,
            &mut ignored,
            &mut ignored,
            &mut ignored,
        );
        let _ = CloseHandle(handle);
        if ok == 0 {
            return None;
        }
        let ticks =
            ((created.dwHighDateTime as u64) << 32) | created.dwLowDateTime as u64;
        Some(ticks.checked_sub(EPOCH_OFFSET)? as f64 / 10_000_000.0)
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn process_start_time_impl(_pid: u32) -> Option<f64> {
    None
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
        // processes at a higher integrity level, which an Unsloth started from an
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

    fn tree_with_venv(base: Option<&Path>, layout: &str) -> tempfile::TempDir {
        let tree = tempfile::tempdir().unwrap();
        let venv = tree.path().join("unsloth_studio");
        std::fs::create_dir_all(venv.join(layout)).unwrap();
        if let Some(base) = base {
            std::fs::write(
                venv.join("pyvenv.cfg"),
                format!("home = {}\nversion_info = 3.13.12\n", base.display()),
            )
            .unwrap();
        }
        tree
    }

    #[test]
    fn our_own_executable_is_resolvable() {
        let pid = std::process::id();
        let exe = executable_path(pid).expect("this platform should resolve its own path");

        assert_eq!(exe, std::env::current_exe().unwrap());
    }

    #[test]
    fn this_platform_reports_its_own_start_time() {
        let started =
            process_start_time_secs(std::process::id()).expect("a start time should be readable");
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        // Started in the past, and this millennium: a wrong epoch base or a
        // wrong clock tick divisor would land far outside that.
        assert!(started <= now + 1.0, "start {started} is after now {now}");
        assert!(started > 1_000_000_000.0, "start {started} is not an epoch");
    }

    #[test]
    fn a_process_is_inside_the_tree_that_contains_it() {
        let exe = std::env::current_exe().unwrap();
        let tree = exe.parent().unwrap().to_path_buf();
        let none = TreeInterpreters {
            shared: Vec::new(),
            base_unknown: false,
        };

        assert_eq!(
            origin_of(std::process::id(), &tree, &none),
            ProcessOrigin::InsideTree
        );
        assert_eq!(
            origin_of(std::process::id(), Path::new("/definitely/elsewhere"), &none),
            ProcessOrigin::Elsewhere
        );
    }

    /// The reported venv case: uv symlinks bin/python at the base interpreter,
    /// so the image path leaves the tree entirely. Reading that as "not ours"
    /// would rewrite the venv underneath a live backend.
    #[test]
    fn a_process_running_a_shared_interpreter_is_not_read_as_elsewhere() {
        let exe = std::env::current_exe().unwrap();
        let shared = TreeInterpreters {
            shared: vec![exe],
            base_unknown: false,
        };

        assert_eq!(
            origin_of(
                std::process::id(),
                Path::new("/definitely/elsewhere"),
                &shared
            ),
            ProcessOrigin::SharedInterpreter
        );
    }

    /// A damaged install is exactly the state a repair runs in, so an
    /// unreadable pyvenv.cfg must not read as "definitely not ours" on the
    /// platforms that have no argv[0] to fall back on.
    #[test]
    fn an_unknown_base_interpreter_is_not_read_as_elsewhere() {
        let unknown = TreeInterpreters {
            shared: Vec::new(),
            base_unknown: true,
        };

        assert_eq!(
            origin_of(
                std::process::id(),
                Path::new("/definitely/elsewhere"),
                &unknown
            ),
            ProcessOrigin::SharedInterpreter
        );
    }

    #[test]
    fn a_venv_whose_config_cannot_be_read_flags_its_base_as_unknown() {
        let tree = tree_with_venv(None, "bin");

        assert!(interpreters_of(tree.path()).base_unknown);
    }

    /// uv's Windows trampoline spawns the base interpreter as a child, so the
    /// venv path is absent from the running image and pyvenv.cfg is the only
    /// thing tying the two together.
    #[test]
    fn the_base_interpreter_from_pyvenv_cfg_is_listed() {
        let base = tempfile::tempdir().unwrap();
        let interpreter = base
            .path()
            .join(if cfg!(windows) { "python.exe" } else { "python3.13" });
        std::fs::write(&interpreter, "").unwrap();
        let tree = tree_with_venv(Some(base.path()), "Scripts");

        let found = interpreters_of(tree.path());

        assert!(!found.base_unknown);
        assert_eq!(found.shared, vec![std::fs::canonicalize(&interpreter).unwrap()]);
    }

    /// A uv venv whose base interpreter was deleted or moved: the config still
    /// names it, so the candidates are built, but none of them resolve. That is
    /// a damaged install, which is exactly when a repair runs.
    #[test]
    fn a_base_interpreter_that_no_longer_exists_is_flagged_unknown() {
        let missing = tempfile::tempdir().unwrap();
        let base = missing.path().join("gone");
        let tree = tree_with_venv(Some(&base), "bin");

        let found = interpreters_of(tree.path());

        assert!(found.base_unknown);
    }

    #[test]
    fn a_tree_with_no_venv_at_all_is_not_flagged_unknown() {
        let tree = tempfile::tempdir().unwrap();

        let found = interpreters_of(tree.path());

        assert!(!found.base_unknown);
        assert!(found.shared.is_empty());
    }

    /// A crashed backend the desktop app has not reaped still answers
    /// kill(pid, 0), and its record would block every repair until the app
    /// exited. Reproduced with a real unreaped child.
    #[cfg(unix)]
    #[test]
    fn a_zombie_is_not_a_live_process() {
        let mut child = std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .unwrap();
        let pid = child.id();

        assert!(!is_zombie(pid), "a running child is not a zombie");

        child.kill().unwrap();
        // Deliberately not reaped yet: Child::wait is what clears the zombie,
        // and leaving it is the state under test.
        let mut zombie = false;
        for _ in 0..200 {
            if is_zombie(pid) {
                zombie = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        child.wait().unwrap();

        assert!(zombie, "a killed but unreaped child should read as a zombie");
        assert!(!is_zombie(std::process::id()), "we are not a zombie");
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

    /// canonicalize hands back an extended-length path on Windows while
    /// QueryFullProcessImageNameW hands back a plain one, and the same file
    /// must compare equal across the two forms.
    #[test]
    fn an_extended_length_path_equals_its_plain_form() {
        assert!(is_same_path(
            Path::new(r"\\?\C:\Users\u\.unsloth\studio\python.exe"),
            Path::new(r"C:\Users\U\.unsloth\studio\python.exe"),
        ));
        assert!(is_same_path(
            Path::new(r"\\?\UNC\server\share\python.exe"),
            Path::new(r"\\server\share\python.exe"),
        ));
        assert!(!is_same_path(
            Path::new(r"\\?\C:\Users\u\python.exe"),
            Path::new(r"C:\Users\u\other.exe"),
        ));
    }
}
