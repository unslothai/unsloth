//! Which install tree a live process is running out of.

use std::path::{Path, PathBuf};

/// None means unknown, never "no such process": another user's pid is refused everywhere.
pub(crate) fn executable_path(pid: u32) -> Option<PathBuf> {
    executable_path_impl(pid)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ProcessOrigin {
    InsideTree,
    /// Running the venv's base interpreter: may be ours or any other program using it.
    SharedInterpreter,
    Elsewhere,
    Unknown,
}

pub(crate) struct TreeInterpreters {
    shared: Vec<PathBuf>,
    /// An unreadable `pyvenv.cfg` is a damaged install, which must not read as "not ours".
    base_unknown: bool,
}

/// `interpreters` covers images outside the tree: uv symlinks `bin/python` to the base on unix,
/// and on Windows `Scripts/python.exe` is a trampoline that spawns the base interpreter.
pub(crate) fn origin_of(pid: u32, tree: &Path, interpreters: &TreeInterpreters) -> ProcessOrigin {
    // argv[0] keeps the path as invoked. Positive evidence only: argv can be anything.
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
    // canonicalize returns `\\?\C:\...` while QueryFullProcessImageNameW returns a plain path.
    simplified(left).to_string_lossy().to_lowercase()
        == simplified(right).to_string_lossy().to_lowercase()
}

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

/// Linux only; macOS (KERN_PROCARGS2) and Windows (remote PEB) are not implemented, so they get
/// a shared-interpreter answer, which still blocks on a port record.
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

/// Both venv layouts (the CLI installer shares the root), canonicalized; unresolvable entries dropped.
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
        // Unknown also covers a base interpreter that was deleted or moved: a damaged install.
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
            // The documented key: the directory holding the interpreter this venv was built from.
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

/// A zombie answers `kill(pid, 0)`, and the app never waits on its backend, so a crashed one
/// would block repairs. Windows has no equivalent.
pub(crate) fn is_zombie(pid: u32) -> bool {
    is_zombie_impl(pid)
}

#[cfg(target_os = "linux")]
fn is_zombie_impl(pid: u32) -> bool {
    // Field 3, read from the last ')' so a comm with spaces or parentheses cannot shift it.
    let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) else {
        return false;
    };
    let Some(after_comm) = stat.rfind(')').map(|at| &stat[at + 1..]) else {
        return false;
    };
    after_comm.split_whitespace().next() == Some("Z")
}

/// Offsets into `struct kinfo_proc` (libc lacks it on Apple), measured on macos-14 arm64:
/// p_stat at 36, record 648 bytes. `p_pid` is read back to check the layout still holds.
#[cfg(target_os = "macos")]
const KINFO_PROC_P_STAT_OFFSET: usize = 36;
#[cfg(target_os = "macos")]
const KINFO_PROC_P_PID_OFFSET: usize = 40;

/// SZOMB from sys/proc.h, which libc does not re-export.
#[cfg(target_os = "macos")]
const SZOMB: u8 = 5;

#[cfg(target_os = "macos")]
fn is_zombie_impl(pid: u32) -> bool {
    if pid > i32::MAX as u32 {
        return false;
    }
    // sysctl, not proc_pidinfo: both proc_pidinfo flavors fail with ESRCH for a zombie.
    match kern_proc_record(pid) {
        Some(record) => record[KINFO_PROC_P_STAT_OFFSET] == SZOMB,
        // Unrecognised record: a pid that exists but proc_pidinfo reports as ESRCH is a zombie.
        None => pid_exists(pid) && !proc_pidinfo_sees(pid),
    }
}

#[cfg(target_os = "macos")]
fn kern_proc_record(pid: u32) -> Option<Vec<u8>> {
    let mut mib: [libc::c_int; 4] = [
        libc::CTL_KERN,
        libc::KERN_PROC,
        libc::KERN_PROC_PID,
        pid as libc::c_int,
    ];
    // The kernel's own size for the record. A short buffer fails with ENOMEM rather than filling partly.
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

/// EPERM is somebody else's live process, which is an answer too.
#[cfg(target_os = "macos")]
fn pid_exists(pid: u32) -> bool {
    if unsafe { libc::kill(pid as i32, 0) } == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// The short flavor does not require the same uid, so a no means the process is gone.
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

/// Compared with the start time the server recorded (psutil create_time, same clock) to detect
/// pid reuse after a crash.
pub(crate) fn process_start_time_secs(pid: u32) -> Option<f64> {
    process_start_time_impl(pid)
}

/// Same one-second window as `_pid_is_studio_backend` in Python.
pub(crate) const PID_START_TIME_TOLERANCE_SECS: f64 = 1.0;

/// Line two of a `studio-{port}-{pid}.pid` record (psutil epoch seconds); blank means unknown.
pub(crate) fn recorded_pid_start_time(path: &Path) -> Option<f64> {
    std::fs::read_to_string(path)
        .ok()?
        .lines()
        .nth(1)?
        .trim()
        .parse()
        .ok()
}

/// A disagreeing start time proves pid reuse, which no executable evidence can override.
pub(crate) fn pid_start_time_matches(pid: u32, recorded: Option<f64>) -> bool {
    let (Some(recorded), Some(actual)) = (recorded, process_start_time_secs(pid)) else {
        return true;
    };
    (actual - recorded).abs() <= PID_START_TIME_TOLERANCE_SECS
}

#[cfg(target_os = "linux")]
fn process_start_time_impl(pid: u32) -> Option<f64> {
    // Field 22, clock ticks since boot. Parsed from the last ')' since comm may contain them.
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
    // Case-insensitive (Windows) and component-wise, so a sibling with a shared prefix cannot match.
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
        // The limited right is granted even for elevated (higher integrity) processes.
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

        // A wrong epoch base or tick divisor would land far outside this range.
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

    /// uv symlinks bin/python at the base interpreter, so the image path leaves the tree.
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
        // Deliberately not reaped yet: that is the state under test.
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
