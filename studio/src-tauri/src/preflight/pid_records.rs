use crate::process_identity::ProcessOrigin;
use std::path::Path;

/// The pid of a live backend of THIS install serving `port`, if recorded. A health probe cannot
/// tell a local backend from an SSH-forwarded one, so per-port and legacy pid records are
/// checked against the live process's start time and executable.
pub(super) fn live_backend_pid_on_port(port: u16) -> Option<u32> {
    let root = record_root();
    let interpreters = crate::process_identity::interpreters_of(&root);
    let probe = Probe {
        is_live: &|pid| {
            crate::desktop_backend_owner::pid_is_not_dead(pid)
                && !crate::process_identity::is_zombie(pid)
        },
        origin: &|pid| crate::process_identity::origin_of(pid, &root, &interpreters),
        started_at: &crate::process_identity::process_start_time_secs,
    };
    live_backend_pid_in(&root, port, &probe)
}

#[cfg(test)]
pub(super) static TEST_RECORD_ROOT: std::sync::Mutex<Option<std::path::PathBuf>> =
    std::sync::Mutex::new(None);

fn record_root() -> std::path::PathBuf {
    #[cfg(test)]
    if let Ok(guard) = TEST_RECORD_ROOT.lock() {
        if let Some(root) = guard.clone() {
            return root;
        }
    }
    crate::diagnostics::studio_dir()
}

struct Probe<'a> {
    /// False only when the pid is provably gone.
    is_live: &'a dyn Fn(u32) -> bool,
    origin: &'a dyn Fn(u32) -> ProcessOrigin,
    started_at: &'a dyn Fn(u32) -> Option<f64>,
}

enum Recorded {
    Stale,
    OurBackend,
    /// May be our backend or any other program sharing the venv's base interpreter.
    MaybeOurs,
    Foreign,
    Opaque,
}

const START_TIME_TOLERANCE_SECS: f64 =
    crate::process_identity::PID_START_TIME_TOLERANCE_SECS;

fn classify(pid: u32, recorded_start: Option<f64>, probe: &Probe) -> Recorded {
    if !(probe.is_live)(pid) {
        return Recorded::Stale;
    }
    // A start time mismatch proves pid reuse, which no executable evidence can override.
    if let (Some(recorded), Some(actual)) = (recorded_start, (probe.started_at)(pid)) {
        if (actual - recorded).abs() > START_TIME_TOLERANCE_SECS {
            return Recorded::Stale;
        }
    }
    match (probe.origin)(pid) {
        ProcessOrigin::InsideTree => Recorded::OurBackend,
        ProcessOrigin::SharedInterpreter => Recorded::MaybeOurs,
        ProcessOrigin::Elsewhere => Recorded::Foreign,
        ProcessOrigin::Unknown => Recorded::Opaque,
    }
}

fn live_backend_pid_in(root: &Path, port: u16, probe: &Probe) -> Option<u32> {
    if let Some(pid) = per_port_record_pid(root, port, probe) {
        return Some(pid);
    }
    legacy_record_pid(root, probe)
}

/// A record naming this port stands unless the pid provably belongs elsewhere.
fn per_port_record_pid(root: &Path, port: u16, probe: &Probe) -> Option<u32> {
    let prefix = format!("studio-{port}-");
    for entry in std::fs::read_dir(root).ok()?.flatten() {
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else {
            continue;
        };
        // The pid comes from the file name, which no partial write can garble.
        let Some(pid) = name
            .strip_prefix(&prefix)
            .and_then(|rest| rest.strip_suffix(".pid"))
            .and_then(|pid| pid.parse::<u32>().ok())
        else {
            continue;
        };
        match classify(pid, recorded_start_time(&entry.path()), probe) {
            Recorded::OurBackend | Recorded::MaybeOurs | Recorded::Opaque => return Some(pid),
            Recorded::Foreign | Recorded::Stale => continue,
        }
    }
    None
}

/// Legacy `studio.pid` has no port or start time: an opaque process is not enough to block
/// a repair, one possibly running our interpreter is.
fn legacy_record_pid(root: &Path, probe: &Probe) -> Option<u32> {
    let body = std::fs::read_to_string(root.join("studio.pid")).ok()?;
    // pid 0 and 1 are never a backend, and signalling either would be a bug.
    let pid = body
        .lines()
        .next()?
        .trim()
        .parse::<u32>()
        .ok()
        .filter(|pid| *pid > 1)?;
    // A pid with a per-port record was already judged under its port; otherwise our backend on an
    // ignored port answers for every port. Mirrors `_legacy_studio_on_port` in Python.
    if has_a_per_port_record(root, pid) {
        return None;
    }
    match classify(pid, None, probe) {
        Recorded::OurBackend | Recorded::MaybeOurs => Some(pid),
        Recorded::Foreign | Recorded::Opaque | Recorded::Stale => None,
    }
}

fn has_a_per_port_record(root: &Path, pid: u32) -> bool {
    let suffix = format!("-{pid}.pid");
    let Ok(entries) = std::fs::read_dir(root) else {
        return false;
    };
    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else {
            continue;
        };
        if name.starts_with("studio-") && name.ends_with(&suffix) {
            return true;
        }
    }
    false
}

fn recorded_start_time(path: &Path) -> Option<f64> {
    crate::process_identity::recorded_pid_start_time(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const RECORDED: u32 = 4242;
    const RECORDED_8888: &str = "studio-8888-4242.pid";

    struct Home {
        dir: tempfile::TempDir,
    }

    impl Home {
        fn new() -> Self {
            Self {
                dir: tempfile::tempdir().unwrap(),
            }
        }

        fn path(&self) -> PathBuf {
            self.dir.path().to_path_buf()
        }

        fn record(&self, name: &str, body: &str) -> &Self {
            std::fs::write(self.dir.path().join(name), body).unwrap();
            self
        }
    }

    fn probe(origin: &dyn Fn(u32) -> ProcessOrigin) -> Probe<'_> {
        Probe {
            is_live: &|_| true,
            origin,
            started_at: &|_| None,
        }
    }

    fn gone<'a>() -> Probe<'a> {
        Probe {
            is_live: &|_| false,
            origin: &|_| unreachable!("a dead pid is never attributed"),
            started_at: &|_| unreachable!("a dead pid is never timed"),
        }
    }

    fn ours(_pid: u32) -> ProcessOrigin {
        ProcessOrigin::InsideTree
    }

    fn shared(_pid: u32) -> ProcessOrigin {
        ProcessOrigin::SharedInterpreter
    }

    fn theirs(_pid: u32) -> ProcessOrigin {
        ProcessOrigin::Elsewhere
    }

    fn opaque(_pid: u32) -> ProcessOrigin {
        ProcessOrigin::Unknown
    }

    #[test]
    fn a_live_record_from_our_tree_is_found() {
        let home = Home::new();
        home.record(RECORDED_8888, "");

        assert_eq!(
            live_backend_pid_in(&home.path(), 8888, &probe(&ours)),
            Some(RECORDED)
        );
    }

    #[test]
    fn a_dead_record_is_ignored() {
        let home = Home::new();
        home.record(RECORDED_8888, "");

        assert_eq!(live_backend_pid_in(&home.path(), 8888, &gone()), None);
    }

    #[test]
    fn a_reused_pid_running_another_tree_is_ignored() {
        let home = Home::new();
        home.record(RECORDED_8888, "");

        assert_eq!(
            live_backend_pid_in(&home.path(), 8888, &probe(&theirs)),
            None
        );
    }

    #[test]
    fn a_recorded_start_time_that_disagrees_settles_a_reused_pid() {
        let home = Home::new();
        home.record(RECORDED_8888, "4242\n1000.0\n127.0.0.1");
        let reused = Probe {
            is_live: &|_| true,
            origin: &shared,
            started_at: &|_| Some(9999.0),
        };

        assert_eq!(live_backend_pid_in(&home.path(), 8888, &reused), None);
    }

    #[test]
    fn a_matching_start_time_leaves_the_record_standing() {
        let home = Home::new();
        home.record(RECORDED_8888, "4242\n1000.0\n127.0.0.1");
        let same = Probe {
            is_live: &|_| true,
            origin: &shared,
            started_at: &|_| Some(1000.4),
        };

        assert_eq!(
            live_backend_pid_in(&home.path(), 8888, &same),
            Some(RECORDED)
        );
    }

    /// Trusted like `_pid_is_studio_backend` on the Python side.
    #[test]
    fn a_record_without_a_start_time_is_not_second_guessed() {
        let home = Home::new();
        home.record(RECORDED_8888, "4242\n\n127.0.0.1");
        let timed = Probe {
            is_live: &|_| true,
            origin: &shared,
            started_at: &|_| Some(9999.0),
        };

        assert_eq!(
            live_backend_pid_in(&home.path(), 8888, &timed),
            Some(RECORDED)
        );
    }

    #[test]
    fn a_shared_interpreter_on_a_recorded_port_blocks() {
        let home = Home::new();
        home.record(RECORDED_8888, "");

        assert_eq!(
            live_backend_pid_in(&home.path(), 8888, &probe(&shared)),
            Some(RECORDED)
        );
    }

    #[test]
    fn a_per_port_record_we_cannot_attribute_still_blocks() {
        let home = Home::new();
        home.record(RECORDED_8888, "");

        assert_eq!(
            live_backend_pid_in(&home.path(), 8888, &probe(&opaque)),
            Some(RECORDED)
        );
    }

    #[test]
    fn a_record_for_another_port_is_ignored() {
        let home = Home::new();
        home.record("studio-8889-4242.pid", "");
        home.record("studio-88881-4242.pid", "");

        assert_eq!(live_backend_pid_in(&home.path(), 8888, &probe(&ours)), None);
    }

    #[test]
    fn a_live_legacy_record_from_our_tree_blocks() {
        let home = Home::new();
        home.record("studio.pid", &RECORDED.to_string());

        assert_eq!(
            live_backend_pid_in(&home.path(), 8888, &probe(&ours)),
            Some(RECORDED)
        );
    }

    #[test]
    fn a_legacy_record_on_a_shared_interpreter_blocks() {
        let home = Home::new();
        home.record("studio.pid", &RECORDED.to_string());

        assert_eq!(
            live_backend_pid_in(&home.path(), 8888, &probe(&shared)),
            Some(RECORDED)
        );
    }

    #[test]
    fn a_legacy_record_for_a_pid_that_serves_a_known_port_is_ignored() {
        let home = Home::new();
        home.record("studio.pid", &RECORDED.to_string());
        home.record("studio-9001-4242.pid", "");

        assert_eq!(live_backend_pid_in(&home.path(), 8888, &probe(&ours)), None);
        assert_eq!(
            live_backend_pid_in(&home.path(), 9001, &probe(&ours)),
            Some(RECORDED)
        );
    }

    #[test]
    fn a_contradicted_per_port_record_silences_the_legacy_one() {
        let home = Home::new();
        home.record("studio.pid", &RECORDED.to_string());
        home.record("studio-9001-4242.pid", "4242\n1000.0\n");
        let reused = Probe {
            is_live: &|_| true,
            origin: &ours,
            started_at: &|_| Some(9999.0),
        };

        assert_eq!(live_backend_pid_in(&home.path(), 8888, &reused), None);
    }

    #[test]
    fn a_legacy_record_from_another_tree_is_ignored() {
        let home = Home::new();
        home.record("studio.pid", &RECORDED.to_string());

        assert_eq!(
            live_backend_pid_in(&home.path(), 8888, &probe(&theirs)),
            None
        );
    }

    #[test]
    fn a_legacy_record_we_cannot_attribute_does_not_block() {
        let home = Home::new();
        home.record("studio.pid", &RECORDED.to_string());

        assert_eq!(
            live_backend_pid_in(&home.path(), 8888, &probe(&opaque)),
            None
        );
    }

    #[test]
    fn a_stale_legacy_record_is_ignored() {
        let home = Home::new();
        home.record("studio.pid", &RECORDED.to_string());

        assert_eq!(live_backend_pid_in(&home.path(), 8888, &gone()), None);
    }

    #[test]
    fn a_malformed_legacy_record_is_ignored() {
        let home = Home::new();
        home.record("studio.pid", "1");

        assert_eq!(live_backend_pid_in(&home.path(), 8888, &probe(&ours)), None);

        home.record("studio.pid", "not a pid");

        assert_eq!(live_backend_pid_in(&home.path(), 8888, &probe(&ours)), None);
    }

    #[test]
    fn nothing_recorded_means_nothing_local() {
        let home = Home::new();
        home.record("studio-8888-notapid.pid", "");

        assert_eq!(live_backend_pid_in(&home.path(), 8888, &probe(&ours)), None);
    }

    #[test]
    fn a_missing_studio_home_is_not_an_error() {
        let home = Home::new();

        assert_eq!(
            live_backend_pid_in(&home.path().join("absent"), 8888, &probe(&ours)),
            None
        );
    }
}

/// Real processes and files, driven through `live_backend_pid_on_port`.
#[cfg(test)]
mod system_tests {
    use super::*;
    use std::path::PathBuf;
    use std::process::{Child, Command};

    static ROOT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct RecordRoot {
        _guard: std::sync::MutexGuard<'static, ()>,
        dir: PathBuf,
        written: Vec<PathBuf>,
    }

    impl RecordRoot {
        fn at_our_own_tree() -> Self {
            let guard = ROOT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let dir = std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .to_path_buf();
            *TEST_RECORD_ROOT.lock().unwrap() = Some(dir.clone());
            Self {
                _guard: guard,
                dir,
                written: Vec::new(),
            }
        }

        fn record(&mut self, name: &str, body: &str) {
            let path = self.dir.join(name);
            std::fs::write(&path, body).unwrap();
            self.written.push(path);
        }
    }

    impl Drop for RecordRoot {
        fn drop(&mut self) {
            for path in &self.written {
                let _ = std::fs::remove_file(path);
            }
            *TEST_RECORD_ROOT.lock().unwrap() = None;
        }
    }

    fn spawn_foreign() -> Child {
        #[cfg(windows)]
        let mut command = {
            let mut c = Command::new("cmd.exe");
            c.args(["/c", "ping", "-n", "60", "127.0.0.1"]);
            c
        };
        #[cfg(not(windows))]
        let mut command = {
            let mut c = Command::new("sleep");
            c.arg("60");
            c
        };
        let child = command.spawn().expect("a foreign process should start");
        wait_for_exec(child.id());
        child
    }

    /// Wait for exec: posix_spawn returns while the child still looks like this test binary.
    fn wait_for_exec(pid: u32) {
        let Ok(own) = std::env::current_exe() else {
            return;
        };
        for _ in 0..400 {
            match crate::process_identity::executable_path(pid) {
                Some(exe) if exe == own => {}
                // Its own image, or the OS will not say; waiting changes neither.
                _ => return,
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    #[test]
    fn a_recorded_live_backend_of_this_install_is_found() {
        let mut root = RecordRoot::at_our_own_tree();
        let me = std::process::id();
        root.record(&format!("studio-8888-{me}.pid"), "");

        assert_eq!(live_backend_pid_on_port(8888), Some(me));
    }

    #[test]
    fn an_unrecorded_port_finds_nothing() {
        let _root = RecordRoot::at_our_own_tree();

        assert_eq!(live_backend_pid_on_port(8890), None);
    }

    #[test]
    fn a_record_for_a_process_that_exited_is_ignored() {
        let mut root = RecordRoot::at_our_own_tree();
        let mut child = spawn_foreign();
        let pid = child.id();
        child.kill().unwrap();
        child.wait().unwrap();
        root.record(&format!("studio-8891-{pid}.pid"), "");

        assert_eq!(live_backend_pid_on_port(8891), None);
    }

    #[test]
    fn a_record_pointing_at_a_foreign_live_process_is_ignored() {
        let mut root = RecordRoot::at_our_own_tree();
        let mut child = spawn_foreign();
        root.record(&format!("studio-8892-{}.pid", child.id()), "");

        let found = live_backend_pid_on_port(8892);
        child.kill().unwrap();
        child.wait().unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn a_record_whose_start_time_disagrees_is_ignored() {
        let mut root = RecordRoot::at_our_own_tree();
        let me = std::process::id();
        root.record(&format!("studio-8894-{me}.pid"), &format!("{me}\n1.0\n"));

        assert_eq!(live_backend_pid_on_port(8894), None);
    }

    #[test]
    fn a_record_whose_start_time_agrees_is_found() {
        let mut root = RecordRoot::at_our_own_tree();
        let me = std::process::id();
        let started = crate::process_identity::process_start_time_secs(me)
            .expect("this platform should report its own start time");
        root.record(
            &format!("studio-8895-{me}.pid"),
            &format!("{me}\n{started}\n"),
        );

        assert_eq!(live_backend_pid_on_port(8895), Some(me));
    }

    /// A crashed unreaped backend keeps a live pid but no socket; its record must not block.
    #[cfg(unix)]
    #[test]
    fn a_record_for_an_unreaped_process_is_ignored() {
        let mut root = RecordRoot::at_our_own_tree();
        let mut child = spawn_foreign();
        let pid = child.id();
        child.kill().unwrap();
        for _ in 0..200 {
            if crate::process_identity::is_zombie(pid) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        root.record(&format!("studio-8896-{pid}.pid"), "");

        let found = live_backend_pid_on_port(8896);
        child.wait().unwrap();

        assert_eq!(found, None);
    }

    #[test]
    fn a_legacy_record_naming_this_install_is_found() {
        let mut root = RecordRoot::at_our_own_tree();
        let me = std::process::id();
        root.record("studio.pid", &me.to_string());

        assert_eq!(live_backend_pid_on_port(8893), Some(me));
    }
}
