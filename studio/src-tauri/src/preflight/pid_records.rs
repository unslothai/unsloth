use crate::process_identity::ProcessOrigin;
use std::path::Path;

/// The pid of a live backend of THIS install that is serving `port`, if one is
/// recorded.
///
/// A mutation rewrites the install tree, so the question it has to answer about
/// an unattributable backend is not "is something listening" but "is a live
/// process running out of the tree I am about to overwrite". A health probe
/// cannot answer that: an Unsloth reached through an SSH forward answers from
/// 127.0.0.1 exactly like a local one.
///
/// The server records itself on bind, in a per-port `studio-{port}-{pid}.pid`
/// file and, for older builds, in a bare `studio.pid`. Those records are hints
/// only: they outlive crashes, the per-port write is best effort, and the OS
/// reuses pids. So each recorded pid is checked against the process actually
/// wearing it, by start time and by what it is running.
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

/// Where the server writes its records. Overridable in tests so the end to end
/// case can be driven without touching the real studio home.
fn record_root() -> std::path::PathBuf {
    #[cfg(test)]
    if let Ok(guard) = TEST_RECORD_ROOT.lock() {
        if let Some(root) = guard.clone() {
            return root;
        }
    }
    crate::diagnostics::studio_dir()
}

/// What the OS is asked about a recorded pid, injected so the decision below
/// can be tested without real processes to point it at.
struct Probe<'a> {
    /// False only when the pid is provably gone.
    is_live: &'a dyn Fn(u32) -> bool,
    origin: &'a dyn Fn(u32) -> ProcessOrigin,
    /// Unix epoch seconds the process started, when the OS will say.
    started_at: &'a dyn Fn(u32) -> Option<f64>,
}

/// A record and the process wearing its pid, once the two have been compared.
enum Recorded {
    /// The pid is gone, or belongs to a process that started at a different
    /// time, so the record is left over from a crash.
    Stale,
    /// Live, and running out of this install.
    OurBackend,
    /// Live, and running an interpreter this install's venv defers to. It may
    /// be our backend and may be any other program sharing that interpreter.
    MaybeOurs,
    /// Live, and running something else entirely.
    Foreign,
    /// Live, and the OS would not say what it is running.
    Opaque,
}

/// The Python side uses the same one-second window in `_pid_is_studio_backend`.
const START_TIME_TOLERANCE_SECS: f64 = 1.0;

fn classify(pid: u32, recorded_start: Option<f64>, probe: &Probe) -> Recorded {
    if !(probe.is_live)(pid) {
        return Recorded::Stale;
    }
    // A recorded start time that disagrees is proof the pid was reused, which
    // no amount of executable evidence can override: the process the record
    // described is gone.
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

/// A record naming this exact port is strong evidence on its own, so anything
/// short of proof that the pid belongs elsewhere leaves it standing.
fn per_port_record_pid(root: &Path, port: u16, probe: &Probe) -> Option<u32> {
    let prefix = format!("studio-{port}-");
    for entry in std::fs::read_dir(root).ok()?.flatten() {
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else {
            continue;
        };
        // The pid comes from the name, not the body: the name is what the
        // binding process wrote about itself, and no partial write can garble
        // it. The body is read only for the start time.
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

/// The legacy record names no port, and a build old enough to be the id-less
/// backend we are asking about is exactly the build that writes only this file.
/// It is also the sole record when a per-port write failed.
///
/// It carries no start time, so a reused pid cannot be ruled out here the way
/// it can above. A process the OS will not describe is therefore not enough to
/// strand a repair on, while one that could be running our own interpreter is.
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
    // A current build writes a per-port record too, so if this pid has one it
    // has already been considered under the port it actually serves, and the
    // portless record must not be read as claiming this one. Without that, the
    // app's own backend on an ignored port answers for every other candidate
    // port and the mutation never reaches the step that stops it.
    // `_legacy_studio_on_port` skips the same case on the Python side.
    if has_a_per_port_record(root, pid) {
        return None;
    }
    match classify(pid, None, probe) {
        Recorded::OurBackend | Recorded::MaybeOurs => Some(pid),
        Recorded::Foreign | Recorded::Opaque | Recorded::Stale => None,
    }
}

/// Whether `pid` is named by a per-port record, on any port.
///
/// Either answer settles the legacy record: a record that still stands has
/// already been judged under the port it names, and one that no longer stands
/// says the pid was reused, which the untimed legacy record cannot see.
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

/// Line two of a record, written by the server as `psutil` epoch seconds. A
/// blank or absent line means the server could not determine it.
fn recorded_start_time(path: &Path) -> Option<f64> {
    std::fs::read_to_string(path)
        .ok()?
        .lines()
        .nth(1)?
        .trim()
        .parse()
        .ok()
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

    /// These pile up: a crashed server never removes its own record.
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

    /// The pid was reused by a process sharing our base interpreter, so the
    /// executable proves nothing. The recorded start time still does.
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

    /// An untimed record cannot be checked, so it is trusted, exactly as
    /// `_pid_is_studio_backend` trusts one on the Python side.
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

    /// A backend behind a symlinked venv or a Windows trampoline cannot be
    /// positively attributed, and a record naming its port must still block.
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
        // A port that merely starts with the digits of ours is another port.
        home.record("studio-88881-4242.pid", "");

        assert_eq!(live_backend_pid_in(&home.path(), 8888, &probe(&ours)), None);
    }

    /// A pre-upgrade server records itself here and nowhere else, and it is
    /// precisely the kind of build that reports no install id.
    #[test]
    fn a_live_legacy_record_from_our_tree_blocks() {
        let home = Home::new();
        home.record("studio.pid", &RECORDED.to_string());

        assert_eq!(
            live_backend_pid_in(&home.path(), 8888, &probe(&ours)),
            Some(RECORDED)
        );
    }

    /// On macOS a backend behind the symlinked venv is never more than
    /// "maybe", and a legacy record is all a pre-upgrade one leaves.
    #[test]
    fn a_legacy_record_on_a_shared_interpreter_blocks() {
        let home = Home::new();
        home.record("studio.pid", &RECORDED.to_string());

        assert_eq!(
            live_backend_pid_in(&home.path(), 8888, &probe(&shared)),
            Some(RECORDED)
        );
    }

    /// The app's own backend on an ignored port writes both records. The
    /// portless one must not then answer for every other candidate port, or a
    /// mutation never reaches the step that stops that backend.
    #[test]
    fn a_legacy_record_for_a_pid_that_serves_a_known_port_is_ignored() {
        let home = Home::new();
        home.record("studio.pid", &RECORDED.to_string());
        home.record("studio-9001-4242.pid", "");

        assert_eq!(live_backend_pid_in(&home.path(), 8888, &probe(&ours)), None);
        // ...while the port it does serve still blocks.
        assert_eq!(
            live_backend_pid_in(&home.path(), 9001, &probe(&ours)),
            Some(RECORDED)
        );
    }

    /// A per-port record that no longer stands settles the legacy one too: it
    /// says the pid was reused, which an untimed record cannot see for itself.
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

    /// The legacy record names no port and carries no start time, so a process
    /// the OS will not describe is not evidence enough to strand a repair.
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

    /// The reported case: the backend on the port is reached over an SSH
    /// forward, so this install recorded nothing anywhere.
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

/// Real processes and real files, driven through `live_backend_pid_on_port`
/// rather than the injected probe above: these are the cases the injected tests
/// cannot vouch for, because the thing under test is what the OS reports.
#[cfg(test)]
mod system_tests {
    use super::*;
    use std::path::PathBuf;
    use std::process::{Child, Command};

    /// Serialises the record-root override, which is process-wide.
    static ROOT_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct RecordRoot {
        _guard: std::sync::MutexGuard<'static, ()>,
        dir: PathBuf,
        written: Vec<PathBuf>,
    }

    impl RecordRoot {
        /// Rooted at our own executable's directory, so this process attributes
        /// as a backend of that tree exactly the way a real one does.
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

    /// A live process that is definitely not running out of our tree.
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

    /// Block until the spawned child is running its own image.
    ///
    /// `Command::spawn` takes the posix_spawn path here, which returns as soon
    /// as the child exists rather than when it has exec'd. In that window the
    /// child is still a copy of this test binary, so its executable path is
    /// inside the record root and it reads as one of ours. Rare, but it turned
    /// a foreign-process test red on a loaded CI machine.
    fn wait_for_exec(pid: u32) {
        let Ok(own) = std::env::current_exe() else {
            return;
        };
        for _ in 0..400 {
            match crate::process_identity::executable_path(pid) {
                Some(exe) if exe == own => {}
                // Its own image, or the OS will not say, which no amount of
                // waiting changes.
                _ => return,
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    /// The reported case reproduced end to end: a backend of this install is
    /// recorded on the port, and the repair has to refuse.
    #[test]
    fn a_recorded_live_backend_of_this_install_is_found() {
        let mut root = RecordRoot::at_our_own_tree();
        let me = std::process::id();
        root.record(&format!("studio-8888-{me}.pid"), "");

        assert_eq!(live_backend_pid_on_port(8888), Some(me));
    }

    /// The other half of it: the port answers, but nothing local claims it, so
    /// the repair proceeds. This is the SSH forward from the report.
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

    /// A live process that is not a backend of ours must not veto a repair,
    /// which is what a stale record with a reused pid looks like.
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

    /// A record whose start time names a different process settles it even
    /// when the executable cannot.
    #[test]
    fn a_record_whose_start_time_disagrees_is_ignored() {
        let mut root = RecordRoot::at_our_own_tree();
        let me = std::process::id();
        root.record(&format!("studio-8894-{me}.pid"), &format!("{me}\n1.0\n"));

        assert_eq!(live_backend_pid_on_port(8894), None);
    }

    /// ...and one that matches is left standing. The recorded time is read
    /// from the OS rather than hardcoded, which also checks the two agree.
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

    /// A crashed backend the app has not reaped keeps its pid and answers a
    /// liveness check, but its socket is gone. Its record must not block.
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
