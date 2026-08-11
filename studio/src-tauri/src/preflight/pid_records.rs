use std::path::Path;

/// The pid of a live backend running out of THIS install, if one is recorded.
///
/// A mutation rewrites the install tree, so the question it has to answer about
/// an unattributable backend is not "is something listening" but "is a live
/// process running out of the tree I am about to overwrite". A health probe
/// cannot answer that: a Studio reached through an SSH forward answers from
/// 127.0.0.1 exactly like a local one.
///
/// Two things are combined to answer it. The server records itself on bind, in
/// a per-port `studio-{port}-{pid}.pid` file and, for older builds, in a bare
/// `studio.pid`. Those records are hints only: they outlive crashes, the
/// per-port write is best effort, and the OS reuses pids. So each recorded pid
/// is then attributed by the executable it is actually running, which is the
/// part that decides.
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

pub(super) fn live_backend_pid_on_port(port: u16) -> Option<u32> {
    let root = record_root();
    let shared = crate::process_identity::shared_interpreters_in(&root);
    let probe = Probe {
        is_live: &|pid| crate::desktop_backend_owner::pid_is_not_dead(pid),
        runs_from_tree: &|pid| crate::process_identity::runs_from(pid, &root, &shared),
    };
    live_backend_pid_in(&root, port, &probe)
}

/// What the OS is asked about a recorded pid, injected so the decision below
/// can be tested without real processes to point it at.
struct Probe<'a> {
    /// False only when the pid is provably gone.
    is_live: &'a dyn Fn(u32) -> bool,
    /// None when the executable could not be read.
    runs_from_tree: &'a dyn Fn(u32) -> Option<bool>,
}

/// Whether a recorded pid still runs, and if so out of our tree.
///
/// `Ok(None)` is "running, but we could not read its executable": another
/// user's process, or a platform with no implementation. It is deliberately
/// distinct from `Ok(Some(false))`, because the two get different answers below.
type Attribution = Result<Option<bool>, Dead>;

struct Dead;

fn attribute(pid: u32, probe: &Probe) -> Attribution {
    if !(probe.is_live)(pid) {
        return Err(Dead);
    }
    Ok((probe.runs_from_tree)(pid))
}

fn live_backend_pid_in(root: &Path, port: u16, probe: &Probe) -> Option<u32> {
    if let Some(pid) = per_port_record_pid(root, port, probe) {
        return Some(pid);
    }
    legacy_record_pid(root, probe)
}

/// A record naming this exact port is strong evidence on its own, so an
/// executable we cannot read leaves it standing. Only positive proof that the
/// process belongs to some other tree clears it.
fn per_port_record_pid(root: &Path, port: u16, probe: &Probe) -> Option<u32> {
    let prefix = format!("studio-{port}-");
    for entry in std::fs::read_dir(root).ok()?.flatten() {
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else {
            continue;
        };
        // Read from the name, not the body: the name is what the binding
        // process wrote about itself, and no partial write can garble it.
        let Some(pid) = name
            .strip_prefix(&prefix)
            .and_then(|rest| rest.strip_suffix(".pid"))
            .and_then(|pid| pid.parse::<u32>().ok())
        else {
            continue;
        };
        match attribute(pid, probe) {
            Ok(Some(false)) | Err(Dead) => continue,
            Ok(_) => return Some(pid),
        }
    }
    None
}

/// The legacy record names no port, and a build old enough to be the id-less
/// backend we are asking about is exactly the build that writes only this file.
/// It is also the sole record when a per-port write failed. Since it cannot be
/// tied to the port, it blocks only on positive attribution: a pid the OS will
/// not tell us about is not enough to strand a repair on.
fn legacy_record_pid(root: &Path, probe: &Probe) -> Option<u32> {
    let body = std::fs::read_to_string(root.join("studio.pid")).ok()?;
    // pid 0 and 1 are never a backend, and signalling either would be a bug.
    let pid = body.trim().parse::<u32>().ok().filter(|pid| *pid > 1)?;
    match attribute(pid, probe) {
        Ok(Some(true)) => Some(pid),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

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

    const RECORDED: u32 = 4242;
    const RECORDED_8888: &str = "studio-8888-4242.pid";

    fn live(runs_from_tree: &dyn Fn(u32) -> Option<bool>) -> Probe<'_> {
        Probe {
            is_live: &|_| true,
            runs_from_tree,
        }
    }

    fn gone<'a>() -> Probe<'a> {
        Probe {
            is_live: &|_| false,
            runs_from_tree: &|_| unreachable!("a dead pid is never attributed"),
        }
    }

    fn ours(_pid: u32) -> Option<bool> {
        Some(true)
    }

    fn theirs(_pid: u32) -> Option<bool> {
        Some(false)
    }

    fn unreadable(_pid: u32) -> Option<bool> {
        None
    }

    #[test]
    fn a_live_record_from_our_tree_is_found() {
        let home = Home::new();
        home.record(RECORDED_8888, "");

        assert_eq!(
            live_backend_pid_in(&home.path(), 8888, &live(&ours)),
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

    /// The pid outlived the record and now belongs to something else. Codex
    /// flagged this as a stale-record veto over a live repair.
    #[test]
    fn a_reused_pid_running_another_tree_is_ignored() {
        let home = Home::new();
        home.record(RECORDED_8888, "");

        assert_eq!(
            live_backend_pid_in(&home.path(), 8888, &live(&theirs)),
            None
        );
    }

    /// ...but a record naming this very port, whose process we simply may not
    /// query, keeps blocking. Guessing "not ours" there would rewrite the venv
    /// underneath a live server.
    #[test]
    fn a_per_port_record_we_cannot_attribute_still_blocks() {
        let home = Home::new();
        home.record(RECORDED_8888, "");

        assert_eq!(
            live_backend_pid_in(&home.path(), 8888, &live(&unreadable)),
            Some(RECORDED)
        );
    }

    #[test]
    fn a_record_for_another_port_is_ignored() {
        let home = Home::new();
        home.record("studio-8889-4242.pid", "");
        // A port that merely starts with the digits of ours is another port.
        home.record("studio-88881-4242.pid", "");

        assert_eq!(live_backend_pid_in(&home.path(), 8888, &live(&ours)), None);
    }

    /// A pre-upgrade server records itself here and nowhere else, and it is
    /// precisely the kind of build that reports no install id.
    #[test]
    fn a_live_legacy_record_from_our_tree_blocks() {
        let home = Home::new();
        home.record("studio.pid", &RECORDED.to_string());

        assert_eq!(
            live_backend_pid_in(&home.path(), 8888, &live(&ours)),
            Some(RECORDED)
        );
    }

    #[test]
    fn a_legacy_record_from_another_tree_is_ignored() {
        let home = Home::new();
        home.record("studio.pid", &RECORDED.to_string());

        assert_eq!(
            live_backend_pid_in(&home.path(), 8888, &live(&theirs)),
            None
        );
    }

    /// The legacy record names no port, so an unreadable process behind it is
    /// not evidence about this one.
    #[test]
    fn a_legacy_record_we_cannot_attribute_does_not_block() {
        let home = Home::new();
        home.record("studio.pid", &RECORDED.to_string());

        assert_eq!(
            live_backend_pid_in(&home.path(), 8888, &live(&unreadable)),
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

        assert_eq!(live_backend_pid_in(&home.path(), 8888, &live(&ours)), None);

        home.record("studio.pid", "not a pid");

        assert_eq!(live_backend_pid_in(&home.path(), 8888, &live(&ours)), None);
    }

    /// The reported case: the backend on the port is reached over an SSH
    /// forward, so this install recorded nothing anywhere.
    #[test]
    fn nothing_recorded_means_nothing_local() {
        let home = Home::new();
        home.record("studio-8888-notapid.pid", "");

        assert_eq!(live_backend_pid_in(&home.path(), 8888, &live(&ours)), None);
    }

    #[test]
    fn a_missing_studio_home_is_not_an_error() {
        let home = Home::new();

        assert_eq!(
            live_backend_pid_in(&home.path().join("absent"), 8888, &live(&ours)),
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
            let dir = std::env::current_exe().unwrap().parent().unwrap().to_path_buf();
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
        command.spawn().expect("a foreign process should start")
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

    #[test]
    fn a_legacy_record_naming_this_install_is_found() {
        let mut root = RecordRoot::at_our_own_tree();
        let me = std::process::id();
        root.record("studio.pid", &me.to_string());

        assert_eq!(live_backend_pid_on_port(8893), Some(me));
    }

    /// uv symlinks the venv interpreter at a base one outside the tree, so the
    /// OS reports the base binary as the image. Reproduced with a real symlink
    /// and a real process: the record must still be able to block. Unix only,
    /// because a Windows venv holds a real python.exe and cannot hit this.
    #[cfg(unix)]
    #[test]
    fn a_symlinked_interpreter_does_not_look_like_a_foreign_process() {
        let tree = tempfile::tempdir().unwrap();
        let bin = tree.path().join("unsloth_studio").join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        let mut child = spawn_foreign();
        let foreign_exe = std::fs::read_link(format!("/proc/{}/exe", child.id()))
            .or_else(|_| which_foreign())
            .unwrap();
        std::os::unix::fs::symlink(&foreign_exe, bin.join("python")).unwrap();

        let shared = crate::process_identity::shared_interpreters_in(tree.path());
        let verdict = crate::process_identity::runs_from(child.id(), tree.path(), &shared);
        // Same process, without the venv link in play, is plainly foreign.
        let without_link = crate::process_identity::runs_from(child.id(), tree.path(), &[]);
        child.kill().unwrap();
        child.wait().unwrap();

        assert_eq!(verdict, None, "a symlinked venv interpreter proves nothing");
        assert_eq!(without_link, Some(false));
    }

    #[cfg(unix)]
    fn which_foreign() -> std::io::Result<PathBuf> {
        std::fs::canonicalize("/bin/sleep").or_else(|_| std::fs::canonicalize("/usr/bin/sleep"))
    }
}
