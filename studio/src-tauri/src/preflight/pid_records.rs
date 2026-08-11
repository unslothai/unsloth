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
pub(super) fn live_backend_pid_on_port(port: u16) -> Option<u32> {
    let root = crate::diagnostics::studio_dir();
    let probe = Probe {
        is_live: &|pid| crate::desktop_backend_owner::pid_is_not_dead(pid),
        runs_from_tree: &|pid| crate::process_identity::runs_from(pid, &root),
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
