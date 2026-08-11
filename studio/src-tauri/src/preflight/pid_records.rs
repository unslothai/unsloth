use std::path::Path;

/// PID of a live Studio backend of THIS install serving `port`, if one is
/// recorded.
///
/// The Python server drops a `studio-{port}-{pid}.pid` file into its studio
/// home on bind, so these records answer the one question an HTTP probe cannot:
/// is the thing on this port a local process out of the tree we are about to
/// rewrite? A backend reached through a port forward, or one belonging to a
/// different install (different studio home, hence different record dir),
/// leaves nothing here.
///
/// Records outlive crashes, so liveness decides; the file alone means nothing.
pub(super) fn live_backend_pid_on_port(port: u16) -> Option<u32> {
    live_backend_pid_in(
        &crate::diagnostics::studio_dir(),
        port,
        crate::desktop_backend_owner::pid_is_not_dead,
    )
}

fn live_backend_pid_in(root: &Path, port: u16, is_live: impl Fn(u32) -> bool) -> Option<u32> {
    let prefix = format!("studio-{port}-");
    for entry in std::fs::read_dir(root).ok()?.flatten() {
        let file_name = entry.file_name();
        let Some(name) = file_name.to_str() else {
            continue;
        };
        // Parsed from the name rather than the body: the name is what the
        // binding process itself wrote, and no read can fail halfway.
        let Some(pid) = name
            .strip_prefix(&prefix)
            .and_then(|rest| rest.strip_suffix(".pid"))
            .and_then(|pid| pid.parse::<u32>().ok())
        else {
            continue;
        };
        if is_live(pid) {
            return Some(pid);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(dir: &Path, name: &str) {
        std::fs::write(dir.join(name), "1").unwrap();
    }

    #[test]
    fn a_live_record_for_the_port_is_found() {
        let dir = tempfile::tempdir().unwrap();
        record(dir.path(), "studio-8888-4242.pid");

        assert_eq!(
            live_backend_pid_in(dir.path(), 8888, |pid| pid == 4242),
            Some(4242)
        );
    }

    /// These files pile up: a crashed server never removes its own.
    #[test]
    fn a_dead_record_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        record(dir.path(), "studio-8888-4242.pid");

        assert_eq!(live_backend_pid_in(dir.path(), 8888, |_| false), None);
    }

    #[test]
    fn a_live_record_for_another_port_is_ignored() {
        let dir = tempfile::tempdir().unwrap();
        record(dir.path(), "studio-8889-4242.pid");
        // A prefix match on the port digits must not count either.
        record(dir.path(), "studio-88881-4243.pid");

        assert_eq!(live_backend_pid_in(dir.path(), 8888, |_| true), None);
    }

    /// The reported case: the server on the port is reached over an SSH tunnel,
    /// so nothing local recorded it.
    #[test]
    fn nothing_recorded_means_nothing_local() {
        let dir = tempfile::tempdir().unwrap();
        record(dir.path(), "studio.pid");
        record(dir.path(), "studio-8888-notapid.pid");

        assert_eq!(live_backend_pid_in(dir.path(), 8888, |_| true), None);
    }

    #[test]
    fn a_missing_studio_home_is_not_an_error() {
        let dir = tempfile::tempdir().unwrap();

        assert_eq!(
            live_backend_pid_in(&dir.path().join("absent"), 8888, |_| true),
            None
        );
    }
}
