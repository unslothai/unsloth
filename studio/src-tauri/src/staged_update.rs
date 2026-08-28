use crate::process_identity::ProcessOrigin;
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};

pub(crate) const STAGE_DIR: &str = ".update-stage";
const PREV_DIR: &str = ".update-prev";
const FAILED_MARKER: &str = ".update-failed.json";
const READY_MARKER: &str = "READY.json";
const PENDING_MARKER: &str = "PENDING.json";
const ROLLBACK_TRASH_PREFIX: &str = ".update-rollback-";
const RUNTIME_ENTRIES: [&str; 4] = [
    "unsloth_studio",
    ".venv_t5_530",
    ".venv_t5_550",
    ".venv_t5_510",
];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct StagedVersions {
    pub backend_version: String,
    #[serde(default)]
    pub shell_version: Option<String>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct StagedUpdateStatus {
    pub state: &'static str,
    pub backend_version: Option<String>,
    pub shell_version: Option<String>,
}

impl StagedUpdateStatus {
    fn with(state: &'static str, versions: Option<StagedVersions>) -> Self {
        Self {
            state,
            backend_version: versions.as_ref().map(|v| v.backend_version.clone()),
            shell_version: versions.and_then(|v| v.shell_version),
        }
    }
}

fn read_versions(path: &Path) -> Option<StagedVersions> {
    serde_json::from_str(&fs::read_to_string(path).ok()?).ok()
}

fn write_versions(path: &Path, versions: &StagedVersions) -> Result<(), String> {
    let body = serde_json::to_string_pretty(versions).map_err(|e| e.to_string())?;
    fs::write(path, body).map_err(|e| format!("{}: {e}", path.display()))
}

pub(crate) fn status(home: &Path) -> StagedUpdateStatus {
    let stage = home.join(STAGE_DIR);
    if let Some(versions) = read_versions(&stage.join(READY_MARKER)) {
        return StagedUpdateStatus::with("ready", Some(versions));
    }
    if stage.is_dir() {
        return StagedUpdateStatus::with("partial", None);
    }
    if let Some(versions) = read_versions(&home.join(FAILED_MARKER)) {
        return StagedUpdateStatus::with("failed", Some(versions));
    }
    StagedUpdateStatus::with("none", None)
}

pub(crate) fn discard(home: &Path) {
    let _ = fs::remove_dir_all(home.join(STAGE_DIR));
}

pub(crate) fn reconcile_at_launch(home: &Path, shell_version: &str) {
    remove_stale_trash(home);
    if let Err(error) = roll_back_unconfirmed(home) {
        warn!("[staged-update] rollback failed: {error}");
    }
    if let Err(error) = activate_ready(home, shell_version) {
        warn!("[staged-update] activation failed: {error}");
    }
}

fn remove_stale_trash(home: &Path) {
    let Ok(entries) = fs::read_dir(home) else {
        return;
    };
    let stale: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(ROLLBACK_TRASH_PREFIX))
        })
        .collect();
    if stale.is_empty() {
        return;
    }
    std::thread::spawn(move || {
        for path in stale {
            let _ = fs::remove_dir_all(path);
        }
    });
}

pub(crate) fn confirm_activated(home: &Path) {
    let prev = home.join(PREV_DIR);
    if fs::remove_file(prev.join(PENDING_MARKER)).is_err() {
        return;
    }
    info!("[staged-update] backend healthy, dropping previous runtime");
    std::thread::spawn(move || {
        let _ = fs::remove_dir_all(prev);
    });
}

fn roll_back_unconfirmed(home: &Path) -> Result<(), String> {
    roll_back_unconfirmed_with(home, live_tree_in_use(home))
}

fn roll_back_unconfirmed_with(home: &Path, in_use: bool) -> Result<(), String> {
    let prev = home.join(PREV_DIR);
    let Some(versions) = read_versions(&prev.join(PENDING_MARKER)) else {
        if prev.is_dir() {
            let _ = fs::remove_dir_all(&prev);
        }
        return Ok(());
    };
    if in_use {
        // Renaming the tree under a live process is unsafe, but a process that never
        // reached validate_candidate_port is not healthy either. Confirming here
        // would drop the only way back. Leave the marker: validate_candidate_port
        // still confirms on a healthy port, and a later launch can still roll back.
        info!("[staged-update] runtime still in use, deferring the rollback decision");
        return Ok(());
    }
    info!(
        "[staged-update] backend {} never became healthy, restoring previous runtime",
        versions.backend_version
    );
    let trash = home.join(format!("{ROLLBACK_TRASH_PREFIX}{}", std::process::id()));
    fs::create_dir_all(&trash).map_err(|e| e.to_string())?;
    swap_entries(home, &prev, &trash, true)?;
    write_versions(&home.join(FAILED_MARKER), &versions)?;
    let _ = fs::remove_dir_all(&prev);
    std::thread::spawn(move || {
        let _ = fs::remove_dir_all(trash);
    });
    Ok(())
}

fn activate_ready(home: &Path, shell_version: &str) -> Result<(), String> {
    let stage = home.join(STAGE_DIR);
    if !stage.is_dir() {
        return Ok(());
    }
    let Some(versions) = read_versions(&stage.join(READY_MARKER)) else {
        info!("[staged-update] removing incomplete stage");
        let _ = fs::remove_dir_all(&stage);
        return Ok(());
    };
    if let Some(required) = versions.shell_version.as_deref() {
        match shell_order(shell_version, required) {
            Ordering::Less => {
                info!("[staged-update] staged backend waits for app {required}");
                return Ok(());
            }
            Ordering::Greater => {
                info!("[staged-update] staged backend was built for app {required}, discarding");
                let _ = fs::remove_dir_all(&stage);
                return Ok(());
            }
            Ordering::Equal => {}
        }
    }
    if live_tree_in_use(home) {
        info!("[staged-update] runtime in use, keeping staged backend for the next launch");
        return Ok(());
    }
    let prev = home.join(PREV_DIR);
    let _ = fs::remove_dir_all(&prev);
    fs::create_dir_all(&prev).map_err(|e| e.to_string())?;
    swap_entries(home, &stage, &prev, false)?;
    if let Err(error) = write_versions(&prev.join(PENDING_MARKER), &versions) {
        // Without the marker the next launch drops the previous runtime instead of
        // restoring it, so an unverified backend would become permanent. Put it back.
        let _ = swap_entries(home, &prev, &stage, false);
        let _ = fs::remove_dir_all(&prev);
        return Err(error);
    }
    let _ = fs::remove_file(home.join(FAILED_MARKER));
    let _ = fs::remove_dir_all(&stage);
    info!(
        "[staged-update] activated backend {}",
        versions.backend_version
    );
    Ok(())
}

fn shell_order(current: &str, required: &str) -> Ordering {
    match crate::desktop_update_policy::compare_versions(current, required) {
        1 => Ordering::Greater,
        -1 => Ordering::Less,
        _ => Ordering::Equal,
    }
}

/// `evict_extra` moves a live entry the incoming tree does not have out of the way
/// instead of leaving it. Rollback needs it: a legacy install with no tiered
/// sidecars has nothing to put in `.update-prev` for them, so without this the
/// failed update's sidecars stay live beside the restored backend, which then
/// imports from an environment that was never confirmed. Activation must not do
/// it -- there the live tree is the one being preserved.
fn swap_entries(
    home: &Path,
    incoming: &Path,
    outgoing: &Path,
    evict_extra: bool,
) -> Result<(), String> {
    let mut moved: Vec<(PathBuf, PathBuf)> = Vec::new();
    let result = (|| {
        for name in RUNTIME_ENTRIES {
            let staged = incoming.join(name);
            let live = home.join(name);
            if !staged.exists() {
                if evict_extra && live.exists() {
                    rename_tracked(&live, &outgoing.join(name), &mut moved)?;
                }
                continue;
            }
            if live.exists() {
                rename_tracked(&live, &outgoing.join(name), &mut moved)?;
            }
            rename_tracked(&staged, &live, &mut moved)?;
        }
        Ok(())
    })();
    if result.is_err() {
        for (from, to) in moved.iter().rev() {
            let _ = fs::rename(to, from);
        }
    }
    result
}

fn rename_tracked(
    from: &Path,
    to: &Path,
    moved: &mut Vec<(PathBuf, PathBuf)>,
) -> Result<(), String> {
    fs::rename(from, to)
        .map_err(|e| format!("{} -> {}: {e}", from.display(), to.display()))?;
    moved.push((from.to_path_buf(), to.to_path_buf()));
    Ok(())
}

fn recorded_pids(home: &Path) -> Vec<u32> {
    let mut pids = Vec::new();
    if let Ok(entries) = fs::read_dir(home) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            let pid = name
                .strip_prefix("studio-")
                .and_then(|rest| rest.strip_suffix(".pid"))
                .and_then(|rest| rest.rsplit('-').next())
                .and_then(|pid| pid.parse::<u32>().ok());
            if let Some(pid) = pid {
                pids.push(pid);
            }
        }
    }
    if let Ok(body) = fs::read_to_string(home.join("studio.pid")) {
        if let Some(pid) = body.lines().next().and_then(|l| l.trim().parse::<u32>().ok()) {
            pids.push(pid);
        }
    }
    pids.retain(|pid| *pid > 1);
    pids.sort_unstable();
    pids.dedup();
    pids
}

fn live_tree_in_use(home: &Path) -> bool {
    #[cfg(windows)]
    {
        if let Some(bin) = crate::process::find_unsloth_binary_in_studio_dir(home) {
            if crate::process::ensure_managed_environment_is_idle(&bin).is_err() {
                return true;
            }
        }
    }
    let interpreters = crate::process_identity::interpreters_of(home);
    recorded_pids(home).into_iter().any(|pid| {
        crate::desktop_backend_owner::pid_is_not_dead(pid)
            && !crate::process_identity::is_zombie(pid)
            && crate::process_identity::origin_of(pid, home, &interpreters)
                != ProcessOrigin::Elsewhere
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_home(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "unsloth-staged-update-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Three code paths drop a tree on a background thread, and Windows refuses to
    /// remove a directory while another handle is still walking inside it, so a
    /// teardown that races one of them fails with ERROR_ACCESS_DENIED rather than
    /// telling anyone anything. Retry until the deleter is done.
    fn cleanup(home: PathBuf) {
        for _ in 0..100 {
            if fs::remove_dir_all(&home).is_ok() || !home.exists() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        fs::remove_dir_all(&home).unwrap();
    }

    fn make_runtime(root: &Path, tag: &str) {
        for name in RUNTIME_ENTRIES {
            fs::create_dir_all(root.join(name)).unwrap();
            fs::write(root.join(name).join("tag"), tag).unwrap();
        }
    }

    fn tag(root: &Path, name: &str) -> String {
        fs::read_to_string(root.join(name).join("tag")).unwrap_or_default()
    }

    fn stage_ready(home: &Path, versions: &StagedVersions) {
        let stage = home.join(STAGE_DIR);
        make_runtime(&stage, "new");
        write_versions(&stage.join(READY_MARKER), versions).unwrap();
    }

    fn versions(shell: Option<&str>) -> StagedVersions {
        StagedVersions {
            backend_version: "2026.9.1".into(),
            shell_version: shell.map(str::to_string),
        }
    }

    #[test]
    fn ready_stage_replaces_the_runtime_and_keeps_the_previous_one_pending() {
        let home = temp_home("activate");
        make_runtime(&home, "old");
        stage_ready(&home, &versions(Some("0.1.900-beta")));

        reconcile_at_launch(&home, "0.1.900-beta");

        assert_eq!(tag(&home, "unsloth_studio"), "new");
        assert_eq!(tag(&home.join(PREV_DIR), "unsloth_studio"), "old");
        assert!(home.join(PREV_DIR).join(PENDING_MARKER).is_file());
        assert!(!home.join(STAGE_DIR).exists());
        assert_eq!(status(&home).state, "none");
        cleanup(home);
    }

    #[test]
    fn stage_built_for_a_newer_shell_waits() {
        let home = temp_home("wait");
        make_runtime(&home, "old");
        stage_ready(&home, &versions(Some("0.1.901-beta")));

        reconcile_at_launch(&home, "0.1.900-beta");

        assert_eq!(tag(&home, "unsloth_studio"), "old");
        assert_eq!(status(&home).state, "ready");
        cleanup(home);
    }

    #[test]
    fn stage_built_for_an_older_shell_is_discarded() {
        let home = temp_home("discard");
        make_runtime(&home, "old");
        stage_ready(&home, &versions(Some("0.1.899-beta")));

        reconcile_at_launch(&home, "0.1.900-beta");

        assert_eq!(tag(&home, "unsloth_studio"), "old");
        assert!(!home.join(STAGE_DIR).exists());
        cleanup(home);
    }

    #[test]
    fn incomplete_stage_is_removed() {
        let home = temp_home("partial");
        make_runtime(&home, "old");
        make_runtime(&home.join(STAGE_DIR), "half");
        assert_eq!(status(&home).state, "partial");

        reconcile_at_launch(&home, "0.1.900-beta");

        assert_eq!(tag(&home, "unsloth_studio"), "old");
        assert!(!home.join(STAGE_DIR).exists());
        cleanup(home);
    }

    #[test]
    fn unconfirmed_activation_rolls_back_and_records_the_failure() {
        let home = temp_home("rollback");
        make_runtime(&home, "old");
        stage_ready(&home, &versions(None));
        reconcile_at_launch(&home, "0.1.900-beta");
        assert_eq!(tag(&home, "unsloth_studio"), "new");

        reconcile_at_launch(&home, "0.1.900-beta");

        assert_eq!(tag(&home, "unsloth_studio"), "old");
        assert_eq!(tag(&home, ".venv_t5_530"), "old");
        assert!(!home.join(PREV_DIR).exists());
        let failed = status(&home);
        assert_eq!(failed.state, "failed");
        assert_eq!(failed.backend_version.as_deref(), Some("2026.9.1"));
        cleanup(home);
    }

    #[test]
    fn confirmation_drops_the_previous_runtime_and_the_next_launch_keeps_the_new_one() {
        let home = temp_home("confirm");
        make_runtime(&home, "old");
        stage_ready(&home, &versions(None));
        reconcile_at_launch(&home, "0.1.900-beta");

        confirm_activated(&home);
        assert!(!home.join(PREV_DIR).join(PENDING_MARKER).exists());
        reconcile_at_launch(&home, "0.1.900-beta");

        assert_eq!(tag(&home, "unsloth_studio"), "new");
        assert_eq!(status(&home).state, "none");
        cleanup(home);
    }

    #[test]
    fn a_failed_swap_restores_every_entry_it_moved() {
        let home = temp_home("swap-fail");
        make_runtime(&home, "old");
        let stage = home.join(STAGE_DIR);
        make_runtime(&stage, "new");
        let prev = home.join(PREV_DIR);
        let blocker = prev.join(".venv_t5_510");
        fs::create_dir_all(&blocker).unwrap();
        fs::write(blocker.join("occupied"), "").unwrap();

        let result = swap_entries(&home, &stage, &prev, false);

        assert!(result.is_err());
        for name in RUNTIME_ENTRIES {
            assert_eq!(tag(&home, name), "old", "{name}");
            assert!(!prev.join(name).join("tag").exists(), "{name}");
        }
        assert_eq!(tag(&stage, "unsloth_studio"), "new");
        cleanup(home);
    }

    #[test]
    fn swapping_back_restores_the_runtime_the_activation_replaced() {
        let home = temp_home("swap-back");
        make_runtime(&home, "old");
        let stage = home.join(STAGE_DIR);
        make_runtime(&stage, "new");
        let prev = home.join(PREV_DIR);
        fs::create_dir_all(&prev).unwrap();

        swap_entries(&home, &stage, &prev, false).unwrap();
        assert_eq!(tag(&home, "unsloth_studio"), "new");
        // What the failed-marker branch runs: the previous runtime goes back and the
        // staged one returns to the stage, so the next launch can try it again.
        swap_entries(&home, &prev, &stage, false).unwrap();

        for name in RUNTIME_ENTRIES {
            assert_eq!(tag(&home, name), "old", "{name}");
            assert_eq!(tag(&stage, name), "new", "{name}");
        }
        cleanup(home);
    }

    #[test]
    fn rollback_takes_back_sidecars_the_failed_update_added() {
        let home = temp_home("rollback-extra");
        // A legacy install: the managed venv is there, the tiered sidecars are not.
        fs::create_dir_all(home.join("unsloth_studio")).unwrap();
        fs::write(home.join("unsloth_studio").join("tag"), "old").unwrap();
        // The staged update builds all of them.
        stage_ready(&home, &versions(None));
        reconcile_at_launch(&home, "0.1.900-beta");
        assert_eq!(tag(&home, "unsloth_studio"), "new");
        assert_eq!(tag(&home, ".venv_t5_530"), "new");

        // The new backend never became healthy.
        reconcile_at_launch(&home, "0.1.900-beta");

        assert_eq!(tag(&home, "unsloth_studio"), "old");
        for name in [".venv_t5_530", ".venv_t5_550", ".venv_t5_510"] {
            // The restored backend must not find the unconfirmed update's sidecars.
            assert!(!home.join(name).exists(), "{name}");
        }
        assert_eq!(status(&home).state, "failed");
        cleanup(home);
    }

    #[test]
    fn a_runtime_still_in_use_defers_instead_of_confirming() {
        let home = temp_home("defer");
        make_runtime(&home, "old");
        stage_ready(&home, &versions(None));
        reconcile_at_launch(&home, "0.1.900-beta");
        assert_eq!(tag(&home, "unsloth_studio"), "new");
        assert!(home.join(PREV_DIR).join(PENDING_MARKER).is_file());

        // Force-closed after activation: a backend is alive on the new tree but it
        // never reached validate_candidate_port, so nothing has vouched for it.
        roll_back_unconfirmed_with(&home, true).unwrap();

        // Neither confirmed nor rolled back -- the way back has to survive.
        assert_eq!(tag(&home, "unsloth_studio"), "new");
        assert!(home.join(PREV_DIR).join(PENDING_MARKER).is_file());

        // Once nothing holds the tree, the rollback it was owed still happens.
        roll_back_unconfirmed_with(&home, false).unwrap();

        assert_eq!(tag(&home, "unsloth_studio"), "old");
        assert_eq!(status(&home).state, "failed");
        cleanup(home);
    }

    #[test]
    fn stale_rollback_trash_is_removed_at_launch() {
        let home = temp_home("trash");
        make_runtime(&home, "old");
        let trash = home.join(format!("{ROLLBACK_TRASH_PREFIX}1"));
        fs::create_dir_all(trash.join("unsloth_studio")).unwrap();

        reconcile_at_launch(&home, "0.1.900-beta");
        for _ in 0..50 {
            if !trash.exists() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }

        assert!(!trash.exists());
        assert_eq!(tag(&home, "unsloth_studio"), "old");
        cleanup(home);
    }

    #[test]
    fn status_reports_a_ready_stage() {
        let home = temp_home("status");
        stage_ready(&home, &versions(Some("0.1.900-beta")));
        assert_eq!(
            status(&home),
            StagedUpdateStatus {
                state: "ready",
                backend_version: Some("2026.9.1".into()),
                shell_version: Some("0.1.900-beta".into()),
            }
        );
        discard(&home);
        assert_eq!(status(&home).state, "none");
        cleanup(home);
    }
}
