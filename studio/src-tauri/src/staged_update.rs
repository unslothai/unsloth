use crate::process_identity::ProcessOrigin;
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

pub(crate) const STAGE_DIR: &str = ".update-stage";
const PREV_DIR: &str = ".update-prev";
const FAILED_MARKER: &str = ".update-failed.json";
const READY_MARKER: &str = "READY.json";
const PENDING_MARKER: &str = "PENDING.json";
const CONFIRMED_MARKER: &str = "CONFIRMED.json";
const ROLLED_BACK_MARKER: &str = "ROLLED_BACK.json";
const ROLLBACK_TRASH_PREFIX: &str = ".update-rollback-";
const RUNTIME_ENTRIES: [&str; 4] = [
    "unsloth_studio",
    ".venv_t5_530",
    ".venv_t5_550",
    ".venv_t5_510",
];
const HELPER_RUNTIME_ENTRIES: [&str; 3] = ["node", "llama.cpp", "whisper.cpp"];

fn all_runtime_entries() -> impl Iterator<Item = &'static str> {
    RUNTIME_ENTRIES.into_iter().chain(HELPER_RUNTIME_ENTRIES)
}

fn live_entry(home: &Path, name: &str) -> PathBuf {
    if HELPER_RUNTIME_ENTRIES.contains(&name) {
        home.parent().unwrap_or(home).join(name)
    } else {
        home.join(name)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct StagedVersions {
    pub backend_version: String,
    #[serde(default)]
    pub shell_version: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct ActivationJournal {
    #[serde(flatten)]
    versions: StagedVersions,
    #[serde(default)]
    previous_entries: Vec<String>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) struct StagedUpdateStatus {
    pub state: &'static str,
    pub backend_version: Option<String>,
    pub shell_version: Option<String>,
    /// A staged child is running now, so a partial stage is being written rather
    /// than left over.
    pub staging: bool,
    pub staging_shell_version: Option<String>,
}

impl StagedUpdateStatus {
    fn with(state: &'static str, versions: Option<StagedVersions>) -> Self {
        Self {
            state,
            backend_version: versions.as_ref().map(|v| v.backend_version.clone()),
            shell_version: versions.and_then(|v| v.shell_version),
            staging: false,
            staging_shell_version: None,
        }
    }
}

fn read_versions(path: &Path) -> Option<StagedVersions> {
    serde_json::from_str(&fs::read_to_string(path).ok()?).ok()
}

fn write_versions(path: &Path, versions: &StagedVersions) -> Result<(), String> {
    let body = serde_json::to_vec_pretty(versions).map_err(|e| e.to_string())?;
    write_atomic(path, &body)
}

fn read_journal(path: &Path) -> Option<ActivationJournal> {
    serde_json::from_str(&fs::read_to_string(path).ok()?).ok()
}

pub(crate) fn pending_versions(home: &Path) -> Option<StagedVersions> {
    read_journal(&home.join(PREV_DIR).join(PENDING_MARKER)).map(|journal| journal.versions)
}

fn write_journal(path: &Path, journal: &ActivationJournal) -> Result<(), String> {
    let body = serde_json::to_vec_pretty(journal).map_err(|e| e.to_string())?;
    write_atomic(path, &body)
}

fn write_atomic(path: &Path, body: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent", path.display()))?;
    fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    let mut temporary = tempfile::Builder::new()
        .prefix(".update-marker-")
        .tempfile_in(parent)
        .map_err(|e| format!("{}: {e}", parent.display()))?;
    temporary
        .write_all(body)
        .and_then(|()| temporary.as_file().sync_all())
        .map_err(|e| format!("{}: {e}", path.display()))?;
    temporary
        .persist(path)
        .map_err(|e| format!("{}: {}", path.display(), e.error))?;
    #[cfg(unix)]
    let _ = File::open(parent).and_then(|directory| directory.sync_all());
    Ok(())
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
        // Activating now would delete .update-prev, and a rollback that failed part
        // way may have left the unconfirmed runtime live: the next candidate would
        // then be recorded against it with the last known-good copy already gone.
        warn!("[staged-update] rollback failed, not activating: {error}");
        return;
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

pub(crate) fn confirm_activated(home: &Path, observed_backend_version: &str) -> bool {
    let prev = home.join(PREV_DIR);
    let Some(versions) = read_versions(&prev.join(PENDING_MARKER)) else {
        return false;
    };
    if crate::desktop_update_policy::compare_versions(
        observed_backend_version,
        &versions.backend_version,
    ) < 0
    {
        warn!(
            "[staged-update] backend {} is older than staged runtime {}",
            observed_backend_version, versions.backend_version
        );
        return false;
    }
    if let Err(error) = write_versions(&prev.join(CONFIRMED_MARKER), &versions) {
        warn!("[staged-update] could not confirm activated runtime: {error}");
        return false;
    }
    info!("[staged-update] backend healthy, dropping previous runtime");
    std::thread::spawn(move || {
        remove_confirmed_previous(&prev);
    });
    true
}

fn remove_confirmed_previous(prev: &Path) {
    if let Ok(entries) = fs::read_dir(prev) {
        for entry in entries.flatten() {
            if entry.file_name() == CONFIRMED_MARKER {
                continue;
            }
            let path = entry.path();
            if path.is_dir() {
                let _ = fs::remove_dir_all(path);
            } else {
                let _ = fs::remove_file(path);
            }
        }
    }
    let _ = fs::remove_file(prev.join(CONFIRMED_MARKER));
    let _ = fs::remove_dir(prev);
}

fn roll_back_unconfirmed(home: &Path) -> Result<(), String> {
    roll_back_unconfirmed_with(home, live_tree_in_use(home))
}

pub(crate) fn roll_back_failed_activation(
    home: &Path,
) -> Result<Option<StagedVersions>, String> {
    let versions = pending_versions(home);
    if versions.is_none() || live_tree_in_use(home) {
        return Ok(None);
    }
    roll_back_unconfirmed_with(home, false)?;
    Ok(versions)
}

pub(crate) fn recover_failed_activation(
    home: &Path,
    restart: impl FnOnce(),
) -> Result<bool, String> {
    if roll_back_failed_activation(home)?.is_none() {
        return Ok(false);
    }
    restart();
    Ok(true)
}

fn roll_back_unconfirmed_with(home: &Path, in_use: bool) -> Result<(), String> {
    let prev = home.join(PREV_DIR);
    if prev.join(CONFIRMED_MARKER).is_file() {
        remove_confirmed_previous(&prev);
        return Ok(());
    }
    let rolled_back = prev.join(ROLLED_BACK_MARKER);
    if rolled_back.is_file() {
        if let Some(versions) = read_versions(&rolled_back) {
            let failed = home.join(FAILED_MARKER);
            if !failed.is_file() {
                write_versions(&failed, &versions)?;
            }
        }
        let _ = fs::remove_dir_all(home.join(STAGE_DIR));
        let _ = fs::remove_dir_all(&prev);
        return Ok(());
    }
    let journal = read_journal(&prev.join(PENDING_MARKER));
    let versions = journal.as_ref().map(|journal| journal.versions.clone());
    let has_previous_runtime = all_runtime_entries().any(|name| prev.join(name).exists());
    if versions.is_none() && !has_previous_runtime {
        if prev.is_dir() {
            let _ = fs::remove_dir_all(&prev);
        }
        return Ok(());
    }
    if in_use {
        // Renaming the tree under a live process is unsafe, but a process that never
        // reached validate_candidate_port is not healthy either. Confirming here
        // would drop the only way back. Leave the marker: validate_candidate_port
        // still confirms on a healthy port, and a later launch can still roll back.
        info!("[staged-update] runtime still in use, deferring the rollback decision");
        return Ok(());
    }
    let versions = versions.or_else(|| read_versions(&home.join(STAGE_DIR).join(READY_MARKER)));
    let previous_entries = journal
        .map(|journal| journal.previous_entries)
        .filter(|entries| !entries.is_empty())
        .unwrap_or_else(|| {
            all_runtime_entries()
                .filter(|name| prev.join(name).exists())
                .map(str::to_string)
                .collect()
        });
    info!("[staged-update] staged backend never became healthy, restoring previous runtime");
    let trash = restore_previous_runtime(home, &prev, &previous_entries)?;
    if let Some(versions) = versions.as_ref() {
        let failed = home.join(FAILED_MARKER);
        let _ = fs::remove_file(&failed);
        write_versions(&failed, versions)?;
        write_versions(&prev.join(ROLLED_BACK_MARKER), versions)?;
    }
    let _ = fs::remove_dir_all(home.join(STAGE_DIR));
    let _ = fs::remove_dir_all(&prev);
    std::thread::spawn(move || {
        let _ = fs::remove_dir_all(trash);
    });
    Ok(())
}

#[cfg(unix)]
fn finalize_stage_for_activation(stage: &Path) -> Result<(), String> {
    let python = stage.join("unsloth_studio").join("bin").join("python");
    let code = concat!(
        "import sys; from pathlib import Path; ",
        "from unsloth_cli._studio_stage import finalize_for_activation; ",
        "finalize_for_activation(Path(sys.argv[1]))"
    );
    let output = std::process::Command::new(&python)
        .args(["-I", "-c", code])
        .arg(stage)
        .current_dir(stage)
        .env_remove("PYTHONHOME")
        .env_remove("PYTHONPATH")
        .env_remove("VIRTUAL_ENV")
        .output()
        .map_err(|error| format!("{}: {error}", python.display()))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(format!(
        "{} exited with {}: {}",
        python.display(),
        output.status,
        stderr.trim()
    ))
}

#[cfg(windows)]
fn finalize_stage_for_activation(_stage: &Path) -> Result<(), String> {
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
    if let Err(error) = finalize_stage_for_activation(&stage) {
        let failed = home.join(FAILED_MARKER);
        let _ = fs::remove_file(&failed);
        write_versions(&failed, &versions)?;
        let _ = fs::remove_dir_all(&stage);
        return Err(format!("staged runtime finalization failed: {error}"));
    }
    let prev = home.join(PREV_DIR);
    let _ = fs::remove_dir_all(&prev);
    fs::create_dir_all(&prev).map_err(|e| e.to_string())?;
    let journal = ActivationJournal {
        versions: versions.clone(),
        previous_entries: all_runtime_entries()
            .filter(|name| live_entry(home, name).exists())
            .map(str::to_string)
            .collect(),
    };
    write_journal(&prev.join(PENDING_MARKER), &journal)?;
    if let Err(error) = swap_entries(home, &stage, &prev, false) {
        let rollback = restore_previous_runtime(home, &prev, &journal.previous_entries);
        if rollback.is_ok() {
            let _ = fs::remove_file(prev.join(PENDING_MARKER));
            let _ = fs::remove_dir_all(&prev);
        }
        return Err(match rollback {
            Ok(trash) => {
                let _ = fs::remove_dir_all(trash);
                error
            }
            Err(rollback_error) => format!("{error}; rollback failed: {rollback_error}"),
        });
    }
    let _ = fs::remove_file(home.join(FAILED_MARKER));
    let _ = fs::remove_dir_all(&stage);
    info!(
        "[staged-update] activated backend {}",
        versions.backend_version
    );
    Ok(())
}

/// A stage is only useful if it actually advanced to what this shell expects.
///
/// `unsloth studio update` upgrades best effort: against a stale mirror or an
/// unreachable index it can succeed from cache without moving the cloned package,
/// and the marker would then advertise a backend older than the shell needs. The
/// restart would activate it and drop straight into preflight repair, or fail to
/// start and roll back, instead of the fast update the pill promised.
pub(crate) fn staged_backend_meets(home: &Path, required: &str) -> Result<(), String> {
    let Some(actual) = status(home).backend_version else {
        return Err("the staged update recorded no backend version".to_string());
    };
    if crate::desktop_update_policy::compare_versions(&actual, required) < 0 {
        return Err(format!(
            "staged backend {actual} is older than the {required} this app needs; \
             the package index may be stale or unreachable"
        ));
    }
    Ok(())
}

fn shell_order(current: &str, required: &str) -> Ordering {
    match crate::desktop_update_policy::compare_versions(current, required) {
        1 => Ordering::Greater,
        -1 => Ordering::Less,
        _ => Ordering::Equal,
    }
}

fn restore_previous_runtime(
    home: &Path,
    previous: &Path,
    previous_entries: &[String],
) -> Result<PathBuf, String> {
    let stage = home.join(STAGE_DIR);
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let trash = home.join(format!(
        "{ROLLBACK_TRASH_PREFIX}{}-{suffix}",
        std::process::id()
    ));
    fs::create_dir_all(&trash).map_err(|e| e.to_string())?;
    let mut moved: Vec<(PathBuf, PathBuf)> = Vec::new();
    let result = (|| {
        for name in all_runtime_entries() {
            let old = previous.join(name);
            let staged = stage.join(name);
            let live = live_entry(home, name);
            if previous_entries.iter().any(|entry| entry == name) {
                if old.exists() {
                    if live.exists() {
                        rename_tracked(&live, &trash.join(name), &mut moved)?;
                    }
                    rename_tracked(&old, &live, &mut moved)?;
                }
            } else if !staged.exists() && live.exists() {
                rename_tracked(&live, &trash.join(name), &mut moved)?;
            }
        }
        Ok(())
    })();
    if result.is_err() {
        for (from, to) in moved.iter().rev() {
            let _ = fs::rename(to, from);
        }
    }
    result.map(|()| trash)
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
        for name in all_runtime_entries() {
            let staged = incoming.join(name);
            let live = live_entry(home, name);
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

/// Pids that claim to be a backend of this install.
///
/// Mirrors `live_sibling_backend` in studio/backend/run.py, which reads all three
/// record kinds because a backend can be in a state where only one exists: a
/// startup marker while it is still binding, a per-port record once it has bound,
/// and a bare `studio.pid` for a pre-upgrade server or one whose per-port write
/// failed. Markers matter most here: the backend keeps its marker after dropping
/// its pid records until shutdown really finishes, and renaming the tree during
/// either window would move it under a process still importing out of it.
fn recorded_pids(home: &Path) -> Vec<u32> {
    let mut pids = Vec::new();
    let mut timed = Vec::new();
    if let Ok(entries) = fs::read_dir(home) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            // Markers use the same body layout as a per-port record, so both carry
            // the start time that settles a reused pid.
            let pid = name
                .strip_suffix(".marker")
                .and_then(|rest| rest.strip_prefix("studio-starting-"))
                .or_else(|| {
                    name.strip_suffix(".pid")
                        .and_then(|rest| rest.strip_prefix("studio-"))
                        .and_then(|rest| rest.rsplit('-').next())
                })
                .and_then(|pid| pid.parse::<u32>().ok());
            let Some(pid) = pid else {
                continue;
            };
            // Judged either way, so the untimed legacy record below must not
            // resurrect a pid this evidence already rejected.
            timed.push(pid);
            if crate::process_identity::pid_start_time_matches(
                pid,
                crate::process_identity::recorded_pid_start_time(&entry.path()),
            ) {
                pids.push(pid);
            }
        }
    }
    if let Ok(body) = fs::read_to_string(home.join("studio.pid")) {
        if let Some(pid) = body.lines().next().and_then(|l| l.trim().parse::<u32>().ok()) {
            // It carries no start time, so on its own it would re-add a pid the
            // timed records just proved was reused.
            if !timed.contains(&pid) {
                pids.push(pid);
            }
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
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            let python = stage.join("unsloth_studio").join("bin").join("python");
            fs::create_dir_all(python.parent().unwrap()).unwrap();
            fs::write(&python, "#!/bin/sh\nexit 0\n").unwrap();
            let mut permissions = fs::metadata(&python).unwrap().permissions();
            permissions.set_mode(0o755);
            fs::set_permissions(&python, permissions).unwrap();
        }
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
        assert_eq!(pending_versions(&home), Some(versions(Some("0.1.900-beta"))));
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

    #[cfg(unix)]
    #[test]
    fn a_stage_that_cannot_finalize_never_replaces_the_live_runtime() {
        let home = temp_home("finalize-failure");
        make_runtime(&home, "old");
        stage_ready(&home, &versions(Some("0.1.900-beta")));
        let python = home
            .join(STAGE_DIR)
            .join("unsloth_studio")
            .join("bin")
            .join("python");
        fs::write(&python, "#!/bin/sh\nexit 42\n").unwrap();

        reconcile_at_launch(&home, "0.1.900-beta");

        assert_eq!(tag(&home, "unsloth_studio"), "old");
        assert!(!home.join(STAGE_DIR).exists());
        assert_eq!(status(&home).state, "failed");
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
    fn failed_activation_rolls_back_once_without_waiting_for_an_app_relaunch() {
        let home = temp_home("same-launch-rollback");
        make_runtime(&home, "old");
        stage_ready(&home, &versions(None));
        activate_ready(&home, "0.1.900-beta").unwrap();

        let restarts = std::cell::Cell::new(0);
        assert!(recover_failed_activation(&home, || restarts.set(restarts.get() + 1)).unwrap());
        assert_eq!(tag(&home, "unsloth_studio"), "old");
        assert!(home.join(FAILED_MARKER).is_file());
        assert!(!recover_failed_activation(&home, || restarts.set(restarts.get() + 1)).unwrap());
        assert_eq!(restarts.get(), 1);
        cleanup(home);
    }

    #[test]
    fn failed_activation_waits_until_the_live_runtime_is_unused() {
        let home = temp_home("same-launch-live");
        make_runtime(&home, "old");
        stage_ready(&home, &versions(None));
        activate_ready(&home, "0.1.900-beta").unwrap();
        let me = std::process::id();
        fs::write(
            home.join(format!("studio-starting-{me}.marker")),
            format!("{me}\n"),
        )
        .unwrap();

        let restarts = std::cell::Cell::new(0);
        assert!(!recover_failed_activation(&home, || restarts.set(1)).unwrap());
        assert_eq!(tag(&home, "unsloth_studio"), "new");
        assert!(home.join(PREV_DIR).join(PENDING_MARKER).is_file());
        assert_eq!(restarts.get(), 0);
        cleanup(home);
    }

    #[test]
    fn confirmation_drops_the_previous_runtime_and_the_next_launch_keeps_the_new_one() {
        let home = temp_home("confirm");
        make_runtime(&home, "old");
        stage_ready(&home, &versions(None));
        reconcile_at_launch(&home, "0.1.900-beta");

        assert!(confirm_activated(&home, "2026.9.1"));
        for _ in 0..50 {
            if !home.join(PREV_DIR).exists() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        reconcile_at_launch(&home, "0.1.900-beta");

        assert_eq!(tag(&home, "unsloth_studio"), "new");
        assert_eq!(status(&home).state, "none");
        cleanup(home);
    }

    #[test]
    fn an_older_backend_cannot_confirm_a_newer_activation() {
        let home = temp_home("confirm-version");
        make_runtime(&home, "old");
        stage_ready(&home, &versions(None));
        activate_ready(&home, "0.1.900-beta").unwrap();

        assert!(!confirm_activated(&home, "2026.8.4"));
        assert!(home.join(PREV_DIR).join(PENDING_MARKER).is_file());
        assert_eq!(tag(&home.join(PREV_DIR), "unsloth_studio"), "old");
        assert!(confirm_activated(&home, "2026.9.2"));
        for _ in 0..50 {
            if !home.join(PREV_DIR).exists() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        cleanup(home);
    }

    #[test]
    fn every_interrupted_activation_boundary_restores_the_old_runtime() {
        for completed_renames in 0..=RUNTIME_ENTRIES.len() * 2 {
            let home = temp_home(&format!("crash-{completed_renames}"));
            make_runtime(&home, "old");
            stage_ready(&home, &versions(None));
            let stage = home.join(STAGE_DIR);
            let prev = home.join(PREV_DIR);
            fs::create_dir_all(&prev).unwrap();
            let journal = ActivationJournal {
                versions: versions(None),
                previous_entries: RUNTIME_ENTRIES.iter().map(|name| (*name).to_string()).collect(),
            };
            write_journal(&prev.join(PENDING_MARKER), &journal).unwrap();

            let mut completed = 0;
            for name in RUNTIME_ENTRIES {
                if completed == completed_renames {
                    break;
                }
                fs::rename(home.join(name), prev.join(name)).unwrap();
                completed += 1;
                if completed == completed_renames {
                    break;
                }
                fs::rename(stage.join(name), home.join(name)).unwrap();
                completed += 1;
            }

            roll_back_unconfirmed_with(&home, false).unwrap();

            for name in RUNTIME_ENTRIES {
                assert_eq!(tag(&home, name), "old", "boundary {completed_renames}: {name}");
            }
            assert!(!prev.exists());
            cleanup(home);
        }
    }

    #[test]
    fn repeating_an_interrupted_rollback_keeps_the_restored_runtime() {
        let home = temp_home("rollback-retry");
        make_runtime(&home, "old");
        stage_ready(&home, &versions(None));
        let stage = home.join(STAGE_DIR);
        let prev = home.join(PREV_DIR);
        fs::create_dir_all(&prev).unwrap();
        let previous_entries: Vec<String> =
            RUNTIME_ENTRIES.iter().map(|name| (*name).to_string()).collect();
        write_journal(
            &prev.join(PENDING_MARKER),
            &ActivationJournal {
                versions: versions(None),
                previous_entries: previous_entries.clone(),
            },
        )
        .unwrap();
        swap_entries(&home, &stage, &prev, false).unwrap();

        let first_trash = restore_previous_runtime(&home, &prev, &previous_entries).unwrap();
        let second_trash = restore_previous_runtime(&home, &prev, &previous_entries).unwrap();

        for name in RUNTIME_ENTRIES {
            assert_eq!(tag(&home, name), "old", "{name}");
        }
        cleanup(first_trash);
        cleanup(second_trash);
        cleanup(home);
    }

    #[test]
    fn an_interrupted_rollback_marker_recreates_the_failed_version() {
        let home = temp_home("rollback-marker-recovery");
        make_runtime(&home, "old");
        let stage = home.join(STAGE_DIR);
        make_runtime(&stage, "bad");
        let prev = home.join(PREV_DIR);
        fs::create_dir_all(&prev).unwrap();
        write_versions(&prev.join(ROLLED_BACK_MARKER), &versions(None)).unwrap();

        roll_back_unconfirmed_with(&home, false).unwrap();

        assert_eq!(read_versions(&home.join(FAILED_MARKER)), Some(versions(None)));
        assert_eq!(tag(&home, "unsloth_studio"), "old");
        assert!(!stage.exists());
        assert!(!prev.exists());
        cleanup(home);
    }

    #[test]
    fn a_durable_confirmation_never_rolls_back_the_new_runtime() {
        let home = temp_home("confirmed-crash");
        make_runtime(&home, "old");
        stage_ready(&home, &versions(None));
        activate_ready(&home, "0.1.900-beta").unwrap();
        let prev = home.join(PREV_DIR);
        write_versions(&prev.join(CONFIRMED_MARKER), &versions(None)).unwrap();

        roll_back_unconfirmed_with(&home, false).unwrap();

        assert_eq!(tag(&home, "unsloth_studio"), "new");
        assert!(!prev.exists());
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
    fn native_helpers_activate_and_roll_back_with_the_python_runtime() {
        let container = temp_home("helpers");
        let home = container.join("studio");
        fs::create_dir_all(&home).unwrap();
        make_runtime(&home, "old");
        stage_ready(&home, &versions(None));
        for name in HELPER_RUNTIME_ENTRIES {
            fs::create_dir_all(container.join(name)).unwrap();
            fs::write(container.join(name).join("tag"), "old").unwrap();
            fs::create_dir_all(home.join(STAGE_DIR).join(name)).unwrap();
            fs::write(home.join(STAGE_DIR).join(name).join("tag"), "new").unwrap();
        }

        activate_ready(&home, "0.1.900-beta").unwrap();

        for name in HELPER_RUNTIME_ENTRIES {
            assert_eq!(tag(&container, name), "new", "{name}");
            assert_eq!(tag(&home.join(PREV_DIR), name), "old", "{name}");
        }

        roll_back_unconfirmed_with(&home, false).unwrap();

        for name in HELPER_RUNTIME_ENTRIES {
            assert_eq!(tag(&container, name), "old", "{name}");
        }
        cleanup(container);
    }

    #[test]
    fn undoing_an_activation_takes_the_staged_only_sidecars_back() {
        let home = temp_home("undo-extra");
        // A legacy install: the managed venv, and no tiered sidecars to swap back.
        fs::create_dir_all(home.join("unsloth_studio")).unwrap();
        fs::write(home.join("unsloth_studio").join("tag"), "old").unwrap();
        let stage = home.join(STAGE_DIR);
        make_runtime(&stage, "new");
        let prev = home.join(PREV_DIR);
        fs::create_dir_all(&prev).unwrap();

        swap_entries(&home, &stage, &prev, false).unwrap();
        assert_eq!(tag(&home, ".venv_t5_530"), "new");

        // What activate_ready runs when the pending marker cannot be written.
        swap_entries(&home, &prev, &stage, true).unwrap();

        assert_eq!(tag(&home, "unsloth_studio"), "old");
        for name in [".venv_t5_530", ".venv_t5_550", ".venv_t5_510"] {
            // No marker is left to undo these later, so they cannot stay live.
            assert!(!home.join(name).exists(), "{name}");
            assert_eq!(tag(&stage, name), "new", "{name}");
        }
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

    /// This process, whose real start time the guard can be measured against.
    fn live_pid_or_skip(home: &Path) -> Option<u32> {
        let me = std::process::id();
        if crate::process_identity::process_start_time_secs(me).is_none() {
            // The OS will not say, so the reuse guard cannot fire and neither can
            // the assertion built on it.
            fs::remove_dir_all(home).ok();
            return None;
        }
        Some(me)
    }

    #[test]
    fn a_startup_marker_counts_as_a_live_tree_record() {
        let home = temp_home("markers");
        let me = std::process::id();
        // What a backend has while it is still binding, and what it keeps after
        // dropping its pid records until shutdown finishes.
        fs::write(
            home.join(format!("studio-starting-{me}.marker")),
            format!("{me}\n"),
        )
        .unwrap();

        assert!(recorded_pids(&home).contains(&me));
        cleanup(home);
    }

    #[test]
    fn a_bare_record_cannot_resurrect_a_pid_the_timed_one_rejected() {
        let home = temp_home("reused");
        let Some(me) = live_pid_or_skip(&home) else {
            return;
        };
        // A start time nowhere near this process's: the record describes something
        // that is gone, and the pid has since been handed out again.
        fs::write(
            home.join(format!("studio-8888-{me}.pid")),
            format!("{me}\n1.0\n"),
        )
        .unwrap();
        fs::write(home.join("studio.pid"), format!("{me}\n")).unwrap();

        assert!(!recorded_pids(&home).contains(&me));
        cleanup(home);
    }

    #[test]
    fn a_bare_record_with_no_timed_evidence_still_counts() {
        let home = temp_home("bare");
        let me = std::process::id();
        // A pre-upgrade backend, or one whose per-port write failed.
        fs::write(home.join("studio.pid"), format!("{me}\n")).unwrap();

        assert!(recorded_pids(&home).contains(&me));
        cleanup(home);
    }

    #[test]
    fn a_stage_that_never_reached_the_pinned_backend_is_rejected() {
        let home = temp_home("version-gate");
        stage_ready(
            &home,
            &StagedVersions {
                // What a stale mirror leaves behind: setup succeeded from cache and
                // the cloned package never moved.
                backend_version: "2026.9.1".into(),
                shell_version: None,
            },
        );

        let err = staged_backend_meets(&home, "2026.9.2").unwrap_err();
        assert!(err.contains("2026.9.1"), "{err}");
        assert!(err.contains("2026.9.2"), "{err}");
        // Equal and newer both pass: the pin is a floor, not an exact match.
        assert!(staged_backend_meets(&home, "2026.9.1").is_ok());
        assert!(staged_backend_meets(&home, "2026.9.0").is_ok());
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
                staging: false,
                staging_shell_version: None,
            }
        );
        discard(&home);
        assert_eq!(status(&home).state, "none");
        cleanup(home);
    }
}
