//! Cleans up what the 805-807 background update left on disk. Nothing here activates a stage.
//!
//! Those releases cloned the managed runtime into `.update-stage`, swapped it in at the next
//! launch and kept the replaced trees in `.update-prev` until a health probe confirmed the new
//! backend. An install that took such an update can still carry a half-written stage, an
//! unconfirmed runtime, or rollback trash. Every step below is a rename or a delete, so
//! running it twice does nothing the first run did not.

use crate::process_identity::ProcessOrigin;
use log::{info, warn};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

const STAGE_DIR: &str = ".update-stage";
const PREV_DIR: &str = ".update-prev";
const FAILED_MARKER: &str = ".update-failed.json";
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

/// The 807 journal also carried `backend_version` and `shell_version`. Unknown fields are
/// ignored, so a journal from any of 805-807 still parses.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq)]
struct ActivationJournal {
    #[serde(default)]
    previous_entries: Vec<String>,
}

fn read_journal(path: &Path) -> Option<ActivationJournal> {
    serde_json::from_str(&fs::read_to_string(path).ok()?).ok()
}

/// Undo whatever a 805-807 background update left behind, then never stage again.
pub(crate) fn reconcile_legacy_at_launch(home: &Path) {
    remove_stale_trash(home);
    // Written by the old shell's fail-fast path and by this wheel's refusal; read by nothing here.
    let _ = fs::remove_file(home.join(FAILED_MARKER));
    // Before the rollback, not after: a READY stage must never be activated, and a rollback that
    // still saw one would leave unrestored entries live beside the restored runtime.
    discard_stage(home);
    if let Err(error) = roll_back_unconfirmed(home) {
        warn!("[staged-update] could not restore the previous runtime: {error}");
    }
}

/// Settle a deferred legacy rollback before an update mutates the live runtime.
///
/// `reconcile_legacy_at_launch` leaves a PENDING journal alone while a backend is on the tree,
/// which is right at launch and wrong afterwards: a classic update would install into a runtime
/// the journal still names as something to undo, and the next idle launch would restore the
/// pre-update trees over it. Refusing beats updating a runtime about to be replaced by a backup.
pub(crate) fn reconcile_before_update(home: &Path) -> Result<(), String> {
    // Nothing a 805-807 update left behind, so nothing to probe for.
    if !home.join(PREV_DIR).is_dir() && !home.join(STAGE_DIR).exists() {
        return Ok(());
    }
    reconcile_before_update_with(home, live_tree_in_use(home))
}

fn reconcile_before_update_with(home: &Path, in_use: bool) -> Result<(), String> {
    // Restarting is the whole fix: an idle launch finishes the rollback itself, and on POSIX the
    // backend this update would replace is usually the one holding the tree.
    const RESTART: &str =
        "An unfinished background update from an earlier release is still waiting on a \
         running backend. Quit Unsloth Studio, reopen it, and update again.";
    if in_use {
        return Err(RESTART.to_string());
    }
    discard_stage(home);
    roll_back_unconfirmed_with(home, false)?;
    if home.join(PREV_DIR).join(PENDING_MARKER).is_file() {
        return Err(RESTART.to_string());
    }
    Ok(())
}

/// A rename, not a delete: `.update-stage` holds a clone of the managed venv and every native
/// helper, and unlinking a torch tree here would hold the runtime gate through all of setup.
/// Move it into the trash namespace a later launch sweeps anyway.
fn discard_stage(home: &Path) {
    discard_stage_with(home, |from, to| fs::rename(from, to));
}

fn discard_stage_with(home: &Path, rename: impl Fn(&Path, &Path) -> std::io::Result<()>) {
    let stage = home.join(STAGE_DIR);
    if !stage.exists() {
        return;
    }
    let trash = trash_path(home, "stage");
    if rename(&stage, &trash).is_ok() {
        std::thread::spawn(move || {
            let _ = fs::remove_dir_all(trash);
        });
        return;
    }
    // Also off the launch path. On Windows the rename fails exactly when a file inside is still
    // open, which is the multi-gigabyte case the background delete exists for. Nothing activates
    // a stage any more, so a tree that outlives this call is inert.
    std::thread::spawn(move || {
        let _ = fs::remove_dir_all(stage);
    });
}

/// Distinct within a launch as well as between launches: a coarse clock can hand two calls the
/// same reading, and the second rename would land on the first call's trash directory.
static TRASH_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn trash_path(home: &Path, label: &str) -> PathBuf {
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let sequence = TRASH_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    home.join(format!(
        "{ROLLBACK_TRASH_PREFIX}{label}-{}-{suffix}-{sequence}",
        std::process::id()
    ))
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

/// Quarantine the confirmed backup, then unlink it off that path.
///
/// A delete in place would take entries out from under the marker vouching for the live runtime,
/// and an interrupted one would leave a `.update-prev` the next launch reads as an unconfirmed
/// activation and rolls back to. The rename is the whole decision, and it is atomic.
fn quarantine_confirmed_previous(home: &Path, prev: PathBuf) {
    let trash = trash_path(home, "confirmed");
    if fs::rename(&prev, &trash).is_err() {
        // Still confirmed and consistent, and every step here repeats safely: leave it for the
        // next launch rather than start a delete that could strand the marker.
        warn!("[staged-update] could not quarantine the confirmed backup, leaving it for the next launch");
        return;
    }
    std::thread::spawn(move || {
        let _ = fs::remove_dir_all(trash);
    });
}

fn roll_back_unconfirmed(home: &Path) -> Result<(), String> {
    // `live_tree_in_use` reads every pid record and, on Windows, probes the managed environment.
    // An install that never took a 805-807 update has nothing to decide and must not pay for it.
    if !home.join(PREV_DIR).is_dir() {
        return Ok(());
    }
    roll_back_unconfirmed_with(home, live_tree_in_use(home))
}

fn roll_back_unconfirmed_with(home: &Path, in_use: bool) -> Result<(), String> {
    let prev = home.join(PREV_DIR);
    if prev.join(CONFIRMED_MARKER).is_file() {
        // 807 vouched for the runtime that is live now. Keep it and drop the copy off the launch path.
        quarantine_confirmed_previous(home, prev);
        return Ok(());
    }
    if prev.join(ROLLED_BACK_MARKER).is_file() {
        // The restore already happened; only the bookkeeping is left.
        let _ = fs::remove_dir_all(&prev);
        return Ok(());
    }
    let journal = read_journal(&prev.join(PENDING_MARKER));
    let has_previous_runtime = all_runtime_entries().any(|name| prev.join(name).exists());
    if journal.is_none() && !has_previous_runtime {
        if prev.is_dir() {
            let _ = fs::remove_dir_all(&prev);
        }
        return Ok(());
    }
    if in_use {
        // Renaming the tree under a live process is unsafe and this runtime was never confirmed,
        // so leave the marker and decide at the next launch.
        info!("[staged-update] runtime still in use, deferring the rollback decision");
        return Ok(());
    }
    let previous_entries = journal
        .map(|journal| journal.previous_entries)
        .filter(|entries| !entries.is_empty())
        .unwrap_or_else(|| {
            all_runtime_entries()
                .filter(|name| prev.join(name).exists())
                .map(str::to_string)
                .collect()
        });
    info!("[staged-update] restoring the runtime a background update replaced");
    let trash = restore_previous_runtime(home, &prev, &previous_entries)?;
    let _ = fs::remove_file(prev.join(PENDING_MARKER));
    let _ = fs::remove_dir_all(&prev);
    std::thread::spawn(move || {
        let _ = fs::remove_dir_all(trash);
    });
    Ok(())
}

/// An entry the previous runtime did not have is moved aside rather than left: a legacy install
/// has nothing to put back for tiered sidecars, so the restored backend would otherwise import
/// from an environment nothing ever confirmed.
fn restore_previous_runtime(
    home: &Path,
    previous: &Path,
    previous_entries: &[String],
) -> Result<PathBuf, String> {
    let trash = trash_path(home, "rollback");
    fs::create_dir_all(&trash).map_err(|e| e.to_string())?;
    let mut moved: Vec<(PathBuf, PathBuf)> = Vec::new();
    let result = (|| {
        for name in all_runtime_entries() {
            let old = previous.join(name);
            let live = live_entry(home, name);
            if previous_entries.iter().any(|entry| entry == name) {
                if old.exists() {
                    if live.exists() {
                        rename_tracked(&live, &trash.join(name), &mut moved)?;
                    }
                    rename_tracked(&old, &live, &mut moved)?;
                }
            } else if live.exists() {
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
/// Mirrors `live_sibling_backend` in studio/backend/run.py and reads all three record kinds,
/// since a backend may have only one: a startup marker while binding, a per-port record once
/// bound, a bare `studio.pid` otherwise. Markers matter most: the backend keeps its marker after
/// dropping its pid records, and renaming the tree in that window moves it under a live importer.
fn recorded_pids(home: &Path) -> Vec<u32> {
    let mut pids = Vec::new();
    let mut timed = Vec::new();
    if let Ok(entries) = fs::read_dir(home) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            // Markers share the per-port record layout, so both carry the start time.
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
            // Judged either way, so the untimed legacy record must not resurrect this pid.
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
            // It carries no start time, so alone it would re-add a pid proved reused.
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

    const READY_MARKER: &str = "READY.json";

    /// Always one level below a private container: `live_entry` resolves the native helpers
    /// against the parent, so a home directly under the temp directory would have the rollback
    /// renaming whatever sits beside it.
    fn temp_home(name: &str) -> PathBuf {
        let home = std::env::temp_dir()
            .join(format!(
                "unsloth-staged-update-{name}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ))
            .join("studio");
        fs::create_dir_all(&home).unwrap();
        home
    }

    /// Two code paths drop a tree on a background thread, and Windows refuses to remove a
    /// directory while another handle walks inside it, so a racing teardown fails with
    /// ERROR_ACCESS_DENIED. Retry until the deleter is done.
    fn cleanup(home: PathBuf) {
        for _ in 0..100 {
            if fs::remove_dir_all(&home).is_ok() || !home.exists() {
                let _ = home.parent().map(fs::remove_dir_all);
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        fs::remove_dir_all(&home).unwrap();
    }

    fn wait_gone(path: &Path) {
        for _ in 0..250 {
            if !path.exists() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
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

    /// The marker bodies 805-807 wrote, versions and all.
    fn write_marker(path: &Path, previous_entries: &[String]) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let body = serde_json::json!({
            "backend_version": "2026.9.1",
            "shell_version": "0.1.900-beta",
            "previous_entries": previous_entries,
        });
        fs::write(path, serde_json::to_vec_pretty(&body).unwrap()).unwrap();
    }

    fn stage_ready(home: &Path) {
        let stage = home.join(STAGE_DIR);
        make_runtime(&stage, "new");
        write_marker(&stage.join(READY_MARKER), &[]);
    }

    /// What 805-807 did at activation: live trees into `.update-prev`, staged trees into their
    /// place, a PENDING journal naming what was displaced.
    fn activate_by_hand(home: &Path) -> Vec<String> {
        let stage = home.join(STAGE_DIR);
        let prev = home.join(PREV_DIR);
        fs::create_dir_all(&prev).unwrap();
        let previous_entries: Vec<String> = all_runtime_entries()
            .filter(|name| live_entry(home, name).exists())
            .map(str::to_string)
            .collect();
        write_marker(&prev.join(PENDING_MARKER), &previous_entries);
        for name in all_runtime_entries() {
            let staged = stage.join(name);
            let live = live_entry(home, name);
            if !staged.exists() {
                continue;
            }
            if live.exists() {
                fs::rename(&live, prev.join(name)).unwrap();
            }
            fs::rename(&staged, &live).unwrap();
        }
        let _ = fs::remove_dir_all(&stage);
        previous_entries
    }

    #[test]
    fn a_ready_stage_is_deleted_and_never_activated() {
        let home = temp_home("ready-stage");
        make_runtime(&home, "old");
        stage_ready(&home);

        reconcile_legacy_at_launch(&home);

        // Renamed aside rather than unlinked, so the stage is unreachable once the call returns.
        assert!(!home.join(STAGE_DIR).exists());
        assert_eq!(tag(&home, "unsloth_studio"), "old");
        assert!(!home.join(PREV_DIR).exists());

        // Safe to repeat: nothing left to do and nothing undone.
        reconcile_legacy_at_launch(&home);
        assert_eq!(tag(&home, "unsloth_studio"), "old");
        cleanup(home);
    }

    #[test]
    fn a_ready_stage_is_discarded_before_the_rollback_it_would_otherwise_survive() {
        // 807 crashed mid-swap: the managed venv is the staged one, the sidecars are not, and the
        // stage still holds the entries the swap never reached.
        let home = temp_home("ready-stage-and-pending");
        make_runtime(&home, "old");
        stage_ready(&home);
        let stage = home.join(STAGE_DIR);
        let prev = home.join(PREV_DIR);
        let previous_entries: Vec<String> = RUNTIME_ENTRIES
            .iter()
            .map(|name| (*name).to_string())
            .collect();
        write_marker(&prev.join(PENDING_MARKER), &previous_entries);
        fs::rename(home.join("unsloth_studio"), prev.join("unsloth_studio")).unwrap();
        fs::rename(stage.join("unsloth_studio"), home.join("unsloth_studio")).unwrap();
        assert_eq!(tag(&home, "unsloth_studio"), "new");
        assert!(stage.join(".venv_t5_530").exists());

        reconcile_legacy_at_launch(&home);

        // The discard runs first, so the rollback cannot leave a fourth staged entry live.
        for name in RUNTIME_ENTRIES {
            assert_eq!(tag(&home, name), "old", "{name}");
        }
        assert!(!stage.exists());
        assert!(!prev.exists());
        cleanup(home);
    }

    #[test]
    fn a_stage_that_cannot_be_renamed_is_still_swept() {
        let home = temp_home("stage-rename-fails");
        make_runtime(&home, "old");
        stage_ready(&home);
        let stage = home.join(STAGE_DIR);

        // Windows refuses the rename while a file inside is open, the one case the background
        // delete exists for, so the fallback runs off the launch path too.
        discard_stage_with(&home, |_, _| Err(std::io::Error::other("rename refused")));

        wait_gone(&stage);
        assert!(!stage.exists());
        assert_eq!(tag(&home, "unsloth_studio"), "old");
        cleanup(home);
    }

    #[test]
    fn two_trash_names_in_one_launch_never_collide() {
        let home = temp_home("trash-names");

        let first = trash_path(&home, "stage");
        let second = trash_path(&home, "stage");

        assert_ne!(first, second);
        for path in [&first, &second] {
            let name = path.file_name().unwrap().to_str().unwrap();
            // Still swept by `remove_stale_trash`, which matches on the prefix alone.
            assert!(name.starts_with(ROLLBACK_TRASH_PREFIX), "{name}");
        }
        cleanup(home);
    }

    #[test]
    fn a_retained_desktop_update_bundle_survives_the_cleanup() {
        // The classic update installs this bundle after the backend step, in the same directory.
        let home = temp_home("bundle");
        make_runtime(&home, "old");
        stage_ready(&home);
        for name in [".desktop-update-bundle", ".desktop-update-bundle.json"] {
            fs::write(home.join(name), b"bundle").unwrap();
        }

        reconcile_legacy_at_launch(&home);

        for name in [".desktop-update-bundle", ".desktop-update-bundle.json"] {
            assert_eq!(fs::read(home.join(name)).unwrap(), b"bundle", "{name}");
        }
        cleanup(home);
    }

    #[test]
    fn a_half_written_stage_is_removed() {
        let home = temp_home("partial-stage");
        make_runtime(&home, "old");
        make_runtime(&home.join(STAGE_DIR), "half");

        reconcile_legacy_at_launch(&home);

        assert_eq!(tag(&home, "unsloth_studio"), "old");
        assert!(!home.join(STAGE_DIR).exists());
        cleanup(home);
    }

    #[test]
    fn the_compatibility_failure_marker_is_removed_at_launch() {
        let home = temp_home("failed-marker");
        make_runtime(&home, "old");
        // What an 807 shell's `--stage` attempt against this wheel leaves behind.
        fs::write(
            home.join(FAILED_MARKER),
            r#"{"backend_version": "2026.9.1", "shell_version": "0.1.807-beta"}"#,
        )
        .unwrap();

        reconcile_legacy_at_launch(&home);

        assert!(!home.join(FAILED_MARKER).exists());
        assert_eq!(tag(&home, "unsloth_studio"), "old");
        cleanup(home);
    }

    #[test]
    fn an_unconfirmed_activation_is_rolled_back_at_launch() {
        let home = temp_home("rollback");
        make_runtime(&home, "old");
        stage_ready(&home);
        activate_by_hand(&home);
        assert_eq!(tag(&home, "unsloth_studio"), "new");

        reconcile_legacy_at_launch(&home);

        assert_eq!(tag(&home, "unsloth_studio"), "old");
        assert_eq!(tag(&home, ".venv_t5_530"), "old");
        assert!(!home.join(PREV_DIR).exists());
        assert!(!home.join(FAILED_MARKER).exists());

        reconcile_legacy_at_launch(&home);
        assert_eq!(tag(&home, "unsloth_studio"), "old");
        cleanup(home);
    }

    #[test]
    fn an_807_journal_with_extra_fields_still_names_the_entries_to_restore() {
        let home = temp_home("journal-compat");
        make_runtime(&home, "old");
        let prev = home.join(PREV_DIR);
        fs::create_dir_all(&prev).unwrap();
        let body = r#"{
            "backend_version": "2026.9.1",
            "shell_version": "0.1.807-beta",
            "previous_entries": ["unsloth_studio", ".venv_t5_530"],
            "unknown_807_field": 7
        }"#;
        fs::write(prev.join(PENDING_MARKER), body).unwrap();

        let journal = read_journal(&prev.join(PENDING_MARKER)).unwrap();

        assert_eq!(journal.previous_entries, ["unsloth_studio", ".venv_t5_530"]);
        cleanup(home);
    }

    #[test]
    fn every_interrupted_activation_boundary_restores_the_old_runtime() {
        for completed_renames in 0..=RUNTIME_ENTRIES.len() * 2 {
            let home = temp_home(&format!("crash-{completed_renames}"));
            make_runtime(&home, "old");
            stage_ready(&home);
            let stage = home.join(STAGE_DIR);
            let prev = home.join(PREV_DIR);
            let previous_entries: Vec<String> =
                RUNTIME_ENTRIES.iter().map(|name| (*name).to_string()).collect();
            write_marker(&prev.join(PENDING_MARKER), &previous_entries);

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
            let _ = fs::remove_dir_all(&stage);

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
        stage_ready(&home);
        let previous_entries = activate_by_hand(&home);
        let prev = home.join(PREV_DIR);

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
    fn an_interrupted_rollback_marker_only_drops_the_bookkeeping() {
        let home = temp_home("rollback-marker-recovery");
        make_runtime(&home, "old");
        let prev = home.join(PREV_DIR);
        write_marker(&prev.join(ROLLED_BACK_MARKER), &[]);

        reconcile_legacy_at_launch(&home);

        // The restore already ran; this launch neither redoes it nor leaves a marker.
        assert!(!home.join(FAILED_MARKER).exists());
        assert_eq!(tag(&home, "unsloth_studio"), "old");
        assert!(!prev.exists());
        cleanup(home);
    }

    #[test]
    fn a_durable_confirmation_never_rolls_back_the_new_runtime() {
        let home = temp_home("confirmed-crash");
        make_runtime(&home, "old");
        stage_ready(&home);
        activate_by_hand(&home);
        let prev = home.join(PREV_DIR);
        write_marker(&prev.join(CONFIRMED_MARKER), &[]);

        reconcile_legacy_at_launch(&home);

        // The superseded runtime is dropped on a background thread.
        wait_gone(&prev);
        assert_eq!(tag(&home, "unsloth_studio"), "new");
        assert!(!prev.exists());
        cleanup(home);
    }

    #[test]
    fn a_confirmed_backup_leaves_the_directory_the_moment_it_is_dropped() {
        let home = temp_home("confirmed-quarantine");
        make_runtime(&home, "old");
        stage_ready(&home);
        activate_by_hand(&home);
        let prev = home.join(PREV_DIR);
        write_marker(&prev.join(CONFIRMED_MARKER), &[]);

        roll_back_unconfirmed_with(&home, false).unwrap();

        // Renamed, not emptied: a delete in place would take entries out from under the
        // confirmation, and an interrupted one would leave a backup with no marker.
        assert!(!prev.exists());
        assert_eq!(tag(&home, "unsloth_studio"), "new");

        reconcile_legacy_at_launch(&home);
        assert_eq!(tag(&home, "unsloth_studio"), "new");
        cleanup(home);
    }

    #[test]
    fn a_deferred_rollback_is_settled_before_an_update_touches_the_runtime() {
        let home = temp_home("before-update");
        make_runtime(&home, "old");
        stage_ready(&home);
        activate_by_hand(&home);
        // The launch found a backend on the tree and left the decision for later.
        roll_back_unconfirmed_with(&home, true).unwrap();
        assert!(home.join(PREV_DIR).join(PENDING_MARKER).is_file());

        reconcile_before_update_with(&home, false).unwrap();

        // Settled here, so no later launch can put this backup back over the update.
        assert_eq!(tag(&home, "unsloth_studio"), "old");
        assert!(!home.join(PREV_DIR).exists());
        cleanup(home);
    }

    #[test]
    fn an_update_is_refused_while_a_deferred_rollback_cannot_be_settled() {
        let home = temp_home("before-update-busy");
        make_runtime(&home, "old");
        stage_ready(&home);
        activate_by_hand(&home);

        let error = reconcile_before_update_with(&home, true).unwrap_err();

        // Restarting is what settles it, so the message says that and not a path.
        assert!(error.contains("Quit Unsloth Studio"), "{error}");
        // Nothing moved, so the launch after this one still has its decision to make.
        assert_eq!(tag(&home, "unsloth_studio"), "new");
        assert!(home.join(PREV_DIR).join(PENDING_MARKER).is_file());
        cleanup(home);
    }

    #[test]
    fn an_install_that_never_staged_has_nothing_to_settle_before_an_update() {
        let home = temp_home("before-update-clean");
        make_runtime(&home, "old");

        reconcile_before_update(&home).unwrap();

        assert_eq!(tag(&home, "unsloth_studio"), "old");
        cleanup(home);
    }

    #[test]
    fn rollback_takes_back_sidecars_the_failed_update_added() {
        let home = temp_home("rollback-extra");
        // A legacy install: the managed venv is there, the tiered sidecars are not.
        fs::create_dir_all(home.join("unsloth_studio")).unwrap();
        fs::write(home.join("unsloth_studio").join("tag"), "old").unwrap();
        // The 807 update built all of them and swapped them in.
        stage_ready(&home);
        activate_by_hand(&home);
        assert_eq!(tag(&home, "unsloth_studio"), "new");
        assert_eq!(tag(&home, ".venv_t5_530"), "new");

        reconcile_legacy_at_launch(&home);

        assert_eq!(tag(&home, "unsloth_studio"), "old");
        for name in [".venv_t5_530", ".venv_t5_550", ".venv_t5_510"] {
            // The restored backend must not find the unconfirmed update's sidecars.
            assert!(!home.join(name).exists(), "{name}");
        }
        cleanup(home);
    }

    #[test]
    fn native_helpers_roll_back_with_the_python_runtime() {
        let home = temp_home("helpers");
        let container = home.parent().unwrap().to_path_buf();
        make_runtime(&home, "old");
        stage_ready(&home);
        for name in HELPER_RUNTIME_ENTRIES {
            fs::create_dir_all(container.join(name)).unwrap();
            fs::write(container.join(name).join("tag"), "old").unwrap();
            fs::create_dir_all(home.join(STAGE_DIR).join(name)).unwrap();
            fs::write(home.join(STAGE_DIR).join(name).join("tag"), "new").unwrap();
        }
        activate_by_hand(&home);
        for name in HELPER_RUNTIME_ENTRIES {
            assert_eq!(tag(&container, name), "new", "{name}");
        }

        roll_back_unconfirmed_with(&home, false).unwrap();

        for name in HELPER_RUNTIME_ENTRIES {
            assert_eq!(tag(&container, name), "old", "{name}");
        }
        cleanup(home);
    }

    #[test]
    fn a_runtime_still_in_use_defers_to_the_next_launch() {
        let home = temp_home("defer");
        make_runtime(&home, "old");
        stage_ready(&home);
        activate_by_hand(&home);
        assert_eq!(tag(&home, "unsloth_studio"), "new");

        // Force-closed after activation: a backend is alive on the new tree, so
        // renaming it out from under that process is not an option.
        roll_back_unconfirmed_with(&home, true).unwrap();

        assert_eq!(tag(&home, "unsloth_studio"), "new");
        assert!(home.join(PREV_DIR).join(PENDING_MARKER).is_file());

        // Once nothing holds the tree, the rollback it was owed still happens.
        roll_back_unconfirmed_with(&home, false).unwrap();

        assert_eq!(tag(&home, "unsloth_studio"), "old");
        assert!(!home.join(PREV_DIR).exists());
        cleanup(home);
    }

    #[test]
    fn stale_rollback_trash_is_removed_at_launch() {
        let home = temp_home("trash");
        make_runtime(&home, "old");
        let trash = home.join(format!("{ROLLBACK_TRASH_PREFIX}1"));
        fs::create_dir_all(trash.join("unsloth_studio")).unwrap();

        reconcile_legacy_at_launch(&home);
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
            // The OS will not say, so neither the reuse guard nor its assertion can fire.
            fs::remove_dir_all(home).ok();
            return None;
        }
        Some(me)
    }

    #[test]
    fn a_startup_marker_counts_as_a_live_tree_record() {
        let home = temp_home("markers");
        let me = std::process::id();
        // What a backend has while binding, and keeps after dropping its pid records.
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
        // A start time nowhere near this process: the pid has since been handed out again.
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
}
