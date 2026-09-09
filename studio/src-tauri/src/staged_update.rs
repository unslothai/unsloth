//! Cleans up what the 805-807 background update left on disk. Nothing here
//! activates a stage.
//!
//! Those releases cloned the managed runtime into `.update-stage`, swapped it in
//! at the next launch and kept the replaced trees in `.update-prev` until the new
//! backend answered a health probe. Background staging is gone, but the on-disk
//! leftovers are not: an install that took a 805-807 update can still be carrying
//! a half-written stage, an activated-but-unconfirmed runtime, or the rollback
//! trash a previous launch never got to delete. Every step below is a rename or a
//! delete, so running it twice does nothing the first run did not already do.

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

/// The 807 journal also carried `backend_version` and `shell_version`. Unknown
/// fields are ignored, so a journal written by any of 805-807 still parses and
/// still names the entries this launch has to put back.
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
    // Written by the old shell's fail-fast path and by this wheel's refusal, and
    // read by nothing here any more.
    let _ = fs::remove_file(home.join(FAILED_MARKER));
    // Before the rollback, not after: a READY stage must never be activated, and
    // a rollback that still saw one would leave the entries it did not restore
    // live beside the restored runtime.
    let _ = fs::remove_dir_all(home.join(STAGE_DIR));
    if let Err(error) = roll_back_unconfirmed(home) {
        warn!("[staged-update] could not restore the previous runtime: {error}");
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

fn roll_back_unconfirmed_with(home: &Path, in_use: bool) -> Result<(), String> {
    let prev = home.join(PREV_DIR);
    if prev.join(CONFIRMED_MARKER).is_file() {
        // 807 vouched for the runtime that is live now. Keep it and drop the copy.
        remove_confirmed_previous(&prev);
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
        // Renaming the tree under a live process is unsafe, and this runtime was
        // never confirmed, so neither answer is safe to force now. Leave the
        // marker and decide at the next launch.
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

/// An entry the previous runtime did not have is moved out of the way rather than
/// left: a legacy install with no tiered sidecars has nothing to put back for
/// them, so leaving them would have the restored backend importing from an
/// environment nothing ever confirmed.
fn restore_previous_runtime(
    home: &Path,
    previous: &Path,
    previous_entries: &[String],
) -> Result<PathBuf, String> {
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

    const READY_MARKER: &str = "READY.json";

    /// Always one level below a private container: `live_entry` resolves the
    /// native helpers against the parent, so a home directly under the system
    /// temp directory would have the rollback renaming whatever `node` or
    /// `llama.cpp` happens to sit beside it.
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

    /// Two code paths drop a tree on a background thread, and Windows refuses to
    /// remove a directory while another handle is still walking inside it, so a
    /// teardown that races one of them fails with ERROR_ACCESS_DENIED rather than
    /// telling anyone anything. Retry until the deleter is done.
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

    /// What 805-807 did at activation: live trees into `.update-prev`, staged
    /// trees into their place, a PENDING journal naming what was displaced.
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

        assert_eq!(tag(&home, "unsloth_studio"), "old");
        assert!(!home.join(STAGE_DIR).exists());
        assert!(!home.join(PREV_DIR).exists());

        // Safe to repeat: nothing left to do and nothing undone.
        reconcile_legacy_at_launch(&home);
        assert_eq!(tag(&home, "unsloth_studio"), "old");
        cleanup(home);
    }

    #[test]
    fn a_retained_desktop_update_bundle_survives_the_cleanup() {
        // The classic update installs this bundle after the backend step, and it
        // lives in the same directory as everything cleaned up above.
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

        // The restore already ran; this launch neither redoes it nor leaves a
        // marker for a status nobody reports any more.
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

        assert_eq!(tag(&home, "unsloth_studio"), "new");
        assert!(!prev.exists());
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
}
