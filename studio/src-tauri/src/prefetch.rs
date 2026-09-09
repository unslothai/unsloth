//! Reads what `unsloth studio prefetch-update` left under `.update-prefetch/`.
//!
//! The CLI does the work; this side only reports it and cleans it up. Nothing
//! here activates anything: the swap is the ordinary update, which finds the
//! wheels in the uv cache the prefetch warmed and downloads nothing.

use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// `_studio_prefetch.PREFETCH_DIR_NAME`.
pub const PREFETCH_DIR_NAME: &str = ".update-prefetch";
/// `_studio_prefetch.MARKER_NAME`, written last and atomically, so its presence
/// is the statement that every wheel it names is cached.
const MARKER_NAME: &str = "PREFETCHED.json";
/// `_studio_prefetch.OWNED_MARKER`. Written before anything else lands in the
/// directory, so an interrupted prefetch still leaves one we may remove.
const OWNED_MARKER: &str = ".unsloth-studio-owned";
/// `_studio_prefetch.MARKER_SCHEMA`.
const MARKER_SCHEMA: u64 = 1;

/// A prefetch older than this is reported stale. The pins it recorded resolved
/// against an index that has had a week to move, and the update would re-resolve
/// them anyway; saying "ready" for it would promise a fast restart we cannot keep.
const MAX_AGE_MS: i64 = 7 * 24 * 60 * 60 * 1000;

#[derive(Clone, Debug, Deserialize)]
struct PrefetchMarker {
    #[serde(default)]
    schema: u64,
    #[serde(default)]
    state: String,
    #[serde(default)]
    backend_version: Option<String>,
    #[serde(default)]
    shell_version: Option<String>,
    #[serde(default)]
    cache_dir: Option<String>,
    #[serde(default)]
    created_at: Option<i64>,
}

/// `none` no prefetch on disk; `ready` everything it planned is cached; `noop`
/// there was nothing to prepare; `partial` the core packages are cached and some
/// requirement file was left to swap time; `stale` a marker this build will not
/// act on (wrong schema, unknown state, or too old).
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PrefetchStatus {
    pub state: String,
    pub backend_version: Option<String>,
    pub shell_version: Option<String>,
    pub cache_dir: Option<String>,
    pub created_at: Option<i64>,
    /// A prefetch child is running now, so this is being written, not stale.
    pub running: bool,
    /// The offer that running child is preparing for, when there is one.
    pub running_shell_version: Option<String>,
}

fn prefetch_dir(home: &Path) -> PathBuf {
    home.join(PREFETCH_DIR_NAME)
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as i64)
        .unwrap_or(0)
}

fn read_marker(home: &Path) -> Option<PrefetchMarker> {
    let raw = fs::read_to_string(prefetch_dir(home).join(MARKER_NAME)).ok()?;
    match serde_json::from_str::<PrefetchMarker>(&raw) {
        Ok(marker) => Some(marker),
        Err(error) => {
            warn!("[prefetch] Could not read the prefetch marker: {error}");
            None
        }
    }
}

pub fn status(home: &Path) -> PrefetchStatus {
    let Some(marker) = read_marker(home) else {
        return PrefetchStatus {
            state: "none".to_string(),
            ..PrefetchStatus::default()
        };
    };
    let known = matches!(marker.state.as_str(), "ready" | "noop" | "partial");
    // A clock that went backwards reads as a negative age, which is not old.
    let expired = marker
        .created_at
        .is_some_and(|created| now_ms().saturating_sub(created) > MAX_AGE_MS);
    let state = if marker.schema != MARKER_SCHEMA || !known || expired {
        "stale"
    } else {
        marker.state.as_str()
    };
    PrefetchStatus {
        state: state.to_string(),
        backend_version: marker.backend_version,
        shell_version: marker.shell_version,
        cache_dir: marker.cache_dir,
        created_at: marker.created_at,
        running: false,
        running_shell_version: None,
    }
}

/// Remove the prefetch directory, but only one Unsloth wrote.
///
/// Same rule the installer applies before it deletes any tree: without the owned
/// marker this is somebody else's directory that happens to share the name, and
/// the cost of leaving it is disk, while the cost of removing it is theirs.
pub fn discard(home: &Path) -> bool {
    let root = prefetch_dir(home);
    if !root.exists() {
        return false;
    }
    if !root.join(OWNED_MARKER).is_file() {
        warn!(
            "[prefetch] {} was not created by Unsloth; leaving it",
            root.display()
        );
        return false;
    }
    match fs::remove_dir_all(&root) {
        Ok(()) => {
            info!("[prefetch] Discarded the prepared update");
            true
        }
        Err(error) => {
            warn!("[prefetch] Could not discard the prepared update: {error}");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_home(name: &str) -> PathBuf {
        let home = std::env::temp_dir()
            .join(format!(
                "unsloth-prefetch-{name}-{}-{}",
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

    fn write_prefetch(home: &Path, body: serde_json::Value, owned: bool) {
        let root = prefetch_dir(home);
        fs::create_dir_all(&root).unwrap();
        if owned {
            fs::write(root.join(OWNED_MARKER), b"").unwrap();
        }
        fs::write(
            root.join(MARKER_NAME),
            serde_json::to_vec_pretty(&body).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn an_absent_prefetch_reports_none() {
        let home = temp_home("absent");
        assert_eq!(status(&home).state, "none");
        assert!(!discard(&home));
        fs::remove_dir_all(home.parent().unwrap()).unwrap();
    }

    #[test]
    fn a_ready_marker_is_reported_with_the_versions_it_recorded() {
        let home = temp_home("ready");
        write_prefetch(
            &home,
            serde_json::json!({
                "schema": 1,
                "state": "ready",
                "backend_version": "2026.9.2",
                "shell_version": "0.1.900-beta",
                "cache_dir": "/home/u/.unsloth/studio/cache/uv",
                "created_at": now_ms(),
            }),
            true,
        );

        let status = status(&home);
        assert_eq!(status.state, "ready");
        assert_eq!(status.backend_version.as_deref(), Some("2026.9.2"));
        assert_eq!(status.shell_version.as_deref(), Some("0.1.900-beta"));
        assert_eq!(
            status.cache_dir.as_deref(),
            Some("/home/u/.unsloth/studio/cache/uv")
        );
        fs::remove_dir_all(home.parent().unwrap()).unwrap();
    }

    #[test]
    fn a_partial_prefetch_is_still_usable_and_says_so() {
        let home = temp_home("partial");
        write_prefetch(
            &home,
            serde_json::json!({"schema": 1, "state": "partial", "created_at": now_ms()}),
            true,
        );
        assert_eq!(status(&home).state, "partial");
        fs::remove_dir_all(home.parent().unwrap()).unwrap();
    }

    #[test]
    fn a_future_schema_an_unknown_state_and_an_old_marker_are_all_stale() {
        for body in [
            serde_json::json!({"schema": 2, "state": "ready", "created_at": now_ms()}),
            serde_json::json!({"schema": 1, "state": "half", "created_at": now_ms()}),
            serde_json::json!({"schema": 1, "state": "ready", "created_at": now_ms() - MAX_AGE_MS - 1}),
        ] {
            let home = temp_home("stale");
            write_prefetch(&home, body, true);
            assert_eq!(status(&home).state, "stale");
            fs::remove_dir_all(home.parent().unwrap()).unwrap();
        }
    }

    #[test]
    fn an_unreadable_marker_reports_none_rather_than_guessing() {
        let home = temp_home("garbage");
        let root = prefetch_dir(&home);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join(MARKER_NAME), b"{not json").unwrap();
        assert_eq!(status(&home).state, "none");
        fs::remove_dir_all(home.parent().unwrap()).unwrap();
    }

    #[test]
    fn discard_removes_an_owned_directory_and_is_repeatable() {
        let home = temp_home("discard");
        write_prefetch(
            &home,
            serde_json::json!({"schema": 1, "state": "ready", "created_at": now_ms()}),
            true,
        );
        fs::create_dir_all(prefetch_dir(&home).join("site")).unwrap();

        assert!(discard(&home));
        assert!(!prefetch_dir(&home).exists());
        assert!(!discard(&home));
        assert_eq!(status(&home).state, "none");
        fs::remove_dir_all(home.parent().unwrap()).unwrap();
    }

    #[test]
    fn a_directory_unsloth_did_not_create_is_left_alone() {
        let home = temp_home("foreign");
        write_prefetch(
            &home,
            serde_json::json!({"schema": 1, "state": "ready", "created_at": now_ms()}),
            false,
        );

        assert!(!discard(&home));
        assert!(prefetch_dir(&home).join(MARKER_NAME).is_file());
        fs::remove_dir_all(home.parent().unwrap()).unwrap();
    }
}
