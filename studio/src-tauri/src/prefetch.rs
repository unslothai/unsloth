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
    /// `_studio_prefetch`'s `core_plan`: the exact pins the swap will read from the
    /// cache. A marker from a build that did not record one is checked for payload only.
    #[serde(default)]
    core_plan: Option<std::collections::BTreeMap<String, String>>,
    /// `_studio_prefetch`'s per-requirement-file records: `{"pins": {...}}` for a file
    /// whose wheels were fetched. A pin cleaned from the cache is a download at restart.
    #[serde(default)]
    requirements: Option<std::collections::BTreeMap<String, RequirementRecord>>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct RequirementRecord {
    #[serde(default)]
    pins: Option<std::collections::BTreeMap<String, String>>,
    #[serde(default)]
    skipped_reason: Option<String>,
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

/// The uv cache directories that hold package bytes, and the files inside them
/// that are bookkeeping rather than payload. Mirrors `_uv_cache_has_packages` in
/// `unsloth_cli/commands/studio.py`, which mirrors `install.sh:_configure_uv_cache`:
/// `wheels-*` is metadata only on uv 0.10, so counting any file at all would read
/// a merely-resolved cache as warm.
const UV_CACHE_BUCKETS: [&str; 5] = ["archive-", "builds-", "built-wheels-", "wheels-", "sdists-"];
const UV_CACHE_BOOKKEEPING: [&str; 3] = ["CACHEDIR.TAG", ".git", ".gitignore"];
const UV_CACHE_METADATA_SUFFIXES: [&str; 4] = [".lock", ".msgpack", ".http", ".rev"];

fn dir_has_payload(dir: &Path, depth: usize) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        if path.is_dir() {
            if depth > 0 && dir_has_payload(&path, depth - 1) {
                return true;
            }
            continue;
        }
        if UV_CACHE_BOOKKEEPING.contains(&name.as_str())
            || UV_CACHE_METADATA_SUFFIXES
                .iter()
                .any(|suffix| name.ends_with(suffix))
        {
            continue;
        }
        return true;
    }
    false
}

/// Whether the cache a prefetch recorded still holds any package bytes.
///
/// `uv cache clean`, a moved `UV_CACHE_DIR` or a deleted Studio home leave the
/// marker behind with nothing under it; reporting that marker ready would promise
/// a fast restart the swap cannot keep.
pub(crate) fn cache_has_packages(cache_dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(cache_dir) else {
        return false;
    };
    entries
        .flatten()
        .filter(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            entry.path().is_dir()
                && UV_CACHE_BUCKETS
                    .iter()
                    .any(|bucket| name.starts_with(bucket))
        })
        .any(|entry| dir_has_payload(&entry.path(), 6))
}

/// Whether the cache still holds the unpacked wheel of `name==version`.
///
/// uv unpacks every wheel it installs from under `archive-v*/<id>/`, where the wheel's
/// `<name>-<version>.dist-info` directory sits at the top; `uv cache clean <package>`
/// removes exactly those entries. `cache_has_packages` cannot tell a cache that lost
/// unsloth from one that kept everything else, and a marker reported ready over such a
/// cache made Restart perform the download it had presented as done.
pub(crate) fn cache_holds_wheel(cache_dir: &Path, name: &str, version: &str) -> bool {
    cached_dist_infos(cache_dir).contains(&wanted_dist_info(name, version))
}

fn wanted_dist_info(name: &str, version: &str) -> String {
    // Compared normalised, not spelled: the dist-info keeps the wheel's own spelling
    // (Faker-20.1.0.dist-info for the plan's faker), and a case-sensitive filesystem
    // would otherwise report a cached wheel absent and the marker stale.
    normalized_dist_info(&format!("{}-{}.dist-info", name.trim(), version.trim()))
}

/// Every unpacked wheel's dist-info name under `archive-v*`, normalised, read once: a
/// status request checks every marker pin against it, where a walk per pin over a
/// large shared cache was pins times entries, polled every second after a reload.
fn cached_dist_infos(cache_dir: &Path) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    let Ok(buckets) = fs::read_dir(cache_dir) else {
        return names;
    };
    for bucket in buckets.flatten() {
        let bucket_name = bucket.file_name().to_string_lossy().into_owned();
        if !bucket_name.starts_with("archive-") || !bucket.path().is_dir() {
            continue;
        }
        let Ok(entries) = fs::read_dir(bucket.path()) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(children) = fs::read_dir(entry.path()) else {
                continue;
            };
            for child in children.flatten() {
                let child_name = child.file_name().to_string_lossy().into_owned();
                if child_name.ends_with(".dist-info") && child.path().is_dir() {
                    names.insert(normalized_dist_info(&child_name));
                }
            }
        }
    }
    names
}

/// PEP 503 spirit for a dist-info directory name: case-folded, with `-`, `_` and `.`
/// read alike, on both sides of the comparison.
fn normalized_dist_info(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace(['-', '.'], "_")
}

/// The cache the next `unsloth studio update` will read, as far as this process can
/// tell: an explicit UV_CACHE_DIR in the environment the CLI inherits, else the cache
/// the install recorded. None when neither is known (the CLI then chooses by content).
fn effective_update_cache(home: &Path, explicit_cache: Option<&str>) -> Option<PathBuf> {
    if let Some(explicit) = explicit_cache {
        let explicit = explicit.trim();
        if !explicit.is_empty() {
            return Some(PathBuf::from(explicit));
        }
    }
    let recorded = fs::read_to_string(home.join("cache").join("uv-cache-dir")).ok()?;
    let recorded = recorded.trim_start_matches('\u{feff}').trim();
    if recorded.is_empty() {
        return None;
    }
    Some(PathBuf::from(recorded))
}

/// Whether `expected` (the live UV_CACHE_DIR, or the install's record) names the
/// cache the marker recorded.
///
/// The marker records the cache resolved the way uv resolves it, against the setup
/// script's working directory; a relative UV_CACHE_DIR in this process's environment
/// is the same cache when the recorded absolute path ends with it. This process
/// cannot resolve the relative spelling itself: its working directory is not the one
/// the update's uv runs from.
fn same_cache(expected: &Path, recorded: &Path) -> bool {
    if comparable_cache_path(expected) == comparable_cache_path(recorded) {
        return true;
    }
    if expected.is_relative() && recorded.is_absolute() {
        // A leading `..` (which only the setup script's working directory could
        // resolve) is dropped by the fold, so what is left is the tail the recorded
        // absolute path has to end with.
        let relative = fold_lexically(expected);
        return !relative.as_os_str().is_empty() && recorded.ends_with(&relative);
    }
    false
}

/// `.` dropped and `..` folded into the component before it: the normalisation the
/// CLI applies (os.path.normpath) to what it records, applied here to what the
/// environment spells, so `/cache/../uv` and `/uv` name one cache.
fn fold_lexically(path: &Path) -> PathBuf {
    let mut folded = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                folded.pop();
            }
            other => folded.push(other.as_os_str()),
        }
    }
    folded
}

/// The folded path as one string: separators unified, no trailing separator, and on
/// Windows case-folded, since two spellings that differ only there open one directory.
fn comparable_cache_path(path: &Path) -> String {
    let text = fold_lexically(path).to_string_lossy().replace('\\', "/");
    let text = text.trim_end_matches('/').to_string();
    if cfg!(windows) {
        text.to_lowercase()
    } else {
        text
    }
}

/// Whether the marker on disk is past the age the status reports as stale.
pub fn marker_expired(home: &Path) -> bool {
    read_marker(home)
        .and_then(|marker| marker.created_at)
        .is_some_and(|created| now_ms().saturating_sub(created) > MAX_AGE_MS)
}

fn cache_holds_plan(
    cached: &std::collections::HashSet<String>,
    plan: &std::collections::BTreeMap<String, String>,
) -> bool {
    plan.iter()
        .all(|(name, version)| cached.contains(&wanted_dist_info(name, version)))
}

/// Every wheel the marker says it fetched: the core plan and each requirement file's
/// pins. A file recorded with a skipped_reason fetched nothing and is not held to
/// anything; `uv cache clean <package>` on any fetched pin makes the marker stale.
fn cache_holds_marker(cache_dir: &Path, marker: &PrefetchMarker) -> bool {
    let cached = cached_dist_infos(cache_dir);
    if let Some(plan) = marker.core_plan.as_ref() {
        if !cache_holds_plan(&cached, plan) {
            return false;
        }
    }
    if let Some(requirements) = marker.requirements.as_ref() {
        for record in requirements.values() {
            if record.skipped_reason.is_some() {
                continue;
            }
            if let Some(pins) = record.pins.as_ref() {
                if !cache_holds_plan(&cached, pins) {
                    return false;
                }
            }
        }
    }
    true
}

pub fn status(home: &Path) -> PrefetchStatus {
    let explicit = std::env::var("UV_CACHE_DIR").ok();
    status_for(home, explicit.as_deref())
}

/// `status` with the environment's UV_CACHE_DIR passed in, so a test host's own
/// cache setting cannot decide what a marker under a temporary home is worth.
fn status_for(home: &Path, explicit_cache: Option<&str>) -> PrefetchStatus {
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
    // `noop` prepared nothing and needs no cache. A marker that recorded no cache
    // directory at all is one this build did not write, and the schema check above
    // has already decided about that.
    let cache_cold = marker.state != "noop"
        && marker.cache_dir.as_deref().is_some_and(|dir| {
            let cache = Path::new(dir);
            !cache_has_packages(cache) || !cache_holds_marker(cache, &marker)
        });
    // A warm cache the update will not read is no preparation: UV_CACHE_DIR changed or
    // cleared since the prefetch, or the recorded install cache moved, and Restart
    // would download what the offer presented as done. The CLI checks the same before
    // it hands the pins over; this keeps the status honest about it.
    let cache_elsewhere = marker.state != "noop"
        && marker.cache_dir.as_deref().is_some_and(|dir| {
            effective_update_cache(home, explicit_cache)
                .is_some_and(|expected| !same_cache(&expected, Path::new(dir)))
        });
    let state =
        if marker.schema != MARKER_SCHEMA || !known || expired || cache_cold || cache_elsewhere {
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

    #[test]
    fn a_cached_wheel_is_found_under_the_wheels_own_spelling() {
        let home = temp_home("wheel-spelling");
        let cache = home.join("cache");
        let archive = cache.join("archive-v0").join("abc123");
        fs::create_dir_all(archive.join("Faker-20.1.0.dist-info")).unwrap();
        fs::create_dir_all(archive.join("ruamel.yaml-0.18.6.dist-info")).unwrap();
        assert!(cache_holds_wheel(&cache, "faker", "20.1.0"));
        assert!(cache_holds_wheel(&cache, "ruamel-yaml", "0.18.6"));
        assert!(!cache_holds_wheel(&cache, "faker", "20.1.1"));
        assert!(!cache_holds_wheel(&cache, "fakers", "20.1.0"));
        let _ = fs::remove_dir_all(&home);
    }

    #[test]
    fn a_marker_for_a_cache_the_update_will_not_read_is_stale() {
        let home = temp_home("cache-elsewhere");
        let warm = home.join("warm-cache");
        let archive = warm.join("archive-v0").join("id1");
        fs::create_dir_all(archive.join("unsloth-2026.9.5.dist-info")).unwrap();
        fs::write(
            archive.join("unsloth-2026.9.5.dist-info").join("RECORD"),
            "x",
        )
        .unwrap();
        fs::create_dir_all(home.join(".update-prefetch")).unwrap();
        fs::write(
            home.join(".update-prefetch").join(".unsloth-studio-owned"),
            "",
        )
        .unwrap();
        let marker = serde_json::json!({
            "schema": MARKER_SCHEMA,
            "state": "ready",
            "cache_dir": warm.to_string_lossy(),
            "core_plan": {"unsloth": "2026.9.5"},
            "created_at": now_ms(),
        });
        fs::write(
            home.join(".update-prefetch").join("PREFETCHED.json"),
            serde_json::to_vec(&marker).unwrap(),
        )
        .unwrap();
        // No record of another cache: the warm one stands.
        assert_eq!(status_for(&home, None).state, "ready");
        // The install recorded a different cache since: the update will read that one.
        fs::create_dir_all(home.join("cache")).unwrap();
        fs::write(
            home.join("cache").join("uv-cache-dir"),
            format!("{}\n", home.join("other").display()),
        )
        .unwrap();
        assert_eq!(status_for(&home, None).state, "stale");
        fs::write(
            home.join("cache").join("uv-cache-dir"),
            format!("{}/\n", warm.display()),
        )
        .unwrap();
        assert_eq!(status_for(&home, None).state, "ready");
        // An explicit UV_CACHE_DIR outranks the record, as it does for the CLI.
        assert_eq!(status_for(&home, Some("/somewhere/else")).state, "stale");
        assert_eq!(
            status_for(&home, Some(&warm.to_string_lossy())).state,
            "ready"
        );
        // A relative UV_CACHE_DIR: the marker recorded it resolved against the setup
        // script's directory, which this process cannot repeat; the recorded path
        // ending with the relative spelling is the same cache.
        assert_eq!(status_for(&home, Some("warm-cache")).state, "ready");
        assert_eq!(status_for(&home, Some("./warm-cache")).state, "ready");
        assert_eq!(status_for(&home, Some("other-cache")).state, "stale");
        // Parent components, which the CLI's normpath folded before recording.
        assert_eq!(status_for(&home, Some("../warm-cache")).state, "ready");
        assert_eq!(status_for(&home, Some("x/../warm-cache")).state, "ready");
        assert_eq!(status_for(&home, Some("../other-cache")).state, "stale");
        let _ = fs::remove_dir_all(&home);
    }

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
        assert_eq!(status_for(&home, None).state, "none");
        assert!(!discard(&home));
        fs::remove_dir_all(home.parent().unwrap()).unwrap();
    }

    /// A uv cache with one unpacked wheel in it, as a prefetch leaves it.
    fn warm_cache(home: &Path) -> PathBuf {
        let cache = home.join("cache").join("uv");
        let unpacked = cache.join("archive-v0").join("abc123").join("unsloth");
        fs::create_dir_all(&unpacked).unwrap();
        fs::write(unpacked.join("__init__.py"), b"").unwrap();
        // The layout uv unpacks a wheel into: its dist-info beside the package.
        fs::create_dir_all(
            cache
                .join("archive-v0")
                .join("abc123")
                .join("unsloth-2026.9.2.dist-info"),
        )
        .unwrap();
        fs::write(
            cache.join("CACHEDIR.TAG"),
            b"Signature: 8a477f597d28d172789f06886806bc55",
        )
        .unwrap();
        cache
    }

    #[test]
    fn a_ready_marker_is_reported_with_the_versions_it_recorded() {
        let home = temp_home("ready");
        let cache = warm_cache(&home);
        write_prefetch(
            &home,
            serde_json::json!({
                "schema": 1,
                "state": "ready",
                "backend_version": "2026.9.2",
                "shell_version": "0.1.900-beta",
                "cache_dir": cache.to_string_lossy(),
                "created_at": now_ms(),
            }),
            true,
        );

        let status = status_for(&home, None);
        assert_eq!(status.state, "ready");
        assert_eq!(status.backend_version.as_deref(), Some("2026.9.2"));
        assert_eq!(status.shell_version.as_deref(), Some("0.1.900-beta"));
        assert_eq!(
            status.cache_dir.as_deref(),
            Some(cache.to_string_lossy().as_ref())
        );
        fs::remove_dir_all(home.parent().unwrap()).unwrap();
    }

    /// `uv cache clean`, a moved UV_CACHE_DIR or a deleted Studio home leave the
    /// marker behind with nothing under it. Reporting it ready would present
    /// Restart and then do at restart every download this feature moved off it.
    #[test]
    fn a_marker_whose_cache_no_longer_holds_packages_is_stale() {
        for (name, prepare) in [
            ("cache-gone", None),
            ("cache-cleaned", Some("empty")),
            ("cache-metadata-only", Some("metadata")),
        ] {
            let home = temp_home(name);
            let cache = home.join("cache").join("uv");
            match prepare {
                None => {}
                Some("empty") => fs::create_dir_all(&cache).unwrap(),
                Some(_) => {
                    // wheels-* holds only resolution metadata on uv 0.10, and the lock
                    // and msgpack files under archive-v0 are bookkeeping, not bytes.
                    let bucket = cache.join("archive-v0").join("abc123");
                    fs::create_dir_all(&bucket).unwrap();
                    fs::write(bucket.join(".lock"), b"").unwrap();
                    fs::write(cache.join("archive-v0").join("index.msgpack"), b"").unwrap();
                    fs::create_dir_all(cache.join("wheels-v5")).unwrap();
                    fs::write(cache.join("CACHEDIR.TAG"), b"").unwrap();
                }
            }
            write_prefetch(
                &home,
                serde_json::json!({
                    "schema": 1,
                    "state": "ready",
                    "shell_version": "0.1.900-beta",
                    "cache_dir": cache.to_string_lossy(),
                    "created_at": now_ms(),
                }),
                true,
            );
            assert_eq!(status_for(&home, None).state, "stale", "{name}");
            fs::remove_dir_all(home.parent().unwrap()).unwrap();
        }
    }

    /// `uv cache clean unsloth` removes one package's entries and leaves the rest of
    /// the cache warm. The marker's plan names that package, so it is stale; a marker
    /// that recorded no plan (an older build) is held to payload only, as before.
    #[test]
    fn a_plan_whose_wheel_was_cleaned_from_the_cache_is_stale() {
        let home = temp_home("plan-cleaned");
        let cache = warm_cache(&home);
        let marker = |plan: serde_json::Value| {
            let mut body = serde_json::json!({
                "schema": 1,
                "state": "ready",
                "backend_version": "2026.9.2",
                "shell_version": "0.1.900-beta",
                "cache_dir": cache.to_string_lossy(),
                "created_at": now_ms(),
            });
            if !plan.is_null() {
                body["core_plan"] = plan;
            }
            body
        };
        write_prefetch(
            &home,
            marker(serde_json::json!({"unsloth": "2026.9.2"})),
            true,
        );
        assert_eq!(status_for(&home, None).state, "ready");
        // A second pin the cache never held: not ready.
        write_prefetch(
            &home,
            marker(serde_json::json!({"unsloth": "2026.9.2", "unsloth-zoo": "2026.9.1"})),
            true,
        );
        assert_eq!(status_for(&home, None).state, "stale");
        // Normalisation: the plan spells the name with a dash, the dist-info with an underscore.
        fs::create_dir_all(
            cache
                .join("archive-v0")
                .join("def456")
                .join("unsloth_zoo-2026.9.1.dist-info"),
        )
        .unwrap();
        assert_eq!(status_for(&home, None).state, "ready");
        // The planned wheel cleaned away while unrelated payload remains: stale.
        fs::remove_dir_all(cache.join("archive-v0").join("abc123")).unwrap();
        fs::create_dir_all(cache.join("archive-v0").join("other").join("numpy")).unwrap();
        fs::write(
            cache
                .join("archive-v0")
                .join("other")
                .join("numpy")
                .join("x.so"),
            b"",
        )
        .unwrap();
        assert_eq!(status_for(&home, None).state, "stale");
        // No plan recorded: payload alone still answers.
        write_prefetch(&home, marker(serde_json::Value::Null), true);
        assert_eq!(status_for(&home, None).state, "ready");
        fs::remove_dir_all(home.parent().unwrap()).unwrap();
    }

    /// The requirement files' pins are fetched wheels too, and `uv cache clean
    /// diffusers` removes one of them as readily as it removes unsloth.
    #[test]
    fn a_requirement_pin_cleaned_from_the_cache_is_stale() {
        let home = temp_home("req-cleaned");
        let cache = warm_cache(&home);
        let marker = |requirements: serde_json::Value| {
            serde_json::json!({
                "schema": 1,
                "state": "ready",
                "backend_version": "2026.9.2",
                "shell_version": "0.1.900-beta",
                "cache_dir": cache.to_string_lossy(),
                "created_at": now_ms(),
                "core_plan": {"unsloth": "2026.9.2"},
                "requirements": requirements,
            })
        };
        write_prefetch(
            &home,
            marker(serde_json::json!({"studio.txt": {"pins": {"diffusers": "0.40.0"}}})),
            true,
        );
        assert_eq!(status_for(&home, None).state, "stale");
        fs::create_dir_all(
            cache
                .join("archive-v0")
                .join("ghi789")
                .join("diffusers-0.40.0.dist-info"),
        )
        .unwrap();
        assert_eq!(status_for(&home, None).state, "ready");
        // A file left to swap time fetched nothing and is held to nothing.
        write_prefetch(
            &home,
            marker(serde_json::json!({
                "studio.txt": {"pins": {"diffusers": "0.40.0"}},
                "base.txt": {"pins": {"numpy": "9.9.9"}, "skipped_reason": "resolve failed: x"}
            })),
            true,
        );
        assert_eq!(status_for(&home, None).state, "ready");
        fs::remove_dir_all(home.parent().unwrap()).unwrap();
    }

    #[test]
    fn a_noop_marker_needs_no_cache() {
        let home = temp_home("noop");
        write_prefetch(
            &home,
            serde_json::json!({
                "schema": 1,
                "state": "noop",
                "cache_dir": home.join("nowhere").to_string_lossy(),
                "created_at": now_ms(),
            }),
            true,
        );
        assert_eq!(status_for(&home, None).state, "noop");
        fs::remove_dir_all(home.parent().unwrap()).unwrap();
    }

    #[test]
    fn a_partial_prefetch_is_still_usable_and_says_so() {
        let home = temp_home("partial");
        let cache = warm_cache(&home);
        write_prefetch(
            &home,
            serde_json::json!({
                "schema": 1,
                "state": "partial",
                "cache_dir": cache.to_string_lossy(),
                "created_at": now_ms(),
            }),
            true,
        );
        assert_eq!(status_for(&home, None).state, "partial");
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
            assert_eq!(status_for(&home, None).state, "stale");
            fs::remove_dir_all(home.parent().unwrap()).unwrap();
        }
    }

    #[test]
    fn an_unreadable_marker_reports_none_rather_than_guessing() {
        let home = temp_home("garbage");
        let root = prefetch_dir(&home);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join(MARKER_NAME), b"{not json").unwrap();
        assert_eq!(status_for(&home, None).state, "none");
        fs::remove_dir_all(home.parent().unwrap()).unwrap();
    }

    #[test]
    fn an_absolute_cache_spelled_with_dots_is_the_recorded_cache() {
        assert!(same_cache(Path::new("/tmp/x/cache/../uv"), Path::new("/tmp/x/uv")));
        assert!(same_cache(Path::new("/tmp/x/./uv/"), Path::new("/tmp/x/uv")));
        assert!(!same_cache(Path::new("/tmp/x/other"), Path::new("/tmp/x/uv")));
        // Relative spellings still match on their folded tail only.
        assert!(same_cache(Path::new("./cache/../uv"), Path::new("/tmp/x/uv")));
        assert!(!same_cache(Path::new("../"), Path::new("/tmp/x/uv")));
    }

    #[test]
    fn a_marker_older_than_the_ceiling_reads_as_expired() {
        let home = temp_home("expired-marker");
        assert!(!marker_expired(&home));
        write_prefetch(
            &home,
            serde_json::json!({"schema": MARKER_SCHEMA, "state": "ready", "created_at": now_ms() - MAX_AGE_MS - 1}),
            true,
        );
        assert!(marker_expired(&home));
        write_prefetch(
            &home,
            serde_json::json!({"schema": MARKER_SCHEMA, "state": "ready", "created_at": now_ms()}),
            true,
        );
        assert!(!marker_expired(&home));
        std::fs::remove_dir_all(&home).unwrap();
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
        assert_eq!(status_for(&home, None).state, "none");
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
