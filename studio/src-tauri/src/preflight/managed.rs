use super::types::ManagedProbe;
use super::version::{
    managed_backend_version_stale_reason, DESKTOP_MANAGEABILITY_VERSION, DESKTOP_PROTOCOL_VERSION,
};
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant, UNIX_EPOCH};
use tokio::io::AsyncReadExt;

// 3: the cached capability gained studio_install_ok / studio_install_reason.
// 4: the fingerprint gained the llama.cpp runtime, so a quarantined file
//    invalidates the entry instead of being served a stale Ready.
const MANAGED_CAPABILITY_CACHE_SCHEMA: u16 = 4;

/// The install is fine; the directory its children must run from is not reachable.
pub(super) const WORKING_DIRECTORY_UNAVAILABLE: &str = "working_directory_unavailable";
/// The profile is reachable but a user-written path setting is not resolvable,
/// so reinstalling hits the same wall. Mirrored in the frontend message map.
pub(super) const PATH_SETTING_UNRESOLVABLE: &str = "path_setting_unresolvable";

/// The reason a managed context failure is reported under, with the setting that
/// caused it where there is one: "one of Unsloth's folder settings" is not
/// something a user can act on, and every pin failure names the setting it could
/// not preserve. The name only, never the value, since this reaches the window.
pub(super) fn context_reason(error: &crate::process::ManagedContextError) -> String {
    match error {
        crate::process::ManagedContextError::WorkingDirectory(_) => {
            WORKING_DIRECTORY_UNAVAILABLE.to_string()
        }
        crate::process::ManagedContextError::PathSetting(detail) => {
            match setting_name(detail) {
                Some(name) => format!("{PATH_SETTING_UNRESOLVABLE}:{name}"),
                None => PATH_SETTING_UNRESOLVABLE.to_string(),
            }
        }
    }
}

/// The leading token of a pin failure, when it looks like an environment name.
fn setting_name(detail: &str) -> Option<&str> {
    let name = detail.split_whitespace().next()?;
    let named = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
    named.then_some(name)
}

/// Whether the reason is a context the app cannot build, not a repairable install.
pub(super) fn is_context_reason(reason: &str) -> bool {
    let head = reason.split(':').next().unwrap_or(reason);
    head == WORKING_DIRECTORY_UNAVAILABLE || head == PATH_SETTING_UNRESOLVABLE
}

const FNV64_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV64_PRIME: u64 = 0x100000001b3;
const HASHED_MARKER_MAX_BYTES: u64 = 64 * 1024;

const FALLBACK_MARKER_NAMES: &[&str] = &[
    // In the fingerprint, not just the cached answer: a repair touching only
    // studio.txt leaves every other marker alone, so a cache entry written
    // while healthy would outlive the dropped manifest. Mirrors MANIFEST_NAME.
    "unsloth_install_manifest.json",
    "pyvenv.cfg",
    "uv.lock",
    "requirements.txt",
    "python.exe",
    "python",
];

#[derive(Debug, Clone, Deserialize, Serialize)]
struct DesktopCapability {
    desktop_protocol_version: Option<u16>,
    desktop_manageability_version: Option<u16>,
    supports_api_only: Option<bool>,
    supports_provision_desktop_auth: Option<bool>,
    supports_desktop_backend_ownership: Option<bool>,
    desktop_auth_stale_reason: Option<String>,
    // A part-way install leaves a CLI that answers `-h` and a backend that dies
    // on `import structlog`, so a running CLI does not mean ready.
    studio_install_ok: Option<bool>,
    studio_install_reason: Option<String>,
    // Smart App Control and antivirus quarantine files out of an otherwise
    // present llama.cpp tree, and nothing repaired that: staleness was decided
    // on the managed Python alone, so the desktop launched happily and the
    // model load failed later looking like a bad GGUF. None when the CLI is too
    // old to answer or nothing is installed yet.
    llama_runtime_ok: Option<bool>,
    llama_runtime_reason: Option<String>,
    version: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ManagedCapabilityCache {
    schema: u16,
    bin_path: String,
    bin_size: u64,
    bin_mtime_ms: u64,
    studio_root_id: Option<String>,
    marker_path: Option<String>,
    marker_size: Option<u64>,
    marker_mtime_ms: Option<u64>,
    llama_runtime: Option<String>,
    desktop_protocol_version: u16,
    desktop_manageability_version: u16,
    capability: DesktopCapability,
}

#[derive(Debug, Clone)]
struct MarkerFingerprint {
    path: String,
    size: u64,
    mtime_ms: u64,
    content_hash: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ManagedBinFingerprint {
    bin_path: String,
    bin_size: u64,
    bin_mtime_ms: u64,
    studio_root_id: Option<String>,
    marker_path: Option<String>,
    marker_size: Option<u64>,
    marker_mtime_ms: Option<u64>,
    llama_runtime: Option<String>,
}

fn modified_ms(metadata: &fs::Metadata) -> Option<u64> {
    metadata
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|duration| u64::try_from(duration.as_millis()).ok())
}
fn hash_bytes(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |mut next, byte| {
        next ^= u64::from(*byte);
        next.wrapping_mul(FNV64_PRIME)
    })
}

fn marker_content_hash(path: &Path, metadata: &fs::Metadata) -> Option<u64> {
    if metadata.len() > HASHED_MARKER_MAX_BYTES {
        return None;
    }
    fs::read(path)
        .ok()
        .map(|bytes| hash_bytes(FNV64_OFFSET_BASIS, &bytes))
}

fn site_packages_dirs(venv_dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    #[cfg(unix)]
    {
        if let Ok(lib_dir) = fs::read_dir(venv_dir.join("lib")) {
            for entry in lib_dir.flatten() {
                out.push(entry.path().join("site-packages"));
            }
        }
    }
    out.push(venv_dir.join("Lib").join("site-packages"));
    // read_dir order is unspecified and the hashes below fold in order.
    out.sort();
    out
}

/// Hash of the .dist-info / .egg-info names present, version included.
///
/// pip uninstall rewrites nothing else that is fingerprinted, so a venv that
/// lost a studio.txt dependency would keep serving the healthy verdict.
fn installed_distributions_hash(site_packages: &Path) -> Option<u64> {
    let mut names: Vec<String> = fs::read_dir(site_packages)
        .ok()?
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            (name.ends_with(".dist-info") || name.ends_with(".egg-info")).then_some(name)
        })
        .collect();
    names.sort();
    Some(names.iter().fold(FNV64_OFFSET_BASIS, |hash, name| {
        hash_bytes(hash, name.as_bytes())
    }))
}

fn marker_candidates_for_bin(bin: &Path) -> Vec<PathBuf> {
    let Some(scripts_dir) = bin.parent() else {
        return Vec::new();
    };
    let Some(venv_dir) = scripts_dir.parent() else {
        return Vec::new();
    };
    let mut out = Vec::new();

    for site_packages in site_packages_dirs(venv_dir) {
        out.push(
            site_packages
                .join("unsloth_cli")
                .join("commands")
                .join("studio.py"),
        );
    }
    for marker_name in FALLBACK_MARKER_NAMES {
        out.push(venv_dir.join(marker_name));
        out.push(scripts_dir.join(marker_name));
    }
    out
}

/// The file whose size and mtime stand for "this CLI", which is the launcher when there
/// is one.
///
/// On Windows there may not be. Antivirus quarantine deletes the generated unsloth.exe
/// and leaves a venv that still runs through its interpreter, and since that is now a
/// supported layout, find_unsloth_binary_in_studio_dir hands back a launcher path that
/// does not exist. Keying the fingerprint on it would fail fs::metadata, so the
/// capability cache could be neither read nor written, and every preflight would pay
/// both the -h and the desktop-capabilities subprocess with their own 10s ceilings.
/// python.exe is the right stand-in: it is what actually starts the CLI there, and an
/// update replaces the whole venv, so it moves when the launcher would have.
fn fingerprint_identity_file(bin: &Path) -> Option<PathBuf> {
    if bin.exists() {
        return Some(bin.to_path_buf());
    }
    #[cfg(windows)]
    {
        let interpreter = bin.parent()?.join("python.exe");
        if interpreter.exists() {
            return Some(interpreter);
        }
    }
    None
}

fn managed_bin_fingerprint(bin: &Path) -> Option<ManagedBinFingerprint> {
    // Metadata from whatever identifies this CLI, but the cache key below stays the
    // launcher path, so the two layouts of one install cannot collide.
    let bin_metadata = fs::metadata(fingerprint_identity_file(bin)?).ok()?;
    let bin_path = bin
        .canonicalize()
        .unwrap_or_else(|_| bin.to_path_buf())
        .to_string_lossy()
        .into_owned();

    let studio_root_id = crate::desktop_backend_owner::read_expected_studio_root_id();
    let mut marker_entries: Vec<MarkerFingerprint> = marker_candidates_for_bin(bin)
        .into_iter()
        .filter_map(|path| {
            let metadata = fs::metadata(&path).ok()?;
            Some(MarkerFingerprint {
                path: path
                    .canonicalize()
                    .unwrap_or(path.clone())
                    .to_string_lossy()
                    .into_owned(),
                size: metadata.len(),
                mtime_ms: modified_ms(&metadata)?,
                content_hash: marker_content_hash(&path, &metadata),
            })
        })
        .collect();
    marker_entries.sort_by(|left, right| left.path.cmp(&right.path));
    let mut marker_hash = marker_entries
        .iter()
        .fold(FNV64_OFFSET_BASIS, |hash, marker| {
            let next = hash_bytes(hash, marker.path.as_bytes());
            let next = hash_bytes(next, &marker.size.to_le_bytes());
            let next = hash_bytes(next, &marker.mtime_ms.to_le_bytes());
            if let Some(content_hash) = marker.content_hash {
                hash_bytes(next, &content_hash.to_le_bytes())
            } else {
                next
            }
        });
    let mut tracked = marker_entries.len();
    if let Some(venv_dir) = bin.parent().and_then(Path::parent) {
        for site_packages in site_packages_dirs(venv_dir) {
            let Some(dist_hash) = installed_distributions_hash(&site_packages) else {
                continue;
            };
            marker_hash = hash_bytes(marker_hash, site_packages.to_string_lossy().as_bytes());
            marker_hash = hash_bytes(marker_hash, &dist_hash.to_le_bytes());
            tracked += 1;
        }
    }
    let marker_path = (tracked > 0).then(|| "markers".to_string());
    let marker_size = (tracked > 0).then_some(tracked as u64);
    let marker_mtime_ms = (tracked > 0).then_some(marker_hash);

    Some(ManagedBinFingerprint {
        bin_path,
        bin_size: bin_metadata.len(),
        bin_mtime_ms: modified_ms(&bin_metadata)?,
        studio_root_id,
        marker_path,
        marker_size,
        marker_mtime_ms,
        llama_runtime: llama_runtime_fingerprint(),
    })
}

/// The folder a leading `~` names, resolved the way the CLI child resolves it.
///
/// `ntpath.expanduser` answers USERPROFILE and `relative_override_pins` resolves
/// the child's tilde through it for that reason. `dirs::home_dir()` reads the
/// known folder instead, which a portable or overridden profile moves, so taking
/// it here would fingerprint a tree the child never looks at and the cache would
/// then survive a quarantine in the tree it does. Same order as process.rs.
fn tilde_home() -> Option<PathBuf> {
    if cfg!(windows) {
        if let Some(profile) = std::env::var_os("USERPROFILE") {
            if !profile.is_empty() {
                return Some(PathBuf::from(profile));
            }
        }
    }
    dirs::home_dir()
}

/// `UNSLOTH_LLAMA_CPP_PATH` resolved the way the CLI child sees it: trimmed, a
/// leading `~` expanded because `default_managed_llama_dir` calls expanduser,
/// and a relative value anchored to this process's working directory because
/// `relative_override_pins` pins it from exactly there before the spawn.
///
/// `~name` is the one shape left alone: nothing here can resolve another user's
/// home. It is then relative, so it drops out below rather than naming a folder
/// called "~name" beside the desktop.
///
/// UNSLOTH_STUDIO_HOME is deliberately not consulted even though the Python
/// resolver honours it: managed spawns scrub it (MANAGED_CHILD_SCRUBBED_ENV), so
/// the CLI answering the capability probe falls through to the legacy root too.
#[cfg_attr(test, allow(dead_code))]
fn llama_runtime_override() -> Option<PathBuf> {
    llama_runtime_override_from(
        std::env::var("UNSLOTH_LLAMA_CPP_PATH").ok().as_deref(),
        tilde_home().as_deref(),
        std::env::current_dir().ok().as_deref(),
    )
}

/// The resolution itself, split out from the reads so the tests can drive it
/// without setting a process-wide variable that the rest of the crate reads.
fn llama_runtime_override_from(
    value: Option<&str>,
    home: Option<&Path>,
    cwd: Option<&Path>,
) -> Option<PathBuf> {
    let value = value?.trim();
    if value.is_empty() {
        return None;
    }
    if value == "~" {
        return home.map(Path::to_path_buf);
    }
    if let Some(rest) = value
        .strip_prefix("~/")
        .or_else(|| cfg!(windows).then(|| value.strip_prefix("~\\")).flatten())
    {
        return Some(home?.join(rest));
    }
    let path = PathBuf::from(value);
    if path.is_absolute() {
        return Some(path);
    }
    // Anchored, not discarded. Dropping it looks safer but is not: the caller
    // then has no override to fall back from, both halves of a launch pair
    // report None, and None matches None, so the cache keeps hitting while the
    // real runtime rots. The child gets this value joined to this same
    // directory, so joining it here watches the tree the child was told about.
    Some(cwd?.join(path))
}

/// The managed llama.cpp install root, the same one default_managed_llama_dir
/// picks in Python.
fn llama_runtime_root() -> Option<PathBuf> {
    // Hermetic under test, and not merely overridable: this walks a real
    // directory, so a developer who happens to have a runtime installed and a CI
    // runner that does not would otherwise run different tests, and every
    // fingerprint assertion in this module would depend on the home directory.
    // Unset means no runtime at all. Mirrors capability_cache_path()'s hook.
    #[cfg(test)]
    {
        return std::env::var_os("UNSLOTH_TEST_LLAMA_RUNTIME_ROOT").map(PathBuf::from);
    }
    // The override is asked for first and its answer is final. Falling back to
    // the legacy tree when it is set but could not be resolved would fingerprint
    // a directory the CLI is not reporting on, so a quarantine in the tree the
    // user actually configured would leave the key unchanged.
    #[cfg(not(test))]
    if std::env::var_os("UNSLOTH_LLAMA_CPP_PATH").is_some_and(|value| !value.is_empty()) {
        return llama_runtime_override();
    }
    #[cfg(not(test))]
    return Some(dirs::home_dir()?.join(".unsloth").join("llama.cpp"));
}

/// A cheap stand-in for "the llama.cpp runtime tree is unchanged": how many
/// files sit in its binary directory and how many bytes they total.
///
/// The rest of this fingerprint covers the managed venv only, so a file
/// quarantined out of the runtime left it identical, the cache hit, and
/// preflight answered Ready without ever asking the CLI. Losing a file changes
/// both halves of this. None when no runtime is installed, which is a
/// NotInstalled case rather than a broken one.
fn llama_runtime_fingerprint() -> Option<String> {
    llama_runtime_fingerprint_at(&llama_runtime_root()?)
}

/// The walk itself, against a given root, so the tests need no shared state.
fn llama_runtime_fingerprint_at(root: &Path) -> Option<String> {
    let mut bin = root.join("build").join("bin");
    if cfg!(windows) {
        bin = bin.join("Release");
    }
    match fs::read_dir(&bin) {
        Ok(entries) => Some(format!("bin:{}", counted(entries))),
        // The binary directory is gone but something is still there. Reading that
        // as None too would make it identical to "nothing was ever installed",
        // and those are the two ends of the transition this cache has to catch: a
        // Ready cached while no runtime existed (installed_runtime_health finds no
        // marker, answers None, and llama_runtime_ok stays null, which is not
        // stale) still matched once a marker appeared over a missing build/bin,
        // so the CLI was never asked and never got to say
        // llama_runtime_dir_missing. Fingerprinting the root's own entries makes
        // the marker's arrival move it.
        Err(_) => fs::read_dir(root)
            .ok()
            .map(|entries| format!("nobin:{}", counted(entries))),
    }
}

/// How many files a directory holds and how many bytes they total. A directory is
/// not a file, so a stray subfolder cannot read as a binary.
fn counted(entries: fs::ReadDir) -> String {
    let mut count: u64 = 0;
    let mut bytes: u64 = 0;
    for entry in entries.flatten() {
        if let Ok(meta) = entry.metadata() {
            if meta.is_file() {
                count += 1;
                bytes += meta.len();
            }
        }
    }
    format!("{count}:{bytes}")
}

fn capability_cache_path() -> Option<PathBuf> {
    #[cfg(test)]
    if let Some(home) = std::env::var_os("UNSLOTH_TEST_DESKTOP_CAPABILITY_CACHE_HOME") {
        return Some(
            PathBuf::from(home)
                .join(".unsloth")
                .join("studio")
                .join("desktop_capability_cache.json"),
        );
    }

    dirs::home_dir().map(|home| {
        home.join(".unsloth")
            .join("studio")
            .join("desktop_capability_cache.json")
    })
}

fn cache_matches(cache: &ManagedCapabilityCache, fingerprint: &ManagedBinFingerprint) -> bool {
    cache.schema == MANAGED_CAPABILITY_CACHE_SCHEMA
        && cache.desktop_protocol_version == DESKTOP_PROTOCOL_VERSION
        && cache.desktop_manageability_version == DESKTOP_MANAGEABILITY_VERSION
        && cache.bin_path == fingerprint.bin_path
        && cache.bin_size == fingerprint.bin_size
        && cache.bin_mtime_ms == fingerprint.bin_mtime_ms
        && cache.studio_root_id == fingerprint.studio_root_id
        && cache.marker_path == fingerprint.marker_path
        && cache.marker_size == fingerprint.marker_size
        && cache.marker_mtime_ms == fingerprint.marker_mtime_ms
        && cache.llama_runtime == fingerprint.llama_runtime
        && desktop_capability_ready(&cache.capability)
}

fn read_cached_capability(fingerprint: &ManagedBinFingerprint) -> Option<DesktopCapability> {
    let path = capability_cache_path()?;
    let bytes = fs::read(path).ok()?;
    let cache = serde_json::from_slice::<ManagedCapabilityCache>(&bytes).ok()?;
    if cache_matches(&cache, fingerprint) {
        Some(cache.capability)
    } else {
        None
    }
}

fn write_cached_capability(fingerprint: &ManagedBinFingerprint, capability: &DesktopCapability) {
    let Some(path) = capability_cache_path() else {
        return;
    };
    let cache = ManagedCapabilityCache {
        schema: MANAGED_CAPABILITY_CACHE_SCHEMA,
        bin_path: fingerprint.bin_path.clone(),
        bin_size: fingerprint.bin_size,
        bin_mtime_ms: fingerprint.bin_mtime_ms,
        studio_root_id: fingerprint.studio_root_id.clone(),
        marker_path: fingerprint.marker_path.clone(),
        marker_size: fingerprint.marker_size,
        marker_mtime_ms: fingerprint.marker_mtime_ms,
        llama_runtime: fingerprint.llama_runtime.clone(),
        desktop_protocol_version: DESKTOP_PROTOCOL_VERSION,
        desktop_manageability_version: DESKTOP_MANAGEABILITY_VERSION,
        capability: capability.clone(),
    };
    if let Some(parent) = path.parent() {
        if fs::create_dir_all(parent).is_err() {
            return;
        }
    }
    let Ok(bytes) = serde_json::to_vec_pretty(&cache) else {
        return;
    };
    if let Err(error) = fs::write(&path, bytes) {
        warn!(
            "Managed preflight: could not write capability cache: {}",
            error
        );
    }
}

async fn run_cli_probe(bin: &Path, args: &[&str]) -> Result<bool, String> {
    let started = Instant::now();
    let Ok(mut cmd) = crate::process::build_managed_cli_command_tokio(bin, args) else {
        info!(
            "Managed preflight probe {:?} has no managed interpreter to run",
            args
        );
        // Ok, not Err: main's Err arm means "the probe could not be set up and the
        // install is untested". A venv with no interpreter beside the launcher IS a
        // result, and the same one this arm always gave.
        return Ok(false);
    };
    cmd.stdout(Stdio::null()).stderr(Stdio::null());

    // Reported, not folded into `false`: the CLI never ran, so calling it broken
    // would start a repair needing the same context. Re-checking afterwards is not
    // enough: a context that recovers in between makes an untested install look bad.
    if let Err(error) = crate::process::apply_managed_cli_context_tokio(&mut cmd) {
        info!(
            "Managed preflight probe {:?} has no usable working directory: {}",
            args, error
        );
        return Err(error);
    }

    #[cfg(target_os = "linux")]
    crate::process::scrub_appimage_python_env_tokio(&mut cmd);

    // Tauri uses the legacy root regardless of UNSLOTH_STUDIO_HOME / STUDIO_HOME;
    // probe subprocesses must follow the same isolation as process.rs.
    cmd.env_remove("UNSLOTH_STUDIO_HOME");
    cmd.env_remove("STUDIO_HOME");

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(crate::process::CREATE_NO_WINDOW);
    }

    let Ok(mut child) = crate::process::with_studio_runtime_launch_guard(|| {
        cmd.spawn().map_err(|error| error.to_string())
    }) else {
        info!(
            "Managed preflight probe {:?} failed to spawn in {}ms",
            args,
            started.elapsed().as_millis()
        );
        return Ok(false);
    };

    let ok = match tokio::time::timeout(Duration::from_secs(10), child.wait()).await {
        Ok(Ok(status)) => status.success(),
        _ => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            false
        }
    };
    info!(
        "Managed preflight probe {:?} finished ok={} in {}ms",
        args,
        ok,
        started.elapsed().as_millis()
    );
    Ok(ok)
}

async fn probe_cli_capability(bin: &Path) -> Result<Option<DesktopCapability>, String> {
    let started = Instant::now();
    let Ok(mut cmd) = crate::process::build_managed_cli_command_tokio(
        bin,
        &["studio", "desktop-capabilities", "--json"],
    ) else {
        info!("Managed desktop-capabilities probe has no managed interpreter to run");
        // As above: a missing interpreter is a verdict, not a failure to ask.
        return Ok(None);
    };
    cmd.stdout(Stdio::piped()).stderr(Stdio::null());

    // As above: a context that cannot be built is not a probe result.
    if let Err(error) = crate::process::apply_managed_cli_context_tokio(&mut cmd) {
        info!(
            "Managed desktop-capabilities probe has no usable working directory: {}",
            error
        );
        return Err(error);
    }

    #[cfg(target_os = "linux")]
    crate::process::scrub_appimage_python_env_tokio(&mut cmd);

    // Tauri uses the legacy root regardless of UNSLOTH_STUDIO_HOME / STUDIO_HOME;
    // probe subprocesses must follow the same isolation as process.rs.
    cmd.env_remove("UNSLOTH_STUDIO_HOME");
    cmd.env_remove("STUDIO_HOME");

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(crate::process::CREATE_NO_WINDOW);
    }

    let Ok(mut child) = crate::process::with_studio_runtime_launch_guard(|| {
        cmd.spawn().map_err(|error| error.to_string())
    }) else {
        info!(
            "Managed desktop-capabilities probe failed to spawn in {}ms",
            started.elapsed().as_millis()
        );
        return Ok(None);
    };
    let Some(mut stdout) = child.stdout.take() else {
        return Ok(None);
    };

    match tokio::time::timeout(Duration::from_secs(10), child.wait()).await {
        Ok(Ok(status)) if status.success() => {}
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            info!(
                "Managed desktop-capabilities probe timed out in {}ms",
                started.elapsed().as_millis()
            );
            return Ok(None);
        }
        _ => {
            info!(
                "Managed desktop-capabilities probe exited unsuccessfully in {}ms",
                started.elapsed().as_millis()
            );
            return Ok(None);
        }
    }

    let mut output = Vec::new();
    if stdout.read_to_end(&mut output).await.is_err() {
        return Ok(None);
    }

    let capability = serde_json::from_slice::<DesktopCapability>(&output).ok();
    info!(
        "Managed desktop-capabilities probe finished ok={} in {}ms",
        capability.is_some(),
        started.elapsed().as_millis()
    );
    Ok(capability)
}

fn desktop_capability_stale_reason(capability: &DesktopCapability) -> Option<String> {
    if capability.desktop_protocol_version != Some(DESKTOP_PROTOCOL_VERSION) {
        return Some("desktop_protocol_incompatible".to_string());
    }
    if capability.supports_api_only != Some(true) {
        return Some("desktop_api_only_unsupported".to_string());
    }
    if capability.supports_provision_desktop_auth != Some(true) {
        return capability
            .desktop_auth_stale_reason
            .clone()
            .or_else(|| Some("desktop_auth_unsupported".to_string()));
    }
    if capability.desktop_manageability_version.unwrap_or(0) < DESKTOP_MANAGEABILITY_VERSION {
        return Some("desktop_manageability_unsupported".to_string());
    }
    if capability.supports_desktop_backend_ownership != Some(true) {
        return Some("desktop_backend_ownership_unsupported".to_string());
    }
    // Half-installed is Stale, not Ready: starting the backend just crashes it.
    // A CLI too old to answer is already rejected above on manageability.
    if capability.studio_install_ok != Some(true) {
        return Some(
            capability
                .studio_install_reason
                .clone()
                .unwrap_or_else(|| "studio_install_incomplete".to_string()),
        );
    }
    // Only an explicit false. None means the CLI predates the field or nothing
    // is installed yet, and neither is a broken runtime: treating them as stale
    // would put every older install into repair on the first launch after an
    // upgrade.
    if capability.llama_runtime_ok == Some(false) {
        return Some(
            capability
                .llama_runtime_reason
                .clone()
                .filter(|reason| !reason.is_empty())
                .unwrap_or_else(|| "llama_runtime_incomplete".to_string()),
        );
    }
    managed_backend_version_stale_reason(capability.version.as_deref())
}

fn desktop_capability_ready(capability: &DesktopCapability) -> bool {
    desktop_capability_stale_reason(capability).is_none()
}

/// The reason an unbuildable context is reported under, if that is what went
/// wrong. Checked after a probe that did run and failed anyway.
fn working_directory_reason() -> Option<String> {
    // The whole context: an unresolvable override fails the same spawn, and is a
    // different thing to fix.
    let error = crate::process::managed_cli_context_error()?;
    info!("Managed preflight: managed context unavailable: {error}");
    Some(context_reason(&error))
}

pub(super) async fn probe_managed_bin(bin: PathBuf) -> ManagedProbe {
    let started = Instant::now();

    // An unmounted roaming profile fails every probe below, which is not a broken
    // install: "cli_unusable" would start a repair needing the same directory.
    if let Some(error) = crate::process::managed_cli_context_error() {
        info!(
            "Managed preflight: no usable managed context for {:?}: {}",
            bin, error
        );
        return ManagedProbe::Stale {
            bin,
            reason: context_reason(&error),
        };
    }

    // Always verify the managed CLI actually launches before trusting the cache.
    // A matching capability fingerprint does not prove the binary can still run:
    // its venv interpreter or a runtime dependency can be broken while the
    // path/size/mtime/markers are unchanged, so the -h probe runs first and a
    // non-launchable install is reported Stale for repair. The capability cache
    // below still skips the heavier desktop-capabilities probe on a hit.
    match run_cli_probe(&bin, &["-h"]).await {
        // The CLI was never asked, so do not report a broken install.
        Err(_) => {
            info!(
                "Managed preflight: no usable managed context for {:?} in {}ms",
                bin,
                started.elapsed().as_millis()
            );
            return ManagedProbe::Stale {
                bin,
                reason: working_directory_reason()
                    .unwrap_or_else(|| WORKING_DIRECTORY_UNAVAILABLE.to_string()),
            };
        }
        Ok(false) => {
            info!(
                "Managed preflight: cli unusable for {:?} in {}ms",
                bin,
                started.elapsed().as_millis()
            );
            // The profile can drop between the check above and the probe, so ask again.
            return ManagedProbe::Stale {
                bin,
                reason: working_directory_reason().unwrap_or_else(|| "cli_unusable".to_string()),
            };
        }
        Ok(true) => {}
    }

    if let Some(fingerprint) = managed_bin_fingerprint(&bin) {
        if read_cached_capability(&fingerprint).is_some() {
            info!(
                "Managed preflight: using cached desktop capability for {:?} in {}ms",
                bin,
                started.elapsed().as_millis()
            );
            return ManagedProbe::Ready { bin };
        }
    }

    let capability = match probe_cli_capability(&bin).await {
        Ok(capability) => capability,
        Err(_) => {
            info!(
                "Managed preflight: no usable managed context for {:?} in {}ms",
                bin,
                started.elapsed().as_millis()
            );
            return ManagedProbe::Stale {
                bin,
                reason: working_directory_reason()
                    .unwrap_or_else(|| WORKING_DIRECTORY_UNAVAILABLE.to_string()),
            };
        }
    };
    if let Some(capability) = capability {
        if let Some(fingerprint) = managed_bin_fingerprint(&bin) {
            write_cached_capability(&fingerprint, &capability);
        }
        if desktop_capability_ready(&capability) {
            info!(
                "Managed preflight: cli ready for {:?} in {}ms",
                bin,
                started.elapsed().as_millis()
            );
            return ManagedProbe::Ready { bin };
        }
        info!(
            "Managed preflight: cli stale for {:?} in {}ms",
            bin,
            started.elapsed().as_millis()
        );
        return ManagedProbe::Stale {
            bin,
            reason: desktop_capability_stale_reason(&capability)
                .unwrap_or_else(|| "desktop_capability_incompatible".to_string()),
        };
    }

    info!(
        "Managed preflight: desktop capability probe failed for {:?} in {}ms",
        bin,
        started.elapsed().as_millis()
    );
    ManagedProbe::Stale {
        bin,
        reason: working_directory_reason()
            .unwrap_or_else(|| "desktop_capability_probe_failed".to_string()),
    }
}

pub(super) async fn probe_managed_install() -> ManagedProbe {
    let started = Instant::now();
    let result = match crate::process::find_unsloth_binary() {
        Some(bin) => probe_managed_bin(bin).await,
        // The managed install lives under the profile, so an unreachable one looks
        // like no install. Say which, or a late network profile sends them to reinstall.
        None => match crate::process::home_dir_available() {
            Ok(()) => ManagedProbe::Missing,
            Err(error) => {
                info!("Managed preflight: {}", error);
                ManagedProbe::Unavailable {
                    reason: WORKING_DIRECTORY_UNAVAILABLE.to_string(),
                }
            }
        },
    };
    info!(
        "Managed preflight: install probe result {:?} in {}ms",
        result,
        started.elapsed().as_millis()
    );
    result
}

pub async fn managed_install_ready() -> bool {
    matches!(probe_managed_install().await, ManagedProbe::Ready { .. })
}

#[cfg(test)]
mod tests {
    use super::super::version::MIN_DESKTOP_BACKEND_VERSION;
    use super::*;

    fn healthy_capability() -> DesktopCapability {
        DesktopCapability {
            desktop_protocol_version: Some(DESKTOP_PROTOCOL_VERSION),
            desktop_manageability_version: Some(DESKTOP_MANAGEABILITY_VERSION),
            supports_api_only: Some(true),
            supports_provision_desktop_auth: Some(true),
            supports_desktop_backend_ownership: Some(true),
            desktop_auth_stale_reason: None,
            studio_install_ok: Some(true),
            studio_install_reason: None,
            llama_runtime_ok: Some(true),
            llama_runtime_reason: None,
            version: Some(MIN_DESKTOP_BACKEND_VERSION.to_string()),
        }
    }

    #[test]
    fn complete_install_is_ready() {
        assert_eq!(desktop_capability_stale_reason(&healthy_capability()), None);
        assert!(desktop_capability_ready(&healthy_capability()));
    }

    #[test]
    fn a_venv_behind_the_desktop_backend_version_is_stale() {
        // The CLI installer shares this venv, so its package version is the only
        // thing that pulls an old-but-launchable install forward via repair.
        let mut capability = healthy_capability();
        capability.version = Some("2026.5.2".to_string());
        assert_eq!(
            desktop_capability_stale_reason(&capability).as_deref(),
            Some("desktop_backend_version_too_old")
        );
        assert!(!desktop_capability_ready(&capability));
    }

    #[test]
    fn incomplete_install_is_stale_with_the_cli_reason() {
        // The venv has the CLI but not structlog, so preflight must repair
        // rather than spawn a backend that cannot import.
        let mut capability = healthy_capability();
        capability.studio_install_ok = Some(false);
        capability.studio_install_reason = Some("studio_install_incomplete".to_string());
        assert_eq!(
            desktop_capability_stale_reason(&capability).as_deref(),
            Some("studio_install_incomplete")
        );
        assert!(!desktop_capability_ready(&capability));
    }

    #[test]
    fn deps_removed_after_install_is_stale() {
        let mut capability = healthy_capability();
        capability.studio_install_ok = Some(false);
        capability.studio_install_reason = Some("studio_deps_missing".to_string());
        assert_eq!(
            desktop_capability_stale_reason(&capability).as_deref(),
            Some("studio_deps_missing")
        );
    }

    #[test]
    fn missing_install_field_falls_back_to_a_generic_reason() {
        let mut capability = healthy_capability();
        capability.studio_install_ok = None;
        capability.studio_install_reason = None;
        assert_eq!(
            desktop_capability_stale_reason(&capability).as_deref(),
            Some("studio_install_incomplete")
        );
    }

    #[test]
    fn a_quarantined_llama_runtime_is_stale() {
        // The venv is fine and the marker still says installed; Smart App Control
        // took a DLL out of the tree underneath it. Repairing here is what stops
        // this surfacing later as a model load failure.
        let mut capability = healthy_capability();
        capability.llama_runtime_ok = Some(false);
        capability.llama_runtime_reason = Some("llama_runtime_payload_incomplete".to_string());
        assert_eq!(
            desktop_capability_stale_reason(&capability).as_deref(),
            Some("llama_runtime_payload_incomplete")
        );
        assert!(!desktop_capability_ready(&capability));
    }

    #[test]
    fn a_broken_runtime_without_a_reason_falls_back_to_a_generic_one() {
        let mut capability = healthy_capability();
        capability.llama_runtime_ok = Some(false);
        capability.llama_runtime_reason = Some(String::new());
        assert_eq!(
            desktop_capability_stale_reason(&capability).as_deref(),
            Some("llama_runtime_incomplete")
        );
    }

    #[test]
    fn an_unknown_llama_runtime_is_not_stale() {
        // None is both "no runtime installed yet" and "this CLI predates the
        // field". Neither is a broken runtime, and calling either one stale would
        // send every existing install through repair on its next launch.
        let mut capability = healthy_capability();
        capability.llama_runtime_ok = None;
        capability.llama_runtime_reason = None;
        assert_eq!(desktop_capability_stale_reason(&capability), None);
        assert!(desktop_capability_ready(&capability));
    }

    /// The payload a CLI without this PR prints, as JSON rather than as a Rust
    /// literal: a struct literal cannot show that a missing key deserializes.
    fn pre_pr_capability_json() -> String {
        format!(
            r#"{{
              "desktop_protocol_version": {protocol},
              "desktop_manageability_version": {manageability},
              "supports_api_only": true,
              "supports_provision_desktop_auth": true,
              "supports_desktop_backend_ownership": true,
              "desktop_auth_stale_reason": null,
              "studio_install_ok": true,
              "studio_install_reason": null,
              "version": "{version}"
            }}"#,
            protocol = DESKTOP_PROTOCOL_VERSION,
            manageability = DESKTOP_MANAGEABILITY_VERSION,
            version = MIN_DESKTOP_BACKEND_VERSION,
        )
    }

    #[test]
    fn a_capability_payload_without_the_new_keys_still_parses_and_is_ready() {
        // The single most damaging way this change could go wrong: every install
        // whose CLI predates it prints a payload with neither key. If that failed
        // to deserialize, or deserialized to something stale, the desktop would
        // send every existing user into repair on their next launch.
        let capability: DesktopCapability =
            serde_json::from_str(&pre_pr_capability_json()).expect("pre-PR payload must parse");
        assert_eq!(capability.llama_runtime_ok, None);
        assert_eq!(capability.llama_runtime_reason, None);
        assert_eq!(desktop_capability_stale_reason(&capability), None);
        assert!(desktop_capability_ready(&capability));
    }

    #[test]
    fn a_payload_from_a_newer_cli_ignores_keys_this_desktop_does_not_know() {
        // The other direction, and the one an upgrade sequence hits: the CLI and
        // the desktop shell update separately, so a newer CLI can answer an older
        // desktop. An unknown key must be ignored, not fatal.
        let json = pre_pr_capability_json().replace(
            "\"studio_install_ok\": true,",
            "\"studio_install_ok\": true, \"a_field_from_the_future\": {\"nested\": [1]},",
        );
        let capability: DesktopCapability =
            serde_json::from_str(&json).expect("an unknown key must not be fatal");
        assert!(desktop_capability_ready(&capability));
    }

    #[test]
    fn a_null_llama_runtime_ok_is_not_a_broken_runtime() {
        // What the CLI prints when the probe itself failed, which must not be
        // read as a verdict. Explicit nulls, not absent keys, since the CLI seeds
        // both keys before trying.
        let json = pre_pr_capability_json().replace(
            "\"studio_install_ok\": true,",
            "\"llama_runtime_ok\": null, \"llama_runtime_reason\": \"\", \"studio_install_ok\": true,",
        );
        let capability: DesktopCapability = serde_json::from_str(&json).unwrap();
        assert_eq!(desktop_capability_stale_reason(&capability), None);
    }

    #[test]
    fn a_broken_runtime_survives_the_json_round_trip() {
        // The reason has to reach the window intact: the frontend switches its
        // message on this exact string.
        let json = pre_pr_capability_json().replace(
            "\"studio_install_ok\": true,",
            "\"llama_runtime_ok\": false, \"llama_runtime_reason\": \"llama_runtime_binaries_missing\", \"studio_install_ok\": true,",
        );
        let capability: DesktopCapability = serde_json::from_str(&json).unwrap();
        assert_eq!(
            desktop_capability_stale_reason(&capability).as_deref(),
            Some("llama_runtime_binaries_missing")
        );
    }

    #[test]
    fn a_cache_file_written_before_this_field_is_ignored_rather_than_fatal() {
        // On disk in every existing install. It must miss and be rewritten, not
        // panic and not be served: schema 3 predates the runtime fingerprint, so
        // its Ready verdict was reached without ever looking at the runtime.
        let old = format!(
            r#"{{
              "schema": 3,
              "bin_path": "/managed/unsloth",
              "bin_size": 1,
              "bin_mtime_ms": 1,
              "studio_root_id": null,
              "marker_path": null,
              "marker_size": null,
              "marker_mtime_ms": null,
              "desktop_protocol_version": {protocol},
              "desktop_manageability_version": {manageability},
              "capability": {capability}
            }}"#,
            protocol = DESKTOP_PROTOCOL_VERSION,
            manageability = DESKTOP_MANAGEABILITY_VERSION,
            capability = pre_pr_capability_json(),
        );
        let cache: ManagedCapabilityCache =
            serde_json::from_str(&old).expect("an old cache file must still parse");
        assert_eq!(cache.llama_runtime, None);
        assert_ne!(cache.schema, MANAGED_CAPABILITY_CACHE_SCHEMA);

        let fingerprint = ManagedBinFingerprint {
            bin_path: "/managed/unsloth".to_string(),
            bin_size: 1,
            bin_mtime_ms: 1,
            studio_root_id: None,
            marker_path: None,
            marker_size: None,
            marker_mtime_ms: None,
            llama_runtime: Some("12:345".to_string()),
        };
        assert!(
            !cache_matches(&cache, &fingerprint),
            "a cache from before the runtime was fingerprinted must not be served"
        );
    }

    #[test]
    fn a_quarantined_runtime_file_changes_the_fingerprint() {
        // The half that makes the stale check reachable at all. Without the
        // runtime in the fingerprint the cache hits, preflight answers Ready from
        // it, and the CLI is never asked whether the runtime is intact.
        let root = std::env::temp_dir().join(format!(
            "unsloth-llama-fingerprint-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        let mut bin = root.join("build").join("bin");
        if cfg!(windows) {
            bin = bin.join("Release");
        }
        fs::create_dir_all(&bin).unwrap();
        fs::write(bin.join("llama.dll"), vec![0u8; 1024]).unwrap();
        fs::write(bin.join("ggml-base.dll"), vec![0u8; 2048]).unwrap();

        let intact =
            llama_runtime_fingerprint_at(&root).expect("an installed runtime must fingerprint");
        assert_eq!(
            intact,
            llama_runtime_fingerprint_at(&root).unwrap(),
            "the same tree must fingerprint the same twice, or every launch misses its own cache"
        );

        fs::remove_file(bin.join("ggml-base.dll")).unwrap();
        let quarantined = llama_runtime_fingerprint_at(&root).unwrap();
        assert_ne!(intact, quarantined);

        // A directory is not a file: a stray subfolder must not read as a binary.
        fs::create_dir(bin.join("some-subdir")).unwrap();
        assert_eq!(quarantined, llama_runtime_fingerprint_at(&root).unwrap());

        // A file replaced by one of a different size is caught by the byte total
        // even though the count is unchanged.
        fs::write(bin.join("llama.dll"), vec![0u8; 4096]).unwrap();
        assert_ne!(quarantined, llama_runtime_fingerprint_at(&root).unwrap());

        // And the whole tree going is distinct from an empty one, because only
        // one of the two means nothing was ever installed.
        fs::remove_dir_all(&root).unwrap();
        assert_eq!(llama_runtime_fingerprint_at(&root), None);
        fs::create_dir_all(&bin).unwrap();
        assert_eq!(
            llama_runtime_fingerprint_at(&root).as_deref(),
            Some("bin:0:0")
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_marker_left_over_a_missing_build_dir_changes_the_fingerprint() {
        // The other transition into a broken runtime, and the one a
        // read_dir(bin).ok()? alone cannot see. Nothing installed and a marker
        // sitting on a tree with no build/bin both fail that read, so both used to
        // fingerprint as None -- while the CLI's verdict moves from "no marker, no
        // opinion" (llama_runtime_ok null, which desktop_capability_stale_reason
        // deliberately does not call stale) to llama_runtime_dir_missing. The
        // cached Ready outlived the change and preflight never re-asked.
        let root = std::env::temp_dir().join(format!(
            "unsloth-llama-nobin-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);

        assert_eq!(
            llama_runtime_fingerprint_at(&root),
            None,
            "no tree at all is the NotInstalled case, and stays None"
        );

        fs::create_dir_all(&root).unwrap();
        let empty_tree = llama_runtime_fingerprint_at(&root);
        assert!(
            empty_tree.is_some(),
            "a runtime root with no build/bin is a broken install, not an absent one"
        );

        fs::write(root.join("unsloth_llama_prebuilt.json"), b"{}").unwrap();
        assert_ne!(
            empty_tree,
            llama_runtime_fingerprint_at(&root),
            "a marker arriving over a missing build/bin must invalidate the cached Ready"
        );

        // And it is still distinct from a tree whose build/bin exists and is
        // empty, so the two are never confused for each other.
        let mut bin = root.join("build").join("bin");
        if cfg!(windows) {
            bin = bin.join("Release");
        }
        fs::create_dir_all(&bin).unwrap();
        assert_ne!(
            llama_runtime_fingerprint_at(&root),
            empty_tree,
            "an empty binary directory is not the same state as a missing one"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn the_runtime_override_is_resolved_the_way_python_resolves_it() {
        // default_managed_llama_dir() strips the value and calls expanduser on it.
        // Reading it raw would fingerprint a folder literally named "~", find
        // nothing, and silently drop the runtime out of the fingerprint for every
        // user who wrote the override that way.
        let home = PathBuf::from(if cfg!(windows) {
            "C:\\Users\\me"
        } else {
            "/home/me"
        });
        let cwd = PathBuf::from(if cfg!(windows) { "C:\\work" } else { "/work" });
        let absolute = if cfg!(windows) {
            "C:\\opt\\llama.cpp"
        } else {
            "/opt/llama.cpp"
        };
        let cases: Vec<(Option<&str>, Option<PathBuf>)> = vec![
            (None, None),
            (Some(""), None),
            (Some("   "), None),
            (Some("~"), Some(home.clone())),
            (Some("~/llama.cpp"), Some(home.join("llama.cpp"))),
            (Some("  ~/llama.cpp  "), Some(home.join("llama.cpp"))),
            (Some(absolute), Some(PathBuf::from(absolute))),
            // Not expanded: nothing here can resolve another user's home. It is
            // then relative, so it is anchored like any other relative value
            // rather than naming a folder called "~someone" beside the desktop.
            (
                Some("~someone/llama.cpp"),
                Some(cwd.join("~someone/llama.cpp")),
            ),
            // Anchored to this process's directory, which is the one
            // relative_override_pins joins the child's copy against.
            (Some("llama.cpp"), Some(cwd.join("llama.cpp"))),
            (Some("./llama.cpp"), Some(cwd.join("./llama.cpp"))),
            (Some("../llama.cpp"), Some(cwd.join("../llama.cpp"))),
        ];
        for (value, expected) in cases {
            assert_eq!(
                llama_runtime_override_from(value, Some(&home), Some(&cwd)),
                expected,
                "value {value:?}"
            );
        }
    }

    #[test]
    fn an_override_that_cannot_be_anchored_is_dropped_rather_than_guessed() {
        // No home to expand against and no directory to anchor to. Guessing here
        // would fingerprint a tree the child was never pointed at, which is worse
        // than no coverage: the cached verdict would then track the wrong folder.
        assert_eq!(llama_runtime_override_from(Some("~/x"), None, None), None);
        assert_eq!(llama_runtime_override_from(Some("x"), None, None), None);
        // An absolute value needs neither, so it still resolves.
        let absolute = if cfg!(windows) {
            "C:\\opt\\llama.cpp"
        } else {
            "/opt/llama.cpp"
        };
        assert_eq!(
            llama_runtime_override_from(Some(absolute), None, None),
            Some(PathBuf::from(absolute))
        );
    }

    #[cfg(windows)]
    #[test]
    fn the_windows_tilde_resolves_through_the_same_profile_the_child_uses() {
        // ntpath.expanduser answers USERPROFILE and relative_override_pins
        // resolves the child's tilde through it, while dirs::home_dir() reads the
        // known folder, which a portable or overridden profile moves. Taking the
        // known folder here would fingerprint a tree the child never looks at, so
        // a quarantine in the tree it does look at would never invalidate the
        // cache. process.rs carries the same note at its own read.
        let _guard = crate::native_path_policy::PROCESS_ENV_LOCK.lock();
        let previous = std::env::var_os("USERPROFILE");
        std::env::set_var("USERPROFILE", "C:\\Portable\\Profile");
        let resolved = tilde_home();
        match previous {
            Some(value) => std::env::set_var("USERPROFILE", value),
            None => std::env::remove_var("USERPROFILE"),
        }
        assert_eq!(resolved, Some(PathBuf::from("C:\\Portable\\Profile")));
    }

    #[test]
    fn older_cli_is_rejected_on_manageability_before_the_install_check() {
        // A CLI predating this feature cannot answer studio_install_ok, so the
        // more specific manageability reason must win in the diagnostics.
        let mut capability = healthy_capability();
        capability.desktop_manageability_version = Some(1);
        capability.studio_install_ok = None;
        assert_eq!(
            desktop_capability_stale_reason(&capability).as_deref(),
            Some("desktop_manageability_unsupported")
        );
    }

    #[test]
    fn a_stale_capability_is_never_served_from_cache() {
        // write_cached_capability runs before the ready check, so an incomplete
        // install does get cached; reusing it would outlive the repair.
        let mut capability = healthy_capability();
        capability.studio_install_ok = Some(false);
        let cache = ManagedCapabilityCache {
            schema: MANAGED_CAPABILITY_CACHE_SCHEMA,
            bin_path: "/managed/unsloth".to_string(),
            bin_size: 1,
            bin_mtime_ms: 1,
            studio_root_id: None,
            marker_path: None,
            marker_size: None,
            marker_mtime_ms: None,
            llama_runtime: None,
            desktop_protocol_version: DESKTOP_PROTOCOL_VERSION,
            desktop_manageability_version: DESKTOP_MANAGEABILITY_VERSION,
            capability,
        };
        let fingerprint = ManagedBinFingerprint {
            bin_path: "/managed/unsloth".to_string(),
            bin_size: 1,
            bin_mtime_ms: 1,
            studio_root_id: None,
            marker_path: None,
            marker_size: None,
            marker_mtime_ms: None,
            llama_runtime: None,
        };
        assert!(!cache_matches(&cache, &fingerprint));
    }

    // Quarantine deletes the generated unsloth.exe, so the supported stubless layout
    // hands back a launcher path that is not on disk. Without a stand-in the
    // fingerprint is None, the capability cache can be neither read nor written, and
    // every preflight pays both probe subprocesses again.
    #[cfg(windows)]
    #[test]
    fn a_quarantined_launcher_is_fingerprinted_through_its_interpreter() {
        let venv = std::env::temp_dir().join(format!(
            "unsloth-fingerprint-quarantined-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let scripts = venv.join("Scripts");
        fs::create_dir_all(&scripts).unwrap();
        let bin = scripts.join("unsloth.exe");
        let interpreter = scripts.join("python.exe");
        fs::write(&interpreter, "python").unwrap();
        assert!(!bin.exists(), "this case is about the launcher being gone");

        let fingerprint = managed_bin_fingerprint(&bin)
            .expect("a stubless venv must still fingerprint, through python.exe");
        // The identity stays the launcher path, so the two layouts of one install
        // cannot share a cache entry.
        assert!(fingerprint.bin_path.ends_with("unsloth.exe"));
        // And it tracks the interpreter, so a venv replaced by an update invalidates.
        fs::write(&interpreter, "python-after-an-update").unwrap();
        let after = managed_bin_fingerprint(&bin).unwrap();
        assert_ne!(fingerprint.bin_size, after.bin_size);

        // With no interpreter either there is nothing to stand in, and None is right.
        fs::remove_file(&interpreter).unwrap();
        assert!(managed_bin_fingerprint(&bin).is_none());

        let _ = fs::remove_dir_all(&venv);
    }

    #[test]
    fn dropping_the_manifest_changes_the_fingerprint() {
        // Otherwise a cache entry written while healthy outlives the manifest,
        // and the probe returns Ready on the very venv this is meant to catch.
        let venv = std::env::temp_dir().join(format!(
            "unsloth-fingerprint-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let scripts = venv.join("bin");
        fs::create_dir_all(&scripts).unwrap();
        let bin = scripts.join("unsloth");
        fs::write(&bin, "#!/bin/sh\nexit 0\n").unwrap();
        let manifest = venv.join("unsloth_install_manifest.json");
        fs::write(&manifest, "{}").unwrap();

        let with_manifest = managed_bin_fingerprint(&bin).unwrap();
        fs::remove_file(&manifest).unwrap();
        let without_manifest = managed_bin_fingerprint(&bin).unwrap();

        assert_ne!(with_manifest, without_manifest);
        let _ = fs::remove_dir_all(&venv);
    }

    fn cache_for(fingerprint: &ManagedBinFingerprint) -> ManagedCapabilityCache {
        ManagedCapabilityCache {
            schema: MANAGED_CAPABILITY_CACHE_SCHEMA,
            bin_path: fingerprint.bin_path.clone(),
            bin_size: fingerprint.bin_size,
            bin_mtime_ms: fingerprint.bin_mtime_ms,
            studio_root_id: fingerprint.studio_root_id.clone(),
            marker_path: fingerprint.marker_path.clone(),
            marker_size: fingerprint.marker_size,
            marker_mtime_ms: fingerprint.marker_mtime_ms,
            llama_runtime: fingerprint.llama_runtime.clone(),
            desktop_protocol_version: DESKTOP_PROTOCOL_VERSION,
            desktop_manageability_version: DESKTOP_MANAGEABILITY_VERSION,
            capability: healthy_capability(),
        }
    }

    #[test]
    fn losing_a_studio_package_changes_the_fingerprint() {
        // pip uninstall rewrites no fingerprinted file: the manifest, pyvenv.cfg
        // and the launcher survive and `unsloth -h` still exits 0. Without the
        // installed distributions in the fingerprint the healthy answer sticks.
        let venv = std::env::temp_dir().join(format!(
            "unsloth-fingerprint-deps-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&venv);
        let scripts = venv.join("bin");
        fs::create_dir_all(&scripts).unwrap();
        let bin = scripts.join("unsloth");
        fs::write(&bin, "#!/bin/sh\nexit 0\n").unwrap();
        fs::write(venv.join("pyvenv.cfg"), "home = /usr/bin\n").unwrap();
        fs::write(venv.join("unsloth_install_manifest.json"), "{}").unwrap();

        // site_packages_dirs() only walks lib/<pyver>/site-packages on unix; on
        // Windows it looks at Lib/site-packages. Building the posix layout
        // everywhere left the dist-info invisible to the fingerprint on Windows,
        // so removing it changed nothing and the assert_ne below could not hold.
        let site_packages = if cfg!(windows) {
            venv.join("Lib").join("site-packages")
        } else {
            venv.join("lib").join("python3.11").join("site-packages")
        };
        fs::create_dir_all(site_packages.join("unsloth_cli").join("commands")).unwrap();
        fs::write(
            site_packages
                .join("unsloth_cli")
                .join("commands")
                .join("studio.py"),
            "# cli\n",
        )
        .unwrap();
        let dist_info = site_packages.join("fastmcp-3.0.2.dist-info");
        fs::create_dir_all(&dist_info).unwrap();
        fs::write(dist_info.join("METADATA"), "Name: fastmcp\n").unwrap();

        let with_dep = managed_bin_fingerprint(&bin).unwrap();
        let healthy_cache = cache_for(&with_dep);
        // read_dir order is unspecified, so an unsorted walk would miss its own
        // cache every launch and the entry would never be worth writing.
        assert_eq!(with_dep, managed_bin_fingerprint(&bin).unwrap());
        assert!(cache_matches(&healthy_cache, &with_dep));

        fs::remove_dir_all(&dist_info).unwrap();
        let without_dep = managed_bin_fingerprint(&bin).unwrap();

        assert_ne!(with_dep, without_dep);
        assert!(
            !cache_matches(&healthy_cache, &without_dep),
            "a removed studio package must not keep serving the cached Ready answer"
        );
        let _ = fs::remove_dir_all(&venv);
    }

    #[test]
    fn a_context_reason_names_the_setting_it_could_not_preserve() {
        use crate::process::ManagedContextError;
        // Every pin failure names the setting first, and the window needs that
        // name: "one of Unsloth's folder settings" is not something to act on.
        let reason = context_reason(&ManagedContextError::PathSetting(
            "HF_HOME names a path this machine cannot resolve".to_string(),
        ));
        assert_eq!(reason, "path_setting_unresolvable:HF_HOME");
        assert!(is_context_reason(&reason));
        // The name is carried, never the value or the sentence around it.
        assert!(!reason.contains("cannot resolve"));
        // A failure that does not start with a setting name still classifies.
        let bare = context_reason(&ManagedContextError::PathSetting(
            "the environment block is too long".to_string(),
        ));
        assert_eq!(bare, PATH_SETTING_UNRESOLVABLE);
        assert!(is_context_reason(&bare));
        assert!(is_context_reason(WORKING_DIRECTORY_UNAVAILABLE));
        assert!(!is_context_reason("cli_unusable"));
    }

    /// A scratch directory of this test's own, emptied before it is handed back.
    ///
    /// Every case below walks real files while cargo runs tests on parallel
    /// threads, so two tests sharing one path would delete each other's tree
    /// mid-walk. Process id plus thread id is what the tests above already key
    /// on, and it is what keeps a leaked directory from ever being reused.
    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "unsloth-managed-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The one directory llama_runtime_fingerprint_at walks, for a given root.
    /// Kept in one place so the tests and the walk cannot drift apart on the
    /// Windows-only Release level.
    fn runtime_bin_dir(root: &Path) -> PathBuf {
        let bin = root.join("build").join("bin");
        if cfg!(windows) {
            bin.join("Release")
        } else {
            bin
        }
    }

    /// A tree shaped like a prebuilt llama.cpp install: a server binary and one
    /// shared library, which is the smallest thing quarantine can take a file
    /// out of.
    fn install_fake_runtime(root: &Path) -> PathBuf {
        let bin = runtime_bin_dir(root);
        fs::create_dir_all(&bin).unwrap();
        fs::write(bin.join("llama-server"), vec![0u8; 4096]).unwrap();
        fs::write(bin.join("libggml-base.so"), vec![0u8; 2048]).unwrap();
        bin
    }

    /// A fingerprint whose venv half is fixed and whose runtime half is read off
    /// disk right now, which is what each launch does: managed_bin_fingerprint
    /// recomputes the whole thing every time and the cache is keyed on the
    /// result. Building it here rather than through managed_bin_fingerprint
    /// keeps UNSLOTH_TEST_LLAMA_RUNTIME_ROOT out of these tests entirely; that
    /// variable is process-wide, and setting it would change what the venv
    /// fingerprint tests above compute while they run beside these.
    fn fingerprint_for_runtime(runtime_root: &Path) -> ManagedBinFingerprint {
        ManagedBinFingerprint {
            bin_path: "/managed/unsloth".to_string(),
            bin_size: 42,
            bin_mtime_ms: 1_700_000_000_000,
            studio_root_id: None,
            marker_path: None,
            marker_size: None,
            marker_mtime_ms: None,
            llama_runtime: llama_runtime_fingerprint_at(runtime_root),
        }
    }

    /// Points capability_cache_path() at a directory of this test's own for as
    /// long as the guard lives.
    ///
    /// The hook is a process-wide environment variable and cargo runs tests on
    /// parallel threads, so the write happens under the crate's
    /// PROCESS_ENV_LOCK and the guard holds that lock for the whole test. That
    /// is the same lock preflight.rs takes around its capability-cache tests,
    /// which set this very variable, and the one main.rs takes around
    /// XDG_DATA_HOME, so no other test can observe a half-installed value or
    /// overwrite this one midway. The per-test directory on top means even a
    /// cache file left behind by a panicking test cannot be read by another.
    struct CapabilityCacheHome {
        home: PathBuf,
        previous: Option<std::ffi::OsString>,
        /// Declared last so it is released after the restore below has run.
        _env: std::sync::MutexGuard<'static, ()>,
    }

    impl CapabilityCacheHome {
        fn new(test_name: &str) -> Self {
            let _env = crate::native_path_policy::PROCESS_ENV_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let home = std::env::temp_dir().join(format!(
                "unsloth-managed-cache-{test_name}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = fs::remove_dir_all(&home);
            fs::create_dir_all(&home).unwrap();
            let previous = std::env::var_os("UNSLOTH_TEST_DESKTOP_CAPABILITY_CACHE_HOME");
            std::env::set_var("UNSLOTH_TEST_DESKTOP_CAPABILITY_CACHE_HOME", &home);
            Self {
                home,
                previous,
                _env,
            }
        }

        /// The file write_cached_capability writes, resolved through the hook
        /// itself rather than rebuilt here, so a change to the layout cannot
        /// leave these tests asserting against a path nothing writes.
        fn cache_file(&self) -> PathBuf {
            let path = capability_cache_path().expect("the test hook must resolve a cache path");
            assert!(
                path.starts_with(&self.home),
                "the hook must resolve inside this test's home, not the real one: {path:?}"
            );
            path
        }
    }

    impl Drop for CapabilityCacheHome {
        fn drop(&mut self) {
            match &self.previous {
                Some(previous) => {
                    std::env::set_var("UNSLOTH_TEST_DESKTOP_CAPABILITY_CACHE_HOME", previous)
                }
                None => std::env::remove_var("UNSLOTH_TEST_DESKTOP_CAPABILITY_CACHE_HOME"),
            }
            let _ = fs::remove_dir_all(&self.home);
        }
    }

    #[test]
    fn an_unchanged_runtime_serves_the_cached_capability_back_from_disk() {
        // The paying case for the cache: a healthy install must hit on its
        // second launch, or the desktop pays both probe subprocesses and their
        // 10s ceilings every time the window opens.
        let home = CapabilityCacheHome::new("hit");
        let root = scratch_dir("runtime-hit");
        install_fake_runtime(&root);

        let fingerprint = fingerprint_for_runtime(&root);
        assert!(
            read_cached_capability(&fingerprint).is_none(),
            "nothing is cached yet, so the first launch must miss"
        );
        write_cached_capability(&fingerprint, &healthy_capability());
        assert!(
            home.cache_file().exists(),
            "the cache must reach disk, not just the struct"
        );

        // The next launch recomputes the fingerprint against the same tree.
        let relaunch = fingerprint_for_runtime(&root);
        let cached = read_cached_capability(&relaunch)
            .expect("an install nothing touched must hit its own cache");
        assert_eq!(cached.llama_runtime_ok, Some(true));
        assert!(desktop_capability_ready(&cached));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_file_quarantined_out_of_the_runtime_misses_the_cache() {
        // The whole point of the change. Smart App Control takes one file out of
        // an otherwise present tree; the venv, the markers and the launcher are
        // all untouched. If this hit, preflight would answer Ready from a cache
        // written while the tree was intact and never ask the CLI, and the user
        // would meet the damage as a model load failure instead.
        let home = CapabilityCacheHome::new("quarantine");
        let root = scratch_dir("runtime-quarantine");
        let bin = install_fake_runtime(&root);

        let healthy = fingerprint_for_runtime(&root);
        write_cached_capability(&healthy, &healthy_capability());
        assert!(read_cached_capability(&healthy).is_some());

        fs::remove_file(bin.join("libggml-base.so")).unwrap();
        let quarantined = fingerprint_for_runtime(&root);
        assert_ne!(healthy.llama_runtime, quarantined.llama_runtime);
        assert!(
            read_cached_capability(&quarantined).is_none(),
            "a quarantined runtime file must not keep serving a cached Ready"
        );
        // And the stale entry is still on disk, so the miss came from the
        // fingerprint rather than from a file that went missing.
        assert!(home.cache_file().exists());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_file_added_to_the_runtime_misses_the_cache() {
        // The other direction of the same walk, and the one a half-finished
        // repair or a partial download leaves behind: a tree that gained a file
        // is not the tree the CLI was asked about.
        let _home = CapabilityCacheHome::new("added");
        let root = scratch_dir("runtime-added");
        let bin = install_fake_runtime(&root);

        let before = fingerprint_for_runtime(&root);
        write_cached_capability(&before, &healthy_capability());

        fs::write(bin.join("llama-quantize"), vec![0u8; 512]).unwrap();
        let after = fingerprint_for_runtime(&root);
        assert!(
            read_cached_capability(&after).is_none(),
            "a runtime that gained a file must be asked about again"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_runtime_directory_that_is_gone_entirely_misses_the_cache() {
        // An uninstall or a wiped ~/.unsloth between launches. The fingerprint
        // drops to None, which must not compare equal to the string written
        // while the tree was there.
        let _home = CapabilityCacheHome::new("removed");
        let root = scratch_dir("runtime-removed");
        install_fake_runtime(&root);

        let installed = fingerprint_for_runtime(&root);
        write_cached_capability(&installed, &healthy_capability());

        fs::remove_dir_all(&root).unwrap();
        let gone = fingerprint_for_runtime(&root);
        assert_eq!(gone.llama_runtime, None);
        assert!(
            read_cached_capability(&gone).is_none(),
            "a runtime that was removed must not be served the entry it had while installed"
        );
    }

    #[test]
    fn an_install_with_no_runtime_at_all_still_hits_its_cache() {
        // A user who never installed a runtime has None on both sides, and None
        // must compare equal to None: treating "no runtime" as a change would
        // make every launch pay both subprocesses forever. The runtime
        // appearing later is a real change and must miss.
        let _home = CapabilityCacheHome::new("no-runtime");
        let root = scratch_dir("runtime-absent");

        let without = fingerprint_for_runtime(&root);
        assert_eq!(without.llama_runtime, None);
        write_cached_capability(&without, &healthy_capability());
        assert!(
            read_cached_capability(&fingerprint_for_runtime(&root)).is_some(),
            "an install with no runtime must not miss its own cache every launch"
        );

        install_fake_runtime(&root);
        let with = fingerprint_for_runtime(&root);
        assert!(with.llama_runtime.is_some());
        assert!(
            read_cached_capability(&with).is_none(),
            "a runtime installed after the cache was written must be probed"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_cached_capability_that_is_itself_broken_is_never_served() {
        // write_cached_capability runs before the ready check, so the entry for
        // a broken runtime does reach disk. Reading it back must still refuse
        // it even though the fingerprint matches exactly, or the repair the
        // stale verdict starts would be undone by the next launch.
        let home = CapabilityCacheHome::new("broken");
        let root = scratch_dir("runtime-broken");
        install_fake_runtime(&root);

        let fingerprint = fingerprint_for_runtime(&root);
        let mut capability = healthy_capability();
        capability.llama_runtime_ok = Some(false);
        capability.llama_runtime_reason = Some("llama_runtime_payload_incomplete".to_string());
        write_cached_capability(&fingerprint, &capability);

        let raw =
            fs::read(home.cache_file()).expect("the broken verdict is written like any other");
        assert!(String::from_utf8_lossy(&raw).contains("llama_runtime_payload_incomplete"));
        assert!(
            read_cached_capability(&fingerprint).is_none(),
            "a broken runtime must be re-probed even when nothing on disk moved"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_schema_three_cache_file_on_disk_misses_and_is_rewritten_as_schema_four() {
        // Byte for byte what the previous release wrote, sitting in every
        // existing install. Everything but the schema and the new key matches
        // the fingerprint, so the miss can only come from the version bump.
        // It must not panic, must not be served, and must be replaced.
        let home = CapabilityCacheHome::new("schema-three");
        let root = scratch_dir("runtime-schema-three");
        install_fake_runtime(&root);
        let fingerprint = fingerprint_for_runtime(&root);

        let path = home.cache_file();
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let previous_release = format!(
            r#"{{
  "schema": 3,
  "bin_path": "{bin_path}",
  "bin_size": {bin_size},
  "bin_mtime_ms": {bin_mtime_ms},
  "studio_root_id": null,
  "marker_path": null,
  "marker_size": null,
  "marker_mtime_ms": null,
  "desktop_protocol_version": {protocol},
  "desktop_manageability_version": {manageability},
  "capability": {capability}
}}"#,
            bin_path = fingerprint.bin_path,
            bin_size = fingerprint.bin_size,
            bin_mtime_ms = fingerprint.bin_mtime_ms,
            protocol = DESKTOP_PROTOCOL_VERSION,
            manageability = DESKTOP_MANAGEABILITY_VERSION,
            capability = pre_pr_capability_json(),
        );
        fs::write(&path, previous_release).unwrap();
        assert!(
            read_cached_capability(&fingerprint).is_none(),
            "a schema 3 entry reached Ready without ever looking at the runtime"
        );

        write_cached_capability(&fingerprint, &healthy_capability());
        let rewritten: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).expect("the rewrite must be JSON");
        assert_eq!(
            rewritten["schema"].as_u64(),
            Some(u64::from(MANAGED_CAPABILITY_CACHE_SCHEMA))
        );
        assert_eq!(
            rewritten["llama_runtime"].as_str(),
            fingerprint.llama_runtime.as_deref(),
            "the rewritten entry must carry the runtime the old one had no room for"
        );
        assert!(
            read_cached_capability(&fingerprint).is_some(),
            "the replacement entry must then hit, or the upgrade never stops re-probing"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn files_below_the_binary_directory_are_not_walked() {
        // The walk is one level deep on purpose: it runs on the launch path.
        // A nested directory must contribute nothing at all, not even its own
        // children, or a model cache parked under build/bin would make the
        // fingerprint cost grow without bound.
        let root = scratch_dir("runtime-nested");
        let bin = install_fake_runtime(&root);
        let flat = llama_runtime_fingerprint_at(&root).unwrap();

        let nested = bin.join("vendor").join("deep");
        fs::create_dir_all(&nested).unwrap();
        fs::write(nested.join("payload.bin"), vec![0u8; 8192]).unwrap();
        assert_eq!(
            flat,
            llama_runtime_fingerprint_at(&root).unwrap(),
            "a subdirectory and its contents are not part of the runtime fingerprint"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_replacement_of_the_same_size_is_a_collision_this_fingerprint_does_not_catch() {
        // Stated rather than glossed: the fingerprint is a count and a byte
        // total, with no mtime and no content hash, so a file rewritten in
        // place at exactly its old length is invisible and the cached Ready
        // survives it. That is accepted here because the case this guards is
        // quarantine and partial extraction, which always change one of the
        // two. A size change of any kind is caught.
        let root = scratch_dir("runtime-collision");
        let bin = install_fake_runtime(&root);
        let original = llama_runtime_fingerprint_at(&root).unwrap();

        fs::write(bin.join("llama-server"), vec![0xABu8; 4096]).unwrap();
        assert_eq!(
            original,
            llama_runtime_fingerprint_at(&root).unwrap(),
            "same count and same bytes is a collision, and this asserts it honestly"
        );

        fs::write(bin.join("llama-server"), vec![0u8; 4097]).unwrap();
        assert_ne!(
            original,
            llama_runtime_fingerprint_at(&root).unwrap(),
            "one byte of size difference must still invalidate"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_runtime_root_that_is_a_file_fingerprints_as_no_runtime() {
        // UNSLOTH_LLAMA_CPP_PATH can point at anything a user typed. Joining
        // build/bin onto a regular file makes read_dir fail, which must degrade
        // to no runtime coverage rather than panic on the launch path.
        let parent = scratch_dir("runtime-is-a-file");
        let root = parent.join("llama.cpp");
        fs::write(&root, "not a directory").unwrap();
        assert_eq!(llama_runtime_fingerprint_at(&root), None);
        let _ = fs::remove_dir_all(&parent);
    }

    #[test]
    fn a_runtime_path_with_spaces_and_non_ascii_fingerprints_normally() {
        // Home directories are named after people. A path this walk cannot
        // handle would silently drop the runtime out of the fingerprint for
        // those users only, which is the hardest kind of gap to notice.
        let parent = scratch_dir("runtime-unicode");
        let root = parent.join("Мой каталог ünïcode llama.cpp");
        install_fake_runtime(&root);
        assert_eq!(
            llama_runtime_fingerprint_at(&root).as_deref(),
            Some("2:6144")
        );
        let _ = fs::remove_dir_all(&parent);
    }

    #[test]
    fn the_runtime_fingerprint_does_not_depend_on_directory_order() {
        // read_dir order is unspecified, and the venv half of this fingerprint
        // sorts for exactly that reason. Two trees with the same files created
        // in opposite orders must agree, or an install would miss its own cache
        // on some launches and not others, which is worse than never caching.
        let parent = scratch_dir("runtime-order");
        let names = ["llama-server", "libggml.so", "llama-cli", "libmtmd.so"];

        let forwards = parent.join("forwards");
        let forwards_bin = runtime_bin_dir(&forwards);
        fs::create_dir_all(&forwards_bin).unwrap();
        for (index, name) in names.iter().enumerate() {
            fs::write(forwards_bin.join(name), vec![0u8; 100 + index]).unwrap();
        }

        let backwards = parent.join("backwards");
        let backwards_bin = runtime_bin_dir(&backwards);
        fs::create_dir_all(&backwards_bin).unwrap();
        for (index, name) in names.iter().enumerate().rev() {
            fs::write(backwards_bin.join(name), vec![0u8; 100 + index]).unwrap();
        }

        assert_eq!(
            llama_runtime_fingerprint_at(&forwards),
            llama_runtime_fingerprint_at(&backwards)
        );
        let _ = fs::remove_dir_all(&parent);
    }

    #[test]
    fn a_large_runtime_directory_fingerprints_fast_enough_for_the_launch_path() {
        // This walk runs before the window is usable, on every launch, ahead of
        // the probes it is meant to save. A tree with a build directory's worth
        // of files must still cost milliseconds, so the bound is generous
        // enough never to flake and tight enough to catch a walk that started
        // reading file contents or recursing.
        let root = scratch_dir("runtime-large");
        let bin = runtime_bin_dir(&root);
        fs::create_dir_all(&bin).unwrap();
        for index in 0..5000 {
            fs::write(bin.join(format!("artifact-{index:05}.o")), [0u8; 1]).unwrap();
        }

        let started = Instant::now();
        let fingerprint = llama_runtime_fingerprint_at(&root);
        let elapsed = started.elapsed();
        assert_eq!(fingerprint.as_deref(), Some("5000:5000"));
        assert!(
            elapsed < Duration::from_secs(2),
            "5000 files took {elapsed:?}, which is too much to spend before the window opens"
        );

        let _ = fs::remove_dir_all(&root);
    }

    // Symlinks are what a developer's own checkout and a hand-placed
    // UNSLOTH_LLAMA_CPP_PATH look like, and they are the one input that can make
    // this walk answer about a tree other than the one it was given.
    #[cfg(unix)]
    #[test]
    fn a_symlinked_runtime_root_fingerprints_the_tree_it_points_at() {
        let parent = scratch_dir("runtime-symlinked-root");
        let real = parent.join("real");
        let bin = install_fake_runtime(&real);
        let linked = parent.join("linked");
        std::os::unix::fs::symlink(&real, &linked).unwrap();

        assert_eq!(
            llama_runtime_fingerprint_at(&real),
            llama_runtime_fingerprint_at(&linked),
            "a symlinked root must see the same tree, or an override written that way is not covered"
        );
        // And it must keep tracking it: a quarantine through the link counts.
        fs::remove_file(bin.join("libggml-base.so")).unwrap();
        assert_eq!(
            llama_runtime_fingerprint_at(&linked).as_deref(),
            Some("1:4096")
        );

        let _ = fs::remove_dir_all(&parent);
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_file_inside_the_runtime_is_not_counted() {
        // DirEntry::metadata does not follow the link, so a symlinked binary
        // reads as neither a file nor a directory and contributes nothing.
        // Documented here because it bounds what this fingerprint promises: a
        // tree whose binaries are symlinks into a store is covered only by the
        // real files beside them. The managed installer writes real files, so
        // the shipped layout is unaffected.
        let parent = scratch_dir("runtime-symlinked-file");
        let root = parent.join("llama.cpp");
        let bin = install_fake_runtime(&root);
        let before = llama_runtime_fingerprint_at(&root).unwrap();

        std::os::unix::fs::symlink(bin.join("llama-server"), bin.join("llama-cli")).unwrap();
        assert_eq!(
            before,
            llama_runtime_fingerprint_at(&root).unwrap(),
            "a symlink to a file inside the directory is invisible to this walk"
        );

        let _ = fs::remove_dir_all(&parent);
    }

    #[cfg(unix)]
    #[test]
    fn a_dangling_symlink_does_not_break_the_walk() {
        // What a quarantine that removed a link target leaves behind. The walk
        // must finish and answer about the real files rather than fail and
        // report the whole runtime missing.
        let parent = scratch_dir("runtime-dangling");
        let root = parent.join("llama.cpp");
        let bin = install_fake_runtime(&root);
        let before = llama_runtime_fingerprint_at(&root).unwrap();

        std::os::unix::fs::symlink(bin.join("gone.so"), bin.join("libggml.so")).unwrap();
        assert_eq!(
            before,
            llama_runtime_fingerprint_at(&root).unwrap(),
            "a broken link must be skipped, not counted and not fatal"
        );

        let _ = fs::remove_dir_all(&parent);
    }

    #[cfg(unix)]
    #[test]
    fn a_runtime_directory_that_cannot_be_read_fingerprints_as_no_runtime() {
        // A tightened-down or half-owned install directory. read_dir fails, and
        // the answer must be None rather than a panic: None only ever costs the
        // probes the cache would have saved.
        use std::os::unix::fs::PermissionsExt;

        let root = scratch_dir("runtime-denied");
        let bin = install_fake_runtime(&root);
        let mut denied = fs::metadata(&bin).unwrap().permissions();
        denied.set_mode(0o000);
        fs::set_permissions(&bin, denied).unwrap();

        // Mode bits do not apply to a privileged user, and some CI images run
        // as root, so the assertion is made only where the denial is real.
        if fs::read_dir(&bin).is_err() {
            assert_eq!(llama_runtime_fingerprint_at(&root), None);
        }

        let mut restored = fs::metadata(&bin).unwrap().permissions();
        restored.set_mode(0o755);
        fs::set_permissions(&bin, restored).unwrap();
        let _ = fs::remove_dir_all(&root);
    }
}
