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
const MANAGED_CAPABILITY_CACHE_SCHEMA: u16 = 3;

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
    })
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
}
