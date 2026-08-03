use super::types::ManagedProbe;
use super::version::{
    backend_version_stale_reason, DESKTOP_MANAGEABILITY_VERSION, DESKTOP_PROTOCOL_VERSION,
};
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant, UNIX_EPOCH};
use tokio::io::AsyncReadExt;
use tokio::process::Command;

// 3: the cached capability gained studio_install_ok / studio_install_reason.
const MANAGED_CAPABILITY_CACHE_SCHEMA: u16 = 3;

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

fn managed_bin_fingerprint(bin: &Path) -> Option<ManagedBinFingerprint> {
    let bin_metadata = fs::metadata(bin).ok()?;
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

async fn run_cli_probe(bin: &Path, args: &[&str]) -> bool {
    let started = Instant::now();
    let mut cmd = Command::new(bin);
    cmd.args(args).stdout(Stdio::null()).stderr(Stdio::null());

    #[cfg(target_os = "linux")]
    if std::env::var_os("APPIMAGE").is_some() {
        cmd.env_remove("LD_LIBRARY_PATH");
        cmd.env_remove("PYTHONHOME");
        cmd.env_remove("PYTHONPATH");
    }

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
        return false;
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
    ok
}

async fn probe_cli_capability(bin: &Path) -> Option<DesktopCapability> {
    let started = Instant::now();
    let mut cmd = Command::new(bin);
    cmd.args(["studio", "desktop-capabilities", "--json"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null());

    #[cfg(target_os = "linux")]
    if std::env::var_os("APPIMAGE").is_some() {
        cmd.env_remove("LD_LIBRARY_PATH");
        cmd.env_remove("PYTHONHOME");
        cmd.env_remove("PYTHONPATH");
    }

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
        return None;
    };
    let Some(mut stdout) = child.stdout.take() else {
        return None;
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
            return None;
        }
        _ => {
            info!(
                "Managed desktop-capabilities probe exited unsuccessfully in {}ms",
                started.elapsed().as_millis()
            );
            return None;
        }
    }

    let mut output = Vec::new();
    if stdout.read_to_end(&mut output).await.is_err() {
        return None;
    }

    let capability = serde_json::from_slice::<DesktopCapability>(&output).ok();
    info!(
        "Managed desktop-capabilities probe finished ok={} in {}ms",
        capability.is_some(),
        started.elapsed().as_millis()
    );
    capability
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
    backend_version_stale_reason(capability.version.as_deref())
}

fn desktop_capability_ready(capability: &DesktopCapability) -> bool {
    desktop_capability_stale_reason(capability).is_none()
}

pub(super) async fn probe_managed_bin(bin: PathBuf) -> ManagedProbe {
    let started = Instant::now();
    // Always verify the managed CLI actually launches before trusting the cache.
    // A matching capability fingerprint does not prove the binary can still run:
    // its venv interpreter or a runtime dependency can be broken while the
    // path/size/mtime/markers are unchanged, so the -h probe runs first and a
    // non-launchable install is reported Stale for repair. The capability cache
    // below still skips the heavier desktop-capabilities probe on a hit.
    if !run_cli_probe(&bin, &["-h"]).await {
        info!(
            "Managed preflight: cli unusable for {:?} in {}ms",
            bin,
            started.elapsed().as_millis()
        );
        return ManagedProbe::Stale {
            bin,
            reason: "cli_unusable".to_string(),
        };
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

    let capability = probe_cli_capability(&bin).await;
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
        reason: "desktop_capability_probe_failed".to_string(),
    }
}

pub(super) async fn probe_managed_install() -> ManagedProbe {
    let started = Instant::now();
    let result = match crate::process::find_unsloth_binary() {
        Some(bin) => probe_managed_bin(bin).await,
        None => ManagedProbe::Missing,
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
            version: Some("2026.7.5".to_string()),
        }
    }

    #[test]
    fn complete_install_is_ready() {
        assert_eq!(desktop_capability_stale_reason(&healthy_capability()), None);
        assert!(desktop_capability_ready(&healthy_capability()));
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
}
