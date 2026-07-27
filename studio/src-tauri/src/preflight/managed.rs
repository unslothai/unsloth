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

const MANAGED_CAPABILITY_CACHE_SCHEMA: u16 = 2;

const FNV64_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV64_PRIME: u64 = 0x100000001b3;
const HASHED_MARKER_MAX_BYTES: u64 = 64 * 1024;

const FALLBACK_MARKER_NAMES: &[&str] = &[
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
    version: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DesktopRuntimeCheck {
    runtime_ready: bool,
    reason: Option<String>,
    module: Option<String>,
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

fn marker_candidates_for_bin(bin: &Path) -> Vec<PathBuf> {
    let Some(scripts_dir) = bin.parent() else {
        return Vec::new();
    };
    let Some(venv_dir) = scripts_dir.parent() else {
        return Vec::new();
    };
    let mut out = Vec::new();

    #[cfg(unix)]
    {
        if let Ok(lib_dir) = fs::read_dir(venv_dir.join("lib")) {
            for entry in lib_dir.flatten() {
                out.push(
                    entry
                        .path()
                        .join("site-packages")
                        .join("unsloth_cli")
                        .join("commands")
                        .join("studio.py"),
                );
            }
        }
    }
    for marker_name in FALLBACK_MARKER_NAMES {
        out.push(venv_dir.join(marker_name));
        out.push(scripts_dir.join(marker_name));
    }

    out.push(
        venv_dir
            .join("Lib")
            .join("site-packages")
            .join("unsloth_cli")
            .join("commands")
            .join("studio.py"),
    );
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
    let marker_hash = marker_entries
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
    let marker_path = (!marker_entries.is_empty()).then(|| "markers".to_string());
    let marker_size = (!marker_entries.is_empty()).then(|| marker_entries.len() as u64);
    let marker_mtime_ms = (!marker_entries.is_empty()).then_some(marker_hash);

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
    let mut cmd = Command::new(bin);
    cmd.args(args).stdout(Stdio::null()).stderr(Stdio::null());

    #[cfg(target_os = "linux")]
    if std::env::var_os("APPIMAGE").is_some() {
        cmd.env_remove("LD_LIBRARY_PATH");
        cmd.env_remove("PYTHONHOME");
        cmd.env_remove("PYTHONPATH");
    }

    cmd.env_remove("UNSLOTH_STUDIO_HOME");
    cmd.env_remove("STUDIO_HOME");

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(crate::process::CREATE_NO_WINDOW);
    }

    let Ok(mut child) = cmd.spawn() else {
        return false;
    };
    match tokio::time::timeout(Duration::from_secs(10), child.wait()).await {
        Ok(Ok(status)) => status.success(),
        _ => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            false
        }
    }
}

/// Room for the payload and what preceded it. Bounded because the binary being
/// probed is a possibly damaged install, and keeping every byte lets an import
/// stuck printing grow this buffer for the whole timeout.
const PROBE_TAIL_BYTES: usize = 64 * 1024;

/// Budget for a whole probe, the drain included. A descendant that inherited
/// stdout keeps the pipe open after the leader exits, so a drain outside the
/// timeout would wedge preflight for good.
const PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// Drained while the wait runs: the backend import can print more than the pipe
/// holds, and waiting first deadlocks, timing out a healthy install into repair.
fn drain_probe_tail(mut stdout: tokio::process::ChildStdout) -> tokio::task::JoinHandle<Vec<u8>> {
    tokio::spawn(async move {
        let mut tail = Vec::new();
        let mut chunk = [0u8; 8192];
        while let Ok(read) = stdout.read(&mut chunk).await {
            if read == 0 {
                break;
            }
            tail.extend_from_slice(&chunk[..read]);
            if tail.len() > PROBE_TAIL_BYTES {
                tail.drain(..tail.len() - PROBE_TAIL_BYTES);
            }
        }
        tail
    })
}

/// The probe could not answer: it would not run, or exited without a payload.
/// Distinct from a CLI that is simply too old to know the command.
const RUNTIME_PROBE_FAILED: &str = "studio_runtime_probe_failed";
/// A hung binary, kept out of the fallback below: an older CLI rejects the
/// unknown command at once, so only a broken one reaches the timeout.
const RUNTIME_PROBE_TIMEOUT: &str = "studio_runtime_probe_timeout";
/// A CLI predating the runtime probe. Repairable: the update installs one that
/// can answer, so the next preflight checks the venv for real.
const RUNTIME_CHECK_UNSUPPORTED: &str = "studio_runtime_check_unsupported";

/// True when the CLI is older than `desktop-runtime-check` yet still launches.
/// Its usage error is indistinguishable from a crashed probe, so `--help`
/// resolves the command without running it.
async fn predates_runtime_check(bin: &Path) -> bool {
    !run_cli_probe(bin, &["studio", "desktop-runtime-check", "--help"]).await
        && run_cli_probe(bin, &["-h"]).await
}

async fn probe_cli_runtime(bin: &Path) -> Result<(), String> {
    let started = Instant::now();
    let mut cmd = Command::new(bin);
    cmd.args(["studio", "desktop-runtime-check", "--json"])
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

    let Ok(mut child) = cmd.spawn() else {
        info!(
            "Managed runtime probe failed to spawn in {}ms",
            started.elapsed().as_millis()
        );
        return Err(RUNTIME_PROBE_FAILED.to_string());
    };
    let Some(stdout) = child.stdout.take() else {
        return Err(RUNTIME_PROBE_FAILED.to_string());
    };

    let mut reader = drain_probe_tail(stdout);
    let deadline = tokio::time::Instant::now() + PROBE_TIMEOUT;

    let status = match tokio::time::timeout_at(deadline, child.wait()).await {
        Ok(Ok(status)) => status,
        _ => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            reader.abort();
            info!(
                "Managed runtime probe timed out in {}ms",
                started.elapsed().as_millis()
            );
            return Err(RUNTIME_PROBE_TIMEOUT.to_string());
        }
    };

    let output = match tokio::time::timeout_at(deadline, &mut reader).await {
        Ok(Ok(output)) => output,
        Ok(Err(_)) => return Err(RUNTIME_PROBE_FAILED.to_string()),
        Err(_) => {
            reader.abort();
            info!(
                "Managed runtime probe left stdout open in {}ms",
                started.elapsed().as_millis()
            );
            return Err(RUNTIME_PROBE_TIMEOUT.to_string());
        }
    };
    let payload = String::from_utf8_lossy(&output)
        .lines()
        .rev()
        .find_map(|line| serde_json::from_str::<DesktopRuntimeCheck>(line).ok());
    let result = match payload {
        Some(payload) if status.success() && payload.runtime_ready => Ok(()),
        Some(payload) => {
            let reason = match payload.reason.as_deref() {
                Some("missing_dependency") => {
                    info!(
                        "Managed runtime probe missing dependency module={}",
                        payload.module.as_deref().unwrap_or("unknown")
                    );
                    "studio_runtime_missing_dependency"
                }
                Some("backend_import_failed") => "studio_runtime_import_failed",
                Some("backend_startup_failed") => super::STUDIO_RUNTIME_STARTUP_FAILED,
                _ => RUNTIME_PROBE_FAILED,
            };
            Err(reason.to_string())
        }
        None => Err(RUNTIME_PROBE_FAILED.to_string()),
    };
    info!(
        "Managed runtime probe finished ok={} in {}ms",
        result.is_ok(),
        started.elapsed().as_millis()
    );
    result
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

    let Ok(mut child) = cmd.spawn() else {
        info!(
            "Managed desktop-capabilities probe failed to spawn in {}ms",
            started.elapsed().as_millis()
        );
        return None;
    };
    let Some(stdout) = child.stdout.take() else {
        return None;
    };

    let mut reader = drain_probe_tail(stdout);
    let deadline = tokio::time::Instant::now() + PROBE_TIMEOUT;

    match tokio::time::timeout_at(deadline, child.wait()).await {
        Ok(Ok(status)) if status.success() => {}
        Err(_) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            reader.abort();
            info!(
                "Managed desktop-capabilities probe timed out in {}ms",
                started.elapsed().as_millis()
            );
            return None;
        }
        _ => {
            reader.abort();
            info!(
                "Managed desktop-capabilities probe exited unsuccessfully in {}ms",
                started.elapsed().as_millis()
            );
            return None;
        }
    }

    let Ok(Ok(output)) = tokio::time::timeout_at(deadline, &mut reader).await else {
        reader.abort();
        return None;
    };

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
    backend_version_stale_reason(capability.version.as_deref())
}

fn desktop_capability_ready(capability: &DesktopCapability) -> bool {
    desktop_capability_stale_reason(capability).is_none()
}

pub(super) async fn probe_managed_bin(bin: PathBuf) -> ManagedProbe {
    let started = Instant::now();
    // Runtime readiness is intentionally uncached: a matching capability
    // fingerprint proves protocol compatibility, not that the venv is complete.
    if let Err(reason) = probe_cli_runtime(&bin).await {
        // A CLI too old for this subcommand is what the interrupted installs
        // shipped with, and -h passes without touching the backend.
        let reason = if reason == RUNTIME_PROBE_FAILED && predates_runtime_check(&bin).await {
            RUNTIME_CHECK_UNSUPPORTED.to_string()
        } else {
            reason
        };
        info!(
            "Managed preflight: runtime unusable for {:?} reason={} in {}ms",
            bin,
            reason,
            started.elapsed().as_millis()
        );
        return ManagedProbe::Stale { bin, reason };
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
        Some(bin) if crate::install::managed_install_in_progress() => {
            info!(
                "Managed preflight: interrupted desktop installation marker found for {:?}",
                bin
            );
            ManagedProbe::Stale {
                bin,
                reason: "install_incomplete".to_string(),
            }
        }
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

/// Ready, or the reason it is not, so repair can tell an unrepairable cause
/// apart from a stale install.
pub async fn managed_install_state() -> Result<(), Option<String>> {
    match probe_managed_install().await {
        ManagedProbe::Ready { .. } => Ok(()),
        ManagedProbe::Stale { reason, .. } => Err(Some(reason)),
        _ => Err(None),
    }
}
