use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncReadExt;
use tokio::process::Command;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DesktopPreflightDisposition {
    NotInstalled,
    ManagedReady,
    ManagedStale,
    AttachedReady,
    ExternalConflict,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DesktopPreflightResult {
    pub disposition: DesktopPreflightDisposition,
    pub reason: Option<String>,
    pub port: Option<u16>,
    pub can_auto_repair: bool,
    pub managed_bin: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalBackendConflict {
    pub port: u16,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ManagedProbe {
    Missing,
    Ready { bin: PathBuf },
    Stale { bin: PathBuf, reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BackendProbe {
    Missing,
    Ready { port: u16 },
    Old { port: u16, reason: String },
    ExternalConflict { port: u16, reason: String },
}

#[derive(Debug, Deserialize)]
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
struct DesktopOwnerHealth {
    kind: Option<String>,
    token_sha256: Option<String>,
}

#[derive(Debug)]
struct BackendHealth {
    desktop_protocol_version: Option<u16>,
    desktop_manageability_version: Option<u16>,
    supports_desktop_auth: Option<bool>,
    supports_desktop_backend_ownership: Option<bool>,
    studio_root_id: Option<String>,
    desktop_owner: Option<DesktopOwnerHealth>,
    version: Option<String>,
    stale_reason: Option<String>,
}

const DESKTOP_PROTOCOL_VERSION: u16 = 1;
const DESKTOP_MANAGEABILITY_VERSION: u16 = 1;
// Explicit backend package minimum, not the desktop app Cargo version: backend
// and app releases can diverge. When bumping, verify this package exists on PyPI.
const MIN_DESKTOP_BACKEND_VERSION: &str = "2026.5.2";

fn release_auto_repair() -> bool {
    !cfg!(debug_assertions)
}

fn parse_version(value: &str) -> Option<[u64; 3]> {
    let value = value.trim();
    let mut parts = value.splitn(3, '.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch_and_suffix = parts.next()?;
    let patch_len = patch_and_suffix
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(patch_and_suffix.len());
    if patch_len == 0 {
        return None;
    }
    let suffix = &patch_and_suffix[patch_len..];
    if !version_suffix_allowed(suffix) {
        return None;
    }
    Some([major, minor, patch_and_suffix[..patch_len].parse().ok()?])
}

fn version_suffix_allowed(suffix: &str) -> bool {
    suffix.is_empty()
        || suffix.starts_with('+')
        || suffix.starts_with(".post")
        || suffix.starts_with(".dev")
        || suffix.starts_with(".rc")
        || suffix.starts_with("post")
        || suffix.starts_with("dev")
        || suffix.starts_with("rc")
        || suffix.starts_with('a')
        || suffix.starts_with('b')
}

fn backend_version_compatible(version: Option<&str>) -> bool {
    let Some(version) = version else {
        return false;
    };
    if cfg!(debug_assertions) && version == "dev" {
        return true;
    }
    let Some(actual) = parse_version(version) else {
        return false;
    };
    let Some(minimum) = parse_version(MIN_DESKTOP_BACKEND_VERSION) else {
        return false;
    };
    actual >= minimum
}

fn backend_version_stale_reason(version: Option<&str>) -> Option<String> {
    if backend_version_compatible(version) {
        return None;
    }
    match version {
        None | Some("") => Some("desktop_backend_version_missing".to_string()),
        Some("dev") if !cfg!(debug_assertions) => {
            Some("desktop_backend_version_invalid".to_string())
        }
        Some(value) if parse_version(value).is_none() => {
            Some("desktop_backend_version_invalid".to_string())
        }
        Some(_) => Some("desktop_backend_version_too_old".to_string()),
    }
}

fn managed_bin_for_result(managed: &ManagedProbe) -> Option<PathBuf> {
    match managed {
        ManagedProbe::Ready { bin } | ManagedProbe::Stale { bin, .. } => Some(bin.clone()),
        ManagedProbe::Missing => None,
    }
}

fn choose_preflight(managed: ManagedProbe, backend: BackendProbe) -> DesktopPreflightResult {
    match (backend, managed) {
        (BackendProbe::ExternalConflict { port, reason }, managed) => DesktopPreflightResult {
            disposition: DesktopPreflightDisposition::ExternalConflict,
            reason: Some(reason),
            port: Some(port),
            can_auto_repair: false,
            managed_bin: managed_bin_for_result(&managed),
        },
        (BackendProbe::Ready { port }, managed) => DesktopPreflightResult {
            disposition: DesktopPreflightDisposition::AttachedReady,
            reason: None,
            port: Some(port),
            can_auto_repair: false,
            managed_bin: managed_bin_for_result(&managed),
        },
        (_, managed) => match managed {
            ManagedProbe::Ready { bin } => DesktopPreflightResult {
                disposition: DesktopPreflightDisposition::ManagedReady,
                reason: None,
                port: None,
                can_auto_repair: false,
                managed_bin: Some(bin),
            },
            ManagedProbe::Stale { bin, reason } => DesktopPreflightResult {
                disposition: DesktopPreflightDisposition::ManagedStale,
                reason: Some(reason),
                port: None,
                can_auto_repair: release_auto_repair(),
                managed_bin: Some(bin),
            },
            ManagedProbe::Missing => DesktopPreflightResult {
                disposition: DesktopPreflightDisposition::NotInstalled,
                reason: None,
                port: None,
                can_auto_repair: false,
                managed_bin: None,
            },
        },
    }
}

async fn run_cli_probe(bin: &std::path::Path, args: &[&str]) -> bool {
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

async fn probe_cli_capability(bin: &std::path::Path) -> Option<DesktopCapability> {
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
            return None;
        }
        _ => return None,
    }

    let mut output = Vec::new();
    if stdout.read_to_end(&mut output).await.is_err() {
        return None;
    }

    serde_json::from_slice::<DesktopCapability>(&output).ok()
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

async fn probe_managed_bin(bin: PathBuf) -> ManagedProbe {
    if !run_cli_probe(&bin, &["-h"]).await {
        return ManagedProbe::Stale {
            bin,
            reason: "cli_unusable".to_string(),
        };
    }

    let capability = probe_cli_capability(&bin).await;
    if let Some(capability) = capability {
        if desktop_capability_ready(&capability) {
            return ManagedProbe::Ready { bin };
        }
        return ManagedProbe::Stale {
            bin,
            reason: desktop_capability_stale_reason(&capability)
                .unwrap_or_else(|| "desktop_capability_incompatible".to_string()),
        };
    }

    ManagedProbe::Stale {
        bin,
        reason: "desktop_capability_probe_failed".to_string(),
    }
}

async fn probe_managed_install() -> ManagedProbe {
    match crate::process::find_unsloth_binary() {
        Some(bin) => probe_managed_bin(bin).await,
        None => ManagedProbe::Missing,
    }
}

pub async fn managed_install_ready() -> bool {
    matches!(probe_managed_install().await, ManagedProbe::Ready { .. })
}

async fn backend_health(client: &reqwest::Client, port: u16) -> Option<BackendHealth> {
    let url = format!("http://127.0.0.1:{port}/api/health");
    let response = client.get(url).send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    let json = response.json::<serde_json::Value>().await.ok()?;
    let healthy = json
        .get("status")
        .and_then(|v| v.as_str())
        .map(|s| s == "healthy")
        .unwrap_or(false);
    let service = json
        .get("service")
        .and_then(|v| v.as_str())
        .map(|s| s == "Unsloth UI Backend")
        .unwrap_or(false);
    if !healthy || !service {
        return None;
    }

    let desktop_protocol_version = json
        .get("desktop_protocol_version")
        .and_then(|v| v.as_u64())
        .and_then(|v| u16::try_from(v).ok());
    let desktop_manageability_version = json
        .get("desktop_manageability_version")
        .and_then(|v| v.as_u64())
        .and_then(|v| u16::try_from(v).ok());
    let supports_desktop_auth = json.get("supports_desktop_auth").and_then(|v| v.as_bool());
    let supports_desktop_backend_ownership = json
        .get("supports_desktop_backend_ownership")
        .and_then(|v| v.as_bool());
    let studio_root_id = json
        .get("studio_root_id")
        .and_then(|v| v.as_str())
        .map(ToOwned::to_owned);
    let desktop_owner = json
        .get("desktop_owner")
        .and_then(|v| serde_json::from_value::<DesktopOwnerHealth>(v.clone()).ok());
    let version = json
        .get("version")
        .and_then(|v| v.as_str())
        .map(ToOwned::to_owned);
    let stale_reason = match supports_desktop_auth {
        Some(false) => json
            .get("desktop_auth_stale_reason")
            .and_then(|v| v.as_str())
            .map(ToOwned::to_owned),
        _ => None,
    };
    Some(BackendHealth {
        desktop_protocol_version,
        desktop_manageability_version,
        supports_desktop_auth,
        supports_desktop_backend_ownership,
        studio_root_id,
        desktop_owner,
        version,
        stale_reason,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BackendRootStatus {
    SameRoot,
    ForeignRoot,
    AmbiguousRoot,
    ExpectedUnavailable,
}

fn backend_root_status(
    health: &BackendHealth,
    expected_studio_root_id: Option<&str>,
) -> BackendRootStatus {
    let Some(expected) = expected_studio_root_id else {
        return BackendRootStatus::ExpectedUnavailable;
    };

    match health.studio_root_id.as_deref() {
        Some(actual) if actual == expected => BackendRootStatus::SameRoot,
        Some(actual) if crate::desktop_backend_owner::is_valid_studio_root_id(actual) => {
            BackendRootStatus::ForeignRoot
        }
        _ => BackendRootStatus::AmbiguousRoot,
    }
}

fn backend_has_verified_desktop_owner(health: &BackendHealth) -> bool {
    let Some(owner) = health.desktop_owner.as_ref() else {
        return false;
    };
    crate::desktop_backend_owner::health_matches_desktop_owner(
        health.studio_root_id.as_deref(),
        owner.kind.as_deref(),
        owner.token_sha256.as_deref(),
    )
}

fn backend_capability_stale_reason(health: &BackendHealth) -> Option<String> {
    if health.desktop_protocol_version != Some(DESKTOP_PROTOCOL_VERSION) {
        return health
            .stale_reason
            .clone()
            .or_else(|| Some("desktop_protocol_incompatible".to_string()));
    }
    if health.supports_desktop_auth != Some(true) {
        return health
            .stale_reason
            .clone()
            .or_else(|| Some("desktop_auth_unsupported".to_string()));
    }
    if health.desktop_manageability_version.unwrap_or(0) < DESKTOP_MANAGEABILITY_VERSION {
        return Some("desktop_manageability_unsupported".to_string());
    }
    if health.supports_desktop_backend_ownership != Some(true) {
        return Some("desktop_backend_ownership_unsupported".to_string());
    }
    backend_version_stale_reason(health.version.as_deref())
}

#[derive(Serialize)]
struct DesktopLoginProbe<'a> {
    secret: &'a str,
}

async fn backend_desktop_auth_status(
    client: &reqwest::Client,
    port: u16,
    health: &BackendHealth,
    expected_studio_root_id: Option<&str>,
) -> BackendProbe {
    let root_status = backend_root_status(health, expected_studio_root_id);
    let verified_owner = backend_has_verified_desktop_owner(health);
    let same_root_external = root_status == BackendRootStatus::SameRoot && !verified_owner;

    match root_status {
        BackendRootStatus::AmbiguousRoot | BackendRootStatus::ExpectedUnavailable => {
            return BackendProbe::ExternalConflict {
                port,
                reason: "ambiguous_root_external_backend_active".to_string(),
            };
        }
        BackendRootStatus::ForeignRoot => {
            return BackendProbe::Old {
                port,
                reason: "studio_root_id_mismatch".to_string(),
            };
        }
        BackendRootStatus::SameRoot => {}
    }

    if let Some(reason) = backend_capability_stale_reason(health) {
        return if same_root_external {
            BackendProbe::ExternalConflict { port, reason }
        } else {
            BackendProbe::Old { port, reason }
        };
    }

    let url = format!("http://127.0.0.1:{port}/api/auth/desktop-login");
    let response = client
        .post(url)
        .json(&DesktopLoginProbe {
            secret: "desktop-preflight-invalid-secret",
        })
        .send()
        .await;

    let Ok(response) = response else {
        let reason = backend_capability_stale_reason(health)
            .unwrap_or_else(|| "desktop_login_probe_failed".to_string());
        return if same_root_external {
            BackendProbe::ExternalConflict { port, reason }
        } else {
            BackendProbe::Old { port, reason }
        };
    };

    match response.status() {
        reqwest::StatusCode::UNAUTHORIZED => BackendProbe::Ready { port },
        reqwest::StatusCode::NOT_FOUND => {
            let reason = "desktop_login_not_found".to_string();
            if same_root_external {
                BackendProbe::ExternalConflict { port, reason }
            } else {
                BackendProbe::Old { port, reason }
            }
        }
        _ => {
            let reason = backend_capability_stale_reason(health)
                .unwrap_or_else(|| "desktop_login_probe_failed".to_string());
            if same_root_external {
                BackendProbe::ExternalConflict { port, reason }
            } else {
                BackendProbe::Old { port, reason }
            }
        }
    }
}

async fn probe_existing_backends(ignored_ports: &[u16]) -> BackendProbe {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
    {
        Ok(client) => client,
        Err(_) => return BackendProbe::Missing,
    };

    // Fan out health probes concurrently. The desktop-auth probe is still
    // sequential per candidate because it has auth-log side effects.
    let ports: Vec<u16> = (8888u16..=8908).collect();
    let mut health_futs = Vec::with_capacity(ports.len());
    for port in ports {
        // why: reqwest::Client is internally Arc-wrapped; clone is a refcount bump
        // (documented cheap). tokio::spawn needs 'static, so each task owns its own clone.
        let c = client.clone();
        health_futs.push(tokio::spawn(async move {
            backend_health(&c, port).await.map(|h| (port, h))
        }));
    }

    let mut candidates: Vec<(u16, BackendHealth)> = Vec::new();
    for fut in health_futs {
        if let Ok(Some(pair)) = fut.await {
            candidates.push(pair);
        }
    }

    let expected_studio_root_id = crate::desktop_backend_owner::read_expected_studio_root_id();
    let mut first_conflict = None;
    let mut first_ready = None;
    let mut first_old = None;
    for (port, health) in candidates {
        if ignored_ports.contains(&port) {
            continue;
        }
        match backend_desktop_auth_status(
            &client,
            port,
            &health,
            expected_studio_root_id.as_deref(),
        )
        .await
        {
            conflict @ BackendProbe::ExternalConflict { .. } if first_conflict.is_none() => {
                first_conflict = Some(conflict)
            }
            ready @ BackendProbe::Ready { .. } if first_ready.is_none() => {
                first_ready = Some(ready)
            }
            old @ BackendProbe::Old { .. } if first_old.is_none() => first_old = Some(old),
            _ => {}
        }
    }

    first_conflict
        .or(first_ready)
        .or(first_old)
        .unwrap_or(BackendProbe::Missing)
}

fn mutation_blocker_from_probe(probe: BackendProbe) -> Option<ExternalBackendConflict> {
    match probe {
        BackendProbe::ExternalConflict { port, reason } => {
            Some(ExternalBackendConflict { port, reason })
        }
        BackendProbe::Ready { port } => Some(ExternalBackendConflict {
            port,
            reason: "same_root_external_backend_active".to_string(),
        }),
        _ => None,
    }
}

pub async fn mutation_blocking_backend_ignoring(
    ignored_ports: &[u16],
) -> Option<ExternalBackendConflict> {
    mutation_blocker_from_probe(probe_existing_backends(ignored_ports).await)
}

pub async fn desktop_preflight_result() -> DesktopPreflightResult {
    let _ = crate::desktop_backend_owner::cleanup_verified_desktop_orphan().await;
    let (managed, backend) = tokio::join!(probe_managed_install(), probe_existing_backends(&[]));
    choose_preflight(managed, backend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn compatible_backend_wins_over_stale_managed_install() {
        let result = choose_preflight(
            ManagedProbe::Stale {
                bin: PathBuf::from("/managed/unsloth"),
                reason: "old cli".to_string(),
            },
            BackendProbe::Ready { port: 8000 },
        );

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::AttachedReady
        );
        assert_eq!(result.port, Some(8000));
        assert_eq!(result.reason, None);
        assert!(!result.can_auto_repair);
        assert_eq!(result.managed_bin, Some(PathBuf::from("/managed/unsloth")));
    }

    #[test]
    fn compatible_backend_wins_over_ready_managed_install() {
        let result = choose_preflight(
            ManagedProbe::Ready {
                bin: PathBuf::from("/managed/unsloth"),
            },
            BackendProbe::Ready { port: 8000 },
        );

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::AttachedReady
        );
        assert_eq!(result.port, Some(8000));
        assert_eq!(result.managed_bin, Some(PathBuf::from("/managed/unsloth")));
        assert!(!result.can_auto_repair);
    }

    #[test]
    fn compatible_backend_wins_over_missing_managed_install() {
        let result = choose_preflight(ManagedProbe::Missing, BackendProbe::Ready { port: 8000 });

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::AttachedReady
        );
        assert_eq!(result.port, Some(8000));
        assert_eq!(result.managed_bin, None);
        assert!(!result.can_auto_repair);
    }

    #[test]
    fn old_backend_falls_back_to_ready_managed_install() {
        let result = choose_preflight(
            ManagedProbe::Ready {
                bin: PathBuf::from("/managed/unsloth"),
            },
            BackendProbe::Old {
                port: 8001,
                reason: "missing endpoint".to_string(),
            },
        );

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::ManagedReady
        );
        assert_eq!(result.reason, None);
        assert_eq!(result.port, None);
        assert_eq!(result.managed_bin, Some(PathBuf::from("/managed/unsloth")));
        assert!(!result.can_auto_repair);
    }

    #[test]
    fn old_backend_falls_back_to_stale_managed_install() {
        let result = choose_preflight(
            ManagedProbe::Stale {
                bin: PathBuf::from("/managed/unsloth"),
                reason: "old cli".to_string(),
            },
            BackendProbe::Old {
                port: 8001,
                reason: "missing endpoint".to_string(),
            },
        );

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::ManagedStale
        );
        assert_eq!(result.reason, Some("old cli".to_string()));
        assert_eq!(result.port, None);
        assert_eq!(result.can_auto_repair, release_auto_repair());
        assert_eq!(result.managed_bin, Some(PathBuf::from("/managed/unsloth")));
    }

    #[test]
    fn managed_ready_when_no_backend() {
        let result = choose_preflight(
            ManagedProbe::Ready {
                bin: PathBuf::from("/managed/unsloth"),
            },
            BackendProbe::Missing,
        );

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::ManagedReady
        );
        assert_eq!(result.managed_bin, Some(PathBuf::from("/managed/unsloth")));
        assert!(!result.can_auto_repair);
    }

    #[test]
    fn managed_stale_when_no_backend() {
        let result = choose_preflight(
            ManagedProbe::Stale {
                bin: PathBuf::from("/managed/unsloth"),
                reason: "old cli".to_string(),
            },
            BackendProbe::Missing,
        );

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::ManagedStale
        );
        assert_eq!(result.reason, Some("old cli".to_string()));
        assert_eq!(result.can_auto_repair, release_auto_repair());
    }

    #[test]
    fn not_installed_when_no_backend_no_managed_binary() {
        let result = choose_preflight(ManagedProbe::Missing, BackendProbe::Missing);

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::NotInstalled
        );
        assert_eq!(result.managed_bin, None);
        assert!(!result.can_auto_repair);
    }

    #[test]
    fn old_backend_with_no_managed_install_uses_install_flow() {
        let result = choose_preflight(
            ManagedProbe::Missing,
            BackendProbe::Old {
                port: 8002,
                reason: "old version".to_string(),
            },
        );

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::NotInstalled
        );
        assert_eq!(result.reason, None);
        assert_eq!(result.port, None);
        assert!(!result.can_auto_repair);
    }

    #[test]
    fn external_conflict_blocks_managed_flow() {
        let result = choose_preflight(
            ManagedProbe::Ready {
                bin: PathBuf::from("/managed/unsloth"),
            },
            BackendProbe::ExternalConflict {
                port: 8888,
                reason: "same_root_external_backend_active".to_string(),
            },
        );

        assert_eq!(
            result.disposition,
            DesktopPreflightDisposition::ExternalConflict
        );
        assert_eq!(result.port, Some(8888));
        assert_eq!(
            result.reason,
            Some("same_root_external_backend_active".to_string())
        );
        assert!(!result.can_auto_repair);
        assert_eq!(result.managed_bin, Some(PathBuf::from("/managed/unsloth")));
    }

    #[test]
    fn mutation_blocker_blocks_ready_external_backends() {
        assert_eq!(
            mutation_blocker_from_probe(BackendProbe::Ready { port: 8890 }),
            Some(ExternalBackendConflict {
                port: 8890,
                reason: "same_root_external_backend_active".to_string(),
            })
        );
    }

    #[test]
    fn backend_version_gate_accepts_minimum_and_newer_versions() {
        assert!(backend_version_compatible(Some(
            MIN_DESKTOP_BACKEND_VERSION
        )));
        assert!(backend_version_compatible(Some("2026.5.3")));
        assert!(backend_version_compatible(Some("2027.1.0")));
    }

    #[test]
    fn backend_version_gate_accepts_pep_style_suffixes() {
        assert!(backend_version_compatible(Some("2026.5.2.post1")));
        assert!(backend_version_compatible(Some("2026.5.2+local")));
        assert!(backend_version_compatible(Some("2026.5.2rc1")));
        assert!(backend_version_compatible(Some("2026.5.2.dev1")));
    }

    #[test]
    fn backend_version_gate_rejects_missing_invalid_and_older_versions() {
        assert_eq!(
            backend_version_stale_reason(None),
            Some("desktop_backend_version_missing".to_string())
        );
        assert_eq!(
            backend_version_stale_reason(Some("not-a-version")),
            Some("desktop_backend_version_invalid".to_string())
        );
        assert_eq!(
            backend_version_stale_reason(Some("2026.5.2.1")),
            Some("desktop_backend_version_invalid".to_string())
        );
        assert_eq!(
            backend_version_stale_reason(Some("2026.5.2foo")),
            Some("desktop_backend_version_invalid".to_string())
        );
        assert_eq!(
            backend_version_stale_reason(Some("2026.5.1")),
            Some("desktop_backend_version_too_old".to_string())
        );
    }

    #[test]
    fn backend_version_gate_allows_dev_only_in_debug_builds() {
        assert_eq!(
            backend_version_compatible(Some("dev")),
            cfg!(debug_assertions)
        );
    }

    #[cfg(unix)]
    struct FakeCli {
        bin: PathBuf,
        dir: PathBuf,
    }

    #[cfg(unix)]
    impl Drop for FakeCli {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    #[cfg(unix)]
    fn fake_cli(test_name: &str, script: &str) -> FakeCli {
        use std::fs;
        use std::os::unix::fs::PermissionsExt;
        use std::time::{SystemTime, UNIX_EPOCH};

        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "unsloth-preflight-{test_name}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let bin = dir.join("unsloth");
        fs::write(&bin, script).unwrap();
        let mut perms = fs::metadata(&bin).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&bin, perms).unwrap();
        FakeCli { bin, dir }
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn managed_cli_stale_when_desktop_capabilities_missing() {
        let fake = fake_cli(
            "cap-missing",
            r#"#!/bin/sh
if [ "$1" = "-h" ]; then exit 0; fi
if [ "$1" = "studio" ] && [ "$2" = "provision-desktop-auth" ] && [ "$3" = "--help" ]; then exit 0; fi
exit 1
"#,
        );
        let bin = fake.bin.clone();

        assert!(matches!(
            probe_managed_bin(bin.clone()).await,
            ManagedProbe::Stale { bin: actual_bin, .. } if actual_bin == bin
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn managed_cli_stale_when_desktop_capabilities_command_missing() {
        let fake = fake_cli(
            "missing-helper",
            r#"#!/bin/sh
if [ "$1" = "-h" ]; then exit 0; fi
exit 1
"#,
        );
        let bin = fake.bin.clone();

        assert!(matches!(
            probe_managed_bin(bin.clone()).await,
            ManagedProbe::Stale { bin: actual_bin, .. } if actual_bin == bin
        ));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn managed_cli_stale_when_help_broken() {
        let fake = fake_cli(
            "broken-help",
            r#"#!/bin/sh
exit 1
"#,
        );
        let bin = fake.bin.clone();

        assert_eq!(
            probe_managed_bin(bin.clone()).await,
            ManagedProbe::Stale {
                bin,
                reason: "cli_unusable".to_string()
            }
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn managed_cli_ready_when_desktop_capabilities_compatible() {
        let fake = fake_cli(
            "cap-true-helper-missing",
            r#"#!/bin/sh
if [ "$1" = "-h" ]; then exit 0; fi
if [ "$1" = "studio" ] && [ "$2" = "desktop-capabilities" ] && [ "$3" = "--json" ]; then
  printf '{"desktop_protocol_version":1,"desktop_manageability_version":1,"supports_api_only":true,"supports_provision_desktop_auth":true,"supports_desktop_backend_ownership":true,"version":"2026.5.2"}'
  exit 0
fi
exit 1
"#,
        );
        let bin = fake.bin.clone();

        assert_eq!(
            probe_managed_bin(bin.clone()).await,
            ManagedProbe::Ready { bin }
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn capability_false_reason_used_when_legacy_helper_missing() {
        let fake = fake_cli(
            "cap-false-helper-missing",
            r#"#!/bin/sh
if [ "$1" = "-h" ]; then exit 0; fi
if [ "$1" = "studio" ] && [ "$2" = "desktop-capabilities" ] && [ "$3" = "--json" ]; then
  printf '{"desktop_protocol_version":1,"desktop_manageability_version":1,"supports_api_only":true,"supports_provision_desktop_auth":false,"supports_desktop_backend_ownership":true,"desktop_auth_stale_reason":"cap_false","version":"2026.5.2"}'
  exit 0
fi
exit 1
"#,
        );
        let bin = fake.bin.clone();

        assert_eq!(
            probe_managed_bin(bin.clone()).await,
            ManagedProbe::Stale {
                bin,
                reason: "cap_false".to_string()
            }
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn capability_false_overrides_working_legacy_helper() {
        let fake = fake_cli(
            "cap-false-helper-ready",
            r#"#!/bin/sh
if [ "$1" = "-h" ]; then exit 0; fi
if [ "$1" = "studio" ] && [ "$2" = "desktop-capabilities" ] && [ "$3" = "--json" ]; then
  printf '{"desktop_protocol_version":1,"desktop_manageability_version":1,"supports_api_only":true,"supports_provision_desktop_auth":false,"supports_desktop_backend_ownership":true,"desktop_auth_stale_reason":"cap_false","version":"2026.5.2"}'
  exit 0
fi
if [ "$1" = "studio" ] && [ "$2" = "provision-desktop-auth" ] && [ "$3" = "--help" ]; then exit 0; fi
exit 1
"#,
        );
        let bin = fake.bin.clone();

        assert_eq!(
            probe_managed_bin(bin.clone()).await,
            ManagedProbe::Stale {
                bin,
                reason: "cap_false".to_string()
            }
        );
    }

    const EXPECTED_ROOT_ID: &str =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const OTHER_ROOT_ID: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const OWNER_TOKEN: &str = "desktop-owner-token";

    fn install_test_owner() {
        crate::desktop_backend_owner::install_test_owner(EXPECTED_ROOT_ID, OWNER_TOKEN);
    }

    fn desktop_ready_health(root_id: &str) -> String {
        desktop_ready_health_with_owner(root_id, true)
    }

    fn desktop_owner_json(include_owner: bool) -> String {
        if include_owner {
            format!(
                r#", "desktop_owner":{{"kind":"tauri","token_sha256":"{}"}}"#,
                crate::desktop_backend_owner::token_sha256(OWNER_TOKEN)
            )
        } else {
            String::new()
        }
    }

    fn desktop_ready_health_with_owner(root_id: &str, include_owner: bool) -> String {
        let owner = desktop_owner_json(include_owner);
        format!(
            r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.5.2","desktop_protocol_version":1,"desktop_manageability_version":1,"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,"studio_root_id":"{root_id}"{owner}}}"#
        )
    }

    async fn backend_server(health_body: impl Into<String>, route_status: &'static str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let health_body = health_body.into();

        tokio::spawn(async move {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut buffer = [0; 2048];
                let n = stream.read(&mut buffer).await.unwrap();
                let request = String::from_utf8_lossy(&buffer[..n]);
                let (status, body) = if request.starts_with("GET /api/health ") {
                    ("200 OK", health_body.as_str())
                } else if request.starts_with("POST /api/auth/desktop-login ") {
                    (route_status, "")
                } else {
                    ("404 Not Found", "")
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(response.as_bytes()).await.unwrap();
            }
        });

        port
    }

    async fn probe_test_backend(
        health_body: impl Into<String>,
        route_status: &'static str,
    ) -> BackendProbe {
        install_test_owner();
        let port = backend_server(health_body, route_status).await;
        let client = reqwest::Client::new();
        let health = backend_health(&client, port).await.unwrap();
        backend_desktop_auth_status(&client, port, &health, Some(EXPECTED_ROOT_ID)).await
    }

    #[tokio::test]
    async fn backend_health_without_desktop_capability_fields_is_still_candidate() {
        let port = backend_server(
            r#"{"status":"healthy","service":"Unsloth UI Backend"}"#,
            "401 Unauthorized",
        )
        .await;
        let client = reqwest::Client::new();

        assert!(backend_health(&client, port).await.is_some());
    }

    #[tokio::test]
    async fn backend_with_auth_support_but_missing_protocol_is_old() {
        let probe = probe_test_backend(
            format!(
                r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.5.2","desktop_manageability_version":1,"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,"studio_root_id":"{EXPECTED_ROOT_ID}"{}}}"#,
                desktop_owner_json(true)
            ),
            "401 Unauthorized",
        )
        .await;

        assert!(matches!(probe, BackendProbe::Old { .. }));
    }

    #[tokio::test]
    async fn backend_with_auth_support_but_unsupported_protocol_is_old() {
        let probe = probe_test_backend(
            format!(
                r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.5.2","desktop_protocol_version":2,"desktop_manageability_version":1,"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,"studio_root_id":"{EXPECTED_ROOT_ID}"{}}}"#,
                desktop_owner_json(true)
            ),
            "401 Unauthorized",
        )
        .await;

        assert!(matches!(probe, BackendProbe::Old { .. }));
    }

    #[tokio::test]
    async fn backend_health_with_desktop_capability_fields_and_401_is_ready() {
        let probe =
            probe_test_backend(desktop_ready_health(EXPECTED_ROOT_ID), "401 Unauthorized").await;

        assert!(matches!(probe, BackendProbe::Ready { .. }));
    }

    #[tokio::test]
    async fn compatible_same_root_without_desktop_owner_is_ready() {
        let probe = probe_test_backend(
            desktop_ready_health_with_owner(EXPECTED_ROOT_ID, false),
            "401 Unauthorized",
        )
        .await;

        assert!(matches!(probe, BackendProbe::Ready { .. }));
    }

    #[tokio::test]
    async fn stale_same_root_without_desktop_owner_is_external_conflict() {
        let probe = probe_test_backend(
            format!(
                r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.5.1","desktop_protocol_version":1,"desktop_manageability_version":1,"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,"studio_root_id":"{EXPECTED_ROOT_ID}"}}"#,
            ),
            "401 Unauthorized",
        )
        .await;

        assert!(matches!(
            probe,
            BackendProbe::ExternalConflict {
                reason,
                ..
            } if reason == "desktop_backend_version_too_old"
        ));
    }

    #[tokio::test]
    async fn backend_root_id_mismatch_is_old_before_auth_probe() {
        let probe =
            probe_test_backend(desktop_ready_health(OTHER_ROOT_ID), "401 Unauthorized").await;

        assert!(matches!(
            probe,
            BackendProbe::Old {
                reason,
                ..
            } if reason == "studio_root_id_mismatch"
        ));
    }

    #[tokio::test]
    async fn backend_missing_root_id_is_external_conflict_before_auth_probe() {
        let probe = probe_test_backend(
            r#"{"status":"healthy","service":"Unsloth UI Backend","desktop_protocol_version":1,"supports_desktop_auth":true}"#,
            "401 Unauthorized",
        )
        .await;

        assert!(matches!(
            probe,
            BackendProbe::ExternalConflict {
                reason,
                ..
            } if reason == "ambiguous_root_external_backend_active"
        ));
    }

    #[tokio::test]
    async fn backend_expected_root_id_missing_is_external_conflict_before_auth_probe() {
        install_test_owner();
        let port = backend_server(desktop_ready_health(EXPECTED_ROOT_ID), "401 Unauthorized").await;
        let client = reqwest::Client::new();
        let health = backend_health(&client, port).await.unwrap();

        assert!(matches!(
            backend_desktop_auth_status(&client, port, &health, None).await,
            BackendProbe::ExternalConflict {
                reason,
                ..
            } if reason == "ambiguous_root_external_backend_active"
        ));
    }

    #[tokio::test]
    async fn backend_route_404_is_old() {
        let probe =
            probe_test_backend(desktop_ready_health(EXPECTED_ROOT_ID), "404 Not Found").await;

        assert!(matches!(
            probe,
            BackendProbe::Old {
                reason,
                ..
            } if reason == "desktop_login_not_found"
        ));
    }

    #[tokio::test]
    async fn backend_route_500_is_old() {
        let probe = probe_test_backend(
            desktop_ready_health(EXPECTED_ROOT_ID),
            "500 Internal Server Error",
        )
        .await;

        assert!(matches!(probe, BackendProbe::Old { .. }));
    }

    #[tokio::test]
    async fn backend_capability_false_is_old_even_when_route_401() {
        install_test_owner();
        let port = backend_server(
            format!(
                r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.5.2","desktop_protocol_version":1,"desktop_manageability_version":1,"supports_desktop_auth":false,"supports_desktop_backend_ownership":true,"desktop_auth_stale_reason":"cap_false","studio_root_id":"{EXPECTED_ROOT_ID}"{}}}"#,
                desktop_owner_json(true)
            ),
            "401 Unauthorized",
        )
        .await;
        let client = reqwest::Client::new();
        let health = backend_health(&client, port).await.unwrap();

        assert!(matches!(
            backend_desktop_auth_status(&client, port, &health, Some(EXPECTED_ROOT_ID)).await,
            BackendProbe::Old {
                reason,
                ..
            } if reason == "cap_false"
        ));
    }
}
