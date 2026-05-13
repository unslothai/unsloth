use log::warn;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(test)]
static TEST_EXPECTED_STUDIO_ROOT_ID: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
#[cfg(test)]
static TEST_METADATA: std::sync::Mutex<Option<DesktopBackendMetadata>> =
    std::sync::Mutex::new(None);

pub(crate) const OWNER_TOKEN_ENV: &str = "UNSLOTH_STUDIO_DESKTOP_OWNER_TOKEN";
pub(crate) const OWNER_KIND_ENV: &str = "UNSLOTH_STUDIO_DESKTOP_OWNER_KIND";
pub(crate) const OWNER_KIND_TAURI: &str = "tauri";

const METADATA_SCHEMA_VERSION: u8 = 1;
const STUDIO_INSTALL_ID_HEX_LEN: usize = 64;
const OWNER_TOKEN_BYTES: usize = 32;
const DESKTOP_PORT_START: u16 = 8888;
const DESKTOP_PORT_END: u16 = 8908;
const LOCAL_HTTP_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Clone, Debug)]
pub(crate) struct PendingBackendOwner {
    pub token: String,
    pub studio_root_id: String,
}

#[derive(Clone, Debug)]
pub(crate) struct BackendOwnerState {
    path: PathBuf,
    metadata: DesktopBackendMetadata,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DesktopBackendMetadata {
    schema_version: u8,
    kind: String,
    token: String,
    token_sha256: String,
    app_pid: u32,
    backend_pid: u32,
    generation: u64,
    requested_port: u16,
    port: Option<u16>,
    studio_root_id: String,
    started_at_ms: u64,
    updated_at_ms: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HealthOwnerMatch {
    None,
    CurrentApp,
    PreviousApp,
    OtherDesktopOwner,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum OwnedBackendReadiness {
    Ready,
    Stale { reason: String },
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) struct VerifiedOwnedBackend {
    pub owner: BackendOwnerState,
    pub port: u16,
    pub backend_pid: u32,
    pub generation: u64,
    pub readiness: OwnedBackendReadiness,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(crate) enum OwnedBackendProbe {
    NoMetadata,
    RemovedMalformed,
    NotVerified { reason: String },
    Unmanageable { port: u16, reason: String },
    Verified(VerifiedOwnedBackend),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreviousAppPidStatus {
    Dead,
    AliveOrCurrent,
    Uncertain,
}

#[derive(Debug, Deserialize)]
struct HealthDesktopOwner {
    kind: Option<String>,
    token_sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HealthResponse {
    status: Option<String>,
    service: Option<String>,
    version: Option<String>,
    desktop_protocol_version: Option<u16>,
    desktop_manageability_version: Option<u16>,
    supports_desktop_auth: Option<bool>,
    supports_desktop_backend_ownership: Option<bool>,
    studio_root_id: Option<String>,
    desktop_owner: Option<HealthDesktopOwner>,
}

#[derive(Debug)]
struct SimpleHttpResponse {
    status: u16,
    body: Vec<u8>,
}

#[derive(Serialize)]
struct DesktopLoginPayload<'a> {
    secret: &'a str,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
}

pub(crate) fn desktop_candidate_ports() -> std::ops::RangeInclusive<u16> {
    DESKTOP_PORT_START..=DESKTOP_PORT_END
}

pub(crate) fn is_valid_studio_root_id(value: &str) -> bool {
    value.len() == STUDIO_INSTALL_ID_HEX_LEN
        && value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

pub(crate) fn parse_studio_root_id(value: &str) -> Option<String> {
    let value = value.trim();
    is_valid_studio_root_id(value).then(|| value.to_string())
}

pub(crate) fn managed_studio_root_id_path(home: &Path) -> PathBuf {
    home.join(".unsloth")
        .join("studio")
        .join("share")
        .join("studio_install_id")
}

fn managed_run_dir(home: &Path) -> PathBuf {
    home.join(".unsloth").join("studio").join("run")
}

fn metadata_path_for_home(home: &Path) -> PathBuf {
    managed_run_dir(home).join("desktop_backend.json")
}

fn auth_secret_path_for_home(home: &Path) -> PathBuf {
    home.join(".unsloth")
        .join("studio")
        .join("auth")
        .join(".desktop_secret")
}

pub(crate) fn read_expected_studio_root_id() -> Option<String> {
    #[cfg(test)]
    if let Ok(guard) = TEST_EXPECTED_STUDIO_ROOT_ID.lock() {
        if let Some(value) = guard.clone() {
            return Some(value);
        }
    }

    let home = dirs::home_dir()?;
    let raw = std::fs::read_to_string(managed_studio_root_id_path(&home)).ok()?;
    parse_studio_root_id(&raw)
}

fn metadata_path() -> Option<PathBuf> {
    dirs::home_dir().map(|home| metadata_path_for_home(&home))
}

pub(crate) fn token_sha256(token: &str) -> String {
    hex_bytes(&Sha256::digest(token.as_bytes()))
}

fn random_owner_token() -> String {
    hex_bytes(&rand::random::<[u8; OWNER_TOKEN_BYTES]>())
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis() as u64
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

pub(crate) fn new_pending_owner() -> Option<PendingBackendOwner> {
    Some(PendingBackendOwner {
        token: random_owner_token(),
        studio_root_id: read_expected_studio_root_id()?,
    })
}

pub(crate) fn activate_owner(
    pending: PendingBackendOwner,
    requested_port: u16,
    generation: u64,
    backend_pid: u32,
) -> Option<BackendOwnerState> {
    let path = metadata_path()?;
    let now = now_ms();
    let metadata = DesktopBackendMetadata {
        schema_version: METADATA_SCHEMA_VERSION,
        kind: OWNER_KIND_TAURI.to_string(),
        token_sha256: token_sha256(&pending.token),
        token: pending.token,
        app_pid: std::process::id(),
        backend_pid,
        generation,
        requested_port,
        port: None,
        studio_root_id: pending.studio_root_id,
        started_at_ms: now,
        updated_at_ms: now,
    };
    let state = BackendOwnerState { path, metadata };
    if let Err(error) = state.write() {
        warn!("Desktop backend owner metadata unavailable: {}", error);
        return None;
    }
    Some(state)
}

#[allow(dead_code)]
impl BackendOwnerState {
    fn from_metadata(path: PathBuf, metadata: DesktopBackendMetadata) -> Self {
        Self { path, metadata }
    }

    fn write(&self) -> Result<(), String> {
        write_metadata(&self.path, &self.metadata)
    }

    pub(crate) fn update_port(&mut self, port: u16) -> Result<(), String> {
        self.metadata.port = Some(port);
        self.metadata.updated_at_ms = now_ms();
        self.write()
    }

    pub(crate) fn remove(self) {
        remove_metadata_file(&self.path);
    }

    pub(crate) fn port(&self) -> Option<u16> {
        self.metadata.port
    }

    pub(crate) fn backend_pid(&self) -> u32 {
        self.metadata.backend_pid
    }

    pub(crate) fn generation(&self) -> u64 {
        self.metadata.generation
    }

    pub(crate) fn requested_port(&self) -> u16 {
        self.metadata.requested_port
    }

    pub(crate) fn token_sha256(&self) -> &str {
        &self.metadata.token_sha256
    }

    pub(crate) fn studio_root_id(&self) -> &str {
        &self.metadata.studio_root_id
    }

    pub(crate) fn verifies_exact_port_blocking(&self, port: u16) -> bool {
        match fetch_health_blocking(port) {
            Ok(Some(health)) => {
                health_verifies_metadata(&health, &self.metadata)
                    && lifecycle_control_block_reason(&health).is_none()
            }
            _ => false,
        }
    }
}

fn write_metadata(path: &Path, metadata: &DesktopBackendMetadata) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "desktop owner metadata path has no parent".to_string())?;
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    set_private_dir_permissions(parent);

    let tmp = parent.join(format!(".desktop_backend.{}.tmp", std::process::id()));
    let body = serde_json::to_vec_pretty(metadata).map_err(|e| e.to_string())?;
    write_private_file(&tmp, &body)?;
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())?;
    set_private_file_permissions(path);
    Ok(())
}

fn write_private_file(path: &Path, body: &[u8]) -> Result<(), String> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .mode(0o600)
            .open(path)
            .map_err(|e| e.to_string())?;
        file.write_all(body).map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        return Ok(());
    }

    #[cfg(not(unix))]
    {
        std::fs::write(path, body).map_err(|e| e.to_string())
    }
}

fn set_private_dir_permissions(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700));
    }
}

fn set_private_file_permissions(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
}

fn remove_metadata_file(path: &Path) {
    if let Err(error) = std::fs::remove_file(path) {
        if error.kind() != std::io::ErrorKind::NotFound {
            warn!(
                "Could not remove desktop backend owner metadata at {}: {}",
                path.display(),
                error
            );
        }
    }
}

fn read_metadata(path: &Path) -> Result<Option<DesktopBackendMetadata>, String> {
    let raw = match std::fs::read(path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.to_string()),
    };
    serde_json::from_slice::<DesktopBackendMetadata>(&raw)
        .map(Some)
        .map_err(|e| e.to_string())
}

fn metadata_is_well_formed(metadata: &DesktopBackendMetadata) -> bool {
    metadata.schema_version == METADATA_SCHEMA_VERSION
        && metadata.kind == OWNER_KIND_TAURI
        && is_valid_studio_root_id(&metadata.studio_root_id)
        && metadata.token_sha256 == token_sha256(&metadata.token)
        && metadata.backend_pid > 0
}

fn owner_matches_metadata(
    metadata: &DesktopBackendMetadata,
    studio_root_id: Option<&str>,
    owner_kind: Option<&str>,
    token_sha256_value: Option<&str>,
) -> bool {
    studio_root_id == Some(metadata.studio_root_id.as_str())
        && owner_kind == Some(OWNER_KIND_TAURI)
        && token_sha256_value == Some(metadata.token_sha256.as_str())
}

fn classify_health_desktop_owner_for_metadata(
    metadata: Option<&DesktopBackendMetadata>,
    expected_studio_root_id: Option<&str>,
    studio_root_id: Option<&str>,
    owner_kind: Option<&str>,
    token_sha256_value: Option<&str>,
) -> HealthOwnerMatch {
    if owner_kind.is_none() && token_sha256_value.is_none() {
        return HealthOwnerMatch::None;
    }

    let Some(metadata) = metadata else {
        return HealthOwnerMatch::OtherDesktopOwner;
    };

    if metadata_is_well_formed(metadata)
        && expected_studio_root_id == Some(metadata.studio_root_id.as_str())
        && owner_matches_metadata(metadata, studio_root_id, owner_kind, token_sha256_value)
    {
        if metadata.app_pid == std::process::id() {
            HealthOwnerMatch::CurrentApp
        } else {
            HealthOwnerMatch::PreviousApp
        }
    } else {
        HealthOwnerMatch::OtherDesktopOwner
    }
}

pub(crate) fn classify_health_desktop_owner(
    studio_root_id: Option<&str>,
    owner_kind: Option<&str>,
    token_sha256_value: Option<&str>,
) -> HealthOwnerMatch {
    let metadata = current_owner_metadata();
    let expected = read_expected_studio_root_id();
    classify_health_desktop_owner_for_metadata(
        metadata.as_ref(),
        expected.as_deref(),
        studio_root_id,
        owner_kind,
        token_sha256_value,
    )
}

fn current_owner_metadata() -> Option<DesktopBackendMetadata> {
    #[cfg(test)]
    if let Ok(guard) = TEST_METADATA.lock() {
        if let Some(metadata) = guard.clone() {
            return Some(metadata);
        }
    }

    let path = metadata_path()?;
    read_metadata(&path).ok().flatten()
}

#[cfg(test)]
pub(crate) fn install_test_owner(root_id: &str, token: &str) {
    let metadata = DesktopBackendMetadata {
        schema_version: METADATA_SCHEMA_VERSION,
        kind: OWNER_KIND_TAURI.to_string(),
        token: token.to_string(),
        token_sha256: token_sha256(token),
        app_pid: std::process::id(),
        backend_pid: 2,
        generation: 3,
        requested_port: 8888,
        port: Some(8888),
        studio_root_id: root_id.to_string(),
        started_at_ms: 1,
        updated_at_ms: 1,
    };
    *TEST_EXPECTED_STUDIO_ROOT_ID.lock().unwrap() = Some(root_id.to_string());
    *TEST_METADATA.lock().unwrap() = Some(metadata);
}

#[cfg(test)]
pub(crate) fn test_owner_state(root_id: &str, token: &str, port: u16) -> BackendOwnerState {
    let metadata = DesktopBackendMetadata {
        schema_version: METADATA_SCHEMA_VERSION,
        kind: OWNER_KIND_TAURI.to_string(),
        token: token.to_string(),
        token_sha256: token_sha256(token),
        app_pid: std::process::id(),
        backend_pid: 2,
        generation: 3,
        requested_port: port,
        port: Some(port),
        studio_root_id: root_id.to_string(),
        started_at_ms: 1,
        updated_at_ms: 1,
    };
    BackendOwnerState {
        path: std::env::temp_dir().join(format!(
            "unsloth-test-owner-state-{}-{}.json",
            std::process::id(),
            now_ms()
        )),
        metadata,
    }
}

fn health_verifies_metadata(health: &HealthResponse, metadata: &DesktopBackendMetadata) -> bool {
    let healthy = health.status.as_deref() == Some("healthy")
        && health.service.as_deref() == Some("Unsloth UI Backend");
    let Some(owner) = health.desktop_owner.as_ref() else {
        return false;
    };
    healthy
        && owner_matches_metadata(
            metadata,
            health.studio_root_id.as_deref(),
            owner.kind.as_deref(),
            owner.token_sha256.as_deref(),
        )
}

fn lifecycle_control_block_reason(health: &HealthResponse) -> Option<String> {
    if health.desktop_protocol_version != Some(crate::preflight::DESKTOP_PROTOCOL_VERSION) {
        return Some("desktop_protocol_incompatible".to_string());
    }
    if health.supports_desktop_auth != Some(true) {
        return Some("desktop_auth_unsupported".to_string());
    }
    if health.desktop_manageability_version.unwrap_or(0)
        < crate::preflight::DESKTOP_MANAGEABILITY_VERSION
    {
        return Some("desktop_manageability_unsupported".to_string());
    }
    if health.supports_desktop_backend_ownership != Some(true) {
        return Some("desktop_backend_ownership_unsupported".to_string());
    }
    None
}

fn ready_for_use_status(health: &HealthResponse) -> OwnedBackendReadiness {
    match crate::preflight::backend_version_stale_reason(health.version.as_deref()) {
        Some(reason) => OwnedBackendReadiness::Stale { reason },
        None => OwnedBackendReadiness::Ready,
    }
}

async fn fetch_health(port: u16) -> Result<Option<HealthResponse>, reqwest::Error> {
    let client = reqwest::Client::builder()
        .timeout(LOCAL_HTTP_TIMEOUT)
        .build()?;
    let response = client
        .get(format!("http://127.0.0.1:{port}/api/health"))
        .send()
        .await?;
    if !response.status().is_success() {
        return Ok(None);
    }
    response.json::<HealthResponse>().await.map(Some)
}

fn fetch_health_blocking(port: u16) -> Result<Option<HealthResponse>, String> {
    let response = http_request_blocking(port, "GET", "/api/health", &[], &[])?;
    if !(200..300).contains(&response.status) {
        return Ok(None);
    }
    serde_json::from_slice::<HealthResponse>(&response.body)
        .map(Some)
        .map_err(|e| e.to_string())
}

async fn desktop_login_route_compatible(port: u16) -> bool {
    let client = match reqwest::Client::builder()
        .timeout(LOCAL_HTTP_TIMEOUT)
        .build()
    {
        Ok(client) => client,
        Err(_) => return false,
    };
    match client
        .post(format!("http://127.0.0.1:{port}/api/auth/desktop-login"))
        .json(&DesktopLoginPayload {
            secret: "desktop-owner-adoption-invalid-secret",
        })
        .send()
        .await
    {
        Ok(response) => response.status() == reqwest::StatusCode::UNAUTHORIZED,
        Err(_) => false,
    }
}

async fn desktop_secret_login_compatible(port: u16) -> Result<(), String> {
    let secret = read_desktop_secret()?.ok_or_else(|| "desktop_auth_secret_missing".to_string())?;
    let client = reqwest::Client::builder()
        .timeout(LOCAL_HTTP_TIMEOUT)
        .build()
        .map_err(|e| e.to_string())?;
    let response = client
        .post(format!("http://127.0.0.1:{port}/api/auth/desktop-login"))
        .json(&DesktopLoginPayload { secret: &secret })
        .send()
        .await
        .map_err(|_| "desktop_auth_secret_probe_failed".to_string())?;
    if response.status().is_success() {
        Ok(())
    } else if response.status() == reqwest::StatusCode::UNAUTHORIZED {
        Err("desktop_auth_secret_rejected".to_string())
    } else {
        Err(format!(
            "desktop_auth_secret_probe_http_{}",
            response.status()
        ))
    }
}

pub(crate) async fn probe_owned_backend_state(
    owner: BackendOwnerState,
    port: Option<u16>,
    require_desktop_secret: bool,
) -> OwnedBackendProbe {
    let ports: Vec<u16> = match port {
        Some(port) => vec![port],
        None => desktop_candidate_ports().collect(),
    };
    let mut verified = Vec::new();
    for port in ports {
        let health = match fetch_health(port).await {
            Ok(Some(health)) => health,
            Ok(None) => continue,
            Err(error) => {
                warn!(
                    "Desktop-owned backend probe skipped port {} after health error: {}",
                    port, error
                );
                continue;
            }
        };
        if !health_verifies_metadata(&health, &owner.metadata) {
            continue;
        }
        if let Some(reason) = lifecycle_control_block_reason(&health) {
            return OwnedBackendProbe::Unmanageable { port, reason };
        }
        if !desktop_login_route_compatible(port).await {
            return OwnedBackendProbe::Unmanageable {
                port,
                reason: "desktop_login_probe_failed".to_string(),
            };
        }
        if require_desktop_secret {
            if let Err(reason) = desktop_secret_login_compatible(port).await {
                return OwnedBackendProbe::Unmanageable { port, reason };
            }
        }
        verified.push((port, ready_for_use_status(&health)));
    }

    if verified.len() != 1 {
        return OwnedBackendProbe::NotVerified {
            reason: if verified.is_empty() {
                "owned_backend_not_found".to_string()
            } else {
                "owned_backend_ambiguous".to_string()
            },
        };
    }

    let (port, readiness) = verified.remove(0);
    OwnedBackendProbe::Verified(VerifiedOwnedBackend {
        backend_pid: owner.backend_pid(),
        generation: owner.generation(),
        owner,
        port,
        readiness,
    })
}

#[allow(dead_code)]
pub(crate) async fn probe_verified_owned_backend() -> Result<OwnedBackendProbe, String> {
    let Some(path) = metadata_path() else {
        return Ok(OwnedBackendProbe::NoMetadata);
    };
    probe_verified_owned_backend_at_path(&path).await
}

async fn probe_verified_owned_backend_at_path(path: &Path) -> Result<OwnedBackendProbe, String> {
    let expected = read_expected_studio_root_id();
    probe_verified_owned_backend_at_path_with_expected(path, expected.as_deref()).await
}

async fn probe_verified_owned_backend_at_path_with_expected(
    path: &Path,
    expected_studio_root_id: Option<&str>,
) -> Result<OwnedBackendProbe, String> {
    let Some(metadata) = read_metadata(path).map_err(|e| format!("read owner metadata: {e}"))?
    else {
        return Ok(OwnedBackendProbe::NoMetadata);
    };

    if !metadata_is_well_formed(&metadata) {
        remove_metadata_file(path);
        return Ok(OwnedBackendProbe::RemovedMalformed);
    }

    match previous_app_pid_status(metadata.app_pid) {
        PreviousAppPidStatus::AliveOrCurrent => {
            return Ok(OwnedBackendProbe::NotVerified {
                reason: "previous_desktop_app_still_running".to_string(),
            });
        }
        PreviousAppPidStatus::Uncertain => {
            return Ok(OwnedBackendProbe::NotVerified {
                reason: "previous_desktop_app_liveness_uncertain".to_string(),
            });
        }
        PreviousAppPidStatus::Dead => {}
    }

    if expected_studio_root_id != Some(metadata.studio_root_id.as_str()) {
        return Ok(OwnedBackendProbe::NotVerified {
            reason: "studio_root_id_mismatch".to_string(),
        });
    }

    let owner = BackendOwnerState::from_metadata(path.to_path_buf(), metadata);
    let port = owner.port();
    Ok(probe_owned_backend_state(owner, port, true).await)
}

fn previous_app_pid_status(pid: u32) -> PreviousAppPidStatus {
    if pid == 0 || pid == std::process::id() {
        return PreviousAppPidStatus::AliveOrCurrent;
    }
    process_liveness(pid)
}

#[cfg(unix)]
fn process_liveness(pid: u32) -> PreviousAppPidStatus {
    if pid > i32::MAX as u32 {
        return PreviousAppPidStatus::Uncertain;
    }
    let rc = unsafe { libc::kill(pid as i32, 0) };
    if rc == 0 {
        return PreviousAppPidStatus::AliveOrCurrent;
    }
    match std::io::Error::last_os_error().raw_os_error() {
        Some(libc::ESRCH) => PreviousAppPidStatus::Dead,
        Some(libc::EPERM) => PreviousAppPidStatus::AliveOrCurrent,
        _ => PreviousAppPidStatus::Uncertain,
    }
}

#[cfg(windows)]
fn process_liveness(pid: u32) -> PreviousAppPidStatus {
    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, ERROR_INVALID_PARAMETER};
    use windows_sys::Win32::System::Threading::{
        OpenProcess, WaitForSingleObject, SYNCHRONIZE, WAIT_OBJECT_0, WAIT_TIMEOUT,
    };

    unsafe {
        let handle = OpenProcess(SYNCHRONIZE, 0, pid);
        if handle == 0 {
            return if GetLastError() == ERROR_INVALID_PARAMETER {
                PreviousAppPidStatus::Dead
            } else {
                PreviousAppPidStatus::Uncertain
            };
        }
        let wait = WaitForSingleObject(handle, 0);
        let _ = CloseHandle(handle);
        match wait {
            WAIT_TIMEOUT => PreviousAppPidStatus::AliveOrCurrent,
            WAIT_OBJECT_0 => PreviousAppPidStatus::Dead,
            _ => PreviousAppPidStatus::Uncertain,
        }
    }
}

#[cfg(not(any(unix, windows)))]
fn process_liveness(_pid: u32) -> PreviousAppPidStatus {
    PreviousAppPidStatus::Uncertain
}

pub(crate) fn exact_port_http_shutdown_blocking(port: u16) -> Result<(), String> {
    let secret =
        read_desktop_secret()?.ok_or_else(|| "desktop auth secret not found".to_string())?;
    let login_body =
        serde_json::to_vec(&DesktopLoginPayload { secret: &secret }).map_err(|e| e.to_string())?;
    let login = http_request_blocking(
        port,
        "POST",
        "/api/auth/desktop-login",
        &["Content-Type: application/json".to_string()],
        &login_body,
    )?;
    if login.status == 401 {
        return Err("desktop auth secret rejected".to_string());
    }
    if !(200..300).contains(&login.status) {
        return Err(format!("desktop login returned HTTP {}", login.status));
    }
    let tokens = serde_json::from_slice::<TokenResponse>(&login.body)
        .map_err(|e| format!("desktop login response invalid: {e}"))?;
    let shutdown = http_request_blocking(
        port,
        "POST",
        "/api/shutdown",
        &[format!("Authorization: Bearer {}", tokens.access_token)],
        &[],
    )?;
    if (200..300).contains(&shutdown.status) {
        Ok(())
    } else {
        Err(format!("shutdown returned HTTP {}", shutdown.status))
    }
}

pub(crate) fn port_is_listening_blocking(port: u16, timeout: Duration) -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    TcpStream::connect_timeout(&addr, timeout).is_ok()
}

fn read_desktop_secret() -> Result<Option<String>, String> {
    let Some(home) = dirs::home_dir() else {
        return Ok(None);
    };
    match std::fs::read_to_string(auth_secret_path_for_home(&home)) {
        Ok(secret) => Ok(Some(secret.trim().to_string())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.to_string()),
    }
}

fn http_request_blocking(
    port: u16,
    method: &str,
    path: &str,
    extra_headers: &[String],
    body: &[u8],
) -> Result<SimpleHttpResponse, String> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let mut stream = TcpStream::connect_timeout(&addr, LOCAL_HTTP_TIMEOUT)
        .map_err(|e| format!("connect to 127.0.0.1:{port}: {e}"))?;
    stream
        .set_read_timeout(Some(LOCAL_HTTP_TIMEOUT))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(LOCAL_HTTP_TIMEOUT))
        .map_err(|e| e.to_string())?;

    let mut request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    )
    .into_bytes();
    for header in extra_headers {
        request.extend_from_slice(header.as_bytes());
        request.extend_from_slice(b"\r\n");
    }
    request.extend_from_slice(b"\r\n");
    request.extend_from_slice(body);
    stream.write_all(&request).map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())?;

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).map_err(|e| e.to_string())?;
    parse_http_response(&raw)
}

fn parse_http_response(raw: &[u8]) -> Result<SimpleHttpResponse, String> {
    let Some(header_end) = raw.windows(4).position(|window| window == b"\r\n\r\n") else {
        return Err("HTTP response missing header terminator".to_string());
    };
    let headers = std::str::from_utf8(&raw[..header_end]).map_err(|e| e.to_string())?;
    let status = headers
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or_else(|| "HTTP response missing status".to_string())?;
    Ok(SimpleHttpResponse {
        status,
        body: raw[header_end + 4..].to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROOT_ID: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const TOKEN: &str = "desktop-owner-token";

    fn metadata(app_pid: u32, port: Option<u16>) -> DesktopBackendMetadata {
        DesktopBackendMetadata {
            schema_version: METADATA_SCHEMA_VERSION,
            kind: OWNER_KIND_TAURI.to_string(),
            token: TOKEN.to_string(),
            token_sha256: token_sha256(TOKEN),
            app_pid,
            backend_pid: 2,
            generation: 3,
            requested_port: 8888,
            port,
            studio_root_id: ROOT_ID.to_string(),
            started_at_ms: 1,
            updated_at_ms: 1,
        }
    }

    fn dead_child_pid() -> u32 {
        #[cfg(windows)]
        let mut child = std::process::Command::new("cmd")
            .args(["/C", "exit", "0"])
            .spawn()
            .unwrap();
        #[cfg(not(windows))]
        let mut child = std::process::Command::new("sh")
            .args(["-c", "exit 0"])
            .spawn()
            .unwrap();
        let pid = child.id();
        let _ = child.wait();
        pid
    }

    fn temp_metadata_path(test_name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "unsloth-owner-{test_name}-{}-{}",
            std::process::id(),
            now_ms()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("desktop_backend.json")
    }

    fn closed_port() -> u16 {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        port
    }

    #[test]
    fn parse_studio_root_id_requires_lowercase_64_hex_chars() {
        assert_eq!(parse_studio_root_id(ROOT_ID), Some(ROOT_ID.to_string()));
        assert_eq!(
            parse_studio_root_id(&format!("\n{ROOT_ID}\n")),
            Some(ROOT_ID.to_string())
        );
        assert_eq!(parse_studio_root_id(""), None);
        assert_eq!(
            parse_studio_root_id(
                "Aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            ),
            None
        );
        assert_eq!(parse_studio_root_id("not-a-root-id"), None);
    }

    #[test]
    fn metadata_well_formed_requires_matching_token_hash() {
        assert_eq!(
            token_sha256(TOKEN),
            "943501cb7d1feb2aa8cde1bf09b80092c25b95dbafaca9ccc12d6785b229a6fd"
        );
        let mut metadata = metadata(1, Some(8888));
        assert!(metadata_is_well_formed(&metadata));
        metadata.token_sha256 = token_sha256("different");
        assert!(!metadata_is_well_formed(&metadata));
    }

    #[test]
    fn health_verification_requires_root_kind_and_token_sha() {
        let metadata = metadata(1, Some(8888));
        let health = HealthResponse {
            status: Some("healthy".to_string()),
            service: Some("Unsloth UI Backend".to_string()),
            version: Some("2026.5.2".to_string()),
            desktop_protocol_version: Some(1),
            desktop_manageability_version: Some(1),
            supports_desktop_auth: Some(true),
            supports_desktop_backend_ownership: Some(true),
            studio_root_id: Some(ROOT_ID.to_string()),
            desktop_owner: Some(HealthDesktopOwner {
                kind: Some(OWNER_KIND_TAURI.to_string()),
                token_sha256: Some(token_sha256(TOKEN)),
            }),
        };
        assert!(health_verifies_metadata(&health, &metadata));

        let mut wrong_root = health;
        wrong_root.studio_root_id =
            Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string());
        assert!(!health_verifies_metadata(&wrong_root, &metadata));
    }

    #[tokio::test]
    async fn unreachable_recorded_port_is_not_verified_not_error() {
        let port = closed_port();
        let path = temp_metadata_path("closed-recorded-port");
        let metadata = metadata(dead_child_pid(), Some(port));
        write_metadata(&path, &metadata).unwrap();

        let probe = probe_verified_owned_backend_at_path_with_expected(&path, Some(ROOT_ID))
            .await
            .unwrap();

        assert!(matches!(
            probe,
            OwnedBackendProbe::NotVerified { reason } if reason == "owned_backend_not_found"
        ));
        assert!(path.exists());
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn current_app_pid_is_not_adoptable() {
        assert_eq!(
            previous_app_pid_status(std::process::id()),
            PreviousAppPidStatus::AliveOrCurrent
        );
    }
}
