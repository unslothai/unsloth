use log::{info, warn};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
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
pub(crate) enum CleanupOutcome {
    NoMetadata,
    NotVerified,
    RemovedMalformed,
    Killed,
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

impl BackendOwnerState {
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
        use std::io::Write;
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

pub(crate) fn health_matches_desktop_owner(
    studio_root_id: Option<&str>,
    owner_kind: Option<&str>,
    token_sha256_value: Option<&str>,
) -> bool {
    let Some(metadata) = current_owner_metadata() else {
        return false;
    };
    metadata_is_well_formed(&metadata)
        && read_expected_studio_root_id().as_deref() == Some(metadata.studio_root_id.as_str())
        && owner_matches_metadata(&metadata, studio_root_id, owner_kind, token_sha256_value)
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
        app_pid: 1,
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

#[derive(Debug, Deserialize)]
struct HealthDesktopOwner {
    kind: Option<String>,
    token_sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HealthResponse {
    status: Option<String>,
    service: Option<String>,
    studio_root_id: Option<String>,
    desktop_owner: Option<HealthDesktopOwner>,
}

async fn fetch_health(port: u16) -> Result<Option<HealthResponse>, reqwest::Error> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
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

pub(crate) async fn cleanup_verified_desktop_orphan() -> Result<CleanupOutcome, String> {
    let Some(path) = metadata_path() else {
        return Ok(CleanupOutcome::NoMetadata);
    };
    cleanup_verified_desktop_orphan_at_path(&path).await
}

async fn cleanup_verified_desktop_orphan_at_path(path: &Path) -> Result<CleanupOutcome, String> {
    let Some(metadata) = read_metadata(path).map_err(|e| format!("read owner metadata: {e}"))?
    else {
        return Ok(CleanupOutcome::NoMetadata);
    };

    if metadata.app_pid == std::process::id() {
        return Ok(CleanupOutcome::NotVerified);
    }

    if !metadata_is_well_formed(&metadata) {
        remove_metadata_file(path);
        return Ok(CleanupOutcome::RemovedMalformed);
    }

    if read_expected_studio_root_id().as_deref() != Some(metadata.studio_root_id.as_str()) {
        return Ok(CleanupOutcome::NotVerified);
    }

    let Some(port) = metadata.port else {
        return Ok(CleanupOutcome::NotVerified);
    };

    let health = match fetch_health(port).await {
        Ok(Some(health)) => health,
        Ok(None) | Err(_) => return Ok(CleanupOutcome::NotVerified),
    };

    if !health_verifies_metadata(&health, &metadata) {
        return Ok(CleanupOutcome::NotVerified);
    }

    kill_verified_process_tree(metadata.backend_pid).await?;
    remove_metadata_file(path);
    info!(
        "Cleaned up verified desktop-owned backend orphan on port {}",
        port
    );
    Ok(CleanupOutcome::Killed)
}

async fn kill_verified_process_tree(pid: u32) -> Result<(), String> {
    #[cfg(unix)]
    {
        if pid > i32::MAX as u32 {
            return Err(format!("PID {pid} exceeds i32 range"));
        }
        let process_group = -(pid as i32);
        if unsafe { libc::kill(process_group, libc::SIGTERM) } != 0 {
            return Err(format!(
                "SIGTERM failed for verified process group {pid}: {}",
                std::io::Error::last_os_error()
            ));
        }
        tokio::time::sleep(Duration::from_millis(750)).await;
        if unsafe { libc::kill(process_group, libc::SIGKILL) } != 0 {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::ESRCH) {
                return Err(format!(
                    "SIGKILL failed for verified process group {pid}: {error}"
                ));
            }
        }
        return Ok(());
    }

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let status = std::process::Command::new("taskkill.exe")
            .creation_flags(crate::process::CREATE_NO_WINDOW)
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map_err(|e| e.to_string())?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("taskkill exited with {status}"))
        }
    }

    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        Err("verified orphan cleanup is not supported on this platform".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROOT_ID: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const TOKEN: &str = "desktop-owner-token";

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
    fn managed_studio_root_id_path_points_at_tauri_default_root() {
        let home = Path::new("home").join("alex");
        assert_eq!(
            managed_studio_root_id_path(&home),
            home.join(".unsloth")
                .join("studio")
                .join("share")
                .join("studio_install_id")
        );
    }

    #[test]
    fn metadata_path_uses_default_tauri_root() {
        let home = Path::new("home").join("alex");
        assert_eq!(
            metadata_path_for_home(&home),
            home.join(".unsloth")
                .join("studio")
                .join("run")
                .join("desktop_backend.json")
        );
    }

    #[test]
    fn token_sha256_is_stable() {
        assert_eq!(
            token_sha256(TOKEN),
            "943501cb7d1feb2aa8cde1bf09b80092c25b95dbafaca9ccc12d6785b229a6fd"
        );
    }

    #[test]
    fn metadata_well_formed_requires_matching_token_hash() {
        let mut metadata = DesktopBackendMetadata {
            schema_version: METADATA_SCHEMA_VERSION,
            kind: OWNER_KIND_TAURI.to_string(),
            token: TOKEN.to_string(),
            token_sha256: token_sha256(TOKEN),
            app_pid: 1,
            backend_pid: 2,
            generation: 3,
            requested_port: 8888,
            port: Some(8888),
            studio_root_id: ROOT_ID.to_string(),
            started_at_ms: 1,
            updated_at_ms: 1,
        };
        assert!(metadata_is_well_formed(&metadata));
        metadata.token_sha256 = token_sha256("different");
        assert!(!metadata_is_well_formed(&metadata));
    }

    #[test]
    fn health_verification_requires_root_kind_and_token_sha() {
        let metadata = DesktopBackendMetadata {
            schema_version: METADATA_SCHEMA_VERSION,
            kind: OWNER_KIND_TAURI.to_string(),
            token: TOKEN.to_string(),
            token_sha256: token_sha256(TOKEN),
            app_pid: 1,
            backend_pid: 2,
            generation: 3,
            requested_port: 8888,
            port: Some(8888),
            studio_root_id: ROOT_ID.to_string(),
            started_at_ms: 1,
            updated_at_ms: 1,
        };
        let health = HealthResponse {
            status: Some("healthy".to_string()),
            service: Some("Unsloth UI Backend".to_string()),
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

    async fn owner_health_server() -> u16 {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let body = format!(
            r#"{{"status":"healthy","service":"Unsloth UI Backend","studio_root_id":"{ROOT_ID}","desktop_owner":{{"kind":"tauri","token_sha256":"{}"}}}}"#,
            token_sha256(TOKEN)
        );
        tokio::spawn(async move {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let mut buffer = [0; 2048];
            let Ok(n) = stream.read(&mut buffer).await else {
                return;
            };
            let request = String::from_utf8_lossy(&buffer[..n]);
            let (status, body) = if request.starts_with("GET /api/health ") {
                ("200 OK", body.as_str())
            } else {
                ("404 Not Found", "")
            };
            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes()).await;
        });
        port
    }

    #[tokio::test]
    async fn cleanup_skips_current_app_metadata_before_kill() {
        *TEST_EXPECTED_STUDIO_ROOT_ID.lock().unwrap() = Some(ROOT_ID.to_string());
        let port = owner_health_server().await;
        let dir = std::env::temp_dir().join(format!(
            "unsloth-current-owner-test-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let path = dir.join("desktop_backend.json");
        let metadata = DesktopBackendMetadata {
            schema_version: METADATA_SCHEMA_VERSION,
            kind: OWNER_KIND_TAURI.to_string(),
            token: TOKEN.to_string(),
            token_sha256: token_sha256(TOKEN),
            app_pid: std::process::id(),
            backend_pid: i32::MAX as u32 + 1,
            generation: 3,
            requested_port: port,
            port: Some(port),
            studio_root_id: ROOT_ID.to_string(),
            started_at_ms: 1,
            updated_at_ms: 1,
        };
        write_metadata(&path, &metadata).unwrap();

        assert_eq!(
            cleanup_verified_desktop_orphan_at_path(&path)
                .await
                .unwrap(),
            CleanupOutcome::NotVerified
        );
        assert!(path.exists());

        let _ = std::fs::remove_dir_all(dir);
    }
}
