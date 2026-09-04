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
// The app's own pid, so the backend can watch the exact owner process instead
// of sampling getppid (racy under a subreaper when the app dies mid-startup).
pub(crate) const OWNER_PID_ENV: &str = "UNSLOTH_STUDIO_DESKTOP_OWNER_PID";
pub(crate) const OWNER_KIND_TAURI: &str = "tauri";

const METADATA_SCHEMA_VERSION: u8 = 1;
const STUDIO_INSTALL_ID_HEX_LEN: usize = 64;
const STUDIO_INSTALL_ID_BYTES: usize = STUDIO_INSTALL_ID_HEX_LEN / 2;
const STUDIO_INSTALL_ID_LOCK_FILE: &str = ".studio_install_id.lock";
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
    pub backend_version: Option<String>,
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

/// `NotVerified` reason for a port that answered the probe in full and named an owner that is
/// not this app's.
///
/// Kept apart from `owned_backend_not_found`, which also covers a port that said nothing at
/// all. The health watchdog needs the difference: silence from a port an Unsloth backend just
/// answered on is a stall and earns the wide busy budget, while a complete answer carrying a
/// different root id, a different token or no desktop owner at all is proof that the backend
/// this app adopted is gone and something else has the port.
pub(crate) const OWNED_BACKEND_OWNER_MISMATCH: &str = "owned_backend_owner_mismatch";

/// Whether a failed probe answered with a different owner rather than falling silent.
pub(crate) fn probe_saw_a_different_owner(probe: &OwnedBackendProbe) -> bool {
    matches!(
        probe,
        OwnedBackendProbe::NotVerified { reason } if reason == OWNED_BACKEND_OWNER_MISMATCH
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PreviousAppPidStatus {
    Dead,
    AliveOrCurrent,
    Uncertain,
}

#[derive(Clone, Debug, Deserialize)]
struct HealthDesktopOwner {
    kind: Option<String>,
    token_sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HealthResponse {
    version: Option<String>,
    native_path_leases_supported: Option<bool>,
    torch_warm_in_progress: Option<bool>,
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

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct DesktopLiveness {
    status: Option<String>,
    service: Option<String>,
    desktop_protocol_version: Option<u16>,
    desktop_manageability_version: Option<u16>,
    supports_desktop_auth: Option<bool>,
    supports_desktop_backend_ownership: Option<bool>,
    studio_root_id: Option<String>,
    desktop_owner: Option<HealthDesktopOwner>,
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

/// Returns the managed Unsloth root ID, creating it when absent.
/// Desktop installs skip the installer step that normally creates it.
pub(crate) fn ensure_managed_studio_root_id() -> Result<String, String> {
    #[cfg(test)]
    if let Ok(guard) = TEST_EXPECTED_STUDIO_ROOT_ID.lock() {
        if let Some(value) = guard.clone() {
            return Ok(value);
        }
    }

    let path = managed_studio_root_id_path(&home_dir_or_error()?);
    ensure_studio_root_id_at(&path, true)?.ok_or_else(|| {
        format!(
            "could not create the desktop ownership id at {}",
            path.display()
        )
    })
}

/// Repairs a missing ID only when a managed install already exists.
pub(crate) fn ensure_installed_studio_root_id() -> Result<Option<String>, String> {
    let path = managed_studio_root_id_path(&home_dir_or_error()?);
    ensure_studio_root_id_at(&path, crate::process::find_unsloth_binary().is_some())
}

fn home_dir_or_error() -> Result<PathBuf, String> {
    dirs::home_dir().ok_or_else(|| "could not resolve the home directory".to_string())
}

fn ensure_studio_root_id_at(
    path: &Path,
    create_when_missing: bool,
) -> Result<Option<String>, String> {
    ensure_studio_root_id_at_with_blank_observer(path, create_when_missing, || {})
}

fn ensure_studio_root_id_at_with_blank_observer(
    path: &Path,
    create_when_missing: bool,
    after_blank_observed: impl FnOnce(),
) -> Result<Option<String>, String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("desktop ownership id path {} has no parent", path.display()))?;

    // Do not create the share directory before Unsloth is installed.
    if !create_when_missing && !path.exists() {
        return Ok(None);
    }

    // Reading an existing id needs no lock: it is published atomically, so an
    // unlocked read only ever sees a complete id. Locking first would make a
    // read-only or full share/ block startup even when the id is right there.
    if let Some(existing) = read_studio_root_id_file(path)? {
        set_private_dir_permissions(parent);
        return Ok(Some(existing));
    }

    std::fs::create_dir_all(parent)
        .map_err(|error| format!("could not create {}: {}", parent.display(), error))?;
    set_private_dir_permissions(parent);
    let _lock = lock_studio_root_id(parent)?;

    // Re-read under the lock; a concurrent caller may have won.
    if let Some(existing) = read_studio_root_id_file(path)? {
        return Ok(Some(existing));
    }
    if !create_when_missing {
        return Ok(None);
    }
    // Remove interrupted blank writes under the install lock.
    if matches!(std::fs::read_to_string(path), Ok(raw) if is_blank_studio_root_id(&raw)) {
        after_blank_observed();
        match std::fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(format!(
                    "could not replace the blank desktop ownership id at {}: {}",
                    path.display(),
                    error
                ))
            }
        }
    }
    if let Some(created) = create_studio_root_id_file(path)? {
        return Ok(Some(created));
    }
    // Adopt the ID created by a concurrent caller.
    match read_studio_root_id_file(path)? {
        Some(winner) => Ok(Some(winner)),
        None => Err(format!(
            "could not create the desktop ownership id at {}; delete that file and reopen Unsloth",
            path.display()
        )),
    }
}

fn lock_studio_root_id(parent: &Path) -> Result<std::fs::File, String> {
    let lock_path = parent.join(STUDIO_INSTALL_ID_LOCK_FILE);
    let mut options = std::fs::OpenOptions::new();
    options.create(true).read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options
        .open(&lock_path)
        .map_err(|error| format!("could not open {}: {}", lock_path.display(), error))?;
    file.lock()
        .map_err(|error| format!("could not lock {}: {}", lock_path.display(), error))?;
    set_private_file_permissions(&lock_path);
    Ok(file)
}

// Match installer behavior: blank IDs are interrupted writes.
fn is_blank_studio_root_id(raw: &str) -> bool {
    raw.trim().is_empty()
}

// create_studio_root_id_file hard-links the temp file onto the real path and then
// removes the temp name, so for the width of that unlink there are two names for one
// file. Windows denies an open of EITHER name while a delete is pending, with
// ERROR_ACCESS_DENIED rather than a sharing violation, so a concurrent reader is turned
// away from a file that is intact on both sides of the window.
const STUDIO_ROOT_ID_READ_ATTEMPTS: usize = 5;
const STUDIO_ROOT_ID_READ_BACKOFF: Duration = Duration::from_millis(20);

fn read_studio_root_id_to_string(path: &Path) -> std::io::Result<String> {
    read_studio_root_id_to_string_with(path, |path| std::fs::read_to_string(path))
}

fn read_studio_root_id_to_string_with(
    path: &Path,
    mut read: impl FnMut(&Path) -> std::io::Result<String>,
) -> std::io::Result<String> {
    let mut attempt = 0;
    loop {
        match read(path) {
            Ok(raw) => return Ok(raw),
            // Absent is an answer, not a transient state.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Err(error),
            Err(error) if is_transient_read_denial(&error) => {
                attempt += 1;
                if attempt >= STUDIO_ROOT_ID_READ_ATTEMPTS {
                    return Err(error);
                }
                std::thread::sleep(STUDIO_ROOT_ID_READ_BACKOFF);
            }
            Err(error) => return Err(error),
        }
    }
}

fn is_transient_read_denial(error: &std::io::Error) -> bool {
    matches!(error.kind(), std::io::ErrorKind::PermissionDenied)
}

fn read_studio_root_id_file(path: &Path) -> Result<Option<String>, String> {
    let raw = match read_studio_root_id_to_string(path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(format!(
                "could not read the desktop ownership id at {}: {}",
                path.display(),
                error
            ))
        }
    };
    if is_blank_studio_root_id(&raw) {
        return Ok(None);
    }
    // Never rewrite malformed IDs because a running backend may still report
    // the previous value.
    parse_studio_root_id(&raw).map(Some).ok_or_else(|| {
        format!(
            "the desktop ownership id at {} is not 64 lowercase hex characters; delete that file and reopen Unsloth",
            path.display()
        )
    })
}

/// Returns the created id, or `None` when another process won the race.
fn create_studio_root_id_file(path: &Path) -> Result<Option<String>, String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("desktop ownership id path {} has no parent", path.display()))?;
    let id = hex_bytes(&rand::random::<[u8; STUDIO_INSTALL_ID_BYTES]>());
    // Unique temp names isolate concurrent publishers.
    let tmp = parent.join(format!(".studio_install_id.{}.tmp", &id[..16]));
    let claimed = claim_private_file(&tmp, path, id.as_bytes());
    let _ = std::fs::remove_file(&tmp);
    claimed.map(|claimed| claimed.then_some(id))
}

/// Atomically publishes a flushed temp file without replacing an existing ID.
fn claim_private_file(tmp: &Path, path: &Path, body: &[u8]) -> Result<bool, String> {
    claim_private_file_with_link(tmp, path, body, |prepared, destination| {
        std::fs::hard_link(prepared, destination)
    })
}

fn claim_private_file_with_link(
    tmp: &Path,
    path: &Path,
    body: &[u8],
    hard_link: impl FnOnce(&Path, &Path) -> std::io::Result<()>,
) -> Result<bool, String> {
    let _ = std::fs::remove_file(tmp);
    write_private_file(tmp, body)?;
    match hard_link(tmp, path) {
        Ok(()) => {
            set_private_file_permissions(path);
            Ok(true)
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
        // The install lock makes rename a safe no-clobber fallback.
        Err(_) => publish_private_file_by_rename(tmp, path),
    }
}

fn publish_private_file_by_rename(tmp: &Path, path: &Path) -> Result<bool, String> {
    if path.exists() {
        return Ok(false);
    }
    std::fs::rename(tmp, path)
        .map_err(|error| format!("could not publish {}: {}", path.display(), error))?;
    set_private_file_permissions(path);
    Ok(true)
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

pub(crate) fn new_pending_owner() -> Result<PendingBackendOwner, String> {
    Ok(PendingBackendOwner {
        token: random_owner_token(),
        studio_root_id: ensure_managed_studio_root_id()?,
    })
}

/// Applies the identity required for ownership and parent-watchdog tracking.
pub(crate) fn apply_owner_env(cmd: &mut std::process::Command, pending: &PendingBackendOwner) {
    cmd.env(OWNER_TOKEN_ENV, pending.token.as_str());
    cmd.env(OWNER_KIND_ENV, OWNER_KIND_TAURI);
    cmd.env(OWNER_PID_ENV, std::process::id().to_string());
}

pub(crate) fn activate_owner(
    pending: PendingBackendOwner,
    requested_port: u16,
    generation: u64,
    backend_pid: u32,
) -> Result<BackendOwnerState, String> {
    let path = metadata_path().ok_or_else(|| "could not resolve the home directory".to_string())?;
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
    state.write().map_err(|error| {
        format!(
            "could not write the desktop backend ownership metadata at {}: {}",
            state.path.display(),
            error
        )
    })?;
    Ok(state)
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
        match fetch_liveness_blocking(port) {
            Ok(Some(liveness)) => {
                liveness_verifies_metadata(&liveness, &self.metadata)
                    && lifecycle_control_block_reason(&liveness).is_none()
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
        Ok(())
    }

    // Flush here too, so a crash right after publishing cannot leave a
    // zero-length id behind. Permissions come from the user profile ACL.
    #[cfg(not(unix))]
    {
        let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;
        file.write_all(body).map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        Ok(())
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

fn liveness_verifies_metadata(
    liveness: &DesktopLiveness,
    metadata: &DesktopBackendMetadata,
) -> bool {
    let alive = matches!(liveness.status.as_deref(), Some("alive") | Some("healthy"))
        && liveness.service.as_deref() == Some("Unsloth UI Backend");
    let Some(owner) = liveness.desktop_owner.as_ref() else {
        return false;
    };
    alive
        && owner_matches_metadata(
            metadata,
            liveness.studio_root_id.as_deref(),
            owner.kind.as_deref(),
            owner.token_sha256.as_deref(),
        )
}

fn lifecycle_control_block_reason(liveness: &DesktopLiveness) -> Option<String> {
    if liveness.desktop_protocol_version != Some(crate::preflight::DESKTOP_PROTOCOL_VERSION) {
        return Some("desktop_protocol_incompatible".to_string());
    }
    if liveness.supports_desktop_auth != Some(true) {
        return Some("desktop_auth_unsupported".to_string());
    }
    if liveness.desktop_manageability_version.unwrap_or(0)
        < crate::preflight::DESKTOP_BACKEND_MANAGEABILITY_VERSION
    {
        return Some("desktop_manageability_unsupported".to_string());
    }
    if liveness.supports_desktop_backend_ownership != Some(true) {
        return Some("desktop_backend_ownership_unsupported".to_string());
    }
    None
}

fn ready_for_use_status(health: Option<&HealthResponse>) -> OwnedBackendReadiness {
    let version = health
        .and_then(|h| h.version.as_deref())
        .filter(|v| !v.is_empty());
    if let Some(reason) = crate::preflight::backend_version_stale_reason(version) {
        return OwnedBackendReadiness::Stale { reason };
    }
    match health.and_then(|health| health.native_path_leases_supported) {
        Some(true) => OwnedBackendReadiness::Ready,
        _ => OwnedBackendReadiness::Stale {
            reason: "native_path_leases_unsupported".to_string(),
        },
    }
}

async fn fetch_liveness(
    port: u16,
    timeout: Duration,
) -> Result<Option<DesktopLiveness>, reqwest::Error> {
    let client = crate::loopback_http::client(timeout)?;
    for path in ["/api/liveness", "/api/health"] {
        let response = client
            .get(format!("http://127.0.0.1:{port}{path}"))
            .send()
            .await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND && path == "/api/liveness" {
            continue;
        }
        if !response.status().is_success() {
            return Ok(None);
        }
        return response.json::<DesktopLiveness>().await.map(Some);
    }
    Ok(None)
}

fn fetch_liveness_blocking(port: u16) -> Result<Option<DesktopLiveness>, String> {
    for path in ["/api/liveness", "/api/health"] {
        let response = http_request_blocking(port, "GET", path, &[], &[])?;
        if response.status == 404 && path == "/api/liveness" {
            continue;
        }
        if !(200..300).contains(&response.status) {
            return Ok(None);
        }
        return serde_json::from_slice::<DesktopLiveness>(&response.body)
            .map(Some)
            .map_err(|e| e.to_string());
    }
    Ok(None)
}
async fn fetch_health(
    port: u16,
    access_token: Option<&str>,
) -> Result<Option<HealthResponse>, String> {
    let client = crate::loopback_http::client(LOCAL_HTTP_TIMEOUT).map_err(|e| e.to_string())?;
    let mut request = client.get(format!("http://127.0.0.1:{port}/api/health"));
    if let Some(access_token) = access_token {
        request = request.bearer_auth(access_token);
    }
    let response = request.send().await.map_err(|e| e.to_string())?;
    if access_token.is_some()
        && matches!(
            response.status(),
            reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN
        )
    {
        return Err("desktop_auth_token_rejected".to_string());
    }
    if !response.status().is_success() {
        return Ok(None);
    }
    response
        .json::<HealthResponse>()
        .await
        .map(Some)
        .map_err(|e| e.to_string())
}

async fn desktop_login_route_compatible(port: u16, timeout: Duration) -> bool {
    let client = match crate::loopback_http::client(timeout) {
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

async fn desktop_secret_login(port: u16, secret: &str) -> Result<String, String> {
    let client = crate::loopback_http::client(LOCAL_HTTP_TIMEOUT).map_err(|e| e.to_string())?;
    let response = client
        .post(format!("http://127.0.0.1:{port}/api/auth/desktop-login"))
        .json(&DesktopLoginPayload { secret })
        .send()
        .await
        .map_err(|_| "desktop_auth_secret_probe_failed".to_string())?;
    if response.status() == reqwest::StatusCode::UNAUTHORIZED {
        Err("desktop_auth_secret_rejected".to_string())
    } else if !response.status().is_success() {
        Err(format!(
            "desktop_auth_secret_probe_http_{}",
            response.status()
        ))
    } else {
        response
            .json::<TokenResponse>()
            .await
            .map(|tokens| tokens.access_token)
            .map_err(|_| "desktop_auth_token_response_invalid".to_string())
    }
}

async fn authenticated_health_ready_status(
    port: u16,
    secret: &str,
) -> Result<OwnedBackendReadiness, String> {
    authenticated_health_ready(port, secret)
        .await
        .map(|(readiness, _, _)| readiness)
}

async fn authenticated_health_ready(
    port: u16,
    secret: &str,
) -> Result<(OwnedBackendReadiness, String, bool), String> {
    let access_token = desktop_secret_login(port, secret).await?;
    let health = fetch_health(port, Some(&access_token))
        .await?
        .ok_or_else(|| "desktop_auth_health_unverified".to_string())?;
    let version = health
        .version
        .as_deref()
        .filter(|version| !version.is_empty())
        .ok_or_else(|| "desktop_auth_health_unverified".to_string())?;
    if let Some(reason) = crate::preflight::backend_version_stale_reason(Some(version)) {
        if reason != "desktop_backend_version_too_old" {
            return Err(reason);
        }
    }
    Ok((
        ready_for_use_status(Some(&health)),
        version.to_string(),
        health.torch_warm_in_progress.unwrap_or(false),
    ))
}

pub(crate) async fn probe_owned_backend_state(
    owner: BackendOwnerState,
    port: Option<u16>,
    require_desktop_secret: bool,
) -> OwnedBackendProbe {
    probe_owned_backend_state_with_timeout(owner, port, require_desktop_secret, LOCAL_HTTP_TIMEOUT)
        .await
}

/// As above, but with an explicit per-request budget.
///
/// The health watchdog needs this. Its probes have to survive the multi-second GIL stalls
/// the backend's warm thread causes while it imports the ML stack, and at the default 2s
/// every request here times out during exactly the stall the watchdog is meant to tolerate,
/// so the backend reads as unverified and gets cleared.
pub(crate) async fn probe_owned_backend_state_with_timeout(
    owner: BackendOwnerState,
    port: Option<u16>,
    require_desktop_secret: bool,
    timeout: Duration,
) -> OwnedBackendProbe {
    probe_owned_backend_state_with_warmup(owner, port, require_desktop_secret, timeout)
        .await
        .0
}

pub(crate) async fn probe_owned_backend_state_for_staged_activation(
    owner: BackendOwnerState,
    port: Option<u16>,
) -> (OwnedBackendProbe, bool) {
    probe_owned_backend_state_with_warmup(owner, port, true, LOCAL_HTTP_TIMEOUT).await
}

async fn probe_owned_backend_state_with_warmup(
    owner: BackendOwnerState,
    port: Option<u16>,
    require_desktop_secret: bool,
    timeout: Duration,
) -> (OwnedBackendProbe, bool) {
    let ports: Vec<u16> = match port {
        Some(port) => vec![port],
        None => desktop_candidate_ports().collect(),
    };
    let mut verified = Vec::new();
    // Set only by a complete, parsed answer that names someone else. A transport error or a
    // non-success status leaves it alone, so silence never reads as a takeover.
    let mut answered_with_a_different_owner = false;
    for port in ports {
        let liveness = match fetch_liveness(port, timeout).await {
            Ok(Some(liveness)) => liveness,
            Ok(None) => continue,
            Err(error) => {
                warn!(
                    "Desktop-owned backend probe skipped port {} after liveness error: {}",
                    port, error
                );
                continue;
            }
        };
        if !liveness_verifies_metadata(&liveness, &owner.metadata) {
            answered_with_a_different_owner = true;
            continue;
        }
        if let Some(reason) = lifecycle_control_block_reason(&liveness) {
            return (OwnedBackendProbe::Unmanageable { port, reason }, false);
        }
        if !desktop_login_route_compatible(port, timeout).await {
            return (
                OwnedBackendProbe::Unmanageable {
                    port,
                    reason: "desktop_login_probe_failed".to_string(),
                },
                false,
            );
        }
        let (readiness, backend_version, torch_warm_in_progress) = if require_desktop_secret {
            let secret = match read_desktop_secret() {
                Ok(Some(secret)) => secret,
                Ok(None) => {
                    return (
                        OwnedBackendProbe::Unmanageable {
                            port,
                            reason: "desktop_auth_secret_missing".to_string(),
                        },
                        false,
                    )
                }
                Err(reason) => return (OwnedBackendProbe::Unmanageable { port, reason }, false),
            };
            match authenticated_health_ready(port, &secret).await {
                Ok((readiness, version, torch_warm_in_progress)) => {
                    (readiness, Some(version), torch_warm_in_progress)
                }
                Err(reason) => return (OwnedBackendProbe::Unmanageable { port, reason }, false),
            }
        } else {
            // Spawned backends were launched from the already-probed managed
            // install. Adopted backends pass `true` on their initial probe;
            // later watchdog checks only need ownership and liveness.
            (OwnedBackendReadiness::Ready, None, false)
        };
        verified.push((port, readiness, backend_version, torch_warm_in_progress));
    }

    if verified.len() != 1 {
        return (
            OwnedBackendProbe::NotVerified {
                reason: if !verified.is_empty() {
                    "owned_backend_ambiguous".to_string()
                } else if answered_with_a_different_owner {
                    OWNED_BACKEND_OWNER_MISMATCH.to_string()
                } else {
                    "owned_backend_not_found".to_string()
                },
            },
            false,
        );
    }

    let (port, readiness, backend_version, torch_warm_in_progress) = verified.remove(0);
    (
        OwnedBackendProbe::Verified(VerifiedOwnedBackend {
            backend_pid: owner.backend_pid(),
            generation: owner.generation(),
            owner,
            port,
            readiness,
            backend_version,
        }),
        torch_warm_in_progress,
    )
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

/// False only when the pid is provably gone.
///
/// A pid we cannot resolve counts as running: on Windows an OpenProcess that
/// fails for anything but a bad pid usually means the process is there and
/// owned by somebody else, and every caller here treats "still running" as the
/// safe answer.
pub(crate) fn pid_is_not_dead(pid: u32) -> bool {
    process_liveness(pid) != PreviousAppPidStatus::Dead
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
    use windows_sys::Win32::Foundation::{
        CloseHandle, GetLastError, ERROR_INVALID_PARAMETER, WAIT_OBJECT_0, WAIT_TIMEOUT,
    };
    use windows_sys::Win32::System::Threading::{
        OpenProcess, WaitForSingleObject, PROCESS_SYNCHRONIZE,
    };

    unsafe {
        let handle = OpenProcess(PROCESS_SYNCHRONIZE, 0, pid);
        if handle.is_null() {
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

    async fn http_sequence_server(
        responses: Vec<(&'static str, &'static str)>,
    ) -> (
        u16,
        std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        tokio::task::JoinHandle<()>,
    ) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen_task = seen.clone();
        let task = tokio::spawn(async move {
            for (status, body) in responses {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut raw = Vec::new();
                loop {
                    let mut chunk = [0u8; 2048];
                    let count = socket.read(&mut chunk).await.unwrap();
                    if count == 0 {
                        break;
                    }
                    raw.extend_from_slice(&chunk[..count]);
                    let Some(header_end) = raw.windows(4).position(|w| w == b"\r\n\r\n") else {
                        continue;
                    };
                    let headers = String::from_utf8_lossy(&raw[..header_end]);
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            line.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .and_then(|value| value.trim().parse::<usize>().ok())
                        })
                        .unwrap_or(0);
                    if raw.len() >= header_end + 4 + content_length {
                        break;
                    }
                }
                seen_task
                    .lock()
                    .unwrap()
                    .push(String::from_utf8_lossy(&raw).into_owned());
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            }
        });
        (port, seen, task)
    }

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

    #[tokio::test]
    async fn authenticated_health_uses_login_bearer_and_accepts_ready_version() {
        let (port, seen, server) = http_sequence_server(vec![
            ("200 OK", r#"{"access_token":"test-access-token"}"#),
            (
                "200 OK",
                r#"{"version":"2026.8.4","native_path_leases_supported":true}"#,
            ),
        ])
        .await;

        let (readiness, version, torch_warm_in_progress) =
            authenticated_health_ready(port, "desktop-test-secret")
                .await
                .unwrap();
        server.await.unwrap();

        assert!(matches!(readiness, OwnedBackendReadiness::Ready));
        assert_eq!(version, "2026.8.4");
        assert!(!torch_warm_in_progress);
        let seen = seen.lock().unwrap();
        assert!(seen[0].contains(r#""secret":"desktop-test-secret""#));
        assert!(seen[1]
            .to_ascii_lowercase()
            .contains("authorization: bearer test-access-token"));
    }

    #[tokio::test]
    async fn authenticated_health_reports_torch_warm_in_progress() {
        let (port, _, server) = http_sequence_server(vec![
            ("200 OK", r#"{"access_token":"test-access-token"}"#),
            (
                "200 OK",
                r#"{"version":"2026.8.4","native_path_leases_supported":true,"torch_warm_in_progress":true}"#,
            ),
        ])
        .await;

        let (readiness, version, torch_warm_in_progress) =
            authenticated_health_ready(port, "desktop-test-secret")
                .await
                .unwrap();
        server.await.unwrap();

        assert!(matches!(readiness, OwnedBackendReadiness::Ready));
        assert_eq!(version, "2026.8.4");
        assert!(torch_warm_in_progress);
    }

    #[tokio::test]
    async fn authenticated_health_marks_missing_or_disabled_native_leases_stale() {
        for health in [
            r#"{"version":"2026.8.4"}"#,
            r#"{"version":"2026.8.4","native_path_leases_supported":false}"#,
        ] {
            let (port, _, server) = http_sequence_server(vec![
                ("200 OK", r#"{"access_token":"test-access-token"}"#),
                ("200 OK", health),
            ])
            .await;

            let readiness = authenticated_health_ready_status(port, "desktop-test-secret")
                .await
                .unwrap();
            server.await.unwrap();

            assert!(matches!(
                readiness,
                OwnedBackendReadiness::Stale { reason }
                    if reason == "native_path_leases_unsupported"
            ));
        }
    }

    #[tokio::test]
    async fn authenticated_health_preserves_genuinely_old_version_as_stale() {
        let (port, _, server) = http_sequence_server(vec![
            ("200 OK", r#"{"access_token":"test-access-token"}"#),
            ("200 OK", r#"{"version":"2026.5.2"}"#),
        ])
        .await;

        let readiness = authenticated_health_ready_status(port, "desktop-test-secret")
            .await
            .unwrap();
        server.await.unwrap();

        assert!(matches!(
            readiness,
            OwnedBackendReadiness::Stale { reason }
                if reason == "desktop_backend_version_too_old"
        ));
    }

    #[tokio::test]
    async fn authenticated_health_rejects_malformed_version() {
        let (port, _, server) = http_sequence_server(vec![
            ("200 OK", r#"{"access_token":"test-access-token"}"#),
            ("200 OK", r#"{"version":"not-a-version"}"#),
        ])
        .await;

        let result = authenticated_health_ready_status(port, "desktop-test-secret").await;
        server.await.unwrap();

        assert_eq!(result.unwrap_err(), "desktop_backend_version_invalid");
    }

    #[tokio::test]
    async fn rejected_desktop_secret_fails_before_health() {
        let (port, seen, server) = http_sequence_server(vec![("401 Unauthorized", "{}")]).await;

        let result = authenticated_health_ready_status(port, "wrong-secret").await;
        server.await.unwrap();

        assert_eq!(result.unwrap_err(), "desktop_auth_secret_rejected");
        assert_eq!(seen.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn rejected_health_bearer_is_not_downgraded_to_auto_repairable_stale() {
        let (port, _, server) = http_sequence_server(vec![
            ("200 OK", r#"{"access_token":"test-access-token"}"#),
            ("401 Unauthorized", "{}"),
        ])
        .await;

        let result = authenticated_health_ready_status(port, "desktop-test-secret").await;
        server.await.unwrap();

        assert_eq!(result.unwrap_err(), "desktop_auth_token_rejected");
    }

    #[tokio::test]
    async fn authenticated_health_rejects_public_payload_without_version() {
        let (port, _, server) = http_sequence_server(vec![
            ("200 OK", r#"{"access_token":"test-access-token"}"#),
            (
                "200 OK",
                r#"{"status":"healthy","service":"Unsloth UI Backend","supports_desktop_auth":true}"#,
            ),
        ])
        .await;

        let result = authenticated_health_ready_status(port, "desktop-test-secret").await;
        server.await.unwrap();

        assert_eq!(result.unwrap_err(), "desktop_auth_health_unverified");
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

    fn owned_liveness(manageability: u16) -> DesktopLiveness {
        DesktopLiveness {
            status: Some("alive".to_string()),
            service: Some("Unsloth UI Backend".to_string()),
            desktop_protocol_version: Some(1),
            desktop_manageability_version: Some(manageability),
            supports_desktop_auth: Some(true),
            supports_desktop_backend_ownership: Some(true),
            studio_root_id: Some(ROOT_ID.to_string()),
            desktop_owner: Some(HealthDesktopOwner {
                kind: Some(OWNER_KIND_TAURI.to_string()),
                token_sha256: Some(token_sha256(TOKEN)),
            }),
        }
    }

    /// A backend that answers the ownership probe's first request and then goes quiet, the
    /// way a saturated one does. The later connections are parked, not closed: closing them
    /// would answer with a reset, which is a different failure entirely.
    async fn owned_backend_that_stalls_after_the_first_request() -> u16 {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let body = format!(
            r#"{{"status":"alive","service":"Unsloth UI Backend","desktop_protocol_version":{},"desktop_manageability_version":{},"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,"studio_root_id":"{}","desktop_owner":{{"kind":"{}","token_sha256":"{}"}},"inference_active":true}}"#,
            crate::preflight::DESKTOP_PROTOCOL_VERSION,
            crate::preflight::DESKTOP_MANAGEABILITY_VERSION,
            ROOT_ID,
            OWNER_KIND_TAURI,
            token_sha256(TOKEN),
        );
        tokio::spawn(async move {
            let mut answered = false;
            let mut parked = Vec::new();
            while let Ok((mut socket, _)) = listener.accept().await {
                if answered {
                    parked.push(socket);
                    continue;
                }
                answered = true;
                let mut chunk = [0u8; 2048];
                let _ = socket.read(&mut chunk).await;
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        port
    }

    #[tokio::test]
    async fn a_stall_after_the_first_request_is_indistinguishable_from_a_foreign_port() {
        // Why the health watchdog cannot ask this probe whether a failure was a stall: the
        // liveness GET succeeds and verifies ownership, then the desktop-login POST runs out
        // of budget, and the answer that comes back carries no trace of which it was. The
        // watchdog has to classify the failure from its own read instead, which is what
        // `commands::adopted_failure_is_a_stall` does. Only the opposite case is decidable
        // here, and is reported as `OWNED_BACKEND_OWNER_MISMATCH`: a port that answered in
        // full for somebody else did not fall silent, so it is not a stall.
        let port = owned_backend_that_stalls_after_the_first_request().await;
        // The probe never touches the file, so nothing has to exist on disk for this.
        let owner = BackendOwnerState::from_metadata(
            std::env::temp_dir().join("unsloth-stall-after-first-request.json"),
            metadata(std::process::id(), Some(port)),
        );

        let probe = probe_owned_backend_state_with_timeout(
            owner,
            Some(port),
            false,
            Duration::from_millis(400),
        )
        .await;

        match probe {
            OwnedBackendProbe::Unmanageable { reason, .. } => {
                assert_eq!(reason, "desktop_login_probe_failed");
            }
            OwnedBackendProbe::NotVerified { reason } => {
                assert_eq!(reason, "owned_backend_not_found");
            }
            other => panic!("a stalled owned backend should not verify: {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_port_taken_over_by_another_backend_is_reported_as_an_owner_mismatch() {
        // The other half of the classification above. A backend the app adopted can exit and
        // have its port rebound by the next Unsloth backend the user starts, which answers
        // the watchdog's pre-probe exactly as the old one did. The probe gets a complete
        // reply here, not silence, so it must say so: the watchdog reads this to keep the
        // dead adopted backend on the normal three-strike budget instead of the busy one.
        //
        // A backend started outside the app omits `desktop_owner` entirely (main.py only
        // emits the key when one is loaded), and a second app instance sends a different
        // token hash. Both are takeovers.
        for body in [
            // Same install, so the root id matches; no desktop owner at all.
            r#"{"status":"alive","service":"Unsloth UI Backend","studio_root_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","supports_desktop_auth":true,"supports_desktop_backend_ownership":true}"#,
            // A desktop-owned backend, but not the one this app is holding.
            r#"{"status":"alive","service":"Unsloth UI Backend","studio_root_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","supports_desktop_auth":true,"supports_desktop_backend_ownership":true,"desktop_owner":{"kind":"tauri","token_sha256":"0000000000000000000000000000000000000000000000000000000000000000"}}"#,
        ] {
            let (port, _, server) = http_sequence_server(vec![("200 OK", body)]).await;
            let owner = BackendOwnerState::from_metadata(
                std::env::temp_dir().join("unsloth-port-taken-over.json"),
                metadata(std::process::id(), Some(port)),
            );

            let probe = probe_owned_backend_state_with_timeout(
                owner,
                Some(port),
                false,
                Duration::from_secs(5),
            )
            .await;
            server.await.unwrap();

            assert!(
                probe_saw_a_different_owner(&probe),
                "a port answering for a different owner read as silence: {probe:?}"
            );
        }
    }

    #[tokio::test]
    async fn a_port_that_says_nothing_is_not_reported_as_an_owner_mismatch() {
        // The guard on the above: a closed port and a stalled one both fail to verify, and
        // neither is evidence that someone else took the port. Only a parsed answer is.
        let port = closed_port();
        let owner = BackendOwnerState::from_metadata(
            std::env::temp_dir().join("unsloth-silent-port.json"),
            metadata(std::process::id(), Some(port)),
        );

        let probe =
            probe_owned_backend_state_with_timeout(owner, Some(port), false, Duration::from_secs(5))
                .await;

        assert!(!probe_saw_a_different_owner(&probe), "{probe:?}");
        assert!(matches!(
            probe,
            OwnedBackendProbe::NotVerified { ref reason } if reason == "owned_backend_not_found"
        ));
    }

    #[test]
    fn legacy_manageability_backend_stays_lifecycle_controllable() {
        // A backend from the previous app version reports manageability 1.
        // studio_install_ok is CLI-side, not part of this backend's HTTP
        // contract: blocking makes preflight answer ExternalConflict and never
        // adopt a process the root id and token already prove is ours.
        assert_eq!(lifecycle_control_block_reason(&owned_liveness(1)), None);
        assert_eq!(
            lifecycle_control_block_reason(&owned_liveness(
                crate::preflight::DESKTOP_MANAGEABILITY_VERSION
            )),
            None
        );

        // The bits a live backend really must carry are still enforced.
        let mut no_ownership = owned_liveness(1);
        no_ownership.supports_desktop_backend_ownership = Some(false);
        assert_eq!(
            lifecycle_control_block_reason(&no_ownership).as_deref(),
            Some("desktop_backend_ownership_unsupported")
        );

        let mut no_auth = owned_liveness(1);
        no_auth.supports_desktop_auth = Some(false);
        assert_eq!(
            lifecycle_control_block_reason(&no_auth).as_deref(),
            Some("desktop_auth_unsupported")
        );

        let mut old_protocol = owned_liveness(1);
        old_protocol.desktop_protocol_version = Some(0);
        assert_eq!(
            lifecycle_control_block_reason(&old_protocol).as_deref(),
            Some("desktop_protocol_incompatible")
        );

        let mut no_manageability = owned_liveness(1);
        no_manageability.desktop_manageability_version = None;
        assert_eq!(
            lifecycle_control_block_reason(&no_manageability).as_deref(),
            Some("desktop_manageability_unsupported")
        );
    }

    #[test]
    fn liveness_verification_requires_root_kind_and_token_sha() {
        let metadata = metadata(1, Some(8888));
        let liveness = owned_liveness(1);
        assert!(liveness_verifies_metadata(&liveness, &metadata));

        let mut wrong_root = liveness;
        wrong_root.studio_root_id =
            Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string());
        assert!(!liveness_verifies_metadata(&wrong_root, &metadata));
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

    fn temp_root_id_path(test_name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "unsloth-root-id-{test_name}-{}-{}-{}",
            std::process::id(),
            now_ms(),
            hex_bytes(&rand::random::<[u8; 4]>())
        ));
        dir.join("share").join("studio_install_id")
    }

    #[test]
    fn missing_studio_root_id_is_created_once_and_then_preserved() {
        let path = temp_root_id_path("create");

        let created = ensure_studio_root_id_at(&path, true).unwrap().unwrap();
        assert!(is_valid_studio_root_id(&created));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), created);

        // A second call must return the same id, not mint a new one.
        assert_eq!(
            ensure_studio_root_id_at(&path, true).unwrap(),
            Some(created.clone())
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap(), created);

        let _ = std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn created_studio_root_id_is_private_to_the_user() {
        use std::os::unix::fs::PermissionsExt;

        let path = temp_root_id_path("permissions");
        ensure_studio_root_id_at(&path, true).unwrap().unwrap();

        let file_mode = std::fs::metadata(&path).unwrap().permissions().mode();
        assert_eq!(file_mode & 0o777, 0o600);
        let dir_mode = std::fs::metadata(path.parent().unwrap())
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(dir_mode & 0o777, 0o700);
        // The temp file used to publish the id must not be left behind.
        let leftovers: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name != "studio_install_id" && name != STUDIO_INSTALL_ID_LOCK_FILE)
            .collect();
        assert!(leftovers.is_empty(), "unexpected leftovers: {leftovers:?}");

        let lock_mode = std::fs::metadata(path.parent().unwrap().join(STUDIO_INSTALL_ID_LOCK_FILE))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(lock_mode & 0o777, 0o600);

        let _ = std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
    }

    #[test]
    fn malformed_studio_root_id_is_an_error_not_a_silent_rewrite() {
        let path = temp_root_id_path("malformed");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "not-a-root-id").unwrap();

        let error = ensure_studio_root_id_at(&path, true).unwrap_err();
        assert!(error.contains(&path.display().to_string()), "{error}");
        // A backend may still be reporting the id this file used to hold.
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "not-a-root-id");

        let _ = std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
    }

    // Injected reader, not a real race: the window is one unlink wide, so a test that
    // raced for it would pass on any machine that never entered it.
    fn denial() -> std::io::Error {
        std::io::Error::new(std::io::ErrorKind::PermissionDenied, "Access is denied.")
    }

    #[test]
    fn a_reader_denied_while_the_temp_name_is_unlinked_still_gets_the_id() {
        let mut seen = 0;
        let raw = read_studio_root_id_to_string_with(Path::new("id"), |_| {
            seen += 1;
            if seen < 3 {
                Err(denial())
            } else {
                Ok("an-id".to_string())
            }
        })
        .unwrap();
        assert_eq!(raw, "an-id");
        assert_eq!(seen, 3, "the read should have been retried until it succeeded");
    }

    #[test]
    fn a_denial_that_never_clears_is_still_reported() {
        // Bounded: a genuinely unreadable file must not become a hang or a silent
        // success, and the last error is what the caller sees.
        let mut seen = 0;
        let error = read_studio_root_id_to_string_with(Path::new("id"), |_| {
            seen += 1;
            Err(denial())
        })
        .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        assert_eq!(seen, STUDIO_ROOT_ID_READ_ATTEMPTS);
    }

    #[test]
    fn a_missing_id_is_answered_without_waiting() {
        // A first start has no id file; retrying would add backoff to every cold
        // launch for the same answer.
        let mut seen = 0;
        let error = read_studio_root_id_to_string_with(Path::new("id"), |_| {
            seen += 1;
            Err(std::io::Error::from(std::io::ErrorKind::NotFound))
        })
        .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
        assert_eq!(seen, 1, "a missing id must not be retried");
    }

    #[test]
    fn an_unrelated_read_error_is_not_retried() {
        let mut seen = 0;
        let error = read_studio_root_id_to_string_with(Path::new("id"), |_| {
            seen += 1;
            Err(std::io::Error::from(std::io::ErrorKind::InvalidData))
        })
        .unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert_eq!(seen, 1, "only a denial is treated as transient");
    }

    #[test]
    fn blank_studio_root_id_is_replaced_like_a_missing_one() {
        // Blank IDs are interrupted writes and must not block later starts.
        for blank in ["", "\n"] {
            let path = temp_root_id_path("blank");
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, blank).unwrap();

            let created = ensure_studio_root_id_at(&path, true).unwrap().unwrap();
            assert!(is_valid_studio_root_id(&created));
            assert_eq!(std::fs::read_to_string(&path).unwrap(), created);

            let _ = std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
        }
    }

    #[test]
    fn stale_blank_observer_cannot_delete_a_competing_callers_id() {
        let path = temp_root_id_path("stale-blank-observer");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "").unwrap();

        let (blank_seen_tx, blank_seen_rx) = std::sync::mpsc::channel();
        let (resume_tx, resume_rx) = std::sync::mpsc::channel();
        let first_path = path.clone();
        let first = std::thread::spawn(move || {
            ensure_studio_root_id_at_with_blank_observer(&first_path, true, || {
                blank_seen_tx.send(()).unwrap();
                resume_rx.recv().unwrap();
            })
            .unwrap()
            .unwrap()
        });
        blank_seen_rx.recv().unwrap();

        // The second caller must wait until blank-file recovery completes.
        let contender = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path.parent().unwrap().join(STUDIO_INSTALL_ID_LOCK_FILE))
            .unwrap();
        let lock_error = contender.try_lock().unwrap_err();
        assert!(matches!(lock_error, std::fs::TryLockError::WouldBlock));

        let (second_tx, second_rx) = std::sync::mpsc::channel();
        let second_path = path.clone();
        let second = std::thread::spawn(move || {
            second_tx
                .send(ensure_studio_root_id_at(&second_path, true))
                .unwrap();
        });

        resume_tx.send(()).unwrap();
        let first_id = first.join().unwrap();
        let second_id = second_rx.recv().unwrap().unwrap().unwrap();
        second.join().unwrap();
        let persisted = std::fs::read_to_string(&path).unwrap();

        assert_eq!(first_id, persisted);
        assert_eq!(second_id, persisted);

        // Windows cannot delete a file that still has an open handle.
        drop(contender);
        let _ = std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
    }

    #[test]
    fn studio_root_id_is_not_created_without_a_managed_install() {
        let path = temp_root_id_path("not-installed");

        assert_eq!(ensure_studio_root_id_at(&path, false).unwrap(), None);
        assert!(!path.parent().unwrap().exists());
    }

    #[test]
    fn existing_studio_root_id_is_read_without_the_install_lock() {
        // A read-only or full share/ must not stop a usable id from being read.
        let path = temp_root_id_path("unlockable");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, ROOT_ID).unwrap();
        // Make the lock file impossible to open.
        std::fs::create_dir_all(path.parent().unwrap().join(STUDIO_INSTALL_ID_LOCK_FILE)).unwrap();

        assert_eq!(
            ensure_studio_root_id_at(&path, true).unwrap(),
            Some(ROOT_ID.to_string())
        );

        let _ = std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
    }

    #[test]
    fn concurrent_creators_converge_on_one_studio_root_id() {
        let path = temp_root_id_path("concurrent");
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));

        let handles: Vec<_> = (0..8)
            .map(|_| {
                let path = path.clone();
                let barrier = std::sync::Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    ensure_studio_root_id_at(&path, true).unwrap().unwrap()
                })
            })
            .collect();
        let ids: Vec<String> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();

        let persisted = std::fs::read_to_string(&path).unwrap();
        assert!(is_valid_studio_root_id(&persisted));
        for id in &ids {
            assert_eq!(id, &persisted);
        }
        // Filter by name so an unrelated file in share/ cannot fail this test.
        let leftovers: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name != "studio_install_id" && name != STUDIO_INSTALL_ID_LOCK_FILE)
            .collect();
        assert!(leftovers.is_empty(), "unexpected leftovers: {leftovers:?}");

        let _ = std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
    }

    #[test]
    fn no_hard_link_fallback_publishes_only_a_complete_destination() {
        let path = temp_root_id_path("rename-fallback");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let tmp = path.parent().unwrap().join("fallback.tmp");
        let body = b"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

        let claimed = claim_private_file_with_link(&tmp, &path, body, |prepared, destination| {
            assert_eq!(std::fs::read(prepared).unwrap(), body);
            assert!(!destination.exists());
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "hard links disabled for test",
            ))
        })
        .unwrap();

        assert!(claimed);
        assert_eq!(std::fs::read(&path).unwrap(), body);
        assert!(!tmp.exists());

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }

        let _ = std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
    }

    #[test]
    fn spawned_backends_always_receive_the_owner_environment() {
        let pending = PendingBackendOwner {
            token: TOKEN.to_string(),
            studio_root_id: ROOT_ID.to_string(),
        };
        let mut cmd = std::process::Command::new("unsloth");
        apply_owner_env(&mut cmd, &pending);

        let env: std::collections::HashMap<String, String> = cmd
            .get_envs()
            .filter_map(|(key, value)| {
                Some((
                    key.to_string_lossy().into_owned(),
                    value?.to_string_lossy().into_owned(),
                ))
            })
            .collect();

        assert_eq!(env.get(OWNER_TOKEN_ENV).map(String::as_str), Some(TOKEN));
        assert_eq!(
            env.get(OWNER_KIND_ENV).map(String::as_str),
            Some(OWNER_KIND_TAURI)
        );
        // The backend arms its parent watchdog only when it knows the owner pid.
        assert_eq!(
            env.get(OWNER_PID_ENV).map(String::as_str),
            Some(std::process::id().to_string().as_str())
        );
    }

    #[test]
    fn current_app_pid_is_not_adoptable() {
        assert_eq!(
            previous_app_pid_status(std::process::id()),
            PreviousAppPidStatus::AliveOrCurrent
        );
    }
}
