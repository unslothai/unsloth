use crate::diagnostics::{self, DiagnosticsState};
use crate::preflight::{DesktopPreflightDisposition, DesktopPreflightResult};
use crate::process::BackendState;
use log::{info, warn};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

static DESKTOP_AUTH_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[derive(Debug, Serialize)]
pub struct DesktopAuthResponse {
    pub access_token: String,
    pub refresh_token: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct DesktopAuthRequest {
    secret: String,
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PortSource {
    Cached,
    Discovered,
}

#[derive(Clone, Copy, Debug)]
struct BackendPort {
    port: u16,
    source: PortSource,
}

#[derive(Debug)]
enum AuthError {
    Connectivity(String),
    StaleResponder(String),
    Failed(String),
}

impl AuthError {
    fn message(self) -> String {
        match self {
            Self::Connectivity(message) | Self::StaleResponder(message) | Self::Failed(message) => {
                message
            }
        }
    }
}

fn auth_secret_path(home: &Path, filename: &str) -> PathBuf {
    home.join(".unsloth")
        .join("studio")
        .join("auth")
        .join(filename)
}

fn auth_url(port: u16, route: &str) -> String {
    format!("http://127.0.0.1:{port}/api/auth/{route}")
}

fn home_dir() -> Result<PathBuf, String> {
    dirs::home_dir().ok_or_else(|| "Could not determine home directory".to_string())
}

fn desktop_secret_path() -> Result<PathBuf, String> {
    Ok(auth_secret_path(&home_dir()?, ".desktop_secret"))
}

fn read_secret_if_exists(path: &Path) -> Result<Option<String>, String> {
    match std::fs::read_to_string(path) {
        Ok(s) => Ok(Some(s.trim().to_string())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => {
            warn!(
                "Desktop auth: ignoring unreadable auth secret at {}: {}",
                path.display(),
                e
            );
            Ok(None)
        }
    }
}

async fn current_backend_port(
    state: &tauri::State<'_, BackendState>,
) -> Result<BackendPort, String> {
    let cached_port = {
        let proc = state.lock().map_err(|e| e.to_string())?;
        if let Some(port) = proc.owned_backend_port() {
            return Ok(BackendPort {
                port,
                source: PortSource::Cached,
            });
        }
        if proc.has_owned_backend() {
            None
        } else {
            proc.port
        }
    };

    if let Some(port) = cached_port {
        return Ok(BackendPort {
            port,
            source: PortSource::Cached,
        });
    }

    if state
        .lock()
        .map(|proc| proc.has_owned_backend())
        .map_err(|e| e.to_string())?
    {
        return Err("Backend is not ready".to_string());
    }

    let port = discover_compatible_backend_port()
        .await
        .ok_or_else(|| "Backend is not ready".to_string())?;

    {
        let mut proc = state.lock().map_err(|e| e.to_string())?;
        if proc.port.is_none() {
            proc.port = Some(port);
        }
    }

    Ok(BackendPort {
        port,
        source: PortSource::Discovered,
    })
}

fn attached_ready_port(preflight: DesktopPreflightResult) -> Option<u16> {
    if preflight.disposition == DesktopPreflightDisposition::AttachedReady {
        preflight.port
    } else {
        None
    }
}

async fn discover_compatible_backend_port() -> Option<u16> {
    attached_ready_port(crate::preflight::desktop_preflight_result().await)
}

fn update_backend_port(state: &tauri::State<'_, BackendState>, port: u16) -> Result<(), String> {
    let mut proc = state.lock().map_err(|e| e.to_string())?;
    proc.port = Some(port);
    Ok(())
}

fn classify_auth_send_error(error: reqwest::Error) -> AuthError {
    let message = format!("Desktop auth failed: {}", error);
    if error.is_connect() || error.is_timeout() {
        AuthError::Connectivity(message)
    } else {
        AuthError::Failed(message)
    }
}

fn should_retry_with_discovered_port(source: PortSource, error: &AuthError) -> bool {
    matches!(
        (source, error),
        (
            PortSource::Cached,
            AuthError::Connectivity(_) | AuthError::StaleResponder(_)
        )
    )
}

fn can_retry_on_discovered_port(state: &BackendState, source: PortSource) -> Result<bool, String> {
    if source != PortSource::Cached {
        return Ok(false);
    }
    state
        .lock()
        .map(|proc| !proc.has_owned_backend())
        .map_err(|e| e.to_string())
}

async fn exchange_desktop_secret(
    client: &Client,
    port: u16,
    secret: &str,
) -> Result<Option<DesktopAuthResponse>, AuthError> {
    let response = client
        .post(auth_url(port, "desktop-login"))
        .json(&DesktopAuthRequest {
            secret: secret.to_string(),
        })
        .send()
        .await
        .map_err(classify_auth_send_error)?;

    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(AuthError::StaleResponder(
            "Running Studio backend is too old for this desktop app. Update that backend and restart."
                .to_string(),
        ));
    }
    if response.status() == reqwest::StatusCode::UNAUTHORIZED {
        return Ok(None);
    }
    if !response.status().is_success() {
        return Err(AuthError::StaleResponder("Desktop auth failed".to_string()));
    }

    response
        .json::<TokenResponse>()
        .await
        .map(|tokens| {
            Some(DesktopAuthResponse {
                access_token: tokens.access_token,
                refresh_token: tokens.refresh_token,
            })
        })
        .map_err(|e| AuthError::Failed(format!("Desktop auth failed: {}", e)))
}

async fn provision_desktop_auth() -> Result<(), String> {
    let bin = crate::process::resolve_backend_binary()?;
    let mut cmd = tokio::process::Command::new(&bin);
    cmd.args(["studio", "provision-desktop-auth"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    #[cfg(target_os = "linux")]
    if std::env::var_os("APPIMAGE").is_some() {
        cmd.env_remove("LD_LIBRARY_PATH");
        cmd.env_remove("PYTHONHOME");
        cmd.env_remove("PYTHONPATH");
    }

    // Tauri uses the legacy root regardless of UNSLOTH_STUDIO_HOME / STUDIO_HOME.
    // Scrub so provisioning writes match what the Rust auth code reads.
    cmd.env_remove("UNSLOTH_STUDIO_HOME");
    cmd.env_remove("STUDIO_HOME");
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(crate::process::CREATE_NO_WINDOW);
    }

    let output = tokio::time::timeout(std::time::Duration::from_secs(30), cmd.output())
        .await
        .map_err(|_| "Desktop auth provisioning timed out after 30s".to_string())?
        .map_err(|e| format!("Desktop auth provisioning failed: {}", e))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(format!(
        "Desktop auth provisioning failed: {}",
        stderr.trim()
    ))
}

async fn retry_on_discovered_port(
    client: &Client,
    state: &tauri::State<'_, BackendState>,
    previous: BackendPort,
    secret: &str,
) -> Result<Option<(Option<DesktopAuthResponse>, BackendPort)>, String> {
    if !can_retry_on_discovered_port(state.inner(), previous.source)? {
        return Ok(None);
    }
    let Some(port) = discover_compatible_backend_port().await else {
        return Ok(None);
    };
    if port == previous.port {
        return Ok(None);
    }

    update_backend_port(state, port)?;
    let backend = BackendPort {
        port,
        source: PortSource::Discovered,
    };
    exchange_desktop_secret(client, port, secret)
        .await
        .map(|tokens| Some((tokens, backend)))
        .map_err(AuthError::message)
}

async fn authenticate_with_stale_port_retry(
    client: &Client,
    state: &tauri::State<'_, BackendState>,
    backend: BackendPort,
    secret: &str,
) -> Result<(Option<DesktopAuthResponse>, BackendPort), String> {
    match exchange_desktop_secret(client, backend.port, secret).await {
        Ok(Some(tokens)) => Ok((Some(tokens), backend)),
        Ok(None) => {
            // 401 means the backend is reachable but the cached secret is stale.
            // Keep using the same backend so the next desktop_auth_inner attempt
            // provisions a new secret for the server we are retrying instead of
            // switching to an unrelated discovered/attached backend.
            Ok((None, backend))
        }
        Err(error) if should_retry_with_discovered_port(backend.source, &error) => {
            if let Some(retried) = retry_on_discovered_port(client, state, backend, secret).await? {
                return Ok(retried);
            }
            Err(error.message())
        }
        Err(error) => Err(error.message()),
    }
}

#[tauri::command]
pub async fn desktop_auth(
    state: tauri::State<'_, BackendState>,
    diagnostics: tauri::State<'_, DiagnosticsState>,
) -> Result<DesktopAuthResponse, String> {
    let result = desktop_auth_inner(&state, diagnostics.inner()).await;
    if let Err(message) = &result {
        let port = state.lock().ok().and_then(|proc| proc.port);
        diagnostics::record_auth_failure(&diagnostics, "desktop_auth", port, message);
    }
    result
}

async fn desktop_auth_inner(
    state: &tauri::State<'_, BackendState>,
    diagnostics: &DiagnosticsState,
) -> Result<DesktopAuthResponse, String> {
    let _auth_guard = DESKTOP_AUTH_LOCK.lock().await;
    let mut backend = current_backend_port(state).await?;
    if backend.source == PortSource::Discovered {
        diagnostics::record_attached_external_backend(diagnostics, backend.port);
    }
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(|e| format!("Desktop auth failed: {}", e))?;

    for attempt in 0..2 {
        if attempt == 1 {
            info!("Desktop auth: provisioning local desktop secret");
            provision_desktop_auth().await?;
        }

        let path = desktop_secret_path()?;
        let Some(secret) = read_secret_if_exists(&path)? else {
            continue;
        };

        info!("Desktop auth: exchanging desktop secret");
        let (tokens, resolved_backend) =
            authenticate_with_stale_port_retry(&client, state, backend, &secret).await?;
        backend = resolved_backend;
        if backend.source == PortSource::Discovered {
            diagnostics::record_attached_external_backend(diagnostics, backend.port);
        }
        if let Some(tokens) = tokens {
            return Ok(tokens);
        }
    }

    Err(
        "Desktop auth failed. Update or repair the managed Studio install, then restart Studio."
            .to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    async fn login_server(status: &str) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let status = status.to_string();

        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buffer = [0; 1024];
            let _ = stream.read(&mut buffer).await.unwrap();
            let response =
                format!("HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            stream.write_all(response.as_bytes()).await.unwrap();
        });

        port
    }

    #[test]
    fn retry_discovery_only_for_cached_recoverable_errors() {
        assert!(should_retry_with_discovered_port(
            PortSource::Cached,
            &AuthError::Connectivity("connection refused".to_string())
        ));
        assert!(should_retry_with_discovered_port(
            PortSource::Cached,
            &AuthError::StaleResponder("old responder".to_string())
        ));
        assert!(!should_retry_with_discovered_port(
            PortSource::Discovered,
            &AuthError::Connectivity("connection refused".to_string())
        ));
        assert!(!should_retry_with_discovered_port(
            PortSource::Cached,
            &AuthError::Failed("Desktop auth failed".to_string())
        ));
    }

    #[test]
    fn owned_handle_disables_discovered_auth_retry() {
        const ROOT_ID: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        const TOKEN: &str = "desktop-owner-token";

        let state = crate::process::new_backend_state();
        assert!(can_retry_on_discovered_port(&state, PortSource::Cached).unwrap());
        assert!(!can_retry_on_discovered_port(&state, PortSource::Discovered).unwrap());

        let owner = crate::desktop_backend_owner::test_owner_state(ROOT_ID, TOKEN, 8890);
        state.lock().unwrap().owned = Some(crate::process::OwnedBackendHandle::adopted(
            owner, 8890, 2, 3,
        ));

        assert!(!can_retry_on_discovered_port(&state, PortSource::Cached).unwrap());
    }

    #[test]
    fn read_secret_handles_missing_trimmed_and_invalid_files() {
        let base =
            std::env::temp_dir().join(format!("unsloth-desktop-secret-{}", std::process::id()));
        let missing = base.with_extension("missing");
        let trimmed = base.with_extension("trimmed");
        let invalid = base.with_extension("invalid");
        let _ = std::fs::remove_file(&missing);
        std::fs::write(&trimmed, "  desktop-secret\n").unwrap();
        std::fs::write(&invalid, [0xff, 0xfe]).unwrap();

        assert_eq!(read_secret_if_exists(&missing).unwrap(), None);
        assert_eq!(
            read_secret_if_exists(&trimmed).unwrap(),
            Some("desktop-secret".to_string())
        );
        assert_eq!(read_secret_if_exists(&invalid).unwrap(), None);
        let _ = std::fs::remove_file(trimmed);
        let _ = std::fs::remove_file(invalid);
    }

    #[tokio::test]
    async fn exchange_desktop_secret_handles_unauthorized_and_not_found() {
        let port = login_server("401 Unauthorized").await;
        let tokens = exchange_desktop_secret(&Client::new(), port, "desktop-stale")
            .await
            .unwrap();
        assert!(tokens.is_none());

        let port = login_server("404 Not Found").await;
        let error = exchange_desktop_secret(&Client::new(), port, "desktop-secret")
            .await
            .unwrap_err()
            .message();
        assert_eq!(
            error,
            "Running Studio backend is too old for this desktop app. Update that backend and restart."
        );
    }
}
