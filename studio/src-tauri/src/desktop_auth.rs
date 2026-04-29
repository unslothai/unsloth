use crate::preflight::{DesktopPreflightDisposition, DesktopPreflightResult};
use crate::process::BackendState;
use log::info;
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
    Failed(String),
}

impl AuthError {
    fn message(self) -> String {
        match self {
            Self::Connectivity(message) | Self::Failed(message) => message,
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
        Err(e) => Err(format!(
            "Failed to read auth secret at {}: {}",
            path.display(),
            e
        )),
    }
}

async fn current_backend_port(
    state: &tauri::State<'_, BackendState>,
) -> Result<BackendPort, String> {
    if let Some(port) = state.lock().map_err(|e| e.to_string())?.port {
        return Ok(BackendPort {
            port,
            source: PortSource::Cached,
        });
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
        (PortSource::Cached, AuthError::Connectivity(_))
    )
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
        return Err(AuthError::Failed(
            "Running Studio backend is too old for this desktop app. Update that backend and restart."
                .to_string(),
        ));
    }
    if response.status() == reqwest::StatusCode::UNAUTHORIZED {
        return Ok(None);
    }
    if !response.status().is_success() {
        return Err(AuthError::Failed("Desktop auth failed".to_string()));
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

async fn authenticate_with_stale_port_retry(
    client: &Client,
    state: &tauri::State<'_, BackendState>,
    backend: BackendPort,
    secret: &str,
) -> Result<(Option<DesktopAuthResponse>, BackendPort), String> {
    match exchange_desktop_secret(client, backend.port, secret).await {
        Ok(tokens) => Ok((tokens, backend)),
        Err(error) if should_retry_with_discovered_port(backend.source, &error) => {
            let Some(port) = discover_compatible_backend_port().await else {
                return Err(error.message());
            };
            update_backend_port(state, port)?;
            let backend = BackendPort {
                port,
                source: PortSource::Discovered,
            };
            exchange_desktop_secret(client, port, secret)
                .await
                .map(|tokens| (tokens, backend))
                .map_err(AuthError::message)
        }
        Err(error) => Err(error.message()),
    }
}

#[tauri::command]
pub async fn desktop_auth(
    state: tauri::State<'_, BackendState>,
) -> Result<DesktopAuthResponse, String> {
    let _auth_guard = DESKTOP_AUTH_LOCK.lock().await;
    let mut backend = current_backend_port(&state).await?;
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
            authenticate_with_stale_port_retry(&client, &state, backend, &secret).await?;
        backend = resolved_backend;
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
    fn auth_secret_path_joins_expected_location() {
        let home = PathBuf::from("/home/alex");
        assert_eq!(
            auth_secret_path(&home, ".desktop_secret"),
            PathBuf::from("/home/alex/.unsloth/studio/auth/.desktop_secret")
        );
    }

    #[test]
    fn auth_url_builds_local_endpoint() {
        assert_eq!(
            auth_url(8890, "desktop-login"),
            "http://127.0.0.1:8890/api/auth/desktop-login"
        );
    }

    #[test]
    fn retry_discovery_only_for_cached_connectivity_errors() {
        assert!(should_retry_with_discovered_port(
            PortSource::Cached,
            &AuthError::Connectivity("connection refused".to_string())
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
    fn attached_ready_port_requires_attached_ready_with_port() {
        let compatible = DesktopPreflightResult {
            disposition: DesktopPreflightDisposition::AttachedReady,
            reason: None,
            port: Some(8890),
            can_auto_repair: false,
            managed_bin: None,
        };
        assert_eq!(attached_ready_port(compatible), Some(8890));

        let missing_port = DesktopPreflightResult {
            disposition: DesktopPreflightDisposition::AttachedReady,
            reason: None,
            port: None,
            can_auto_repair: false,
            managed_bin: None,
        };
        assert_eq!(attached_ready_port(missing_port), None);

        let managed_ready = DesktopPreflightResult {
            disposition: DesktopPreflightDisposition::ManagedReady,
            reason: None,
            port: Some(8890),
            can_auto_repair: false,
            managed_bin: None,
        };
        assert_eq!(attached_ready_port(managed_ready), None);
    }

    #[tokio::test]
    async fn exchange_desktop_secret_returns_none_for_unauthorized() {
        let port = login_server("401 Unauthorized").await;
        let tokens = exchange_desktop_secret(&Client::new(), port, "desktop-stale")
            .await
            .unwrap();

        assert!(tokens.is_none());
    }

    #[tokio::test]
    async fn exchange_desktop_secret_reports_unsupported_backend_on_not_found() {
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
