use crate::commands;
use crate::process::BackendState;
use log::{info, warn};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

static DESKTOP_AUTH_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[derive(Debug, Serialize)]
pub struct DesktopAuthResponse {
    pub access_token: String,
    pub refresh_token: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct LoginRequest {
    username: String,
    password: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChangePasswordRequest {
    current_password: String,
    new_password: String,
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: String,
    must_change_password: bool,
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

fn auth_password_path(home: &Path, filename: &str) -> PathBuf {
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

fn desktop_password_path() -> Result<PathBuf, String> {
    Ok(auth_password_path(&home_dir()?, ".desktop_password"))
}

fn bootstrap_password_path() -> Result<PathBuf, String> {
    Ok(auth_password_path(&home_dir()?, ".bootstrap_password"))
}

fn read_password(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path)
        .map(|s| s.trim().to_string())
        .map_err(|e| format!("Failed to read password at {}: {}", path.display(), e))
}

fn read_password_if_exists(path: &Path) -> Result<Option<String>, String> {
    match std::fs::read_to_string(path) {
        Ok(s) => Ok(Some(s.trim().to_string())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!(
            "Failed to read password at {}: {}",
            path.display(),
            e
        )),
    }
}

fn generate_desktop_password() -> String {
    (0..64)
        .map(|_| {
            let idx = rand::random_range(0..62u8);
            let c = match idx {
                0..26 => b'a' + idx,
                26..52 => b'A' + (idx - 26),
                52..62 => b'0' + (idx - 52),
                _ => unreachable!(),
            };
            c as char
        })
        .collect()
}

fn write_desktop_password(password: &str) -> Result<(), String> {
    let path = desktop_password_path()?;
    write_desktop_password_to_path(&path, password)
}

fn write_desktop_password_to_path(path: &Path, password: &str) -> Result<(), String> {
    let auth_dir = path
        .parent()
        .ok_or_else(|| "Failed to resolve auth directory".to_string())?;

    std::fs::create_dir_all(auth_dir)
        .map_err(|e| format!("Failed to create auth directory: {}", e))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        use std::os::unix::fs::PermissionsExt;

        if path.exists() {
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
                .map_err(|e| format!("Failed to set permissions: {}", e))?;
        }

        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)
            .map_err(|e| format!("Failed to write desktop password: {}", e))?;
        file.write_all(password.as_bytes())
            .map_err(|e| format!("Failed to write desktop password: {}", e))?;
    }

    #[cfg(not(unix))]
    std::fs::write(path, password.as_bytes())
        .map_err(|e| format!("Failed to write desktop password: {}", e))?;

    Ok(())
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

    let port = commands::find_existing_server()
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

async fn login_with_password(
    client: &Client,
    port: u16,
    password: &str,
) -> Result<Option<TokenResponse>, AuthError> {
    let response = client
        .post(auth_url(port, "login"))
        .json(&LoginRequest {
            username: "unsloth".to_string(),
            password: password.to_string(),
        })
        .send()
        .await
        .map_err(classify_auth_send_error)?;

    if !response.status().is_success() {
        return Ok(None);
    }

    response
        .json::<TokenResponse>()
        .await
        .map(Some)
        .map_err(|e| AuthError::Failed(format!("Desktop auth failed: {}", e)))
}

async fn change_password(
    client: &Client,
    port: u16,
    access_token: &str,
    current_password: &str,
    new_password: &str,
) -> Result<TokenResponse, AuthError> {
    let response = client
        .post(auth_url(port, "change-password"))
        .bearer_auth(access_token)
        .json(&ChangePasswordRequest {
            current_password: current_password.to_string(),
            new_password: new_password.to_string(),
        })
        .send()
        .await
        .map_err(classify_auth_send_error)?;

    if !response.status().is_success() {
        return Err(AuthError::Failed("Desktop auth failed".to_string()));
    }

    response
        .json::<TokenResponse>()
        .await
        .map_err(|e| AuthError::Failed(format!("Desktop auth failed: {}", e)))
}

async fn authenticate_with_password(
    client: &Client,
    port: u16,
    password: &str,
) -> Result<Option<DesktopAuthResponse>, AuthError> {
    let Some(tokens) = login_with_password(client, port, password).await? else {
        return Ok(None);
    };

    if !tokens.must_change_password {
        return Ok(Some(DesktopAuthResponse {
            access_token: tokens.access_token,
            refresh_token: tokens.refresh_token,
        }));
    }

    let new_password = generate_desktop_password();
    write_desktop_password(&new_password).map_err(AuthError::Failed)?;
    let fresh =
        change_password(client, port, &tokens.access_token, password, &new_password).await?;

    Ok(Some(DesktopAuthResponse {
        access_token: fresh.access_token,
        refresh_token: fresh.refresh_token,
    }))
}

async fn authenticate_with_stale_port_retry(
    client: &Client,
    state: &tauri::State<'_, BackendState>,
    backend: BackendPort,
    password: &str,
) -> Result<(Option<DesktopAuthResponse>, BackendPort), String> {
    match authenticate_with_password(client, backend.port, password).await {
        Ok(tokens) => Ok((tokens, backend)),
        Err(error) if should_retry_with_discovered_port(backend.source, &error) => {
            let Some(port) = commands::find_existing_server().await else {
                return Err(error.message());
            };
            update_backend_port(state, port)?;
            let backend = BackendPort {
                port,
                source: PortSource::Discovered,
            };
            authenticate_with_password(client, port, password)
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

    let desktop_path = desktop_password_path()?;
    match read_password_if_exists(&desktop_path) {
        Ok(Some(password)) => {
            info!("Desktop auth: trying stored desktop password");
            let (tokens, resolved_backend) =
                authenticate_with_stale_port_retry(&client, &state, backend, &password).await?;
            backend = resolved_backend;
            if let Some(tokens) = tokens {
                return Ok(tokens);
            }
        }
        Ok(None) => {}
        Err(e) => warn!("{}", e),
    }

    info!("Desktop auth: trying bootstrap password");
    let bootstrap_path = bootstrap_password_path()?;
    match read_password(&bootstrap_path) {
        Ok(password) => {
            match authenticate_with_stale_port_retry(&client, &state, backend, &password)
                .await?
                .0
            {
                Some(tokens) => Ok(tokens),
                None => Err("Desktop auth failed".to_string()),
            }
        }
        Err(e) => {
            warn!("{}", e);
            Err("Desktop auth failed".to_string())
        }
    }
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
    fn auth_password_path_joins_expected_location() {
        let home = PathBuf::from("/home/alex");
        assert_eq!(
            auth_password_path(&home, ".desktop_password"),
            PathBuf::from("/home/alex/.unsloth/studio/auth/.desktop_password")
        );
    }

    #[test]
    fn auth_url_builds_local_endpoint() {
        assert_eq!(
            auth_url(8890, "login"),
            "http://127.0.0.1:8890/api/auth/login"
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

    #[cfg(unix)]
    #[test]
    fn write_desktop_password_to_path_sets_existing_file_0600() {
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join(format!(
            "unsloth-desktop-auth-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(".desktop_password");
        std::fs::write(&path, b"old").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();

        write_desktop_password_to_path(&path, "new").unwrap();

        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn login_with_password_returns_none_for_non_success_http_status() {
        let port = login_server("500 Internal Server Error").await;
        let tokens = login_with_password(&Client::new(), port, "stale-password")
            .await
            .unwrap();

        assert!(tokens.is_none());
    }
}
