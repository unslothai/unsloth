use crate::install;
use crate::process::{self, BackendState};
use log::{error, info};
use tauri::AppHandle;

/// Check if the unsloth binary is installed in the managed venv.
#[tauri::command]
pub fn check_install_status() -> bool {
    process::find_unsloth_binary().is_some()
}

/// Start the backend server on the given port.
#[tauri::command]
pub async fn start_server(
    app: AppHandle,
    state: tauri::State<'_, BackendState>,
    port: u16,
) -> Result<(), String> {
    info!("start_server command called with port {}", port);
    process::start_backend(&app, &state, port)
}

/// Stop the backend server.
#[tauri::command]
pub fn stop_server(state: tauri::State<'_, BackendState>) -> Result<(), String> {
    info!("stop_server command called");
    process::stop_backend(&state)
}

/// Check if a healthy Unsloth backend is running on the given port.
/// Expects JSON response with status=="healthy" AND service=="Unsloth UI Backend".
#[tauri::command]
pub async fn check_health(port: u16) -> Result<bool, String> {
    match check_health_inner(port).await {
        Ok(healthy) => Ok(healthy),
        Err(e) => {
            // Network errors are not command errors — just means not healthy
            info!("Health check on port {} failed: {}", port, e);
            Ok(false)
        }
    }
}

async fn check_health_inner(port: u16) -> Result<bool, reqwest::Error> {
    let url = format!("http://127.0.0.1:{}/api/health", port);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()?;
    let resp = client.get(&url).send().await?;
    let json: serde_json::Value = resp.json().await?;

    let healthy = json
        .get("status")
        .and_then(|v| v.as_str())
        .map(|s| s == "healthy")
        .unwrap_or(false);
    let correct_service = json
        .get("service")
        .and_then(|v| v.as_str())
        .map(|s| s == "Unsloth UI Backend")
        .unwrap_or(false);

    Ok(healthy && correct_service)
}

/// Scan ports 8888-8907 looking for an existing healthy backend.
/// Range matches the backend's auto-increment range (up to +20 ports in run.py).
#[tauri::command]
pub async fn find_existing_server() -> Option<u16> {
    info!("Scanning ports 8888-8907 for existing backend...");
    for port in 8888..=8907 {
        match check_health_inner(port).await {
            Ok(true) => {
                info!("Found existing backend on port {}", port);
                return Some(port);
            }
            _ => continue,
        }
    }
    info!("No existing backend found");
    None
}

/// Return buffered server logs.
#[tauri::command]
pub fn get_server_logs(state: tauri::State<'_, BackendState>) -> Vec<String> {
    match state.lock() {
        Ok(proc) => proc.logs.iter().cloned().collect(),
        Err(e) => {
            error!("Failed to lock state for logs: {}", e);
            vec![]
        }
    }
}

/// Read the bootstrap password from ~/.unsloth/studio/auth/.bootstrap_password
#[tauri::command]
pub fn get_bootstrap_password() -> Result<String, String> {
    let home = dirs::home_dir().ok_or("Could not determine home directory")?;
    let path = home
        .join(".unsloth")
        .join("studio")
        .join("auth")
        .join(".bootstrap_password");

    std::fs::read_to_string(&path)
        .map(|s| s.trim().to_string())
        .map_err(|e| {
            format!(
                "Failed to read bootstrap password at {}: {}",
                path.display(),
                e
            )
        })
}

/// Open the Unsloth Studio directory in the system file manager.
#[tauri::command]
pub fn open_logs_dir() -> Result<(), String> {
    let home = dirs::home_dir().ok_or("Could not determine home directory")?;
    let dir = home.join(".unsloth").join("studio");

    if !dir.exists() {
        return Err(format!("Directory does not exist: {}", dir.display()));
    }

    open::that(&dir).map_err(|e| format!("Failed to open directory: {}", e))
}

/// Start the first-launch installation process.
/// Runs the platform installer script with --tauri flag and streams progress events.
/// Returns "NEEDS_ELEVATION" if system packages need elevated install (Linux only).
#[tauri::command]
pub async fn start_install(
    app: AppHandle,
    state: tauri::State<'_, install::InstallState>,
) -> Result<(), String> {
    let state = state.inner().clone();
    tokio::task::spawn_blocking(move || install::run_install(app, state))
        .await
        .map_err(|e| format!("Install task panicked: {e}"))?
}

/// Install system packages with elevated permissions (Linux only).
/// Called by frontend after user approves the elevation dialog.
#[cfg(target_os = "linux")]
#[tauri::command]
pub fn install_system_packages(packages: Vec<String>) -> Result<(), String> {
    install::install_system_packages(&packages)
}

/// Stub for non-Linux platforms — elevation is handled by the scripts themselves.
#[cfg(not(target_os = "linux"))]
#[tauri::command]
pub fn install_system_packages(_packages: Vec<String>) -> Result<(), String> {
    Err("Elevated package install is only supported on Linux".to_string())
}
