use crate::install;
use crate::process::{self, BackendState, ShutdownFlag};
use log::{error, info, warn};
use tauri::{AppHandle, Emitter, Manager};

/// Check if unsloth is installed AND functional.
/// Runs `unsloth -h` to verify the import chain works — a partial install
/// (binary exists but deps missing) will fail on import and return false,
/// which sends the user to the install screen for a clean re-install.
#[tauri::command]
pub async fn check_install_status() -> bool {
    let Some(bin) = process::find_unsloth_binary() else {
        return false;
    };

    let mut cmd = tokio::process::Command::new(&bin);
    cmd.arg("-h")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    // Match the same AppImage env clearing used in process.rs and install.rs,
    // otherwise the probe can fail due to bundled libs even when the install is fine.
    #[cfg(target_os = "linux")]
    if std::env::var_os("APPIMAGE").is_some() || std::env::var_os("APPDIR").is_some() {
        cmd.env_remove("LD_LIBRARY_PATH");
        cmd.env_remove("PYTHONHOME");
        cmd.env_remove("PYTHONPATH");
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            warn!("Install check: failed to spawn {:?}: {}", bin, e);
            return false;
        }
    };

    match tokio::time::timeout(std::time::Duration::from_secs(10), child.wait()).await {
        Ok(Ok(status)) => {
            let ok = status.success();
            if !ok {
                warn!("Install check: `unsloth -h` exited with {}", status);
            }
            ok
        }
        Ok(Err(e)) => {
            warn!("Install check: wait failed: {}", e);
            false
        }
        Err(_) => {
            warn!("Install check: `unsloth -h` timed out after 10s");
            let _ = child.kill().await;
            false
        }
    }
}

/// Start the backend server on the given port.
/// Also spawns a health watchdog that monitors the backend and emits
/// `server-crashed` if it becomes unresponsive (deadlock, OOM, etc.).
#[tauri::command]
pub async fn start_server(
    app: AppHandle,
    state: tauri::State<'_, BackendState>,
    shutdown: tauri::State<'_, ShutdownFlag>,
    port: u16,
) -> Result<(), String> {
    info!("start_server command called with port {}", port);
    process::start_backend(&app, &state, port, &shutdown)?;

    // Spawn health watchdog — detects deadlocks and hangs that stdout-based
    // crash detection misses (process alive, pipe open, but not responding).
    let watchdog_state = state.inner().clone();
    let watchdog_shutdown = shutdown.inner().clone();
    let watchdog_app = app.clone();
    tokio::spawn(async move {
        health_watchdog(watchdog_app, watchdog_state, watchdog_shutdown).await;
    });

    Ok(())
}

/// Stop the backend server.
/// Sends SIGTERM to the process group, which triggers uvicorn's graceful
/// shutdown (same codepath as /api/shutdown). Falls back to SIGKILL after 5s.
#[tauri::command]
pub fn stop_server(
    state: tauri::State<'_, BackendState>,
    shutdown: tauri::State<'_, ShutdownFlag>,
) -> Result<(), String> {
    info!("stop_server command called");
    process::stop_backend(&state, &shutdown)
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

/// Scan ports 8888-8908 looking for an existing healthy backend.
/// Range matches run.py: initial port 8888, then _find_free_port(8889, max_attempts=20)
/// which can reach 8908.
#[tauri::command]
pub async fn find_existing_server() -> Option<u16> {
    info!("Scanning ports 8888-8908 for existing backend...");
    for port in 8888..=8908 {
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

/// Periodic health check that detects deadlocked or hung backends.
/// Starts 30s after the backend is launched (to allow initial startup),
/// then pings /api/health every 15s. After 3 consecutive failures (45s)
/// with the process still alive, emits `server-crashed` so the frontend
/// can offer a restart.
async fn health_watchdog(app: AppHandle, state: BackendState, shutdown: ShutdownFlag) {
    use std::sync::atomic::Ordering;

    // Give the backend time to start up
    tokio::time::sleep(std::time::Duration::from_secs(30)).await;

    let mut consecutive_failures: u32 = 0;

    loop {
        tokio::time::sleep(std::time::Duration::from_secs(15)).await;

        if shutdown.load(Ordering::SeqCst) {
            info!("Health watchdog: shutdown flag set, exiting");
            break;
        }

        let (port, has_child) = {
            let proc = match state.lock() {
                Ok(p) => p,
                Err(_) => break,
            };
            (proc.port, proc.child.is_some())
        };

        // Stop watching if the backend is gone
        if !has_child {
            info!("Health watchdog: backend stopped, exiting");
            break;
        }

        let Some(port) = port else {
            continue; // Port not yet known
        };

        match check_health_inner(port).await {
            Ok(true) => {
                consecutive_failures = 0;
            }
            _ => {
                consecutive_failures += 1;
                warn!(
                    "Health watchdog: failure {}/3 on port {}",
                    consecutive_failures, port
                );
                if consecutive_failures >= 3 {
                    error!("Health watchdog: backend unresponsive for 45s, killing and declaring dead");
                    // Kill the zombie process so retry can start fresh
                    let _ = process::stop_backend(&state, &shutdown);
                    let _ = app.emit("server-crashed", ());
                    break;
                }
            }
        }
    }
}

