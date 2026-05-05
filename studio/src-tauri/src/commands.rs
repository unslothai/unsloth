use crate::diagnostics::{self, DiagnosticsState};
use crate::install;
use crate::process::{self, BackendState, ShutdownFlag};
use crate::update;
use log::{error, info, warn};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter};

const BACKEND_STARTUP_GRACE_PERIOD: Duration = Duration::from_secs(5 * 60);
const HEALTH_WATCHDOG_INTERVAL: Duration = Duration::from_secs(15);
const HEALTH_WATCHDOG_MAX_FAILURES: u32 = 3;

fn should_count_watchdog_failure(has_seen_healthy: bool, elapsed_since_start: Duration) -> bool {
    has_seen_healthy || elapsed_since_start >= BACKEND_STARTUP_GRACE_PERIOD
}

async fn managed_install_ready_after_repair() -> bool {
    crate::preflight::managed_install_ready().await
}

fn should_emit_repair_failed(msg: &str) -> bool {
    !msg.contains("NEEDS_ELEVATION")
}

#[tauri::command]
pub async fn desktop_preflight(
    diagnostics: tauri::State<'_, DiagnosticsState>,
) -> Result<crate::preflight::DesktopPreflightResult, String> {
    let result = crate::preflight::desktop_preflight_result().await;
    diagnostics::record_preflight(&diagnostics, &result);
    Ok(result)
}

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

    #[cfg(windows)]
    {
        cmd.creation_flags(crate::process::CREATE_NO_WINDOW);
    }

    // Match the same AppImage env clearing used in process.rs and install.rs,
    // otherwise the probe can fail due to bundled libs even when the install is fine.
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
    diagnostics: tauri::State<'_, DiagnosticsState>,
    port: u16,
) -> Result<(), String> {
    info!("start_server command called with port {}", port);

    let diagnostics_state = diagnostics.inner().clone();
    let generation = process::start_backend(&app, &state, port, &shutdown, &diagnostics_state)?;

    // Spawn health watchdog for the owned backend — detects
    // deadlocks and hangs that stdout-based crash detection misses.
    let watchdog_state = state.inner().clone();
    let watchdog_shutdown = shutdown.inner().clone();
    let watchdog_app = app.clone();
    tokio::spawn(async move {
        health_watchdog(
            watchdog_app,
            watchdog_state,
            watchdog_shutdown,
            diagnostics_state,
            generation,
        )
        .await;
    });

    Ok(())
}

/// Start the managed backend without reusing an existing backend.
#[tauri::command]
pub async fn start_managed_server(
    app: AppHandle,
    state: tauri::State<'_, BackendState>,
    shutdown: tauri::State<'_, ShutdownFlag>,
    diagnostics: tauri::State<'_, DiagnosticsState>,
    port: u16,
) -> Result<(), String> {
    info!("start_managed_server command called with port {}", port);
    let diagnostics_state = diagnostics.inner().clone();
    let generation = process::start_backend(&app, &state, port, &shutdown, &diagnostics_state)?;

    let watchdog_state = state.inner().clone();
    let watchdog_shutdown = shutdown.inner().clone();
    let watchdog_app = app.clone();
    tokio::spawn(async move {
        health_watchdog(
            watchdog_app,
            watchdog_state,
            watchdog_shutdown,
            diagnostics_state,
            generation,
        )
        .await;
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
    diagnostics: tauri::State<'_, DiagnosticsState>,
) -> Result<(), String> {
    info!("stop_server command called");
    process::stop_backend(&state, &shutdown, Some(diagnostics.inner()))
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
    diagnostics: tauri::State<'_, DiagnosticsState>,
) -> Result<(), String> {
    let state = state.inner().clone();
    let diagnostics_state = diagnostics.inner().clone();
    tokio::task::spawn_blocking(move || install::run_install(app, state, diagnostics_state))
        .await
        .map_err(|e| format!("Install task panicked: {e}"))?
}

/// Record that the user canceled a pending system-package elevation flow.
#[tauri::command]
pub fn cancel_pending_elevation(
    state: tauri::State<'_, install::InstallState>,
    diagnostics: tauri::State<'_, DiagnosticsState>,
) -> Result<(), String> {
    let _ = install::record_pending_elevation_canceled(&state, diagnostics.inner());
    Ok(())
}

/// Install system packages with elevated permissions (Linux only).
/// Called by frontend after user approves the elevation dialog.
/// Only allows packages that the install script reported as needed.
#[cfg(target_os = "linux")]
#[tauri::command]
pub fn install_system_packages(
    packages: Vec<String>,
    state: tauri::State<'_, install::InstallState>,
    diagnostics: tauri::State<'_, DiagnosticsState>,
) -> Result<(), String> {
    // Cross-check against the packages the install script actually reported
    let allowed = state
        .lock()
        .map(|s| s.needed_packages.clone())
        .unwrap_or_default();
    for pkg in &packages {
        if !allowed.contains(pkg) {
            return Err(format!(
                "Package '{}' was not requested by the install script",
                pkg
            ));
        }
    }
    install::install_system_packages(&packages, &state, diagnostics.inner())
}

/// Stub for non-Linux platforms — elevation is handled by the scripts themselves.
#[cfg(not(target_os = "linux"))]
#[tauri::command]
pub fn install_system_packages(
    _packages: Vec<String>,
    _state: tauri::State<'_, install::InstallState>,
    _diagnostics: tauri::State<'_, DiagnosticsState>,
) -> Result<(), String> {
    Err("Elevated package install is only supported on Linux".to_string())
}

/// Run backend update: stop server, run `unsloth studio update`, emit progress.
/// Does NOT restart the backend — the frontend handles shell update + relaunch after.
#[tauri::command]
pub async fn start_backend_update(
    app: AppHandle,
    backend_state: tauri::State<'_, BackendState>,
    shutdown: tauri::State<'_, ShutdownFlag>,
    update_state: tauri::State<'_, update::UpdateState>,
    install_state: tauri::State<'_, install::InstallState>,
    diagnostics: tauri::State<'_, DiagnosticsState>,
) -> Result<(), String> {
    info!("start_backend_update command called");

    // Signal the health watchdog to exit immediately, before any guards.
    // This closes the race window where the watchdog could emit server-crashed
    // between our command being called and stop_backend completing.
    shutdown.store(true, std::sync::atomic::Ordering::SeqCst);

    // Guard: reject if install is running
    if install_state
        .lock()
        .map(|s| s.child.is_some())
        .unwrap_or(false)
    {
        return Err("Cannot update while installation is in progress.".to_string());
    }

    // Guard: reject if update is already running
    if update_state
        .lock()
        .map(|s| s.child.is_some())
        .unwrap_or(false)
    {
        return Err("Update is already running.".to_string());
    }

    // Stop backend if running
    if backend_state
        .lock()
        .map(|s| s.child.is_some())
        .unwrap_or(false)
    {
        info!("Stopping backend before update...");
        process::stop_backend(&backend_state, &shutdown, Some(diagnostics.inner()))?;
    }

    // Run update in a blocking thread
    let state = update_state.inner().clone();
    let diagnostics_state = diagnostics.inner().clone();
    tokio::task::spawn_blocking(move || update::run_backend_update(app, state, diagnostics_state))
        .await
        .map_err(|e| format!("Update task panicked: {e}"))?
}

/// Repair a stale managed Studio install.
#[tauri::command]
pub async fn start_managed_repair(
    app: AppHandle,
    backend_state: tauri::State<'_, BackendState>,
    shutdown: tauri::State<'_, ShutdownFlag>,
    update_state: tauri::State<'_, update::UpdateState>,
    install_state: tauri::State<'_, install::InstallState>,
    diagnostics: tauri::State<'_, DiagnosticsState>,
) -> Result<(), String> {
    info!("start_managed_repair command called");

    if install_state
        .lock()
        .map(|s| s.child.is_some())
        .unwrap_or(false)
    {
        return Err("Cannot repair while installation is in progress.".to_string());
    }

    if update_state
        .lock()
        .map(|s| s.child.is_some())
        .unwrap_or(false)
    {
        return Err("Repair is already running.".to_string());
    }

    let diagnostics_state = diagnostics.inner().clone();
    let repair_group_id = install::take_pending_repair_group_for_resume(&install_state)
        .unwrap_or_else(|| diagnostics::begin_repair_group(&diagnostics_state));

    shutdown.store(true, std::sync::atomic::Ordering::SeqCst);

    if backend_state
        .lock()
        .map(|s| s.child.is_some())
        .unwrap_or(false)
    {
        info!("Stopping backend before repair...");
        process::stop_backend(&backend_state, &shutdown, Some(&diagnostics_state))?;
    }

    let _ = app.emit("repair-progress", "Updating existing Studio install...");
    let update_app = app.clone();
    let update_state = update_state.inner().clone();
    let update_diagnostics = diagnostics_state.clone();
    let update_repair_group_id = repair_group_id.clone();
    let update_result = tokio::task::spawn_blocking(move || {
        update::run_backend_update_for_repair(
            update_app,
            update_state,
            update_diagnostics,
            update_repair_group_id,
        )
    })
    .await
    .map_err(|e| format!("Repair update task panicked: {e}"))?;

    match update_result {
        Ok(()) if managed_install_ready_after_repair().await => {
            info!("Managed repair complete after update");
            diagnostics::finish_repair_group(&diagnostics_state, &repair_group_id, "success", None);
            let _ = app.emit("repair-complete", ());
            return Ok(());
        }
        Ok(()) => {
            warn!("Managed repair update finished, but preflight is still not ready; falling back to installer");
            let _ = app.emit(
                "repair-progress",
                "Update finished, but Studio is still not ready. Running bundled installer...",
            );
        }
        Err(msg) => {
            if msg.to_ascii_lowercase().contains("already running") {
                error!("Managed repair update conflict: {}", msg);
                diagnostics::finish_repair_group(
                    &diagnostics_state,
                    &repair_group_id,
                    "failed",
                    Some(msg.clone()),
                );
                let _ = app.emit("repair-failed", &msg);
                return Err(msg);
            }

            warn!(
                "Managed repair update failed, falling back to bundled installer: {}",
                msg
            );
            let _ = app.emit(
                "repair-progress",
                "Update failed. Running bundled installer...",
            );
        }
    }

    let install_app = app.clone();
    let install_state = install_state.inner().clone();
    let install_diagnostics = diagnostics_state.clone();
    let install_repair_group_id = repair_group_id.clone();
    let install_result = tokio::task::spawn_blocking(move || {
        install::run_install_for_repair(
            install_app,
            install_state,
            install_diagnostics,
            install_repair_group_id,
        )
    })
    .await
    .map_err(|e| format!("Repair install task panicked: {e}"))?;

    if let Err(msg) = install_result {
        diagnostics::finish_repair_group(
            &diagnostics_state,
            &repair_group_id,
            if msg == "NEEDS_ELEVATION" {
                "needs_elevation"
            } else {
                "failed"
            },
            Some(msg.clone()),
        );
        if should_emit_repair_failed(&msg) {
            error!("Managed repair installer failed: {}", msg);
            let _ = app.emit("repair-failed", &msg);
        }
        return Err(msg);
    }

    if managed_install_ready_after_repair().await {
        info!("Managed repair complete after installer");
        diagnostics::finish_repair_group(&diagnostics_state, &repair_group_id, "success", None);
        let _ = app.emit("repair-complete", ());
        return Ok(());
    }

    let msg = "Repair finished, but Studio install is still not desktop-ready.".to_string();
    error!("{}", msg);
    diagnostics::finish_repair_group(
        &diagnostics_state,
        &repair_group_id,
        "failed",
        Some(msg.clone()),
    );
    let _ = app.emit("repair-failed", &msg);
    Err(msg)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    #[test]
    fn repair_elevation_is_not_a_terminal_repair_failure() {
        assert!(!super::should_emit_repair_failed("NEEDS_ELEVATION"));
        assert!(super::should_emit_repair_failed(
            "Installer exited with code 1"
        ));
    }

    #[test]
    fn watchdog_ignores_startup_failures_within_grace_period() {
        assert!(!super::should_count_watchdog_failure(
            false,
            super::BACKEND_STARTUP_GRACE_PERIOD - Duration::from_secs(1)
        ));
    }

    #[test]
    fn watchdog_counts_failures_after_backend_was_healthy() {
        assert!(super::should_count_watchdog_failure(
            true,
            Duration::from_secs(1)
        ));
    }

    #[test]
    fn watchdog_counts_startup_failures_after_grace_period() {
        assert!(super::should_count_watchdog_failure(
            false,
            super::BACKEND_STARTUP_GRACE_PERIOD
        ));
    }
}

/// Periodic health check that detects deadlocked or hung backends.
/// During startup, failures are ignored for a generous grace period so a slow
/// but legitimate backend boot is not killed. After the backend has answered at
/// least once, or after the startup grace expires, 3 consecutive failed checks
/// emit `server-crashed` so the frontend can offer a restart.
async fn health_watchdog(
    app: AppHandle,
    state: BackendState,
    shutdown: ShutdownFlag,
    diagnostics: DiagnosticsState,
    generation: u64,
) {
    use std::sync::atomic::Ordering;

    let started_at = Instant::now();
    let mut consecutive_failures: u32 = 0;
    let mut has_seen_healthy = false;

    loop {
        tokio::time::sleep(HEALTH_WATCHDOG_INTERVAL).await;

        if shutdown.load(Ordering::SeqCst) {
            info!("Health watchdog: shutdown flag set, exiting");
            break;
        }

        let (port, has_child, current_generation) = {
            let proc = match state.lock() {
                Ok(p) => p,
                Err(_) => break,
            };
            (proc.port, proc.child.is_some(), proc.generation)
        };

        if current_generation != generation {
            info!("Health watchdog: backend generation changed, exiting");
            break;
        }

        // Stop watching if the backend is gone
        if !has_child {
            info!("Health watchdog: backend stopped, exiting");
            break;
        }

        let should_count_failure =
            should_count_watchdog_failure(has_seen_healthy, started_at.elapsed());

        let Some(port) = port else {
            if should_count_failure {
                consecutive_failures += 1;
                warn!(
                    "Health watchdog: backend has not reported a port ({}/{})",
                    consecutive_failures, HEALTH_WATCHDOG_MAX_FAILURES
                );
            }

            if consecutive_failures >= HEALTH_WATCHDOG_MAX_FAILURES {
                error!(
                    "Health watchdog: backend never reported a port, killing and declaring dead"
                );
                diagnostics::record_backend_watchdog(
                    &diagnostics,
                    generation,
                    "no_port_after_grace",
                );
                let _ = process::stop_backend(&state, &shutdown, Some(&diagnostics));
                let _ = app.emit("server-crashed", ());
                break;
            }

            continue;
        };

        match check_health_inner(port).await {
            Ok(true) => {
                has_seen_healthy = true;
                consecutive_failures = 0;
            }
            _ if !should_count_failure => {
                info!(
                    "Health watchdog: startup health check failed on port {} before grace period elapsed",
                    port
                );
            }
            _ => {
                consecutive_failures += 1;
                warn!(
                    "Health watchdog: failure {}/{} on port {}",
                    consecutive_failures, HEALTH_WATCHDOG_MAX_FAILURES, port
                );
                if consecutive_failures >= HEALTH_WATCHDOG_MAX_FAILURES {
                    error!("Health watchdog: backend unresponsive, killing and declaring dead");
                    diagnostics::record_backend_watchdog(
                        &diagnostics,
                        generation,
                        "unresponsive_health_check",
                    );
                    // Kill the zombie process so retry can start fresh
                    let _ = process::stop_backend(&state, &shutdown, Some(&diagnostics));
                    let _ = app.emit("server-crashed", ());
                    break;
                }
            }
        }
    }
}
