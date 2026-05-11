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

fn external_conflict_message(conflict: &crate::preflight::ExternalBackendConflict) -> String {
    if conflict.reason == "desktop_owned_backend_active" {
        return format!(
            "A desktop-owned Studio server for this install is already running on port {}. Quit the other desktop app instance, then try again.",
            conflict.port
        );
    }
    format!(
        "A Studio server for this install is already running from a terminal on port {}. Stop that server, or run `unsloth studio update` from that terminal before using desktop repair/update.",
        conflict.port
    )
}

fn owned_backend_port(state: &tauri::State<'_, BackendState>) -> Result<Option<u16>, String> {
    state
        .lock()
        .map(|proc| proc.owned_backend_port())
        .map_err(|e| e.to_string())
}

fn has_owned_backend(state: &tauri::State<'_, BackendState>) -> Result<bool, String> {
    state
        .lock()
        .map(|proc| proc.has_owned_backend())
        .map_err(|e| e.to_string())
}

async fn block_external_conflict(ignored_ports: &[u16]) -> Result<(), String> {
    if let Some(conflict) =
        crate::preflight::mutation_blocking_backend_ignoring(ignored_ports).await
    {
        return Err(external_conflict_message(&conflict));
    }
    Ok(())
}

#[tauri::command]
pub async fn desktop_preflight(
    app: AppHandle,
    state: tauri::State<'_, BackendState>,
    shutdown: tauri::State<'_, ShutdownFlag>,
    diagnostics: tauri::State<'_, DiagnosticsState>,
) -> Result<crate::preflight::DesktopPreflightResult, String> {
    let (result, adopted_watchdog_generation) =
        crate::preflight::desktop_preflight_result_with_state(state.inner()).await?;
    diagnostics::record_preflight(&diagnostics, &result);

    if let Some((generation, newly_adopted)) = adopted_watchdog_generation {
        if newly_adopted {
            if let Some(port) = result.port {
                diagnostics::begin_adopted_backend_session(&diagnostics, port, generation);
            }
        }
        if process::claim_adopted_watchdog_if_current(state.inner(), generation) {
            shutdown.store(false, std::sync::atomic::Ordering::SeqCst);
            let watchdog_state = state.inner().clone();
            let watchdog_shutdown = shutdown.inner().clone();
            let watchdog_diagnostics = diagnostics.inner().clone();
            tokio::spawn(async move {
                health_watchdog(
                    app,
                    watchdog_state,
                    watchdog_shutdown,
                    watchdog_diagnostics,
                    generation,
                    true,
                )
                .await;
            });
        }
    }

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
            false,
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
            false,
        )
        .await;
    });

    Ok(())
}

/// Stop the current desktop-owned backend if this app can safely control it.
#[tauri::command]
pub async fn stop_server(
    state: tauri::State<'_, BackendState>,
    shutdown: tauri::State<'_, ShutdownFlag>,
    diagnostics: tauri::State<'_, DiagnosticsState>,
) -> Result<(), String> {
    info!("stop_server command called");
    let state = state.inner().clone();
    let shutdown = shutdown.inner().clone();
    let diagnostics = diagnostics.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        process::stop_backend(&state, &shutdown, Some(&diagnostics))
    })
    .await
    .map_err(|e| format!("stop backend task failed: {e}"))?
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

async fn check_watchdog_health(
    state: &BackendState,
    generation: u64,
    port: u16,
    has_adopted: bool,
) -> bool {
    if !has_adopted {
        return check_health_inner(port).await.unwrap_or(false);
    }

    let snapshot = match process::owned_backend_snapshot(state) {
        Ok(Some(snapshot))
            if snapshot.is_adopted
                && snapshot.generation == generation
                && snapshot.port == Some(port) =>
        {
            snapshot
        }
        _ => return false,
    };
    let Some(owner) = snapshot.owner else {
        return false;
    };
    matches!(
        crate::desktop_backend_owner::probe_owned_backend_state(owner, Some(port), false).await,
        crate::desktop_backend_owner::OwnedBackendProbe::Verified(_)
    )
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

    if install_state
        .lock()
        .map(|s| s.child.is_some())
        .unwrap_or(false)
    {
        return Err("Cannot update while installation is in progress.".to_string());
    }

    if update_state
        .lock()
        .map(|s| s.child.is_some())
        .unwrap_or(false)
    {
        return Err("Update is already running.".to_string());
    }

    let owned_port = owned_backend_port(&backend_state)?;
    let has_owned = has_owned_backend(&backend_state)?;
    if has_owned {
        if let Some(port) = owned_port {
            block_external_conflict(&[port]).await?;
        }

        info!("Stopping backend before update...");
        process::stop_backend_for_mutation(&backend_state, &shutdown, Some(diagnostics.inner()))?;
        block_external_conflict(&[]).await?;
    } else {
        block_external_conflict(&[]).await?;
    }

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

    let owned_port = owned_backend_port(&backend_state)?;
    let has_owned = has_owned_backend(&backend_state)?;
    if has_owned {
        if let Some(port) = owned_port {
            block_external_conflict(&[port]).await?;
        }

        info!("Stopping backend before repair...");
        process::stop_backend_for_mutation(&backend_state, &shutdown, Some(&diagnostics_state))?;
        block_external_conflict(&[]).await?;
    } else {
        block_external_conflict(&[]).await?;
    }

    let repair_group_id = install::take_pending_repair_group_for_resume(&install_state)
        .unwrap_or_else(|| diagnostics::begin_repair_group(&diagnostics_state));

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

    if let Err(msg) = block_external_conflict(&[]).await {
        diagnostics::finish_repair_group(
            &diagnostics_state,
            &repair_group_id,
            "failed",
            Some(msg.clone()),
        );
        let _ = app.emit("repair-failed", &msg);
        return Err(msg);
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
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    const ROOT_ID: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const OWNER_TOKEN: &str = "desktop-owner-token";

    fn ready_health(include_owner: bool) -> String {
        let owner = if include_owner {
            format!(
                r#", "desktop_owner":{{"kind":"tauri","token_sha256":"{}"}}"#,
                crate::desktop_backend_owner::token_sha256(OWNER_TOKEN)
            )
        } else {
            String::new()
        };
        format!(
            r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.5.3","desktop_protocol_version":1,"desktop_manageability_version":1,"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,"studio_root_id":"{ROOT_ID}"{owner}}}"#
        )
    }

    async fn command_test_backend(health_body: String) -> u16 {
        let mut listener = None;
        for port in 8888u16..=8908 {
            if let Ok(bound) = TcpListener::bind(("127.0.0.1", port)).await {
                listener = Some(bound);
                break;
            }
        }
        let listener = listener.expect("test needs a free desktop preflight port");
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            for _ in 0..2 {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let mut buffer = [0; 2048];
                let Ok(n) = stream.read(&mut buffer).await else {
                    return;
                };
                let request = String::from_utf8_lossy(&buffer[..n]);
                let (status, body) = if request.starts_with("GET /api/health ") {
                    ("200 OK", health_body.as_str())
                } else if request.starts_with("POST /api/auth/desktop-login ") {
                    ("401 Unauthorized", "")
                } else {
                    ("404 Not Found", "")
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
            }
        });
        port
    }

    #[test]
    fn repair_elevation_is_not_a_terminal_repair_failure() {
        assert!(!super::should_emit_repair_failed("NEEDS_ELEVATION"));
        assert!(super::should_emit_repair_failed(
            "Installer exited with code 1"
        ));
    }

    #[tokio::test]
    async fn mutation_guard_blocks_second_external_backend_when_owned_child_is_ignored() {
        crate::desktop_backend_owner::install_test_owner(ROOT_ID, OWNER_TOKEN);
        let owned_port = command_test_backend(ready_health(true)).await;
        let external_port = command_test_backend(ready_health(false)).await;

        let err = super::block_external_conflict(&[owned_port])
            .await
            .expect_err("external non-owned backend should block mutation");

        assert!(err.contains(&format!("port {external_port}")));
        assert!(err.contains("Stop that server"));
    }

    #[test]
    fn watchdog_failure_policy_counts_only_after_health_or_grace_period() {
        for (has_seen_healthy, elapsed, expected) in [
            (
                false,
                super::BACKEND_STARTUP_GRACE_PERIOD - Duration::from_secs(1),
                false,
            ),
            (true, Duration::from_secs(1), true),
            (false, super::BACKEND_STARTUP_GRACE_PERIOD, true),
        ] {
            assert_eq!(
                super::should_count_watchdog_failure(has_seen_healthy, elapsed),
                expected
            );
        }
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
    count_failures_immediately: bool,
) {
    use std::sync::atomic::Ordering;

    let started_at = Instant::now();
    let mut consecutive_failures: u32 = 0;
    let mut has_seen_healthy = count_failures_immediately;

    loop {
        tokio::time::sleep(HEALTH_WATCHDOG_INTERVAL).await;

        if shutdown.load(Ordering::SeqCst) {
            info!("Health watchdog: shutdown flag set, exiting");
            break;
        }

        let (port, has_owned, has_adopted, current_generation) = {
            let proc = match state.lock() {
                Ok(p) => p,
                Err(_) => break,
            };
            (
                proc.port,
                proc.has_owned_backend(),
                proc.has_adopted_backend(),
                proc.generation,
            )
        };

        if current_generation != generation {
            info!("Health watchdog: backend generation changed, exiting");
            break;
        }

        // Stop watching if the backend is gone
        if !has_owned {
            info!("Health watchdog: backend stopped, exiting");
            break;
        }

        let should_count_failure =
            port.is_some() || should_count_watchdog_failure(has_seen_healthy, started_at.elapsed());

        let Some(port) = port else {
            if has_adopted {
                diagnostics::record_backend_watchdog(
                    &diagnostics,
                    generation,
                    "adopted_port_missing",
                );
                error!("Health watchdog: adopted backend lost its port, declaring dead");
                process::clear_adopted_backend_if_current(
                    &state,
                    generation,
                    None,
                    "watchdog adopted port missing",
                );
                let _ = app.emit("server-crashed", ());
                break;
            }
            if !should_count_failure {
                info!("Health watchdog: backend has not reported a validated port yet");
                continue;
            }
            consecutive_failures += 1;
            warn!(
                "Health watchdog: missing validated port failure {}/{}",
                consecutive_failures, HEALTH_WATCHDOG_MAX_FAILURES
            );
            if consecutive_failures >= HEALTH_WATCHDOG_MAX_FAILURES {
                diagnostics::record_backend_watchdog(
                    &diagnostics,
                    generation,
                    "missing_validated_port",
                );
                error!("Health watchdog: backend never reported a validated port, killing and declaring dead");
                let _ = process::stop_backend(&state, &shutdown, Some(&diagnostics));
                let _ = app.emit("server-crashed", ());
                break;
            }
            continue;
        };

        if check_watchdog_health(&state, generation, port, has_adopted).await {
            has_seen_healthy = true;
            consecutive_failures = 0;
        } else if !should_count_failure {
            info!(
                "Health watchdog: startup health check failed on port {} before grace period elapsed",
                port
            );
        } else {
            consecutive_failures += 1;
            warn!(
                "Health watchdog: failure {}/{} on port {}",
                consecutive_failures, HEALTH_WATCHDOG_MAX_FAILURES, port
            );
            if consecutive_failures >= HEALTH_WATCHDOG_MAX_FAILURES {
                diagnostics::record_backend_watchdog(
                    &diagnostics,
                    generation,
                    "unresponsive_health_check",
                );
                if has_adopted {
                    error!(
                        "Health watchdog: adopted backend unresponsive, clearing state and declaring dead"
                    );
                    process::clear_adopted_backend_if_current(
                        &state,
                        generation,
                        Some(port),
                        "watchdog health check failures",
                    );
                } else {
                    error!("Health watchdog: backend unresponsive, killing and declaring dead");
                    // Kill the zombie process so retry can start fresh
                    let _ = process::stop_backend(&state, &shutdown, Some(&diagnostics));
                }
                let _ = app.emit("server-crashed", ());
                break;
            }
        }
    }

    process::clear_adopted_watchdog_if_current(&state, generation);
}
