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
/// Per-probe HTTP budget for the launcher's liveness probe.
///
/// Generous on purpose. The backend imports torch and transformers on a background warm
/// thread, and those C-extension imports hold the GIL, so the event loop can go silent for
/// seconds at a time on a cold start (3735ms measured on a Mac, with the process quiet for
/// ~27s around it). A 2s budget turns that into a probe timeout, and three of those in a row
/// kill a backend that is merely busy. Kept below HEALTH_WATCHDOG_INTERVAL so a slow probe
/// cannot overlap the next one.
///
/// This is not the preflight budget: `preflight::backend::probe_ownerless_spawned_backend`
/// deliberately keeps its own 2s, because a preflight timeout dead-ends the launch rather
/// than retrying, and `backend/tests/test_health_answers_within_probe_budget.py` derives
/// `_HEALTH_DETECT_BUDGET_S` from that number.
const HEALTH_PROBE_TIMEOUT: Duration = Duration::from_secs(10);

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
            "A desktop-owned Unsloth server for this install is already running on port {}. Quit the other desktop app instance, then try again.",
            conflict.port
        );
    }
    format!(
        "An Unsloth server for this install is already running from a terminal on port {}. Stop that server, or run `unsloth studio update` from that terminal before using desktop repair/update.",
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
    let started = Instant::now();
    let (result, adopted_watchdog_generation) =
        crate::preflight::desktop_preflight_result_with_state(state.inner()).await?;
    diagnostics::record_preflight(&diagnostics, &result);

    info!(
        "desktop_preflight completed disposition={:?} port={:?} in {}ms",
        result.disposition,
        result.port,
        started.elapsed().as_millis()
    );

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

    let started = Instant::now();
    let diagnostics_state = diagnostics.inner().clone();
    let generation = process::start_backend(&app, &state, port, &shutdown, &diagnostics_state)?;

    info!(
        "start_managed_server spawned generation={} in {}ms",
        generation,
        started.elapsed().as_millis()
    );

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

/// What one launcher probe learned about the backend process.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct BackendLiveness {
    /// The port answered with an Unsloth backend reply.
    alive: bool,
    /// The backend answered but has not finished its background warm, so the ML-stack
    /// imports on its warm thread are still in flight. Alive, but not yet done starting.
    warming_up: bool,
}

/// Check if an Unsloth backend is running on the given port.
/// Expects JSON with status=="alive" (or "healthy") AND service=="Unsloth UI Backend".
#[tauri::command]
pub async fn check_health(port: u16) -> Result<bool, String> {
    match check_health_inner(port).await {
        Ok(liveness) => Ok(liveness.alive),
        Err(e) => {
            // Network errors are not command errors — just means not healthy
            info!("Health check on port {} failed: {}", port, e);
            Ok(false)
        }
    }
}

/// Probe the backend for process liveness.
///
/// `/api/liveness` rather than `/api/health`: health awaits hardware detection through
/// `_await_hardware_detection`, so it deliberately bills the probe for work that says
/// nothing about whether the process is alive. Liveness reads module-level caches only,
/// which is the reason it was added. Backends older than that route answer 404, so fall
/// back to health for them, the same order `process::generic_backend_health_ok` and
/// `desktop_backend_owner::fetch_liveness` already use.
async fn check_health_inner(port: u16) -> Result<BackendLiveness, reqwest::Error> {
    let client = crate::loopback_http::client(HEALTH_PROBE_TIMEOUT)?;
    let mut json = None;
    for path in ["/api/liveness", "/api/health"] {
        let resp = client
            .get(format!("http://127.0.0.1:{}{}", port, path))
            .send()
            .await?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND && path == "/api/liveness" {
            continue;
        }
        if !resp.status().is_success() {
            return Ok(BackendLiveness::default());
        }
        json = Some(resp.json::<serde_json::Value>().await?);
        break;
    }
    let Some(json) = json else {
        return Ok(BackendLiveness::default());
    };

    // Liveness answers "alive" and health answers "healthy". Accept either, so the fallback
    // above and a downgraded backend both still validate.
    let live = json
        .get("status")
        .and_then(|v| v.as_str())
        .map(|s| s == "alive" || s == "healthy")
        .unwrap_or(false);
    let correct_service = json
        .get("service")
        .and_then(|v| v.as_str())
        .map(|s| s == "Unsloth UI Backend")
        .unwrap_or(false);
    // Both routes carry `torch_warm_in_progress` while the backend's coordinated warm thread
    // is running, and drop it the moment that thread is done or was never started. That is
    // the whole warm, not just its first stage: hardware detection settles early and the
    // transformers, datasets and unsloth_zoo imports that follow hold the GIL just as hard,
    // so this is the field that says "startup is still in flight".
    let warming = json
        .get("torch_warm_in_progress")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    // `hardware_detecting` is the older, narrower signal, kept as a fallback for backends
    // that predate the field above; on those it is the only warm-up marker there is. It also
    // means "this hardware verdict is provisional", which is why a deferred warm sets
    // `hardware_detection_deferred` alongside it: nothing will ever settle the verdict then,
    // and counting that as warming up would hold the startup grace open until it expired.
    // A backend too old to send any of these reads as settled, which is what this launcher
    // assumed before, so a downgrade loses the extra grace and nothing else.
    let detecting = json
        .get("hardware_detecting")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let deferred = json
        .get("hardware_detection_deferred")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let alive = live && correct_service;
    Ok(BackendLiveness {
        alive,
        warming_up: alive && (warming || (detecting && !deferred)),
    })
}

async fn check_watchdog_health(
    state: &BackendState,
    generation: u64,
    port: u16,
    has_adopted: bool,
) -> BackendLiveness {
    if !has_adopted {
        return check_health_inner(port).await.unwrap_or_default();
    }

    let snapshot = match process::owned_backend_snapshot(state) {
        Ok(Some(snapshot))
            if snapshot.is_adopted
                && snapshot.generation == generation
                && snapshot.port == Some(port) =>
        {
            snapshot
        }
        _ => return BackendLiveness::default(),
    };
    let Some(owner) = snapshot.owner else {
        return BackendLiveness::default();
    };
    let verified = matches!(
        crate::desktop_backend_owner::probe_owned_backend_state(owner, Some(port), false).await,
        crate::desktop_backend_owner::OwnedBackendProbe::Verified(_)
    );
    // The ownership probe carries no warm-up signal, and it does not need one: an adopted
    // backend was already serving before this app attached, so its watchdog starts with
    // count_failures_immediately = true and never consults the startup grace.
    BackendLiveness {
        alive: verified,
        warming_up: false,
    }
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

/// Open an existing directory in the system file manager. Validates the path
/// up front so callers get a clean error instead of a raw OS failure.
fn open_existing_dir_with<E>(
    dir: &std::path::Path,
    opener: impl FnOnce(&std::path::Path) -> Result<(), E>,
) -> Result<(), String>
where
    E: std::fmt::Display,
{
    if !dir.is_dir() {
        return Err(format!("Directory does not exist: {}", dir.display()));
    }
    opener(dir).map_err(|error| format!("Failed to open directory: {error}"))
}

fn open_existing_dir(dir: &std::path::Path) -> Result<(), String> {
    open_existing_dir_with(dir, |path| open::that_detached(path))
}

/// Open the Unsloth directory in the system file manager.
#[tauri::command]
pub fn open_logs_dir(window: tauri::WebviewWindow) -> Result<(), String> {
    crate::native_intents::ensure_main_window(&window)?;
    let home = dirs::home_dir().ok_or("Could not determine home directory")?;
    open_existing_dir(&home.join(".unsloth").join("studio"))
}

/// Open a models directory (resolved by the backend, e.g. the HF cache) in the
/// system file manager.
#[tauri::command]
pub fn open_models_dir(window: tauri::WebviewWindow, path: String) -> Result<(), String> {
    crate::native_intents::ensure_main_window(&window)?;
    open_existing_dir(std::path::Path::new(&path))
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

/// Repair a stale managed Unsloth install.
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

    let _ = app.emit("repair-progress", "Updating existing Unsloth install...");
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
                "Update finished, but Unsloth is still not ready. Running bundled installer...",
            );
        }
        Err(msg) => {
            // A stop is the user quitting or cancelling, not a broken install. Running
            // the installer here rewrites a working venv and leaves it half-built when
            // the app exits underneath it.
            if msg == update::UPDATE_STOPPED {
                info!("Managed repair update stopped; skipping installer fallback");
                // Only a user stop reaches this branch, and the support report prints
                // final_status verbatim. Matches record_pending_elevation_canceled.
                diagnostics::finish_repair_group(
                    &diagnostics_state,
                    &repair_group_id,
                    "canceled",
                    Some(msg.clone()),
                );
                return Err(msg);
            }
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

    let msg = "Repair finished, but Unsloth install is still not desktop-ready.".to_string();
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
    use std::fs;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
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
            r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.8.4","desktop_protocol_version":1,"desktop_manageability_version":1,"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,"studio_root_id":"{ROOT_ID}"{owner}}}"#
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
    fn existing_directory_helper_invokes_opener_and_surfaces_errors() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("unsloth-open-dir-{}-{nanos}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let mut opened = false;
        super::open_existing_dir_with(&dir, |path| {
            opened = true;
            assert_eq!(path, dir);
            Ok::<_, &str>(())
        })
        .unwrap();
        assert!(opened);

        let error = super::open_existing_dir_with(&dir, |_| Err("opener failed")).unwrap_err();
        assert!(error.contains("opener failed"));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn existing_directory_helper_rejects_missing_path_without_opening() {
        let missing = std::env::temp_dir().join("unsloth-definitely-missing-open-dir");
        let error = super::open_existing_dir_with(&missing, |_| {
            panic!("opener must not run for an invalid directory");
            #[allow(unreachable_code)]
            Ok::<_, &str>(())
        })
        .unwrap_err();
        assert!(error.contains("Directory does not exist"));
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

    /// Stub backend for the launcher probe. `liveness` of `None` models a backend older
    /// than the route, which answers 404 there. Returns the port and the paths it was asked
    /// for, so a test can assert which route the probe actually used.
    async fn probe_test_backend(
        liveness: Option<String>,
        health: String,
    ) -> (u16, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("probe test needs a loopback port");
        let port = listener.local_addr().unwrap().port();
        let paths = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&paths);
        tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    return;
                };
                let mut buffer = [0; 2048];
                let Ok(n) = stream.read(&mut buffer).await else {
                    return;
                };
                let request = String::from_utf8_lossy(&buffer[..n]);
                let path = request
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or_default()
                    .to_string();
                recorded.lock().unwrap().push(path.clone());
                let (status, body) = match path.as_str() {
                    "/api/liveness" => match liveness.as_deref() {
                        Some(body) => ("200 OK", body),
                        None => ("404 Not Found", ""),
                    },
                    "/api/health" => ("200 OK", health.as_str()),
                    _ => ("404 Not Found", ""),
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
            }
        });
        (port, paths)
    }

    #[tokio::test]
    async fn liveness_is_probed_instead_of_the_detection_gated_health_route() {
        // /api/health awaits hardware detection on purpose, so probing it bills the
        // watchdog for a torch import. Liveness must be the only route touched.
        let (port, paths) = probe_test_backend(
            Some(r#"{"status":"alive","service":"Unsloth UI Backend"}"#.to_string()),
            ready_health(false),
        )
        .await;

        let liveness = super::check_health_inner(port).await.unwrap();

        assert!(liveness.alive);
        assert!(!liveness.warming_up);
        assert_eq!(paths.lock().unwrap().as_slice(), ["/api/liveness"]);
    }

    #[tokio::test]
    async fn a_backend_without_the_liveness_route_still_validates_through_health() {
        // The desktop app talks to backends of varying versions; a downgrade answers
        // "healthy" from /api/health and 404s liveness.
        let (port, paths) = probe_test_backend(None, ready_health(false)).await;

        let liveness = super::check_health_inner(port).await.unwrap();

        assert!(liveness.alive);
        assert_eq!(
            paths.lock().unwrap().as_slice(),
            ["/api/liveness", "/api/health"]
        );
    }

    #[tokio::test]
    async fn a_warming_backend_is_alive_but_not_finished_starting() {
        let (port, _) = probe_test_backend(
            Some(
                r#"{"status":"alive","service":"Unsloth UI Backend","hardware_detecting":true}"#
                    .to_string(),
            ),
            ready_health(false),
        )
        .await;

        let liveness = super::check_health_inner(port).await.unwrap();

        assert!(liveness.alive);
        assert!(
            liveness.warming_up,
            "an unsettled hardware verdict means the torch import is still in flight"
        );
    }

    #[tokio::test]
    async fn a_late_warm_stage_still_counts_as_warming_up() {
        // The regression: hardware detection is the warm's first stage, so its marker is
        // gone while transformers, datasets and unsloth_zoo are still importing. Ending the
        // startup grace here puts a GIL stall from those imports straight back into the
        // three-strikes count that killed a healthy backend.
        let (port, _) = probe_test_backend(
            Some(
                r#"{"status":"alive","service":"Unsloth UI Backend","torch_warm_in_progress":true}"#
                    .to_string(),
            ),
            ready_health(false),
        )
        .await;

        let liveness = super::check_health_inner(port).await.unwrap();

        assert!(liveness.alive);
        assert!(
            liveness.warming_up,
            "a settled hardware verdict does not mean the warm is over; the imports that \
             hold the GIL longest run after it"
        );
    }

    #[tokio::test]
    async fn a_finished_warm_ends_the_startup_grace() {
        // The other half: the field has to disappear, or the grace never ends on its own
        // and a genuinely hung backend waits out the full five minutes.
        let (port, _) = probe_test_backend(
            Some(
                r#"{"status":"alive","service":"Unsloth UI Backend","torch_warm_in_progress":false}"#
                    .to_string(),
            ),
            ready_health(false),
        )
        .await;

        let liveness = super::check_health_inner(port).await.unwrap();

        assert!(liveness.alive);
        assert!(!liveness.warming_up);
    }

    #[tokio::test]
    async fn a_backend_predating_the_warm_field_still_gets_its_grace() {
        // Backwards compatibility: a downgraded backend sends only hardware_detecting, and
        // it is the only warm-up signal that backend has.
        let (port, _) = probe_test_backend(
            Some(
                r#"{"status":"alive","service":"Unsloth UI Backend","hardware_detecting":true}"#
                    .to_string(),
            ),
            ready_health(false),
        )
        .await;

        let liveness = super::check_health_inner(port).await.unwrap();

        assert!(liveness.warming_up);
    }

    #[tokio::test]
    async fn a_deferred_warm_is_not_reported_as_still_warming_up() {
        // With the warm switched off nothing will ever settle the verdict, so treating it
        // as warming up would hold the startup grace open until it expires on its own.
        let (port, _) = probe_test_backend(
            Some(
                r#"{"status":"alive","service":"Unsloth UI Backend","hardware_detecting":true,"hardware_detection_deferred":true}"#
                    .to_string(),
            ),
            ready_health(false),
        )
        .await;

        let liveness = super::check_health_inner(port).await.unwrap();

        assert!(liveness.alive);
        assert!(!liveness.warming_up);
    }

    #[tokio::test]
    async fn a_foreign_service_on_the_port_is_not_alive() {
        let (port, _) = probe_test_backend(
            Some(r#"{"status":"alive","service":"Some Other App"}"#.to_string()),
            ready_health(false),
        )
        .await;

        assert_eq!(
            super::check_health_inner(port).await.unwrap(),
            super::BackendLiveness::default()
        );
    }

    #[test]
    fn the_probe_budget_fits_inside_one_watchdog_interval() {
        // A probe that outlives the interval would let the next tick start on top of it.
        assert!(super::HEALTH_PROBE_TIMEOUT < super::HEALTH_WATCHDOG_INTERVAL);
    }

    #[test]
    fn the_startup_grace_survives_the_mac_cold_start_timeline() {
        // Replays the macOS report this grace period exists for. The warm thread held the
        // GIL through `import torch`, three probes in a row timed out inside the first
        // minute, and the watchdog SIGTERMed a backend that was starting normally.
        for (label, elapsed) in [
            ("no validated port yet", Duration::from_secs(15)),
            ("probe timeout 1/3", Duration::from_secs(30)),
            ("probe timeout 2/3", Duration::from_secs(47)),
            ("probe timeout 3/3", Duration::from_secs(64)),
        ] {
            assert!(
                !super::should_count_watchdog_failure(false, elapsed),
                "{label} at {}s was counted against a backend still inside the {}s startup grace",
                elapsed.as_secs(),
                super::BACKEND_STARTUP_GRACE_PERIOD.as_secs()
            );
        }
        assert!(!super::should_count_watchdog_failure(
            false,
            super::BACKEND_STARTUP_GRACE_PERIOD - Duration::from_millis(1)
        ));
        assert!(super::should_count_watchdog_failure(
            false,
            super::BACKEND_STARTUP_GRACE_PERIOD
        ));
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
/// but legitimate backend boot is not killed. After the backend has answered a probe
/// that says its warm-up is finished, or after the startup grace expires, 3 consecutive
/// failed checks emit `server-crashed` so the frontend can offer a restart.
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
            should_count_watchdog_failure(has_seen_healthy, started_at.elapsed());

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

        let liveness = check_watchdog_health(&state, generation, port, has_adopted).await;
        if liveness.alive {
            // One answer is proof of life, not proof that startup is over. The backend
            // imports the ML stack on a warm thread and those C-extension imports hold the
            // GIL, so a process that replies now can still miss the next three probes while
            // it is perfectly healthy. Only end the startup grace once the backend reports
            // that whole warm finished, not merely its first (hardware detection) stage.
            if liveness.warming_up {
                info!(
                    "Health watchdog: backend on port {} is alive but still warming up, holding the startup grace period",
                    port
                );
            } else {
                has_seen_healthy = true;
            }
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
