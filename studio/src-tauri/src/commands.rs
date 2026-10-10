use crate::diagnostics::{self, DiagnosticsState};
use crate::install;
use crate::process::{self, BackendState, ShutdownFlag};
use crate::update;
use log::{error, info, warn};
use std::net::{Ipv4Addr, SocketAddr};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter};

const BACKEND_STARTUP_GRACE_PERIOD: Duration = Duration::from_secs(5 * 60);
const HEALTH_WATCHDOG_INTERVAL: Duration = Duration::from_secs(15);
const HEALTH_WATCHDOG_MAX_FAILURES: u32 = 3;
/// Busy budget: a timed-out strike costs ~25s (sleep + probe), so 12 strikes is ~300s.
/// Only stalls after a busy answer get it; refused connections use the plain budget.
const HEALTH_WATCHDOG_MAX_FAILURES_BUSY: u32 = 12;
/// Generous: warm-thread imports hold the GIL for seconds. Must stay below HEALTH_WATCHDOG_INTERVAL.
/// Not the preflight budget, which stays 2s (test_health_answers_within_probe_budget.py).
const HEALTH_PROBE_TIMEOUT: Duration = Duration::from_secs(10);
/// How long a loopback connect gets to be refused. Windows retransmits the SYN (~2030ms first),
/// so its window is (2030, 2500], capped at a quarter of HEALTH_PROBE_TIMEOUT.
#[cfg(windows)]
const REFUSAL_PROBE_TIMEOUT: Duration = Duration::from_millis(2_500);
#[cfg(not(windows))]
const REFUSAL_PROBE_TIMEOUT: Duration = Duration::from_millis(250);
/// Last-chance probe budget; above the interval on purpose since it runs at most once, before a kill.
const HEALTH_CONFIRM_PROBE_TIMEOUT: Duration = Duration::from_secs(30);

fn should_count_watchdog_failure(has_seen_healthy: bool, elapsed_since_start: Duration) -> bool {
    has_seen_healthy || elapsed_since_start >= BACKEND_STARTUP_GRACE_PERIOD
}

/// A warming answer clears the latch (adopted backends start latched); no answer leaves it alone.
fn watchdog_seen_healthy_after(previous: bool, alive: bool, warming_up: bool) -> bool {
    if !alive {
        return previous;
    }
    !warming_up
}

/// Only an answer updates the latch; silence keeps the last answer, since a stall gives no answer.
fn watchdog_inference_active_after(previous: bool, alive: bool, inference_active: bool) -> bool {
    if !alive {
        return previous;
    }
    inference_active
}

/// A timeout means something accepted and stalled; any other error means nothing is serving (death).
fn liveness_from_probe_error(error: &reqwest::Error) -> BackendLiveness {
    BackendLiveness {
        probe_timed_out: error.is_timeout(),
        ..BackendLiveness::default()
    }
}

/// A failed adopted check is a stall if the pre-probe answered, unless a different owner answered.
fn adopted_failure_is_a_stall(verified: bool, served_alive: bool, different_owner: bool) -> bool {
    !verified && served_alive && !different_owner
}

/// Not `alive && inference_active`: a saturated adopted backend fails ownership re-verification,
/// but its pre-probe busy marker is still evidence that it is generating.
fn watchdog_confirm_keeps_backend(confirmed: &BackendLiveness) -> bool {
    confirmed.inference_active && (confirmed.alive || confirmed.probe_timed_out)
}

/// One long last-chance probe before a kill, for a generation that started between probes.
/// Only on a stall; a refused port still dies at three strikes.
fn watchdog_should_confirm_before_death(
    consecutive_failures: u32,
    budget: u32,
    probe_timed_out: bool,
) -> bool {
    consecutive_failures >= budget && probe_timed_out
}

/// Probes await up to ~40s; a restart (new generation) or stop may land meanwhile, so recheck
/// before killing or the replacement gets killed.
fn watchdog_may_still_act(
    current_generation: u64,
    watched_generation: u64,
    has_owned: bool,
    shutting_down: bool,
) -> bool {
    current_generation == watched_generation && has_owned && !shutting_down
}

fn watchdog_failure_budget(inference_active: bool, probe_timed_out: bool) -> u32 {
    if inference_active && probe_timed_out {
        HEALTH_WATCHDOG_MAX_FAILURES_BUSY
    } else {
        HEALTH_WATCHDOG_MAX_FAILURES
    }
}

async fn managed_install_ready_after_repair() -> bool {
    crate::preflight::managed_install_ready().await
}

fn should_emit_repair_failed(msg: &str) -> bool {
    !msg.contains("NEEDS_ELEVATION")
}

fn external_conflict_message(conflict: &crate::preflight::ExternalBackendConflict) -> String {
    match conflict.reason.as_str() {
        "desktop_owned_backend_active" => format!(
            "A desktop-owned Unsloth server for this install is already running on port {}. Quit the other desktop app instance, then try again.",
            conflict.port
        ),
        // Do not describe a backend from an unknown install as terminal-started.
        "ambiguous_root_external_backend_active" => format!(
            "An Unsloth server is already running on port {}, and this app cannot confirm which install it belongs to. Stop that server, then try again.",
            conflict.port
        ),
        _ => format!(
            "An Unsloth server for this install is already running from a terminal on port {}. Stop that server, or run `unsloth studio update` from that terminal before using desktop repair/update.",
            conflict.port
        ),
    }
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
    update_state: tauri::State<'_, update::UpdateState>,
    install_state: tauri::State<'_, install::InstallState>,
    diagnostics: tauri::State<'_, DiagnosticsState>,
) -> Result<crate::preflight::DesktopPreflightResult, String> {
    let started = Instant::now();
    // The installer phase does not hold the runtime gate; checked before and after the probe.
    let mutating = || {
        install::is_install_running(install_state.inner())
            || update::is_update_running(update_state.inner())
            || update::is_repair_running(update_state.inner())
    };
    let mutating_before = mutating();
    let (result, adopted_watchdog_generation) =
        crate::preflight::desktop_preflight_result_with_state(state.inner()).await?;
    let result = if mutating_before || mutating() {
        crate::preflight::busy_managed_environment(result)
    } else {
        result
    };
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

/// Runs `unsloth -h` so a partial install (deps missing) reports false.
#[tauri::command]
pub async fn check_install_status() -> bool {
    let Some(bin) = process::find_unsloth_binary() else {
        return false;
    };

    let mut cmd = match process::build_managed_cli_command_tokio(&bin, &["-h"]) {
        Ok(cmd) => cmd,
        Err(e) => {
            warn!(
                "Install check: cannot run the managed CLI for {:?}: {}",
                bin, e
            );
            return false;
        }
    };
    cmd.stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    if let Err(error) = process::apply_managed_cli_context_tokio(&mut cmd) {
        warn!("Install check: no usable working directory: {}", error);
        return false;
    }

    #[cfg(windows)]
    {
        cmd.creation_flags(crate::process::CREATE_NO_WINDOW);
    }

    #[cfg(target_os = "linux")]
    crate::process::scrub_appimage_python_env_tokio(&mut cmd);

    // Tauri uses the legacy root regardless of UNSLOTH_STUDIO_HOME / STUDIO_HOME;
    // probe subprocesses must follow the same isolation as process.rs.
    cmd.env_remove("UNSLOTH_STUDIO_HOME");
    cmd.env_remove("STUDIO_HOME");

    let mut child = match process::with_studio_runtime_launch_guard(|| {
        cmd.spawn().map_err(|error| error.to_string())
    }) {
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

/// Start the backend and spawn a health watchdog that emits `server-crashed` if it hangs.
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
    alive: bool,
    /// Answered, but the background warm (ML-stack imports) is still in flight.
    warming_up: bool,
    /// On the adopted path this is the pre-probe reading; pair with `alive` for "answered now".
    inference_active: bool,
    /// Timed out rather than refused: a process still holds the port (stall, not death).
    probe_timed_out: bool,
    /// An HTTP RESPONSE, whatever its status. Weaker than `alive`, which also requires the payload to name this service.
    answered: bool,
}

/// Check if an Unsloth backend is running on the given port.
/// Expects JSON with status=="alive" (or "healthy") AND service=="Unsloth UI Backend".
#[tauri::command]
pub async fn check_health(port: u16) -> Result<bool, String> {
    match check_health_inner(port, HEALTH_PROBE_TIMEOUT).await {
        Ok(liveness) => Ok(liveness.alive),
        Err(e) => {
            info!("Health check on port {} failed: {}", port, e);
            Ok(false)
        }
    }
}

/// A stall arrives through the `Err` arm; a refused connection is not a timeout.
#[tauri::command]
pub async fn check_backend_present(
    state: tauri::State<'_, BackendState>,
    port: u16,
) -> Result<bool, String> {
    // Silence does not say which side went quiet, so ownership of the port settles it.
    let state = state.inner();
    backend_presence(port, || we_manage_a_backend_on(state, port)).await
}

/// *we_manage_it* is called AFTER the probe: an ownership answer taken before it died would report an exited backend as running.
async fn backend_presence(
    port: u16,
    we_manage_it: impl Fn() -> bool,
) -> Result<bool, String> {
    match check_health_inner(port, HEALTH_PROBE_TIMEOUT).await {
        Ok(liveness) => Ok(backend_is_present(&liveness, we_manage_it())),
        Err(e) => {
            info!("Backend presence check on port {} failed: {}", port, e);
            Ok(backend_is_present(
                &liveness_from_probe_error(&e),
                we_manage_it(),
            ))
        }
    }
}

/// Not `check_backend_present`: our own backend may not have bound its port yet. Ownership asked first.
#[tauri::command]
pub async fn check_backend_is_gone(
    state: tauri::State<'_, BackendState>,
    port: u16,
) -> Result<bool, String> {
    let state = state.inner();
    Ok(backend_is_gone(port, REFUSAL_PROBE_TIMEOUT, || {
        we_could_be_bringing_up(state, port)
    })
    .await)
}

async fn backend_is_gone(port: u16, budget: Duration, we_manage_it: impl Fn() -> bool) -> bool {
    // Port 0 is the placeholder base the webview holds before a validated port arrives.
    if port == 0 {
        return false;
    }
    // A backend we started may not have bound its port yet: keep waiting.
    if we_manage_it() {
        return false;
    }
    if !matches!(connect_outcome(port, budget).await, ConnectOutcome::Refused) {
        return false;
    }
    // Re-check ownership: Windows refusals can take ~2s, so the earlier read may be stale.
    !we_manage_it()
}

#[derive(Debug, PartialEq, Eq)]
enum ConnectOutcome {
    Refused,
    Accepted,
    /// No answer inside the budget, or an error not naming a closed port: #10520 lands here.
    Unsettled,
}

async fn connect_outcome(port: u16, budget: Duration) -> ConnectOutcome {
    let target = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let settled = tokio::time::timeout(budget, tokio::net::TcpStream::connect(target))
        .await
        .ok()
        .map(|attempt| attempt.map(|_stream| ()));
    classify_connect(settled)
}

/// `None` is a spent budget.
fn classify_connect(settled: Option<std::io::Result<()>>) -> ConnectOutcome {
    match settled {
        Some(Ok(())) => ConnectOutcome::Accepted,
        Some(Err(err)) if err.kind() == std::io::ErrorKind::ConnectionRefused => {
            ConnectOutcome::Refused
        }
        Some(Err(_)) => ConnectOutcome::Unsettled,
        None => ConnectOutcome::Unsettled,
    }
}

/// Whether this app's own backend handle names *port*. The handle outlives the process it names, so the owner is checked too.
fn we_manage_a_backend_on(state: &BackendState, port: u16) -> bool {
    // Handle and process state in one pass under the same lock, so they cannot disagree.
    process::owned_backend_on_port_is_running(state, port)
}

/// Deliberately wider than presence: a live backend of ours that has not reported a port yet
/// may be about to bind THIS one, and a refusal about it is not proof of anything.
fn we_could_be_bringing_up(state: &BackendState, port: u16) -> bool {
    process::owned_backend_could_bind_port(state, port)
}

fn backend_is_present(liveness: &BackendLiveness, we_manage_it: bool) -> bool {
    // A healthy answer is presence whoever owns the port; a timeout or unhealthy reply counts only for a backend we manage.
    liveness.alive || ((liveness.probe_timed_out || liveness.answered) && we_manage_it)
}

/// Uses /api/liveness (health awaits hardware detection), falling back to /api/health on 404.
/// The budget is a parameter so the last-chance probe can use a wider one.
async fn check_health_inner(
    port: u16,
    budget: Duration,
) -> Result<BackendLiveness, reqwest::Error> {
    let client = crate::loopback_http::client(budget)?;
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
            // Not healthy, but not silence either: something answered on that port.
            return Ok(BackendLiveness {
                answered: true,
                ..BackendLiveness::default()
            });
        }
        json = match resp.json::<serde_json::Value>().await {
            Ok(value) => Some(value),
            Err(error) => {
                // Propagating the error made presence indistinguishable from a refusal.
                info!("Backend on port {} answered unparseable JSON: {}", port, error);
                return Ok(BackendLiveness {
                    answered: true,
                    ..BackendLiveness::default()
                });
            }
        };
        break;
    }
    let Some(json) = json else {
        return Ok(BackendLiveness {
            answered: true,
            ..BackendLiveness::default()
        });
    };

    // Liveness answers "alive", health "healthy"; accept either.
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
    // `torch_warm_in_progress` covers the whole warm, not just hardware detection.
    let warming = json
        .get("torch_warm_in_progress")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    // Fallback for older backends. A deferred warm sets hardware_detection_deferred and must not count
    // as warming, or the grace would never end.
    let detecting = json
        .get("hardware_detecting")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let deferred = json
        .get("hardware_detection_deferred")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Absent on old backends, which read as not busy.
    let inference_active = json
        .get("inference_active")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let alive = live && correct_service;
    Ok(BackendLiveness {
        alive,
        warming_up: alive && (warming || (detecting && !deferred)),
        inference_active: alive && inference_active,
        probe_timed_out: false,
        answered: true,
    })
}

async fn check_watchdog_health(
    state: &BackendState,
    generation: u64,
    port: u16,
    has_adopted: bool,
    budget: Duration,
) -> BackendLiveness {
    if !has_adopted {
        return match check_health_inner(port, budget).await {
            Ok(liveness) => liveness,
            Err(error) => liveness_from_probe_error(&error),
        };
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
    // Classify the failure here first: the ownership probe folds every transport error into not-verified.
    let served = match check_health_inner(port, budget).await {
        Ok(liveness) => liveness,
        Err(error) => return liveness_from_probe_error(&error),
    };
    // HEALTH_PROBE_TIMEOUT, not the 2s default, or every request times out during the GIL stall.
    let ownership = crate::desktop_backend_owner::probe_owned_backend_state_with_timeout(
        owner,
        Some(port),
        false,
        budget,
    )
    .await;
    let verified = matches!(
        ownership,
        crate::desktop_backend_owner::OwnedBackendProbe::Verified(_)
    );
    // The one failure the probe is certain about: a complete answer naming another owner.
    let different_owner = crate::desktop_backend_owner::probe_saw_a_different_owner(&ownership);
    adopted_backend_liveness(verified, &served, different_owner)
}

/// `alive` and `warming_up` stay gated on verification so a foreign process cannot hold the grace open.
fn adopted_backend_liveness(
    verified: bool,
    served: &BackendLiveness,
    different_owner: bool,
) -> BackendLiveness {
    BackendLiveness {
        alive: verified,
        warming_up: verified && served.warming_up,
        // Not gated on verification: the pre-probe busy marker is the evidence the confirm probe needs.
        // A takeover (different_owner) still clears it.
        inference_active: served.inference_active && !different_owner,
        // Ownership's extra requests can time out on a saturated backend; the pre-probe answer
        // makes it a stall.
        probe_timed_out: adopted_failure_is_a_stall(verified, served.alive, different_owner),
        answered: served.answered || served.alive,
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

/// Validates the path first so callers get a clean error, not a raw OS failure.
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
    open_existing_dir_with(dir, |path| crate::process::open_detached(path))
}

/// Open the Unsloth logs directory in the system file manager.
#[tauri::command]
pub fn open_logs_dir(webview: tauri::Webview) -> Result<(), String> {
    crate::native_intents::ensure_main_window(&webview)?;
    open_existing_dir(&diagnostics::logs_dir())
}

/// Open a models directory (resolved by the backend) in the system file manager.
#[tauri::command]
pub fn open_models_dir(webview: tauri::Webview, path: String) -> Result<(), String> {
    crate::native_intents::ensure_main_window(&webview)?;
    open_existing_dir(std::path::Path::new(&path))
}

/// Run the platform installer with --tauri, streaming progress.
/// Returns "NEEDS_ELEVATION" if system packages need elevated install (Linux only).
#[tauri::command]
pub async fn start_install(
    app: AppHandle,
    state: tauri::State<'_, install::InstallState>,
    backend_state: tauri::State<'_, BackendState>,
    update_state: tauri::State<'_, update::UpdateState>,
    diagnostics: tauri::State<'_, DiagnosticsState>,
) -> Result<(), String> {
    if has_owned_backend(&backend_state)? {
        return Err(
            "The Unsloth backend is still running. Stop it before starting installation."
                .to_string(),
        );
    }
    // A repair's installer phase runs through run_install_for_repair, so one started here races it.
    if update::is_repair_running(update_state.inner()) {
        return Err("Cannot install while a repair is in progress.".to_string());
    }
    block_external_conflict(&[]).await?;

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

/// Install system packages elevated (Linux); only packages the install script reported are allowed.
#[cfg(target_os = "linux")]
#[tauri::command]
pub fn install_system_packages(
    packages: Vec<String>,
    state: tauri::State<'_, install::InstallState>,
    diagnostics: tauri::State<'_, DiagnosticsState>,
) -> Result<(), String> {
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

/// Non-Linux stub: elevation is handled by the scripts themselves.
#[cfg(not(target_os = "linux"))]
#[tauri::command]
pub fn install_system_packages(
    _packages: Vec<String>,
    _state: tauri::State<'_, install::InstallState>,
    _diagnostics: tauri::State<'_, DiagnosticsState>,
) -> Result<(), String> {
    Err("Elevated package install is only supported on Linux".to_string())
}

fn backend_update_confirmed(training_active: bool, confirm: impl FnOnce() -> bool) -> bool {
    !training_active || confirm()
}

#[tauri::command]
pub async fn confirm_backend_update(app: AppHandle) -> bool {
    // blocking_show parks its thread until the user answers: keep it off the async workers.
    tauri::async_runtime::spawn_blocking(move || {
        backend_update_confirmed(crate::training_is_active(&app), || {
            crate::confirm_update_during_training(&app)
        })
    })
    .await
    .unwrap_or(false)
}

/// Stop the server and run `unsloth studio update`. Does NOT restart; the frontend relaunches.
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
    // A repair holds no child handle between its update and its installer: invisible to the above.
    if update::is_repair_running(update_state.inner()) {
        return Err("Cannot update while a repair is in progress.".to_string());
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

/// Whether a native path lease this app signs can be verified: the key is per process, so only a
/// backend THIS process spawned holds it; adopted ones advertise support with another key.
#[tauri::command]
pub async fn native_path_leases_usable(
    backend_state: tauri::State<'_, BackendState>,
) -> Result<bool, String> {
    Ok(
        matches!(process::owned_backend_snapshot(backend_state.inner())?,
            Some(snapshot) if !snapshot.is_adopted),
    )
}

/// Repair a stale managed Unsloth install.
/// `force_installer` skips `studio update`, which keeps a CPU-only torch; only the installer re-selects it.
#[tauri::command]
pub async fn start_managed_repair(
    app: AppHandle,
    backend_state: tauri::State<'_, BackendState>,
    shutdown: tauri::State<'_, ShutdownFlag>,
    update_state: tauri::State<'_, update::UpdateState>,
    install_state: tauri::State<'_, install::InstallState>,
    diagnostics: tauri::State<'_, DiagnosticsState>,
    force_installer: Option<bool>,
) -> Result<(), String> {
    let force_installer = force_installer.unwrap_or(false);
    info!(
        "start_managed_repair command called (force_installer={})",
        force_installer
    );

    if install_state
        .lock()
        .map(|s| s.child.is_some())
        .unwrap_or(false)
    {
        return Err("Cannot repair while installation is in progress.".to_string());
    }

    // Held to the end: process handles are empty between stop, update and installer, so duplicates raced.
    let _repair = update::RepairInFlight::claim(update_state.inner())?;

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

    let update_result = if force_installer {
        let _ = app.emit("repair-progress", "Running bundled installer...");
        Ok(())
    } else {
        let _ = app.emit("repair-progress", "Updating existing Unsloth install...");
        let update_app = app.clone();
        let update_state = update_state.inner().clone();
        let update_diagnostics = diagnostics_state.clone();
        let update_repair_group_id = repair_group_id.clone();
        tokio::task::spawn_blocking(move || {
            update::run_backend_update_for_repair(
                update_app,
                update_state,
                update_diagnostics,
                update_repair_group_id,
            )
        })
        .await
        .map_err(|e| format!("Repair update task panicked: {e}"))?
    };

    match update_result {
        Ok(()) if !force_installer && managed_install_ready_after_repair().await => {
            info!("Managed repair complete after update");
            diagnostics::finish_repair_group(&diagnostics_state, &repair_group_id, "success", None);
            let _ = app.emit("repair-complete", ());
            return Ok(());
        }
        Ok(()) if force_installer => {}
        Ok(()) => {
            warn!("Managed repair update finished, but preflight is still not ready; falling back to installer");
            let _ = app.emit(
                "repair-progress",
                "Update finished, but Unsloth is still not ready. Running bundled installer...",
            );
        }
        Err(msg) => {
            // A stop is the user quitting, not a broken install; running the installer would
            // half-build the venv.
            if msg == update::UPDATE_STOPPED {
                info!("Managed repair update stopped; skipping installer fallback");
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
            force_installer,
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

#[allow(clippy::items_after_test_module)]
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
        // Only an owned backend can advertise lease support; ready_health(false)
        // stands in for a terminal-started server, which never can.
        let leases = if include_owner {
            r#""native_path_leases_supported":true,"#
        } else {
            ""
        };
        format!(
            r#"{{"status":"healthy","service":"Unsloth UI Backend","version":"2026.8.4","desktop_protocol_version":1,"desktop_manageability_version":1,"supports_desktop_auth":true,"supports_desktop_backend_ownership":true,{leases}"studio_root_id":"{ROOT_ID}"{owner}}}"#
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

    #[test]
    fn keeping_training_at_the_update_prompt_declines_the_update() {
        assert!(!super::backend_update_confirmed(true, || false));
    }

    #[test]
    fn updating_anyway_during_training_confirms_the_update() {
        assert!(super::backend_update_confirmed(true, || true));
    }

    #[test]
    fn updating_without_training_does_not_ask() {
        let mut asked = false;
        assert!(super::backend_update_confirmed(false, || {
            asked = true;
            false
        }));
        assert!(!asked);
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

    /// Stub backend; `liveness: None` models a backend older than the route (404).
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
        // Health awaits hardware detection, so liveness must be the only route touched.
        let (port, paths) = probe_test_backend(
            Some(r#"{"status":"alive","service":"Unsloth UI Backend"}"#.to_string()),
            ready_health(false),
        )
        .await;

        let liveness = super::check_health_inner(port, super::HEALTH_PROBE_TIMEOUT)
            .await
            .unwrap();

        assert!(liveness.alive);
        assert!(!liveness.warming_up);
        assert_eq!(paths.lock().unwrap().as_slice(), ["/api/liveness"]);
    }

    #[tokio::test]
    async fn a_backend_without_the_liveness_route_still_validates_through_health() {
        let (port, paths) = probe_test_backend(None, ready_health(false)).await;

        let liveness = super::check_health_inner(port, super::HEALTH_PROBE_TIMEOUT)
            .await
            .unwrap();

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

        let liveness = super::check_health_inner(port, super::HEALTH_PROBE_TIMEOUT)
            .await
            .unwrap();

        assert!(liveness.alive);
        assert!(
            liveness.warming_up,
            "an unsettled hardware verdict means the torch import is still in flight"
        );
    }

    #[tokio::test]
    async fn a_late_warm_stage_still_counts_as_warming_up() {
        // Hardware detection ends early; the grace must last until the whole warm finishes.
        let (port, _) = probe_test_backend(
            Some(
                r#"{"status":"alive","service":"Unsloth UI Backend","torch_warm_in_progress":true}"#
                    .to_string(),
            ),
            ready_health(false),
        )
        .await;

        let liveness = super::check_health_inner(port, super::HEALTH_PROBE_TIMEOUT)
            .await
            .unwrap();

        assert!(liveness.alive);
        assert!(
            liveness.warming_up,
            "a settled hardware verdict does not mean the warm is over; the imports that \
             hold the GIL longest run after it"
        );
    }

    #[tokio::test]
    async fn a_finished_warm_ends_the_startup_grace() {
        let (port, _) = probe_test_backend(
            Some(
                r#"{"status":"alive","service":"Unsloth UI Backend","torch_warm_in_progress":false}"#
                    .to_string(),
            ),
            ready_health(false),
        )
        .await;

        let liveness = super::check_health_inner(port, super::HEALTH_PROBE_TIMEOUT)
            .await
            .unwrap();

        assert!(liveness.alive);
        assert!(!liveness.warming_up);
    }

    #[tokio::test]
    async fn a_backend_predating_the_warm_field_still_gets_its_grace() {
        let (port, _) = probe_test_backend(
            Some(
                r#"{"status":"alive","service":"Unsloth UI Backend","hardware_detecting":true}"#
                    .to_string(),
            ),
            ready_health(false),
        )
        .await;

        let liveness = super::check_health_inner(port, super::HEALTH_PROBE_TIMEOUT)
            .await
            .unwrap();

        assert!(liveness.warming_up);
    }

    #[tokio::test]
    async fn a_deferred_warm_is_not_reported_as_still_warming_up() {
        let (port, _) = probe_test_backend(
            Some(
                r#"{"status":"alive","service":"Unsloth UI Backend","hardware_detecting":true,"hardware_detection_deferred":true}"#
                    .to_string(),
            ),
            ready_health(false),
        )
        .await;

        let liveness = super::check_health_inner(port, super::HEALTH_PROBE_TIMEOUT)
            .await
            .unwrap();

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
            super::check_health_inner(port, super::HEALTH_PROBE_TIMEOUT)
                .await
                .unwrap(),
            super::BackendLiveness {
                // It answered, and that is all `answered` says: not OUR backend.
                answered: true,
                ..super::BackendLiveness::default()
            }
        );
        assert!(!super::backend_is_present(
            &super::check_health_inner(port, super::HEALTH_PROBE_TIMEOUT)
                .await
                .unwrap(),
            false
        ));
    }

    #[test]
    fn the_probe_budget_fits_inside_one_watchdog_interval() {
        // A probe that outlives the interval would let the next tick start on top of it.
        assert!(super::HEALTH_PROBE_TIMEOUT < super::HEALTH_WATCHDOG_INTERVAL);
    }

    #[test]
    fn the_frontend_retry_ladder_outlives_one_probe_budget() {
        // The ladder lives in TypeScript and the budget here; this guard keeps them in step.
        let src = include_str!("../../frontend/src/features/auth/api.ts").replace("\r\n", "\n");
        let marker = "const TAURI_FETCH_RETRY_DELAYS_MS = [";
        let start = src
            .find(marker)
            .expect("the Tauri fetch retry ladder moved; update this guard")
            + marker.len();
        let ladder = &src[start..];
        let ladder = &ladder[..ladder.find(']').expect("unterminated retry ladder")];
        let total_ms: u64 = ladder
            .split(',')
            .map(str::trim)
            .filter(|delay| !delay.is_empty())
            .map(|delay| {
                delay
                    .parse::<u64>()
                    .expect("a retry delay stopped being a plain number of milliseconds")
            })
            .sum();
        assert!(
            total_ms >= super::HEALTH_PROBE_TIMEOUT.as_millis() as u64,
            "the webview gives up after {total_ms}ms while one native liveness probe is \
             allowed {}ms, so a backend the launcher still considers alive is reported to \
             the user as not running",
            super::HEALTH_PROBE_TIMEOUT.as_millis()
        );
        assert!(
            src.contains("invoke<boolean>(\"check_backend_present\""),
            "the transport-failure path no longer asks the native side before it tells the \
             user to relaunch"
        );
        // And not the health command, which reports a spent budget as a refused connection.
        assert!(
            !src.contains("invoke<boolean>(\"check_health\""),
            "the transport-failure path is back on check_health, which collapses a stalled \
             probe onto \"not running\""
        );
    }

    #[test]
    fn a_stalled_probe_is_not_reported_as_an_absent_backend() {
        let stalled = super::BackendLiveness {
            alive: false,
            probe_timed_out: true,
            ..Default::default()
        };
        let closed = super::BackendLiveness::default();
        let answered = super::BackendLiveness {
            alive: true,
            ..Default::default()
        };

        // Through the rule the command applies: an inlined copy asserts its own arithmetic.
        assert!(!stalled.alive, "a stall is not an answer");
        assert!(
            super::backend_is_present(&stalled, true),
            "a stalled probe must read as a backend that is still present"
        );
        assert!(
            !super::backend_is_present(&stalled, false),
            "a stall on a port this app does not manage proves nothing about our backend"
        );
        assert!(
            !super::backend_is_present(&closed, true),
            "a refused connection must still read as absent"
        );
        assert!(super::backend_is_present(&answered, false));
        assert!(
            !stalled.alive,
            "check_health still reports a stall as not usable"
        );
    }

    #[test]
    fn the_startup_grace_survives_the_mac_cold_start_timeline() {
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
    fn the_adopted_ownership_probe_uses_the_watchdog_budget() {
        // The ownership probe must use HEALTH_PROBE_TIMEOUT or the warm-up read never runs.
        // Normalise CRLF: include_str! embeds Windows checkouts verbatim.
        let src = include_str!("commands.rs").replace("\r\n", "\n");
        let start = src
            .find("async fn check_watchdog_health")
            .expect("check_watchdog_health moved; update this guard");
        let body = &src[start..];
        let body = &body[..body.find("\n}\n").expect("could not find the function end")];
        assert!(
            body.contains("probe_owned_backend_state_with_timeout"),
            "the adopted path is back on the default-timeout ownership probe"
        );
        assert!(
            body.contains("HEALTH_PROBE_TIMEOUT"),
            "the adopted ownership probe no longer uses the watchdog's probe budget"
        );
    }

    #[test]
    fn an_adopted_backend_that_is_still_warming_gets_the_grace_back() {
        // Adopted watchdogs start latched; a cold-starting backend relaunched onto must still get the grace.
        let mut has_seen_healthy = true; // adopted: count_failures_immediately
        has_seen_healthy = super::watchdog_seen_healthy_after(has_seen_healthy, true, true);
        assert!(
            !has_seen_healthy,
            "a backend that answered \"still warming\" left the latch set, so the grace stayed off"
        );
        for elapsed in [15, 30, 47, 64] {
            assert!(
                !super::should_count_watchdog_failure(
                    has_seen_healthy,
                    Duration::from_secs(elapsed)
                ),
                "a stalled probe at {elapsed}s was counted against an adopted backend that had \
                 just reported it was still warming"
            );
        }
        // The grace is still bounded: a backend that never finishes warming is not immortal.
        assert!(super::should_count_watchdog_failure(
            has_seen_healthy,
            super::BACKEND_STARTUP_GRACE_PERIOD
        ));
    }

    #[test]
    fn the_latch_tracks_the_last_answer_and_ignores_silence() {
        assert!(super::watchdog_seen_healthy_after(false, true, false));
        assert!(!super::watchdog_seen_healthy_after(true, true, true));
        assert!(super::watchdog_seen_healthy_after(true, false, false));
        assert!(!super::watchdog_seen_healthy_after(false, false, false));
        assert!(super::watchdog_seen_healthy_after(false, true, false));
    }

    #[test]
    fn a_backend_that_stalls_while_generating_is_not_killed_at_three_strikes() {
        // A saturated host (0.28 tok/s) can miss probes while still streaming.
        let mut generating = false;
        generating = super::watchdog_inference_active_after(generating, true, true);
        assert!(generating);
        generating = super::watchdog_inference_active_after(generating, false, false);
        assert!(generating);
        assert_eq!(
            super::watchdog_failure_budget(generating, true),
            super::HEALTH_WATCHDOG_MAX_FAILURES_BUSY
        );
        assert!(super::HEALTH_WATCHDOG_MAX_FAILURES_BUSY > super::HEALTH_WATCHDOG_MAX_FAILURES);
    }

    #[test]
    fn a_dead_port_is_still_declared_dead_at_three_strikes() {
        assert_eq!(
            super::watchdog_failure_budget(true, false),
            super::HEALTH_WATCHDOG_MAX_FAILURES
        );
        assert_eq!(
            super::watchdog_failure_budget(false, true),
            super::HEALTH_WATCHDOG_MAX_FAILURES
        );
    }

    /// Accepted streams are parked, not dropped: dropping would answer with a reset.
    async fn stalling_test_backend() -> u16 {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("probe test needs a loopback port");
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            let mut parked = Vec::new();
            while let Ok((stream, _)) = listener.accept().await {
                parked.push(stream);
            }
        });
        port
    }

    async fn answering_test_backend(status: &'static str, body: &'static str) -> u16 {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("probe test needs a loopback port");
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let mut buffer = [0; 2048];
                let Ok(_) = stream.read(&mut buffer).await else {
                    return;
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

    #[tokio::test]
    async fn ownership_is_read_after_the_probe_not_before_it() {
        // The closure answers "we manage it" only once the request reached the server, so true means it ran AFTER the probe.
        let probe_arrived = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("probe test needs a loopback port");
        let port = listener.local_addr().unwrap().port();
        let seen = std::sync::Arc::clone(&probe_arrived);
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let mut buffer = [0; 2048];
                let Ok(_) = stream.read(&mut buffer).await else {
                    return;
                };
                seen.store(true, std::sync::atomic::Ordering::SeqCst);
                let response = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
                let _ = stream.write_all(response.as_bytes()).await;
            }
        });

        let asked = std::sync::Arc::clone(&probe_arrived);
        assert_eq!(
            super::backend_presence(port, move || asked.load(std::sync::atomic::Ordering::SeqCst))
                .await,
            Ok(true),
            "ownership was read before the probe, so a backend that changed under it would \
             still have been reported as running"
        );
    }

    #[tokio::test]
    async fn a_managed_backend_that_answers_unhealthily_is_still_present() {
        let port = answering_test_backend("503 Service Unavailable", "").await;
        assert_eq!(
            super::backend_presence(port, || true).await,
            Ok(true),
            "a managed backend answering 503 was reported absent"
        );
        assert_eq!(super::backend_presence(port, || false).await, Ok(false));
    }

    #[tokio::test]
    async fn a_reply_this_build_cannot_parse_is_still_an_answer() {
        let port = answering_test_backend("200 OK", "not json at all").await;
        assert_eq!(
            super::backend_presence(port, || true).await,
            Ok(true),
            "a managed backend answering an unparseable body was reported absent"
        );
        assert_eq!(super::backend_presence(port, || false).await, Ok(false));

        let liveness = super::check_health_inner(port, super::HEALTH_PROBE_TIMEOUT)
            .await
            .expect("an answered probe is not a transport error");
        assert!(!liveness.alive, "an unparseable reply is not a healthy backend");
        assert!(liveness.answered, "the reply arrived, so the port is not silent");
    }

    #[tokio::test]
    async fn a_port_that_accepts_and_never_answers_reads_as_a_stall() {
        let port = stalling_test_backend().await;
        let client = crate::loopback_http::client(Duration::from_millis(250)).unwrap();

        let error = client
            .get(format!("http://127.0.0.1:{port}/api/liveness"))
            .send()
            .await
            .expect_err("a parked connection cannot answer");

        assert!(
            super::liveness_from_probe_error(&error).probe_timed_out,
            "a spent probe budget is not being classified as a stall, so a backend that is \
             merely busy gets the three-strike budget"
        );
    }

    #[tokio::test]
    async fn check_backend_present_reports_a_stalled_port_as_present() {
        // Costs HEALTH_PROBE_TIMEOUT in wall clock.
        let port = stalling_test_backend().await;
        assert_eq!(
            super::backend_presence(port, || true).await,
            Ok(true),
            "a backend holding the port and not answering was reported absent, which is the \
             relaunch prompt this command was added to prevent"
        );
    }

    #[tokio::test]
    async fn a_stalled_port_this_app_does_not_manage_is_not_our_backend() {
        let port = stalling_test_backend().await;
        assert_eq!(
            super::backend_presence(port, || false).await,
            Ok(false),
            "a stalled port with no managed backend behind it was reported as present"
        );
    }

    #[tokio::test]
    async fn check_backend_present_reports_a_closed_port_as_absent() {
        let port = {
            let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
            let port = listener.local_addr().unwrap().port();
            drop(listener);
            port
        };
        assert_eq!(
            super::backend_presence(port, || true).await,
            Ok(false),
            "a closed port must still read as an absent backend"
        );
    }

    /// Below the ephemeral range so no sibling test's port-0 binding can be handed this port.
    async fn a_closed_port_below_the_ephemeral_range() -> u16 {
        for candidate in 20_064..20_128u16 {
            if let Ok(listener) = TcpListener::bind(("127.0.0.1", candidate)).await {
                drop(listener);
                return candidate;
            }
        }
        panic!("every port in the probe window was already bound")
    }

    async fn describe_connect(port: u16, budget: Duration) -> String {
        let target =
            std::net::SocketAddr::from((std::net::Ipv4Addr::LOCALHOST, port));
        let started = std::time::Instant::now();
        let settled = tokio::time::timeout(budget, tokio::net::TcpStream::connect(target)).await;
        let elapsed = started.elapsed();
        match settled {
            Err(_) => format!("no answer inside {budget:?} (waited {elapsed:?})"),
            Ok(Ok(_)) => format!("accepted after {elapsed:?}"),
            Ok(Err(error)) => format!(
                "kind={:?} raw_os_error={:?} after {elapsed:?} ({error})",
                error.kind(),
                error.raw_os_error(),
            ),
        }
    }

    /// Floors are well under REFUSAL_PROBE_TIMEOUT so only a real drop fails this.
    #[test]
    fn the_refusal_budget_clears_the_wait_this_platform_actually_takes() {
        let floor = if cfg!(windows) {
            Duration::from_millis(2_100)
        } else {
            Duration::from_millis(50)
        };
        assert!(
            super::REFUSAL_PROBE_TIMEOUT >= floor,
            "REFUSAL_PROBE_TIMEOUT is {:?}, under the {floor:?} this platform needs to see a \
             refusal at all. Below the real wait every dead port reads as Unsettled and the \
             fast path never fires.",
            super::REFUSAL_PROBE_TIMEOUT,
        );
    }

    #[tokio::test]
    async fn a_closed_port_nobody_here_owns_is_gone() {
        let port = a_closed_port_below_the_ephemeral_range().await;
        let observed = describe_connect(port, super::REFUSAL_PROBE_TIMEOUT).await;
        assert!(
            super::backend_is_gone(port, super::REFUSAL_PROBE_TIMEOUT, || false).await,
            "a refused loopback connect on a port nothing here owns is not reported as gone. \
             Port {port} answered: {observed}"
        );
    }

    #[tokio::test]
    async fn a_closed_port_we_are_bringing_up_is_not_gone() {
        // Port 0 is safe here: ownership is read before anything connects.
        let port = {
            let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
            let port = listener.local_addr().unwrap().port();
            drop(listener);
            port
        };
        assert!(
            !super::backend_is_gone(port, super::REFUSAL_PROBE_TIMEOUT, || true).await,
            "a backend of ours that is still starting was reported as gone"
        );
    }

    #[tokio::test]
    async fn ownership_taken_while_the_probe_ran_still_keeps_the_ladder() {
        let port = a_closed_port_below_the_ephemeral_range().await;
        let asked = std::sync::atomic::AtomicUsize::new(0);
        let gone = super::backend_is_gone(port, super::REFUSAL_PROBE_TIMEOUT, || {
            asked.fetch_add(1, std::sync::atomic::Ordering::SeqCst) > 0
        })
        .await;
        assert_eq!(
            asked.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "ownership was not re-read after the connect, so the answer rests on a stale look"
        );
        assert!(
            !gone,
            "a backend that became ours while the probe was in flight was reported as gone"
        );
    }

    #[tokio::test]
    async fn a_port_something_is_listening_on_is_not_gone() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(
            !super::backend_is_gone(port, super::REFUSAL_PROBE_TIMEOUT, || false).await,
            "a port that accepted a connection was reported as gone"
        );
    }

    #[tokio::test]
    async fn the_placeholder_port_is_never_an_answer() {
        assert!(!super::backend_is_gone(0, super::REFUSAL_PROBE_TIMEOUT, || false).await);
    }

    #[test]
    fn a_connect_that_never_answers_is_unsettled_not_refused() {
        // No test can drop a SYN, so the rule is checked on its own.
        assert_eq!(
            super::classify_connect(None),
            super::ConnectOutcome::Unsettled,
            "a spent budget was reported as proof that nothing is listening"
        );
        assert_eq!(
            super::classify_connect(Some(Err(std::io::Error::from(
                std::io::ErrorKind::TimedOut
            )))),
            super::ConnectOutcome::Unsettled
        );
        assert_eq!(
            super::classify_connect(Some(Err(std::io::Error::from(
                std::io::ErrorKind::PermissionDenied
            )))),
            super::ConnectOutcome::Unsettled,
            "a blocked connect is not evidence the port is empty"
        );
        assert_eq!(
            super::classify_connect(Some(Err(std::io::Error::from(
                std::io::ErrorKind::ConnectionRefused
            )))),
            super::ConnectOutcome::Refused
        );
        assert_eq!(
            super::classify_connect(Some(Ok(()))),
            super::ConnectOutcome::Accepted
        );
    }

    #[test]
    fn the_refusal_probe_cannot_eat_the_ladder_it_short_circuits() {
        assert!(super::REFUSAL_PROBE_TIMEOUT * 4 <= super::HEALTH_PROBE_TIMEOUT);
    }

    #[test]
    fn the_fast_path_asks_for_absence_and_not_for_presence() {
        // `check_backend_present` reports our unbound backend as absent; the ladder must not use it.
        let src = include_str!("../../frontend/src/features/auth/api.ts").replace("\r\n", "\n");
        assert!(
            src.contains("invoke<boolean>(\"check_backend_is_gone\""),
            "the retry ladder no longer has a fast path for a backend that is provably gone"
        );
        let marker = "if (attempt === 0 && (await nativeBackendIsGone()))";
        assert!(
            src.contains(marker),
            "the fast path stopped being gated on the first failure and a positive answer, \
             so a slow backend can be abandoned before the ladder has run"
        );
    }

    #[tokio::test]
    async fn a_port_with_nothing_on_it_reads_as_death_not_a_stall() {
        let client = crate::loopback_http::client(Duration::from_secs(5)).unwrap();

        // Draw from below the ephemeral range: a freed port-0 port can be rebound by a sibling test,
        // and retrying a port that answered would consume another test's one-shot connection.
        let port = {
            let mut free = None;
            for candidate in 20_000..20_064u16 {
                if let Ok(listener) = TcpListener::bind(("127.0.0.1", candidate)).await {
                    drop(listener);
                    free = Some(candidate);
                    break;
                }
            }
            free.expect("every port in the probe window was already bound")
        };

        let error = client
            .get(format!("http://127.0.0.1:{port}/api/liveness"))
            .send()
            .await
            .expect_err("nothing is listening on this port");

        let liveness = super::liveness_from_probe_error(&error);
        assert!(!liveness.probe_timed_out, "a refused port read as a stall");
        assert_eq!(
            super::watchdog_failure_budget(true, liveness.probe_timed_out),
            super::HEALTH_WATCHDOG_MAX_FAILURES,
            "a dead port inherits the busy budget from the last answer it gave"
        );
    }

    #[tokio::test]
    async fn the_busy_marker_survives_the_wire() {
        // End to end, so the field name is checked against what main.py publishes.
        let (busy_port, _) = probe_test_backend(
            Some(
                r#"{"status":"alive","service":"Unsloth UI Backend","inference_active":true}"#
                    .to_string(),
            ),
            ready_health(false),
        )
        .await;
        let (idle_port, _) = probe_test_backend(
            Some(r#"{"status":"alive","service":"Unsloth UI Backend"}"#.to_string()),
            ready_health(false),
        )
        .await;

        assert!(
            super::check_health_inner(busy_port, super::HEALTH_PROBE_TIMEOUT)
                .await
                .unwrap()
                .inference_active
        );
        assert!(
            !super::check_health_inner(idle_port, super::HEALTH_PROBE_TIMEOUT)
                .await
                .unwrap()
                .inference_active
        );
    }

    #[test]
    fn both_watchdog_paths_classify_a_failed_probe() {
        // Normalise CRLF: include_str! embeds CRLF on Windows checkouts.
        let src = include_str!("commands.rs").replace("\r\n", "\n");
        let start = src
            .find("async fn check_watchdog_health")
            .expect("check_watchdog_health moved; update this guard");
        let body = &src[start..];
        let body = &body[..body.find("\n}\n").expect("could not find the function end")];

        assert_eq!(
            body.matches("liveness_from_probe_error").count(),
            2,
            "the owned and adopted branches must both classify a failed probe"
        );
    }

    #[test]
    fn a_backend_that_answers_idle_gives_the_wide_budget_back() {
        let generating = super::watchdog_inference_active_after(true, true, false);
        assert!(!generating);
        assert_eq!(
            super::watchdog_failure_budget(generating, true),
            super::HEALTH_WATCHDOG_MAX_FAILURES
        );
    }

    #[test]
    fn an_adopted_backend_that_answers_and_then_stalls_keeps_the_busy_budget() {
        assert!(super::adopted_failure_is_a_stall(false, true, false));
        assert_eq!(
            super::watchdog_failure_budget(
                true,
                super::adopted_failure_is_a_stall(false, true, false)
            ),
            super::HEALTH_WATCHDOG_MAX_FAILURES_BUSY
        );
    }

    #[test]
    fn a_stalled_adopted_backend_that_says_it_is_generating_survives_the_confirmation() {
        let served = super::BackendLiveness {
            alive: true,
            warming_up: false,
            inference_active: true,
            probe_timed_out: false,
            answered: true,
        };
        let confirmed = super::adopted_backend_liveness(false, &served, false);
        assert!(
            !confirmed.alive,
            "an unverified re-check is not a live answer"
        );
        assert!(super::watchdog_confirm_keeps_backend(&confirmed));

        let quiet =
            super::adopted_backend_liveness(false, &super::BackendLiveness::default(), false);
        assert!(!super::watchdog_confirm_keeps_backend(&quiet));

        let idle = super::adopted_backend_liveness(
            true,
            &super::BackendLiveness {
                alive: true,
                warming_up: false,
                inference_active: false,
                probe_timed_out: false,
                answered: true,
            },
            false,
        );
        assert!(!super::watchdog_confirm_keeps_backend(&idle));

        let taken_over = super::adopted_backend_liveness(false, &served, true);
        assert!(!super::watchdog_confirm_keeps_backend(&taken_over));
    }

    #[test]
    fn a_port_another_backend_took_over_stays_on_the_normal_budget() {
        // A freed port rebound by another Unsloth backend is a takeover, not a stall.
        assert!(!super::adopted_failure_is_a_stall(false, true, true));
        assert_eq!(
            super::watchdog_failure_budget(
                true,
                super::adopted_failure_is_a_stall(false, true, true)
            ),
            super::HEALTH_WATCHDOG_MAX_FAILURES
        );
        assert!(!super::watchdog_should_confirm_before_death(
            super::HEALTH_WATCHDOG_MAX_FAILURES,
            super::HEALTH_WATCHDOG_MAX_FAILURES,
            super::adopted_failure_is_a_stall(false, true, true),
        ));
    }

    #[test]
    fn an_adopted_port_that_answered_nothing_is_not_called_a_stall() {
        assert!(!super::adopted_failure_is_a_stall(false, false, false));
        assert_eq!(
            super::watchdog_failure_budget(
                true,
                super::adopted_failure_is_a_stall(false, false, false)
            ),
            super::HEALTH_WATCHDOG_MAX_FAILURES
        );
        assert!(!super::adopted_failure_is_a_stall(true, true, false));
    }

    #[test]
    fn a_generation_that_starts_between_probes_still_gets_one_last_chance() {
        assert!(super::watchdog_should_confirm_before_death(
            super::HEALTH_WATCHDOG_MAX_FAILURES,
            super::HEALTH_WATCHDOG_MAX_FAILURES,
            true,
        ));
        assert!(!super::watchdog_should_confirm_before_death(
            super::HEALTH_WATCHDOG_MAX_FAILURES - 1,
            super::HEALTH_WATCHDOG_MAX_FAILURES,
            true,
        ));
        assert!(!super::watchdog_should_confirm_before_death(
            super::HEALTH_WATCHDOG_MAX_FAILURES,
            super::HEALTH_WATCHDOG_MAX_FAILURES,
            false,
        ));
    }

    #[test]
    fn the_last_chance_budget_is_wider_than_the_one_that_gave_up() {
        assert!(super::HEALTH_CONFIRM_PROBE_TIMEOUT > super::HEALTH_PROBE_TIMEOUT);
    }

    #[derive(Clone, Copy, Debug)]
    enum Probe {
        Answered { warming: bool, busy: bool },
        TimedOut,
        Refused,
    }

    fn answered(busy: bool) -> Probe {
        Probe::Answered {
            warming: false,
            busy,
        }
    }

    /// Mirrors health_watchdog's failure accounting (it needs an AppHandle and real sleeps) using the
    /// same helpers; returns the second the backend was declared dead.
    fn simulate_watchdog(probes: &[Probe], confirms: &[Probe]) -> Option<u64> {
        let interval = super::HEALTH_WATCHDOG_INTERVAL.as_secs();
        let probe_budget = super::HEALTH_PROBE_TIMEOUT.as_secs();
        let confirm_budget = super::HEALTH_CONFIRM_PROBE_TIMEOUT.as_secs();

        let mut failures: u32 = 0;
        let mut was_generating = false;
        let mut elapsed: u64 = 0;
        let mut next_confirm = 0usize;

        for probe in probes {
            elapsed += interval;
            let liveness = match *probe {
                Probe::Answered { warming, busy } => super::BackendLiveness {
                    alive: true,
                    warming_up: warming,
                    inference_active: busy,
                    probe_timed_out: false,
                    answered: true,
                },
                Probe::TimedOut => {
                    elapsed += probe_budget;
                    super::BackendLiveness {
                        probe_timed_out: true,
                        ..super::BackendLiveness::default()
                    }
                }
                Probe::Refused => super::BackendLiveness::default(),
            };

            if liveness.alive {
                was_generating = super::watchdog_inference_active_after(
                    was_generating,
                    liveness.alive,
                    liveness.inference_active,
                );
                failures = 0;
                continue;
            }

            let budget = super::watchdog_failure_budget(was_generating, liveness.probe_timed_out);
            failures += 1;
            if super::watchdog_should_confirm_before_death(
                failures,
                budget,
                liveness.probe_timed_out,
            ) {
                let confirm = confirms
                    .get(next_confirm)
                    .copied()
                    .unwrap_or(Probe::TimedOut);
                next_confirm += 1;
                match confirm {
                    Probe::Answered { busy: true, .. } => {
                        was_generating = true;
                        failures = 0;
                        continue;
                    }
                    Probe::Answered { busy: false, .. } => {}
                    Probe::TimedOut => elapsed += confirm_budget,
                    Probe::Refused => {}
                }
            }
            if failures >= budget {
                return Some(elapsed);
            }
        }
        None
    }

    #[test]
    fn simulated_timelines_kill_only_what_should_be_killed() {
        let plain = super::HEALTH_WATCHDOG_MAX_FAILURES as usize;
        let busy = super::HEALTH_WATCHDOG_MAX_FAILURES_BUSY as usize;
        let interval = super::HEALTH_WATCHDOG_INTERVAL.as_secs();
        let probe_budget = super::HEALTH_PROBE_TIMEOUT.as_secs();
        let confirm_budget = super::HEALTH_CONFIRM_PROBE_TIMEOUT.as_secs();
        let stall_cycle = interval + probe_budget;

        assert_eq!(simulate_watchdog(&vec![answered(false); 200], &[]), None);
        assert_eq!(simulate_watchdog(&vec![answered(true); 200], &[]), None);
        assert_eq!(
            simulate_watchdog(
                &vec![
                    Probe::Answered {
                        warming: true,
                        busy: false
                    };
                    200
                ],
                &[]
            ),
            None
        );

        let mut dead = vec![answered(false)];
        dead.extend(vec![Probe::Refused; plain]);
        assert_eq!(simulate_watchdog(&dead, &[]), Some(interval * 4));
        let mut died_generating = vec![answered(true)];
        died_generating.extend(vec![Probe::Refused; plain]);
        assert_eq!(simulate_watchdog(&died_generating, &[]), Some(interval * 4));

        let mut stalling = vec![answered(true)];
        stalling.extend(vec![Probe::TimedOut; busy - 1]);
        assert_eq!(simulate_watchdog(&stalling, &[]), None);

        let mut wedged = vec![answered(true)];
        wedged.extend(vec![Probe::TimedOut; busy + 4]);
        assert_eq!(
            simulate_watchdog(&wedged, &[Probe::TimedOut]),
            Some(interval + busy as u64 * stall_cycle + confirm_budget)
        );

        let mut started_between_probes = vec![answered(false)];
        started_between_probes.extend(vec![Probe::TimedOut; 40]);
        assert_eq!(
            simulate_watchdog(
                &started_between_probes,
                &[answered(true); 8] // one per spent budget
            ),
            None
        );
        assert_eq!(
            simulate_watchdog(&started_between_probes, &[Probe::TimedOut]),
            Some(interval + plain as u64 * stall_cycle + confirm_budget)
        );
        assert_eq!(
            simulate_watchdog(&started_between_probes, &[answered(false)]),
            Some(interval + plain as u64 * stall_cycle)
        );

        let mut finished = vec![answered(true), answered(false)];
        finished.extend(vec![Probe::Refused; plain]);
        assert_eq!(simulate_watchdog(&finished, &[]), Some(interval * 5));

        let mut stalled_then_gone = vec![answered(true)];
        stalled_then_gone.extend(vec![Probe::TimedOut; 5]);
        stalled_then_gone.push(Probe::Refused);
        assert_eq!(
            simulate_watchdog(&stalled_then_gone, &[]),
            Some(interval + 5 * stall_cycle + interval)
        );

        let flapping: Vec<Probe> = (0..200)
            .map(|i| {
                if i % 2 == 0 {
                    answered(true)
                } else {
                    Probe::TimedOut
                }
            })
            .collect();
        assert_eq!(simulate_watchdog(&flapping, &[]), None);
    }

    /// Answers only after `delay`: a loop running behind, not gone.
    async fn slow_test_backend(delay: Duration, body: &'static str) -> u16 {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("probe test needs a loopback port");
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                tokio::spawn(async move {
                    let mut buffer = [0; 2048];
                    if stream.read(&mut buffer).await.is_err() {
                        return;
                    }
                    tokio::time::sleep(delay).await;
                    let response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = stream.write_all(response.as_bytes()).await;
                });
            }
        });
        port
    }

    #[tokio::test]
    async fn a_wider_budget_reaches_a_backend_the_per_cycle_one_gives_up_on() {
        let port = slow_test_backend(
            Duration::from_millis(600),
            r#"{"status":"alive","service":"Unsloth UI Backend","inference_active":true}"#,
        )
        .await;

        let gave_up = super::check_health_inner(port, Duration::from_millis(150))
            .await
            .expect_err("a 150ms budget cannot outlast a 600ms answer");
        assert!(
            super::liveness_from_probe_error(&gave_up).probe_timed_out,
            "a slow-but-answering backend must read as a stall, not as a dead port"
        );

        let confirmed = super::check_health_inner(port, Duration::from_secs(5))
            .await
            .expect("a 5s budget outlasts a 600ms answer");
        assert!(confirmed.alive);
        assert!(
            confirmed.inference_active,
            "the wider probe must still parse the busy marker, or the reprieve never fires"
        );
    }

    #[test]
    fn conflict_message_only_blames_a_terminal_for_a_same_install_backend() {
        let message = |reason: &str| {
            super::external_conflict_message(&crate::preflight::ExternalBackendConflict {
                port: 8890,
                reason: reason.to_string(),
            })
        };

        assert!(message("same_root_external_backend_active").contains("from a terminal"));

        for reason in [
            "desktop_owned_backend_active",
            "ambiguous_root_external_backend_active",
        ] {
            let message = message(reason);
            assert!(!message.contains("from a terminal"), "{message}");
            assert!(!message.contains("unsloth studio update"), "{message}");
            assert!(message.contains("port 8890"), "{message}");
        }
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

    #[test]
    fn a_restart_during_the_last_chance_probe_is_not_declared_dead() {
        // A probe answer names a service, not a generation; anything but the same live generation must stop.
        assert!(super::watchdog_may_still_act(7, 7, true, false));
        assert!(!super::watchdog_may_still_act(8, 7, true, false));
        assert!(!super::watchdog_may_still_act(7, 7, false, false));
        assert!(!super::watchdog_may_still_act(7, 7, true, true));
    }

    #[test]
    fn the_watchdog_rereads_the_generation_after_the_confirm_probe() {
        // Up to 40s separates the generation check from the kill; without the re-read a restart gets killed.
        // Normalise CRLF: include_str! embeds CRLF on the Windows runner.
        let src = include_str!("commands.rs").replace("\r\n", "\n");
        let start = src
            .find("async fn health_watchdog")
            .expect("health_watchdog moved; update this guard");
        let body = &src[start..];
        let body = &body[..body.find("\n}\n").expect("could not find the function end")];

        let confirm = body
            .find("HEALTH_CONFIRM_PROBE_TIMEOUT")
            .expect("the last-chance probe is gone; update this guard");
        let guard = body
            .find("watchdog_may_still_act")
            .expect("the post-probe generation re-read is gone");
        let kill = body
            .rfind("stop_backend")
            .expect("the unresponsive-backend kill moved");
        assert!(
            confirm < guard && guard < kill,
            "the generation re-read must sit between the last-chance probe and the kill"
        );
    }
}

/// Periodic check for hung backends. Failures are ignored during the startup grace; afterwards 3
/// failures (12 when last seen generating and probes time out) emit `server-crashed`.
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
    let mut was_generating = false;

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

        let liveness =
            check_watchdog_health(&state, generation, port, has_adopted, HEALTH_PROBE_TIMEOUT)
                .await;
        if liveness.alive {
            // Only end the grace once the whole warm finished: warm imports hold the GIL and can miss probes.
            if liveness.warming_up {
                info!(
                    "Health watchdog: backend on port {} is alive but still warming up, holding the startup grace period",
                    port
                );
            }
            has_seen_healthy =
                watchdog_seen_healthy_after(has_seen_healthy, liveness.alive, liveness.warming_up);
            was_generating = watchdog_inference_active_after(
                was_generating,
                liveness.alive,
                liveness.inference_active,
            );
            consecutive_failures = 0;
        } else if !should_count_failure {
            info!(
                "Health watchdog: startup health check failed on port {} before grace period elapsed",
                port
            );
        } else {
            let budget = watchdog_failure_budget(was_generating, liveness.probe_timed_out);
            consecutive_failures += 1;
            warn!(
                "Health watchdog: failure {}/{} on port {}{}",
                consecutive_failures,
                budget,
                port,
                if budget > HEALTH_WATCHDOG_MAX_FAILURES {
                    " (backend last seen generating, probe timed out)"
                } else {
                    ""
                }
            );
            if watchdog_should_confirm_before_death(
                consecutive_failures,
                budget,
                liveness.probe_timed_out,
            ) {
                // Before killing, one wider probe; only a backend reporting it is generating gets the
                // count reset.
                let confirmed = check_watchdog_health(
                    &state,
                    generation,
                    port,
                    has_adopted,
                    HEALTH_CONFIRM_PROBE_TIMEOUT,
                )
                .await;
                if watchdog_confirm_keeps_backend(&confirmed) {
                    warn!(
                        "Health watchdog: backend on port {} answered the last-chance probe and is still generating, not declaring it dead",
                        port
                    );
                    has_seen_healthy = watchdog_seen_healthy_after(
                        has_seen_healthy,
                        confirmed.alive,
                        confirmed.warming_up,
                    );
                    was_generating = true;
                    consecutive_failures = 0;
                    continue;
                }
            }
            // Re-read before acting: a restart during the probes would make stop_backend kill the replacement.
            let (current_generation, still_owned) = {
                let proc = match state.lock() {
                    Ok(p) => p,
                    Err(_) => break,
                };
                (proc.generation, proc.has_owned_backend())
            };
            if !watchdog_may_still_act(
                current_generation,
                generation,
                still_owned,
                shutdown.load(Ordering::SeqCst),
            ) {
                info!(
                    "Health watchdog: backend changed while the health probe was in flight, exiting without declaring it dead"
                );
                break;
            }
            if consecutive_failures >= budget {
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
                    let _ = process::stop_backend(&state, &shutdown, Some(&diagnostics));
                }
                let _ = app.emit("server-crashed", ());
                break;
            }
        }
    }

    process::clear_adopted_watchdog_if_current(&state, generation);
}
