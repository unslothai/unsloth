use crate::diagnostics::{self, DiagnosticsState};
use crate::install;
use crate::process::{self, BackendState, ShutdownFlag};
use crate::desktop_updater;
use crate::staged_update;
use crate::update;
use log::{error, info, warn};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter};

const BACKEND_STARTUP_GRACE_PERIOD: Duration = Duration::from_secs(5 * 60);
const HEALTH_WATCHDOG_INTERVAL: Duration = Duration::from_secs(15);
const HEALTH_WATCHDOG_MAX_FAILURES: u32 = 3;
/// Strikes allowed when the last answered probe said the backend was generating and the
/// ones since then timed out rather than being refused.
///
/// Three strikes is ~75s of silence here, not 45s: the loop sleeps HEALTH_WATCHDOG_INTERVAL
/// and only then probes, so a strike that TIMES OUT costs 15s + 10s, where a refused one
/// costs only the 15s. A saturated host clears 75s easily -- a model that does not fit runs
/// at fractions of a token per second, and the loop serving it can miss several probes in a
/// row while the response is still being produced. Killing there ends a stream the user is
/// waiting on and reports it as "Server stopped unexpectedly". Twelve strikes is ~300s.
///
/// Only the stalled case gets this. A refused connection still counts against the plain
/// budget, so a backend that really died is reported just as fast as before.
const HEALTH_WATCHDOG_MAX_FAILURES_BUSY: u32 = 12;
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
/// Budget for the single last-chance probe spent before a stalled backend is declared dead.
///
/// Deliberately above HEALTH_WATCHDOG_INTERVAL, unlike the per-cycle budget: this one is not
/// part of the cadence. It runs at most once per backend, and only when the alternative is
/// killing it, so overlapping the next probe is not a concern -- there is no next probe if
/// this one does not save the backend. 30s covers a loop running a couple of probe budgets
/// behind, which is the regime where a generation is still producing tokens but nothing
/// answers within 10s.
const HEALTH_CONFIRM_PROBE_TIMEOUT: Duration = Duration::from_secs(30);

fn should_count_watchdog_failure(has_seen_healthy: bool, elapsed_since_start: Duration) -> bool {
    has_seen_healthy || elapsed_since_start >= BACKEND_STARTUP_GRACE_PERIOD
}

/// Whether the startup grace has ended, given what the latest probe reported.
///
/// A backend that answers while still warming clears the latch instead of leaving it: an
/// adopted backend starts latched (it was serving before this app attached), so a watchdog
/// that only declined to *set* it would keep counting failures against a host that has just
/// said it is still importing the ML stack. A probe that got no answer leaves the latch
/// alone, since silence says nothing about whether startup finished.
fn watchdog_seen_healthy_after(previous: bool, alive: bool, warming_up: bool) -> bool {
    if !alive {
        return previous;
    }
    !warming_up
}

/// Whether the backend was generating, given what the latest probe reported.
///
/// Same shape as the warm-up latch above: only an answer updates it. A probe that got
/// nothing leaves the last answer standing, which is the whole point here, since the
/// stall being ridden out is exactly when no answer comes back.
fn watchdog_inference_active_after(previous: bool, alive: bool, inference_active: bool) -> bool {
    if !alive {
        return previous;
    }
    inference_active
}

/// What a failed probe says about the port.
///
/// A spent budget means a process accepted the connection and did not answer, which is the
/// stall the busy budget exists for. Anything else (refused, reset, no route) means nothing
/// is serving there, and that is death.
fn liveness_from_probe_error(error: &reqwest::Error) -> BackendLiveness {
    BackendLiveness {
        probe_timed_out: error.is_timeout(),
        ..BackendLiveness::default()
    }
}

/// Whether a failed adopted-path check is a stall rather than a dead port.
///
/// The ownership probe cannot answer this on its own: every transport error inside it becomes
/// `NotVerified`, indistinguishable from a port that a different process now owns. The
/// pre-probe can: it answered, from an Unsloth backend, on this port. Silence from the
/// requests after that is the same stall the owned path already rides out.
///
/// `different_owner` is the one thing the probe *can* say for certain, so it overrides both:
/// an adopted backend that exits while the busy latch is set leaves a freed port, and a
/// restarted Unsloth backend binding it answers the pre-probe just as the old one did. That
/// answer is not silence -- the ownership probe got a complete reply naming a different root,
/// token or no desktop owner at all -- so it is a takeover, not a stall, and the port must be
/// cleared on the normal three-strike budget rather than held for twelve.
fn adopted_failure_is_a_stall(verified: bool, served_alive: bool, different_owner: bool) -> bool {
    !verified && served_alive && !different_owner
}

/// Whether the last-chance probe brought back evidence that a generation is still running.
///
/// Not `alive && inference_active`. On the adopted path `alive` means "the ownership probe
/// re-verified us", and that probe's two extra loopback requests are exactly what a
/// saturated backend drops -- the one this probe exists to reach. Such a backend answers
/// the pre-probe with the busy marker set and then goes quiet, so requiring verification
/// threw away the answer and cleared a backend mid-response, which is the loss this whole
/// path is for. A stalled port that says it is generating is kept; silence is not, and a
/// port a different Unsloth backend took over never reaches here with the marker set.
fn watchdog_confirm_keeps_backend(confirmed: &BackendLiveness) -> bool {
    confirmed.inference_active && (confirmed.alive || confirmed.probe_timed_out)
}

/// Whether to spend one long last-chance probe before declaring a backend dead.
///
/// The busy budget can only widen on a marker that some probe actually brought back, and
/// there is a window where none can: a generation that starts just after an idle answer
/// and saturates the loop before the next probe leaves the latch false for the rest of its
/// life, so it dies on the plain three-strike budget with a response in flight -- the exact
/// loss this change is for. Nothing in the backend can close that window, because a starved
/// loop cannot answer at all, and the desktop UI cannot either: #8945 is an external
/// agentic client posting straight to `/v1/messages`, with no frontend in the path.
///
/// What can close it is asking once more, with a budget wide enough that a loop running
/// tens of seconds behind still gets a word in. Only ever spent on a stall: a refused port
/// is death and is still reported at three strikes, at the same speed, with no extra wait.
fn watchdog_should_confirm_before_death(
    consecutive_failures: u32,
    budget: u32,
    probe_timed_out: bool,
) -> bool {
    consecutive_failures >= budget && probe_timed_out
}

/// Whether the watchdog may still act on the backend it set out to watch.
///
/// Every probe is an await, and the last-chance one is 30s wide on top of the 10s cycle
/// probe, so up to 40s of wall clock passes between reading the state at the top of the
/// loop and spending the verdict on it. A stop or a restart lands in that window as a
/// swapped handle: `start_backend` bumps the generation and stores a new child, and
/// `stop_backend` has no generation guard of its own, so a watchdog acting on its
/// pre-await snapshot kills the replacement and reports the healthy new backend as
/// crashed. The generation is the identity the probe never carried, and the shutdown flag
/// covers the stop that has not been followed by a start yet.
fn watchdog_may_still_act(
    current_generation: u64,
    watched_generation: u64,
    has_owned: bool,
    shutting_down: bool,
) -> bool {
    current_generation == watched_generation && has_owned && !shutting_down
}

/// Consecutive failures tolerated before the backend is declared dead.
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
    let generation = match process::start_backend(&app, &state, port, &shutdown, &diagnostics_state)
    {
        Ok(generation) => generation,
        Err(_error) if process::request_staged_rollback_restart(&app, &state) => return Ok(()),
        Err(error) => return Err(error),
    };

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
    let generation = match process::start_backend(&app, &state, port, &shutdown, &diagnostics_state)
    {
        Ok(generation) => generation,
        Err(_error) if process::request_staged_rollback_restart(&app, &state) => return Ok(()),
        Err(error) => return Err(error),
    };

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
    /// The backend answered and had at least one generation in flight. On the adopted path
    /// this is the pre-probe's reading, which survives an ownership re-check that went
    /// silent; readers that need "answered right now" pair it with `alive`.
    inference_active: bool,
    /// The probe ran out of budget rather than being refused, so a process is still
    /// holding the port. Silence from a closed port is death; silence from an accepted
    /// connection is a stall.
    probe_timed_out: bool,
}

/// Check if an Unsloth backend is running on the given port.
/// Expects JSON with status=="alive" (or "healthy") AND service=="Unsloth UI Backend".
#[tauri::command]
pub async fn check_health(port: u16) -> Result<bool, String> {
    match check_health_inner(port, HEALTH_PROBE_TIMEOUT).await {
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
///
/// The budget is a parameter rather than the constant, because the last-chance probe the
/// watchdog spends before declaring a stalled backend dead needs a wider one: a loop that
/// is starved past 10s is exactly the loop whose answer decides whether a generation is
/// still running, and abandoning that read at 10s is what leaves the question unanswered.
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

    // Absent on a backend too old to publish it, which reads as "not busy": the widened
    // budget is the only thing such a backend loses.
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
    // Ask the backend before asking who owns it. The ownership probe folds every transport
    // error into "not verified", so classifying the failure has to happen here or a stalled
    // adopted backend is indistinguishable from a dead one and loses the busy budget.
    //
    // Nothing is skipped by returning early: the probe's own first step is this same GET,
    // on the same port with the same budget and the same /api/liveness -> /api/health
    // fallback, so a port that answers nothing here cannot verify ownership either.
    let served = match check_health_inner(port, budget).await {
        Ok(liveness) => liveness,
        Err(error) => return liveness_from_probe_error(&error),
    };
    // HEALTH_PROBE_TIMEOUT, not the ownership path's default 2s. Every request inside the
    // probe would otherwise time out during the very GIL stall this watchdog is supposed to
    // ride out, so the backend came back unverified and the warm-up read below -- which is
    // gated on that verification -- never ran at all.
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
    // The one failure the probe can be certain about: a complete answer naming an owner that
    // is not ours. Everything else it reports could be silence.
    let different_owner = crate::desktop_backend_owner::probe_saw_a_different_owner(&ownership);
    // The ownership probe answers "is this still our process", not "is startup over", which
    // is why the read above is consulted as well. An adopted backend having served one
    // request before this app attached is the same fallacy the owned path fixes below: a
    // force-quit during a cold start leaves the backend running and still importing the ML
    // stack, and the app relaunches straight onto it. Without a warm-up signal that host is
    // declared dead three probes later while it is perfectly healthy.
    //
    adopted_backend_liveness(verified, &served, different_owner)
}

/// Fold the adopted path's two readings -- what the backend served and what the ownership
/// probe made of it -- into one verdict.
///
/// `alive` and `warming_up` stay gated on verification, so a foreign process on the port
/// cannot hold the startup grace open.
fn adopted_backend_liveness(
    verified: bool,
    served: &BackendLiveness,
    different_owner: bool,
) -> BackendLiveness {
    BackendLiveness {
        alive: verified,
        warming_up: verified && served.warming_up,
        // NOT gated on verification, for the same reason `probe_timed_out` below is not.
        // The ownership probe's extra requests are the ones a saturated backend drops, so
        // requiring verification here discards the one thing the pre-probe did bring back:
        // this backend, on this port, said a generation was in flight. That is precisely
        // the evidence the last-chance confirmation spends, and gating it meant a stalled
        // adopted backend answered "still generating" and was cleared anyway, mid-response.
        //
        // A takeover still clears it: `different_owner` is a complete answer naming someone
        // else, so the generation it reports is not ours to protect. Every other reader
        // gates on `alive` already (`watchdog_inference_active_after` returns the previous
        // latch when the probe did not answer), so nothing else changes shape here.
        inference_active: served.inference_active && !different_owner,
        // The read above answering does not mean the whole check answered. The ownership
        // probe issues two more loopback requests -- its own `/api/liveness` and the
        // `/api/auth/desktop-login` compatibility POST -- each with its own budget, and it
        // folds a timeout on either into "not verified". Hard-coding this false there gave
        // the three-strike budget to an adopted backend that had just told us it was
        // generating, which is the kill this whole change exists to prevent. Worse, the
        // narrower budget is re-read on every failure, so one such cycle could fire the
        // kill immediately on a count already past three.
        //
        // The read is the evidence: an Unsloth backend answered on this port, so silence
        // from the rest of the check is a stall, not an empty port. A refused port never
        // reaches here -- it returns above with `probe_timed_out` set from the error, and a
        // port a different Unsloth backend has taken over is excluded by `different_owner`,
        // since that one answered rather than fell silent.
        probe_timed_out: adopted_failure_is_a_stall(verified, served.alive, different_owner),
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
    open_existing_dir_with(dir, |path| crate::process::open_detached(path))
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
    backend_state: tauri::State<'_, BackendState>,
    diagnostics: tauri::State<'_, DiagnosticsState>,
) -> Result<(), String> {
    if has_owned_backend(&backend_state)? {
        return Err(
            "The Unsloth backend is still running. Stop it before starting installation."
                .to_string(),
        );
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

#[tauri::command]
pub async fn start_staged_update(
    app: AppHandle,
    update_state: tauri::State<'_, update::UpdateState>,
    install_state: tauri::State<'_, install::InstallState>,
    desktop_update: tauri::State<'_, desktop_updater::DesktopUpdateState>,
    diagnostics: tauri::State<'_, DiagnosticsState>,
) -> Result<(), String> {
    info!("start_staged_update command called");

    if install_state
        .lock()
        .map(|s| s.child.is_some())
        .unwrap_or(false)
    {
        return Err("Cannot prepare an update while installation is in progress.".to_string());
    }
    if update::is_staged_update_running(&update_state) || update::is_update_running(&update_state) {
        return Err("Update is already running.".to_string());
    }

    let shell_version = desktop_updater::pending_version(&desktop_update)?;
    let backend_version = desktop_updater::pending_backend_version(&desktop_update)?;
    let state = update_state.inner().clone();
    let diagnostics_state = diagnostics.inner().clone();
    tokio::task::spawn_blocking(move || {
        update::run_staged_update(app, state, diagnostics_state, shell_version, backend_version)
    })
    .await
    .map_err(|e| format!("Staged update task panicked: {e}"))?
}

#[tauri::command]
pub fn cancel_staged_update(
    update_state: tauri::State<'_, update::UpdateState>,
    diagnostics: tauri::State<'_, DiagnosticsState>,
) -> Result<(), String> {
    let staged = update_state
        .lock()
        .map(|s| s.child.is_some() && s.staged)
        .unwrap_or(false);
    if !staged {
        return Ok(());
    }
    update::record_update_intentional_stop(&update_state, &diagnostics);
    update::stop_update(&update_state)?;
    process::with_studio_runtime_launch_guard(|| {
        staged_update::discard(&diagnostics::studio_dir());
        Ok(())
    })
}

#[tauri::command]
pub fn staged_update_status(
    update_state: tauri::State<'_, update::UpdateState>,
) -> staged_update::StagedUpdateStatus {
    let mut status = staged_update::status(&diagnostics::studio_dir());
    status.staging = update::is_staged_update_running(&update_state);
    status.staging_shell_version = update::staged_update_shell_version(&update_state);
    status
}

#[tauri::command]
pub fn discard_staged_update(
    update_state: tauri::State<'_, update::UpdateState>,
) -> Result<(), String> {
    if update::is_update_running(&update_state) {
        return Err("Update is already running.".to_string());
    }
    process::with_studio_runtime_launch_guard(|| {
        staged_update::discard(&diagnostics::studio_dir());
        Ok(())
    })
}

/// Repair a stale managed Unsloth install.
/// Whether a native path lease this app signs can actually be verified.
///
/// The key is per process, so only a backend THIS process spawned holds it. An
/// adopted survivor and an attached terminal-started backend both advertise
/// `native_path_leases_supported` with a key that is not ours, and the boolean
/// cannot tell them apart, so answer from the positive fact instead. Restarting
/// them would also work, but adoption exists so a possibly mid-training backend
/// is not killed, and the UI's only restart path runs a network update.
#[tauri::command]
pub async fn native_path_leases_usable(
    backend_state: tauri::State<'_, BackendState>,
) -> Result<bool, String> {
    Ok(
        matches!(process::owned_backend_snapshot(backend_state.inner())?,
            Some(snapshot) if !snapshot.is_adopted),
    )
}

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

        let liveness = super::check_health_inner(port, super::HEALTH_PROBE_TIMEOUT)
            .await
            .unwrap();

        assert!(liveness.alive);
        assert!(!liveness.warming_up);
        assert_eq!(paths.lock().unwrap().as_slice(), ["/api/liveness"]);
    }

    #[tokio::test]
    async fn a_backend_without_the_liveness_route_still_validates_through_health() {
        // The desktop app talks to backends of varying versions; a downgrade answers
        // "healthy" from /api/health and 404s liveness.
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

        let liveness = super::check_health_inner(port, super::HEALTH_PROBE_TIMEOUT)
            .await
            .unwrap();

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

        let liveness = super::check_health_inner(port, super::HEALTH_PROBE_TIMEOUT)
            .await
            .unwrap();

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
    fn the_adopted_ownership_probe_uses_the_watchdog_budget() {
        // The warm-up read for an adopted backend is gated on ownership verifying, and that
        // probe defaults to a 2s per-request budget. At 2s every request inside it times out
        // during the very GIL stall the watchdog exists to ride out, so the backend reads as
        // unverified, the warm-up read never runs, and the grace never reopens. Binding the
        // call to HEALTH_PROBE_TIMEOUT here keeps the two from drifting apart again.
        // Normalise line endings first. include_str! embeds the file exactly as checked
        // out, and on Windows that is CRLF, so a bare "\n}\n" never matches and this
        // panicked on the Tauri CI runner while passing everywhere else.
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
        // The other half of the reported crash. A force-quit during a cold start leaves the
        // backend running and still importing the ML stack, and the relaunched app adopts it
        // rather than spawning a new one. Adopted watchdogs start latched, so before this the
        // grace never applied: three GIL-stalled probes cleared the backend at ~45s and put a
        // "server stopped unexpectedly" screen in front of a host that was starting normally.
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
        // alive + done warming latches; alive + warming clears; no answer changes nothing,
        // because a probe that timed out says nothing about whether startup finished.
        assert!(super::watchdog_seen_healthy_after(false, true, false));
        assert!(!super::watchdog_seen_healthy_after(true, true, true));
        assert!(super::watchdog_seen_healthy_after(true, false, false));
        assert!(!super::watchdog_seen_healthy_after(false, false, false));
        // A warm that finishes after a stall re-latches, so the grace does not linger.
        assert!(super::watchdog_seen_healthy_after(false, true, false));
    }

    #[test]
    fn a_backend_that_stalls_while_generating_is_not_killed_at_three_strikes() {
        // #8945. Four concurrent requests against a 27B Q8 on an APU ran at 0.28 tok/s, the
        // loop serving them went quiet, and three probes later the response the user was
        // waiting on was killed and reported as "Server stopped unexpectedly".
        let mut generating = false;
        generating = super::watchdog_inference_active_after(generating, true, true);
        assert!(generating);
        // Silence leaves the last answer standing: no answer is exactly what a stall gives.
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
        // The budget widens for a stall, never for a refusal, so a backend that really
        // exited is reported just as fast as before. Same for one that was idle.
        assert_eq!(
            super::watchdog_failure_budget(true, false),
            super::HEALTH_WATCHDOG_MAX_FAILURES
        );
        assert_eq!(
            super::watchdog_failure_budget(false, true),
            super::HEALTH_WATCHDOG_MAX_FAILURES
        );
    }

    /// A port that completes the handshake and then says nothing, which is what a stalled
    /// backend looks like from here. The accepted streams are parked, not dropped: closing
    /// one would answer the probe with a reset and the request would fail early.
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

    #[tokio::test]
    async fn a_port_that_accepts_and_never_answers_reads_as_a_stall() {
        // The premise of the busy budget: a saturated backend still holds its port open, so
        // the probe runs out of budget rather than being refused.
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
    async fn a_port_with_nothing_on_it_reads_as_death_not_a_stall() {
        // The other half: a backend that really exited leaves a closed port, and that must
        // still be declared dead at three strikes.
        let port = {
            let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
            let port = listener.local_addr().unwrap().port();
            drop(listener);
            port
        };
        let client = crate::loopback_http::client(Duration::from_secs(5)).unwrap();

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
        // End to end through the same client and parser the watchdog uses, so the field name
        // is checked against what main.py publishes rather than against itself.
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
        // Absent on an older backend, which reads as idle rather than as busy forever.
        assert!(
            !super::check_health_inner(idle_port, super::HEALTH_PROBE_TIMEOUT)
                .await
                .unwrap()
                .inference_active
        );
    }

    #[test]
    fn both_watchdog_paths_classify_a_failed_probe() {
        // The adopted path folds every transport error into "not verified", so it has to
        // classify the failure itself. Hard-coding it false there killed a stalled adopted
        // backend at three strikes while the owned path rode the same stall out.
        // Normalise line endings first: include_str! embeds CRLF on Windows checkouts.
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
        // The latch follows the last answer, so a finished stream drops straight back to
        // three strikes rather than leaving the wide budget armed for the whole session.
        let generating = super::watchdog_inference_active_after(true, true, false);
        assert!(!generating);
        assert_eq!(
            super::watchdog_failure_budget(generating, true),
            super::HEALTH_WATCHDOG_MAX_FAILURES
        );
    }

    #[test]
    fn an_adopted_backend_that_answers_and_then_stalls_keeps_the_busy_budget() {
        // The ownership check issues two more loopback requests after the read that already
        // answered, and folds a timeout on either into "not verified". Reporting that as a
        // refusal handed a generating adopted backend the three-strike budget.
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
        // The shape the last-chance probe really brings back from a saturated adopted
        // backend: the pre-probe answered with the busy marker, and the ownership probe's
        // two extra loopback requests timed out behind it, so nothing is verified. Judging
        // that on `alive` threw the answer away and cleared the backend with the response
        // still streaming, which is the kill this whole path exists to prevent.
        let served = super::BackendLiveness {
            alive: true,
            warming_up: false,
            inference_active: true,
            probe_timed_out: false,
        };
        let confirmed = super::adopted_backend_liveness(false, &served, false);
        assert!(
            !confirmed.alive,
            "an unverified re-check is not a live answer"
        );
        assert!(super::watchdog_confirm_keeps_backend(&confirmed));

        // Silence is still death: the same unverified re-check with nothing served.
        let quiet =
            super::adopted_backend_liveness(false, &super::BackendLiveness::default(), false);
        assert!(!super::watchdog_confirm_keeps_backend(&quiet));

        // So is an idle backend that answered: only a generation buys the reprieve.
        let idle = super::adopted_backend_liveness(
            true,
            &super::BackendLiveness {
                alive: true,
                warming_up: false,
                inference_active: false,
                probe_timed_out: false,
            },
            false,
        );
        assert!(!super::watchdog_confirm_keeps_backend(&idle));

        // And a port a different Unsloth backend took over answers with its own generation,
        // which is not ours to hold the port open for.
        let taken_over = super::adopted_backend_liveness(false, &served, true);
        assert!(!super::watchdog_confirm_keeps_backend(&taken_over));
    }

    #[test]
    fn a_port_another_backend_took_over_stays_on_the_normal_budget() {
        // An adopted backend that exits mid-generation leaves the busy latch set, and the
        // freed port is routinely rebound by the next Unsloth backend the user starts. That
        // one answers the pre-probe exactly as the old one did, so `served.alive` is true
        // while the ownership probe rejects it -- with a complete answer, not silence.
        //
        // Calling that a stall gave a foreign port twelve strikes instead of three: 12 * 15s
        // = ~180s of a backend the app still shows as running, against ~45s, plus a
        // last-chance probe that is spent for nothing.
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
        // The read is the only evidence there is. Without it the failure is a dead port,
        // and a dead port must still be reported at three strikes.
        assert!(!super::adopted_failure_is_a_stall(false, false, false));
        assert_eq!(
            super::watchdog_failure_budget(
                true,
                super::adopted_failure_is_a_stall(false, false, false)
            ),
            super::HEALTH_WATCHDOG_MAX_FAILURES
        );
        // And a check that passed is not a failure at all, so it is not a stall either.
        assert!(!super::adopted_failure_is_a_stall(true, true, false));
    }

    #[test]
    fn a_generation_that_starts_between_probes_still_gets_one_last_chance() {
        // The latch can only be set by a probe that answered. A generation that starts just
        // after an idle answer and saturates the loop before the next probe never sets it,
        // so it dies on three strikes with a response in flight unless something asks again.
        assert!(super::watchdog_should_confirm_before_death(
            super::HEALTH_WATCHDOG_MAX_FAILURES,
            super::HEALTH_WATCHDOG_MAX_FAILURES,
            true,
        ));
        // Not before the budget is actually spent: the normal cadence is untouched.
        assert!(!super::watchdog_should_confirm_before_death(
            super::HEALTH_WATCHDOG_MAX_FAILURES - 1,
            super::HEALTH_WATCHDOG_MAX_FAILURES,
            true,
        ));
        // And never for a refused port, so a backend that really exited is reported at the
        // same speed as before, with no extra wait bolted on.
        assert!(!super::watchdog_should_confirm_before_death(
            super::HEALTH_WATCHDOG_MAX_FAILURES,
            super::HEALTH_WATCHDOG_MAX_FAILURES,
            false,
        ));
    }

    #[test]
    fn the_last_chance_budget_is_wider_than_the_one_that_gave_up() {
        // The point of the extra probe is a budget the per-cycle one cannot afford. Equal
        // budgets would just repeat the read that already failed.
        assert!(super::HEALTH_CONFIRM_PROBE_TIMEOUT > super::HEALTH_PROBE_TIMEOUT);
    }

    /// What one watchdog cycle got back from the port.
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

    /// Replay a timeline through the watchdog's failure accounting and report the wall
    /// clock at which the backend was declared dead, or `None` if it survived.
    ///
    /// Mirrors the loop in `health_watchdog` rather than calling it: the loop needs an
    /// `AppHandle` and a real 15s sleep per cycle. Every decision it makes comes from the
    /// helpers below, so the arithmetic being checked here is the arithmetic it runs.
    /// `confirms` supplies what the last-chance probe gets back, in order.
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

        // A healthy idle backend is never touched.
        assert_eq!(simulate_watchdog(&vec![answered(false); 200], &[]), None);
        // Neither is one that answers while it generates, which is the #8945 log: liveness
        // came back every 15s right up to the end.
        assert_eq!(simulate_watchdog(&vec![answered(true); 200], &[]), None);
        // A backend still importing the ML stack answers, so it never counts a failure.
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

        // A backend that exited leaves a refused port, and dies at three strikes with no
        // last-chance wait bolted on: 3 x 15s, exactly what it cost before this change.
        let mut dead = vec![answered(false)];
        dead.extend(vec![Probe::Refused; plain]);
        assert_eq!(simulate_watchdog(&dead, &[]), Some(interval * 4));
        // Even one that was generating when it went: the busy budget is for silence from a
        // port that is still accepting, never for a refusal.
        let mut died_generating = vec![answered(true)];
        died_generating.extend(vec![Probe::Refused; plain]);
        assert_eq!(simulate_watchdog(&died_generating, &[]), Some(interval * 4));

        // The regression this PR is for: answered busy, then the loop goes quiet. It has to
        // survive far past three strikes.
        let mut stalling = vec![answered(true)];
        stalling.extend(vec![Probe::TimedOut; busy - 1]);
        assert_eq!(simulate_watchdog(&stalling, &[]), None);

        // A backend that is genuinely wedged is still killed, just later. Bounded, and the
        // bound is checked so nobody can widen it by accident.
        let mut wedged = vec![answered(true)];
        wedged.extend(vec![Probe::TimedOut; busy + 4]);
        assert_eq!(
            simulate_watchdog(&wedged, &[Probe::TimedOut]),
            Some(interval + busy as u64 * stall_cycle + confirm_budget)
        );

        // Codex #3: the generation starts between two probes, so no answer ever reports it.
        // Before the last-chance probe this died at three strikes; now the probe that a
        // starved loop can still answer within 30s reports the generation and it lives.
        let mut started_between_probes = vec![answered(false)];
        started_between_probes.extend(vec![Probe::TimedOut; 40]);
        assert_eq!(
            simulate_watchdog(
                &started_between_probes,
                &[answered(true); 8] // one per spent budget
            ),
            None
        );
        // Without an answer to that probe it still dies, one confirm budget later than the
        // three strikes it used to take. That is the whole cost to a hung idle backend.
        assert_eq!(
            simulate_watchdog(&started_between_probes, &[Probe::TimedOut]),
            Some(interval + plain as u64 * stall_cycle + confirm_budget)
        );
        // And an answer that says the backend is idle is not a reprieve: a backend that is
        // alive but unresponsive and doing nothing dies exactly where it did before.
        assert_eq!(
            simulate_watchdog(&started_between_probes, &[answered(false)]),
            Some(interval + plain as u64 * stall_cycle)
        );

        // A stream that finishes hands the wide budget straight back.
        let mut finished = vec![answered(true), answered(false)];
        finished.extend(vec![Probe::Refused; plain]);
        assert_eq!(simulate_watchdog(&finished, &[]), Some(interval * 5));

        // Mixed causes: stalling on the busy budget and then the port disappears. The
        // budget is re-read per failure, so the refusal ends it immediately rather than
        // waiting out the remaining busy strikes.
        let mut stalled_then_gone = vec![answered(true)];
        stalled_then_gone.extend(vec![Probe::TimedOut; 5]);
        stalled_then_gone.push(Probe::Refused);
        assert_eq!(
            simulate_watchdog(&stalled_then_gone, &[]),
            Some(interval + 5 * stall_cycle + interval)
        );

        // A backend that answers every other probe never accumulates a budget at all.
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

    /// A port that answers, but only after `delay`. This is the regime the last-chance
    /// probe exists for: the loop is running behind, not gone.
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
        // Measured, not asserted from the constants: the same port, the same parser, one
        // budget that expires before the answer and one that does not.
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
        // The watched generation is the only identity this task has: `check_health_inner`
        // matches on a service name, so a probe answer says "an Unsloth backend is on this
        // port", never "the one I was started for". Anything that is not still generation
        // G with a handle stored and no stop in flight has to end the loop rather than
        // reach `stop_backend`, which takes whatever handle is stored *now*.
        assert!(super::watchdog_may_still_act(7, 7, true, false));
        // Restarted under the probe: start_backend bumped the generation and stored a new
        // child, so the kill below would land on the replacement.
        assert!(!super::watchdog_may_still_act(8, 7, true, false));
        // Stopped and not started again: the handle is gone, so there is nothing to kill
        // and "server stopped unexpectedly" would contradict the stop the user asked for.
        assert!(!super::watchdog_may_still_act(7, 7, false, false));
        // A stop in flight: stop_backend sets the flag before it takes the handle, so this
        // is the same restart caught one instant earlier.
        assert!(!super::watchdog_may_still_act(7, 7, true, true));
    }

    #[test]
    fn the_watchdog_rereads_the_generation_after_the_confirm_probe() {
        // HEALTH_CONFIRM_PROBE_TIMEOUT is 30s on top of the 10s cycle probe, so up to 40s
        // of wall clock separates the generation check at the top of the loop from the
        // kill at the bottom, and both stop_server and start_server are async commands
        // that run while this task is parked on the await. Losing the re-read reinstates a
        // 40s window in which a user restarting a hung backend has the new one killed.
        // Normalised line endings for the same reason as the guard above: include_str!
        // embeds CRLF on the Windows runner.
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

/// Periodic health check that detects deadlocked or hung backends.
/// During startup, failures are ignored for a generous grace period so a slow
/// but legitimate backend boot is not killed. After the backend has answered a probe
/// that says its warm-up is finished, or after the startup grace expires, 3 consecutive
/// failed checks emit `server-crashed` so the frontend can offer a restart.
/// A backend last seen generating gets the wider `HEALTH_WATCHDOG_MAX_FAILURES_BUSY`
/// budget for probes that time out, since a saturated host stalls one that is healthy.
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
                let stopped = process::stop_backend(&state, &shutdown, Some(&diagnostics));
                if stopped.is_ok() && process::request_staged_rollback_restart(&app, &state) {
                    break;
                }
                let _ = app.emit("server-crashed", ());
                break;
            }
            continue;
        };

        let liveness =
            check_watchdog_health(&state, generation, port, has_adopted, HEALTH_PROBE_TIMEOUT)
                .await;
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
                // The budget is spent and the port is still accepting connections. Before
                // ending a response that may still be streaming, ask once with a budget the
                // per-cycle one cannot afford. An answer here is not a reprieve on its own:
                // only a backend that says it is generating gets the count reset, so one
                // that is alive but idle and unresponsive still dies, as it did before.
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
            // The verdict below is about the backend that was current when this cycle read
            // the state, and both probes above have awaited since. Re-read before spending
            // it: `stop_backend` acts on whatever handle is stored now, so a restart during
            // the last-chance probe would have this task kill the replacement instead.
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
                    // Kill the zombie process so retry can start fresh
                    let stopped = process::stop_backend(&state, &shutdown, Some(&diagnostics));
                    if stopped.is_ok() && process::request_staged_rollback_restart(&app, &state) {
                        break;
                    }
                }
                let _ = app.emit("server-crashed", ());
                break;
            }
        }
    }

    process::clear_adopted_watchdog_if_current(&state, generation);
}
