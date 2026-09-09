use crate::diagnostics::{self, AttemptLog, DiagnosticsState};
use crate::process::trim_line_endings;
use log::{error, info, warn};
use process_wrap::std::*;
use std::io::BufRead;
use std::process::{Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};


#[derive(Default)]
pub struct UpdateProcess {
    pub child: Option<Box<dyn ChildWrapper + Send>>,
    pub intentional_stop: bool,
    pub current_attempt: Option<AttemptLog>,
}

pub type UpdateState = Arc<Mutex<UpdateProcess>>;

pub fn new_update_state() -> UpdateState {
    Arc::new(Mutex::new(UpdateProcess::default()))
}

const UPDATE_ARGS: &[&str] = &["studio", "update"];
const PREFETCH_ARGS: &[&str] = &["studio", "prefetch-update"];
const SHELL_VERSION_ENV: &str = "UNSLOTH_TAURI_SHELL_VERSION";

/// typer answers an unknown subcommand with click's usage exit, so a backend that
/// predates PR C fails the prefetch this way and only this way. Paired with the
/// message below, because 2 on its own is also how a bad option exits.
const PREFETCH_UNSUPPORTED_EXIT: i32 = 2;
const PREFETCH_UNSUPPORTED_MESSAGE: &str = "No such command";
/// `_studio_prefetch.EXIT_BUSY`: a prefetch is already running, which is not a
/// failure to report to anyone.
const PREFETCH_BUSY_EXIT: i32 = 3;

/// The offer stands and the swap is the classic update; there is nothing to say.
pub const PREFETCH_UNSUPPORTED: &str = "prefetch-unsupported";
/// Another prefetch owns the work. Whatever it produces is what gets adopted.
pub const PREFETCH_BUSY: &str = "prefetch-busy";

pub(crate) enum UpdateKind {
    Backend,
    Repair(String),
    /// Background download into the uv cache. Touches no installed file, so it
    /// carries none of the protections the two above need.
    Prefetch {
        shell_version: Option<String>,
    },
}

impl UpdateKind {
    fn args(&self) -> &'static [&'static str] {
        match self {
            UpdateKind::Prefetch { .. } => PREFETCH_ARGS,
            _ => UPDATE_ARGS,
        }
    }

    fn progress_event(&self) -> &'static str {
        match self {
            UpdateKind::Backend => "update-progress",
            UpdateKind::Repair(_) => "repair-progress",
            UpdateKind::Prefetch { .. } => "prefetch-progress",
        }
    }

    fn terminal_events(&self) -> Option<(&'static str, &'static str)> {
        match self {
            UpdateKind::Backend => Some(("update-complete", "update-failed")),
            UpdateKind::Repair(_) => None,
            UpdateKind::Prefetch { .. } => Some(("prefetch-complete", "prefetch-failed")),
        }
    }

    /// False for the prefetch, and that is the whole reason it is a separate
    /// command: no runtime gate, no idle scan, no handoff env, so it can run
    /// beside a working backend without blocking anything or being blocked.
    fn mutates_live_environment(&self) -> bool {
        !matches!(self, UpdateKind::Prefetch { .. })
    }
}

fn build_update_command(bin: &std::path::Path, args: &[&str]) -> Result<Command, String> {
    // Only the Windows arm below mutates it.
    #[cfg_attr(not(windows), allow(unused_mut))]
    // Isolated, as this call site shipped: it is the one managed invocation nobody types by
    // hand and the one that decides which install gets rewritten, so a user-site unsloth_cli
    // must not answer `from unsloth_cli import app` here.
    let mut cmd = crate::process::build_managed_cli_command_with(
        bin,
        args,
        crate::process::Isolation::Isolated,
    )?;
    // The only managed invocation that scrubs: a foreign PYTHONHOME stops the managed
    // interpreter finding its own site-packages, and a PYTHONPATH pointing at another checkout
    // updates the wrong install.
    cmd.env_remove("PYTHONHOME");
    cmd.env_remove("PYTHONPATH");
    Ok(cmd)
}

fn configure_tauri_update_environment(cmd: &mut Command) {
    // The desktop owns its shortcuts and frontend bundle; this update needs only backend deps.
    cmd.env_remove("UNSLOTH_STUDIO_HOME");
    cmd.env_remove("STUDIO_HOME");
    cmd.env("UNSLOTH_TAURI_UPDATE", "1");
    cmd.env("SKIP_STUDIO_FRONTEND", "1");
    cmd.env(
        "UNSLOTH_DESKTOP_BACKEND_VERSION",
        crate::preflight::expected_backend_version(),
    );
}

// The shell holds the retained POSIX flock around the whole update child, so the CLI must
// inherit the gate rather than take it again. Set everywhere, as Windows always did.
fn configure_runtime_gate_environment(cmd: &mut Command) {
    cmd.env(crate::process::STUDIO_RUNTIME_GATE_HANDOFF_ENV, "1");
}

type ChildStreams = (
    Option<std::process::ChildStdout>,
    Option<std::process::ChildStderr>,
);

/// Everything an update child and a prefetch child both need. The gate handoff is
/// deliberately NOT here: it is the one thing the two do not share.
fn prepare_child_command(bin: &std::path::Path, args: &[&str]) -> Result<Command, String> {
    let mut cmd = build_update_command(bin, args)?;
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    // A login-started desktop inherits C:\Windows\system32, which the CLI refuses to run from.
    crate::process::apply_managed_cli_context(&mut cmd).map_err(|error| {
        format!(
            "Failed to pick a working directory for the update: {}",
            error
        )
    })?;

    // PYTHONPATH is dropped by the context itself on Windows, where -I covers only the first
    // interpreter and the update starts more.

    #[cfg(target_os = "linux")]
    crate::process::scrub_appimage_python_env(&mut cmd);

    // Keep the update on the desktop-managed install and skip assets already in the bundle.
    configure_tauri_update_environment(&mut cmd);

    // read_lossy_lines decodes as UTF-8; the child is Python, which otherwise uses the locale page.
    #[cfg(windows)]
    {
        cmd.env("PYTHONUTF8", "1");
        cmd.env("PYTHONIOENCODING", "utf-8");
    }

    Ok(cmd)
}

/// Start the prepared command and hand its pipes back, with the caller's lock on
/// the slot still held so nothing can take it in between.
#[cfg_attr(not(windows), allow(unused_mut))]
fn spawn_prepared(mut cmd: Command, update: &mut UpdateProcess) -> Result<ChildStreams, String> {
    #[cfg(windows)]
    let mut child: Box<dyn ChildWrapper + Send> = {
        use std::os::windows::process::CommandExt;

        cmd.creation_flags(crate::process::CREATE_NO_WINDOW);
        let child = cmd
            .spawn()
            .map_err(|e| format!("Failed to spawn update: {}", e))?;
        Box::new(child)
    };

    #[cfg(unix)]
    let mut child: Box<dyn ChildWrapper + Send> = {
        let mut wrap = CommandWrap::from(cmd);
        wrap.wrap(ProcessGroup::leader());
        wrap.spawn()
            .map_err(|e| format!("Failed to spawn update: {}", e))?
    };

    let stdout = child.stdout().take();
    let stderr = child.stderr().take();
    update.child = Some(child);
    Ok((stdout, stderr))
}

fn spawn_update(bin: &std::path::Path, state: &UpdateState) -> Result<ChildStreams, String> {
    spawn_child(
        bin,
        state,
        &UpdateKind::Backend,
        "Update is already running.",
    )
}

/// The prefetch child: same isolation and the same managed install, minus the gate.
fn spawn_prefetch(
    bin: &std::path::Path,
    state: &UpdateState,
    kind: &UpdateKind,
) -> Result<ChildStreams, String> {
    spawn_child(bin, state, kind, "A prefetch is already running.")
}

/// The one place a managed CLI child is started for either flow.
fn spawn_child(
    bin: &std::path::Path,
    state: &UpdateState,
    kind: &UpdateKind,
    busy: &str,
) -> Result<ChildStreams, String> {
    let mut update = state.lock().map_err(|e| e.to_string())?;
    if update.child.is_some() {
        return Err(busy.to_string());
    }
    update.intentional_stop = false;

    spawn_prepared(build_child_command(bin, kind)?, &mut update)
}

/// The child's whole command, decided by the kind and nothing else.
///
/// Split out from the spawn so both shapes can be asserted without starting a
/// process: whether the gate handoff is set is the difference between an update
/// and a prefetch, and that is exactly the sort of thing that is only ever wrong
/// once, in a release.
fn build_child_command(bin: &std::path::Path, kind: &UpdateKind) -> Result<Command, String> {
    let mut cmd = prepare_child_command(bin, kind.args())?;
    if kind.mutates_live_environment() {
        configure_runtime_gate_environment(&mut cmd);
    } else {
        // Removed rather than left alone: this process may itself have been
        // started with a handoff, and a prefetch that inherited it would tell the
        // CLI it is holding a gate that nothing is holding for it.
        cmd.env_remove(crate::process::STUDIO_RUNTIME_GATE_HANDOFF_ENV);
    }
    // The version the prefetch records in its marker, so a later reader can tell
    // whether the cache was warmed for the offer it is looking at.
    if let UpdateKind::Prefetch {
        shell_version: Some(version),
    } = kind
    {
        cmd.env(SHELL_VERSION_ENV, version);
    }
    Ok(cmd)
}

// ── Stream ──

fn read_lossy_lines<R: std::io::Read>(
    stream: R,
    mut on_line: impl FnMut(String),
) -> std::io::Result<()> {
    let mut reader = std::io::BufReader::new(stream);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        if reader.read_until(b'\n', &mut buf)? == 0 {
            return Ok(());
        }
        on_line(String::from_utf8_lossy(trim_line_endings(&buf)).into_owned());
    }
}

fn structured_update_error(text: &str) -> Option<String> {
    text.strip_prefix("[TAURI:ERROR] ")
        .map(str::trim)
        .filter(|message| !message.is_empty())
        .map(str::to_owned)
}

fn stream_output(
    app: &AppHandle,
    progress_event: &'static str,
    diagnostics: DiagnosticsState,
    attempt: AttemptLog,
    explicit_error: Arc<Mutex<Option<String>>>,
    stdout: Option<std::process::ChildStdout>,
    stderr: Option<std::process::ChildStderr>,
) -> Vec<std::thread::JoinHandle<()>> {
    let mut threads = Vec::new();

    if let Some(out) = stdout {
        let app_clone = app.clone();
        let diagnostics_clone = diagnostics.clone();
        let attempt_clone = attempt.clone();
        let explicit_error_clone = explicit_error.clone();
        threads.push(std::thread::spawn(move || {
            if let Err(e) = read_lossy_lines(out, |text| {
                diagnostics::append_phase_line(&attempt_clone.handle, "stdout", &text);
                if let Some(step) = text.strip_prefix("[TAURI:STEP] ") {
                    diagnostics::record_step(&diagnostics_clone, &attempt_clone, step);
                } else if let Some(progress) = text.strip_prefix("[TAURI:PROGRESS] ") {
                    diagnostics::record_progress(&diagnostics_clone, &attempt_clone, progress);
                } else if let Some(marker) = text.strip_prefix("[TAURI:DIAG] ") {
                    diagnostics::record_diag_marker(&diagnostics_clone, &attempt_clone, marker);
                }
                if let Some(message) = structured_update_error(&text) {
                    if let Ok(mut error) = explicit_error_clone.lock() {
                        *error = Some(message);
                    }
                }
                info!("[update][stdout] {}", text);
                let _ = app_clone.emit(progress_event, &text);
            }) {
                warn!("[update] Error reading stdout: {}", e);
            }
        }));
    }

    if let Some(err) = stderr {
        let app_clone = app.clone();
        let attempt_clone = attempt.clone();
        threads.push(std::thread::spawn(move || {
            if let Err(e) = read_lossy_lines(err, |text| {
                diagnostics::append_phase_line(&attempt_clone.handle, "stderr", &text);
                warn!("[update][stderr] {}", text);
                let _ = app_clone.emit(progress_event, &text);
            }) {
                warn!("[update] Error reading stderr: {}", e);
            }
        }));
    }

    threads
}


fn wait_for_exit(state: &UpdateState) -> Result<(ExitStatus, bool), String> {
    const MAX_WAIT_ITERATIONS: u32 = 72_000; // 2h at 100ms intervals
    for _ in 0..MAX_WAIT_ITERATIONS {
        let mut update = state.lock().map_err(|e| e.to_string())?;
        let intentional = update.intentional_stop;

        match update.child.as_mut() {
            Some(child) => match child.try_wait() {
                Ok(Some(status)) => {
                    update.child = None;
                    return Ok((status, intentional));
                }
                Ok(None) => {}
                Err(e) => {
                    update.child = None;
                    return Err(format!("Error waiting for update: {}", e));
                }
            },
            None if intentional => return Err(UPDATE_STOPPED.to_string()),
            None => return Err("Update process disappeared unexpectedly.".to_string()),
        }

        drop(update);
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    let _ = stop_update(state);
    Err("Update timed out after 2 hours".to_string())
}


pub fn run_backend_update(
    app: AppHandle,
    state: UpdateState,
    diagnostics: DiagnosticsState,
) -> Result<(), String> {
    run_update(app, state, diagnostics, UpdateKind::Backend)
}

pub(crate) fn run_backend_update_for_repair(
    app: AppHandle,
    state: UpdateState,
    diagnostics: DiagnosticsState,
    repair_group_id: String,
) -> Result<(), String> {
    run_update(app, state, diagnostics, UpdateKind::Repair(repair_group_id))
}

fn run_update(
    app: AppHandle,
    state: UpdateState,
    diagnostics: DiagnosticsState,
    kind: UpdateKind,
) -> Result<(), String> {
    let attempt = match &kind {
        UpdateKind::Repair(group_id) => {
            diagnostics::begin_repair_child(&diagnostics, group_id, "update")
        }
        _ => diagnostics::begin_update_attempt(&diagnostics),
    };
    if let Ok(mut update) = state.lock() {
        update.current_attempt = Some(attempt.clone());
    }

    let bin = match crate::process::find_unsloth_binary() {
        Some(bin) => bin,
        None => {
            let msg = "Unsloth binary not found. Cannot run update.".to_string();
            diagnostics::finish_attempt(&diagnostics, &attempt, None, false, Some(msg.clone()));
            clear_current_attempt(&state);
            return Err(msg);
        }
    };

    info!("[update] Starting backend update via {:?}", bin);
    diagnostics::append_phase_line(
        &attempt.handle,
        "meta",
        &format!("Starting backend update via {:?}", bin),
    );
    let progress_event = kind.progress_event();
    let _ = app.emit(progress_event, "Starting backend update...");

    let explicit_error = Arc::new(Mutex::new(None));
    // Update mutates the managed environment for its whole lifetime. Synchronous, so the
    // thread-owned Win32 mutex never crosses an await.
    let result = crate::process::with_studio_runtime_launch_guard(|| {
        crate::process::ensure_managed_environment_is_idle(&bin)?;
        // Under the gate and after the idle scan. A 805-807 rollback the last launch deferred
        // still names the live runtime as something to undo, and updating on top of that journal
        // has the next idle launch restoring the pre-update trees over everything installed here.
        crate::staged_update::reconcile_before_update(&crate::diagnostics::studio_dir())?;
        let (stdout, stderr) =
            spawn_update(&bin, &state).map_err(|msg| format!("spawn_update: {msg}"))?;
        let threads = stream_output(
            &app,
            progress_event,
            diagnostics.clone(),
            attempt.clone(),
            explicit_error.clone(),
            stdout,
            stderr,
        );

        let result = wait_for_exit(&state);
        for handle in threads {
            let _ = handle.join();
        }
        result
    });
    // Read only after the guard returned, so both reader threads are joined.
    let explicit_error = explicit_error.lock().ok().and_then(|error| error.clone());

    match result {
        Ok((status, _)) if status.success() => {
            diagnostics::finish_attempt(
                &diagnostics,
                &attempt,
                Some(status.to_string()),
                false,
                None,
            );
            clear_current_attempt(&state);
            info!("[update] Backend update complete");
            if let Some((complete, _)) = kind.terminal_events() {
                let _ = app.emit(complete, ());
            }
            Ok(())
        }
        Ok((status, intentional)) if intentional => {
            diagnostics::finish_attempt(
                &diagnostics,
                &attempt,
                Some(status.to_string()),
                true,
                Some(UPDATE_STOPPED.to_string()),
            );
            clear_current_attempt(&state);
            info!("[update] Update stopped intentionally");
            Err(UPDATE_STOPPED.to_string())
        }
        Ok((status, intentional)) => {
            let code = status.code().unwrap_or(-1);
            let msg = explicit_error.unwrap_or_else(|| format!("Update exited with code {}", code));
            diagnostics::finish_attempt(
                &diagnostics,
                &attempt,
                Some(status.to_string()),
                intentional,
                Some(msg.clone()),
            );
            clear_current_attempt(&state);
            error!("[update] {}", msg);
            if let Some((_, failed)) = kind.terminal_events() {
                let _ = app.emit(failed, &msg);
            }
            Err(msg)
        }
        Err(msg) => {
            diagnostics::finish_attempt(&diagnostics, &attempt, None, false, Some(msg.clone()));
            clear_current_attempt(&state);
            error!("[update] {}", msg);
            if let Some((_, failed)) = kind.terminal_events() {
                let _ = app.emit(failed, &msg);
            }
            Err(msg)
        }
    }
}

// ── Prefetch ──

/// The prefetch slot, kept apart from the update slot on purpose.
///
/// `is_update_running` and the quit dialog both read `UpdateState`, and a
/// background download is not a reason to warn anyone about quitting or to refuse
/// a real update. Sharing one slot would make it both.
#[derive(Clone)]
pub struct PrefetchState {
    process: UpdateState,
    /// The offered shell version the RUNNING prefetch was started for.
    ///
    /// The marker on disk only names a prefetch that finished, and a webview
    /// reload loses the renderer's own record, so without this a reloaded window
    /// cannot tell whether the run in progress is preparing the offer it is
    /// showing or an older one.
    running_version: Arc<Mutex<Option<String>>>,
}

pub fn new_prefetch_state() -> PrefetchState {
    PrefetchState {
        process: new_update_state(),
        running_version: Arc::new(Mutex::new(None)),
    }
}

pub fn is_prefetch_running(state: &PrefetchState) -> bool {
    is_update_running(&state.process)
}

pub fn running_prefetch_version(state: &PrefetchState) -> Option<String> {
    state
        .running_version
        .lock()
        .ok()
        .and_then(|version| version.clone())
}

pub fn stop_prefetch(state: &PrefetchState) -> Result<(), String> {
    stop_update(&state.process)
}

/// What the child said about why it stopped, gathered while it was still running.
#[derive(Default)]
struct PrefetchOutcome {
    explicit_error: Option<String>,
    unsupported: bool,
}

fn stream_prefetch_output(
    app: &AppHandle,
    outcome: Arc<Mutex<PrefetchOutcome>>,
    stdout: Option<std::process::ChildStdout>,
    stderr: Option<std::process::ChildStderr>,
) -> Vec<std::thread::JoinHandle<()>> {
    let mut threads = Vec::new();

    // No diagnostics attempt for either stream: this runs on a timer in the
    // background, and filling the support bundle with hourly downloads would
    // push out the update the user actually wants read back to them.
    if let Some(out) = stdout {
        let app_clone = app.clone();
        let outcome_clone = outcome.clone();
        threads.push(std::thread::spawn(move || {
            if let Err(e) = read_lossy_lines(out, |text| {
                if let Some(message) = structured_update_error(&text) {
                    if let Ok(mut outcome) = outcome_clone.lock() {
                        outcome.explicit_error = Some(message);
                    }
                }
                info!("[prefetch][stdout] {}", text);
                let _ = app_clone.emit("prefetch-progress", &text);
            }) {
                warn!("[prefetch] Error reading stdout: {}", e);
            }
        }));
    }

    if let Some(err) = stderr {
        let app_clone = app.clone();
        let outcome_clone = outcome.clone();
        threads.push(std::thread::spawn(move || {
            if let Err(e) = read_lossy_lines(err, |text| {
                // typer prints the usage error here, and only the pair of exit
                // code and message identifies a backend that has no such command.
                if text.contains(PREFETCH_UNSUPPORTED_MESSAGE) {
                    if let Ok(mut outcome) = outcome_clone.lock() {
                        outcome.unsupported = true;
                    }
                }
                info!("[prefetch][stderr] {}", text);
                let _ = app_clone.emit("prefetch-progress", &text);
            }) {
                warn!("[prefetch] Error reading stderr: {}", e);
            }
        }));
    }

    threads
}

/// Map a non-zero prefetch exit onto something the desktop can act on.
fn prefetch_failure(code: i32, outcome: &PrefetchOutcome) -> String {
    if code == PREFETCH_BUSY_EXIT {
        return PREFETCH_BUSY.to_string();
    }
    if code == PREFETCH_UNSUPPORTED_EXIT && outcome.unsupported {
        return PREFETCH_UNSUPPORTED.to_string();
    }
    outcome
        .explicit_error
        .clone()
        .unwrap_or_else(|| format!("Prefetch exited with code {}", code))
}

/// Warm the uv cache for the next update, in the background.
///
/// Deliberately not routed through `run_update`: that function takes the runtime
/// gate and runs the idle scan around its child, which is exactly what a
/// background download must not do.
pub(crate) fn run_prefetch_update(
    app: AppHandle,
    state: PrefetchState,
    shell_version: Option<String>,
) -> Result<(), String> {
    let kind = UpdateKind::Prefetch {
        shell_version: shell_version.clone(),
    };
    let bin = match crate::process::find_unsloth_binary() {
        Some(bin) => bin,
        None => return Err("Unsloth binary not found. Cannot prepare an update.".to_string()),
    };

    info!("[prefetch] Preparing the next update via {:?}", bin);
    let outcome = Arc::new(Mutex::new(PrefetchOutcome::default()));
    // Recorded BEFORE the spawn, and cleared on every way out, so there is no window
    // where the status says a prefetch is running and cannot say what for. A reader
    // that saw that window would read the run it just started as one for an older
    // offer, and cancel it.
    if let Ok(mut running) = state.running_version.lock() {
        *running = shell_version;
    }
    let (stdout, stderr) = match spawn_prefetch(&bin, &state.process, &kind) {
        Ok(streams) => streams,
        Err(msg) => {
            if let Ok(mut running) = state.running_version.lock() {
                *running = None;
            }
            return Err(format!("spawn_prefetch: {msg}"));
        }
    };
    let threads = stream_prefetch_output(&app, outcome.clone(), stdout, stderr);

    let result = wait_for_exit(&state.process);
    for handle in threads {
        let _ = handle.join();
    }
    if let Ok(mut running) = state.running_version.lock() {
        *running = None;
    }
    // Read only after both readers are joined, so the last line still counts.
    let outcome = outcome
        .lock()
        .map(|guard| PrefetchOutcome {
            explicit_error: guard.explicit_error.clone(),
            unsupported: guard.unsupported,
        })
        .unwrap_or_default();

    let (complete, failed) = kind
        .terminal_events()
        .expect("a prefetch always has terminal events");
    match result {
        Ok((status, _)) if status.success() => {
            info!("[prefetch] Update prepared");
            let _ = app.emit(complete, ());
            Ok(())
        }
        Ok((_, intentional)) if intentional => {
            info!("[prefetch] Prefetch stopped intentionally");
            Err(UPDATE_STOPPED.to_string())
        }
        Ok((status, _)) => {
            let msg = prefetch_failure(status.code().unwrap_or(-1), &outcome);
            info!("[prefetch] {}", msg);
            let _ = app.emit(failed, &msg);
            Err(msg)
        }
        Err(msg) => {
            warn!("[prefetch] {}", msg);
            let _ = app.emit(failed, &msg);
            Err(msg)
        }
    }
}

fn clear_current_attempt(state: &UpdateState) {
    if let Ok(mut update) = state.lock() {
        update.current_attempt = None;
    }
}

pub fn is_update_running(state: &UpdateState) -> bool {
    state
        .lock()
        .map(|update| update.child.is_some())
        .unwrap_or(false)
}

pub fn record_update_intentional_stop(state: &UpdateState, diagnostics: &DiagnosticsState) {
    let attempt = state
        .lock()
        .ok()
        .and_then(|update| update.current_attempt.clone());
    if let Some(attempt) = attempt {
        diagnostics::finish_attempt(
            diagnostics,
            &attempt,
            None,
            true,
            Some("intentional_stop".to_string()),
        );
    }
}

pub const UPDATE_STOPPED: &str = "Update stopped.";

#[cfg(unix)]
fn process_group_alive(process_group: i32) -> bool {
    let result = unsafe { libc::kill(-process_group, 0) };
    result == 0 || std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

#[cfg(unix)]
fn signal_process_group(process_group: i32, signal: i32) -> Result<(), String> {
    let result = unsafe { libc::kill(-process_group, signal) };
    if result == 0 {
        return Ok(());
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        return Ok(());
    }
    Err(format!(
        "Could not signal update process group {process_group}: {error}"
    ))
}

pub fn stop_update(state: &UpdateState) -> Result<(), String> {
    let mut child = {
        let mut update = match state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!("Update state mutex poisoned, recovering for cleanup");
                poisoned.into_inner()
            }
        };
        update.intentional_stop = true;
        update.child.take()
    };

    let Some(ref mut child) = child else {
        return Ok(());
    };

    let pid = child.id();
    info!("Stopping update process group (pid {})", pid);

    #[cfg(unix)]
    {
        if pid > i32::MAX as u32 {
            warn!("PID {} exceeds i32 range, using direct kill", pid);
            let _ = child.kill();
            let _ = child.wait();
            return Ok(());
        }
        let process_group = pid as i32;
        signal_process_group(process_group, libc::SIGTERM)?;
        let mut leader_exited = false;
        for _ in 0..50 {
            if !leader_exited {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        leader_exited = true;
                        info!("Update leader exited with status: {:?}", status);
                    }
                    Ok(None) => {}
                    Err(error) => warn!("Could not poll update leader: {error}"),
                }
            }
            if !process_group_alive(process_group) {
                if !leader_exited {
                    let _ = child.wait();
                }
                info!("Update process group stopped gracefully");
                return Ok(());
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        warn!("Update process group did not exit gracefully, force killing");
        signal_process_group(process_group, libc::SIGKILL)?;
        if !leader_exited {
            let _ = child.wait();
        }
        for _ in 0..50 {
            if !process_group_alive(process_group) {
                info!("Update process group force stopped");
                return Ok(());
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        return Err(format!(
            "Update process group {process_group} is still running after SIGKILL"
        ));
    }

    #[cfg(windows)]
    {
        crate::process::force_kill_process_tree(pid, child, "Update");
        return Ok(());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// A directory that looks enough like a managed install for
    /// `build_update_command` to resolve an interpreter beside the launcher.
    fn managed_binary_for_test(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "unsloth-update-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let python = dir.join(if cfg!(windows) {
            "python.exe"
        } else {
            "python"
        });
        let bin = dir.join(if cfg!(windows) {
            "unsloth.exe"
        } else {
            "unsloth"
        });
        std::fs::write(&python, b"").unwrap();
        std::fs::write(&bin, b"").unwrap();
        bin
    }

    #[test]
    fn tauri_backend_update_skips_the_web_frontend_build() {
        use std::ffi::OsStr;

        let mut cmd = Command::new("unused");
        configure_tauri_update_environment(&mut cmd);

        for name in ["UNSLOTH_STUDIO_HOME", "STUDIO_HOME"] {
            assert!(cmd
                .get_envs()
                .any(|(key, value)| key == OsStr::new(name) && value.is_none()));
        }
        for (name, expected) in [("UNSLOTH_TAURI_UPDATE", "1"), ("SKIP_STUDIO_FRONTEND", "1")] {
            assert!(cmd.get_envs().any(|(key, value)| {
                key == OsStr::new(name) && value == Some(OsStr::new(expected))
            }));
        }
    }

    #[test]
    fn lossy_reader_keeps_invalid_utf8_and_later_lines() {
        let mut lines = Vec::new();
        read_lossy_lines(Cursor::new(b"bad\xff\r\n[TAURI:STEP] next\n"), |line| {
            lines.push(line)
        })
        .unwrap();

        assert_eq!(lines, ["bad\u{fffd}", "[TAURI:STEP] next"]);
    }

    #[test]
    fn structured_update_error_is_promoted_from_stdout() {
        assert_eq!(
            structured_update_error("[TAURI:ERROR] Access denied reading llama.cpp"),
            Some("Access denied reading llama.cpp".to_string())
        );
        assert_eq!(structured_update_error("[TAURI:ERROR]   "), None);
        assert_eq!(structured_update_error("ordinary update output"), None);
    }

    #[cfg(windows)]
    #[test]
    fn windows_update_command_uses_python_not_replaceable_console_stub() {
        use std::ffi::OsString;

        let dir =
            std::env::temp_dir().join(format!("unsloth-update-command-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let python = dir.join("python.exe");
        let bin = dir.join("unsloth.exe");
        std::fs::write(&python, b"").unwrap();

        let cmd = build_update_command(&bin, UPDATE_ARGS).unwrap();

        assert_eq!(cmd.get_program(), python.as_os_str());
        assert_ne!(cmd.get_program(), bin.as_os_str());
        assert_eq!(
            cmd.get_args().map(OsString::from).collect::<Vec<_>>(),
            vec![
                // -I here and nowhere else: this invocation decides which install gets
                // rewritten, and a user-site unsloth_cli would update the wrong one.
                OsString::from("-X"),
                OsString::from("utf8"),
                OsString::from("-I"),
                OsString::from("-c"),
                OsString::from(crate::process::WINDOWS_CLI_ENTRYPOINT),
                OsString::from("studio"),
                OsString::from("update")
            ]
        );
        // PYTHONHOME / PYTHONPATH handling is asserted in
        // windows_update_command_still_scrubs_the_python_search_path below.
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn windows_update_command_fails_closed_without_managed_python() {
        let bin = std::env::temp_dir()
            .join("missing-managed-python")
            .join("unsloth.exe");
        assert!(build_update_command(&bin, UPDATE_ARGS)
            .unwrap_err()
            .contains("python.exe"));
    }

    // Without -E the child reads PYTHONHOME and PYTHONPATH; see build_update_command.
    #[test]
    fn update_command_scrubs_the_python_search_path() {
        let dir = std::env::temp_dir().join(format!(
            "unsloth-update-scrub-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let python = dir.join("python.exe");
        let bin = dir.join("unsloth.exe");
        std::fs::write(&python, "").unwrap();
        std::fs::write(&bin, "").unwrap();

        let cmd = build_update_command(&bin, UPDATE_ARGS).unwrap();
        for name in ["PYTHONHOME", "PYTHONPATH"] {
            assert!(
                cmd.get_envs()
                    .any(|(key, value)| key == std::ffi::OsStr::new(name) && value.is_none()),
                "{name} is not scrubbed for the updater"
            );
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    // macOS and Linux still exec the console script.
    #[cfg(not(windows))]
    #[test]
    fn posix_update_command_still_execs_the_console_script() {
        use std::ffi::OsString;

        let bin = std::path::Path::new("/opt/unsloth/bin/unsloth");
        let cmd = build_update_command(bin, UPDATE_ARGS).unwrap();

        assert_eq!(cmd.get_program(), bin.as_os_str());
        assert_eq!(
            cmd.get_args().map(OsString::from).collect::<Vec<_>>(),
            vec![OsString::from("studio"), OsString::from("update")]
        );
        for name in ["PYTHONHOME", "PYTHONPATH"] {
            assert!(cmd
                .get_envs()
                .any(|(key, value)| key == std::ffi::OsStr::new(name) && value.is_none()));
        }
    }

    #[test]
    fn the_prefetch_is_a_separate_command_with_its_own_events() {
        let kind = UpdateKind::Prefetch {
            shell_version: Some("0.1.900-beta".to_string()),
        };

        assert_eq!(kind.args(), &["studio", "prefetch-update"]);
        assert_eq!(kind.progress_event(), "prefetch-progress");
        assert_eq!(
            kind.terminal_events(),
            Some(("prefetch-complete", "prefetch-failed"))
        );
        // The whole point of the separate command: nothing it does needs the gate,
        // the idle scan, or the launcher transaction on the CLI side.
        assert!(!kind.mutates_live_environment());
        assert!(UpdateKind::Backend.mutates_live_environment());
        assert_eq!(UpdateKind::Backend.args(), &["studio", "update"]);
    }

    /// A prefetch that claimed the parent's gate would tell the CLI it is covered
    /// by a lock nothing is holding, and a real update starting beside it would
    /// then find the environment "idle" while a download is writing the cache.
    #[test]
    fn the_prefetch_child_never_inherits_the_runtime_gate() {
        use std::ffi::OsStr;

        let bin = managed_binary_for_test("prefetch-gate");
        let prefetch = build_child_command(
            &bin,
            &UpdateKind::Prefetch {
                shell_version: Some("0.1.900-beta".to_string()),
            },
        )
        .unwrap();

        let handoff = prefetch
            .get_envs()
            .find(|(key, _)| *key == OsStr::new(crate::process::STUDIO_RUNTIME_GATE_HANDOFF_ENV));
        assert_eq!(handoff.map(|(_, value)| value), Some(None));
        assert!(prefetch.get_envs().any(|(key, value)| {
            key == OsStr::new(SHELL_VERSION_ENV) && value == Some(OsStr::new("0.1.900-beta"))
        }));

        let update = build_child_command(&bin, &UpdateKind::Backend).unwrap();
        assert!(update.get_envs().any(|(key, value)| {
            key == OsStr::new(crate::process::STUDIO_RUNTIME_GATE_HANDOFF_ENV)
                && value == Some(OsStr::new("1"))
        }));
        // The shell version belongs to the marker a prefetch writes; an update
        // has nothing to record it in.
        assert!(!update
            .get_envs()
            .any(|(key, _)| key == OsStr::new(SHELL_VERSION_ENV)));
        std::fs::remove_dir_all(bin.parent().unwrap()).unwrap();
    }

    #[test]
    fn a_prefetch_without_an_offered_version_sets_no_shell_version() {
        use std::ffi::OsStr;

        let bin = managed_binary_for_test("prefetch-no-version");
        let cmd = build_child_command(
            &bin,
            &UpdateKind::Prefetch {
                shell_version: None,
            },
        )
        .unwrap();

        assert!(!cmd
            .get_envs()
            .any(|(key, _)| key == OsStr::new(SHELL_VERSION_ENV)));
        std::fs::remove_dir_all(bin.parent().unwrap()).unwrap();
    }

    /// The quit dialog and `is_update_running` both read `UpdateState`, so a
    /// prefetch in its own slot cannot make either of them fire.
    #[test]
    fn a_running_prefetch_is_invisible_to_the_update_state() {
        let update = new_update_state();
        let prefetch = new_prefetch_state();

        let mut command = Command::new(if cfg!(windows) { "cmd" } else { "/bin/sh" });
        if cfg!(windows) {
            command.args(["/C", "ping -n 30 127.0.0.1 > NUL"]);
        } else {
            command.args(["-c", "sleep 30"]);
        }
        command.stdout(Stdio::null()).stderr(Stdio::null());
        let mut wrapped = CommandWrap::from(command);
        #[cfg(unix)]
        wrapped.wrap(ProcessGroup::leader());
        prefetch.process.lock().unwrap().child = Some(wrapped.spawn().unwrap());

        assert!(is_prefetch_running(&prefetch));
        assert!(!is_update_running(&update));

        stop_prefetch(&prefetch).unwrap();
        assert!(!is_prefetch_running(&prefetch));
    }

    #[test]
    fn a_backend_without_the_command_is_reported_as_unsupported_not_failed() {
        let unsupported = PrefetchOutcome {
            explicit_error: None,
            unsupported: true,
        };
        assert_eq!(
            prefetch_failure(PREFETCH_UNSUPPORTED_EXIT, &unsupported),
            PREFETCH_UNSUPPORTED
        );
        // Exit 2 alone is also how a bad option exits, so the message has to be there.
        assert_eq!(
            prefetch_failure(PREFETCH_UNSUPPORTED_EXIT, &PrefetchOutcome::default()),
            "Prefetch exited with code 2"
        );
        assert_eq!(
            prefetch_failure(PREFETCH_BUSY_EXIT, &PrefetchOutcome::default()),
            PREFETCH_BUSY
        );
        assert_eq!(
            prefetch_failure(
                1,
                &PrefetchOutcome {
                    explicit_error: Some("no space left".to_string()),
                    unsupported: false,
                }
            ),
            "no space left"
        );
    }

    // POSIX updates fail "busy" against the shell's own retained flock unless the child
    // inherits it, so the handoff is set on every platform.
    #[test]
    fn update_child_uses_the_parent_runtime_gate_on_every_platform() {
        use std::ffi::OsStr;

        let mut cmd = Command::new("unused");
        configure_runtime_gate_environment(&mut cmd);

        assert!(cmd.get_envs().any(|(key, value)| {
            key == OsStr::new(crate::process::STUDIO_RUNTIME_GATE_HANDOFF_ENV)
                && value == Some(OsStr::new("1"))
        }));
    }

    #[cfg(unix)]
    #[test]
    fn stop_update_kills_descendants_after_the_group_leader_exits() {
        let dir = tempfile::tempdir().unwrap();
        let child_pid_file = dir.path().join("child.pid");
        let mut command = Command::new("/bin/sh");
        command
            .args([
                "-c",
                "trap 'exit 0' TERM; /bin/sh -c 'trap \"\" TERM; while :; do sleep 1; done' & echo $! > \"$1\"; while :; do sleep 1; done",
                "update-test",
            ])
            .arg(&child_pid_file)
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let mut wrapped = CommandWrap::from(command);
        wrapped.wrap(ProcessGroup::leader());
        let child = wrapped.spawn().unwrap();
        let process_group = child.id() as i32;
        let state = new_update_state();
        state.lock().unwrap().child = Some(child);

        for _ in 0..50 {
            if child_pid_file.is_file() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let descendant = std::fs::read_to_string(&child_pid_file)
            .unwrap()
            .trim()
            .parse::<i32>()
            .unwrap();

        stop_update(&state).unwrap();

        assert!(!process_group_alive(process_group));
        assert_eq!(unsafe { libc::kill(descendant, 0) }, -1);
    }
}
