use crate::diagnostics::{self, AttemptLog, DiagnosticsState};
use crate::process::trim_line_endings;
use log::{error, info, warn};
use process_wrap::std::*;
use std::io::BufRead;
use std::process::{Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};

// ── Types ──

#[derive(Default)]
pub struct UpdateProcess {
    pub child: Option<Box<dyn ChildWrapper + Send>>,
    pub intentional_stop: bool,
    pub current_attempt: Option<AttemptLog>,
    pub staged: bool,
    pub staged_shell_version: Option<String>,
}

pub type UpdateState = Arc<Mutex<UpdateProcess>>;

/// report staged work owned by this app or a surviving cli process.
pub(crate) fn is_staged_update_running(state: &UpdateState) -> bool {
    let local = state
        .lock()
        .map(|s| s.child.is_some() && s.staged)
        .unwrap_or(false);
    local || staged_update_is_owned_elsewhere()
}

pub(crate) fn staged_update_is_owned_elsewhere() -> bool {
    let home = crate::diagnostics::studio_dir();
    staged_update_is_owned_elsewhere_at(&home, || {
        crate::process::with_studio_runtime_launch_guard(|| Ok(()))
    })
}

fn staged_update_is_owned_elsewhere_at(
    home: &std::path::Path,
    try_gate: impl FnOnce() -> Result<(), String>,
) -> bool {
    if !home.join(crate::staged_update::STAGE_DIR).is_dir() {
        return false;
    }
    try_gate().is_err()
}

pub(crate) fn staged_update_shell_version(state: &UpdateState) -> Option<String> {
    state.lock().ok().and_then(|update| {
        (update.child.is_some() && update.staged)
            .then(|| update.staged_shell_version.clone())
            .flatten()
    })
}

pub fn new_update_state() -> UpdateState {
    Arc::new(Mutex::new(UpdateProcess::default()))
}

const UPDATE_ARGS: &[&str] = &["studio", "update"];
const STAGE_ARGS: &[&str] = &["studio", "update", "--stage"];
const SHELL_VERSION_ENV: &str = "UNSLOTH_TAURI_SHELL_VERSION";

pub(crate) enum UpdateKind {
    Backend,
    Repair(String),
    Staged {
        shell_version: Option<String>,
        backend_version: Option<String>,
    },
}

impl UpdateKind {
    fn args(&self) -> &'static [&'static str] {
        match self {
            UpdateKind::Staged { .. } => STAGE_ARGS,
            _ => UPDATE_ARGS,
        }
    }

    fn progress_event(&self) -> &'static str {
        match self {
            UpdateKind::Backend => "update-progress",
            UpdateKind::Repair(_) => "repair-progress",
            UpdateKind::Staged { .. } => "stage-progress",
        }
    }

    fn terminal_events(&self) -> Option<(&'static str, &'static str)> {
        match self {
            UpdateKind::Backend => Some(("update-complete", "update-failed")),
            UpdateKind::Repair(_) => None,
            UpdateKind::Staged { .. } => Some(("stage-complete", "stage-failed")),
        }
    }

    fn mutates_live_environment(&self) -> bool {
        !matches!(self, UpdateKind::Staged { .. })
    }
}

// ── Spawn ──
fn build_update_command(bin: &std::path::Path, args: &[&str]) -> Result<Command, String> {
    // Only the Windows arm below mutates it.
    #[cfg_attr(not(windows), allow(unused_mut))]
    // Isolated, as this call site shipped. It is the one managed invocation nobody
    // types by hand, and the one that decides which install gets rewritten: a
    // user-site unsloth_cli must not be able to answer `from unsloth_cli import app`
    // here. Everything else inherits, because the console script does.
    let mut cmd = crate::process::build_managed_cli_command_with(
        bin,
        args,
        crate::process::Isolation::Isolated,
    )?;
    // The only managed invocation that scrubs, and the only one that shipped doing it.
    // Elsewhere inheriting is the point, since the console script honours these. Here
    // the failure is unrecoverable: a foreign PYTHONHOME stops the managed interpreter
    // finding its own site-packages, and a PYTHONPATH pointing at another checkout
    // makes `from unsloth_cli import app` update the wrong install.
    cmd.env_remove("PYTHONHOME");
    cmd.env_remove("PYTHONPATH");
    Ok(cmd)
}

fn configure_tauri_update_environment(cmd: &mut Command) {
    // The desktop owns both its shortcuts and its frontend bundle. The managed
    // Python update only needs backend dependencies and native helpers.
    cmd.env_remove("UNSLOTH_STUDIO_HOME");
    cmd.env_remove("STUDIO_HOME");
    cmd.env("UNSLOTH_TAURI_UPDATE", "1");
    cmd.env("SKIP_STUDIO_FRONTEND", "1");
    cmd.env(
        "UNSLOTH_DESKTOP_BACKEND_VERSION",
        crate::preflight::expected_backend_version(),
    );
}

fn configure_staged_update_environment(cmd: &mut Command) {
    for name in [
        "UNSLOTH_LOCAL_LLAMA_CPP_DIR",
        "UNSLOTH_LLAMA_FORCE_COMPILE",
        "UNSLOTH_LLAMA_FORCE_COMPILE_REF",
        "UNSLOTH_LLAMA_PR",
        "UNSLOTH_LLAMA_PR_FORCE",
    ] {
        cmd.env_remove(name);
    }
}

fn configure_runtime_gate_environment(cmd: &mut Command, kind: &UpdateKind) {
    if kind.mutates_live_environment() {
        cmd.env(crate::process::STUDIO_RUNTIME_GATE_HANDOFF_ENV, "1");
    } else {
        cmd.env_remove(crate::process::STUDIO_RUNTIME_GATE_HANDOFF_ENV);
    }
}

fn spawn_update(
    bin: &std::path::Path,
    state: &UpdateState,
    kind: &UpdateKind,
) -> Result<
    (
        Option<std::process::ChildStdout>,
        Option<std::process::ChildStderr>,
    ),
    String,
> {
    let mut update = state.lock().map_err(|e| e.to_string())?;
    if update.child.is_some() {
        return Err("Update is already running.".to_string());
    }
    update.intentional_stop = false;
    update.staged = matches!(kind, UpdateKind::Staged { .. });
    update.staged_shell_version = match kind {
        UpdateKind::Staged { shell_version, .. } => shell_version.clone(),
        _ => None,
    };

    let mut cmd = build_update_command(bin, kind.args())?;
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    if let UpdateKind::Staged {
        shell_version: Some(version),
        ..
    } = kind
    {
        cmd.env(SHELL_VERSION_ENV, version);
    }

    // A login-started desktop inherits C:\Windows\system32, which the CLI refuses
    // to run from; the Windows branch above hits the same guard. Pin both.
    crate::process::apply_managed_cli_context(&mut cmd).map_err(|error| {
        format!(
            "Failed to pick a working directory for the update: {}",
            error
        )
    })?;

    // PYTHONPATH is dropped by the context itself on Windows, where -I covers
    // only the first interpreter and the update starts more.

    #[cfg(target_os = "linux")]
    crate::process::scrub_appimage_python_env(&mut cmd);

    // Keep the update on the desktop-managed install and avoid rebuilding assets
    // that are already compiled into the signed Tauri bundle.
    configure_tauri_update_environment(&mut cmd);
    if matches!(kind, UpdateKind::Staged { .. }) {
        configure_staged_update_environment(&mut cmd);
    }
    configure_runtime_gate_environment(&mut cmd, kind);

    // read_lossy_lines decodes as UTF-8, and here the child is Python itself,
    // which otherwise encodes redirected streams with the locale code page.
    #[cfg(windows)]
    {
        cmd.env("PYTHONUTF8", "1");
        cmd.env("PYTHONIOENCODING", "utf-8");
    }

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

// ── Wait ──

fn wait_for_exit(state: &UpdateState) -> Result<(ExitStatus, bool), String> {
    const MAX_WAIT_ITERATIONS: u32 = 72_000; // 2h at 100ms intervals
    for _ in 0..MAX_WAIT_ITERATIONS {
        let mut update = state.lock().map_err(|e| e.to_string())?;
        let intentional = update.intentional_stop;

        match update.child.as_mut() {
            Some(child) => match child.try_wait() {
                Ok(Some(status)) => {
                    update.child = None;
                    update.staged_shell_version = None;
                    return Ok((status, intentional));
                }
                Ok(None) => {}
                Err(e) => {
                    update.child = None;
                    update.staged_shell_version = None;
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

// ── Public API ──

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

pub(crate) fn run_staged_update(
    app: AppHandle,
    state: UpdateState,
    diagnostics: DiagnosticsState,
    shell_version: Option<String>,
    backend_version: Option<String>,
) -> Result<(), String> {
    run_update(
        app,
        state,
        diagnostics,
        UpdateKind::Staged {
            shell_version,
            backend_version,
        },
    )
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
    let run_child = || {
        let (stdout, stderr) =
            spawn_update(&bin, &state, &kind).map_err(|msg| format!("spawn_update: {msg}"))?;
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
    };
    // the staged cli owns the gate so it remains held after a hard desktop exit.
    let result = if kind.mutates_live_environment() {
        crate::process::with_studio_runtime_launch_guard(|| {
            crate::process::ensure_managed_environment_is_idle(&bin)?;
            run_child()
        })
    } else {
        run_child()
    };
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
            if let UpdateKind::Staged {
                backend_version: Some(required),
                ..
            } = &kind
            {
                let home = crate::diagnostics::studio_dir();
                if let Err(msg) = crate::staged_update::staged_backend_meets(&home, required) {
                    crate::staged_update::discard(&home);
                    error!("[update] {msg}");
                    if let Some((_, failed)) = kind.terminal_events() {
                        let _ = app.emit(failed, &msg);
                    }
                    return Err(msg);
                }
            }
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
        update.staged_shell_version = None;
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
                // -I here and nowhere else. This is the invocation that decides
                // which install gets rewritten, and it shipped isolated; a
                // user-site unsloth_cli answering `from unsloth_cli import app`
                // would update the wrong one. Every invocation a user could have
                // typed instead inherits, because the console script does.
                OsString::from("-X"),
                OsString::from("utf8"),
                OsString::from("-I"),
                OsString::from("-c"),
                OsString::from(crate::process::WINDOWS_CLI_ENTRYPOINT),
                OsString::from("studio"),
                OsString::from("update")
            ]
        );
        // The updater's PYTHONHOME / PYTHONPATH handling is asserted once, in
        // windows_update_command_still_scrubs_the_python_search_path below. This
        // test owns the program and the argument vector.
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

    // The Windows trampoline moved into process.rs; nothing about the POSIX
    // Dropping -I made this load bearing rather than belt and braces: without -E the
    // child now reads both. See build_update_command for what each one breaks.
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

    // command may move with it. macOS and Linux still exec the console script.
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
    fn staged_update_drops_source_build_overrides() {
        use std::ffi::OsStr;

        let mut cmd = Command::new("unused");
        configure_staged_update_environment(&mut cmd);

        for name in [
            "UNSLOTH_LOCAL_LLAMA_CPP_DIR",
            "UNSLOTH_LLAMA_FORCE_COMPILE",
            "UNSLOTH_LLAMA_FORCE_COMPILE_REF",
            "UNSLOTH_LLAMA_PR",
            "UNSLOTH_LLAMA_PR_FORCE",
        ] {
            assert!(cmd
                .get_envs()
                .any(|(key, value)| key == OsStr::new(name) && value.is_none()));
        }
    }

    #[test]
    fn staged_update_child_acquires_its_own_runtime_gate() {
        use std::ffi::OsStr;

        let mut cmd = Command::new("unused");
        configure_runtime_gate_environment(
            &mut cmd,
            &UpdateKind::Staged {
                shell_version: None,
                backend_version: None,
            },
        );

        assert!(cmd.get_envs().any(|(key, value)| {
            key == OsStr::new(crate::process::STUDIO_RUNTIME_GATE_HANDOFF_ENV) && value.is_none()
        }));
    }

    #[test]
    fn live_update_child_uses_the_parent_runtime_gate() {
        use std::ffi::OsStr;

        let mut cmd = Command::new("unused");
        configure_runtime_gate_environment(&mut cmd, &UpdateKind::Backend);

        assert!(cmd.get_envs().any(|(key, value)| {
            key == OsStr::new(crate::process::STUDIO_RUNTIME_GATE_HANDOFF_ENV)
                && value == Some(OsStr::new("1"))
        }));
    }

    #[cfg(unix)]
    #[test]
    fn surviving_stage_gate_blocks_reopen_until_the_owner_exits() {
        use std::os::fd::AsRawFd;

        let home = tempfile::tempdir().unwrap();
        std::fs::create_dir(home.path().join(crate::staged_update::STAGE_DIR)).unwrap();
        let gate_path = home.path().join(".studio-runtime.lock");
        let owner = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&gate_path)
            .unwrap();
        assert_eq!(unsafe { libc::flock(owner.as_raw_fd(), libc::LOCK_EX) }, 0);

        let probe = || {
            let candidate = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .open(&gate_path)
                .map_err(|error| error.to_string())?;
            let result =
                unsafe { libc::flock(candidate.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            if result == 0 {
                unsafe { libc::flock(candidate.as_raw_fd(), libc::LOCK_UN) };
                Ok(())
            } else {
                Err(std::io::Error::last_os_error().to_string())
            }
        };

        assert!(staged_update_is_owned_elsewhere_at(home.path(), probe));
        drop(owner);
        assert!(!staged_update_is_owned_elsewhere_at(home.path(), probe));
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
