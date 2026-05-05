use crate::diagnostics::{self, AttemptLog, DiagnosticsState};
use log::{error, info, warn};
use process_wrap::std::*;
use std::io::BufRead;
use std::process::{Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};

// ── Types ──

pub struct UpdateProcess {
    pub child: Option<Box<dyn ChildWrapper + Send>>,
    pub intentional_stop: bool,
    pub current_attempt: Option<AttemptLog>,
}

impl Default for UpdateProcess {
    fn default() -> Self {
        Self {
            child: None,
            intentional_stop: false,
            current_attempt: None,
        }
    }
}

pub type UpdateState = Arc<Mutex<UpdateProcess>>;

pub fn new_update_state() -> UpdateState {
    Arc::new(Mutex::new(UpdateProcess::default()))
}

// ── Spawn ──

fn spawn_update(
    bin: &std::path::Path,
    state: &UpdateState,
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

    let mut cmd = Command::new(bin);
    cmd.args(["studio", "update"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // AppImage sets LD_LIBRARY_PATH to its bundled libs, which breaks Python
    #[cfg(target_os = "linux")]
    if std::env::var_os("APPIMAGE").is_some() {
        cmd.env_remove("LD_LIBRARY_PATH");
        cmd.env_remove("PYTHONHOME");
        cmd.env_remove("PYTHONPATH");
    }

    // Tauri manages the legacy root; scrub so 'unsloth studio update' targets
    // the same install the desktop app uses, not an inherited custom root.
    cmd.env_remove("UNSLOTH_STUDIO_HOME");
    cmd.env_remove("STUDIO_HOME");

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

fn stream_output(
    app: &AppHandle,
    progress_event: &'static str,
    diagnostics: DiagnosticsState,
    attempt: AttemptLog,
    stdout: Option<std::process::ChildStdout>,
    stderr: Option<std::process::ChildStderr>,
) -> Vec<std::thread::JoinHandle<()>> {
    let mut threads = Vec::new();

    if let Some(out) = stdout {
        let app_clone = app.clone();
        let diagnostics_clone = diagnostics.clone();
        let attempt_clone = attempt.clone();
        threads.push(std::thread::spawn(move || {
            let reader = std::io::BufReader::new(out);
            for line in reader.lines() {
                match line {
                    Ok(text) => {
                        diagnostics::append_phase_line(&attempt_clone.handle, "stdout", &text);
                        if let Some(step) = text.strip_prefix("[TAURI:STEP] ") {
                            diagnostics::record_step(&diagnostics_clone, &attempt_clone, step);
                        } else if let Some(progress) = text.strip_prefix("[TAURI:PROGRESS] ") {
                            diagnostics::record_progress(
                                &diagnostics_clone,
                                &attempt_clone,
                                progress,
                            );
                        } else if let Some(marker) = text.strip_prefix("[TAURI:DIAG] ") {
                            diagnostics::record_diag_marker(
                                &diagnostics_clone,
                                &attempt_clone,
                                marker,
                            );
                        }
                        info!("[update][stdout] {}", text);
                        let _ = app_clone.emit(progress_event, &text);
                    }
                    Err(e) => {
                        warn!("[update] Error reading stdout: {}", e);
                        break;
                    }
                }
            }
        }));
    }

    if let Some(err) = stderr {
        let app_clone = app.clone();
        let attempt_clone = attempt.clone();
        threads.push(std::thread::spawn(move || {
            let reader = std::io::BufReader::new(err);
            for line in reader.lines() {
                match line {
                    Ok(text) => {
                        diagnostics::append_phase_line(&attempt_clone.handle, "stderr", &text);
                        warn!("[update][stderr] {}", text);
                        let _ = app_clone.emit(progress_event, &text);
                    }
                    Err(e) => {
                        warn!("[update] Error reading stderr: {}", e);
                        break;
                    }
                }
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
                    return Ok((status, intentional));
                }
                Ok(None) => {}
                Err(e) => {
                    update.child = None;
                    return Err(format!("Error waiting for update: {}", e));
                }
            },
            None if intentional => return Err("Update stopped.".to_string()),
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
    run_backend_update_with_terminal_events(app, state, diagnostics, true, None)
}

pub(crate) fn run_backend_update_for_repair(
    app: AppHandle,
    state: UpdateState,
    diagnostics: DiagnosticsState,
    repair_group_id: String,
) -> Result<(), String> {
    run_backend_update_with_terminal_events(app, state, diagnostics, false, Some(repair_group_id))
}

fn run_backend_update_with_terminal_events(
    app: AppHandle,
    state: UpdateState,
    diagnostics: DiagnosticsState,
    terminal_events: bool,
    repair_group_id: Option<String>,
) -> Result<(), String> {
    let attempt = match repair_group_id.as_deref() {
        Some(group_id) => diagnostics::begin_repair_child(&diagnostics, group_id, "update"),
        None => diagnostics::begin_update_attempt(&diagnostics),
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
    let progress_event = if terminal_events {
        "update-progress"
    } else {
        "repair-progress"
    };
    let _ = app.emit(progress_event, "Starting backend update...");

    let (stdout, stderr) = match spawn_update(&bin, &state) {
        Ok(handles) => handles,
        Err(msg) => {
            diagnostics::finish_attempt(
                &diagnostics,
                &attempt,
                None,
                false,
                Some(format!("spawn_update: {msg}")),
            );
            clear_current_attempt(&state);
            return Err(msg);
        }
    };
    let threads = stream_output(
        &app,
        progress_event,
        diagnostics.clone(),
        attempt.clone(),
        stdout,
        stderr,
    );

    let result = wait_for_exit(&state);
    for handle in threads {
        let _ = handle.join();
    }

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
            if terminal_events {
                let _ = app.emit("update-complete", ());
            }
            Ok(())
        }
        Ok((status, intentional)) if intentional => {
            diagnostics::finish_attempt(
                &diagnostics,
                &attempt,
                Some(status.to_string()),
                true,
                Some("Update stopped.".to_string()),
            );
            clear_current_attempt(&state);
            info!("[update] Update stopped intentionally");
            Err("Update stopped.".to_string())
        }
        Ok((status, intentional)) => {
            let code = status.code().unwrap_or(-1);
            let msg = format!("Update exited with code {}", code);
            diagnostics::finish_attempt(
                &diagnostics,
                &attempt,
                Some(status.to_string()),
                intentional,
                Some(msg.clone()),
            );
            clear_current_attempt(&state);
            error!("[update] {}", msg);
            if terminal_events {
                let _ = app.emit("update-failed", &msg);
            }
            Err(msg)
        }
        Err(msg) => {
            diagnostics::finish_attempt(&diagnostics, &attempt, None, false, Some(msg.clone()));
            clear_current_attempt(&state);
            error!("[update] {}", msg);
            if terminal_events {
                let _ = app.emit("update-failed", &msg);
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
        unsafe {
            libc::kill(-(pid as i32), libc::SIGTERM);
        }
        for _ in 0..50 {
            match child.try_wait() {
                Ok(Some(status)) => {
                    info!("Update exited gracefully with status: {:?}", status);
                    return Ok(());
                }
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(100)),
                Err(_) => break,
            }
        }
        warn!("Update did not exit gracefully, force killing");
    }

    #[cfg(windows)]
    {
        crate::process::force_kill_process_tree(pid, child, "Update");
        return Ok(());
    }

    #[cfg(unix)]
    {
        let _ = child.kill();
        let _ = child.wait();
        info!("Update process group force stopped");
        Ok(())
    }
}
