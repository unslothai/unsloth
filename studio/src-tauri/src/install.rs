use crate::diagnostics::{self, AttemptLog, DiagnosticsState};
use log::{error, info, warn};
use process_wrap::std::*;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager};

// ── Types ──

pub struct InstallProcess {
    /// Process group handle — killing this kills the entire subprocess tree.
    pub child: Option<Box<dyn ChildWrapper + Send>>,
    pub intentional_stop: bool,
    /// Packages needing elevated install, parsed from [TAURI:NEED_SUDO] output.
    pub needed_packages: Vec<String>,
    /// Current diagnostics attempt; kept after NEEDS_ELEVATION so apt output can be linked.
    pub current_attempt: Option<AttemptLog>,
}

impl Default for InstallProcess {
    fn default() -> Self {
        Self {
            child: None,
            intentional_stop: false,
            needed_packages: Vec::new(),
            current_attempt: None,
        }
    }
}

pub type InstallState = Arc<Mutex<InstallProcess>>;

pub fn new_install_state() -> InstallState {
    Arc::new(Mutex::new(InstallProcess::default()))
}

use crate::process::trim_line_endings;

// ── Script Resolution ──

/// Returns (script_path, args) depending on dev vs production mode.
/// Dev mode: repo root script + --tauri --local
/// Production: bundled resource + --tauri
fn resolve_install_script(app: &AppHandle) -> Result<(PathBuf, Vec<String>), String> {
    let mut args = vec!["--tauri".to_string()];

    if cfg!(debug_assertions) {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent() // studio/
            .and_then(|p| p.parent()) // repo root
            .ok_or("Cannot resolve repo root from CARGO_MANIFEST_DIR")?;

        let script = if cfg!(unix) {
            repo_root.join("install.sh")
        } else {
            repo_root.join("install.ps1")
        };

        if !script.exists() {
            return Err(format!("Install script not found: {}", script.display()));
        }

        args.push("--local".to_string());
        info!("Dev mode: using repo script at {}", script.display());
        Ok((script, args))
    } else {
        let name = if cfg!(unix) {
            "install.sh"
        } else {
            "install.ps1"
        };
        let script = app
            .path()
            .resolve(name, tauri::path::BaseDirectory::Resource)
            .map_err(|e| format!("Failed to resolve bundled {}: {}", name, e))?;
        info!("Production: using bundled script at {}", script.display());
        Ok((script, args))
    }
}

// ── Emit Helpers ──

#[derive(Clone, Copy)]
enum InstallEventMode {
    Full,
    Repair,
}

impl InstallEventMode {
    fn progress_event(self) -> &'static str {
        match self {
            Self::Full => "install-progress",
            Self::Repair => "repair-progress",
        }
    }

    fn emit_install_structured_events(self) -> bool {
        matches!(self, Self::Full)
    }

    fn emit_terminal_events(self) -> bool {
        matches!(self, Self::Full)
    }

    fn needs_elevation_event(self) -> &'static str {
        match self {
            Self::Full => "install-needs-elevation",
            Self::Repair => "repair-needs-elevation",
        }
    }
}

fn emit_mode_progress(app: &AppHandle, mode: InstallEventMode, message: &str) {
    info!("[install] {}", message);
    let _ = app.emit(mode.progress_event(), message);
}

fn emit_failed(app: &AppHandle, message: &str) {
    error!("[install] FAILED: {}", message);
    let _ = app.emit("install-failed", message);
}

fn emit_complete(app: &AppHandle) {
    info!("[install] Installation complete");
    let _ = app.emit("install-complete", ());
}

// ── Spawn ──

/// Spawns the install script in a process group.
/// Returns (stdout, stderr) handles for streaming.
/// The GroupChild is stored in state so stop_install() can kill the entire tree.
fn spawn_script(
    script: &Path,
    args: &[String],
    state: &InstallState,
) -> Result<
    (
        Option<std::process::ChildStdout>,
        Option<std::process::ChildStderr>,
    ),
    String,
> {
    let mut install = state.lock().map_err(|e| e.to_string())?;
    if install.child.is_some() {
        return Err("Installation is already running.".to_string());
    }
    install.intentional_stop = false;
    install.needed_packages.clear();

    // Scripts create ~/.unsloth/studio/ themselves, but need a writable cwd.
    let home = dirs::home_dir().ok_or("Could not determine home directory")?;
    let work_dir = home.join(".unsloth");
    if !work_dir.exists() {
        std::fs::create_dir_all(&work_dir)
            .map_err(|e| format!("Failed to create {}: {}", work_dir.display(), e))?;
    }

    #[cfg(unix)]
    let mut cmd = Command::new("bash");
    #[cfg(unix)]
    cmd.arg(script)
        .args(args)
        .current_dir(&work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(windows)]
    let mut cmd = Command::new("powershell.exe");
    #[cfg(windows)]
    cmd.args([
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-WindowStyle",
        "Hidden",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
    ])
    .arg(script)
    .args(args)
    .current_dir(&work_dir)
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());

    // AppImage sets LD_LIBRARY_PATH to its bundled libs, which breaks Python
    // spawned by the install script. Only clear inside AppImage — native installs
    // may need these for custom CUDA or conda paths.
    #[cfg(target_os = "linux")]
    if std::env::var_os("APPIMAGE").is_some() {
        cmd.env_remove("LD_LIBRARY_PATH");
        cmd.env_remove("PYTHONHOME");
        cmd.env_remove("PYTHONPATH");
    }

    // Tauri only does default-root installs; install.sh / install.ps1 reject
    // these under --tauri. Scrub so an inherited value can't trip the guard.
    cmd.env_remove("UNSLOTH_STUDIO_HOME");
    cmd.env_remove("STUDIO_HOME");

    // On Windows, launch the installer directly with CREATE_NO_WINDOW.
    // The app process is assigned to a KILL_ON_JOB_CLOSE job in main.rs, so
    // child cleanup on crash comes from inherited job membership instead.
    #[cfg(windows)]
    let mut child: Box<dyn ChildWrapper + Send> = {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(crate::process::CREATE_NO_WINDOW);
        let child = cmd
            .spawn()
            .map_err(|e| format!("Failed to spawn install script: {}", e))?;
        Box::new(child)
    };

    #[cfg(unix)]
    let mut child: Box<dyn ChildWrapper + Send> = {
        // Keep the whole installer tree in a process group on Unix.
        let mut wrap = CommandWrap::from(cmd);
        wrap.wrap(ProcessGroup::leader());
        wrap.spawn()
            .map_err(|e| format!("Failed to spawn install script: {}", e))?
    };

    let stdout = child.stdout().take();
    let stderr = child.stderr().take();
    install.child = Some(child);
    Ok((stdout, stderr))
}

// ── Stream ──

/// Spawns reader threads for stdout/stderr.
/// Parses [TAURI:*] lines from stdout for structured events.
fn stream_output(
    app: &AppHandle,
    state: &InstallState,
    event_mode: InstallEventMode,
    diagnostics: DiagnosticsState,
    attempt: AttemptLog,
    stdout: Option<std::process::ChildStdout>,
    stderr: Option<std::process::ChildStderr>,
) -> Vec<std::thread::JoinHandle<()>> {
    let mut threads = Vec::new();

    if let Some(out) = stdout {
        let app_clone = app.clone();
        let state_clone = Arc::clone(state);
        let diagnostics_clone = diagnostics.clone();
        let attempt_clone = attempt.clone();
        threads.push(std::thread::spawn(move || {
            let mut reader = std::io::BufReader::new(out);
            let mut buf = Vec::new();
            loop {
                buf.clear();
                match reader.read_until(b'\n', &mut buf) {
                    Ok(0) => break,
                    Ok(_) => {
                        let text = String::from_utf8_lossy(trim_line_endings(&buf)).into_owned();
                        diagnostics::append_phase_line(&attempt_clone.handle, "stdout", &text);
                        // Parse structured Tauri protocol lines
                        if let Some(packages) = text.strip_prefix("[TAURI:NEED_SUDO] ") {
                            let pkgs: Vec<String> =
                                packages.split_whitespace().map(String::from).collect();
                            if let Ok(mut install) = state_clone.lock() {
                                install.needed_packages = pkgs.clone();
                            }
                            diagnostics::record_elevation_packages(
                                &diagnostics_clone,
                                &attempt_clone,
                                &pkgs,
                            );
                        } else if let Some(step) = text.strip_prefix("[TAURI:STEP] ") {
                            diagnostics::record_step(&diagnostics_clone, &attempt_clone, step);
                            if !event_mode.emit_install_structured_events() {
                                info!("[install][stdout] {}", text);
                                let _ = app_clone.emit(event_mode.progress_event(), &text);
                                continue;
                            }
                            let _ = app_clone.emit("install-step", step);
                        } else if let Some(detail) = text.strip_prefix("[TAURI:PROGRESS] ") {
                            diagnostics::record_progress(
                                &diagnostics_clone,
                                &attempt_clone,
                                detail,
                            );
                            if !event_mode.emit_install_structured_events() {
                                info!("[install][stdout] {}", text);
                                let _ = app_clone.emit(event_mode.progress_event(), detail);
                                continue;
                            }
                            let _ = app_clone.emit("install-progress-detail", detail);
                        } else if let Some(marker) = text.strip_prefix("[TAURI:DIAG] ") {
                            diagnostics::record_diag_marker(
                                &diagnostics_clone,
                                &attempt_clone,
                                marker,
                            );
                        }
                        // Always forward the raw line
                        info!("[install][stdout] {}", text);
                        let _ = app_clone.emit(event_mode.progress_event(), &text);
                    }
                    Err(e) => {
                        warn!("[install] Error reading stdout: {}", e);
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
            let mut reader = std::io::BufReader::new(err);
            let mut buf = Vec::new();
            loop {
                buf.clear();
                match reader.read_until(b'\n', &mut buf) {
                    Ok(0) => break,
                    Ok(_) => {
                        let text = String::from_utf8_lossy(trim_line_endings(&buf)).into_owned();
                        diagnostics::append_phase_line(&attempt_clone.handle, "stderr", &text);
                        warn!("[install][stderr] {}", text);
                        let _ = app_clone.emit(event_mode.progress_event(), &text);
                    }
                    Err(e) => {
                        warn!("[install] Error reading stderr: {}", e);
                        break;
                    }
                }
            }
        }));
    }

    threads
}

// ── Wait & Finalize ──

/// Waits for the install process to exit. Returns (exit_status, was_intentional_stop).
/// Times out after 2 hours to prevent infinite loops if the child hangs.
fn wait_for_exit(state: &InstallState) -> Result<(ExitStatus, bool), String> {
    const MAX_WAIT_ITERATIONS: u32 = 72_000; // 2h at 100ms intervals
    for _ in 0..MAX_WAIT_ITERATIONS {
        let mut install = state.lock().map_err(|e| e.to_string())?;
        let intentional = install.intentional_stop;

        match install.child.as_mut() {
            Some(child) => match child.try_wait() {
                Ok(Some(status)) => {
                    install.child = None;
                    return Ok((status, intentional));
                }
                Ok(None) => {}
                Err(e) => {
                    install.child = None;
                    return Err(format!("Error waiting for installer: {}", e));
                }
            },
            None if intentional => return Err("Installation stopped.".to_string()),
            None => return Err("Installer process disappeared unexpectedly.".to_string()),
        }

        drop(install);
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    // Timed out — kill and report
    let _ = stop_install(state);
    Err("Installation timed out after 2 hours".to_string())
}

// ── Public API ──

/// Run the install script. Returns Ok(()) on success.
/// Returns Err("NEEDS_ELEVATION") if system packages need elevated install (Linux only).
/// Returns Err(message) on other failures.
pub fn run_install(
    app: AppHandle,
    state: InstallState,
    diagnostics: DiagnosticsState,
) -> Result<(), String> {
    run_install_with_event_mode(app, state, diagnostics, InstallEventMode::Full, None)
}

pub(crate) fn run_install_for_repair(
    app: AppHandle,
    state: InstallState,
    diagnostics: DiagnosticsState,
    repair_group_id: String,
) -> Result<(), String> {
    run_install_with_event_mode(
        app,
        state,
        diagnostics,
        InstallEventMode::Repair,
        Some(repair_group_id),
    )
}

fn run_install_with_event_mode(
    app: AppHandle,
    state: InstallState,
    diagnostics: DiagnosticsState,
    event_mode: InstallEventMode,
    repair_group_id: Option<String>,
) -> Result<(), String> {
    let attempt = match repair_group_id.as_deref() {
        Some(group_id) => diagnostics::begin_repair_child(&diagnostics, group_id, "install"),
        None => diagnostics::begin_install_attempt(&diagnostics),
    };
    if let Ok(mut install) = state.lock() {
        install.current_attempt = Some(attempt.clone());
    }

    emit_mode_progress(&app, event_mode, "Starting installation...");

    let (script, args) = match resolve_install_script(&app) {
        Ok(resolved) => resolved,
        Err(msg) => {
            diagnostics::finish_attempt(
                &diagnostics,
                &attempt,
                None,
                false,
                Some(format!("resolve_install_script: {msg}")),
            );
            clear_current_attempt(&state);
            return Err(msg);
        }
    };
    diagnostics::append_phase_line(
        &attempt.handle,
        "meta",
        &format!("Using script: {}", script.display()),
    );
    emit_mode_progress(
        &app,
        event_mode,
        &format!("Using script: {}", script.display()),
    );

    let (stdout, stderr) = match spawn_script(&script, &args, &state) {
        Ok(handles) => handles,
        Err(msg) => {
            diagnostics::finish_attempt(
                &diagnostics,
                &attempt,
                None,
                false,
                Some(format!("spawn_install_script: {msg}")),
            );
            clear_current_attempt(&state);
            return Err(msg);
        }
    };
    let threads = stream_output(
        &app,
        &state,
        event_mode,
        diagnostics.clone(),
        attempt.clone(),
        stdout,
        stderr,
    );

    // Wait for exit, join reader threads
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
            if event_mode.emit_terminal_events() {
                emit_complete(&app);
            }
            Ok(())
        }
        Ok((status, intentional)) => {
            let code = status.code().unwrap_or(-1);
            if code == 2 {
                // Script needs elevated package install — report to frontend
                let packages = state
                    .lock()
                    .map(|i| i.needed_packages.clone())
                    .unwrap_or_default();
                diagnostics::record_elevation_packages(&diagnostics, &attempt, &packages);
                diagnostics::finish_attempt(
                    &diagnostics,
                    &attempt,
                    Some(status.to_string()),
                    intentional,
                    Some("needs_elevation".to_string()),
                );
                info!("[install] Needs elevation for packages: {:?}", packages);
                let _ = app.emit(event_mode.needs_elevation_event(), &packages);
                Err("NEEDS_ELEVATION".to_string())
            } else {
                let msg = format!("Installer exited with code {}", code);
                diagnostics::finish_attempt(
                    &diagnostics,
                    &attempt,
                    Some(status.to_string()),
                    intentional,
                    Some(msg.clone()),
                );
                clear_current_attempt(&state);
                if event_mode.emit_terminal_events() {
                    emit_failed(&app, &msg);
                }
                Err(msg)
            }
        }
        Err(msg) if msg == "Installation stopped." => {
            diagnostics::finish_attempt(&diagnostics, &attempt, None, true, Some(msg.clone()));
            clear_current_attempt(&state);
            info!("[install] Installation stopped intentionally");
            Err(msg)
        }
        Err(msg) => {
            diagnostics::finish_attempt(&diagnostics, &attempt, None, false, Some(msg.clone()));
            clear_current_attempt(&state);
            if event_mode.emit_terminal_events() {
                emit_failed(&app, &msg);
            }
            Err(msg)
        }
    }
}

fn clear_current_attempt(state: &InstallState) {
    if let Ok(mut install) = state.lock() {
        install.current_attempt = None;
    }
}

pub fn take_pending_repair_group_for_resume(state: &InstallState) -> Option<String> {
    let mut install = state.lock().ok()?;
    let repair_group_id = install
        .current_attempt
        .as_ref()
        .and_then(|attempt| attempt.repair_group_id.clone());
    if repair_group_id.is_some() {
        install.current_attempt = None;
    }
    repair_group_id
}

pub fn record_pending_elevation_canceled(
    state: &InstallState,
    diagnostics: &DiagnosticsState,
) -> bool {
    let attempt = state
        .lock()
        .ok()
        .and_then(|mut install| install.current_attempt.take());
    let Some(attempt) = attempt else {
        return false;
    };
    diagnostics::finish_attempt(
        diagnostics,
        &attempt,
        None,
        true,
        Some("elevation_canceled".to_string()),
    );
    if let Some(repair_group_id) = attempt.repair_group_id.as_deref() {
        diagnostics::finish_repair_group(
            diagnostics,
            repair_group_id,
            "canceled",
            Some("elevation_canceled".to_string()),
        );
    }
    true
}

pub fn record_install_intentional_stop(state: &InstallState, diagnostics: &DiagnosticsState) {
    let attempt = state
        .lock()
        .ok()
        .and_then(|install| install.current_attempt.clone());
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

/// Stop a running install process gracefully.
/// Unix: SIGTERM to process group -> wait up to 5s -> SIGKILL
/// Windows: hidden taskkill /T /F to terminate the installer tree
pub fn stop_install(state: &InstallState) -> Result<(), String> {
    let mut child = {
        let mut install = match state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!("Install state mutex poisoned, recovering for cleanup");
                poisoned.into_inner()
            }
        };
        install.intentional_stop = true;
        install.child.take()
    };

    let Some(ref mut child) = child else {
        return Ok(());
    };

    let pid = child.id();
    info!("Stopping installer process group (pid {})", pid);

    // Try graceful SIGTERM first so pip/cmake can clean up temp files
    #[cfg(unix)]
    {
        if pid > i32::MAX as u32 {
            // PID too large for i32 negation, fall back to direct kill
            warn!("PID {} exceeds i32 range, using direct kill", pid);
            let _ = child.kill();
            let _ = child.wait();
            return Ok(());
        }
        unsafe {
            libc::kill(-(pid as i32), libc::SIGTERM);
        }
        // Wait up to 5s for graceful exit
        for _ in 0..50 {
            match child.try_wait() {
                Ok(Some(status)) => {
                    info!("Installer exited gracefully with status: {:?}", status);
                    return Ok(());
                }
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(100)),
                Err(_) => break,
            }
        }
        warn!("Installer did not exit gracefully, force killing");
    }

    #[cfg(windows)]
    {
        crate::process::force_kill_process_tree(pid, child, "Installer");
        return Ok(());
    }

    #[cfg(unix)]
    {
        // Force kill (SIGKILL on Unix)
        let _ = child.kill();
        let _ = child.wait();
        info!("Installer process group force stopped");
        Ok(())
    }
}

/// Install apt system packages with elevated permissions (Linux only).
/// Uses `elevated-command` crate for native auth dialog.
#[cfg(target_os = "linux")]
pub fn install_system_packages(
    packages: &[String],
    state: &InstallState,
    diagnostics: &DiagnosticsState,
) -> Result<(), String> {
    use regex::Regex;
    use std::path::Path;
    use std::process::Command as StdCommand;

    let current_attempt = state
        .lock()
        .ok()
        .and_then(|install| install.current_attempt.clone());
    let elevation_attempt = current_attempt
        .as_ref()
        .and_then(|attempt| attempt.repair_group_id.as_deref())
        .map(|group_id| diagnostics::begin_repair_child(diagnostics, group_id, "elevation"))
        .or_else(|| current_attempt.clone());
    if let Some(attempt) = elevation_attempt.as_ref() {
        diagnostics::append_phase_line(
            &attempt.handle,
            "meta",
            &format!("Starting elevated apt install for: {}", packages.join(", ")),
        );
        diagnostics::record_elevation_packages(diagnostics, attempt, packages);
    }

    // Validate package names to prevent injection via elevated command.
    let valid_pkg = Regex::new(r"^[a-zA-Z0-9][a-zA-Z0-9.+\-]*$").unwrap();
    for pkg in packages {
        if !valid_pkg.is_match(pkg) {
            let msg = format!("Invalid package name: {}", pkg);
            finish_elevation_failure(diagnostics, elevation_attempt.as_ref(), None, msg.clone());
            clear_current_attempt(state);
            return Err(msg);
        }
    }

    // install.sh reports Debian package names. Do not pass them to dnf,
    // zypper, or pacman where names differ; show an explicit support boundary
    // instead of offering an elevation flow that is likely to fail.
    if !Path::new("/usr/bin/apt-get").exists() {
        let msg = "Automatic system package installation is supported on apt-based Linux distributions (Ubuntu/Debian) only. Install the missing dependencies with your package manager and retry."
            .to_string();
        finish_elevation_failure(diagnostics, elevation_attempt.as_ref(), None, msg.clone());
        clear_current_attempt(state);
        return Err(msg);
    }

    info!(
        "[install] Elevated install of apt packages: {}",
        packages.join(", ")
    );

    let mut update_cmd = StdCommand::new("apt-get");
    update_cmd.args(["update", "-y"]);
    match elevated_command::Command::new(update_cmd).output() {
        Ok(elevated_update) => {
            if let Some(attempt) = elevation_attempt.as_ref() {
                diagnostics::append_phase_line(
                    &attempt.handle,
                    "apt-update-status",
                    &elevated_update.status.to_string(),
                );
                append_capped_output(
                    &attempt.handle,
                    "apt-update-stdout",
                    &elevated_update.stdout,
                );
                append_capped_output(
                    &attempt.handle,
                    "apt-update-stderr",
                    &elevated_update.stderr,
                );
            }
            if !elevated_update.status.success() {
                let stderr = capped_output_text(&elevated_update.stderr);
                warn!(
                    "[install] apt-get update failed before elevated install; continuing with cached package metadata: {}",
                    stderr
                );
                if let Some(attempt) = elevation_attempt.as_ref() {
                    diagnostics::append_phase_line(
                        &attempt.handle,
                        "apt-update-warning",
                        "apt-get update failed; continuing with apt-get install",
                    );
                }
            }
        }
        Err(error) => {
            warn!(
                "[install] Elevated apt update could not run before install; continuing with apt-get install: {}",
                error
            );
            if let Some(attempt) = elevation_attempt.as_ref() {
                diagnostics::append_phase_line(
                    &attempt.handle,
                    "apt-update-error",
                    &format!(
                        "apt-get update could not run; continuing with apt-get install: {error}"
                    ),
                );
            }
        }
    }

    let mut install_cmd = StdCommand::new("apt-get");
    install_cmd.args(["install", "-y"]).args(packages);

    let elevated_install = match elevated_command::Command::new(install_cmd).output() {
        Ok(output) => output,
        Err(error) => {
            let msg = format!("Elevated install failed: {}", error);
            finish_elevation_failure(diagnostics, elevation_attempt.as_ref(), None, msg.clone());
            clear_current_attempt(state);
            return Err(msg);
        }
    };
    if let Some(attempt) = elevation_attempt.as_ref() {
        diagnostics::append_phase_line(
            &attempt.handle,
            "apt-install-status",
            &elevated_install.status.to_string(),
        );
        append_capped_output(
            &attempt.handle,
            "apt-install-stdout",
            &elevated_install.stdout,
        );
        append_capped_output(
            &attempt.handle,
            "apt-install-stderr",
            &elevated_install.stderr,
        );
    }

    if !elevated_install.status.success() {
        let stderr = capped_output_text(&elevated_install.stderr);
        finish_elevation_failure(
            diagnostics,
            elevation_attempt.as_ref(),
            Some(elevated_install.status.to_string()),
            format!("Package installation failed: {stderr}"),
        );
        clear_current_attempt(state);
        return Err(format!("Package installation failed: {}", stderr));
    }

    if let Some(attempt) = elevation_attempt.as_ref() {
        diagnostics::finish_attempt(
            diagnostics,
            attempt,
            Some(elevated_install.status.to_string()),
            false,
            None,
        );
    }
    info!("[install] Elevated apt package install succeeded");
    Ok(())
}

#[cfg(target_os = "linux")]
fn finish_elevation_failure(
    diagnostics: &DiagnosticsState,
    attempt: Option<&AttemptLog>,
    exit_status: Option<String>,
    message: String,
) {
    if let Some(attempt) = attempt {
        diagnostics::finish_attempt(
            diagnostics,
            attempt,
            exit_status,
            false,
            Some(message.clone()),
        );
        if let Some(repair_group_id) = attempt.repair_group_id.as_deref() {
            diagnostics::finish_repair_group(diagnostics, repair_group_id, "failed", Some(message));
        }
    }
}

#[cfg(target_os = "linux")]
fn append_capped_output(handle: &diagnostics::PhaseLogHandle, stream: &str, bytes: &[u8]) {
    diagnostics::append_phase_line(handle, stream, &capped_output_text(bytes));
}

#[cfg(any(target_os = "linux", test))]
fn capped_output_text(bytes: &[u8]) -> String {
    const MAX_ELEVATED_OUTPUT_BYTES: usize = 64 * 1024;
    let text = String::from_utf8_lossy(bytes);
    if text.len() <= MAX_ELEVATED_OUTPUT_BYTES {
        return text.into_owned();
    }
    let boundary = diagnostics::valid_utf8_boundary(&text, MAX_ELEVATED_OUTPUT_BYTES);
    let mut truncated = text[..boundary].to_string();
    truncated.push_str("\n[elevated output truncated after 64KiB]");
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elevated_output_cap_is_utf8_boundary_safe() {
        let text = "é".repeat(40_000);
        let capped = capped_output_text(text.as_bytes());
        assert!(capped.ends_with("[elevated output truncated after 64KiB]"));
        assert!(capped.is_char_boundary(capped.len()));
    }

    #[test]
    fn repair_install_mode_uses_repair_elevation_event() {
        assert_eq!(
            InstallEventMode::Full.needs_elevation_event(),
            "install-needs-elevation"
        );
        assert_eq!(
            InstallEventMode::Repair.needs_elevation_event(),
            "repair-needs-elevation"
        );
        assert!(!InstallEventMode::Repair.emit_terminal_events());
    }
}
