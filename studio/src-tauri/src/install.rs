use crate::diagnostics::{self, AttemptLog, DiagnosticsState};
use log::{error, info, warn};
use process_wrap::std::*;
use std::collections::VecDeque;
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

const FAILURE_CONTEXT_LINES: usize = 8;
const FAILURE_CONTEXT_LINE_BYTES: usize = 1_000;

fn generic_failure_message(code: i32) -> String {
    format!(
        "Installation failed with exit code {}. Open the installer logs for details.",
        code
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum InstallOutputStream {
    Stdout,
    Stderr,
}

struct InstallOutputLine {
    stream: InstallOutputStream,
    text: String,
}

#[derive(Default)]
struct InstallFailureContext {
    explicit_error: Option<String>,
    explicit_error_stream: Option<InstallOutputStream>,
    default_error: Option<String>,
    output_tail: VecDeque<InstallOutputLine>,
}

impl InstallFailureContext {
    fn observe_stdout(&mut self, text: &str) -> bool {
        if text.starts_with("[TAURI:ERROR_CLEAR] ") {
            self.clear_failure(InstallOutputStream::Stdout);
            return true;
        }
        if text.starts_with("[TAURI:OUTPUT_CLEAR] ") {
            self.clear_stream(InstallOutputStream::Stdout);
            return true;
        }
        if let Some(message) = text.strip_prefix("[TAURI:ERROR] ") {
            let message = message.trim();
            if !message.is_empty() {
                self.explicit_error = Some(Self::bounded_line(message));
                self.explicit_error_stream = Some(InstallOutputStream::Stdout);
            }
            return false;
        }
        if let Some(message) = text.strip_prefix("[TAURI:ERROR_OUTPUT] ") {
            self.capture_output_error(InstallOutputStream::Stdout, message);
            return true;
        }
        if let Some(message) = text.strip_prefix("[TAURI:ERROR_DEFAULT] ") {
            let message = message.trim();
            if !message.is_empty() {
                self.default_error = Some(Self::bounded_line(message));
            }
            return true;
        }
        if !text.starts_with("[TAURI:") {
            self.push_output(InstallOutputStream::Stdout, text);
        }
        false
    }

    fn observe_stderr(&mut self, text: &str) -> bool {
        if text.starts_with("[TAURI:ERROR_CLEAR] ") {
            self.clear_failure(InstallOutputStream::Stderr);
            return true;
        }
        if text.starts_with("[TAURI:OUTPUT_CLEAR] ") {
            self.clear_stream(InstallOutputStream::Stderr);
            return true;
        }
        if let Some(message) = text.strip_prefix("[TAURI:ERROR_OUTPUT] ") {
            self.capture_output_error(InstallOutputStream::Stderr, message);
            return true;
        }
        self.push_output(InstallOutputStream::Stderr, text);
        false
    }

    fn capture_output_error(&mut self, stream: InstallOutputStream, fallback: &str) {
        let fallback = fallback.trim();
        let detail = self
            .output_tail
            .iter()
            .rev()
            .find(|line| line.stream == stream)
            .map(|line| line.text.as_str());
        if let Some(error) = match (fallback.is_empty(), detail) {
            (_, Some(detail)) if fallback == detail => Some(detail.to_owned()),
            (false, Some(detail)) => Some(Self::bounded_line(&format!("{fallback}: {detail}"))),
            (false, None) => Some(Self::bounded_line(fallback)),
            (true, Some(detail)) => Some(detail.to_owned()),
            (true, None) => None,
        } {
            self.explicit_error = Some(error);
            self.explicit_error_stream = Some(stream);
        }
    }

    fn clear_failure(&mut self, stream: InstallOutputStream) {
        if self.explicit_error_stream == Some(stream) {
            self.explicit_error = None;
            self.explicit_error_stream = None;
        }
        if stream == InstallOutputStream::Stdout {
            self.default_error = None;
        }
        self.clear_stream(stream);
    }

    fn clear_stream(&mut self, stream: InstallOutputStream) {
        self.output_tail.retain(|line| line.stream != stream);
    }

    fn push_output(&mut self, stream: InstallOutputStream, text: &str) {
        let text = text.trim();
        if text.is_empty() {
            return;
        }
        let text = Self::bounded_line(text);
        self.output_tail
            .push_back(InstallOutputLine { stream, text });
        while self.output_tail.len() > FAILURE_CONTEXT_LINES {
            self.output_tail.pop_front();
        }
    }

    fn bounded_line(text: &str) -> String {
        let mut text = diagnostics::redact_for_display(text);
        let boundary =
            diagnostics::valid_utf8_boundary(&text, text.len().min(FAILURE_CONTEXT_LINE_BYTES));
        text.truncate(boundary);
        text
    }

    fn message(&self, code: i32) -> String {
        let detail = self
            .explicit_error
            .as_deref()
            .or(self.default_error.as_deref())
            .or_else(|| self.output_tail.back().map(|line| line.text.as_str()));
        match detail {
            Some(detail) => format!("Installation failed: {}", detail),
            None => generic_failure_message(code),
        }
    }
}

fn is_elevation_request(code: i32, packages: &[String]) -> bool {
    code == 2 && !packages.is_empty()
}

/// Windows PowerShell 5.1 applies `RemoteSigned` authorization differently to
/// Win32 verbatim paths (`\\?\C:\...`) than to their ordinary drive/UNC forms.
/// Tauri resolves resources through `canonicalize`, which always returns the
/// verbatim form, so convert only the two lossless filesystem forms PowerShell
/// understands before passing a script to `-File`.
#[cfg(windows)]
fn powershell_script_path(path: &Path) -> PathBuf {
    use std::ffi::OsString;
    use std::os::windows::ffi::{OsStrExt, OsStringExt};

    // Everything after `\\?\` reaches the object manager, which is case insensitive.
    fn is(unit: Option<&u16>, ascii: u8) -> bool {
        unit.is_some_and(|value| *value < 128 && (*value as u8).eq_ignore_ascii_case(&ascii))
    }

    let wide: Vec<u16> = path.as_os_str().encode_wide().collect();
    let verbatim: Vec<u16> = r"\\?\".encode_utf16().collect();
    if !wide.starts_with(&verbatim) {
        return path.to_path_buf();
    }
    let rest = &wide[verbatim.len()..];

    let is_drive = rest.first().is_some_and(|value| {
        (b'A' as u16..=b'Z' as u16).contains(value) || (b'a' as u16..=b'z' as u16).contains(value)
    }) && rest.get(1) == Some(&(b':' as u16))
        && rest.get(2) == Some(&(b'\\' as u16));

    let normalized = if is(rest.first(), b'U')
        && is(rest.get(1), b'N')
        && is(rest.get(2), b'C')
        && rest.get(3) == Some(&(b'\\' as u16))
    {
        let mut value: Vec<u16> = r"\\".encode_utf16().collect();
        value.extend_from_slice(&rest[4..]);
        value
    } else if is_drive {
        rest.to_vec()
    } else {
        return path.to_path_buf();
    };

    // Only the verbatim form addresses a path past MAX_PATH; stripping it there
    // would trade an authorization error for a "path too long" one. MAX_PATH
    // counts the terminating NUL, so 259 units is the longest legacy path.
    if normalized.len() >= 260 {
        return path.to_path_buf();
    }

    PathBuf::from(OsString::from_wide(&normalized))
}

/// Everything passed to `powershell.exe` before the script's own arguments.
///
/// Separate from `spawn_script` so a test can assert the property that matters:
/// that this flag set authorizes this path spelling. #7819 broke first-run
/// install by editing only the flags, which no test over `powershell_script_path`
/// can see.
#[cfg(windows)]
fn powershell_launch_args(script: &Path) -> Vec<std::ffi::OsString> {
    use std::ffi::OsString;

    // No -WindowStyle Hidden / -ExecutionPolicy Bypass: that pair is a Microsoft
    // detection signature, CREATE_NO_WINDOW hides the console, and NSIS writes
    // resources without a mark-of-the-web so RemoteSigned loads them.
    let mut launch: Vec<OsString> = [
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "RemoteSigned",
        "-File",
    ]
    .iter()
    .map(OsString::from)
    .collect();

    // Load-bearing: RemoteSigned rejects the `\\?\` spelling Tauri resolves to.
    launch.push(powershell_script_path(script).into_os_string());
    launch
}

/// `Command::new` searches the running executable's own directory before the
/// system one, and a `currentUser` install puts that directory somewhere the
/// user can write, so resolve the interpreter absolutely.
#[cfg(windows)]
fn powershell_exe() -> PathBuf {
    let system_root = std::env::var_os("SystemRoot").unwrap_or_else(|| r"C:\Windows".into());
    let absolute = Path::new(&system_root).join(r"System32\WindowsPowerShell\v1.0\powershell.exe");
    if absolute.is_file() {
        absolute
    } else {
        PathBuf::from("powershell.exe")
    }
}

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
    let mut cmd = Command::new(powershell_exe());
    #[cfg(windows)]
    cmd.args(powershell_launch_args(script))
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
/// Parses structured events from stdout and failure controls from both streams.
fn stream_output(
    app: &AppHandle,
    state: &InstallState,
    event_mode: InstallEventMode,
    diagnostics: DiagnosticsState,
    attempt: AttemptLog,
    stdout: Option<std::process::ChildStdout>,
    stderr: Option<std::process::ChildStderr>,
) -> (
    Vec<std::thread::JoinHandle<()>>,
    Arc<Mutex<InstallFailureContext>>,
) {
    let mut threads = Vec::new();
    let failure_context = Arc::new(Mutex::new(InstallFailureContext::default()));

    if let Some(out) = stdout {
        let app_clone = app.clone();
        let state_clone = Arc::clone(state);
        let diagnostics_clone = diagnostics.clone();
        let attempt_clone = attempt.clone();
        let failure_context_clone = Arc::clone(&failure_context);
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
                        let is_failure_control = failure_context_clone
                            .lock()
                            .map(|mut context| context.observe_stdout(&text))
                            .unwrap_or(false);
                        if is_failure_control {
                            info!("[install][stdout] {}", text);
                            continue;
                        }
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
        let failure_context_clone = Arc::clone(&failure_context);
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
                        let is_failure_control = failure_context_clone
                            .lock()
                            .map(|mut context| context.observe_stderr(&text))
                            .unwrap_or(false);
                        if is_failure_control {
                            info!("[install][stderr] {}", text);
                            continue;
                        }
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

    (threads, failure_context)
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
    let (threads, failure_context) = stream_output(
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
            let packages = state
                .lock()
                .map(|install| install.needed_packages.clone())
                .unwrap_or_default();
            if is_elevation_request(code, &packages) {
                // Script needs elevated package install — report to frontend
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
                let msg = failure_context
                    .lock()
                    .map(|context| context.message(code))
                    .unwrap_or_else(|_| generic_failure_message(code));
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

/// True while an installer runs; quitting now would leave a broken venv.
pub fn is_install_running(state: &InstallState) -> bool {
    state
        .lock()
        .map(|install| install.child.is_some())
        .unwrap_or(false)
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

    #[cfg(windows)]
    #[test]
    fn powershell_script_path_normalizes_verbatim_filesystem_paths() {
        assert_eq!(
            powershell_script_path(Path::new(r"\\?\C:\Users\Owner\install.ps1")),
            PathBuf::from(r"C:\Users\Owner\install.ps1")
        );
        assert_eq!(
            powershell_script_path(Path::new(r"\\?\UNC\server\share\install.ps1")),
            PathBuf::from(r"\\server\share\install.ps1")
        );
        assert_eq!(
            powershell_script_path(Path::new(r"C:\Users\Owner\install.ps1")),
            PathBuf::from(r"C:\Users\Owner\install.ps1")
        );
        assert_eq!(
            powershell_script_path(Path::new(r"\\?\Volume{1234}\install.ps1")),
            PathBuf::from(r"\\?\Volume{1234}\install.ps1")
        );
    }

    #[cfg(windows)]
    #[test]
    fn powershell_script_path_leaves_unsupported_spellings_verbatim() {
        // The object manager is case insensitive, so `unc` must normalize too.
        assert_eq!(
            powershell_script_path(Path::new(r"\\?\unc\server\share\install.ps1")),
            PathBuf::from(r"\\server\share\install.ps1")
        );
        assert_eq!(
            powershell_script_path(Path::new(r"\\?\c:\Users\Owner\install.ps1")),
            PathBuf::from(r"c:\Users\Owner\install.ps1")
        );
        // A drive letter with no separator is drive-relative, not the same path.
        for unchanged in [
            r"\\?\C:",
            r"\\?\",
            r"\\?\1:\install.ps1",
            r"\\?\GLOBALROOT\Device\HarddiskVolume1\install.ps1",
            r"\\.\C:\install.ps1",
            r"\\server\share\install.ps1",
            r"install.ps1",
        ] {
            assert_eq!(
                powershell_script_path(Path::new(unchanged)),
                PathBuf::from(unchanged),
                "{unchanged} should be passed through untouched"
            );
        }
        // MAX_PATH counts the terminating NUL, so 259 units still fits but 260 does not.
        let fits = format!(r"C:\{}\install.ps1", "a".repeat(244));
        assert_eq!(fits.len(), 259);
        assert_eq!(
            powershell_script_path(Path::new(&format!(r"\\?\{fits}"))),
            PathBuf::from(&fits)
        );
        for over in [
            format!(r"\\?\C:\{}\install.ps1", "a".repeat(245)),
            format!(r"\\?\C:\{}\install.ps1", "a".repeat(300)),
        ] {
            assert_eq!(
                powershell_script_path(Path::new(&over)),
                PathBuf::from(&over),
                "a path past MAX_PATH must stay verbatim"
            );
        }
    }

    /// Run the real interpreter with the real flags against a script addressed
    /// the way Tauri addresses it. Fails if the execution policy is swapped as
    /// in #7819, or if `powershell_script_path` is dropped from the call site;
    /// both leave the assertions over the normalizer itself passing.
    #[cfg(windows)]
    #[test]
    fn powershell_runs_a_script_addressed_the_way_tauri_addresses_it() {
        use std::fs;

        // A temp file has no Zone.Identifier, so RemoteSigned admits it unsigned.
        // The path spelling and the flag set are what is under test, not signing.
        let dir = std::env::temp_dir().join(format!(
            "unsloth-launch-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        let script = dir.join("install.ps1");
        fs::write(&script, "Write-Output 'unsloth-launcher-ok'\r\n").expect("write script");

        // Resource resolution bottoms out in `canonicalize`, documented to return
        // extended-length syntax. Assert that, so the test cannot pass vacuously.
        let resolved = fs::canonicalize(&script).expect("canonicalize");
        assert!(
            resolved.as_os_str().to_string_lossy().starts_with(r"\\?\"),
            "expected a verbatim path to exercise, got {resolved:?}"
        );

        let output = Command::new(powershell_exe())
            .args(powershell_launch_args(&resolved))
            .output()
            .expect("spawn powershell");

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let _ = fs::remove_dir_all(&dir);

        assert!(
            output.status.success() && stdout.contains("unsloth-launcher-ok"),
            "the launcher shape failed to authorize {resolved:?}\n\
             status: {:?}\nstdout: {stdout}\nstderr: {stderr}",
            output.status.code()
        );
    }

    /// Pin the flag set, so a policy swap shows up as a diff. `-File` stays last
    /// so the script path is never parsed as a flag.
    #[cfg(windows)]
    #[test]
    fn powershell_launch_args_pin_the_defender_friendly_shape() {
        let args = powershell_launch_args(Path::new(r"\\?\C:\Users\Owner\install.ps1"));
        let args: Vec<String> = args
            .iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();

        assert_eq!(
            args,
            vec![
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "RemoteSigned",
                "-File",
                r"C:\Users\Owner\install.ps1",
            ]
        );
        // The pair Microsoft ships as a detection test must not come back.
        assert!(!args.iter().any(|a| a == "Bypass" || a == "-WindowStyle"));
    }

    #[cfg(windows)]
    #[test]
    fn powershell_exe_resolves_under_the_system_directory() {
        let resolved = powershell_exe();
        assert!(resolved.is_absolute(), "{resolved:?} should be absolute");
        assert!(resolved.is_file(), "{resolved:?} should exist");
        let system_root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into());
        assert!(
            resolved.starts_with(&system_root),
            "{resolved:?} escaped {system_root}"
        );
    }

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
        assert!(!is_elevation_request(2, &[]));
        assert!(is_elevation_request(2, &["cmake".to_string()]));
        assert!(!is_elevation_request(1, &["cmake".to_string()]));
    }

    #[test]
    fn explicit_installer_error_beats_stderr_noise() {
        let mut context = InstallFailureContext::default();
        context.observe_stdout("[TAURI:ERROR] Failed to install PyTorch");
        context.observe_stderr("rollback cleanup failed");
        assert_eq!(
            context.message(7),
            "Installation failed: Failed to install PyTorch"
        );
    }

    #[test]
    fn command_error_includes_preceding_output_from_the_same_stream() {
        let mut context = InstallFailureContext::default();
        context.observe_stdout("unrelated stdout");
        context.observe_stderr("resolver error: no space left on device");
        assert!(context.observe_stderr("[TAURI:ERROR_OUTPUT] install unsloth failed (exit code 1)"));
        context.observe_stdout("[TAURI:ERROR_DEFAULT] Failed to install unsloth");
        assert_eq!(
            context.message(1),
            "Installation failed: install unsloth failed (exit code 1): resolver error: no space left on device"
        );
    }

    #[test]
    fn command_error_without_output_uses_its_fallback() {
        let mut context = InstallFailureContext::default();
        context.observe_stdout("unrelated output from an earlier step");
        assert!(context.observe_stdout("[TAURI:OUTPUT_CLEAR] create venv"));
        assert!(context.observe_stdout("[TAURI:ERROR_OUTPUT] create venv failed (exit code 2)"));
        assert_eq!(
            context.message(2),
            "Installation failed: create venv failed (exit code 2)"
        );
    }

    #[test]
    fn recovered_retry_clears_stale_installer_error() {
        let mut context = InstallFailureContext::default();
        context.observe_stderr("ERROR: transient PyTorch download failure");
        assert!(context.observe_stderr("[TAURI:ERROR_OUTPUT] install PyTorch failed (exit code 1)"));
        assert!(context.observe_stdout("[TAURI:ERROR_CLEAR] install PyTorch recovered after retry"));
        assert!(context.observe_stderr("[TAURI:ERROR_CLEAR] install PyTorch recovered after retry"));
        context.observe_stderr("ERROR: studio setup failed");
        let message = context.message(7);
        assert!(message.contains("studio setup failed"));
        assert!(!message.contains("install PyTorch"));
        assert!(!message.contains("transient PyTorch"));
    }

    #[test]
    fn recovery_clear_is_order_independent_across_streams() {
        let mut context = InstallFailureContext::default();
        assert!(context.observe_stdout("[TAURI:ERROR_CLEAR] install PyTorch recovered"));
        context.observe_stderr("ERROR: transient PyTorch download failure");
        assert!(context.observe_stderr("[TAURI:ERROR_OUTPUT] install PyTorch failed (exit code 1)"));
        assert!(context.observe_stderr("[TAURI:ERROR_CLEAR] install PyTorch recovered"));
        context.observe_stdout("[TAURI:ERROR] later setup failure");
        assert!(context.observe_stderr("[TAURI:ERROR_CLEAR] delayed recovery clear"));
        assert_eq!(
            context.message(1),
            "Installation failed: later setup failure"
        );
    }

    #[test]
    fn successful_fallback_clears_unstructured_stderr() {
        let mut context = InstallFailureContext::default();
        context.observe_stderr("bitsandbytes pre-release install failed");
        assert!(context.observe_stderr("[TAURI:ERROR_CLEAR] bitsandbytes pypi fallback recovered"));
        context.observe_stderr("mkdir: cannot create directory: Permission denied");
        assert_eq!(
            context.message(1),
            "Installation failed: mkdir: cannot create directory: Permission denied"
        );
    }

    #[test]
    fn setup_failure_uses_explicit_producer_error_before_default() {
        let mut context = InstallFailureContext::default();
        context.observe_stdout("CMake not found -- installing via winget");
        context.observe_stdout("[TAURI:ERROR] UNSLOTH_LLAMA_PR=invalid is not a valid PR number");
        assert!(context.observe_stdout("[TAURI:ERROR_DEFAULT] studio setup failed (exit code 4)"));
        assert_eq!(
            context.message(4),
            "Installation failed: UNSLOTH_LLAMA_PR=invalid is not a valid PR number"
        );
    }

    #[test]
    fn setup_failure_uses_default_without_specific_output() {
        let mut context = InstallFailureContext::default();
        context.observe_stdout("Finishing setup");
        assert!(context.observe_stdout("[TAURI:ERROR_DEFAULT] studio setup failed (exit code 4)"));
        context.observe_stderr("restored previous environment");
        assert_eq!(
            context.message(4),
            "Installation failed: studio setup failed (exit code 4)"
        );
    }

    #[test]
    fn explicit_setup_error_survives_optional_output_and_footer() {
        let mut context = InstallFailureContext::default();
        context.observe_stdout("[TAURI:ERROR] llama.cpp setup did not produce a usable server");
        context.observe_stderr("whisper.cpp source build failed (exit code 1)");
        context.observe_stdout(
            "whisper.cpp    prebuilt install failed; browser and Transformers dictation remain available",
        );
        for index in 0..10 {
            context.observe_stdout(&format!("setup footer line {index}"));
        }
        assert!(context.observe_stdout("[TAURI:ERROR_DEFAULT] studio setup failed (exit code 1)"));
        assert_eq!(
            context.message(1),
            "Installation failed: llama.cpp setup did not produce a usable server"
        );
    }

    #[test]
    fn setup_default_outranks_nonfatal_failure_output() {
        let mut context = InstallFailureContext::default();
        context.observe_stdout("long paths failed to enable");
        context.observe_stderr("Triton install failed; torch.compile may not work");
        assert!(context.observe_stdout("[TAURI:ERROR_DEFAULT] studio setup failed (exit code 3)"));
        assert_eq!(
            context.message(3),
            "Installation failed: studio setup failed (exit code 3)"
        );
    }

    #[test]
    fn latest_output_is_used_without_structured_context() {
        let mut context = InstallFailureContext::default();
        context.observe_stderr("first diagnostic");
        context.observe_stderr("mv: cannot move build: Permission denied");
        assert_eq!(
            context.message(1),
            "Installation failed: mv: cannot move build: Permission denied"
        );
    }

    #[test]
    fn structured_rollback_progress_does_not_replace_failure_output() {
        let mut context = InstallFailureContext::default();
        context.observe_stderr("ln: cannot create symbolic link: Permission denied");
        context.observe_stdout(
            "[TAURI:PROGRESS] restoring previous environment after failed install...",
        );
        context.observe_stdout("[TAURI:PROGRESS] restored previous environment");
        assert_eq!(
            context.message(1),
            "Installation failed: ln: cannot create symbolic link: Permission denied"
        );
    }

    #[test]
    fn installer_exit_code_is_not_duplicated() {
        let mut context = InstallFailureContext::default();
        context.observe_stdout("[TAURI:ERROR] Failed to install PyTorch (exit code 7)");
        let message = context.message(7);
        assert_eq!(
            message,
            "Installation failed: Failed to install PyTorch (exit code 7)"
        );
        assert_eq!(message.matches("exit code 7").count(), 1);
    }

    #[test]
    fn stderr_fallback_redacts_secrets() {
        let mut context = InstallFailureContext::default();
        context.observe_stderr("ERROR: download failed for https://user:pass@example.com/package");
        let message = context.message(1);
        assert!(message.contains("ERROR: download failed"));
        assert!(message.contains("https://<redacted>@example.com/package"));
        assert!(!message.contains("user:pass"));
    }

    #[test]
    fn failure_context_is_bounded_and_utf8_safe() {
        let mut context = InstallFailureContext::default();
        context.observe_stdout(&format!(
            "[TAURI:ERROR] {}https://user:secret@example.com/package",
            "é".repeat(500)
        ));
        for index in 0..20 {
            context.observe_stderr(&format!("{index}: {}", "é".repeat(1_000)));
        }
        let explicit_error = context.explicit_error.as_ref().unwrap();
        assert!(explicit_error.len() <= FAILURE_CONTEXT_LINE_BYTES);
        assert!(explicit_error.is_char_boundary(explicit_error.len()));
        assert!(!explicit_error.contains("secret"));
        assert_eq!(context.output_tail.len(), FAILURE_CONTEXT_LINES);
        assert!(context
            .output_tail
            .iter()
            .all(|line| line.text.len() <= FAILURE_CONTEXT_LINE_BYTES));
        assert!(context
            .output_tail
            .iter()
            .all(|line| line.text.is_char_boundary(line.text.len())));
    }
}
