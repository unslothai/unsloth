use crate::diagnostics::{self, BackendLog, DiagnosticsState};
use log::{error, info, warn};
use process_wrap::std::*;
use regex::Regex;
use std::collections::VecDeque;
use std::io::BufRead;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager};

const MAX_LOG_LINES: usize = 1000;

pub struct BackendProcess {
    pub child: Option<Box<dyn ChildWrapper + Send>>,
    pub port: Option<u16>,
    pub logs: VecDeque<String>,
    pub intentional_stop: bool,
    pub generation: u64,
    pub diagnostics_session: Option<BackendLog>,
}

impl Default for BackendProcess {
    fn default() -> Self {
        Self {
            child: None,
            port: None,
            logs: VecDeque::with_capacity(MAX_LOG_LINES),
            intentional_stop: false,
            generation: 0,
            diagnostics_session: None,
        }
    }
}

pub type BackendState = Arc<Mutex<BackendProcess>>;
pub type ShutdownFlag = Arc<AtomicBool>;

pub fn new_backend_state() -> BackendState {
    Arc::new(Mutex::new(BackendProcess::default()))
}

pub fn new_shutdown_flag() -> ShutdownFlag {
    Arc::new(AtomicBool::new(false))
}

pub(crate) fn trim_line_endings(bytes: &[u8]) -> &[u8] {
    let mut end = bytes.len();
    while end > 0 && matches!(bytes[end - 1], b'\n' | b'\r') {
        end -= 1;
    }
    &bytes[..end]
}

/// Windows `CREATE_NO_WINDOW` flag — suppresses console windows for child processes.
#[cfg(windows)]
pub(crate) const CREATE_NO_WINDOW: u32 = 0x08000000;

/// Force-kill a Windows process tree via hidden `taskkill /T /F`, falling
/// back to `child.kill()` if taskkill itself fails.  Reaps the child afterward.
#[cfg(windows)]
pub(crate) fn force_kill_process_tree(
    pid: u32,
    child: &mut Box<dyn ChildWrapper + Send>,
    label: &str,
) {
    use std::os::windows::process::CommandExt;

    let taskkill_status = Command::new("taskkill.exe")
        .creation_flags(CREATE_NO_WINDOW)
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    match taskkill_status {
        Ok(status) if status.success() => {}
        Ok(status) => {
            warn!(
                "taskkill returned non-zero status for {} pid {}: {}",
                label, pid, status
            );
            let _ = child.kill();
        }
        Err(e) => {
            warn!("taskkill failed for {} pid {}: {}", label, pid, e);
            let _ = child.kill();
        }
    }

    let _ = child.wait();
    info!("{} process tree force stopped", label);
}

/// Returns the path to the unsloth binary inside the managed venv, if it exists.
/// Checks the new layout (~/.unsloth/studio/unsloth_studio/) first,
/// then falls back to the old layout (~/.unsloth/studio/.venv/) for compat.
fn find_unsloth_binary_in_studio_dir(studio: &std::path::Path) -> Option<std::path::PathBuf> {
    // New layout (upstream scripts >= March 2026)
    let new_base = studio.join("unsloth_studio");
    // Old layout (bundled scripts, older upstream)
    let old_base = studio.join(".venv");

    for base in [new_base, old_base] {
        #[cfg(unix)]
        let bin = base.join("bin").join("unsloth");
        #[cfg(windows)]
        let bin = base.join("Scripts").join("unsloth.exe");

        if bin.exists() {
            return Some(bin);
        }
    }

    None
}

pub fn find_unsloth_binary() -> Option<std::path::PathBuf> {
    let home = dirs::home_dir()?;
    let studio = home.join(".unsloth").join("studio");

    find_unsloth_binary_in_studio_dir(&studio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_studio_dir(test_name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "unsloth-{test_name}-{}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn finds_new_layout_before_legacy_layout() {
        let temp = temp_studio_dir("new-before-legacy");

        #[cfg(unix)]
        let new_bin = temp.join("unsloth_studio/bin/unsloth");
        #[cfg(unix)]
        let old_bin = temp.join(".venv/bin/unsloth");
        #[cfg(windows)]
        let new_bin = temp.join("unsloth_studio/Scripts/unsloth.exe");
        #[cfg(windows)]
        let old_bin = temp.join(".venv/Scripts/unsloth.exe");

        fs::create_dir_all(new_bin.parent().unwrap()).unwrap();
        fs::create_dir_all(old_bin.parent().unwrap()).unwrap();
        fs::write(&new_bin, "").unwrap();
        fs::write(&old_bin, "").unwrap();

        assert_eq!(find_unsloth_binary_in_studio_dir(&temp), Some(new_bin));
        fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn finds_legacy_layout_when_new_missing() {
        let temp = temp_studio_dir("legacy");

        #[cfg(unix)]
        let old_bin = temp.join(".venv/bin/unsloth");
        #[cfg(windows)]
        let old_bin = temp.join(".venv/Scripts/unsloth.exe");

        fs::create_dir_all(old_bin.parent().unwrap()).unwrap();
        fs::write(&old_bin, "").unwrap();

        assert_eq!(find_unsloth_binary_in_studio_dir(&temp), Some(old_bin));
        fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn returns_none_when_no_managed_layout_exists() {
        let temp = temp_studio_dir("none");

        assert_eq!(find_unsloth_binary_in_studio_dir(&temp), None);
        fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn backend_args_always_enable_api_only() {
        assert_eq!(
            backend_args(8888),
            vec!["studio", "--api-only", "-H", "127.0.0.1", "-p", "8888"]
        );
    }
}

/// Find the unsloth binary, preferring the dev repo if available.
/// In dev mode (debug builds), checks for a local .venv in the repo first.
/// Falls back to find_unsloth_binary() which checks ~/.unsloth/studio/unsloth_studio/
/// (new layout) then ~/.unsloth/studio/.venv/ (old layout).
pub(crate) fn resolve_backend_binary() -> Result<std::path::PathBuf, String> {
    // In dev mode, check for local repo venv first
    #[cfg(debug_assertions)]
    {
        // CARGO_MANIFEST_DIR is set at compile time to studio/src-tauri/
        // Repo root is 2 levels up: studio/src-tauri -> studio -> repo_root
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let repo_root = std::path::Path::new(manifest_dir)
            .parent() // studio/
            .and_then(|p| p.parent()); // repo_root/

        if let Some(root) = repo_root {
            #[cfg(unix)]
            let dev_bin = root.join(".venv/bin/unsloth");
            #[cfg(windows)]
            let dev_bin = root.join(".venv/Scripts/unsloth.exe");

            if dev_bin.exists() {
                info!("Dev mode: using local repo backend at {:?}", dev_bin);
                return Ok(dev_bin.to_path_buf());
            }
        }
        info!("Dev mode: no local .venv found, falling back to installed backend");
    }

    find_unsloth_binary()
        .ok_or_else(|| "Unsloth binary not found. Please install Unsloth Studio first.".to_string())
}

fn backend_args(port: u16) -> Vec<String> {
    [
        "studio",
        "--api-only",
        "-H",
        "127.0.0.1",
        "-p",
        &port.to_string(),
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

/// Spawn the backend process and wire up stdout/stderr reader threads.
pub fn start_backend(
    app: &AppHandle,
    state: &BackendState,
    port: u16,
    shutdown: &ShutdownFlag,
    diagnostics_state: &DiagnosticsState,
) -> Result<u64, String> {
    let bin = match resolve_backend_binary() {
        Ok(bin) => bin,
        Err(msg) => {
            diagnostics::record_backend_start_failure(
                diagnostics_state,
                Some(port),
                None,
                "resolve_backend_binary",
                &msg,
            );
            return Err(msg);
        }
    };

    shutdown.store(false, Ordering::SeqCst);

    // Reset state and invalidate any readers from an older backend generation
    // before we release the lock and spawn a new child.
    let generation = {
        let mut proc = state.lock().map_err(|e| e.to_string())?;
        if proc.child.is_some() {
            return Err("Backend is already running.".to_string());
        }
        proc.generation = proc.generation.wrapping_add(1);
        proc.port = None;
        proc.logs.clear();
        proc.intentional_stop = false;
        proc.diagnostics_session = None;
        proc.generation
    };

    let backend_log = diagnostics::begin_backend_session(diagnostics_state, port, generation);

    let args = backend_args(port);
    info!("Starting backend: {:?} {}", bin, args.join(" "));
    diagnostics::append_phase_line(
        &backend_log.handle,
        "meta",
        &format!("Starting backend: {:?} {}", bin, args.join(" ")),
    );

    let mut cmd = Command::new(&bin);
    cmd.args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(native_state) = app.try_state::<crate::native_intents::NativeIntakeState>() {
        cmd.env(
            crate::native_backend_lease::LEASE_SECRET_ENV,
            native_state.lease_secret_env(),
        );
    }

    // AppImage sets LD_LIBRARY_PATH to its bundled libs, which breaks the spawned
    // Python process (wrong libpython/libz → "No module named encodings").
    // Only clear when running inside an AppImage — native .deb/.rpm installs may
    // need these env vars for custom CUDA or conda paths.
    #[cfg(target_os = "linux")]
    if std::env::var_os("APPIMAGE").is_some() {
        cmd.env_remove("LD_LIBRARY_PATH");
        cmd.env_remove("PYTHONHOME");
        cmd.env_remove("PYTHONPATH");
    }

    // Tauri uses the legacy root regardless of UNSLOTH_STUDIO_HOME / STUDIO_HOME;
    // scrub so the spawned Python backend can't diverge. UNSLOTH_LLAMA_CPP_PATH
    // is a pre-existing user-controlled llama.cpp dir override; keep it.
    cmd.env_remove("UNSLOTH_STUDIO_HOME");
    cmd.env_remove("STUDIO_HOME");

    // On Windows, launch the backend directly with hidden-window flags.
    // The app process is assigned to a KILL_ON_JOB_CLOSE job in main.rs, so
    // children inherit crash-safe cleanup without the buggy per-child JobObject wrapper.
    #[cfg(windows)]
    let mut child: Box<dyn ChildWrapper + Send> = {
        use std::os::windows::process::CommandExt;

        const CREATE_NEW_PROCESS_GROUP: u32 = 0x00000200;
        cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
        let child = cmd.spawn().map_err(|e| {
            let msg = format!("Failed to spawn backend: {}", e);
            diagnostics::record_backend_start_failure(
                diagnostics_state,
                Some(port),
                Some(generation),
                "spawn_backend",
                &msg,
            );
            msg
        })?;
        Box::new(child)
    };

    #[cfg(unix)]
    let mut child: Box<dyn ChildWrapper + Send> = {
        // Keep the backend tree in a process group on Unix for cleanup.
        let mut wrap = CommandWrap::from(cmd);
        wrap.wrap(ProcessGroup::leader());
        wrap.spawn().map_err(|e| {
            let msg = format!("Failed to spawn backend: {}", e);
            diagnostics::record_backend_start_failure(
                diagnostics_state,
                Some(port),
                Some(generation),
                "spawn_backend",
                &msg,
            );
            msg
        })?
    };

    let stdout = child.stdout().take();
    let stderr = child.stderr().take();

    // Store child in state for the already-selected generation.
    {
        let mut proc = state.lock().map_err(|e| e.to_string())?;
        proc.child = Some(child);
        proc.diagnostics_session = Some(backend_log.clone());
    }

    // Spawn stdout reader thread
    if let Some(stdout) = stdout {
        let app_handle = app.clone();
        let state_clone = Arc::clone(state);
        let diagnostics_clone = diagnostics_state.clone();
        let backend_log_clone = backend_log.clone();
        std::thread::spawn(move || {
            read_output_stream(
                stdout,
                &app_handle,
                &state_clone,
                &diagnostics_clone,
                &backend_log_clone,
                false,
                generation,
            );
        });
    }

    // Spawn stderr reader thread
    if let Some(stderr) = stderr {
        let app_handle = app.clone();
        let state_clone = Arc::clone(state);
        let diagnostics_clone = diagnostics_state.clone();
        let backend_log_clone = backend_log.clone();
        std::thread::spawn(move || {
            read_output_stream(
                stderr,
                &app_handle,
                &state_clone,
                &diagnostics_clone,
                &backend_log_clone,
                true,
                generation,
            );
        });
    }

    Ok(generation)
}

/// Read lines from a child process stream (stdout or stderr).
/// For stdout, parse TAURI_PORT=(\d+) to detect the actual port.
/// When stdout closes and the stop was not intentional, emit server-crashed.
fn read_output_stream<R: std::io::Read>(
    stream: R,
    app: &AppHandle,
    state: &BackendState,
    diagnostics_state: &DiagnosticsState,
    backend_log: &BackendLog,
    is_stderr: bool,
    generation: u64,
) {
    let mut reader = std::io::BufReader::new(stream);
    let port_re = Regex::new(r"TAURI_PORT=(\d+)").unwrap();
    let mut buf = Vec::new();

    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) => break,
            Ok(_) => {
                let text = String::from_utf8_lossy(trim_line_endings(&buf)).into_owned();
                let log_line = if is_stderr {
                    format!("[stderr] {}", text)
                } else {
                    text.clone()
                };

                diagnostics::append_phase_line(
                    &backend_log.handle,
                    if is_stderr { "stderr" } else { "stdout" },
                    &text,
                );

                // Check for TAURI_PORT on stdout only
                let detected_port = if !is_stderr {
                    port_re
                        .captures(&text)
                        .and_then(|caps| caps.get(1))
                        .and_then(|port_str| port_str.as_str().parse::<u16>().ok())
                } else {
                    None
                };

                // Buffer the log line only for the current backend generation.
                // Old reader threads can briefly outlive a stop/start cycle;
                // they must not overwrite the new backend's port or logs.
                let mut should_record_port = None;
                let current_generation = if let Ok(mut proc) = state.lock() {
                    if proc.generation != generation {
                        false
                    } else {
                        if let Some(port) = detected_port {
                            proc.port = Some(port);
                            should_record_port = Some(port);
                        }
                        if proc.logs.len() >= MAX_LOG_LINES {
                            proc.logs.pop_front();
                        }
                        proc.logs.push_back(log_line.clone());
                        true
                    }
                } else {
                    false
                };

                if !current_generation {
                    break;
                }

                if let Some(port) = should_record_port {
                    diagnostics::record_backend_port(
                        diagnostics_state,
                        &backend_log.session_id,
                        port,
                    );
                    info!("Detected backend port: {}", port);
                    let _ = app.emit("server-port", port);
                }

                info!("[backend] {}", log_line);

                // Emit to frontend
                let _ = app.emit("server-log", &log_line);
            }
            Err(e) => {
                warn!(
                    "Error reading backend {}: {}",
                    if is_stderr { "stderr" } else { "stdout" },
                    e
                );
                break;
            }
        }
    }

    // Stream closed. Only the stdout reader checks for crashes.
    if !is_stderr {
        let mut exit_record: Option<(String, bool)> = None;
        let mut emit_crash = false;
        if let Ok(mut proc) = state.lock() {
            if proc.generation != generation {
                return;
            }
            let intentional = proc.intentional_stop;
            let exited = if let Some(ref mut child) = proc.child {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        info!("Backend stdout stream ended with status: {}", status);
                        exit_record = Some((status.to_string(), intentional));
                        true
                    }
                    Ok(None) => {
                        warn!("Backend stdout stream ended, but process is still running");
                        false
                    }
                    Err(e) => {
                        warn!("Failed to query backend status after stdout closed: {}", e);
                        false
                    }
                }
            } else {
                false
            };

            if exited {
                proc.child = None;
                proc.diagnostics_session = None;
                emit_crash = !intentional;
            }
        }
        if let Some((status, intentional)) = exit_record {
            diagnostics::record_backend_exit(
                diagnostics_state,
                &backend_log.session_id,
                Some(status),
                intentional,
                None,
            );
        }
        if emit_crash {
            error!("Backend process stdout closed unexpectedly (crash detected)");
            let _ = app.emit("server-crashed", ());
        }
    }
}

/// Graceful shutdown of the backend process and its entire subprocess tree.
/// Unix: SIGTERM to process group -> wait up to 5s -> SIGKILL to group
/// Windows: CTRL_BREAK_EVENT -> wait up to 5s -> hidden taskkill /T /F
pub fn stop_backend(
    state: &BackendState,
    shutdown: &ShutdownFlag,
    diagnostics_state: Option<&DiagnosticsState>,
) -> Result<(), String> {
    shutdown.store(true, Ordering::SeqCst);
    if let Some(diagnostics_state) = diagnostics_state {
        diagnostics::record_backend_intentional_stop(diagnostics_state);
    }

    // Extract the child and mark intentional stop.
    // We take the child OUT of the mutex so we don't hold the lock during the wait loop.
    let mut child = {
        let mut proc = match state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                warn!("Backend state mutex poisoned, recovering for cleanup");
                poisoned.into_inner()
            }
        };
        proc.intentional_stop = true;
        proc.child.take()
    };

    let Some(ref mut child) = child else {
        return Ok(()); // Nothing running
    };

    let pid = child.id();
    info!("Stopping backend process group (pid {})", pid);

    // Send SIGTERM to the entire process group (negative PID = group signal).
    // This gives Python and any workers a chance to shut down gracefully.
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
    }
    #[cfg(windows)]
    {
        unsafe {
            windows_sys::Win32::System::Console::GenerateConsoleCtrlEvent(
                windows_sys::Win32::System::Console::CTRL_BREAK_EVENT,
                pid,
            );
        }
    }

    // Poll for up to 5 seconds (50 iterations * 100ms)
    for _ in 0..50 {
        match child.try_wait() {
            Ok(Some(status)) => {
                info!("Backend exited with status: {}", status);
                return Ok(());
            }
            Ok(None) => {
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            Err(e) => {
                warn!("Error polling backend process: {}", e);
                break;
            }
        }
    }

    #[cfg(windows)]
    {
        warn!(
            "Backend did not exit gracefully, force killing process tree (pid {})",
            pid
        );
        force_kill_process_tree(pid, child, "Backend");
        return Ok(());
    }

    #[cfg(unix)]
    {
        // Force kill the process group on Unix
        warn!(
            "Backend did not exit gracefully, force killing group (pid {})",
            pid
        );
        let _ = child.kill();

        // Reap the process
        let _ = child.wait();
        info!("Backend process group forcefully stopped");
        Ok(())
    }
}
