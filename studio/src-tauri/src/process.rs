use log::{error, info, warn};
use process_wrap::std::*;
use regex::Regex;
use std::collections::VecDeque;
use std::io::BufRead;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};

const MAX_LOG_LINES: usize = 1000;

pub struct BackendProcess {
    pub child: Option<Box<dyn ChildWrapper + Send>>,
    pub port: Option<u16>,
    pub logs: VecDeque<String>,
    pub intentional_stop: bool,
}

impl Default for BackendProcess {
    fn default() -> Self {
        Self {
            child: None,
            port: None,
            logs: VecDeque::with_capacity(MAX_LOG_LINES),
            intentional_stop: false,
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
pub fn find_unsloth_binary() -> Option<std::path::PathBuf> {
    let home = dirs::home_dir()?;
    let studio = home.join(".unsloth").join("studio");

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

/// Find the unsloth binary, preferring the dev repo if available.
/// In dev mode (debug builds), checks for a local .venv in the repo first.
/// Falls back to find_unsloth_binary() which checks ~/.unsloth/studio/unsloth_studio/
/// (new layout) then ~/.unsloth/studio/.venv/ (old layout).
fn resolve_backend_binary() -> Result<std::path::PathBuf, String> {
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

/// Check if the installed unsloth CLI supports --api-only by probing --help output.
/// Times out after 5s to avoid blocking startup if the binary hangs.
fn supports_api_only(bin: &std::path::Path) -> bool {
    let mut cmd = std::process::Command::new(bin);
    cmd.args(["studio", "--help"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null());

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(_) => return false,
    };

    // Poll for up to 5 seconds
    for _ in 0..50 {
        match child.try_wait() {
            Ok(Some(_)) => {
                let stdout = child.stdout.take();
                if let Some(mut out) = stdout {
                    let mut buf = String::new();
                    use std::io::Read;
                    let _ = out.read_to_string(&mut buf);
                    return buf.contains("--api-only");
                }
                return false;
            }
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(100)),
            Err(_) => return false,
        }
    }
    // Timed out — kill and assume not supported
    let _ = child.kill();
    let _ = child.wait();
    false
}

/// Spawn the backend process and wire up stdout/stderr reader threads.
pub fn start_backend(
    app: &AppHandle,
    state: &BackendState,
    port: u16,
    shutdown: &ShutdownFlag,
) -> Result<(), String> {
    let bin = resolve_backend_binary()?;

    shutdown.store(false, Ordering::SeqCst);

    // Reset state
    {
        let mut proc = state.lock().map_err(|e| e.to_string())?;
        if proc.child.is_some() {
            return Err("Backend is already running.".to_string());
        }
        proc.port = None;
        proc.logs.clear();
        proc.intentional_stop = false;
    }

    // In dev, always use --api-only (local code has it).
    // In production, probe the binary to check if the PyPI version supports it.
    let use_api_only = if cfg!(debug_assertions) {
        true
    } else {
        supports_api_only(&bin)
    };

    info!(
        "Starting backend: {:?} studio {}-H 127.0.0.1 -p {}",
        bin,
        if use_api_only { "--api-only " } else { "" },
        port
    );

    let mut cmd = Command::new(&bin);
    cmd.arg("studio");
    if use_api_only {
        cmd.arg("--api-only");
    }
    cmd.args(["-H", "127.0.0.1", "-p", &port.to_string()])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

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

    // On Windows, launch the backend directly with hidden-window flags.
    // The app process is assigned to a KILL_ON_JOB_CLOSE job in main.rs, so
    // children inherit crash-safe cleanup without the buggy per-child JobObject wrapper.
    #[cfg(windows)]
    let mut child: Box<dyn ChildWrapper + Send> = {
        use std::os::windows::process::CommandExt;

        const CREATE_NEW_PROCESS_GROUP: u32 = 0x00000200;
        cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
        let child = cmd
            .spawn()
            .map_err(|e| format!("Failed to spawn backend: {}", e))?;
        Box::new(child)
    };

    #[cfg(unix)]
    let mut child: Box<dyn ChildWrapper + Send> = {
        // Keep the backend tree in a process group on Unix for cleanup.
        let mut wrap = CommandWrap::from(cmd);
        wrap.wrap(ProcessGroup::leader());
        wrap.spawn()
            .map_err(|e| format!("Failed to spawn backend: {}", e))?
    };

    let stdout = child.stdout().take();
    let stderr = child.stderr().take();

    // Store child in state
    {
        let mut proc = state.lock().map_err(|e| e.to_string())?;
        proc.child = Some(child);
    }

    // Spawn stdout reader thread
    if let Some(stdout) = stdout {
        let app_handle = app.clone();
        let state_clone = Arc::clone(state);
        std::thread::spawn(move || {
            read_output_stream(stdout, &app_handle, &state_clone, false);
        });
    }

    // Spawn stderr reader thread
    if let Some(stderr) = stderr {
        let app_handle = app.clone();
        let state_clone = Arc::clone(state);
        std::thread::spawn(move || {
            read_output_stream(stderr, &app_handle, &state_clone, true);
        });
    }

    Ok(())
}

/// Read lines from a child process stream (stdout or stderr).
/// For stdout, parse TAURI_PORT=(\d+) to detect the actual port.
/// When stdout closes and the stop was not intentional, emit server-crashed.
fn read_output_stream<R: std::io::Read>(
    stream: R,
    app: &AppHandle,
    state: &BackendState,
    is_stderr: bool,
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

                // Check for TAURI_PORT on stdout only
                if !is_stderr {
                    if let Some(caps) = port_re.captures(&text) {
                        if let Some(port_str) = caps.get(1) {
                            if let Ok(port) = port_str.as_str().parse::<u16>() {
                                info!("Detected backend port: {}", port);
                                if let Ok(mut proc) = state.lock() {
                                    proc.port = Some(port);
                                }
                                let _ = app.emit("server-port", port);
                            }
                        }
                    }
                }

                // Buffer the log line
                if let Ok(mut proc) = state.lock() {
                    if proc.logs.len() >= MAX_LOG_LINES {
                        proc.logs.pop_front();
                    }
                    proc.logs.push_back(log_line.clone());
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
        if let Ok(mut proc) = state.lock() {
            let intentional = proc.intentional_stop;
            let exited = if let Some(ref mut child) = proc.child {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        info!("Backend stdout stream ended with status: {}", status);
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
                if !intentional {
                    error!("Backend process stdout closed unexpectedly (crash detected)");
                    let _ = app.emit("server-crashed", ());
                }
            }
        }
    }
}

/// Graceful shutdown of the backend process and its entire subprocess tree.
/// Unix: SIGTERM to process group -> wait up to 5s -> SIGKILL to group
/// Windows: CTRL_BREAK_EVENT -> wait up to 5s -> hidden taskkill /T /F
pub fn stop_backend(state: &BackendState, shutdown: &ShutdownFlag) -> Result<(), String> {
    shutdown.store(true, Ordering::SeqCst);

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
