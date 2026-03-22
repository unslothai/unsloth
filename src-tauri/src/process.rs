use log::{error, info, warn};
use regex::Regex;
use std::collections::VecDeque;
use std::io::BufRead;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};

const MAX_LOG_LINES: usize = 1000;

pub struct BackendProcess {
    pub child: Option<Child>,
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

pub fn new_backend_state() -> BackendState {
    Arc::new(Mutex::new(BackendProcess::default()))
}

/// Returns the path to the unsloth binary inside the managed venv, if it exists.
pub fn find_unsloth_binary() -> Option<std::path::PathBuf> {
    let home = dirs::home_dir()?;
    let base = home.join(".unsloth").join("studio").join(".venv");

    #[cfg(unix)]
    let bin = base.join("bin").join("unsloth");
    #[cfg(windows)]
    let bin = base.join("Scripts").join("unsloth.exe");

    if bin.exists() {
        Some(bin)
    } else {
        None
    }
}

/// Spawn the backend process and wire up stdout/stderr reader threads.
pub fn start_backend(
    app: &AppHandle,
    state: &BackendState,
    port: u16,
) -> Result<(), String> {
    let bin = find_unsloth_binary().ok_or_else(|| {
        "Unsloth binary not found. Please install Unsloth Studio first.".to_string()
    })?;

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

    info!("Starting backend: {:?} studio --api-only -H 127.0.0.1 -p {}", bin, port);

    let mut cmd = Command::new(&bin);
    cmd.args(["studio", "--api-only", "-H", "127.0.0.1", "-p", &port.to_string()])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // On Windows, create a new process group so CTRL_BREAK_EVENT works.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x00000200;
        cmd.creation_flags(CREATE_NEW_PROCESS_GROUP);
    }

    let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn backend: {}", e))?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

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
    let reader = std::io::BufReader::new(stream);
    let port_re = Regex::new(r"TAURI_PORT=(\d+)").unwrap();

    for line in reader.lines() {
        match line {
            Ok(text) => {
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

                // Emit to frontend
                let _ = app.emit("server-log", &log_line);
            }
            Err(e) => {
                warn!("Error reading backend {}: {}", if is_stderr { "stderr" } else { "stdout" }, e);
                break;
            }
        }
    }

    // Stream closed. Only the stdout reader checks for crashes.
    if !is_stderr {
        let intentional = state
            .lock()
            .map(|proc| proc.intentional_stop)
            .unwrap_or(false);
        if !intentional {
            error!("Backend process stdout closed unexpectedly (crash detected)");
            let _ = app.emit("server-crashed", ());
        }
        // Clean up the child handle since the process has exited
        if let Ok(mut proc) = state.lock() {
            // Try to wait on the child to reap it
            if let Some(ref mut child) = proc.child {
                let _ = child.wait();
            }
            proc.child = None;
        }
    }
}

/// Graceful shutdown of the backend process.
/// Unix: SIGTERM -> wait up to 5s -> SIGKILL
/// Windows: CTRL_BREAK_EVENT -> wait up to 5s -> TerminateProcess
pub fn stop_backend(state: &BackendState) -> Result<(), String> {
    // Extract the child and mark intentional stop.
    // We take the child OUT of the mutex so we don't hold the lock during the wait loop.
    let mut child = {
        let mut proc = state.lock().map_err(|e| e.to_string())?;
        proc.intentional_stop = true;
        proc.child.take()
    };

    let Some(ref mut child) = child else {
        return Ok(()); // Nothing running
    };

    let pid = child.id();
    info!("Stopping backend process (pid {})", pid);

    // Send the initial signal
    #[cfg(unix)]
    {
        unsafe {
            libc::kill(pid as i32, libc::SIGTERM);
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

    // Force kill
    warn!("Backend did not exit gracefully, force killing (pid {})", pid);
    #[cfg(unix)]
    {
        unsafe {
            libc::kill(pid as i32, libc::SIGKILL);
        }
    }
    #[cfg(windows)]
    {
        let _ = child.kill();
    }

    // Reap the process
    let _ = child.wait();
    info!("Backend process forcefully stopped");
    Ok(())
}
