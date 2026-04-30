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
}

impl Default for InstallProcess {
    fn default() -> Self {
        Self {
            child: None,
            intentional_stop: false,
            needed_packages: Vec::new(),
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
    stdout: Option<std::process::ChildStdout>,
    stderr: Option<std::process::ChildStderr>,
) -> Vec<std::thread::JoinHandle<()>> {
    let mut threads = Vec::new();

    if let Some(out) = stdout {
        let app_clone = app.clone();
        let state_clone = Arc::clone(state);
        threads.push(std::thread::spawn(move || {
            let mut reader = std::io::BufReader::new(out);
            let mut buf = Vec::new();
            loop {
                buf.clear();
                match reader.read_until(b'\n', &mut buf) {
                    Ok(0) => break,
                    Ok(_) => {
                        let text = String::from_utf8_lossy(trim_line_endings(&buf)).into_owned();
                        // Parse structured Tauri protocol lines
                        if let Some(packages) = text.strip_prefix("[TAURI:NEED_SUDO] ") {
                            let pkgs: Vec<String> =
                                packages.split_whitespace().map(String::from).collect();
                            if let Ok(mut install) = state_clone.lock() {
                                install.needed_packages = pkgs;
                            }
                        } else if let Some(step) = text.strip_prefix("[TAURI:STEP] ") {
                            if !event_mode.emit_install_structured_events() {
                                info!("[install][stdout] {}", text);
                                let _ = app_clone.emit(event_mode.progress_event(), &text);
                                continue;
                            }
                            let _ = app_clone.emit("install-step", step);
                        } else if let Some(detail) = text.strip_prefix("[TAURI:PROGRESS] ") {
                            if !event_mode.emit_install_structured_events() {
                                info!("[install][stdout] {}", text);
                                let _ = app_clone.emit(event_mode.progress_event(), detail);
                                continue;
                            }
                            let _ = app_clone.emit("install-progress-detail", detail);
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
        threads.push(std::thread::spawn(move || {
            let mut reader = std::io::BufReader::new(err);
            let mut buf = Vec::new();
            loop {
                buf.clear();
                match reader.read_until(b'\n', &mut buf) {
                    Ok(0) => break,
                    Ok(_) => {
                        let text = String::from_utf8_lossy(trim_line_endings(&buf)).into_owned();
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
pub fn run_install(app: AppHandle, state: InstallState) -> Result<(), String> {
    run_install_with_event_mode(app, state, InstallEventMode::Full)
}

pub(crate) fn run_install_for_repair(app: AppHandle, state: InstallState) -> Result<(), String> {
    run_install_with_event_mode(app, state, InstallEventMode::Repair)
}

fn run_install_with_event_mode(
    app: AppHandle,
    state: InstallState,
    event_mode: InstallEventMode,
) -> Result<(), String> {
    emit_mode_progress(&app, event_mode, "Starting installation...");

    let (script, args) = resolve_install_script(&app)?;
    emit_mode_progress(
        &app,
        event_mode,
        &format!("Using script: {}", script.display()),
    );

    let (stdout, stderr) = spawn_script(&script, &args, &state)?;
    let threads = stream_output(&app, &state, event_mode, stdout, stderr);

    // Wait for exit, join reader threads
    let result = wait_for_exit(&state);
    for handle in threads {
        let _ = handle.join();
    }

    match result {
        Ok((status, _)) if status.success() => {
            if event_mode.emit_terminal_events() {
                emit_complete(&app);
            }
            Ok(())
        }
        Ok((status, _)) => {
            let code = status.code().unwrap_or(-1);
            if code == 2 {
                // Script needs elevated package install — report to frontend
                let packages = state
                    .lock()
                    .map(|i| i.needed_packages.clone())
                    .unwrap_or_default();
                info!("[install] Needs elevation for packages: {:?}", packages);
                let _ = app.emit(event_mode.needs_elevation_event(), &packages);
                Err("NEEDS_ELEVATION".to_string())
            } else {
                let msg = format!("Installer exited with code {}", code);
                if event_mode.emit_terminal_events() {
                    emit_failed(&app, &msg);
                }
                Err(msg)
            }
        }
        Err(msg) if msg == "Installation stopped." => {
            info!("[install] Installation stopped intentionally");
            Err(msg)
        }
        Err(msg) => {
            if event_mode.emit_terminal_events() {
                emit_failed(&app, &msg);
            }
            Err(msg)
        }
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

/// Install system packages with elevated permissions (Linux only).
/// Uses `elevated-command` crate for native auth dialog.
#[cfg(target_os = "linux")]
pub fn install_system_packages(packages: &[String]) -> Result<(), String> {
    use regex::Regex;
    use std::path::Path;
    use std::process::Command as StdCommand;

    // Validate package names to prevent injection via elevated command.
    let valid_pkg = Regex::new(r"^[a-zA-Z0-9][a-zA-Z0-9.+\-]*$").unwrap();
    for pkg in packages {
        if !valid_pkg.is_match(pkg) {
            return Err(format!("Invalid package name: {}", pkg));
        }
    }

    // AppImage bundles run on non-Debian distros too. Pick the first package
    // manager we find. Names in `packages` are Debian-style; callers that want
    // cross-distro support should translate before invoking.
    let (program, base_args): (&str, &[&str]) = if Path::new("/usr/bin/apt-get").exists() {
        ("apt-get", &["install", "-y"])
    } else if Path::new("/usr/bin/dnf").exists() {
        ("dnf", &["install", "-y"])
    } else if Path::new("/usr/bin/zypper").exists() {
        ("zypper", &["install", "-y"])
    } else if Path::new("/usr/bin/pacman").exists() {
        ("pacman", &["-S", "--noconfirm"])
    } else {
        return Err(
            "No supported system package manager found (apt-get, dnf, zypper, pacman)".to_string(),
        );
    };

    info!(
        "[install] Elevated install of packages via {}: {}",
        program,
        packages.join(", ")
    );

    let mut cmd = StdCommand::new(program);
    cmd.args(base_args).args(packages);

    let elevated = elevated_command::Command::new(cmd)
        .output()
        .map_err(|e| format!("Elevated install failed: {}", e))?;

    if !elevated.status.success() {
        let stderr = String::from_utf8_lossy(&elevated.stderr);
        return Err(format!("Package installation failed: {}", stderr));
    }

    info!("[install] Elevated package install succeeded");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
