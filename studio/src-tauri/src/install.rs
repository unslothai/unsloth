use command_group::{CommandGroup, GroupChild};
use log::{error, info, warn};
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager};

// ── Types ──

pub struct InstallProcess {
    /// Process group handle — killing this kills the entire subprocess tree.
    pub child: Option<GroupChild>,
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
        let name = if cfg!(unix) { "install.sh" } else { "install.ps1" };
        let script = app
            .path()
            .resolve(name, tauri::path::BaseDirectory::Resource)
            .map_err(|e| format!("Failed to resolve bundled {}: {}", name, e))?;
        info!("Production: using bundled script at {}", script.display());
        Ok((script, args))
    }
}

// ── Emit Helpers ──

fn emit_progress(app: &AppHandle, message: &str) {
    info!("[install] {}", message);
    let _ = app.emit("install-progress", message);
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
) -> Result<(Option<std::process::ChildStdout>, Option<std::process::ChildStderr>), String> {
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
    let mut cmd = Command::new("powershell");
    #[cfg(windows)]
    cmd.args(["-ExecutionPolicy", "Bypass", "-File"])
        .arg(script)
        .args(args)
        .current_dir(&work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    // AppImage/Flatpak set LD_LIBRARY_PATH to their bundled libs, which breaks
    // Python spawned by the install script. Clear these env vars.
    #[cfg(target_os = "linux")]
    {
        cmd.env_remove("LD_LIBRARY_PATH");
        cmd.env_remove("PYTHONHOME");
        cmd.env_remove("PYTHONPATH");
    }

    // Spawn in a process group so kill() terminates the entire tree
    let mut group_child = cmd
        .group()
        .spawn()
        .map_err(|e| format!("Failed to spawn install script: {}", e))?;

    let stdout = group_child.inner().stdout.take();
    let stderr = group_child.inner().stderr.take();
    install.child = Some(group_child);
    Ok((stdout, stderr))
}

// ── Stream ──

/// Spawns reader threads for stdout/stderr.
/// Parses [TAURI:*] lines from stdout for structured events.
fn stream_output(
    app: &AppHandle,
    state: &InstallState,
    stdout: Option<std::process::ChildStdout>,
    stderr: Option<std::process::ChildStderr>,
) -> Vec<std::thread::JoinHandle<()>> {
    let mut threads = Vec::new();

    if let Some(out) = stdout {
        let app_clone = app.clone();
        let state_clone = Arc::clone(state);
        threads.push(std::thread::spawn(move || {
            let reader = std::io::BufReader::new(out);
            for line in reader.lines() {
                match line {
                    Ok(text) => {
                        // Parse structured Tauri protocol lines
                        if let Some(packages) = text.strip_prefix("[TAURI:NEED_SUDO] ") {
                            let pkgs: Vec<String> =
                                packages.split_whitespace().map(String::from).collect();
                            if let Ok(mut install) = state_clone.lock() {
                                install.needed_packages = pkgs;
                            }
                        } else if let Some(step) = text.strip_prefix("[TAURI:STEP] ") {
                            let _ = app_clone.emit("install-step", step);
                        } else if let Some(detail) = text.strip_prefix("[TAURI:PROGRESS] ") {
                            let _ = app_clone.emit("install-progress-detail", detail);
                        }
                        // Always forward the raw line
                        info!("[install][stdout] {}", text);
                        let _ = app_clone.emit("install-progress", &text);
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
            let reader = std::io::BufReader::new(err);
            for line in reader.lines() {
                match line {
                    Ok(text) => {
                        warn!("[install][stderr] {}", text);
                        let _ = app_clone.emit("install-progress", &text);
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
fn wait_for_exit(state: &InstallState) -> Result<(ExitStatus, bool), String> {
    loop {
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
}

// ── Public API ──

/// Run the install script. Returns Ok(()) on success.
/// Returns Err("NEEDS_ELEVATION") if system packages need elevated install (Linux only).
/// Returns Err(message) on other failures.
pub fn run_install(app: AppHandle, state: InstallState) -> Result<(), String> {
    emit_progress(&app, "Starting installation...");

    let (script, args) = resolve_install_script(&app)?;
    emit_progress(&app, &format!("Using script: {}", script.display()));

    let (stdout, stderr) = spawn_script(&script, &args, &state)?;
    let threads = stream_output(&app, &state, stdout, stderr);

    // Wait for exit, join reader threads
    let result = wait_for_exit(&state);
    for handle in threads {
        let _ = handle.join();
    }

    match result {
        Ok((status, _)) if status.success() => {
            emit_complete(&app);
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
                let _ = app.emit("install-needs-elevation", &packages);
                Err("NEEDS_ELEVATION".to_string())
            } else {
                let msg = format!("Installer exited with code {}", code);
                emit_failed(&app, &msg);
                Err(msg)
            }
        }
        Err(msg) if msg == "Installation stopped." => {
            info!("[install] Installation stopped intentionally");
            Err(msg)
        }
        Err(msg) => {
            emit_failed(&app, &msg);
            Err(msg)
        }
    }
}

/// Stop a running install process.
/// Kills the entire process group (bash/powershell + all subprocesses like uv, cmake, pip).
pub fn stop_install(state: &InstallState) -> Result<(), String> {
    let mut child = {
        let mut install = state.lock().map_err(|e| e.to_string())?;
        install.intentional_stop = true;
        install.child.take()
    };

    let Some(ref mut child) = child else {
        return Ok(());
    };

    let pid = child.id();
    info!("Stopping installer process group (pid {})", pid);
    // GroupChild::kill() sends SIGKILL to the entire process group on Unix
    // and terminates all processes in the job object on Windows.
    let _ = child.kill();
    let _ = child.wait();
    info!("Installer process group stopped");
    Ok(())
}

/// Install system packages with elevated permissions (Linux only).
/// Uses `elevated-command` crate for native auth dialog.
#[cfg(target_os = "linux")]
pub fn install_system_packages(packages: &[String]) -> Result<(), String> {
    use regex::Regex;
    use std::process::Command as StdCommand;

    // Validate package names to prevent injection via elevated command.
    // Only allow standard Debian package name characters.
    let valid_pkg = Regex::new(r"^[a-zA-Z0-9][a-zA-Z0-9.+\-]*$").unwrap();
    for pkg in packages {
        if !valid_pkg.is_match(pkg) {
            return Err(format!("Invalid package name: {}", pkg));
        }
    }

    info!(
        "[install] Elevated install of packages: {}",
        packages.join(", ")
    );

    let mut cmd = StdCommand::new("apt-get");
    cmd.args(["install", "-y"]).args(packages);

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
