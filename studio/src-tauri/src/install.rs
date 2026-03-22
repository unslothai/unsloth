use log::{error, info, warn};
use std::io::BufRead;
use std::process::{Child, ChildStderr, ChildStdout, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager};

pub struct InstallProcess {
    pub child: Option<Child>,
    pub intentional_stop: bool,
}

impl Default for InstallProcess {
    fn default() -> Self {
        Self {
            child: None,
            intentional_stop: false,
        }
    }
}

pub type InstallState = Arc<Mutex<InstallProcess>>;

pub fn new_install_state() -> InstallState {
    Arc::new(Mutex::new(InstallProcess::default()))
}

/// Emit an install-progress event to the frontend.
fn emit_progress(app: &AppHandle, message: &str) {
    info!("[install] {}", message);
    let _ = app.emit("install-progress", message);
}

/// Emit an install-failed event to the frontend.
fn emit_failed(app: &AppHandle, message: &str) {
    error!("[install] FAILED: {}", message);
    let _ = app.emit("install-failed", message);
}

/// Emit an install-complete event to the frontend.
fn emit_complete(app: &AppHandle) {
    info!("[install] Installation complete");
    let _ = app.emit("install-complete", ());
}

fn wait_for_install_exit(state: &InstallState) -> Result<(ExitStatus, bool), String> {
    loop {
        let mut install = state.lock().map_err(|e| e.to_string())?;
        let intentional_stop = install.intentional_stop;

        match install.child.as_mut() {
            Some(child) => match child.try_wait() {
                Ok(Some(status)) => {
                    install.child = None;
                    return Ok((status, intentional_stop));
                }
                Ok(None) => {}
                Err(e) => {
                    install.child = None;
                    return Err(format!("Failed while waiting for installer: {}", e));
                }
            },
            None if intentional_stop => return Err("Installation stopped.".to_string()),
            None => return Err("Installer process handle disappeared unexpectedly.".to_string()),
        }

        drop(install);
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

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
    info!("Stopping installer process (pid {})", pid);
    let _ = child.kill();
    let _ = child.wait();
    info!("Installer process stopped");
    Ok(())
}

/// Stream stdout and stderr from a child process, emitting install-progress events.
fn stream_child_output(
    app: &AppHandle,
    stdout: Option<ChildStdout>,
    stderr: Option<ChildStderr>,
) -> Vec<std::thread::JoinHandle<()>> {
    let mut threads = Vec::new();

    if let Some(out) = stdout {
        let app_stdout = app.clone();
        threads.push(std::thread::spawn(move || {
            let reader = std::io::BufReader::new(out);
            for line in reader.lines() {
                match line {
                    Ok(text) => {
                        info!("[install][stdout] {}", text);
                        let _ = app_stdout.emit("install-progress", &text);
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
        let app_stderr = app.clone();
        threads.push(std::thread::spawn(move || {
            let reader = std::io::BufReader::new(err);
            for line in reader.lines() {
                match line {
                    Ok(text) => {
                        warn!("[install][stderr] {}", text);
                        let _ = app_stderr.emit("install-progress", &text);
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

fn finalize_install(
    app: &AppHandle,
    state: &InstallState,
    threads: Vec<std::thread::JoinHandle<()>>,
) -> Result<(), String> {
    let result = wait_for_install_exit(state);

    for handle in threads {
        let _ = handle.join();
    }

    match result {
        Ok((status, _)) if status.success() => {
            emit_complete(app);
            Ok(())
        }
        Ok((status, _)) => {
            let code = status.code().unwrap_or(-1);
            let msg = format!("Installer exited with code {}", code);
            emit_failed(app, &msg);
            Err(msg)
        }
        Err(msg) if msg == "Installation stopped." => {
            info!("[install] Installation stopped intentionally");
            Err(msg)
        }
        Err(msg) => {
            emit_failed(app, &msg);
            Err(msg)
        }
    }
}

#[cfg(unix)]
pub fn run_install(app: AppHandle, state: InstallState) -> Result<(), String> {
    emit_progress(&app, "Starting installation...");

    let required_tools = ["bash", "git", "cmake", "gcc", "curl"];
    let mut missing = Vec::new();
    for tool in &required_tools {
        let status = Command::new("which")
            .arg(tool)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        match status {
            Ok(s) if s.success() => {}
            _ => missing.push(*tool),
        }
    }

    if !missing.is_empty() {
        let msg = format!(
            "Missing required tools: {}. Please install them and try again.\n\
             On Ubuntu/Debian: sudo apt install {}\n\
             On Fedora: sudo dnf install {}\n\
             On macOS: brew install {}",
            missing.join(", "),
            missing.join(" "),
            missing.join(" "),
            missing.join(" "),
        );
        emit_failed(&app, &msg);
        return Err(msg);
    }
    emit_progress(&app, "All required tools found.");

    let home = dirs::home_dir().ok_or_else(|| {
        let msg = "Could not determine home directory".to_string();
        emit_failed(&app, &msg);
        msg
    })?;
    let unsloth_dir = home.join(".unsloth");
    if !unsloth_dir.exists() {
        std::fs::create_dir_all(&unsloth_dir).map_err(|e| {
            let msg = format!("Failed to create ~/.unsloth/: {}", e);
            emit_failed(&app, &msg);
            msg
        })?;
    }
    emit_progress(&app, "~/.unsloth/ directory ready.");

    let install_script = app
        .path()
        .resolve("resources/install.sh", tauri::path::BaseDirectory::Resource)
        .map_err(|e| {
            let msg = format!("Failed to resolve install.sh resource: {}", e);
            emit_failed(&app, &msg);
            msg
        })?;

    if !install_script.exists() {
        let msg = format!(
            "install.sh not found at resolved path: {}",
            install_script.display()
        );
        emit_failed(&app, &msg);
        return Err(msg);
    }
    emit_progress(
        &app,
        &format!("Using install script: {}", install_script.display()),
    );

    emit_progress(&app, "Running install.sh...");
    let (stdout, stderr) = {
        let mut install = state.lock().map_err(|e| e.to_string())?;
        if install.child.is_some() {
            let msg = "Installation is already running.".to_string();
            emit_failed(&app, &msg);
            return Err(msg);
        }
        install.intentional_stop = false;

        let mut child = Command::new("bash")
            .arg(&install_script)
            .current_dir(&unsloth_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                let msg = format!("Failed to spawn install.sh: {}", e);
                emit_failed(&app, &msg);
                msg
            })?;

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        install.child = Some(child);
        (stdout, stderr)
    };
    let threads = stream_child_output(&app, stdout, stderr);

    finalize_install(&app, &state, threads)
}

#[cfg(windows)]
pub fn run_install(app: AppHandle, state: InstallState) -> Result<(), String> {
    emit_progress(&app, "Starting installation on Windows...");

    let home = dirs::home_dir().ok_or_else(|| {
        let msg = "Could not determine home directory".to_string();
        emit_failed(&app, &msg);
        msg
    })?;
    let unsloth_dir = home.join(".unsloth");

    if !unsloth_dir.exists() {
        std::fs::create_dir_all(&unsloth_dir).map_err(|e| {
            let msg = format!("Failed to create {}: {}", unsloth_dir.display(), e);
            emit_failed(&app, &msg);
            msg
        })?;
    }
    emit_progress(&app, &format!("{} directory ready.", unsloth_dir.display()));

    let install_script = app
        .path()
        .resolve(
            "resources/install.ps1",
            tauri::path::BaseDirectory::Resource,
        )
        .map_err(|e| {
            let msg = format!("Failed to resolve install.ps1 resource: {}", e);
            emit_failed(&app, &msg);
            msg
        })?;

    if !install_script.exists() {
        let msg = format!(
            "install.ps1 not found at resolved path: {}",
            install_script.display()
        );
        emit_failed(&app, &msg);
        return Err(msg);
    }
    emit_progress(
        &app,
        &format!("Using install script: {}", install_script.display()),
    );

    emit_progress(&app, "Running install.ps1...");
    let (stdout, stderr) = {
        let mut install = state.lock().map_err(|e| e.to_string())?;
        if install.child.is_some() {
            let msg = "Installation is already running.".to_string();
            emit_failed(&app, &msg);
            return Err(msg);
        }
        install.intentional_stop = false;

        let mut child = Command::new("powershell")
            .args([
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                &install_script.to_string_lossy(),
            ])
            .current_dir(&unsloth_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                let msg = format!("Failed to spawn install.ps1: {}", e);
                emit_failed(&app, &msg);
                msg
            })?;

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        install.child = Some(child);
        (stdout, stderr)
    };
    let threads = stream_child_output(&app, stdout, stderr);

    finalize_install(&app, &state, threads)
}
