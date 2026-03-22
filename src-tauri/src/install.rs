use log::{error, info, warn};
use std::io::BufRead;
use std::process::{Command, Stdio};
use tauri::{AppHandle, Emitter};
#[cfg(unix)]
use tauri::Manager;

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

/// Stream stdout and stderr from a child process, emitting install-progress events.
fn stream_child_output(app: &AppHandle, child: &mut std::process::Child) {
    // Read stdout in a separate thread
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let app_stdout = app.clone();
    let stdout_thread = stdout.map(|out| {
        std::thread::spawn(move || {
            let reader = std::io::BufReader::new(out);
            for line in reader.lines() {
                match line {
                    Ok(text) => {
                        let _ = app_stdout.emit("install-progress", &text);
                    }
                    Err(e) => {
                        warn!("[install] Error reading stdout: {}", e);
                        break;
                    }
                }
            }
        })
    });

    let app_stderr = app.clone();
    let stderr_thread = stderr.map(|err| {
        std::thread::spawn(move || {
            let reader = std::io::BufReader::new(err);
            for line in reader.lines() {
                match line {
                    Ok(text) => {
                        let _ = app_stderr.emit("install-progress", &text);
                    }
                    Err(e) => {
                        warn!("[install] Error reading stderr: {}", e);
                        break;
                    }
                }
            }
        })
    });

    // Wait for reader threads to finish
    if let Some(handle) = stdout_thread {
        let _ = handle.join();
    }
    if let Some(handle) = stderr_thread {
        let _ = handle.join();
    }
}

// ─── Unix implementation ────────────────────────────────────────────────────

#[cfg(unix)]
pub fn run_install(app: AppHandle) -> Result<(), String> {
    emit_progress(&app, "Starting installation...");

    // 1. Pre-flight: check required tools
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

    // 2. Create ~/.unsloth/ directory
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

    // 3. Resolve bundled install.sh from Tauri resources
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

    // 4. Spawn bash install.sh with cwd = ~/.unsloth/
    emit_progress(&app, "Running install.sh...");
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

    // 5. Stream stdout/stderr line-by-line
    stream_child_output(&app, &mut child);

    // 6. Check exit status
    let status = child.wait().map_err(|e| {
        let msg = format!("Failed to wait on install.sh: {}", e);
        emit_failed(&app, &msg);
        msg
    })?;

    if status.success() {
        emit_complete(&app);
        Ok(())
    } else {
        let code = status.code().unwrap_or(-1);
        let msg = format!("install.sh exited with code {}", code);
        emit_failed(&app, &msg);
        Err(msg)
    }
}

// ─── Windows implementation ─────────────────────────────────────────────────

#[cfg(windows)]
pub fn run_install(app: AppHandle) -> Result<(), String> {
    emit_progress(&app, "Starting installation on Windows...");

    let home = dirs::home_dir().ok_or_else(|| {
        let msg = "Could not determine home directory".to_string();
        emit_failed(&app, &msg);
        msg
    })?;

    // Pre-compute all paths as owned Strings to avoid lifetime issues with to_string_lossy()
    let unsloth_dir = home.join(".unsloth");
    let venv_dir = unsloth_dir.join("unsloth_studio");
    let python_path = venv_dir.join("Scripts").join("python.exe");
    let unsloth_exe = venv_dir.join("Scripts").join("unsloth.exe");

    let unsloth_dir_str = unsloth_dir.to_string_lossy().to_string();
    let venv_dir_str = venv_dir.to_string_lossy().to_string();
    let python_path_str = python_path.to_string_lossy().to_string();
    let unsloth_exe_str = unsloth_exe.to_string_lossy().to_string();

    // Create ~/.unsloth/ directory
    if !unsloth_dir.exists() {
        std::fs::create_dir_all(&unsloth_dir).map_err(|e| {
            let msg = format!("Failed to create {}: {}", unsloth_dir_str, e);
            emit_failed(&app, &msg);
            msg
        })?;
    }

    // Define the 4 install steps
    struct InstallStep {
        description: String,
        program: String,
        args: Vec<String>,
    }

    let steps = vec![
        InstallStep {
            description: "Installing uv package manager...".to_string(),
            program: "powershell".to_string(),
            args: vec![
                "-Command".to_string(),
                "irm https://astral.sh/uv/install.ps1 | iex".to_string(),
            ],
        },
        InstallStep {
            description: "Creating Python virtual environment...".to_string(),
            program: "uv".to_string(),
            args: vec![
                "venv".to_string(),
                venv_dir_str.clone(),
                "--python".to_string(),
                "3.13".to_string(),
            ],
        },
        InstallStep {
            description: "Installing Unsloth...".to_string(),
            program: "uv".to_string(),
            args: vec![
                "pip".to_string(),
                "install".to_string(),
                format!("--python={}", python_path_str),
                "unsloth".to_string(),
                "--torch-backend=auto".to_string(),
            ],
        },
        InstallStep {
            description: "Running Unsloth Studio setup...".to_string(),
            program: unsloth_exe_str.clone(),
            args: vec!["studio".to_string(), "setup".to_string()],
        },
    ];

    for (i, step) in steps.iter().enumerate() {
        emit_progress(&app, &format!("Step {}/{}: {}", i + 1, steps.len(), step.description));

        let mut child = Command::new(&step.program)
            .args(&step.args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                let msg = format!("Failed to spawn '{}': {}", step.program, e);
                emit_failed(&app, &msg);
                msg
            })?;

        stream_child_output(&app, &mut child);

        let status = child.wait().map_err(|e| {
            let msg = format!("Failed to wait on '{}': {}", step.program, e);
            emit_failed(&app, &msg);
            msg
        })?;

        if !status.success() {
            let code = status.code().unwrap_or(-1);
            let msg = format!(
                "Step {} ('{}') failed with exit code {}",
                i + 1,
                step.description,
                code
            );
            emit_failed(&app, &msg);
            return Err(msg);
        }

        emit_progress(&app, &format!("Step {}/{} completed successfully.", i + 1, steps.len()));
    }

    emit_complete(&app);
    Ok(())
}
