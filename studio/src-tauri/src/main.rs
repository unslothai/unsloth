#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app_layout;
mod commands;
mod desktop_auth;
mod desktop_backend_owner;
mod desktop_update_policy;
mod desktop_updater;
mod diagnostics;
mod install;
mod loopback_http;
mod native_backend_lease;
mod native_clipboard;
mod native_file_dialogs;
mod native_intents;
mod native_path_policy;
mod preflight;
mod process;
mod update;
mod windows_job;
mod x11_threads;

use log::{info, warn};
use process::new_backend_state;
use simplelog::{
    CombinedLogger, Config, LevelFilter, SharedLogger, TermLogger, TerminalMode, WriteLogger,
};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Once, OnceLock};
use tauri::menu::{MenuBuilder, MenuItemBuilder};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{Emitter, Manager};
use tauri_plugin_window_state::{AppHandleExt, StateFlags};

/// Serializes the exit paths that reap the backend: `request_quit` (tray "Quit" and,
/// outside macOS, the close button), the Unix signal listener, and `RunEvent::Exit`.
/// Exactly one runs cleanup; the others block, so the process never exits mid-reap.
static TERMINATION_CLEANUP: Once = Once::new();

const IN_APP_RELAUNCH_MARKER_FILE: &str = "in-app-relaunch-v1";

/// Resolved once, at setup, where the marker is consumed, so no later caller can flip the answer.
static LAUNCHED_HIDDEN: OnceLock<bool> = OnceLock::new();

/// Marks the next start as an in-app relaunch, whose inherited `--hidden` is not a login start.
#[tauri::command]
fn mark_in_app_relaunch(app: tauri::AppHandle) -> Result<(), String> {
    // Nothing to suppress without an inherited `--hidden`; the caller stops the restart on failure.
    if !argv_has_hidden_flag() {
        return Ok(());
    }
    let dir = in_app_relaunch_config_dir(&app)?;
    fs::create_dir_all(&dir).map_err(|error| {
        format!(
            "Failed to create app configuration directory {}: {error}",
            dir.display()
        )
    })?;
    write_in_app_relaunch_marker(&dir)
}

/// Undo the marker when its relaunch never happens, so an unrelated later start cannot consume it.
#[tauri::command]
fn clear_in_app_relaunch(app: tauri::AppHandle) -> Result<(), String> {
    let dir = in_app_relaunch_config_dir(&app)?;
    take_in_app_relaunch_marker(&dir);
    Ok(())
}

fn in_app_relaunch_config_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_config_dir()
        .map_err(|error| format!("Could not determine app configuration directory: {error}"))
}

fn in_app_relaunch_marker_path(config_dir: &Path) -> PathBuf {
    config_dir.join(IN_APP_RELAUNCH_MARKER_FILE)
}

fn write_in_app_relaunch_marker(config_dir: &Path) -> Result<(), String> {
    let path = in_app_relaunch_marker_path(config_dir);
    fs::write(&path, b"relaunching\n").map_err(|error| {
        format!(
            "Failed to write in-app relaunch marker {}: {error}",
            path.display()
        )
    })
}

/// Consumes the marker. A failed removal can only show a would-be-hidden start, never the reverse.
fn take_in_app_relaunch_marker(config_dir: &Path) -> bool {
    let path = in_app_relaunch_marker_path(config_dir);
    if !path.exists() {
        return false;
    }
    if let Err(error) = fs::remove_file(&path) {
        warn!(
            "Could not remove the in-app relaunch marker {}: {error}",
            path.display()
        );
    }
    true
}

/// args_os, not args: args panics on non-Unicode argv, e.g. argv[0] under a non-UTF-8 path.
fn argv_has_hidden_flag() -> bool {
    std::env::args_os().any(|arg| arg == *OsStr::new("--hidden"))
}

fn resolve_launched_hidden(app: &tauri::AppHandle) -> bool {
    // Consume unconditionally, so a marker left by a relaunch that never happened cannot outlive it.
    let relaunched = in_app_relaunch_config_dir(app)
        .map(|dir| take_in_app_relaunch_marker(&dir))
        .unwrap_or(false);
    argv_has_hidden_flag() && !relaunched
}

/// True for a login autostart start (window stays in the tray); false for an in-app relaunch,
/// which inherits `--hidden` without being one.
#[tauri::command]
fn was_launched_hidden(app: tauri::AppHandle) -> bool {
    *LAUNCHED_HIDDEN.get_or_init(|| resolve_launched_hidden(&app))
}

#[tauri::command]
fn reveal_main_window(app: tauri::AppHandle) {
    show_main_window(&app);
}

#[tauri::command]
fn get_launch_at_login(app: tauri::AppHandle) -> Result<bool, String> {
    use tauri_plugin_autostart::ManagerExt;
    // is_enabled only checks the entry file exists, so a DE-disabled entry would read as on.
    if cfg!(target_os = "linux") && linux_autostart_disabled(&app) {
        return Ok(false);
    }
    app.autolaunch()
        .is_enabled()
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn set_launch_at_login(app: tauri::AppHandle, enabled: bool) -> Result<bool, String> {
    use tauri_plugin_autostart::ManagerExt;
    let autolaunch = app.autolaunch();
    let result = if enabled {
        autolaunch.enable()
    } else {
        autolaunch.disable()
    };
    result.map_err(|error| error.to_string())?;
    if enabled {
        harden_autostart_entry(&app);
    }
    autolaunch
        .is_enabled()
        .map_err(|error| error.to_string())
}

fn harden_autostart_entry(app: &tauri::AppHandle) {
    if cfg!(target_os = "linux") {
        guard_linux_autostart_entry(app);
    }
    #[cfg(windows)]
    quote_windows_run_value(app);
    #[cfg(target_os = "macos")]
    rewrite_macos_launch_agent(app);
}

/// auto-launch writes the Run value unquoted, so a spaced path resolves to the wrong executable.
/// None when already quoted.
#[cfg_attr(not(windows), allow(dead_code))]
fn quoted_windows_run_command(value: &str) -> Option<String> {
    if value.starts_with('"') {
        return None;
    }
    let (path, args) = match value.strip_suffix(" --hidden") {
        Some(path) => (path, " --hidden"),
        None => (value, ""),
    };
    Some(format!("\"{path}\"{args}"))
}

#[cfg(windows)]
fn quote_windows_run_value(app: &tauri::AppHandle) {
    use winreg::enums::{HKEY_CURRENT_USER, KEY_QUERY_VALUE, KEY_SET_VALUE};
    use winreg::RegKey;

    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let Ok(run) = hkcu.open_subkey_with_flags(
        r"Software\Microsoft\Windows\CurrentVersion\Run",
        KEY_QUERY_VALUE | KEY_SET_VALUE,
    ) else {
        return;
    };
    let name = &app.package_info().name;
    let Ok(value) = run.get_value::<String, _>(name) else {
        return;
    };
    if let Some(quoted) = quoted_windows_run_command(&value) {
        let _ = run.set_value(name, &quoted);
    }
}

/// Exec is whitespace-delimited: quote a path holding reserved characters, escaping `"`, `` ` ``,
/// `$` and `\` inside, and double a literal `%` regardless, since it would start a field code.
fn exec_quoted(binary: &str) -> String {
    let binary = binary.replace('%', "%%");
    const RESERVED: &str = " \t\n\"'\\><~|&;$*?#()`";
    if !binary.chars().any(|c| RESERVED.contains(c)) {
        return binary;
    }
    let mut quoted = String::with_capacity(binary.len() + 2);
    quoted.push('"');
    for c in binary.chars() {
        // The string rule runs before the quoting rule, so the escaping backslash is itself
        // escaped; a single one is invalid and launchers drop the whole Exec.
        match c {
            '"' | '`' | '$' => {
                quoted.push_str("\\\\");
                quoted.push(c);
            }
            '\\' => quoted.push_str("\\\\\\\\"),
            _ => quoted.push(c),
        }
    }
    quoted.push('"');
    quoted
}

/// Quote auto-launch's unquoted Exec binary and add TryExec, which launchers skip when missing, so
/// a removed install goes inert without postrm touching user homes. TryExec stays unquoted.
fn hardened_autostart_entry(entry: &str) -> Option<String> {
    if entry.lines().any(|line| line.starts_with("TryExec=")) {
        return None;
    }
    let exec = entry.lines().find_map(|line| line.strip_prefix("Exec="))?;
    let (binary, args) = match exec.strip_suffix(" --hidden") {
        Some(binary) => (binary, " --hidden"),
        None => (exec, ""),
    };
    let hardened = entry
        .lines()
        .map(|line| {
            if line.starts_with("Exec=") {
                format!("Exec={}{}", exec_quoted(binary), args)
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    // TryExec skips the quoting rule but is still a string, so backslashes still escape.
    Some(format!(
        "{hardened}\nTryExec={}",
        binary.replace('\\', "\\\\")
    ))
}

fn linux_autostart_entry_path(app: &tauri::AppHandle) -> Option<std::path::PathBuf> {
    // auto-launch hardcodes ~/.config regardless of XDG_CONFIG_HOME; mirror it or we miss the file.
    Some(
        dirs::home_dir()?
            .join(".config")
            .join("autostart")
            .join(format!("{}.desktop", app.package_info().name)),
    )
}

/// DE startup UIs disable an entry via Hidden=true or X-GNOME-Autostart-enabled=false, not deletion.
fn autostart_entry_disabled(entry: &str) -> bool {
    entry.lines().any(|line| {
        matches!(
            line.trim(),
            "Hidden=true" | "X-GNOME-Autostart-enabled=false"
        )
    })
}

fn linux_autostart_disabled(app: &tauri::AppHandle) -> bool {
    linux_autostart_entry_path(app)
        .and_then(|path| fs::read_to_string(path).ok())
        .is_some_and(|entry| autostart_entry_disabled(&entry))
}

fn guard_linux_autostart_entry(app: &tauri::AppHandle) {
    let Some(path) = linux_autostart_entry_path(app) else {
        return;
    };
    let Ok(entry) = fs::read_to_string(&path) else {
        return;
    };
    if let Some(hardened) = hardened_autostart_entry(&entry) {
        let _ = fs::write(&path, hardened);
    }
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn xml_escaped(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// The plugin's template, but escaped: it interpolates the path unescaped, so a path with & or <
/// writes a malformed LaunchAgent.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn macos_launch_agent_plist(label: &str, binary: &str) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
         <plist version=\"1.0\">\n\
         <dict>\n  \
         <key>Label</key>\n  \
         <string>{}</string>\n  \
         <key>ProgramArguments</key>\n  \
         <array>\n    \
         <string>{}</string>\n    \
         <string>--hidden</string>\n  \
         </array>\n  \
         <key>RunAtLoad</key>\n  \
         <true/>\n\
         </dict>\n\
         </plist>",
        xml_escaped(label),
        xml_escaped(binary),
    )
}

#[cfg(target_os = "macos")]
fn rewrite_macos_launch_agent(app: &tauri::AppHandle) {
    let Some(home_dir) = dirs::home_dir() else {
        return;
    };
    let name = &app.package_info().name;
    let path = home_dir
        .join("Library")
        .join("LaunchAgents")
        .join(format!("{name}.plist"));
    if !path.exists() {
        return;
    }
    let Ok(binary) = std::env::current_exe() else {
        return;
    };
    let plist = macos_launch_agent_plist(name, &binary.display().to_string());
    let _ = fs::write(&path, plist);
}

/// A moved or re-downloaded AppImage leaves a stale path that `is_enabled` still accepts, so
/// repoint the entry at the current executable on every startup.
fn reconcile_autostart_entry(app: &tauri::AppHandle) {
    use tauri_plugin_autostart::ManagerExt;
    // A dev run would repoint the entry at target/debug.
    if cfg!(debug_assertions) {
        return;
    }
    // enable() would rewrite the file without the user's DE-set disabled marker.
    if cfg!(target_os = "linux") && linux_autostart_disabled(app) {
        return;
    }
    let autolaunch = app.autolaunch();
    if !autolaunch.is_enabled().unwrap_or(false) {
        return;
    }
    if autolaunch.enable().is_ok() {
        harden_autostart_entry(app);
    }
}

#[tauri::command]
fn has_saved_window_state(app: tauri::AppHandle) -> bool {
    let Ok(dir) = app.path().app_config_dir() else {
        return false;
    };
    dir.join(app.filename()).is_file()
}

fn setup_logging() {
    let mut loggers: Vec<Box<dyn SharedLogger>> = vec![];

    // Always log to stderr for development
    loggers.push(TermLogger::new(
        LevelFilter::Info,
        Config::default(),
        TerminalMode::Stderr,
        simplelog::ColorChoice::Auto,
    ));

    // Try to set up file logging to ~/.unsloth/studio/tauri.log
    if let Some(home) = dirs::home_dir() {
        let log_dir = home.join(".unsloth").join("studio");
        if fs::create_dir_all(&log_dir).is_ok() {
            let log_path = log_dir.join("tauri.log");
            let rotated_path = log_dir.join("tauri.log.1");
            let max_log_bytes = 5 * 1024 * 1024;
            if fs::metadata(&log_path)
                .map(|meta| meta.len() >= max_log_bytes)
                .unwrap_or(false)
            {
                let _ = fs::remove_file(&rotated_path);
                let _ = fs::rename(&log_path, &rotated_path);
            }
            if let Ok(file) = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_path)
            {
                loggers.push(WriteLogger::new(LevelFilter::Info, Config::default(), file));
            }
        }
    }

    if !loggers.is_empty() {
        let _ = CombinedLogger::init(loggers);
    }
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
fn setup_custom_titlebar(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let window = app.get_webview_window("main").ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound, "main window not found")
    })?;
    window.set_decorations(false)?;
    Ok(())
}

/// A Tauri quit never fires beforeunload, so the frontend mirrors quit protection here.
pub type TrainingActivityState = std::sync::Arc<std::sync::Mutex<bool>>;

fn new_training_activity_state() -> TrainingActivityState {
    std::sync::Arc::new(std::sync::Mutex::new(false))
}

#[tauri::command]
fn set_training_active(state: tauri::State<'_, TrainingActivityState>, active: bool) {
    if let Ok(mut running) = state.lock() {
        *running = active;
    }
}

/// Ask before quitting while training is starting or active (true to proceed).
/// request_quit only, as below.
fn confirm_quit_during_training(app: &tauri::AppHandle) -> bool {
    use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

    let Some(state) = app.try_state::<TrainingActivityState>() else {
        return true;
    };
    if !state.lock().map(|running| *running).unwrap_or(false) {
        return true;
    }
    app.dialog()
        .message(
            "Training is starting or still running. Quitting now can stop the \
             run and lose progress since the last checkpoint.",
        )
        .kind(MessageDialogKind::Warning)
        .title("Training in progress")
        .buttons(MessageDialogButtons::OkCancelCustom(
            "Quit anyway".to_string(),
            "Keep training".to_string(),
        ))
        .blocking_show()
}

/// Renderer-only activity, mirrored here for the same reason: Hub downloads, which the
/// backend runs but only the frontend tracks, and the Tauri shell self-update, which
/// `update::is_update_running` stops covering once `downloadAndInstall` takes over.
#[derive(Default)]
pub struct RendererActivity {
    pub downloads: bool,
    pub shell_update: bool,
}

pub type RendererActivityState = std::sync::Arc<std::sync::Mutex<RendererActivity>>;

fn new_renderer_activity_state() -> RendererActivityState {
    std::sync::Arc::new(std::sync::Mutex::new(RendererActivity::default()))
}

/// The command body without the `tauri::State` wrapper, so tests can drive it directly.
fn apply_renderer_activity(state: &RendererActivityState, kind: &str, active: bool) {
    if let Ok(mut activity) = state.lock() {
        match kind {
            "downloads" => activity.downloads = active,
            "shell_update" => activity.shell_update = active,
            // An unknown kind is a renderer/Rust mismatch, never a reason to flip a flag.
            _ => {}
        }
    }
}

#[tauri::command]
fn set_renderer_activity(state: tauri::State<'_, RendererActivityState>, kind: &str, active: bool) {
    apply_renderer_activity(state.inner(), kind, active);
}

/// Ask before quitting mid shell update (true to proceed). request_quit only, as below.
fn confirm_quit_during_shell_update(app: &tauri::AppHandle) -> bool {
    use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

    let Some(state) = app.try_state::<RendererActivityState>() else {
        return true;
    };
    if !state
        .lock()
        .map(|activity| activity.shell_update)
        .unwrap_or(false)
    {
        return true;
    }
    app.dialog()
        .message(
            "The app update is still installing. Quitting now interrupts the \
             installer part-way and can leave the app half-updated.",
        )
        .kind(MessageDialogKind::Warning)
        .title("Update in progress")
        .buttons(MessageDialogButtons::OkCancelCustom(
            "Quit anyway".to_string(),
            "Keep updating".to_string(),
        ))
        .blocking_show()
}

/// Ask before quitting mid-download (true to proceed). request_quit only, as below.
fn confirm_quit_during_downloads(app: &tauri::AppHandle) -> bool {
    use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

    let Some(state) = app.try_state::<RendererActivityState>() else {
        return true;
    };
    if !state
        .lock()
        .map(|activity| activity.downloads)
        .unwrap_or(false)
    {
        return true;
    }
    app.dialog()
        .message(
            "Downloads are still running. Quitting now stops the backend and \
             cancels the downloads in progress.",
        )
        .kind(MessageDialogKind::Warning)
        .title("Downloads in progress")
        .buttons(MessageDialogButtons::OkCancelCustom(
            "Quit anyway".to_string(),
            "Keep downloading".to_string(),
        ))
        .blocking_show()
}

/// Ask before quitting mid-install (true to proceed): `cleanup_child_processes` SIGTERMs the
/// installer, leaving a venv that looks healthy but cannot start. request_quit only, since
/// RunEvent::Exit must never block on a dialog nobody can answer.
fn confirm_quit_during_install(app: &tauri::AppHandle) -> bool {
    use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

    let Some(install_state) = app.try_state::<install::InstallState>() else {
        return true;
    };
    if !install::is_install_running(&install_state) {
        return true;
    }
    app.dialog()
        .message(
            "Unsloth is still installing. Quitting now stops it part-way and \
             leaves the installation incomplete, so it will need to be repaired before \
             it can start.",
        )
        .kind(MessageDialogKind::Warning)
        .title("Installation in progress")
        .buttons(MessageDialogButtons::OkCancelCustom(
            "Quit anyway".to_string(),
            "Keep installing".to_string(),
        ))
        .blocking_show()
}

/// Ask before quitting mid-update (true to proceed): the update is killed part-way
/// through installing dependencies, which leaves the venv unusable until it is repaired.
fn confirm_quit_during_update(app: &tauri::AppHandle) -> bool {
    use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

    let Some(update_state) = app.try_state::<update::UpdateState>() else {
        return true;
    };
    if !update::is_update_running(&update_state) {
        return true;
    }
    app.dialog()
        .message(
            "Unsloth is still updating. Quitting now stops it part-way and \
             leaves the installation incomplete, so it will need to be repaired before \
             it can start.",
        )
        .kind(MessageDialogKind::Warning)
        .title("Update in progress")
        .buttons(MessageDialogButtons::OkCancelCustom(
            "Quit anyway".to_string(),
            "Keep updating".to_string(),
        ))
        .blocking_show()
}

fn cleanup_child_processes(app: &tauri::AppHandle) {
    // `call_once_force` rather than `call_once`: a panicking cleanup poisons the `Once`
    // instead of deadlocking the exit paths waiting on it, and the next caller retries
    // it rather than exiting on a backend that was never reaped.
    TERMINATION_CLEANUP.call_once_force(|state| {
        if state.is_poisoned() {
            warn!("Previous termination cleanup panicked, retrying it");
        }

        let diagnostics_state = app
            .try_state::<diagnostics::DiagnosticsState>()
            .map(|state| state.inner().clone());
        if let Some(install_state) = app.try_state::<install::InstallState>() {
            if let Some(diagnostics) = diagnostics_state.as_ref() {
                install::record_install_intentional_stop(&install_state, diagnostics);
            }
            let _ = install::stop_install(&install_state);
        }
        if let Some(update_state) = app.try_state::<update::UpdateState>() {
            if let Some(diagnostics) = diagnostics_state.as_ref() {
                update::record_update_intentional_stop(&update_state, diagnostics);
            }
            let _ = update::stop_update(&update_state);
        }
        if let Some(backend_state) = app.try_state::<process::BackendState>() {
            let shutdown = app
                .try_state::<process::ShutdownFlag>()
                .expect("ShutdownFlag must be managed");
            let _ = process::stop_backend(&backend_state, &shutdown, diagnostics_state.as_ref());
        }
    });
}

/// The backend is spawned as its own process-group leader, so nothing reaps it when the
/// desktop app is terminated by a signal (logout, `kill`, a shell Ctrl-C). Tauri installs
/// no handler for those, so without this the app would die and leave the backend running.
/// Windows gets the same guarantee from the kill-on-close job object in `windows_job`.
#[cfg(unix)]
fn setup_unix_termination_signals(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    use signal_hook::consts::signal::{SIGHUP, SIGINT, SIGQUIT, SIGTERM};
    use signal_hook::iterator::Signals;

    let mut signals = Signals::new([SIGHUP, SIGINT, SIGQUIT, SIGTERM])?;
    let app_handle = app.handle().clone();
    std::thread::Builder::new()
        .name("desktop-termination-signal".to_string())
        .spawn(move || {
            // Terminating is not conditional on cleanup succeeding. If cleanup panicked
            // and took the exit with it, the app would keep running after SIGTERM, the
            // session manager would SIGKILL it once its timeout expired, and the backend
            // would be orphaned: exactly the failure this listener exists to prevent.
            let cleanup_and_exit = |app: &tauri::AppHandle, exit_code: i32| {
                let cleanup = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    cleanup_child_processes(app)
                }));
                if cleanup.is_err() {
                    warn!("Termination cleanup panicked, exiting anyway");
                }
                app.exit(exit_code);
            };

            let mut cleanup_started = false;
            for signal in signals.forever() {
                let name = signal_hook::low_level::signal_name(signal).unwrap_or("unknown signal");
                let exit_code = 128 + signal;
                if cleanup_started {
                    // Cleanup blocks for as long as the backend takes to die. A repeat
                    // signal is the user asking to stop waiting, so leave straight away
                    // instead of swallowing it and forcing them to reach for SIGKILL.
                    warn!("Received {name} ({signal}) while cleaning up, exiting immediately");
                    std::process::exit(exit_code);
                }
                cleanup_started = true;
                info!("Received Unix termination signal {name} ({signal})");

                // Clean up on its own thread so this one stays in `signals.forever()` and
                // can still observe a repeat signal. If that thread cannot be spawned we
                // fall back to cleaning up here, which gives up the repeat-signal escape
                // hatch until cleanup finishes.
                let cleanup_handle = app_handle.clone();
                let cleanup = std::thread::Builder::new()
                    .name("desktop-termination-cleanup".to_string())
                    .spawn(move || cleanup_and_exit(&cleanup_handle, exit_code));
                if let Err(error) = cleanup {
                    warn!("Could not spawn termination cleanup thread: {error}");
                    cleanup_and_exit(&app_handle, exit_code);
                }
            }
        })?;
    Ok(())
}

fn show_main_window(app: &tauri::AppHandle) {
    // Hidden login starts run as an accessory app (no Dock icon); restore the regular policy.
    #[cfg(target_os = "macos")]
    let _ = app.set_activation_policy(tauri::ActivationPolicy::Regular);
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

/// One quit confirmation at a time: the dialogs are unparented, so repeat close
/// presses would stack another one.
static QUIT_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

/// Clears the flag on cancel or panic, so a quit cannot leave the close button inert.
struct QuitGuard;

impl Drop for QuitGuard {
    fn drop(&mut self) {
        QUIT_IN_PROGRESS.store(false, Ordering::SeqCst);
    }
}

fn begin_quit() -> Option<QuitGuard> {
    (!QUIT_IN_PROGRESS.swap(true, Ordering::SeqCst)).then_some(QuitGuard)
}

/// Confirm, reap the backend, then exit. Off the caller's thread because the confirmations
/// block, and never exit first: that would orphan the backend tree.
fn request_quit(app: &tauri::AppHandle) {
    let Some(guard) = begin_quit() else {
        info!("Quit already awaiting confirmation, ignoring the repeat request");
        return;
    };
    let app = app.clone();
    // The guard moves into the closure, so a failed spawn drops it and the next request retries.
    let spawned = std::thread::Builder::new()
        .name("request-quit".to_string())
        .spawn(move || {
            let _guard = guard;
            if !confirm_quit_during_install(&app) {
                return;
            }
            if !confirm_quit_during_update(&app) {
                return;
            }
            if !confirm_quit_during_shell_update(&app) {
                return;
            }
            if !confirm_quit_during_training(&app) {
                return;
            }
            if !confirm_quit_during_downloads(&app) {
                return;
            }
            cleanup_child_processes(&app);
            app.exit(0);
        });
    if let Err(error) = spawned {
        warn!("Could not spawn the quit thread: {error}");
    }
}

fn setup_tray(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let open = MenuItemBuilder::with_id("open", "Open Unsloth").build(app)?;
    let toggle = MenuItemBuilder::with_id("toggle", "Start/Stop Server").build(app)?;
    let quit = MenuItemBuilder::with_id("quit", "Quit").build(app)?;
    let menu = MenuBuilder::new(app)
        .items(&[&open, &toggle, &quit])
        .build()?;

    TrayIconBuilder::new()
        .menu(&menu)
        .tooltip("Unsloth")
        .icon(app.default_window_icon().unwrap().clone())
        .on_menu_event(move |app, event| match event.id().as_ref() {
            "open" => show_main_window(app),
            "toggle" => {
                let _ = app.emit("tray-toggle-server", ());
            }
            "quit" => request_quit(app),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                show_main_window(tray.app_handle());
            }
        })
        .build(app)?;

    Ok(())
}

// Root shared by the WebView profile and the version stamp below.
fn webview_profile_root(bundle_id: &str) -> Option<std::path::PathBuf> {
    use std::path::PathBuf;
    let from_env = |key: &str| {
        std::env::var(key)
            .ok()
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    #[cfg(target_os = "windows")]
    {
        from_env("LOCALAPPDATA").map(|d| d.join(bundle_id))
    }
    #[cfg(target_os = "macos")]
    {
        from_env("HOME").map(|h| h.join("Library/Application Support").join(bundle_id))
    }
    #[cfg(target_os = "linux")]
    {
        // Tauri points the WebView at LocalData/<bid> and wry keys the WebKitGTK
        // base-cache dir to it. dirs, which resolves LocalData, drops a relative
        // XDG_DATA_HOME as the XDG spec requires; following one would miss the
        // real cache and delete under $PWD.
        from_env("XDG_DATA_HOME")
            .filter(|d| d.is_absolute())
            .or_else(|| from_env("HOME").map(|h| h.join(".local/share")))
            .map(|d| d.join(bundle_id))
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        let _ = (bundle_id, from_env);
        None
    }
}

// Clear WebView caches after an update, before the webview initializes: the
// in-app update runs setup.sh/setup.ps1 while the old WebView still holds these
// files, so its clear fails and a relaunch can serve the previous frontend.
// Version-stamped, because this runs before the single-instance plugin: ungated,
// a duplicate launch would delete a live instance's profile and every ordinary
// launch would discard the Windows compiled-JS cache for nothing.
// Cache-only; LocalStorage, IndexedDB, cookies and app data are kept. Mirrors
// setup.sh _clear_webview_caches / setup.ps1. The returned lock must be held for
// the rest of the process: it is what stops a duplicate launch from clearing the
// profile this instance is already using.
#[must_use]
fn clear_webview_caches(bundle_id: &str, version: &str) -> Option<fs::File> {
    let root = webview_profile_root(bundle_id)?;
    // Claim the profile BEFORE reading the stamp, and keep the lock even when it
    // matches: a stamped launch holding nothing would let a newer executable started
    // alongside it delete this instance's live profile. The OS drops the lock if we
    // crash, so a dead process never blocks a clear.
    let _ = fs::create_dir_all(&root);
    let lock = fs::File::create(root.join(".webview-cache-lock")).ok()?;
    lock.try_lock().ok()?;
    let stamp = root.join(".webview-cache-cleared");
    if fs::read_to_string(&stamp).is_ok_and(|v| v.trim() == version) {
        return Some(lock);
    }

    let mut paths: Vec<std::path::PathBuf> = Vec::new();
    #[cfg(target_os = "windows")]
    {
        let profile = root.join("EBWebView").join("Default");
        for sub in ["Cache", "Code Cache", "GPUCache", "Service Worker"] {
            paths.push(profile.join(sub));
        }
    }
    #[cfg(target_os = "macos")]
    if let Ok(home) = std::env::var("HOME") {
        // WKWebView keeps every cache-typed store under Library/Caches/<bid>;
        // Library/WebKit/<bid> is user storage and is left alone.
        paths.push(std::path::PathBuf::from(home).join("Library/Caches").join(bundle_id));
    }
    #[cfg(target_os = "linux")]
    for sub in ["WebKitCache", "CacheStorage", "serviceworkers"] {
        paths.push(root.join(sub));
    }

    let mut cleared = true;
    for p in &paths {
        // Absent is normal; anything else is the silent failure that shows up
        // as a stale frontend, so log it.
        if let Err(e) = fs::remove_dir_all(p) {
            if e.kind() != std::io::ErrorKind::NotFound {
                warn!("could not clear WebView cache {}: {e}", p.display());
                cleared = false;
            }
        }
    }
    // Stamping a partial clear (an update can still hold a cache file open)
    // would make every later launch skip the retry and keep the stale cache.
    if cleared {
        let _ = fs::write(&stamp, version);
    }
    Some(lock)
}

fn main() {
    // Must precede any Xlib call: GTK3 never calls XInitThreads and this
    // process drives X from several threads. See x11_threads for the crash.
    x11_threads::init_x11_threads();

    // Fix PATH for GUI apps (macOS .app bundles, Linux AppImage, Windows)
    // GUI apps don't inherit shell dotfile PATH — this spawns the user's
    // login shell to source .zshrc/.bashrc/.profile and sets PATH properly.
    let _ = fix_path_env::fix();

    setup_logging();
    info!("Unsloth desktop app starting");
    windows_job::initialize();

    // Must run before the Builder: the config-defined window (and its
    // WebView, which locks these files) exists by the time setup hooks run.
    let context = tauri::generate_context!();
    // Bound, not dropped: the lock has to outlive the clear (see the function).
    let _webview_profile_lock = clear_webview_caches(
        &context.config().identifier,
        context.package_info().version.to_string().as_str(),
    );

    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            show_main_window(app);
        }))
        .plugin(tauri_plugin_deep_link::init())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            Some(vec!["--hidden"]),
        ))
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(
            tauri_plugin_window_state::Builder::new()
                .with_state_flags(StateFlags::SIZE | StateFlags::POSITION | StateFlags::MAXIMIZED)
                .skip_initial_state("main")
                .build(),
        )
        .manage(diagnostics::new_diagnostics_state())
        .manage(install::new_install_state())
        .manage(new_training_activity_state())
        .manage(new_renderer_activity_state())
        .manage(native_intents::new_native_intake_state())
        .manage(new_backend_state())
        .manage(process::new_shutdown_flag())
        .manage(update::new_update_state())
        .invoke_handler(tauri::generate_handler![
            set_training_active,
            set_renderer_activity,
            app_layout::has_initialized_app_window_layout,
            app_layout::mark_app_window_layout_initialized,
            app_layout::reset_app_window_layout_initialized,
            commands::check_install_status,
            commands::desktop_preflight,
            commands::start_install,
            commands::start_server,
            commands::start_managed_server,
            commands::stop_server,
            commands::check_health,
            commands::get_server_logs,
            commands::open_logs_dir,
            commands::open_models_dir,
            commands::start_backend_update,
            commands::start_managed_repair,
            commands::cancel_pending_elevation,
            commands::install_system_packages,
            desktop_auth::desktop_auth,
            desktop_update_policy::check_desktop_manual_update,
            desktop_update_policy::desktop_update_policy,
            desktop_updater::check_desktop_update,
            diagnostics::collect_support_diagnostics,
            native_clipboard::read_native_clipboard_files,
            native_clipboard::read_native_clipboard_png,
            native_file_dialogs::save_native_file,
            native_file_dialogs::pick_native_chat_import,
            native_file_dialogs::pick_native_training_config,
            native_intents::drain_native_intents,
            native_intents::register_native_model_path,
            native_intents::register_native_attachment_path,
            native_intents::register_native_dataset_path,
            native_intents::pick_native_model,
            native_intents::pick_hugging_face_cache_dir,
            native_intents::consume_native_path_token,
            native_intents::register_artifact_path,
            native_intents::reveal_path_token,
            native_intents::open_path_token,
            has_saved_window_state,
            was_launched_hidden,
            mark_in_app_relaunch,
            clear_in_app_relaunch,
            reveal_main_window,
            get_launch_at_login,
            set_launch_at_login,
        ])
        .setup(|app| {
            // Resolve here, before any window path can ask: this consumes the relaunch marker.
            let launched_hidden = was_launched_hidden(app.handle().clone());
            #[cfg(not(target_os = "macos"))]
            let _ = launched_hidden;
            #[cfg(target_os = "macos")]
            if launched_hidden {
                app.set_activation_policy(tauri::ActivationPolicy::Accessory);
            }
            reconcile_autostart_entry(app.handle());
            #[cfg(any(target_os = "linux", all(debug_assertions, windows)))]
            {
                use tauri_plugin_deep_link::DeepLinkExt;
                if let Err(error) = app.deep_link().register_all() {
                    warn!("Failed to register deep-link handlers: {error}");
                }
            }
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            setup_custom_titlebar(app)?;
            setup_tray(app)?;
            #[cfg(unix)]
            setup_unix_termination_signals(app)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            // Record real drops here, in Rust, so the renderer can only register paths the
            // OS actually handed to the native intake commands.
            if let tauri::WindowEvent::DragDrop(tauri::DragDropEvent::Drop { paths, .. }) = event {
                window
                    .state::<native_intents::NativeIntakeState>()
                    .note_dropped_paths(paths);
            }
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                // Never close directly: the only window, so closing exits before the reap.
                api.prevent_close();
                if cfg!(target_os = "macos") {
                    // Closing a window leaves the app in the Dock; Reopen restores it.
                    let _ = window.hide();
                } else {
                    // Elsewhere the close button means quit, not hiding what the user
                    // believes they just closed.
                    request_quit(window.app_handle());
                }
            }
        })
        .build(context)
        .expect("error while building tauri application")
        .run(|app, event| match event {
            #[cfg(target_os = "macos")]
            tauri::RunEvent::Reopen {
                has_visible_windows: false,
                ..
            } => show_main_window(app),
            tauri::RunEvent::Exit => {
                // Safety net for framework-driven exits. When another path already owns
                // cleanup, this blocks the main event-loop thread until that path is
                // done: worst case roughly 15s, waiting on the graceful-then-force stop
                // of the installer (5s), the updater (5s) and the backend (5s).
                cleanup_child_processes(app);
            }
            _ => {}
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// args() panics on a non-Unicode argv[0], so the flag scan has to run over OsStr.
    #[test]
    fn hidden_flag_scan_tolerates_non_unicode_args() {
        use std::ffi::OsString;
        #[cfg(unix)]
        use std::os::unix::ffi::OsStringExt;

        #[cfg(unix)]
        let args: Vec<OsString> = vec![
            OsString::from_vec(b"/opt/\xff\xfe/Unsloth".to_vec()),
            OsString::from("--hidden"),
        ];
        #[cfg(not(unix))]
        let args: Vec<OsString> = vec![OsString::from("Unsloth.exe"), OsString::from("--hidden")];

        assert!(args.iter().any(|arg| arg == OsStr::new("--hidden")));
        assert!(!args[..1].iter().any(|arg| arg == OsStr::new("--hidden")));
    }

    #[test]
    fn relaunch_marker_is_consumed_by_the_next_start() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!take_in_app_relaunch_marker(dir.path()));

        write_in_app_relaunch_marker(dir.path()).unwrap();
        assert!(take_in_app_relaunch_marker(dir.path()));
        assert!(!take_in_app_relaunch_marker(dir.path()));
        assert!(!in_app_relaunch_marker_path(dir.path()).exists());
    }

    #[cfg(target_os = "linux")]
    const BID: &str = "ai.unsloth.studio";

    // Only XDG_DATA_HOME is swapped, and nothing else in the crate reads it, so the
    // tests running beside these stay clear. HOME is left alone for the same reason:
    // `dirs::home_dir` is all over this crate.
    #[cfg(target_os = "linux")]
    fn with_xdg_data_home<T>(value: &str, f: impl FnOnce() -> T) -> T {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved = std::env::var_os("XDG_DATA_HOME");
        std::env::set_var("XDG_DATA_HOME", value);
        let out = f();
        match saved {
            Some(v) => std::env::set_var("XDG_DATA_HOME", v),
            None => std::env::remove_var("XDG_DATA_HOME"),
        }
        out
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn a_relative_xdg_data_home_resolves_like_tauri_does() {
        // dirs, which resolves the LocalData dir Tauri hands the WebView, drops a
        // relative XDG_DATA_HOME; following one would leave the real cache in
        // place and rm -rf under the working directory.
        let (got, expected) = with_xdg_data_home("reldata", || {
            (
                webview_profile_root(BID),
                dirs::data_local_dir().map(|d| d.join(BID)),
            )
        });
        assert_eq!(got, expected, "diverged from the resolver Tauri uses");
        assert!(
            got.is_none_or(|p| p.is_absolute()),
            "resolved a deletion target relative to the working directory"
        );

        // An absolute override is still honoured.
        let dir = tempfile::tempdir().unwrap();
        let got = with_xdg_data_home(dir.path().to_str().unwrap(), || webview_profile_root(BID));
        assert_eq!(got, Some(dir.path().join(BID)));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn a_partial_clear_is_not_stamped_and_retries() {
        let dir = tempfile::tempdir().unwrap();
        let xdg = dir.path().to_str().unwrap();
        let root = dir.path().join(BID);
        fs::create_dir_all(root.join("CacheStorage")).unwrap();
        // A plain file where a cache dir belongs fails remove_dir_all with something
        // other than NotFound, exactly as a locked directory does.
        fs::write(root.join("WebKitCache"), b"still open").unwrap();

        let lock = with_xdg_data_home(xdg, || clear_webview_caches(BID, "1.0.0"));
        assert!(!root.join("CacheStorage").exists(), "the deletable cache stayed");
        assert!(
            !root.join(".webview-cache-cleared").exists(),
            "stamping a partial clear makes every later launch skip the retry"
        );
        drop(lock);

        // The next launch, with the obstruction gone, clears and stamps.
        fs::remove_file(root.join("WebKitCache")).unwrap();
        fs::create_dir_all(root.join("WebKitCache")).unwrap();
        let lock = with_xdg_data_home(xdg, || clear_webview_caches(BID, "1.0.0"));
        assert!(!root.join("WebKitCache").exists(), "the retry did not run");
        assert!(root.join(".webview-cache-cleared").exists(), "the retry did not stamp");
        drop(lock);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn a_live_instance_lock_excludes_a_duplicate_launch() {
        let dir = tempfile::tempdir().unwrap();
        let xdg = dir.path().to_str().unwrap();
        let root = dir.path().join(BID);
        fs::create_dir_all(root.join("WebKitCache")).unwrap();

        // The first launch clears and holds the lock, as main() does.
        let live = with_xdg_data_home(xdg, || clear_webview_caches(BID, "1.0.0"));
        assert!(live.is_some(), "the clearing instance must get the lock");
        assert!(!root.join("WebKitCache").exists(), "the first launch did not clear");

        // Single-instance arbitration is still several steps away, so a second
        // launch must leave the first one's live profile alone.
        fs::create_dir_all(root.join("WebKitCache")).unwrap();
        let second = with_xdg_data_home(xdg, || clear_webview_caches(BID, "1.0.1"));
        assert!(second.is_none(), "a duplicate launch took the lock");
        assert!(root.join("WebKitCache").exists(), "deleted a live instance's cache");

        // Exit or crash drops the lock, so the next launch clears.
        drop(live);

        // A launch whose stamp already matches must still TAKE the lock, or a
        // newer executable started alongside it could delete the live profile.
        fs::create_dir_all(root.join("WebKitCache")).unwrap();
        let stamped = with_xdg_data_home(xdg, || clear_webview_caches(BID, "1.0.0"));
        assert!(stamped.is_some(), "a stamped launch dropped the profile lock");
        assert!(root.join("WebKitCache").exists(), "a stamped launch cleared anyway");
        let racer = with_xdg_data_home(xdg, || clear_webview_caches(BID, "1.0.1"));
        assert!(racer.is_none(), "a newer launch took the lock from a stamped instance");
        assert!(root.join("WebKitCache").exists(), "deleted a stamped instance's cache");
        drop(stamped);

        let next = with_xdg_data_home(xdg, || clear_webview_caches(BID, "1.0.1"));
        assert!(!root.join("WebKitCache").exists(), "a released lock still blocked the clear");
        drop(next);
    }

    // One test, not three: `QUIT_IN_PROGRESS` is process-global and cargo tests run in parallel.
    #[test]
    fn quit_guard_admits_one_quit_and_always_re_arms() {
        assert!(!QUIT_IN_PROGRESS.load(Ordering::SeqCst));

        {
            let _first = begin_quit().expect("the first quit must be admitted");
            assert!(
                begin_quit().is_none(),
                "a repeat close must not stack a second confirmation"
            );
        }

        // Cancelling drops the guard, which re-arms the close button.
        {
            let _second = begin_quit().expect("cancelling must re-arm the close button");
        }

        let unwound = std::panic::catch_unwind(|| {
            let _guard = begin_quit().expect("admitted");
            panic!("quit thread unwound");
        });
        assert!(unwound.is_err());
        assert!(
            begin_quit().is_some(),
            "a panicking quit must release the guard"
        );
    }

    #[test]
    fn autostart_hardening_appends_the_exec_binary_without_args() {
        let entry = "[Desktop Entry]\nType=Application\nExec=/usr/bin/unsloth-studio --hidden\nTerminal=false";
        let hardened = hardened_autostart_entry(entry).expect("guard must be added");
        assert!(hardened.starts_with(entry));
        assert!(hardened.ends_with("\nTryExec=/usr/bin/unsloth-studio"));
    }

    #[test]
    fn autostart_hardening_quotes_a_binary_path_with_spaces() {
        let entry = "[Desktop Entry]\nExec=/home/n/My Apps/Unsloth.AppImage --hidden\nTerminal=false";
        let hardened = hardened_autostart_entry(entry).expect("guard must be added");
        assert!(hardened.contains("Exec=\"/home/n/My Apps/Unsloth.AppImage\" --hidden\n"));
        // TryExec is a plain string field, so the path stays unquoted.
        assert!(hardened.ends_with("\nTryExec=/home/n/My Apps/Unsloth.AppImage"));
    }

    #[test]
    fn autostart_hardening_escapes_exec_reserved_characters() {
        let entry = "[Desktop Entry]\nExec=/opt/a\"b$c/app --hidden";
        let hardened = hardened_autostart_entry(entry).expect("guard must be added");
        // Doubled: the string escape rule is undone before the quoting rule.
        assert!(hardened.contains(r#"Exec="/opt/a\\"b\\$c/app" --hidden"#));
    }

    /// A single escaping backslash is an invalid escape, so launchers discard the whole Exec.
    #[test]
    fn autostart_hardening_doubles_the_escaping_backslash() {
        let entry = "[Desktop Entry]\nExec=/opt/a b`c$d/app --hidden";
        let hardened = hardened_autostart_entry(entry).expect("guard must be added");
        assert!(hardened.contains(r#"Exec="/opt/a b\\`c\\$d/app" --hidden"#));
    }

    #[test]
    fn autostart_hardening_escapes_a_literal_backslash_in_both_fields() {
        let entry = "[Desktop Entry]\nExec=/opt/a b\\c/app --hidden";
        let hardened = hardened_autostart_entry(entry).expect("guard must be added");
        // Four in Exec (string rule then quoting rule), two in the plain TryExec.
        assert!(hardened.contains(r#"Exec="/opt/a b\\\\c/app" --hidden"#));
        assert!(hardened.ends_with(r"TryExec=/opt/a b\\c/app"));
    }

    #[test]
    fn autostart_hardening_is_idempotent() {
        let entry = "[Desktop Entry]\nExec=/a --hidden\nTryExec=/a";
        assert!(hardened_autostart_entry(entry).is_none());
    }

    #[test]
    fn autostart_hardening_needs_an_exec_line() {
        assert!(hardened_autostart_entry("[Desktop Entry]\nType=Application").is_none());
    }

    #[test]
    fn exec_quoting_escapes_percent_field_codes() {
        let entry = "[Desktop Entry]\nExec=/home/n/Unsloth%20Studio.AppImage --hidden";
        let hardened = hardened_autostart_entry(entry).expect("guard must be added");
        assert!(hardened.contains("Exec=/home/n/Unsloth%%20Studio.AppImage --hidden\n"));
        // TryExec is a plain string field, so the raw path stays.
        assert!(hardened.ends_with("\nTryExec=/home/n/Unsloth%20Studio.AppImage"));
    }

    #[test]
    fn autostart_disabled_markers_are_detected() {
        assert!(autostart_entry_disabled("[Desktop Entry]\nHidden=true"));
        assert!(autostart_entry_disabled(
            "[Desktop Entry]\nX-GNOME-Autostart-enabled=false"
        ));
        assert!(!autostart_entry_disabled("[Desktop Entry]\nExec=/a --hidden"));
    }

    #[test]
    fn macos_plist_escapes_xml_metacharacters() {
        let plist = macos_launch_agent_plist(
            "Unsloth",
            "/Applications/AI & ML/Unsloth.app/Contents/MacOS/unsloth-studio",
        );
        assert!(plist.contains(
            "<string>/Applications/AI &amp; ML/Unsloth.app/Contents/MacOS/unsloth-studio</string>"
        ));
        assert!(plist.contains("<string>--hidden</string>"));
        assert!(plist.contains("<key>RunAtLoad</key>"));
    }

    #[test]
    fn windows_run_command_quotes_a_spaced_path() {
        let quoted =
            quoted_windows_run_command(r"C:\Users\Jane Doe\AppData\Local\Unsloth\Unsloth.exe --hidden");
        assert_eq!(
            quoted.as_deref(),
            Some(r#""C:\Users\Jane Doe\AppData\Local\Unsloth\Unsloth.exe" --hidden"#),
        );
    }

    #[test]
    fn windows_run_command_leaves_quoted_values_alone() {
        assert!(quoted_windows_run_command(r#""C:\Unsloth\Unsloth.exe" --hidden"#).is_none());
    }

    #[test]
    fn windows_run_command_quotes_a_bare_path() {
        assert_eq!(
            quoted_windows_run_command(r"C:\Unsloth\Unsloth.exe").as_deref(),
            Some(r#""C:\Unsloth\Unsloth.exe""#),
        );
    }

    #[test]
    fn autostart_hardening_keeps_foreign_args_untouched() {
        let entry = "[Desktop Entry]\nExec=/plain/app";
        let hardened = hardened_autostart_entry(entry).expect("guard must be added");
        assert!(hardened.contains("Exec=/plain/app\n"));
        assert!(hardened.ends_with("\nTryExec=/plain/app"));
    }

    fn renderer_activity(state: &RendererActivityState) -> (bool, bool) {
        let activity = state.lock().expect("the activity mutex must not be poisoned");
        (activity.downloads, activity.shell_update)
    }

    #[test]
    fn renderer_activity_starts_clear_and_round_trips_each_kind() {
        let state = new_renderer_activity_state();
        assert_eq!(renderer_activity(&state), (false, false));

        apply_renderer_activity(&state, "downloads", true);
        assert_eq!(renderer_activity(&state), (true, false));
        apply_renderer_activity(&state, "downloads", false);
        assert_eq!(renderer_activity(&state), (false, false));

        apply_renderer_activity(&state, "shell_update", true);
        assert_eq!(renderer_activity(&state), (false, true));
        apply_renderer_activity(&state, "shell_update", false);
        assert_eq!(renderer_activity(&state), (false, false));
    }

    #[test]
    fn renderer_activity_kinds_are_independent() {
        let state = new_renderer_activity_state();

        apply_renderer_activity(&state, "downloads", true);
        apply_renderer_activity(&state, "shell_update", true);
        assert_eq!(renderer_activity(&state), (true, true));

        // Downloads finishing must not clear an update that is still installing.
        apply_renderer_activity(&state, "downloads", false);
        assert_eq!(renderer_activity(&state), (false, true));
    }

    #[test]
    fn renderer_activity_ignores_an_unknown_kind() {
        let state = new_renderer_activity_state();
        apply_renderer_activity(&state, "shell_update", true);

        // A renderer/Rust mismatch must never flip a flag it did not name.
        apply_renderer_activity(&state, "training", true);
        apply_renderer_activity(&state, "", false);
        apply_renderer_activity(&state, "Downloads", true);
        assert_eq!(renderer_activity(&state), (false, true));
    }
}
