#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app_layout;
mod app_menu;
mod browser_capture;
#[cfg(target_os = "macos")]
mod browser_context_downloads;
mod browser_downloads;
mod browser_proxy;
mod browser_webview;
mod commands;
#[cfg(target_os = "linux")]
mod debian_update;
mod desktop_auth;
mod desktop_backend_owner;
mod desktop_update_policy;
mod desktop_updater;
mod diagnostics;
mod install;
mod install_watchdog;
#[cfg(target_os = "linux")]
mod linux_webkit;
mod loopback_http;
#[cfg(target_os = "macos")]
mod macos_event_guard;
#[cfg(target_os = "macos")]
mod macos_tray;
mod native_backend_lease;
mod native_clipboard;
mod native_file_dialogs;
mod native_intents;
mod native_path_policy;
mod preflight;
mod process;
mod process_identity;
mod shell_path;
mod staged_update;
mod update;
mod webview_permissions;
mod windows_job;
mod x11_threads;

use log::{info, warn};
use process::new_backend_state;
use simplelog::{
    CombinedLogger, Config, LevelFilter, SharedLogger, TermLogger, TerminalMode, WriteLogger,
};
use std::ffi::OsStr;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use tauri::menu::{MenuBuilder, MenuItem, MenuItemBuilder};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{Emitter, Manager};
use tauri_plugin_window_state::{AppHandleExt, StateFlags};

/// Serializes the exit paths that reap the backend (request_quit, the Unix signal listener,
/// RunEvent::Exit). Exactly one runs cleanup; the others block, so the process never exits mid-reap.
static TERMINATION_CLEANUP: Mutex<bool> = Mutex::new(false);

const IN_APP_RELAUNCH_MARKER_FILE: &str = "in-app-relaunch-v1";

const CLOSE_TO_TRAY_PREFERENCE_FILE: &str = "close-to-tray-v1";

/// The user's launch-at-login answer, stored separately because the Windows HKCU Run value is
/// deleted by the NSIS uninstaller, antivirus and registry cleaners without asking.
const LAUNCH_AT_LOGIN_PREFERENCE_FILE: &str = "launch-at-login-v1";

struct CloseToTrayState(AtomicBool);

fn new_close_to_tray_state() -> CloseToTrayState {
    CloseToTrayState(AtomicBool::new(false))
}

/// Resolved once, at setup, where the marker is consumed, so no later caller can flip the answer.
static LAUNCHED_HIDDEN: OnceLock<bool> = OnceLock::new();

/// Marks the next start as an in-app relaunch, whose inherited `--hidden` is not a login start.
#[tauri::command]
fn mark_in_app_relaunch(app: tauri::AppHandle) -> Result<(), String> {
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

/// A failed removal can only show a would-be-hidden start, never the reverse.
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

/// args_os, not args: args panics on non-Unicode argv.
fn argv_has_hidden_flag() -> bool {
    std::env::args_os().any(|arg| arg == *OsStr::new("--hidden"))
}

fn resolve_launched_hidden(app: &tauri::AppHandle) -> bool {
    // Consume unconditionally, so a stale marker cannot outlive its relaunch.
    let relaunched = in_app_relaunch_config_dir(app)
        .map(|dir| take_in_app_relaunch_marker(&dir))
        .unwrap_or(false);
    argv_has_hidden_flag() && !relaunched
}

/// True for a login autostart; false for an in-app relaunch that inherits `--hidden`.
#[tauri::command]
fn was_launched_hidden(app: tauri::AppHandle) -> bool {
    *LAUNCHED_HIDDEN.get_or_init(|| resolve_launched_hidden(&app))
}

#[tauri::command]
fn reveal_main_window(app: tauri::AppHandle) {
    show_main_window(&app);
}

fn close_to_tray_preference_path(config_dir: &Path) -> PathBuf {
    config_dir.join(CLOSE_TO_TRAY_PREFERENCE_FILE)
}

fn read_close_to_tray_preference(config_dir: &Path) -> bool {
    let path = close_to_tray_preference_path(config_dir);
    match fs::read_to_string(&path) {
        Ok(value) => match value.trim().parse::<bool>() {
            Ok(enabled) => enabled,
            Err(error) => {
                warn!(
                    "Ignoring invalid close-to-tray preference {}: {error}",
                    path.display()
                );
                false
            }
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
        Err(error) => {
            warn!(
                "Could not read close-to-tray preference {}: {error}",
                path.display()
            );
            false
        }
    }
}

fn write_close_to_tray_preference(config_dir: &Path, enabled: bool) -> Result<(), String> {
    fs::create_dir_all(config_dir).map_err(|error| {
        format!(
            "Failed to create app configuration directory {}: {error}",
            config_dir.display()
        )
    })?;
    let path = close_to_tray_preference_path(config_dir);
    fs::write(&path, format!("{enabled}\n")).map_err(|error| {
        format!(
            "Failed to save close-to-tray preference {}: {error}",
            path.display()
        )
    })
}

fn launch_at_login_preference_path(config_dir: &Path) -> PathBuf {
    config_dir.join(LAUNCH_AT_LOGIN_PREFERENCE_FILE)
}

/// None is not false: with no record, inventing one could enable autostart nobody asked for.
fn read_launch_at_login_preference(config_dir: &Path) -> Option<bool> {
    let path = launch_at_login_preference_path(config_dir);
    match fs::read_to_string(&path) {
        Ok(value) => match value.trim().parse::<bool>() {
            Ok(enabled) => Some(enabled),
            Err(error) => {
                warn!(
                    "Ignoring invalid launch-at-login preference {}: {error}",
                    path.display()
                );
                None
            }
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => {
            warn!(
                "Could not read launch-at-login preference {}: {error}",
                path.display()
            );
            None
        }
    }
}

fn write_launch_at_login_preference(config_dir: &Path, enabled: bool) -> Result<(), String> {
    fs::create_dir_all(config_dir).map_err(|error| {
        format!(
            "Failed to create app configuration directory {}: {error}",
            config_dir.display()
        )
    })?;
    let path = launch_at_login_preference_path(config_dir);
    fs::write(&path, format!("{enabled}\n")).map_err(|error| {
        format!(
            "Failed to save launch-at-login preference {}: {error}",
            path.display()
        )
    })
}

/// A failed write must not fail the toggle: the OS entry is already written.
fn store_launch_at_login_preference(app: &tauri::AppHandle, enabled: bool) {
    match app.path().app_config_dir() {
        Ok(dir) => {
            if let Err(error) = write_launch_at_login_preference(&dir, enabled) {
                warn!("Could not record the launch-at-login preference: {error}");
            }
        }
        Err(error) => warn!("Could not determine app configuration directory: {error}"),
    }
}

fn stored_launch_at_login_preference(app: &tauri::AppHandle) -> Option<bool> {
    app.path()
        .app_config_dir()
        .ok()
        .and_then(|dir| read_launch_at_login_preference(&dir))
}

fn initialize_close_to_tray(app: &tauri::AppHandle) {
    let enabled = app
        .path()
        .app_config_dir()
        .map(|dir| read_close_to_tray_preference(&dir))
        .unwrap_or_else(|error| {
            warn!("Could not determine app configuration directory: {error}");
            false
        });
    app.state::<CloseToTrayState>()
        .0
        .store(enabled, Ordering::SeqCst);
}

#[tauri::command]
fn get_close_to_tray(state: tauri::State<'_, CloseToTrayState>) -> Option<bool> {
    cfg!(any(target_os = "windows", target_os = "linux")).then(|| state.0.load(Ordering::SeqCst))
}

#[tauri::command]
fn set_close_to_tray(
    app: tauri::AppHandle,
    state: tauri::State<'_, CloseToTrayState>,
    enabled: bool,
) -> Result<bool, String> {
    if !cfg!(any(target_os = "windows", target_os = "linux")) {
        return Err("Close to system tray is only configurable on Windows and Linux".to_string());
    }
    let config_dir = app
        .path()
        .app_config_dir()
        .map_err(|error| format!("Could not determine app configuration directory: {error}"))?;
    write_close_to_tray_preference(&config_dir, enabled)?;
    state.0.store(enabled, Ordering::SeqCst);
    Ok(enabled)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MainWindowCloseAction {
    Hide,
    Quit,
}

fn main_window_close_action(close_to_tray: bool) -> MainWindowCloseAction {
    if cfg!(target_os = "macos")
        || (cfg!(any(target_os = "windows", target_os = "linux")) && close_to_tray)
    {
        MainWindowCloseAction::Hide
    } else {
        MainWindowCloseAction::Quit
    }
}

/// On Windows reads both keys itself: `is_enabled` misreports re-enabled entries as off.
fn autostart_enabled(app: &tauri::AppHandle) -> Result<bool, String> {
    // is_enabled only checks the entry file exists, so a DE-disabled entry would read as on.
    if cfg!(target_os = "linux") && linux_autostart_disabled(app) {
        return Ok(false);
    }
    #[cfg(windows)]
    {
        // An unreadable Run key reads as present and disabled, which restores nothing.
        let (present, disabled) = windows_autostart_entry_state(app);
        Ok(present && !disabled)
    }
    #[cfg(not(windows))]
    {
        use tauri_plugin_autostart::ManagerExt;
        app.autolaunch()
            .is_enabled()
            .map_err(|error| error.to_string())
    }
}

#[tauri::command]
fn get_launch_at_login(app: tauri::AppHandle) -> Result<bool, String> {
    autostart_enabled(&app)
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
    // After the OS write, on both arms: a stored `false` stops the restore from fighting the user.
    store_launch_at_login_preference(&app, enabled);
    autostart_enabled(&app)
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

/// StartupApproved state is the FIRST byte: 02 enabled, 03 disabled, 06 re-enabled. Reading the
/// trailing timestamp bytes (as auto-launch does) misreads re-enabled entries. Absent means enabled.
#[cfg_attr(not(windows), allow(dead_code))]
fn startup_approved_disabled(bytes: &[u8]) -> bool {
    match bytes.first() {
        Some(0x02) | Some(0x06) | None => false,
        Some(_) => true,
    }
}

/// Restore a *deleted* entry, never a disabled one: disabling is a user decision, while no
/// Windows UI deletes the Run value.
#[cfg_attr(not(windows), allow(dead_code))]
fn should_restore_autostart_entry(
    stored: Option<bool>,
    entry_present: bool,
    disabled: bool,
) -> bool {
    stored == Some(true) && !entry_present && !disabled
}

/// Unreadable registry reads as present and disabled, which restores nothing.
#[cfg(windows)]
fn windows_autostart_entry_state(app: &tauri::AppHandle) -> (bool, bool) {
    use winreg::enums::{HKEY_CURRENT_USER, KEY_QUERY_VALUE};
    use winreg::RegKey;

    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let name = &app.package_info().name;
    let Ok(run) = hkcu.open_subkey_with_flags(
        r"Software\Microsoft\Windows\CurrentVersion\Run",
        KEY_QUERY_VALUE,
    ) else {
        return (true, true);
    };
    let present = run.get_value::<String, _>(name).is_ok();
    let disabled = hkcu
        .open_subkey_with_flags(
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run",
            KEY_QUERY_VALUE,
        )
        .ok()
        .and_then(|approved| approved.get_raw_value(name).ok())
        .is_some_and(|value| startup_approved_disabled(&value.bytes));
    (present, disabled)
}

/// Windows only: macOS and GNOME delete the login item as the user's off switch.
fn restore_missing_autostart_entry(app: &tauri::AppHandle) -> bool {
    #[cfg(windows)]
    {
        let (present, disabled) = windows_autostart_entry_state(app);
        let restore = should_restore_autostart_entry(
            stored_launch_at_login_preference(app),
            present,
            disabled,
        );
        if restore {
            info!(
                "The \"Run Unsloth at login\" entry is gone but was last set to on; restoring it."
            );
        }
        restore
    }
    #[cfg(not(windows))]
    {
        let _ = app;
        false
    }
}

/// Exec is whitespace-delimited: quote paths with reserved characters, escape `"`, `` ` ``, `$`, `\`,
/// and always double a literal `%`.
fn exec_quoted(binary: &str) -> String {
    let binary = binary.replace('%', "%%");
    const RESERVED: &str = " \t\n\"'\\><~|&;$*?#()`";
    if !binary.chars().any(|c| RESERVED.contains(c)) {
        return binary;
    }
    let mut quoted = String::with_capacity(binary.len() + 2);
    quoted.push('"');
    for c in binary.chars() {
        // The string rule runs before the quoting rule, so the escaping backslash is itself escaped.
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

/// TryExec lets launchers skip the entry once the install is removed. TryExec stays unquoted.
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
    // auto-launch hardcodes ~/.config regardless of XDG_CONFIG_HOME; mirror it.
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

/// The plugin's template interpolates the path unescaped, so & or < breaks the LaunchAgent.
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

/// A moved AppImage leaves a stale path `is_enabled` still accepts, so repoint on every startup.
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
    if autostart_enabled(app).unwrap_or(false) {
        // Adopt an entry made before this install kept a record, so its deletion can be undone.
        if stored_launch_at_login_preference(app).is_none() {
            store_launch_at_login_preference(app, true);
        }
    } else if !restore_missing_autostart_entry(app) {
        return;
    }
    if app.autolaunch().enable().is_ok() {
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

/// Rotates `tauri.log` at 5 MiB while the app runs, not only at launch. Rotation closes the
/// handle before renaming, which Windows requires.
struct RotatingLogFile {
    path: PathBuf,
    rotated_path: PathBuf,
    max_bytes: u64,
    file: Option<fs::File>,
    written: u64,
}

impl RotatingLogFile {
    fn open(
        path: PathBuf,
        rotated_path: PathBuf,
        max_bytes: u64,
    ) -> std::io::Result<RotatingLogFile> {
        let mut rotating = RotatingLogFile {
            path,
            rotated_path,
            max_bytes,
            file: None,
            written: 0,
        };
        rotating.reopen()?;
        Ok(rotating)
    }

    fn reopen(&mut self) -> std::io::Result<()> {
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        // An already-oversized file rotates on its first write.
        self.written = file.metadata().map(|meta| meta.len()).unwrap_or(0);
        self.file = Some(file);
        Ok(())
    }

    /// Best effort: if reopening fails, drop lines rather than crash.
    fn rotate(&mut self) {
        self.file = None;
        let _ = fs::remove_file(&self.rotated_path);
        let _ = fs::rename(&self.path, &self.rotated_path);
        if self.reopen().is_err() {
            self.file = None;
            self.written = 0;
        }
    }
}

impl Write for RotatingLogFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.written >= self.max_bytes {
            self.rotate();
        }
        match self.file.as_mut() {
            Some(file) => {
                let written = file.write(buf)?;
                self.written += written as u64;
                Ok(written)
            }
            // No usable handle: report the bytes as taken so the logger does not spin.
            None => Ok(buf.len()),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self.file.as_mut() {
            Some(file) => file.flush(),
            None => Ok(()),
        }
    }
}

fn setup_logging() {
    let mut loggers: Vec<Box<dyn SharedLogger>> = vec![];

    loggers.push(TermLogger::new(
        LevelFilter::Info,
        Config::default(),
        TerminalMode::Stderr,
        simplelog::ColorChoice::Auto,
    ));

    if let Some(home) = dirs::home_dir() {
        let log_dir = home.join(".unsloth").join("studio");
        if fs::create_dir_all(&log_dir).is_ok() {
            let log_path = log_dir.join("tauri.log");
            let rotated_path = log_dir.join("tauri.log.1");
            let max_log_bytes = 5 * 1024 * 1024;
            if let Ok(file) = RotatingLogFile::open(log_path.clone(), rotated_path, max_log_bytes) {
                loggers.push(WriteLogger::new(LevelFilter::Info, Config::default(), file));
                let _ = PANIC_LOG_PATH.set(log_path);
            }
        }
    }

    if !loggers.is_empty() {
        let _ = CombinedLogger::init(loggers);
    }
}

static PANIC_LOG_PATH: OnceLock<PathBuf> = OnceLock::new();

/// Own file handle, not `log`: a panic raised inside the logger's lock would deadlock.
fn log_panics() {
    static PANICS: AtomicU64 = AtomicU64::new(0);
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if let Some(path) = PANIC_LOG_PATH.get() {
            // Symbolizing is slow, so only the first few panics get a backtrace.
            let backtrace = if PANICS.fetch_add(1, Ordering::Relaxed) < 4 {
                format!("\n{}", std::backtrace::Backtrace::force_capture())
            } else {
                String::new()
            };
            let now = time::OffsetDateTime::now_utc();
            let thread = std::thread::current();
            if let Ok(mut file) = fs::OpenOptions::new().append(true).open(path) {
                let _ = writeln!(
                    file,
                    "{:02}:{:02}:{:02} [ERROR] thread '{}' {info}{backtrace}",
                    now.hour(),
                    now.minute(),
                    now.second(),
                    thread.name().unwrap_or("<unnamed>"),
                );
            }
        }
        default_hook(info);
    }));
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
fn setup_custom_titlebar(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let window = app.get_webview_window("main").ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound, "main window not found")
    })?;
    window.set_decorations(false)?;
    Ok(())
}

// tao reapplies `resizable: false` on the first configure, wiping setResizable calls made while hidden.
#[cfg(target_os = "linux")]
fn keep_resizable_across_first_configure(
    app: &tauri::App,
) -> Result<(), Box<dyn std::error::Error>> {
    use gtk::prelude::*;
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    let window = app.get_webview_window("main").ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound, "main window not found")
    })?;
    let gtk_window = window.gtk_window()?;
    let requested: Rc<Cell<Option<bool>>> = Rc::default();
    let handlers: Rc<RefCell<Vec<glib::SignalHandlerId>>> = Rc::default();

    let seen = requested.clone();
    // "event" fires before tao's configure handler; unrealized configures skip it, so refresh each time.
    let before = gtk_window.connect_event(move |window, event| {
        if event.event_type() == gdk::EventType::Configure {
            seen.set(Some(window.is_resizable()));
        }
        glib::Propagation::Proceed
    });
    let owned = handlers.clone();
    let after = gtk_window.connect_configure_event(move |window, _| {
        if let Some(resizable) = requested.take() {
            window.set_resizable(resizable);
            for id in owned.borrow_mut().drain(..) {
                window.disconnect(id);
            }
        }
        false
    });
    handlers.borrow_mut().extend([before, after]);
    Ok(())
}

// WebKitGTK disables media streams and denies unhandled permission requests by default, which
// breaks dictation. Allow only user-media requests; every other kind keeps the default deny.
#[cfg(target_os = "linux")]
fn setup_linux_media_permissions(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    use webkit2gtk::{
        glib::Cast, PermissionRequestExt, SettingsExt, UserMediaPermissionRequest, WebViewExt,
    };

    let window = app.get_webview_window("main").ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound, "main window not found")
    })?;
    window.with_webview(|webview| {
        let webview = webview.inner();
        if let Some(settings) = webview.settings() {
            settings.set_enable_media_stream(true);
        }
        webview.connect_permission_request(|_webview, request| {
            match request.downcast_ref::<UserMediaPermissionRequest>() {
                Some(request) => {
                    request.allow();
                    true
                }
                None => false,
            }
        });
    })?;
    Ok(())
}

// Compile in both Windows profiles so dependency skew fails in CI.
#[cfg(windows)]
#[cfg_attr(debug_assertions, allow(dead_code))]
fn setup_windows_browser_guards(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    use webview2_com::Microsoft::Web::WebView2::Win32::{
        ICoreWebView2Controller, ICoreWebView2_11, COREWEBVIEW2_CONTEXT_MENU_TARGET_KIND,
        COREWEBVIEW2_CONTEXT_MENU_TARGET_KIND_PAGE,
    };
    use webview2_com::{AcceleratorKeyPressedEventHandler, ContextMenuRequestedEventHandler};
    use windows_core::Interface;
    use windows_core::BOOL;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        GetKeyState, VK_CONTROL, VK_F5, VK_MENU, VK_R, VK_SHIFT,
    };

    let window = app.get_webview_window("main").ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound, "main window not found")
    })?;
    window.with_webview(|webview| unsafe {
        // Keep the menus and non-refresh accelerators the WebView2 settings would disable together.
        // The explicit type makes dependency version mismatches fail here.
        let controller: ICoreWebView2Controller = webview.controller();
        let accelerator_handler =
            AcceleratorKeyPressedEventHandler::create(Box::new(move |_, event_args| {
                let Some(event_args) = event_args else {
                    return Ok(());
                };
                let mut virtual_key = 0;
                event_args.VirtualKey(&mut virtual_key)?;
                let control_down = GetKeyState(i32::from(VK_CONTROL)) < 0;
                // AltGr reports as Ctrl+Alt, so never treat it as Ctrl+R.
                let alt_down = GetKeyState(i32::from(VK_MENU)) < 0;
                // Preserve Ctrl+Shift+R as recovery from a broken renderer.
                let shift_down = GetKeyState(i32::from(VK_SHIFT)) < 0;
                if virtual_key == u32::from(VK_F5)
                    || (virtual_key == u32::from(VK_R) && control_down && !alt_down && !shift_down)
                {
                    event_args.SetHandled(true)?;
                }
                Ok(())
            }));
        let mut accelerator_token = 0;
        if let Err(error) =
            controller.add_AcceleratorKeyPressed(&accelerator_handler, &mut accelerator_token)
        {
            warn!("Could not block Windows refresh shortcuts: {error}");
        }

        let core_webview = match controller.CoreWebView2() {
            Ok(core_webview) => core_webview,
            Err(error) => {
                warn!("Could not access the Windows WebView: {error}");
                return;
            }
        };
        let core_webview11 = match core_webview.cast::<ICoreWebView2_11>() {
            Ok(core_webview11) => core_webview11,
            Err(error) => {
                warn!("Could not customize the Windows context menu: {error}");
                return;
            }
        };
        let context_menu_handler =
            ContextMenuRequestedEventHandler::create(Box::new(move |_, event_args| {
                let Some(event_args) = event_args else {
                    return Ok(());
                };
                let target = event_args.ContextMenuTarget()?;
                let mut kind = COREWEBVIEW2_CONTEXT_MENU_TARGET_KIND::default();
                let mut is_editable = BOOL::default();
                let mut has_link_uri = BOOL::default();
                let mut has_selection = BOOL::default();
                target.Kind(&mut kind)?;
                target.IsEditable(&mut is_editable)?;
                target.HasLinkUri(&mut has_link_uri)?;
                target.HasSelection(&mut has_selection)?;
                // Suppress only bare-page browser commands; keep media, selection, editable and link menus.
                let is_bare_page = kind == COREWEBVIEW2_CONTEXT_MENU_TARGET_KIND_PAGE
                    && !is_editable.as_bool()
                    && !has_link_uri.as_bool()
                    && !has_selection.as_bool();
                if is_bare_page {
                    event_args.SetHandled(true)?;
                }
                Ok(())
            }));
        let mut context_menu_token = 0;
        if let Err(error) =
            core_webview11.add_ContextMenuRequested(&context_menu_handler, &mut context_menu_token)
        {
            warn!("Could not filter the Windows context menu: {error}");
        }
    })?;
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

fn training_is_active(app: &tauri::AppHandle) -> bool {
    let Some(state) = app.try_state::<TrainingActivityState>() else {
        return false;
    };
    state.lock().map(|running| *running).unwrap_or(false)
}

fn install_is_active(app: &tauri::AppHandle) -> bool {
    let Some(state) = app.try_state::<install::InstallState>() else {
        return false;
    };
    install::is_install_running(&state)
}

/// Called only from the shared confirmation sequence below.
fn confirm_quit_during_training(app: &tauri::AppHandle) -> bool {
    use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

    if !training_is_active(app) {
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

fn confirm_update_during_training(app: &tauri::AppHandle) -> bool {
    use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

    app.dialog()
        .message(
            "Training is starting or still running. Updating now stops the \
             run and loses progress since the last checkpoint.",
        )
        .kind(MessageDialogKind::Warning)
        .title("Training in progress")
        .buttons(MessageDialogButtons::OkCancelCustom(
            "Update anyway".to_string(),
            "Keep training".to_string(),
        ))
        .blocking_show()
}

/// renderer-owned downloads, shell updates and unsaved transcripts must also protect native quit.
#[derive(Default)]
pub struct RendererActivity {
    pub downloads: bool,
    pub shell_update: bool,
    pub unsaved_transcript: bool,
}

pub type RendererActivityState = std::sync::Arc<std::sync::Mutex<RendererActivity>>;

fn new_renderer_activity_state() -> RendererActivityState {
    std::sync::Arc::new(std::sync::Mutex::new(RendererActivity::default()))
}

fn apply_renderer_activity(state: &RendererActivityState, kind: &str, active: bool) {
    if let Ok(mut activity) = state.lock() {
        match kind {
            "downloads" => activity.downloads = active,
            "shell_update" => activity.shell_update = active,
            "unsaved_transcript" => activity.unsaved_transcript = active,
            // An unknown kind is a renderer/Rust mismatch, never a reason to flip a flag.
            _ => {}
        }
    }
}

#[tauri::command]
fn set_renderer_activity(state: tauri::State<'_, RendererActivityState>, kind: &str, active: bool) {
    apply_renderer_activity(state.inner(), kind, active);
}

fn current_renderer_activity(app: &tauri::AppHandle) -> RendererActivity {
    let Some(state) = app.try_state::<RendererActivityState>() else {
        return RendererActivity::default();
    };
    state
        .lock()
        .map(|activity| RendererActivity {
            downloads: activity.downloads,
            shell_update: activity.shell_update,
            unsaved_transcript: activity.unsaved_transcript,
        })
        .unwrap_or_default()
}

fn confirm_quit_with_unsaved_transcript(app: &tauri::AppHandle) -> bool {
    use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

    if !current_renderer_activity(app).unsaved_transcript {
        return true;
    }
    app.dialog()
        .message("A transcript could not be saved. Download a copy before quitting to keep it.")
        .kind(MessageDialogKind::Warning)
        .title("Unsaved transcript")
        .buttons(MessageDialogButtons::OkCancelCustom(
            "Quit anyway".to_string(),
            "Keep open".to_string(),
        ))
        .blocking_show()
}

fn confirm_quit_during_shell_update(app: &tauri::AppHandle) -> bool {
    use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

    if !current_renderer_activity(app).shell_update {
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

fn confirm_quit_during_downloads(app: &tauri::AppHandle) -> bool {
    use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

    if !current_renderer_activity(app).downloads {
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

/// Cleanup SIGTERMs the installer, leaving a venv that cannot start. Never called from
/// RunEvent::Exit, which must not block on a dialog.
fn confirm_quit_during_install(app: &tauri::AppHandle) -> bool {
    use tauri_plugin_dialog::{DialogExt, MessageDialogButtons, MessageDialogKind};

    if !install_is_active(app) {
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

/// A killed update leaves the venv unusable until repaired.
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

/// Let a later exit run cleanup again after a cancelled or failed installer spent the guard.
fn reset_termination_cleanup() {
    match TERMINATION_CLEANUP.lock() {
        Ok(mut done) => *done = false,
        Err(poisoned) => *poisoned.into_inner() = false,
    }
}

/// AppKit termination must decide synchronously: terminate now or NSTerminateLater.
#[cfg(target_os = "macos")]
fn quit_requires_confirmation(app: &tauri::AppHandle) -> bool {
    let update_active = app
        .try_state::<update::UpdateState>()
        .is_some_and(|state| update::is_update_running(&state));
    let renderer = current_renderer_activity(app);
    install_is_active(app)
        || update_active
        || renderer.shell_update
        || renderer.unsaved_transcript
        || training_is_active(app)
        || renderer.downloads
}

fn cleanup_child_processes(app: &tauri::AppHandle) {
    // Resettable, not `Once`. The flag is set after the body, so a panicking cleanup poisons the
    // lock and the next caller retries instead of exiting with an unreaped backend.
    let mut done = match TERMINATION_CLEANUP.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            warn!("Previous termination cleanup panicked, retrying it");
            poisoned.into_inner()
        }
    };
    if *done {
        return;
    }
    {
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
    }
    *done = true;
}

/// The backend is its own process-group leader and Tauri installs no signal handler, so reap it
/// here. Windows uses the kill-on-close job in `windows_job`.
#[cfg(unix)]
fn setup_unix_termination_signals(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    use signal_hook::consts::signal::{SIGHUP, SIGINT, SIGQUIT, SIGTERM};
    use signal_hook::iterator::Signals;

    let mut signals = Signals::new([SIGHUP, SIGINT, SIGQUIT, SIGTERM])?;
    let app_handle = app.handle().clone();
    std::thread::Builder::new()
        .name("desktop-termination-signal".to_string())
        .spawn(move || {
            // Exit regardless of cleanup success, or SIGTERM leaves the app running and the backend
            // orphaned.
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
                    // A repeat signal means stop waiting: exit straight away.
                    warn!("Received {name} ({signal}) while cleaning up, exiting immediately");
                    std::process::exit(exit_code);
                }
                cleanup_started = true;
                info!("Received Unix termination signal {name} ({signal})");

                // Clean up on another thread so this one can still observe a repeat signal.
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
    // Not get_webview_window: that is None while browser views are children of the window.
    if let Some(window) = app.get_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
    browser_webview::window_changed(app, None);
}

struct QuitConfirmationState {
    in_progress: bool,
    #[cfg(target_os = "macos")]
    termination_reply_pending: bool,
}

static QUIT_CONFIRMATION_STATE: Mutex<QuitConfirmationState> = Mutex::new(QuitConfirmationState {
    in_progress: false,
    #[cfg(target_os = "macos")]
    termination_reply_pending: false,
});

fn lock_quit_confirmation_state() -> MutexGuard<'static, QuitConfirmationState> {
    QUIT_CONFIRMATION_STATE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Clears the flag on cancel, spawn failure, or panic, so a quit cannot leave the app inert.
struct QuitGuard {
    active: bool,
}

impl QuitGuard {
    fn new() -> Self {
        Self { active: true }
    }

    /// Atomically close this confirmation to new attachments and take the AppKit reply, if any.
    fn finish(mut self) -> bool {
        let mut state = lock_quit_confirmation_state();
        state.in_progress = false;
        #[cfg(target_os = "macos")]
        let reply_pending = std::mem::take(&mut state.termination_reply_pending);
        #[cfg(not(target_os = "macos"))]
        let reply_pending = false;
        self.active = false;
        reply_pending
    }
}

impl Drop for QuitGuard {
    fn drop(&mut self) {
        if self.active {
            let mut state = lock_quit_confirmation_state();
            state.in_progress = false;
            #[cfg(target_os = "macos")]
            {
                state.termination_reply_pending = false;
            }
        }
    }
}

fn begin_quit() -> Option<QuitGuard> {
    let mut state = lock_quit_confirmation_state();
    if state.in_progress {
        return None;
    }
    state.in_progress = true;
    Some(QuitGuard::new())
}

#[cfg(target_os = "macos")]
enum TerminationConfirmation {
    Now,
    Start(QuitGuard),
    Attached,
    Duplicate,
}

/// Atomically attach AppKit to a visible confirmation or reserve the guard for a new one.
#[cfg(target_os = "macos")]
fn begin_or_attach_termination(requires_confirmation: bool) -> TerminationConfirmation {
    let mut state = lock_quit_confirmation_state();
    if state.termination_reply_pending {
        return TerminationConfirmation::Duplicate;
    }
    if state.in_progress {
        state.termination_reply_pending = true;
        return TerminationConfirmation::Attached;
    }
    if !requires_confirmation {
        return TerminationConfirmation::Now;
    }
    state.in_progress = true;
    state.termination_reply_pending = true;
    TerminationConfirmation::Start(QuitGuard::new())
}

#[cfg(target_os = "macos")]
fn take_pending_termination_reply() -> bool {
    let mut state = lock_quit_confirmation_state();
    std::mem::take(&mut state.termination_reply_pending)
}

/// Renderer closing overlay. Reaping takes up to ~18s on Windows (probes, shutdown requests,
/// CTRL_BREAK waits), during which the window would look frozen.
const APP_CLOSING_EVENT: &str = "app-closing";

const APP_CLOSING_CANCELLED_EVENT: &str = "app-closing-cancelled";

/// A guard rather than paired emits, so early returns and unwinds always retract the overlay.
struct ClosingOverlay<E: Fn(&str)> {
    emit: E,
    retract: bool,
}

impl<E: Fn(&str)> ClosingOverlay<E> {
    fn raise(emit: E) -> Self {
        emit(APP_CLOSING_EVENT);
        Self {
            emit,
            retract: true,
        }
    }

    fn keep(mut self) {
        self.retract = false;
    }
}

impl<E: Fn(&str)> Drop for ClosingOverlay<E> {
    fn drop(&mut self) {
        if self.retract {
            (self.emit)(APP_CLOSING_CANCELLED_EVENT);
        }
    }
}

/// Windows only. Skips windows nobody can see (tray quit, `--hidden` start); a minimized window
/// counts as visible. `None` means no main window; an unreadable visibility raises the overlay.
fn quit_raises_the_overlay(
    windows: bool,
    main_window_visible: impl FnOnce() -> Option<Result<bool, String>>,
) -> bool {
    if !windows {
        return false;
    }
    match main_window_visible() {
        None => false,
        Some(Ok(visible)) => visible,
        Some(Err(error)) => {
            warn!("Could not read the main window visibility ({error}); covering the quit anyway");
            true
        }
    }
}

/// Confirm, cover, reap; returns whether to exit. Blocking parts are injected so the order is
/// testable. The overlay goes up after the confirm dialogs, never before, and before the reap.
fn quit_sequence(
    confirm: impl Fn() -> bool,
    cover: impl FnOnce() -> bool,
    reap: impl FnOnce(),
    emit: impl Fn(&str),
) -> bool {
    if !confirm() {
        return false;
    }
    // Asked after the confirmations, so a declined quit never does the blocking visibility read.
    let overlay = cover().then(|| ClosingOverlay::raise(emit));
    reap();
    if let Some(overlay) = overlay {
        overlay.keep();
    }
    true
}

/// `reserved_guard` lets AppKit atomically attach-or-reserve before spawning.
fn spawn_quit_confirmation<F>(
    app: &tauri::AppHandle,
    reserved_guard: Option<QuitGuard>,
    done: F,
) -> bool
where
    F: FnOnce(&tauri::AppHandle, bool) + Send + 'static,
{
    let guard = match reserved_guard {
        Some(guard) => guard,
        None => {
            let Some(guard) = begin_quit() else {
                info!("Quit already awaiting confirmation, ignoring the repeat request");
                return false;
            };
            guard
        }
    };
    let app = app.clone();
    let spawned = std::thread::Builder::new()
        .name("request-quit".to_string())
        .spawn(move || {
            // Driven from here so the tray Quit is covered on the same terms as the close button.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                quit_sequence(
                    || {
                        confirm_quit_during_install(&app)
                            && confirm_quit_during_update(&app)
                            && confirm_quit_during_shell_update(&app)
                            && confirm_quit_during_training(&app)
                            && confirm_quit_with_unsaved_transcript(&app)
                            && confirm_quit_during_downloads(&app)
                    },
                    || {
                        quit_raises_the_overlay(
                            // Windows only: there stop_backend's serial budgets made the window
                            // look frozen.
                            cfg!(target_os = "windows"),
                            || {
                                app.get_window("main")
                                    .map(|window| window.is_visible().map_err(|e| e.to_string()))
                            },
                        )
                    },
                    || cleanup_child_processes(&app),
                    |event| {
                        let _ = app.emit(event, ());
                    },
                )
            }));
            let proceed = result.as_ref().copied().unwrap_or(false);
            let reply_pending = guard.finish();
            #[cfg(target_os = "macos")]
            if reply_pending {
                reply_to_termination_request(&app, proceed);
            }
            if let Err(payload) = result {
                std::panic::resume_unwind(payload);
            }
            done(&app, proceed);
        });
    if let Err(error) = spawned {
        warn!("Could not spawn the quit thread: {error}");
        return false;
    }
    true
}

/// Never exit first: that would orphan the backend tree. Exit's cleanup is an idempotent safety net.
fn request_quit(app: &tauri::AppHandle) {
    spawn_quit_confirmation(app, None, |app, proceed| {
        if proceed {
            app.exit(0);
        }
    });
}

#[cfg(target_os = "macos")]
const APP_QUIT_MENU_ID: &str = "app-quit";

/// Replace the native Quit role so Cmd+Q enters the same guarded confirmation path as the tray.
#[cfg(target_os = "macos")]
fn setup_quit_menu(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let handle = app.handle();
    let menu = tauri::menu::Menu::default(handle)?;
    let Some(app_menu) = menu
        .items()?
        .first()
        .and_then(|item| item.as_submenu().cloned())
    else {
        return Ok(());
    };
    if let Some(quit) = app_menu.items()?.last() {
        app_menu.remove(quit)?;
    }
    let quit = MenuItemBuilder::with_id(APP_QUIT_MENU_ID, "Quit Unsloth")
        .accelerator("CmdOrCtrl+Q")
        .build(app)?;
    app_menu.append(&quit)?;
    app_menu::setup_app_menus(app, &menu)?;
    app.set_menu(menu)?;
    app.on_menu_event(|app, event| {
        if event.id() == APP_QUIT_MENU_ID {
            request_quit(app);
        } else {
            app_menu::handle_menu_event(app, event.id().as_ref());
        }
    });
    Ok(())
}

#[cfg(target_os = "macos")]
static TERMINATE_APP_HANDLE: std::sync::OnceLock<tauri::AppHandle> = std::sync::OnceLock::new();

#[cfg(target_os = "macos")]
extern "C-unwind" fn application_should_terminate(
    _this: *mut objc2::runtime::AnyObject,
    _cmd: objc2::runtime::Sel,
    _sender: *mut objc2::runtime::AnyObject,
) -> usize {
    const NS_TERMINATE_CANCEL: usize = 0;
    const NS_TERMINATE_NOW: usize = 1;
    const NS_TERMINATE_LATER: usize = 2;

    let Some(app) = TERMINATE_APP_HANDLE.get() else {
        return NS_TERMINATE_NOW;
    };
    match begin_or_attach_termination(quit_requires_confirmation(app)) {
        TerminationConfirmation::Now => NS_TERMINATE_NOW,
        TerminationConfirmation::Attached => NS_TERMINATE_LATER,
        TerminationConfirmation::Duplicate => NS_TERMINATE_CANCEL,
        TerminationConfirmation::Start(guard) => {
            if spawn_quit_confirmation(app, Some(guard), |_, _| {}) {
                NS_TERMINATE_LATER
            } else {
                // The worker never started, so no caller can deliver the promised reply.
                take_pending_termination_reply();
                NS_TERMINATE_CANCEL
            }
        }
    }
}

/// AppKit expects it on the main thread; if that loop is gone, so is the request.
#[cfg(target_os = "macos")]
fn reply_to_termination_request(app: &tauri::AppHandle, proceed: bool) {
    use objc2::runtime::{AnyObject, Bool};

    let result = app.run_on_main_thread(move || unsafe {
        let nsapp: *mut AnyObject =
            objc2::msg_send![objc2::class!(NSApplication), sharedApplication];
        let () = objc2::msg_send![nsapp, replyToApplicationShouldTerminate: Bool::new(proceed)];
    });
    if let Err(error) = result {
        warn!("Could not reply to the pending termination request: {error}");
    }
}

/// tao leaves `applicationShouldTerminate:` unimplemented, so Dock/logout/AppleScript quits skip
/// the menu handler. Add it: defer with NSTerminateLater while a run is active.
#[cfg(target_os = "macos")]
fn setup_terminate_interception(app: &tauri::App) {
    use objc2::ffi::{class_addMethod, object_getClass};
    use objc2::runtime::{AnyObject, Imp, Sel};

    let _ = TERMINATE_APP_HANDLE.set(app.handle().clone());
    unsafe {
        let nsapp: *mut AnyObject =
            objc2::msg_send![objc2::class!(NSApplication), sharedApplication];
        let delegate: *mut AnyObject = objc2::msg_send![nsapp, delegate];
        if delegate.is_null() {
            warn!("No NSApplication delegate; external quits will not be confirmed");
            return;
        }
        let imp: Imp = std::mem::transmute(
            application_should_terminate
                as extern "C-unwind" fn(*mut AnyObject, Sel, *mut AnyObject) -> usize,
        );
        let added = class_addMethod(
            object_getClass(delegate).cast_mut(),
            objc2::sel!(applicationShouldTerminate:),
            imp,
            c"Q@:@".as_ptr(),
        );
        if !added.as_bool() {
            warn!(
                "Could not hook applicationShouldTerminate; external quits will not be confirmed"
            );
        }
    }
}

struct TrayServerToggle(MenuItem<tauri::Wry>);

fn tray_toggle_label(status: &str) -> (&'static str, bool) {
    match status {
        "running" => ("Stop Server", true),
        "stopped" | "error" => ("Start Server", true),
        "starting" => ("Starting\u{2026}", false),
        _ => ("Start Server", false),
    }
}

#[tauri::command]
fn set_tray_server_status(app: tauri::AppHandle, status: String) {
    if let Some(toggle) = app.try_state::<TrayServerToggle>() {
        let (text, enabled) = tray_toggle_label(&status);
        let _ = toggle.0.set_text(text);
        let _ = toggle.0.set_enabled(enabled);
    }
}

fn setup_tray(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let open = MenuItemBuilder::with_id("open", "Open Unsloth").build(app)?;
    let toggle = MenuItemBuilder::with_id("toggle", "Start/Stop Server").build(app)?;
    let quit = MenuItemBuilder::with_id("quit", "Quit").build(app)?;
    let menu = MenuBuilder::new(app)
        .items(&[&open, &toggle, &quit])
        .build()?;
    app.manage(TrayServerToggle(toggle));

    // macOS renders tray images at 18 points: embed the 36 px scale; template mode adapts the color.
    #[cfg(target_os = "macos")]
    let tray_icon = tauri::include_image!("./icons/tray-icon@2x.png");
    #[cfg(not(target_os = "macos"))]
    let tray_icon = tauri::include_image!("./icons/tray-icon-color.png");

    let tray = TrayIconBuilder::new()
        .menu(&menu)
        .tooltip("Unsloth")
        .icon(tray_icon)
        .icon_as_template(cfg!(target_os = "macos"))
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

    #[cfg(target_os = "macos")]
    if let Err(error) = macos_tray::install_appearance_observer(&tray) {
        warn!("Could not install the macOS tray appearance observer: {error}");
    }
    #[cfg(not(target_os = "macos"))]
    drop(tray);

    Ok(())
}

// Same call Tauri's PathResolver uses for LocalData/<bid>, with passwd/known-folder fallbacks.
fn webview_profile_root(bundle_id: &str) -> Option<std::path::PathBuf> {
    dirs::data_local_dir().map(|d| d.join(bundle_id))
}

// Clear WebView caches after an update, which ran setup while the old WebView held them.
// Version-stamped, cache-only, mirroring setup.sh _clear_webview_caches / setup.ps1.
#[must_use]
fn clear_webview_caches(bundle_id: &str, version: &str) -> Option<fs::File> {
    let root = webview_profile_root(bundle_id)?;
    // Claim the profile lock BEFORE reading the stamp and keep it, so a newer executable started
    // alongside cannot delete this instance's live profile.
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
    if let Some(caches) = dirs::cache_dir() {
        // Library/WebKit/<bid> is user storage and is left alone.
        paths.push(caches.join(bundle_id));
    }
    #[cfg(target_os = "linux")]
    for sub in ["WebKitCache", "CacheStorage", "serviceworkers"] {
        paths.push(root.join(sub));
    }

    let mut cleared = true;
    for p in &paths {
        // Absent is normal; anything else shows up as a stale frontend, so log it.
        if let Err(e) = fs::remove_dir_all(p) {
            if e.kind() != std::io::ErrorKind::NotFound {
                warn!("could not clear WebView cache {}: {e}", p.display());
                cleared = false;
            }
        }
    }
    // Do not stamp a partial clear, or later launches skip the retry.
    if cleared {
        let _ = fs::write(&stamp, version);
    }
    Some(lock)
}

// Never dropped: the lock has to outlive the clear (see the function).
static WEBVIEW_PROFILE_LOCK: std::sync::OnceLock<Option<fs::File>> = std::sync::OnceLock::new();

// Register directly after tauri_plugin_single_instance: setup hooks run in registration order
// inside Builder::build(), before the window's WebView locks these files.
fn webview_cache_plugin<R: tauri::Runtime>() -> tauri::plugin::TauriPlugin<R> {
    tauri::plugin::Builder::new("unsloth-webview-cache")
        .setup(|app, _api| {
            let version = app.package_info().version.to_string();
            let _ =
                WEBVIEW_PROFILE_LOCK.set(clear_webview_caches(&app.config().identifier, &version));
            Ok(())
        })
        .build()
}

/// Mirror endpoints for the webview CSP: this process, plus any backend it adopts.
fn configured_hf_endpoints() -> Vec<String> {
    let mut raw: Vec<String> = ["HF_ENDPOINT", "HF_DATASETS_SERVER"]
        .into_iter()
        .filter_map(|key| std::env::var(key).ok())
        .collect();
    // An adopted backend may have been started with an HF_ENDPOINT this process lacks.
    raw.extend(desktop_backend_owner::recorded_hf_endpoints());
    csp_sources_from(raw)
}

fn csp_sources_from(raw: Vec<String>) -> Vec<String> {
    let mut sources: Vec<String> = Vec::new();
    for value in raw {
        let value = value.trim().to_string();
        if value.is_empty() {
            continue;
        }
        let with_scheme =
            if value.contains("://") { value } else { format!("https://{value}") };
        let trimmed = with_scheme.trim_end_matches('/');
        let normalised = match split_scheme(trimmed) {
            Some((scheme, rest)) => format!("{scheme}://{rest}"),
            None => trimmed.to_string(),
        };
        if !is_usable_csp_source(&normalised) {
            continue;
        }
        let source = csp_origin_of(&normalised);
        if !sources.contains(&source) {
            sources.push(source);
        }
    }
    sources
}

/// Split "scheme://rest", scheme lowercased (RFC 3986 3.1).
fn split_scheme(endpoint: &str) -> Option<(String, &str)> {
    endpoint
        .split_once("://")
        .map(|(scheme, rest)| (scheme.to_ascii_lowercase(), rest))
}

/// Mirrors `utils/hf_endpoint.py::is_loopback_host`.
fn is_loopback_host(authority: &str) -> bool {
    let host = match authority.rsplit_once(':') {
        Some((h, port)) if !authority.ends_with(']') && port.chars().all(|c| c.is_ascii_digit()) => h,
        _ => authority,
    };
    let host = host.trim_start_matches('[').trim_end_matches(']').to_ascii_lowercase();
    if host == "localhost" || host.ends_with(".localhost") {
        return true;
    }
    // Parsed, not string-matched: 0:0:0:0:0:0:0:1 is ::1, and 127.x.y.z all count.
    matches!(host.parse::<std::net::IpAddr>(), Ok(ip) if ip.is_loopback())
}

/// A host-source is matched as a string (CSP3 6.7.2.5) and the browser sends the compressed form.
fn canonical_authority(authority: &str) -> String {
    let (host, port) = match authority.strip_prefix('[') {
        Some(rest) => match rest.split_once(']') {
            Some((host, tail)) => (host, tail),
            None => return authority.to_string(),
        },
        None => return authority.to_string(),
    };
    match host.parse::<std::net::Ipv6Addr>() {
        Ok(ip) => format!("[{ip}]{port}"),
        Err(_) => authority.to_string(),
    }
}

/// A host-source with a path is matched EXACTLY unless it ends in a slash (CSP3 6.7.2.7).
fn csp_origin_of(endpoint: &str) -> String {
    match split_scheme(endpoint) {
        Some((scheme, rest)) => {
            let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
            format!("{scheme}://{}", canonical_authority(authority))
        }
        None => endpoint.to_string(),
    }
}

/// Mirrors `utils/hf_endpoint.py::_sanitize`, so the CSP and /api/health agree.
fn is_usable_csp_source(endpoint: &str) -> bool {
    let (scheme, authority) = match split_scheme(endpoint) {
        Some((scheme, rest)) if scheme == "http" || scheme == "https" => (scheme, rest),
        _ => return false,
    };
    if endpoint
        .chars()
        .any(|c| c.is_whitespace() || c.is_control() || matches!(c, ';' | ',' | '\'' | '"' | '\\'))
    {
        return false;
    }
    // No IDNA encoder here, and latin-1 CSP headers on the backend.
    if !endpoint.is_ascii() {
        return false;
    }
    let host = authority
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default();
    if host.is_empty() || host.contains('@') || authority.contains('?') || authority.contains('#') {
        return false;
    }
    if host.contains('*') {
        return false;
    }
    // Hub calls carry the user's token: http off-box puts it on the wire.
    if scheme == "http" && !is_loopback_host(host) {
        return false;
    }
    // "javascript:alert(1)" arrives as "https://javascript:alert(1)".
    match host.rsplit_once(':') {
        Some((_, port)) if !host.ends_with(']') => {
            !port.is_empty() && port.chars().all(|c| c.is_ascii_digit())
        }
        _ => true,
    }
}

/// Matched on the directive's first token, so a hostname containing it cannot match.
fn append_connect_sources(policy: &mut String, sources: &[String]) -> bool {
    append_sources_to(policy, "connect-src", sources)
}

/// img-src and media-src carry a bare `https:`, which does not cover a plain-HTTP loopback mirror.
fn append_sources_to(policy: &mut String, directive_name: &str, sources: &[String]) -> bool {
    let mut appended = false;
    let rebuilt: Vec<String> = policy
        .split(';')
        .map(|directive| directive.trim())
        .filter(|directive| !directive.is_empty())
        .map(|directive| {
            let is_target = directive
                .split_whitespace()
                .next()
                .is_some_and(|name| name.eq_ignore_ascii_case(directive_name));
            if !is_target || appended {
                return directive.to_string();
            }
            let existing: Vec<&str> = directive.split_whitespace().collect();
            let mut tokens = existing.clone();
            for source in sources {
                if !existing.iter().any(|token| *token == source.as_str()) {
                    tokens.push(source);
                }
            }
            appended = true;
            tokens.join(" ")
        })
        .collect();
    if appended {
        *policy = rebuilt.join("; ");
    }
    appended
}

/// Tauri builds the CSP header from the Context config per request, so this covers the static window.
fn extend_csp_with_hf_endpoints<R: tauri::Runtime>(context: &mut tauri::Context<R>) {
    let endpoints = configured_hf_endpoints();
    if endpoints.is_empty() {
        return;
    }
    if let Some(tauri::utils::config::Csp::Policy(policy)) =
        context.config_mut().app.security.csp.as_mut()
    {
        append_connect_sources(policy, &endpoints);
        let assets: Vec<String> = endpoints
            .iter()
            .filter(|e| e.starts_with("http://"))
            .cloned()
            .collect();
        if !assets.is_empty() {
            append_sources_to(policy, "img-src", &assets);
            append_sources_to(policy, "media-src", &assets);
        }
    }
}

fn main() {
    #[cfg(target_os = "linux")]
    if let Some(result) = debian_update::run_installer() {
        match result {
            Ok(()) => std::process::exit(0),
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1);
            }
        }
    }

    // Must precede any Xlib call; see x11_threads.
    x11_threads::init_x11_threads();

    // Pick a WebKitGTK rendering fallback before any GTK/WebKit object is initialized.
    #[cfg(target_os = "linux")]
    let webkit_rendering_workaround = linux_webkit::configure_renderer();
    // Fix PATH for GUI apps (macOS .app bundles, Linux AppImage, Windows)
    // GUI apps don't inherit shell dotfile PATH — this spawns the user's
    // login shell to source .zshrc/.bashrc/.profile and sets PATH properly.
    shell_path::fix_path();

    setup_logging();
    log_panics();
    info!("Unsloth desktop app starting");
    #[cfg(target_os = "macos")]
    macos_event_guard::install();

    #[cfg(target_os = "linux")]
    if let Some((variables, reason)) = webkit_rendering_workaround {
        let applied = variables
            .iter()
            .map(|variable| format!("{variable}=1"))
            .collect::<Vec<_>>()
            .join(" ");
        info!("{reason}; set {applied} for WebKitGTK compatibility");
    }
    windows_job::initialize();

    let mut context = tauri::generate_context!();
    extend_csp_with_hf_endpoints(&mut context);
    // Restore while hidden, else the 760x560 setup size overwrites the saved layout.
    let restore_initial_layout = dirs::config_dir().is_some_and(|dir| {
        app_layout::should_restore_initial_window_state(
            &dir.join(&context.config().identifier),
            tauri_plugin_window_state::DEFAULT_FILENAME,
        )
    });
    info!("Native saved app layout restore enabled: {restore_initial_layout}");
    let mut window_state = tauri_plugin_window_state::Builder::new()
        .with_state_flags(StateFlags::SIZE | StateFlags::POSITION | StateFlags::MAXIMIZED);
    if !restore_initial_layout {
        window_state = window_state.skip_initial_state("main");
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            show_main_window(app);
        }))
        .plugin(webview_cache_plugin())
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
        .plugin(window_state.build())
        .manage(app_layout::NativeLayoutRestored(
            std::sync::atomic::AtomicBool::new(restore_initial_layout),
        ))
        .manage(diagnostics::new_diagnostics_state())
        .manage(install::new_install_state())
        .manage(new_training_activity_state())
        .manage(new_renderer_activity_state())
        .manage(native_intents::new_native_intake_state())
        .manage(new_backend_state())
        .manage(process::new_shutdown_flag())
        .manage(update::new_update_state())
        .manage(desktop_updater::new_desktop_update_state())
        .manage(new_close_to_tray_state())
        .manage(native_file_dialogs::ChatImportRegistry::default())
        .manage(native_file_dialogs::NativeSaveRegistry::default())
        .manage(browser_webview::new_browser_views())
        .manage(browser_downloads::new_browser_downloads())
        .invoke_handler(tauri::generate_handler![
            app_menu::set_app_menu_actions,
            browser_webview::browser_view_supported,
            browser_webview::browser_view_show,
            browser_webview::browser_view_navigate,
            browser_webview::browser_view_action,
            browser_webview::browser_view_zoom,
            browser_webview::browser_view_find,
            browser_webview::browser_view_annotate,
            browser_webview::browser_view_close,
            browser_webview::browser_view_clear_data,
            browser_webview::browser_view_mute,
            browser_capture::browser_capture,
            browser_downloads::browser_download_save,
            browser_downloads::browser_download_reveal,
            browser_downloads::browser_download_open,
            browser_downloads::browser_download_exists,
            browser_downloads::browser_download_forget,
            browser_downloads::browser_download_decide,
            browser_downloads::browser_download_folder,
            browser_downloads::browser_download_folder_pick,
            browser_downloads::browser_download_folder_reset,
            browser_capture::browser_view_print,
            set_training_active,
            set_renderer_activity,
            app_layout::has_initialized_app_window_layout,
            app_layout::mark_app_window_layout_initialized,
            app_layout::reset_app_window_layout_initialized,
            app_layout::take_native_layout_restored,
            commands::check_install_status,
            commands::desktop_preflight,
            commands::start_install,
            commands::start_server,
            commands::start_managed_server,
            commands::stop_server,
            commands::check_health,
            commands::check_backend_present,
            commands::check_backend_is_gone,
            commands::get_server_logs,
            commands::open_logs_dir,
            commands::open_models_dir,
            commands::confirm_backend_update,
            commands::start_backend_update,
            commands::start_managed_repair,
            commands::native_path_leases_usable,
            commands::cancel_pending_elevation,
            commands::install_system_packages,
            desktop_auth::desktop_auth,
            desktop_update_policy::check_desktop_manual_update,
            desktop_update_policy::desktop_update_policy,
            desktop_updater::check_desktop_update,
            desktop_updater::download_desktop_update,
            desktop_updater::install_desktop_update,
            desktop_updater::desktop_update_bundle_status,
            desktop_updater::desktop_update_cleanup_armed,
            desktop_updater::resume_desktop_update_cleanup,
            diagnostics::collect_support_diagnostics,
            native_clipboard::read_native_clipboard_files,
            native_clipboard::read_native_clipboard_png,
            native_file_dialogs::save_native_file,
            native_file_dialogs::begin_native_file_save,
            native_file_dialogs::append_native_file_save_chunk,
            native_file_dialogs::finish_native_file_save,
            native_file_dialogs::cancel_native_file_save,
            native_file_dialogs::save_native_file_from_url,
            native_file_dialogs::download_logs_to_downloads,
            native_file_dialogs::pick_native_chat_import,
            native_file_dialogs::read_native_chat_import_chunk,
            native_file_dialogs::pick_native_training_config,
            native_intents::drain_native_intents,
            native_intents::register_native_model_path,
            native_intents::register_native_attachment_path,
            native_intents::register_native_dataset_path,
            native_intents::read_native_attachment_file,
            native_intents::pick_native_model,
            native_intents::pick_hugging_face_cache_dir,
            native_intents::pick_native_document_folder,
            native_intents::consume_native_path_token,
            native_intents::register_artifact_path,
            native_intents::reveal_path_token,
            native_intents::open_path_token,
            webview_permissions::reset_microphone_permission,
            has_saved_window_state,
            was_launched_hidden,
            mark_in_app_relaunch,
            clear_in_app_relaunch,
            reveal_main_window,
            get_close_to_tray,
            set_close_to_tray,
            get_launch_at_login,
            set_launch_at_login,
            set_tray_server_status,
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

            initialize_close_to_tray(app.handle());
            reconcile_autostart_entry(app.handle());
            if let Err(error) = process::with_studio_runtime_launch_guard(|| {
                staged_update::reconcile_legacy_at_launch(&diagnostics::studio_dir());
                Ok(())
            }) {
                warn!("Legacy staged update cleanup deferred: {error}");
            }
            if let Err(error) = desktop_backend_owner::ensure_installed_studio_root_id() {
                warn!("Desktop backend ownership id unavailable: {error}");
            }
            #[cfg(any(target_os = "linux", all(debug_assertions, windows)))]
            {
                use tauri_plugin_deep_link::DeepLinkExt;
                if let Err(error) = app.deep_link().register_all() {
                    warn!("Failed to register deep-link handlers: {error}");
                }
            }
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            setup_custom_titlebar(app)?;
            #[cfg(target_os = "linux")]
            setup_linux_media_permissions(app)?;
            #[cfg(target_os = "linux")]
            keep_resizable_across_first_configure(app)?;
            #[cfg(all(windows, not(debug_assertions)))]
            setup_windows_browser_guards(app)?;
            #[cfg(target_os = "macos")]
            {
                setup_quit_menu(app)?;
                setup_terminate_interception(app);
            }
            setup_tray(app)?;
            #[cfg(unix)]
            setup_unix_termination_signals(app)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if window.label() == "main" {
                match event {
                    tauri::WindowEvent::Focused(focused) => {
                        browser_webview::window_changed(window.app_handle(), Some(*focused))
                    }
                    tauri::WindowEvent::Resized(_) => {
                        browser_webview::window_changed(window.app_handle(), None)
                    }
                    _ => {}
                }
            }
            // Record real drops in Rust, so the renderer can only register paths the OS handed over.
            if let tauri::WindowEvent::DragDrop(tauri::DragDropEvent::Drop { paths, .. }) = event {
                window
                    .state::<native_intents::NativeIntakeState>()
                    .note_dropped_paths(paths);
            }
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                // Never close directly: the only window, so closing exits before the reap.
                api.prevent_close();
                let close_to_tray = window.state::<CloseToTrayState>().0.load(Ordering::SeqCst);
                match main_window_close_action(close_to_tray) {
                    MainWindowCloseAction::Hide => {
                        let _ = window.hide();
                        browser_webview::window_changed(window.app_handle(), None);
                    }
                    MainWindowCloseAction::Quit => request_quit(window.app_handle()),
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
                // Safety net for framework-driven exits; blocks the event loop until another cleanup path
                // finishes (about 15s worst case on Unix, 18s on Windows).
                #[cfg(target_os = "macos")]
                macos_tray::remove_appearance_observer();

                cleanup_child_processes(app);
            }
            _ => {}
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_file_rotates_mid_session_and_keeps_one_generation() {
        let directory = tempfile::tempdir().unwrap();
        let log_path = directory.path().join("tauri.log");
        let rotated_path = directory.path().join("tauri.log.1");

        let mut file = RotatingLogFile::open(log_path.clone(), rotated_path.clone(), 8).unwrap();
        file.write_all(b"aaaaaaaaaa").unwrap();
        file.flush().unwrap();
        // The threshold is checked before a write, so the crossing line is written whole.
        assert!(!rotated_path.exists());

        file.write_all(b"bbbb").unwrap();
        file.flush().unwrap();
        assert_eq!(fs::read_to_string(&rotated_path).unwrap(), "aaaaaaaaaa");
        assert_eq!(fs::read_to_string(&log_path).unwrap(), "bbbb");

        file.write_all(b"cccccccccc").unwrap();
        file.write_all(b"dddd").unwrap();
        file.flush().unwrap();
        assert_eq!(fs::read_to_string(&rotated_path).unwrap(), "bbbbcccccccccc");
        assert_eq!(fs::read_to_string(&log_path).unwrap(), "dddd");
    }

    #[test]
    fn connect_sources_appended_after_existing_directive() {
        let mut policy = "default-src 'self'; connect-src 'self' https://huggingface.co; img-src 'self' data:".to_string();
        assert!(append_connect_sources(
            &mut policy,
            &["https://hf-mirror.com".to_string(), "https://ds.example.com".to_string()],
        ));
        assert_eq!(
            policy,
            "default-src 'self'; connect-src 'self' https://huggingface.co https://hf-mirror.com https://ds.example.com; img-src 'self' data:",
        );
    }

    #[test]
    fn already_listed_sources_are_not_duplicated() {
        let mut policy = "connect-src 'self' https://huggingface.co".to_string();
        assert!(append_connect_sources(&mut policy, &["https://huggingface.co".to_string()]));
        assert_eq!(policy, "connect-src 'self' https://huggingface.co");
    }

    #[test]
    fn policy_without_connect_src_is_left_untouched() {
        let mut policy = "default-src 'self'; img-src 'self' data:".to_string();
        assert!(!append_connect_sources(&mut policy, &["https://hf-mirror.com".to_string()]));
        assert_eq!(policy, "default-src 'self'; img-src 'self' data:");
    }

    #[test]
    fn hostname_containing_directive_name_cannot_match() {
        let mut policy = "connect-src 'self'; img-src connect-src.evil.com".to_string();
        assert!(append_connect_sources(&mut policy, &["https://hf-mirror.com".to_string()]));
        assert_eq!(policy, "connect-src 'self' https://hf-mirror.com; img-src connect-src.evil.com");
    }

    #[test]
    fn a_path_prefixed_mirror_is_reduced_to_its_origin() {
        assert_eq!(csp_origin_of("https://hub.internal/hf"), "https://hub.internal");
        assert_eq!(
            csp_origin_of("https://hub.internal:8443/hf/v2"),
            "https://hub.internal:8443"
        );
        assert_eq!(csp_origin_of("https://hf-mirror.com"), "https://hf-mirror.com");
        assert_eq!(csp_origin_of("http://localhost:8080"), "http://localhost:8080");
    }

    #[test]
    fn a_plain_https_origin_is_a_usable_csp_source() {
        assert!(is_usable_csp_source("https://hf-mirror.com"));
        assert!(is_usable_csp_source("http://localhost:8080"));
        assert!(is_usable_csp_source("https://hub.internal:8443/hf"));
    }

    #[test]
    fn an_endpoint_that_could_forge_a_directive_is_rejected() {
        for hostile in [
            "https://hf-mirror.com; script-src *",
            "https://hf-mirror.com *",
            "https://hf-mirror.com\nscript-src *",
            "https://hf-mirror.com\r\nscript-src *",
            "https://hf-mirror.com\tfoo",
            "https://hf-mirror.com,https://evil.com",
            "https://hf-mirror.com'",
        ] {
            assert!(!is_usable_csp_source(hostile), "should reject {hostile:?}");
        }
    }

    #[test]
    fn a_non_http_or_hostless_endpoint_is_rejected() {
        for bad in [
            "javascript:alert(1)",
            "file:///etc/passwd",
            "data:text/html,x",
            "ftp://hf-mirror.com",
            "https://",
            "https://user:pass@hf-mirror.com",
            "https://hf-mirror.com?x=1",
            "https://hf-mirror.com#f",
            "https://javascript:alert(1)",
            "https://hf-mirror.com:",
            "https://hf-mirror.com:80x",
            "https://*",
            "https://*.evil.com",
        ] {
            assert!(!is_usable_csp_source(bad), "should reject {bad:?}");
        }
    }

    #[test]
    fn plain_http_is_loopback_only() {
        assert!(is_usable_csp_source("http://127.0.0.1:9700"));
        assert!(is_usable_csp_source("http://localhost:8080"));
        assert!(is_usable_csp_source("http://127.1.2.3"));
        assert!(!is_usable_csp_source("http://192.168.1.10:8080"));
        assert!(!is_usable_csp_source("http://hf-mirror.com"));
        assert!(!is_usable_csp_source("http://10.0.0.5:8080"));
        assert!(is_usable_csp_source("https://192.168.1.10:8080"));
        assert!(is_usable_csp_source("https://hf-mirror.com"));
    }

    #[test]
    fn csp_sources_are_normalised_and_deduplicated() {
        let sources = csp_sources_from(vec![
            "https://hf-mirror.com".to_string(),
            "HTTPS://hf-mirror.com/".to_string(),
            "hf-mirror.com".to_string(),
            "  ".to_string(),
            "http://192.168.1.10".to_string(),
            "https://ds.internal/hf".to_string(),
        ]);
        assert_eq!(sources, vec!["https://hf-mirror.com", "https://ds.internal"]);
    }

    #[test]
    fn a_unicode_host_is_refused_and_its_punycode_form_is_not() {
        assert!(!is_usable_csp_source("https://例子.测试"));
        assert!(is_usable_csp_source("https://xn--fsqu00a.xn--0zwm56d"));
    }

    #[test]
    fn every_ipv6_loopback_spelling_is_recognised_and_compressed() {
        assert!(is_usable_csp_source("http://[0:0:0:0:0:0:0:1]:9700"));
        assert!(is_usable_csp_source("http://[::1]:9700"));
        assert!(is_usable_csp_source("http://127.9.9.9"));
        assert!(!is_usable_csp_source("http://[2001:db8::1]:9700"));
        assert_eq!(
            csp_origin_of("http://[0:0:0:0:0:0:0:1]:9700"),
            "http://[::1]:9700"
        );
        assert_eq!(csp_origin_of("http://[0:0:0:0:0:0:0:1]"), "http://[::1]");
    }

    #[test]
    fn the_scheme_is_matched_case_insensitively() {
        assert!(is_usable_csp_source("HTTPS://hf-mirror.com"));
        assert!(is_usable_csp_source("Https://hf-mirror.com"));
        assert!(is_usable_csp_source("HTTP://127.0.0.1:9700"));
        assert!(!is_usable_csp_source("HTTP://hf-mirror.com"));
        assert_eq!(csp_origin_of("HTTPS://hf-mirror.com/hf"), "https://hf-mirror.com");
    }

    #[test]
    fn a_loopback_http_mirror_reaches_the_asset_directives_too() {
        let mut policy = "connect-src 'self'; img-src 'self' data: https:; \
media-src 'self' https:"
            .to_string();
        let assets = ["http://127.0.0.1:9700".to_string()];
        assert!(append_sources_to(&mut policy, "img-src", &assets));
        assert!(append_sources_to(&mut policy, "media-src", &assets));
        assert!(policy.contains("img-src 'self' data: https: http://127.0.0.1:9700"));
        assert!(policy.contains("media-src 'self' https: http://127.0.0.1:9700"));
        assert!(policy.contains("connect-src 'self';"));
    }

    #[test]
    fn an_oversized_log_from_a_previous_session_rotates_on_the_first_write() {
        let directory = tempfile::tempdir().unwrap();
        let log_path = directory.path().join("tauri.log");
        let rotated_path = directory.path().join("tauri.log.1");
        fs::write(&log_path, b"stale-and-oversized").unwrap();

        let mut file = RotatingLogFile::open(log_path.clone(), rotated_path.clone(), 8).unwrap();
        file.write_all(b"fresh").unwrap();
        file.flush().unwrap();

        assert_eq!(
            fs::read_to_string(&rotated_path).unwrap(),
            "stale-and-oversized"
        );
        assert_eq!(fs::read_to_string(&log_path).unwrap(), "fresh");
    }

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

    #[test]
    fn close_to_tray_preference_defaults_off_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();

        assert!(!read_close_to_tray_preference(dir.path()));
        write_close_to_tray_preference(dir.path(), false).unwrap();
        assert!(!read_close_to_tray_preference(dir.path()));
        write_close_to_tray_preference(dir.path(), true).unwrap();
        assert!(read_close_to_tray_preference(dir.path()));
    }

    #[test]
    fn invalid_close_to_tray_preference_falls_back_to_disabled() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(close_to_tray_preference_path(dir.path()), b"maybe\n").unwrap();

        assert!(!read_close_to_tray_preference(dir.path()));
    }

    #[test]
    fn launch_at_login_preference_round_trips_and_starts_with_no_record() {
        let dir = tempfile::tempdir().unwrap();

        assert_eq!(read_launch_at_login_preference(dir.path()), None);
        write_launch_at_login_preference(dir.path(), true).unwrap();
        assert_eq!(read_launch_at_login_preference(dir.path()), Some(true));
        write_launch_at_login_preference(dir.path(), false).unwrap();
        assert_eq!(read_launch_at_login_preference(dir.path()), Some(false));
    }

    #[test]
    fn an_unreadable_launch_at_login_preference_restores_nothing() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(launch_at_login_preference_path(dir.path()), b"maybe\n").unwrap();

        assert_eq!(read_launch_at_login_preference(dir.path()), None);
        assert!(!should_restore_autostart_entry(None, false, false));
    }

    #[test]
    fn only_a_deleted_entry_with_a_stored_yes_is_restored() {
        assert!(should_restore_autostart_entry(Some(true), false, false));

        assert!(!should_restore_autostart_entry(Some(true), true, true));
        assert!(!should_restore_autostart_entry(Some(true), false, true));
        assert!(!should_restore_autostart_entry(None, false, false));
        assert!(!should_restore_autostart_entry(Some(false), false, false));
        assert!(!should_restore_autostart_entry(Some(true), true, false));
    }

    #[test]
    fn startup_approved_bytes_are_read_from_the_state_byte() {
        assert!(!startup_approved_disabled(&[
            0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]));
        assert!(startup_approved_disabled(&[
            0x03, 0, 0, 0, 0x1e, 0x38, 0x9f, 0x4c, 0x7d, 0x2a, 0xdb, 0x01
        ]));
        assert!(!startup_approved_disabled(&[
            0x06, 0, 0, 0, 0x1e, 0x38, 0x9f, 0x4c, 0x7d, 0x2a, 0xdb, 0x01
        ]));
        assert!(!startup_approved_disabled(&[0x02, 0, 0, 0]));
        assert!(!startup_approved_disabled(&[]));
    }

    #[test]
    fn a_re_enabled_entry_is_not_treated_as_deleted() {
        let re_enabled = [
            0x06, 0, 0, 0, 0x1e, 0x38, 0x9f, 0x4c, 0x7d, 0x2a, 0xdb, 0x01,
        ];
        let disabled = startup_approved_disabled(&re_enabled);

        assert!(!disabled);
        assert!(!should_restore_autostart_entry(Some(true), true, disabled));
    }

    #[test]
    fn enabled_close_to_tray_hides_the_main_window_on_supported_desktops() {
        let expected = if cfg!(any(
            target_os = "windows",
            target_os = "linux",
            target_os = "macos"
        )) {
            MainWindowCloseAction::Hide
        } else {
            MainWindowCloseAction::Quit
        };
        assert_eq!(main_window_close_action(true), expected);

        let disabled_expected = if cfg!(target_os = "macos") {
            MainWindowCloseAction::Hide
        } else {
            MainWindowCloseAction::Quit
        };
        assert_eq!(main_window_close_action(false), disabled_expected);
    }

    #[cfg(target_os = "linux")]
    const BID: &str = "ai.unsloth.studio";

    // Only XDG_DATA_HOME is swapped, under the crate-wide env lock; HOME is left alone.
    #[cfg(target_os = "linux")]
    fn with_xdg_data_home<T>(value: &str, f: impl FnOnce() -> T) -> T {
        // The crate-wide lock: the path policy reads XDG_DATA_HOME when choosing test dirs.
        let _guard = crate::native_path_policy::PROCESS_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
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
        // dirs drops a relative XDG_DATA_HOME; following one would rm -rf under the working directory.
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
        // A plain file fails remove_dir_all with something other than NotFound, like a locked dir.
        fs::write(root.join("WebKitCache"), b"still open").unwrap();

        let lock = with_xdg_data_home(xdg, || clear_webview_caches(BID, "1.0.0"));
        assert!(
            !root.join("CacheStorage").exists(),
            "the deletable cache stayed"
        );
        assert!(
            !root.join(".webview-cache-cleared").exists(),
            "stamping a partial clear makes every later launch skip the retry"
        );
        drop(lock);

        fs::remove_file(root.join("WebKitCache")).unwrap();
        fs::create_dir_all(root.join("WebKitCache")).unwrap();
        let lock = with_xdg_data_home(xdg, || clear_webview_caches(BID, "1.0.0"));
        assert!(!root.join("WebKitCache").exists(), "the retry did not run");
        assert!(
            root.join(".webview-cache-cleared").exists(),
            "the retry did not stamp"
        );
        drop(lock);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn a_live_instance_lock_excludes_a_duplicate_launch() {
        let dir = tempfile::tempdir().unwrap();
        let xdg = dir.path().to_str().unwrap();
        let root = dir.path().join(BID);
        fs::create_dir_all(root.join("WebKitCache")).unwrap();

        let live = with_xdg_data_home(xdg, || clear_webview_caches(BID, "1.0.0"));
        assert!(live.is_some(), "the clearing instance must get the lock");
        assert!(
            !root.join("WebKitCache").exists(),
            "the first launch did not clear"
        );

        fs::create_dir_all(root.join("WebKitCache")).unwrap();
        let second = with_xdg_data_home(xdg, || clear_webview_caches(BID, "1.0.1"));
        assert!(second.is_none(), "a duplicate launch took the lock");
        assert!(
            root.join("WebKitCache").exists(),
            "deleted a live instance's cache"
        );

        drop(live);

        fs::create_dir_all(root.join("WebKitCache")).unwrap();
        let stamped = with_xdg_data_home(xdg, || clear_webview_caches(BID, "1.0.0"));
        assert!(
            stamped.is_some(),
            "a stamped launch dropped the profile lock"
        );
        assert!(
            root.join("WebKitCache").exists(),
            "a stamped launch cleared anyway"
        );
        let racer = with_xdg_data_home(xdg, || clear_webview_caches(BID, "1.0.1"));
        assert!(
            racer.is_none(),
            "a newer launch took the lock from a stamped instance"
        );
        assert!(
            root.join("WebKitCache").exists(),
            "deleted a stamped instance's cache"
        );
        drop(stamped);

        let next = with_xdg_data_home(xdg, || clear_webview_caches(BID, "1.0.1"));
        assert!(
            !root.join("WebKitCache").exists(),
            "a released lock still blocked the clear"
        );
        drop(next);
    }

    // One test, not three: the quit coordinator is process-global and cargo tests run in parallel.
    #[test]
    fn quit_guard_admits_one_quit_and_always_re_arms() {
        assert!(!lock_quit_confirmation_state().in_progress);

        #[cfg(target_os = "macos")]
        {
            assert!(matches!(
                begin_or_attach_termination(false),
                TerminationConfirmation::Now
            ));
            assert!(!take_pending_termination_reply());
        }

        {
            let first = begin_quit().expect("the first quit must be admitted");
            assert!(
                begin_quit().is_none(),
                "a repeat close must not stack a second confirmation"
            );

            #[cfg(target_os = "macos")]
            {
                assert!(matches!(
                    begin_or_attach_termination(false),
                    TerminationConfirmation::Attached
                ));
                assert!(matches!(
                    begin_or_attach_termination(true),
                    TerminationConfirmation::Duplicate
                ));
                assert!(
                    first.finish(),
                    "finishing must atomically take the attached AppKit reply"
                );
                assert!(!take_pending_termination_reply());
            }

            #[cfg(not(target_os = "macos"))]
            assert!(!first.finish());
        }

        #[cfg(target_os = "macos")]
        {
            let guard = match begin_or_attach_termination(true) {
                TerminationConfirmation::Start(guard) => guard,
                _ => panic!("an idle active state must start its own confirmation"),
            };
            assert!(guard.finish());
            assert!(!lock_quit_confirmation_state().in_progress);
        }

        {
            let _second = begin_quit().expect("cancelling must re-arm the close button");
        }

        let unwound = std::panic::catch_unwind(|| {
            let _guard = begin_quit().expect("admitted");
            #[cfg(target_os = "macos")]
            assert!(matches!(
                begin_or_attach_termination(true),
                TerminationConfirmation::Attached
            ));
            panic!("quit thread unwound");
        });
        assert!(unwound.is_err());
        #[cfg(target_os = "macos")]
        assert!(
            !take_pending_termination_reply(),
            "unwinding must not leave a stale duplicate request"
        );
        assert!(
            begin_quit().is_some(),
            "a panicking quit must release the guard"
        );
    }

    #[test]
    fn a_quit_that_reaches_exit_covers_the_reap() {
        let events = std::cell::RefCell::new(Vec::new());

        let quitting = quit_sequence(
            || true,
            || true,
            || events.borrow_mut().push("reap".to_string()),
            |event| events.borrow_mut().push(event.to_string()),
        );

        assert!(quitting);
        assert_eq!(events.into_inner(), ["app-closing", "reap"]);
    }

    #[test]
    fn a_quit_with_nothing_to_cover_reaps_without_asking_for_an_overlay() {
        let events = std::cell::RefCell::new(Vec::new());

        let quitting = quit_sequence(
            || true,
            || false,
            || events.borrow_mut().push("reap".to_string()),
            |event| events.borrow_mut().push(event.to_string()),
        );

        assert!(
            quitting,
            "the overlay is presentation, not part of quitting"
        );
        assert_eq!(
            events.into_inner(),
            ["reap"],
            "window-close and tray paths can bypass the overlay entirely, and a tray quit with no \
             window on screen must not emit it"
        );
    }

    #[test]
    fn a_declined_quit_never_raises_the_overlay() {
        let events = std::cell::RefCell::new(Vec::new());

        let quitting = quit_sequence(
            || false,
            || true,
            || events.borrow_mut().push("reap".to_string()),
            |event| events.borrow_mut().push(event.to_string()),
        );

        assert!(!quitting, "a declined confirmation must not reach the exit");
        assert!(
            events.into_inner().is_empty(),
            "the confirmations are dialogs, and an overlay behind one would claim the app \
             is closing while it asks whether to keep going"
        );
    }

    #[test]
    fn a_declined_quit_never_asks_whether_the_window_is_visible() {
        let asked = std::cell::Cell::new(false);

        let quitting = quit_sequence(
            || false,
            || {
                asked.set(true);
                true
            },
            || panic!("a declined quit must not reap"),
            |_| panic!("a declined quit must not emit"),
        );

        assert!(!quitting);
        assert!(
            !asked.get(),
            "the window question is only worth asking once the quit is committed"
        );
    }

    #[test]
    fn a_panicking_reap_takes_the_overlay_back_down() {
        let events = std::cell::RefCell::new(Vec::new());

        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            quit_sequence(
                || true,
                || true,
                || panic!("the reap panicked"),
                |event| events.borrow_mut().push(event.to_string()),
            )
        }));

        assert!(unwound.is_err());
        assert_eq!(
            events.into_inner(),
            ["app-closing", "app-closing-cancelled"],
            "a panicking reap leaves the app up, so the overlay cannot cover it"
        );
    }

    #[test]
    fn a_panicking_reap_with_nothing_to_cover_stays_silent() {
        let events = std::cell::RefCell::new(Vec::new());

        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            quit_sequence(
                || true,
                || false,
                || panic!("the reap panicked"),
                |event| events.borrow_mut().push(event.to_string()),
            )
        }));

        assert!(unwound.is_err());
        assert!(
            events.into_inner().is_empty(),
            "nothing was raised, so the unwind has nothing to retract: a cancel here would \
             be the guard half-armed"
        );
    }

    #[test]
    fn a_quit_with_the_window_hidden_raises_no_overlay() {
        assert!(
            !quit_raises_the_overlay(true, || Some(Ok(false))),
            "there is no frozen window to explain when no window is on screen"
        );
    }

    #[test]
    fn a_minimized_window_still_gets_the_overlay() {
        assert!(quit_raises_the_overlay(true, || Some(Ok(true))));
    }

    #[test]
    fn a_quit_with_no_main_window_at_all_raises_no_overlay() {
        assert!(
            !quit_raises_the_overlay(true, || None),
            "a window that does not exist cannot be looking frozen"
        );
    }

    #[test]
    fn an_unreadable_window_visibility_still_gets_the_overlay() {
        assert!(
            quit_raises_the_overlay(true, || Some(Err("no window handle".to_string()))),
            "an emit into a hidden window costs nothing, and an unexplained frozen window \
             is the failure this exists to fix"
        );
    }

    #[test]
    fn the_window_visibility_is_never_read_off_windows() {
        let asked = std::cell::Cell::new(false);

        let raised = quit_raises_the_overlay(false, || {
            asked.set(true);
            Some(Ok(true))
        });

        assert!(!raised, "only Windows shows the freeze this covers");
        assert!(
            !asked.get(),
            "the getter blocks on a round trip to the event loop, and nothing off Windows \
             raises an overlay to spend it on"
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
        let entry =
            "[Desktop Entry]\nExec=/home/n/My Apps/Unsloth.AppImage --hidden\nTerminal=false";
        let hardened = hardened_autostart_entry(entry).expect("guard must be added");
        assert!(hardened.contains("Exec=\"/home/n/My Apps/Unsloth.AppImage\" --hidden\n"));
        assert!(hardened.ends_with("\nTryExec=/home/n/My Apps/Unsloth.AppImage"));
    }

    #[test]
    fn autostart_hardening_escapes_exec_reserved_characters() {
        let entry = "[Desktop Entry]\nExec=/opt/a\"b$c/app --hidden";
        let hardened = hardened_autostart_entry(entry).expect("guard must be added");
        assert!(hardened.contains(r#"Exec="/opt/a\\"b\\$c/app" --hidden"#));
    }

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
        assert!(hardened.ends_with("\nTryExec=/home/n/Unsloth%20Studio.AppImage"));
    }

    #[test]
    fn autostart_disabled_markers_are_detected() {
        assert!(autostart_entry_disabled("[Desktop Entry]\nHidden=true"));
        assert!(autostart_entry_disabled(
            "[Desktop Entry]\nX-GNOME-Autostart-enabled=false"
        ));
        assert!(!autostart_entry_disabled(
            "[Desktop Entry]\nExec=/a --hidden"
        ));
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
        let quoted = quoted_windows_run_command(
            r"C:\Users\Jane Doe\AppData\Local\Unsloth\Unsloth.exe --hidden",
        );
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

    // One element per field of RendererActivity, so no kind ships untested.
    fn renderer_activity(state: &RendererActivityState) -> (bool, bool, bool) {
        let activity = state
            .lock()
            .expect("the activity mutex must not be poisoned");
        (
            activity.downloads,
            activity.shell_update,
            activity.unsaved_transcript,
        )
    }

    #[test]
    fn renderer_activity_starts_clear_and_round_trips_each_kind() {
        let state = new_renderer_activity_state();
        assert_eq!(renderer_activity(&state), (false, false, false));

        apply_renderer_activity(&state, "downloads", true);
        assert_eq!(renderer_activity(&state), (true, false, false));
        apply_renderer_activity(&state, "downloads", false);
        assert_eq!(renderer_activity(&state), (false, false, false));

        apply_renderer_activity(&state, "shell_update", true);
        assert_eq!(renderer_activity(&state), (false, true, false));
        apply_renderer_activity(&state, "shell_update", false);
        assert_eq!(renderer_activity(&state), (false, false, false));

        apply_renderer_activity(&state, "unsaved_transcript", true);
        assert_eq!(renderer_activity(&state), (false, false, true));
        apply_renderer_activity(&state, "unsaved_transcript", false);
        assert_eq!(renderer_activity(&state), (false, false, false));
    }

    #[test]
    fn renderer_activity_kinds_are_independent() {
        let state = new_renderer_activity_state();

        apply_renderer_activity(&state, "downloads", true);
        apply_renderer_activity(&state, "shell_update", true);
        apply_renderer_activity(&state, "unsaved_transcript", true);
        assert_eq!(renderer_activity(&state), (true, true, true));

        apply_renderer_activity(&state, "downloads", false);
        assert_eq!(renderer_activity(&state), (false, true, true));

        apply_renderer_activity(&state, "unsaved_transcript", false);
        assert_eq!(renderer_activity(&state), (false, true, false));
    }

    #[test]
    fn renderer_activity_ignores_an_unknown_kind() {
        let state = new_renderer_activity_state();
        apply_renderer_activity(&state, "shell_update", true);

        apply_renderer_activity(&state, "training", true);
        apply_renderer_activity(&state, "", false);
        apply_renderer_activity(&state, "Downloads", true);
        apply_renderer_activity(&state, "unsaved-transcript", true);
        assert_eq!(renderer_activity(&state), (false, true, false));
    }

    #[test]
    fn the_tray_toggle_names_the_action_a_click_would_take() {
        assert_eq!(tray_toggle_label("running"), ("Stop Server", true));
        assert_eq!(tray_toggle_label("stopped"), ("Start Server", true));
        assert_eq!(tray_toggle_label("error"), ("Start Server", true));
        assert_eq!(tray_toggle_label("starting"), ("Starting\u{2026}", false));
    }

    /// Keep this list in step with the BackendStatus union in use-tauri-backend.ts.
    #[test]
    fn a_status_the_tray_cannot_act_on_greys_the_toggle() {
        for status in [
            "checking",
            "not-installed",
            "installing",
            "install-error",
            "needs-elevation",
            "repairing",
            "repair-error",
        ] {
            assert_eq!(
                tray_toggle_label(status),
                ("Start Server", false),
                "{status} offered a click the renderer would drop"
            );
        }
    }

    /// Match the whole string: a prefix or case fold would let "run" offer to stop a stopped server.
    #[test]
    fn an_unrecognised_status_greys_the_toggle_rather_than_guessing() {
        for status in ["", " ", "Running", "RUNNING", "running ", "run", "{}"] {
            assert_eq!(
                tray_toggle_label(status),
                ("Start Server", false),
                "{status:?} was read as a known status"
            );
        }
    }
}
