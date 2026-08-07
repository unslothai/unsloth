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
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;
use tauri::menu::{MenuBuilder, MenuItemBuilder};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{Emitter, Manager};
use tauri_plugin_window_state::{AppHandleExt, StateFlags};

/// Serializes the exit paths that reap the backend: `request_quit` (tray "Quit" and,
/// outside macOS, the close button), the Unix signal listener, and `RunEvent::Exit`.
/// Exactly one runs cleanup; the others block, so the process never exits mid-reap.
static TERMINATION_CLEANUP: Once = Once::new();

#[tauri::command]
fn has_saved_window_state(app: tauri::AppHandle) -> bool {
    let Ok(dir) = app.path().app_config_dir() else {
        return false;
    };
    dir.join(app.filename()).is_file()
}

/// Append target for `tauri.log` that rotates while the app runs, not only at launch.
///
/// The size check used to happen once in `setup_logging`, so a session left open for
/// days (the backend's own log lines are mirrored here as they arrive) grew the file
/// past the cap until the next restart. Tracking the byte count as we write applies the
/// same 5 MiB threshold continuously. Rotation closes the handle before renaming, which
/// Windows requires and which also means the reopened file is the one being written.
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
        // Pick up where a previous session left off, so an already-oversized file
        // rotates on its first write instead of growing for another whole session.
        self.written = file.metadata().map(|meta| meta.len()).unwrap_or(0);
        self.file = Some(file);
        Ok(())
    }

    /// Move the current log aside and start a fresh one. Best effort: if reopening
    /// fails we drop lines rather than take the app down over a log file.
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
            if let Ok(file) = RotatingLogFile::open(log_path, rotated_path, max_log_bytes) {
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

    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            show_main_window(app);
        }))
        .plugin(tauri_plugin_deep_link::init())
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
        ])
        .setup(|app| {
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
        .build(tauri::generate_context!())
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

    #[test]
    fn log_file_rotates_mid_session_and_keeps_one_generation() {
        let directory = tempfile::tempdir().unwrap();
        let log_path = directory.path().join("tauri.log");
        let rotated_path = directory.path().join("tauri.log.1");

        let mut file = RotatingLogFile::open(log_path.clone(), rotated_path.clone(), 8).unwrap();
        file.write_all(b"aaaaaaaaaa").unwrap();
        file.flush().unwrap();
        // Still the first generation: the threshold is checked before a write, so the
        // line that crosses it is written whole rather than split across two files.
        assert!(!rotated_path.exists());

        file.write_all(b"bbbb").unwrap();
        file.flush().unwrap();
        assert_eq!(fs::read_to_string(&rotated_path).unwrap(), "aaaaaaaaaa");
        assert_eq!(fs::read_to_string(&log_path).unwrap(), "bbbb");

        // A second rotation replaces .1 rather than piling up generations.
        file.write_all(b"cccccccccc").unwrap();
        file.write_all(b"dddd").unwrap();
        file.flush().unwrap();
        assert_eq!(fs::read_to_string(&rotated_path).unwrap(), "bbbbcccccccccc");
        assert_eq!(fs::read_to_string(&log_path).unwrap(), "dddd");
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

        assert_eq!(fs::read_to_string(&rotated_path).unwrap(), "stale-and-oversized");
        assert_eq!(fs::read_to_string(&log_path).unwrap(), "fresh");
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
