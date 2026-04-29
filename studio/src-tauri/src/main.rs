#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod desktop_auth;
mod install;
mod preflight;
mod process;
mod update;
mod windows_job;

use log::info;
use process::new_backend_state;
use simplelog::{
    CombinedLogger, Config, LevelFilter, SharedLogger, TermLogger, TerminalMode, WriteLogger,
};
use std::fs;
use tauri::menu::{MenuBuilder, MenuItemBuilder};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{Emitter, Manager};

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

fn setup_tray(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let open = MenuItemBuilder::with_id("open", "Open Studio").build(app)?;
    let toggle = MenuItemBuilder::with_id("toggle", "Start/Stop Server").build(app)?;
    let quit = MenuItemBuilder::with_id("quit", "Quit").build(app)?;
    let menu = MenuBuilder::new(app)
        .items(&[&open, &toggle, &quit])
        .build()?;

    TrayIconBuilder::new()
        .menu(&menu)
        .tooltip("Unsloth Studio (Desktop)")
        .icon(app.default_window_icon().unwrap().clone())
        .on_menu_event(move |app, event| match event.id().as_ref() {
            "open" => {
                if let Some(window) = app.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
            "toggle" => {
                let _ = app.emit("tray-toggle-server", ());
            }
            "quit" => {
                let install_state = app.state::<crate::install::InstallState>();
                let _ = crate::install::stop_install(&install_state);
                let update_state = app.state::<crate::update::UpdateState>();
                let _ = crate::update::stop_update(&update_state);
                // Detach the 5s graceful-wait so the tray click does not
                // block the Tauri main loop. Exit runs the RunEvent::Exit
                // safety net which also calls stop_backend synchronously.
                let shutdown = app.state::<crate::process::ShutdownFlag>().inner().clone();
                let backend_state = app.state::<crate::process::BackendState>().inner().clone();
                crate::process::stop_backend_detached(backend_state, shutdown);
                app.exit(0);
            }
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                if let Some(window) = tray.app_handle().get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
        })
        .build(app)?;

    Ok(())
}

fn main() {
    // Fix PATH for GUI apps (macOS .app bundles, Linux AppImage, Windows)
    // GUI apps don't inherit shell dotfile PATH — this spawns the user's
    // login shell to source .zshrc/.bashrc/.profile and sets PATH properly.
    let _ = fix_path_env::fix();

    setup_logging();
    info!("Unsloth Studio desktop app starting");
    windows_job::initialize();

    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(install::new_install_state())
        .manage(new_backend_state())
        .manage(process::new_shutdown_flag())
        .manage(update::new_update_state())
        .invoke_handler(tauri::generate_handler![
            commands::check_install_status,
            commands::desktop_preflight,
            commands::start_install,
            commands::start_server,
            commands::start_managed_server,
            commands::stop_server,
            commands::check_health,
            commands::get_server_logs,
            commands::open_logs_dir,
            commands::start_backend_update,
            commands::start_managed_repair,
            commands::install_system_packages,
            desktop_auth::desktop_auth,
        ])
        .setup(|app| {
            setup_tray(app)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                // Hide window instead of closing — this is a tray app.
                // Processes keep running so the backend stays available.
                // Full cleanup happens via:
                //   - Tray "Quit" menu item (explicit user action)
                //   - RunEvent::Exit handler (OS shutdown, SIGTERM, etc.)
                let _ = window.hide();
                api.prevent_close();
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                // Cleanup on ALL exit paths — safety net for non-tray exits
                if let Some(install_state) = app.try_state::<install::InstallState>() {
                    let _ = install::stop_install(&install_state);
                }
                if let Some(update_state) = app.try_state::<update::UpdateState>() {
                    let _ = update::stop_update(&update_state);
                }
                if let Some(backend_state) = app.try_state::<process::BackendState>() {
                    let shutdown = app
                        .try_state::<process::ShutdownFlag>()
                        .expect("ShutdownFlag must be managed");
                    let _ = process::stop_backend(&backend_state, &shutdown);
                }
            }
        });
}
