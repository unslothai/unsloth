#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod install;
mod process;

use log::info;
use process::new_backend_state;
use simplelog::{
    CombinedLogger, Config, LevelFilter, SharedLogger, TermLogger, TerminalMode, WriteLogger,
};
use std::fs;

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

fn main() {
    setup_logging();
    info!("Unsloth Studio desktop app starting");

    tauri::Builder::default()
        .plugin(tauri_plugin_process::init())
        .manage(new_backend_state())
        .invoke_handler(tauri::generate_handler![
            commands::check_install_status,
            commands::start_install,
            commands::start_server,
            commands::stop_server,
            commands::check_health,
            commands::find_existing_server,
            commands::get_server_logs,
            commands::get_bootstrap_password,
            commands::open_logs_dir,
        ])
        // Note: window close handler will be added in Task 5 (system tray)
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
