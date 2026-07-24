#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app_layout;
mod commands;
mod desktop_auth;
mod desktop_backend_owner;
mod desktop_update_policy;
mod diagnostics;
mod install;
mod native_backend_lease;
mod native_file_dialogs;
mod native_intents;
mod native_path_policy;
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
use tauri_plugin_window_state::{AppHandleExt, StateFlags};

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

fn cleanup_child_processes(app: &tauri::AppHandle) {
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

fn setup_tray(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    let open = MenuItemBuilder::with_id("open", "Open Unsloth").build(app)?;
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
                // Run cleanup off the menu callback, but only exit after the
                // backend tree has been reaped. Exiting first can terminate this
                // process while a detached cleanup thread is still waiting,
                // leaving the backend orphaned.
                let app_handle = app.clone();
                std::thread::spawn(move || {
                    cleanup_child_processes(&app_handle);
                    app_handle.exit(0);
                });
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

// Clear WebView caches before the webview initializes. An in-app update runs
// setup.sh/setup.ps1 while the old WebView still holds these files (its clear
// silently fails), so without this a relaunch can serve the previous frontend
// from cache. Cache-only paths; LocalStorage, IndexedDB, cookies, and app
// data are kept. Mirrors setup.sh _clear_webview_caches / setup.ps1.
fn clear_webview_caches(bundle_id: &str) {
    use std::path::PathBuf;
    let mut paths: Vec<PathBuf> = Vec::new();
    #[cfg(target_os = "windows")]
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        if !local.is_empty() {
            let profile = PathBuf::from(local)
                .join(bundle_id)
                .join("EBWebView")
                .join("Default");
            for sub in ["Cache", "Code Cache", "GPUCache", "Service Worker"] {
                paths.push(profile.join(sub));
            }
        }
    }
    #[cfg(target_os = "macos")]
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() {
            let home = PathBuf::from(home);
            paths.push(home.join("Library/Caches").join(bundle_id));
            let data = home.join("Library/WebKit").join(bundle_id).join("WebsiteData");
            for sub in ["CacheStorage", "ServiceWorkers", "DiskCache"] {
                paths.push(data.join(sub));
            }
        }
    }
    #[cfg(target_os = "linux")]
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() {
            let home = PathBuf::from(home);
            let cache = std::env::var("XDG_CACHE_HOME")
                .ok()
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
                .unwrap_or_else(|| home.join(".cache"));
            paths.push(cache.join(bundle_id));
            // wry keys the WebKitGTK base-cache dir to the app data dir.
            let data = std::env::var("XDG_DATA_HOME")
                .ok()
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
                .unwrap_or_else(|| home.join(".local/share"))
                .join(bundle_id);
            for sub in ["WebKitCache", "CacheStorage", "serviceworkers", "ServiceWorkers"] {
                paths.push(data.join(sub));
            }
        }
    }
    for p in paths {
        let _ = fs::remove_dir_all(&p);
    }
}

fn main() {
    // Fix PATH for GUI apps (macOS .app bundles, Linux AppImage, Windows)
    // GUI apps don't inherit shell dotfile PATH — this spawns the user's
    // login shell to source .zshrc/.bashrc/.profile and sets PATH properly.
    let _ = fix_path_env::fix();

    setup_logging();
    info!("Unsloth Studio desktop app starting");
    windows_job::initialize();

    // Must run before the Builder: the config-defined window (and its
    // WebView, which locks these files) exists by the time setup hooks run.
    let context = tauri::generate_context!();
    clear_webview_caches(&context.config().identifier);

    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.set_focus();
            }
        }))
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
        .manage(native_intents::new_native_intake_state())
        .manage(new_backend_state())
        .manage(process::new_shutdown_flag())
        .manage(update::new_update_state())
        .invoke_handler(tauri::generate_handler![
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
            diagnostics::collect_support_diagnostics,
            native_file_dialogs::save_native_file,
            native_file_dialogs::pick_native_chat_import,
            native_intents::drain_native_intents,
            native_intents::register_native_model_path,
            native_intents::pick_native_model,
            native_intents::pick_hugging_face_cache_dir,
            native_intents::consume_native_path_token,
            native_intents::register_artifact_path,
            native_intents::reveal_path_token,
            native_intents::open_path_token,
            has_saved_window_state,
        ])
        .setup(|app| {
            #[cfg(any(target_os = "windows", target_os = "linux"))]
            setup_custom_titlebar(app)?;
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
        .build(context)
        .expect("error while building tauri application")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                // Cleanup on ALL exit paths — safety net for non-tray exits
                cleanup_child_processes(app);
            }
        });
}
