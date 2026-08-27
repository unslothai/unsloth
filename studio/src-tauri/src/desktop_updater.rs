use serde::Serialize;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_updater::{Update, UpdaterExt};

const DOWNLOAD_EVENT: &str = "desktop-update-download";
const DOWNLOAD_EVENT_STEP: u64 = 512 * 1024;

#[derive(Default)]
pub(crate) struct DesktopUpdate {
    update: Option<Update>,
    bundle: Option<Vec<u8>>,
    downloading: bool,
}

pub(crate) type DesktopUpdateState = Arc<Mutex<DesktopUpdate>>;

pub(crate) fn new_desktop_update_state() -> DesktopUpdateState {
    Arc::new(Mutex::new(DesktopUpdate::default()))
}

pub(crate) fn pending_version(state: &DesktopUpdateState) -> Result<Option<String>, String> {
    let guard = state.lock().map_err(|error| error.to_string())?;
    Ok(guard.update.as_ref().map(|update| update.version.clone()))
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DesktopUpdateMetadata {
    current_version: String,
    version: String,
    date: Option<String>,
    body: Option<String>,
    raw_json: serde_json::Value,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DesktopUpdateDownload {
    version: String,
    downloaded: u64,
    total: Option<u64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DesktopUpdateBundleStatus {
    version: Option<String>,
    downloaded: bool,
    downloading: bool,
}

/// Re-arm crash cleanup when the installer never took over.
///
/// `on_before_exit` clears kill-on-close assuming the installer is about to
/// replace this process; a failed or cancelled install leaves the app running
/// with no reaper for its children.
#[tauri::command]
pub(crate) async fn resume_desktop_update_cleanup() -> Result<(), String> {
    // Both halves of what the pre-exit hook consumed: the exit guard it spent, so a
    // retry actually reaps the backend again, and kill-on-close itself.
    crate::reset_termination_cleanup();
    #[cfg(windows)]
    {
        crate::windows_job::resume_after_update_installer().map_err(|error| error.to_string())?;
    }
    Ok(())
}

/// Whether crash cleanup is armed right now, for a UI that has just remounted.
#[tauri::command]
pub(crate) async fn desktop_update_cleanup_armed() -> Result<bool, String> {
    #[cfg(windows)]
    {
        crate::windows_job::kill_on_close_armed().map_err(|error| error.to_string())
    }
    #[cfg(not(windows))]
    {
        Ok(true)
    }
}

#[tauri::command]
pub(crate) async fn check_desktop_update(
    webview: tauri::Webview,
    state: tauri::State<'_, DesktopUpdateState>,
) -> Result<Option<DesktopUpdateMetadata>, String> {
    let app = webview.app_handle().clone();
    let builder = webview.updater_builder().on_before_exit(move || {
        #[cfg(windows)]
        {
            crate::cleanup_child_processes(&app);
            if let Err(error) = crate::windows_job::suspend_for_update_installer() {
                log::error!(
                    "Could not suspend Windows job cleanup for the updater; refusing to launch the installer: {error}"
                );
                std::process::exit(1);
            }
        }
        app.cleanup_before_exit();
    });

    let updater = builder.build().map_err(|error| error.to_string())?;
    let update = updater.check().await.map_err(|error| error.to_string())?;
    let mut guard = state.lock().map_err(|error| error.to_string())?;
    let Some(update) = update else {
        guard.update = None;
        guard.bundle = None;
        return Ok(None);
    };

    let date = update
        .date
        .map(|date| date.format(&time::format_description::well_known::Rfc3339))
        .transpose()
        .map_err(|error| error.to_string())?;
    let metadata = DesktopUpdateMetadata {
        current_version: update.current_version.clone(),
        version: update.version.clone(),
        date,
        body: update.body.clone(),
        raw_json: update.raw_json.clone(),
    };
    if guard.update.as_ref().map(|u| u.version.as_str()) != Some(update.version.as_str()) {
        guard.bundle = None;
    }
    guard.update = Some(update);
    Ok(Some(metadata))
}

#[tauri::command]
pub(crate) async fn download_desktop_update(
    app: AppHandle,
    state: tauri::State<'_, DesktopUpdateState>,
) -> Result<(), String> {
    let update = {
        let mut guard = state.lock().map_err(|error| error.to_string())?;
        let Some(update) = guard.update.clone() else {
            return Err("No desktop update has been checked.".to_string());
        };
        if guard.bundle.is_some() {
            return Ok(());
        }
        if guard.downloading {
            return Err("Desktop update download is already running.".to_string());
        }
        guard.downloading = true;
        update
    };

    let version = update.version.clone();
    let mut downloaded: u64 = 0;
    let mut last_emitted: u64 = 0;
    let progress_app = app.clone();
    let result = update
        .download(
            |chunk, total| {
                downloaded += chunk as u64;
                if downloaded - last_emitted >= DOWNLOAD_EVENT_STEP || Some(downloaded) == total {
                    last_emitted = downloaded;
                    let _ = progress_app.emit(
                        DOWNLOAD_EVENT,
                        DesktopUpdateDownload {
                            version: version.clone(),
                            downloaded,
                            total,
                        },
                    );
                }
            },
            || {},
        )
        .await;

    let mut guard = state.lock().map_err(|error| error.to_string())?;
    guard.downloading = false;
    match result {
        Ok(bytes) => {
            if guard.update.as_ref().map(|u| u.version.as_str()) == Some(update.version.as_str()) {
                guard.bundle = Some(bytes);
            }
            Ok(())
        }
        Err(error) => Err(error.to_string()),
    }
}

#[tauri::command]
pub(crate) async fn install_desktop_update(
    state: tauri::State<'_, DesktopUpdateState>,
) -> Result<(), String> {
    let (update, bundle) = {
        let mut guard = state.lock().map_err(|error| error.to_string())?;
        let Some(update) = guard.update.clone() else {
            return Err("No desktop update has been checked.".to_string());
        };
        let Some(bundle) = guard.bundle.take() else {
            return Err("Desktop update has not been downloaded.".to_string());
        };
        (update, bundle)
    };
    if let Err(error) = update.install(&bundle) {
        if let Ok(mut guard) = state.lock() {
            guard.bundle = Some(bundle);
        }
        return Err(error.to_string());
    }
    Ok(())
}

#[tauri::command]
pub(crate) fn desktop_update_bundle_status(
    state: tauri::State<'_, DesktopUpdateState>,
) -> Result<DesktopUpdateBundleStatus, String> {
    let guard = state.lock().map_err(|error| error.to_string())?;
    Ok(DesktopUpdateBundleStatus {
        version: guard.update.as_ref().map(|u| u.version.clone()),
        downloaded: guard.bundle.is_some(),
        downloading: guard.downloading,
    })
}
