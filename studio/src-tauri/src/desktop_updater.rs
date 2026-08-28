use log::warn;
use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_updater::{Update, UpdaterExt};

const DOWNLOAD_EVENT: &str = "desktop-update-download";
const DOWNLOAD_EVENT_STEP: u64 = 512 * 1024;
const BUNDLE_FILE: &str = ".desktop-update-bundle";

/// The prepared bundle lives on disk, not in this struct.
///
/// Preparation finishes long before the user restarts, and the whole point of the
/// flow is that they keep training or generating in the meantime. A platform
/// installer held in memory for that entire window is resident memory taken from
/// exactly the workloads the background update exists to avoid interrupting.
#[derive(Default)]
pub(crate) struct DesktopUpdate {
    update: Option<Update>,
    bundle: Option<PathBuf>,
    downloading: bool,
}

fn bundle_path() -> PathBuf {
    crate::diagnostics::studio_dir().join(BUNDLE_FILE)
}

fn discard_bundle(path: Option<&PathBuf>) {
    if let Some(path) = path {
        let _ = fs::remove_file(path);
    }
}

pub(crate) type DesktopUpdateState = Arc<Mutex<DesktopUpdate>>;

pub(crate) fn new_desktop_update_state() -> DesktopUpdateState {
    Arc::new(Mutex::new(DesktopUpdate::default()))
}

/// Drop a bundle left behind by a previous run.
///
/// Which bundle is prepared is in-memory state, so one that outlived its process is
/// claimed by nothing and would sit at full installer size forever. Call this only
/// from the app setup hook: `.manage(...)` runs while the builder is still being
/// assembled, before the single-instance plugin turns a duplicate launch away, so a
/// second process would delete the bundle the first one is holding a path to.
pub(crate) fn discard_stale_bundle() {
    let _ = fs::remove_file(bundle_path());
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
        discard_bundle(guard.bundle.take().as_ref());
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
        discard_bundle(guard.bundle.take().as_ref());
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
            if guard.update.as_ref().map(|u| u.version.as_str()) != Some(update.version.as_str()) {
                // A newer check landed mid-download; these bytes are already stale.
                return Ok(());
            }
            let path = bundle_path();
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            fs::write(&path, &bytes).map_err(|error| format!("{}: {error}", path.display()))?;
            guard.bundle = Some(path);
            Ok(())
        }
        Err(error) => Err(error.to_string()),
    }
}

#[tauri::command]
pub(crate) async fn install_desktop_update(
    state: tauri::State<'_, DesktopUpdateState>,
) -> Result<(), String> {
    let (update, path) = {
        let mut guard = state.lock().map_err(|error| error.to_string())?;
        let Some(update) = guard.update.clone() else {
            return Err("No desktop update has been checked.".to_string());
        };
        let Some(path) = guard.bundle.take() else {
            return Err("Desktop update has not been downloaded.".to_string());
        };
        (update, path)
    };
    // Read back only for the install itself, so the bytes are resident for the
    // seconds it takes rather than for the whole session.
    let bundle = match fs::read(&path) {
        Ok(bundle) => bundle,
        Err(error) => {
            let _ = fs::remove_file(&path);
            return Err(format!("Prepared update is unreadable: {error}"));
        }
    };
    if let Err(error) = update.install(&bundle) {
        // Keep the file: the retry path installs it again without downloading.
        if let Ok(mut guard) = state.lock() {
            guard.bundle = Some(path);
        } else {
            warn!("[desktop-update] state poisoned; dropping the prepared bundle");
        }
        return Err(error.to_string());
    }
    let _ = fs::remove_file(&path);
    Ok(())
}

#[tauri::command]
pub(crate) fn desktop_update_bundle_status(
    state: tauri::State<'_, DesktopUpdateState>,
) -> Result<DesktopUpdateBundleStatus, String> {
    let guard = state.lock().map_err(|error| error.to_string())?;
    Ok(DesktopUpdateBundleStatus {
        version: guard.update.as_ref().map(|u| u.version.clone()),
        downloaded: guard.bundle.as_ref().is_some_and(|path| path.is_file()),
        downloading: guard.downloading,
    })
}
