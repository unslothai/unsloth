use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_updater::{Update, UpdaterExt};

const DOWNLOAD_EVENT: &str = "desktop-update-download";
const DOWNLOAD_EVENT_STEP: u64 = 512 * 1024;
const BUNDLE_FILE: &str = ".desktop-update-bundle";
const BUNDLE_METADATA_FILE: &str = ".desktop-update-bundle.json";

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

fn bundle_metadata_path() -> PathBuf {
    crate::diagnostics::studio_dir().join(BUNDLE_METADATA_FILE)
}

#[derive(Deserialize, Serialize)]
struct PreparedBundle {
    version: String,
    size: u64,
    sha256: String,
}

fn sha256(bytes: &[u8]) -> String {
    crate::native_backend_lease::hex_bytes(&Sha256::digest(bytes))
}

fn sync_parent(path: &Path) {
    #[cfg(unix)]
    if let Some(parent) = path.parent() {
        let _ = File::open(parent).and_then(|directory| directory.sync_all());
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", path.display()))?;
    fs::create_dir_all(parent).map_err(|error| format!("{}: {error}", parent.display()))?;
    let mut temporary = tempfile::Builder::new()
        .prefix(".desktop-update-")
        .tempfile_in(parent)
        .map_err(|error| format!("{}: {error}", parent.display()))?;
    temporary
        .write_all(bytes)
        .and_then(|()| temporary.as_file().sync_all())
        .map_err(|error| format!("{}: {error}", path.display()))?;
    temporary
        .persist(path)
        .map_err(|error| format!("{}: {}", path.display(), error.error))?;
    sync_parent(path);
    Ok(())
}

fn discard_prepared_bundle(bundle: &Path, metadata: &Path) {
    let _ = fs::remove_file(bundle);
    let _ = fs::remove_file(metadata);
}

fn persist_prepared_bundle(
    bundle: &Path,
    metadata: &Path,
    version: &str,
    bytes: &[u8],
) -> Result<(), String> {
    discard_prepared_bundle(bundle, metadata);
    write_atomic(bundle, bytes)?;
    let record = PreparedBundle {
        version: version.to_string(),
        size: bytes.len() as u64,
        sha256: sha256(bytes),
    };
    let body = serde_json::to_vec_pretty(&record).map_err(|error| error.to_string())?;
    if let Err(error) = write_atomic(metadata, &body) {
        discard_prepared_bundle(bundle, metadata);
        return Err(error);
    }
    Ok(())
}

fn read_prepared_bundle(
    bundle: &Path,
    metadata: &Path,
    expected_version: &str,
) -> Result<Vec<u8>, String> {
    let record: PreparedBundle = serde_json::from_slice(
        &fs::read(metadata).map_err(|error| format!("{}: {error}", metadata.display()))?,
    )
    .map_err(|error| format!("{}: {error}", metadata.display()))?;
    if record.version != expected_version {
        return Err(format!(
            "prepared update {} does not match {}",
            record.version, expected_version
        ));
    }
    let bytes = fs::read(bundle).map_err(|error| format!("{}: {error}", bundle.display()))?;
    if bytes.len() as u64 != record.size || sha256(&bytes) != record.sha256 {
        return Err("prepared update failed its integrity check".to_string());
    }
    Ok(bytes)
}

fn rehydrate_prepared_bundle(expected_version: &str) -> Option<PathBuf> {
    let bundle = bundle_path();
    let metadata = bundle_metadata_path();
    match read_prepared_bundle(&bundle, &metadata, expected_version) {
        Ok(_) => Some(bundle),
        Err(_) => {
            discard_prepared_bundle(&bundle, &metadata);
            None
        }
    }
}

pub(crate) type DesktopUpdateState = Arc<Mutex<DesktopUpdate>>;

pub(crate) fn new_desktop_update_state() -> DesktopUpdateState {
    Arc::new(Mutex::new(DesktopUpdate::default()))
}

pub(crate) fn pending_version(state: &DesktopUpdateState) -> Result<Option<String>, String> {
    let guard = state.lock().map_err(|error| error.to_string())?;
    Ok(guard.update.as_ref().map(|update| update.version.clone()))
}

/// `pypi_version` from latest.json: the backend this shell is built against.
pub(crate) fn pending_backend_version(
    state: &DesktopUpdateState,
) -> Result<Option<String>, String> {
    let guard = state.lock().map_err(|error| error.to_string())?;
    Ok(guard.update.as_ref().and_then(|update| {
        update
            .raw_json
            .get("pypi_version")
            .and_then(|value| value.as_str())
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    }))
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
        discard_prepared_bundle(&bundle_path(), &bundle_metadata_path());
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
    guard.bundle = rehydrate_prepared_bundle(&update.version);
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
            persist_prepared_bundle(&path, &bundle_metadata_path(), &update.version, &bytes)?;
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
    let metadata = bundle_metadata_path();
    let bundle = match read_prepared_bundle(&path, &metadata, &update.version) {
        Ok(bundle) => bundle,
        Err(error) => {
            discard_prepared_bundle(&path, &metadata);
            return Err(format!("Prepared update is invalid: {error}"));
        }
    };
    if let Err(error) = update.install(&bundle) {
        discard_prepared_bundle(&path, &metadata);
        return Err(error.to_string());
    }
    discard_prepared_bundle(&path, &metadata);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prepared_bundle_survives_state_recreation() {
        let directory = tempfile::tempdir().unwrap();
        let bundle = directory.path().join(BUNDLE_FILE);
        let metadata = directory.path().join(BUNDLE_METADATA_FILE);
        let bytes = b"signed updater bundle";

        persist_prepared_bundle(&bundle, &metadata, "0.1.901", bytes).unwrap();

        assert_eq!(
            read_prepared_bundle(&bundle, &metadata, "0.1.901").unwrap(),
            bytes
        );
    }

    #[test]
    fn a_corrupt_prepared_bundle_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let bundle = directory.path().join(BUNDLE_FILE);
        let metadata = directory.path().join(BUNDLE_METADATA_FILE);
        persist_prepared_bundle(&bundle, &metadata, "0.1.901", b"verified").unwrap();
        fs::write(&bundle, b"corrupt!").unwrap();

        let error = read_prepared_bundle(&bundle, &metadata, "0.1.901").unwrap_err();

        assert!(error.contains("integrity"), "{error}");
    }

    #[test]
    fn a_bundle_for_another_release_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let bundle = directory.path().join(BUNDLE_FILE);
        let metadata = directory.path().join(BUNDLE_METADATA_FILE);
        persist_prepared_bundle(&bundle, &metadata, "0.1.901", b"verified").unwrap();

        let error = read_prepared_bundle(&bundle, &metadata, "0.1.902").unwrap_err();

        assert!(error.contains("does not match"), "{error}");
    }
}
