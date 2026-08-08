use serde::Serialize;
use tauri::Manager;
use tauri_plugin_updater::UpdaterExt;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DesktopUpdateMetadata {
    rid: tauri::ResourceId,
    current_version: String,
    version: String,
    date: Option<String>,
    body: Option<String>,
    raw_json: serde_json::Value,
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

#[tauri::command]
pub(crate) async fn check_desktop_update(
    webview: tauri::Webview,
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
    let Some(update) = updater.check().await.map_err(|error| error.to_string())? else {
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
        rid: webview.resources_table().add(update),
    };

    Ok(Some(metadata))
}
