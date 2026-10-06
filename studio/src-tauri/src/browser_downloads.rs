//! Paths of files saved from the browser panel, so Download history can reveal them or flag
//! them deleted. The webview only gets opaque ids.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tauri::{AppHandle, Manager, Runtime, State};

/// Well past MAX_DOWNLOADS in history-store.ts. The history forgets ids it drops, so only orphans
/// (a tab closed mid-download) build up toward this, not rows the history still shows.
const MAX_ENTRIES: usize = 1000;
const ID_BYTES: usize = 16;
const FILE_NAME: &str = "browser-downloads.json";

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
struct Entry {
    id: String,
    path: PathBuf,
}

/// Loaded from disk on first use, newest last.
#[derive(Default)]
pub struct BrowserDownloads {
    entries: Mutex<Option<Vec<Entry>>>,
}

pub fn new_browser_downloads() -> BrowserDownloads {
    BrowserDownloads::default()
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SavedDownload {
    id: String,
    name: String,
}

fn store_path<R: Runtime>(app: &AppHandle<R>) -> Option<PathBuf> {
    app.path()
        .app_local_data_dir()
        .ok()
        .map(|dir| dir.join(FILE_NAME))
}

fn new_id() -> String {
    let bytes = rand::random::<[u8; ID_BYTES]>();
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn valid_id(id: &str) -> bool {
    id.len() == ID_BYTES * 2 && id.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn load(path: Option<&Path>) -> Vec<Entry> {
    let Some(path) = path else {
        return Vec::new();
    };
    fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Vec<Entry>>(&bytes).ok())
        .map(|entries| {
            entries
                .into_iter()
                .filter(|entry| valid_id(&entry.id))
                .collect()
        })
        .unwrap_or_default()
}

/// Written to a temp file and renamed, so a crash never leaves half a list.
fn save(path: Option<&Path>, entries: &[Entry]) {
    let Some(path) = path else {
        return;
    };
    let Some(parent) = path.parent() else {
        return;
    };
    let Ok(bytes) = serde_json::to_vec(entries) else {
        return;
    };
    let _ = fs::create_dir_all(parent).and_then(|()| {
        let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
        temporary.write_all(&bytes)?;
        temporary
            .persist(path)
            .map(|_| ())
            .map_err(|error| error.error)
    });
}

fn with_entries<R: Runtime, T>(
    app: &AppHandle<R>,
    state: &BrowserDownloads,
    change: impl FnOnce(&mut Vec<Entry>) -> (T, bool),
) -> T {
    let path = store_path(app);
    let mut guard = state.entries.lock().unwrap();
    let entries = guard.get_or_insert_with(|| load(path.as_deref()));
    let (result, changed) = change(entries);
    if changed {
        save(path.as_deref(), entries);
    }
    result
}

fn push(entries: &mut Vec<Entry>, path: PathBuf) -> String {
    let id = new_id();
    entries.push(Entry {
        id: id.clone(),
        path,
    });
    let excess = entries.len().saturating_sub(MAX_ENTRIES);
    entries.drain(..excess);
    id
}

/// Remember where a download landed; its id for the panel's history.
pub(crate) fn record<R: Runtime>(app: &AppHandle<R>, path: PathBuf) -> String {
    let state = app.state::<BrowserDownloads>();
    with_entries(app, &state, |entries| (push(entries, path), true))
}

fn path_of(entries: &[Entry], id: &str) -> Option<PathBuf> {
    entries
        .iter()
        .find(|entry| entry.id == id)
        .map(|entry| entry.path.clone())
}

/// Save a panel file where the user picks, like `save_native_file`, and remember it.
#[tauri::command]
pub async fn browser_download_save(
    webview: tauri::Webview,
    app: AppHandle,
    request: tauri::ipc::Request<'_>,
) -> Result<Option<SavedDownload>, String> {
    crate::native_intents::ensure_main_window(&webview)?;
    let Some(path) = crate::native_file_dialogs::save_request_with_dialog(&app, &request).await?
    else {
        return Ok(None);
    };
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok(Some(SavedDownload {
        id: record(&app, path),
        name,
    }))
}

/// Show a remembered download in Finder or Explorer, selected.
#[tauri::command]
pub fn browser_download_reveal(
    webview: tauri::Webview,
    app: AppHandle,
    state: State<'_, BrowserDownloads>,
    id: String,
) -> Result<(), String> {
    crate::native_intents::ensure_main_window(&webview)?;
    let path = with_entries(&app, &state, |entries| (path_of(entries, &id), false))
        .ok_or_else(|| "Unknown download.".to_string())?;
    if !path.is_file() {
        return Err("The file was moved or deleted.".to_string());
    }
    crate::native_intents::reveal_in_file_manager(&path)
}

/// Whether each download is still where it was saved, in the order asked.
#[tauri::command]
pub fn browser_download_exists(
    webview: tauri::Webview,
    app: AppHandle,
    state: State<'_, BrowserDownloads>,
    ids: Vec<String>,
) -> Result<Vec<bool>, String> {
    crate::native_intents::ensure_main_window(&webview)?;
    let paths = with_entries(&app, &state, |entries| {
        let paths: Vec<Option<PathBuf>> = ids.iter().map(|id| path_of(entries, id)).collect();
        (paths, false)
    });
    Ok(paths
        .into_iter()
        .map(|path| path.is_some_and(|path| path.is_file()))
        .collect())
}

/// Forget downloads taken off the history (the files stay). Always by id: accounts share this list.
#[tauri::command]
pub fn browser_download_forget(
    webview: tauri::Webview,
    app: AppHandle,
    state: State<'_, BrowserDownloads>,
    ids: Vec<String>,
) -> Result<(), String> {
    crate::native_intents::ensure_main_window(&webview)?;
    with_entries(&app, &state, |entries| {
        let before = entries.len();
        entries.retain(|entry| !ids.contains(&entry.id));
        ((), entries.len() != before)
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_random_hex_and_checked() {
        let id = new_id();
        assert!(valid_id(&id));
        assert_ne!(id, new_id());
        assert!(!valid_id("../etc/passwd"));
        assert!(!valid_id(""));
    }

    #[test]
    fn the_list_keeps_only_the_newest() {
        let mut entries = Vec::new();
        let first = push(&mut entries, PathBuf::from("/tmp/first"));
        for index in 0..MAX_ENTRIES {
            push(&mut entries, PathBuf::from(format!("/tmp/{index}")));
        }
        assert_eq!(entries.len(), MAX_ENTRIES);
        assert!(path_of(&entries, &first).is_none());
        assert_eq!(
            path_of(&entries, &entries[MAX_ENTRIES - 1].id.clone()),
            Some(PathBuf::from(format!("/tmp/{}", MAX_ENTRIES - 1)))
        );
    }

    #[test]
    fn saved_lists_load_back_and_bad_ids_are_dropped() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(FILE_NAME);
        let mut entries = Vec::new();
        push(&mut entries, PathBuf::from("/tmp/a.zip"));
        entries.push(Entry {
            id: "nope".into(),
            path: PathBuf::from("/tmp/b.zip"),
        });
        save(Some(&path), &entries);
        let loaded = load(Some(&path));
        assert_eq!(loaded, entries[..1].to_vec());
        assert!(load(Some(&dir.path().join("missing.json"))).is_empty());
    }
}
