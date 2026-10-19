//! Browser panel downloads: folder, approval for site downloads, and where each landed. The webview only gets opaque ids.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;
use tauri::{AppHandle, Manager, Runtime, State, Url};
use tauri_plugin_dialog::DialogExt;

/// Well past MAX_DOWNLOADS in history-store.ts. The history forgets ids it drops, so only orphans
/// (a tab closed mid-download) build up toward this, not rows the history still shows.
const MAX_ENTRIES: usize = 1000;
const ID_BYTES: usize = 16;
const FILE_NAME: &str = "browser-downloads.json";
const FOLDER_FILE_NAME: &str = "browser-download-folder.json";
const STAGING_DIR: &str = "browser-download-staging";
const ASK_HEADER: &str = "x-unsloth-ask";
const SOURCE_HEADER: &str = "x-unsloth-source";
/// Per-tab cap on unanswered page downloads, so an ignored prompt can't fill the disk with staged files.
pub(crate) const MAX_UNANSWERED_PER_TAB: usize = 3;
/// An unanswered prompt (window reloaded, say) counts as Cancel after this.
const ANSWER_TIMEOUT: Duration = Duration::from_secs(10 * 60);

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
struct Entry {
    id: String,
    path: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Decision {
    Allow { ask: bool },
    Deny,
}

struct Pending {
    tab_id: String,
    url: Url,
    name: String,
    staged: PathBuf,
    decision: Option<Decision>,
    finished: Option<bool>,
}

#[derive(Default)]
pub struct BrowserDownloads {
    /// Loaded from disk on first use, newest last.
    entries: Mutex<Option<Vec<Entry>>>,
    pending: Mutex<HashMap<String, Pending>>,
    /// None = system Downloads; outer None = not loaded yet.
    folder: Mutex<Option<Option<PathBuf>>>,
    staging_cleared: AtomicBool,
    naming: Mutex<()>,
}

pub fn new_browser_downloads() -> BrowserDownloads {
    BrowserDownloads::default()
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SavedDownload {
    id: String,
    name: String,
    /// False when the internet mark couldn't be written (FAT, exFAT, some network drives): the panel
    /// warns. None when nothing marks it (no web source, or Linux).
    marked: Option<bool>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadFolder {
    path: String,
    custom: bool,
}

#[derive(Serialize, Deserialize)]
struct FolderFile {
    path: PathBuf,
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

fn save(path: Option<&Path>, value: &impl Serialize) {
    if let Some(path) = path {
        let _ = write_json(path, value);
    }
}

/// Written to a temp file and renamed, so a crash never leaves half a file.
fn write_json(path: &Path, value: &impl Serialize) -> std::io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| std::io::Error::other("no parent folder"))?;
    let bytes = serde_json::to_vec(value)?;
    fs::create_dir_all(parent)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(&bytes)?;
    temporary
        .persist(path)
        .map(|_| ())
        .map_err(|error| error.error)
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
        save(path.as_deref(), &*entries);
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

/// Save a panel file in the download folder, or via a dialog when `x-unsloth-ask` is set, and remember it.
#[tauri::command]
pub async fn browser_download_save(
    webview: tauri::Webview,
    app: AppHandle,
    request: tauri::ipc::Request<'_>,
) -> Result<Option<SavedDownload>, String> {
    crate::native_intents::ensure_main_window(&webview)?;
    let ask = request
        .headers()
        .get(ASK_HEADER)
        .is_some_and(|value| value.as_bytes() == b"1");
    let source = request
        .headers()
        .get(SOURCE_HEADER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| Url::parse(value).ok())
        .filter(|url| matches!(url.scheme(), "http" | "https"));
    let folder = download_folder(&app)?;
    let saved = if ask {
        crate::native_file_dialogs::save_request_with_dialog(&app, &request, Some(&folder)).await?
    } else {
        let state = app.state::<BrowserDownloads>();
        let _naming = state.naming.lock().unwrap();
        Some(crate::native_file_dialogs::save_request_in(
            &request, &folder,
        )?)
    };
    let Some(path) = saved else {
        return Ok(None);
    };
    // Quarantined like a page download, so Gatekeeper or SmartScreen checks it.
    let marked = source
        .as_ref()
        .and_then(|source| crate::browser_webview::mark_downloaded(&path, source));
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok(Some(SavedDownload {
        id: record(&app, path),
        name,
        marked,
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

/// Extensions that run code when opened, shared with download-safety.ts.
pub(crate) fn runs_code(path: &Path) -> bool {
    static EXTENSIONS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    let extensions = EXTENSIONS.get_or_init(|| {
        #[derive(Deserialize)]
        struct Policy {
            extensions: Vec<String>,
        }
        serde_json::from_str::<Policy>(include_str!(
            "../../frontend/src/features/browser/dangerous-file-types.json"
        ))
        .map(|policy| policy.extensions)
        .unwrap_or_default()
    });
    // Windows drops trailing dots and spaces, so `setup.exe.` still runs.
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_default();
    let name = name.trim_end_matches(['.', ' ']);
    name.rsplit_once('.').is_some_and(|(_, ext)| {
        // Windows compares names upper-cased, so `.m\u{17f}i` (long s) is `.MSI`.
        let folded = [ext.to_lowercase(), ext.to_uppercase().to_lowercase()];
        extensions.iter().any(|known| folded.contains(known))
    })
}

/// Open a download with its default app. Programs and scripts are refused: they stay a reveal away.
#[tauri::command]
pub fn browser_download_open(
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
    if runs_code(&path) {
        return Err("Programs and scripts open from their folder.".to_string());
    }
    tauri_plugin_opener::open_path(&path, None::<&str>).map_err(|error| error.to_string())
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

fn folder_file<R: Runtime>(app: &AppHandle<R>) -> Option<PathBuf> {
    app.path()
        .app_local_data_dir()
        .ok()
        .map(|dir| dir.join(FOLDER_FILE_NAME))
}

fn custom_folder<R: Runtime>(app: &AppHandle<R>) -> Option<PathBuf> {
    let state = app.state::<BrowserDownloads>();
    let mut guard = state.folder.lock().unwrap();
    let folder = guard.get_or_insert_with(|| {
        folder_file(app)
            .and_then(|path| fs::read(path).ok())
            .and_then(|bytes| serde_json::from_slice::<FolderFile>(&bytes).ok())
            .map(|file| file.path)
    });
    folder.clone().filter(|path| path.is_dir())
}

pub(crate) fn download_folder<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf, String> {
    if let Some(folder) = custom_folder(app) {
        return Ok(folder);
    }
    let folder = app
        .path()
        .download_dir()
        .ok()
        .or_else(dirs::home_dir)
        .ok_or_else(|| "Could not find a Downloads folder.".to_string())?;
    // XDG can name a Downloads folder that was never created.
    fs::create_dir_all(&folder)
        .map_err(|error| format!("Failed to prepare {}: {error}", folder.display()))?;
    Ok(folder)
}

fn folder_info<R: Runtime>(app: &AppHandle<R>) -> Result<DownloadFolder, String> {
    Ok(DownloadFolder {
        path: crate::native_file_dialogs::display_path(&download_folder(app)?),
        custom: custom_folder(app).is_some(),
    })
}

/// Cached only once written, so Settings never shows a choice a restart would lose.
fn set_folder<R: Runtime>(app: &AppHandle<R>, folder: Option<PathBuf>) -> Result<(), String> {
    let file = folder_file(app).ok_or_else(|| "Could not find the app data folder.".to_string())?;
    let written = match &folder {
        Some(path) => write_json(&file, &FolderFile { path: path.clone() }),
        None => match fs::remove_file(&file) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(error),
            _ => Ok(()),
        },
    };
    written.map_err(|error| format!("Failed to save the download location: {error}"))?;
    *app.state::<BrowserDownloads>().folder.lock().unwrap() = Some(folder);
    Ok(())
}

#[tauri::command]
pub fn browser_download_folder(
    webview: tauri::Webview,
    app: AppHandle,
) -> Result<DownloadFolder, String> {
    crate::native_intents::ensure_main_window(&webview)?;
    folder_info(&app)
}

/// The webview never names a path: the folder comes from the system dialog.
#[tauri::command]
pub async fn browser_download_folder_pick(
    webview: tauri::Webview,
    app: AppHandle,
) -> Result<Option<DownloadFolder>, String> {
    crate::native_intents::ensure_main_window(&webview)?;
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog()
        .file()
        .set_title("Choose download location")
        .set_directory(download_folder(&app)?)
        .pick_folder(move |path| {
            let _ = tx.send(path);
        });
    let Some(picked) = rx.await.map_err(|_| "Dialog closed".to_string())? else {
        return Ok(None);
    };
    let path = picked
        .into_path()
        .map_err(|_| "Only local folders are supported.".to_string())?;
    set_folder(&app, Some(path))?;
    folder_info(&app).map(Some)
}

#[tauri::command]
pub fn browser_download_folder_reset(
    webview: tauri::Webview,
    app: AppHandle,
) -> Result<DownloadFolder, String> {
    crate::native_intents::ensure_main_window(&webview)?;
    set_folder(&app, None)?;
    folder_info(&app)
}

/// A fresh staging folder for one page download, with its id; the staging root is emptied on first use after launch.
pub(crate) fn staging_dir<R: Runtime>(app: &AppHandle<R>) -> Option<(String, PathBuf)> {
    let root = app.path().app_cache_dir().ok()?.join(STAGING_DIR);
    let state = app.state::<BrowserDownloads>();
    if !state.staging_cleared.swap(true, Ordering::SeqCst) {
        let _ = fs::remove_dir_all(&root);
    }
    let id = new_id();
    let dir = root.join(&id);
    fs::create_dir_all(&dir).ok()?;
    Some((id, dir))
}

pub(crate) fn unanswered<R: Runtime>(app: &AppHandle<R>, tab_id: &str) -> usize {
    app.state::<BrowserDownloads>()
        .pending
        .lock()
        .unwrap()
        .values()
        .filter(|entry| entry.tab_id == tab_id && entry.decision.is_none())
        .count()
}

pub(crate) fn add_pending<R: Runtime>(
    app: &AppHandle<R>,
    id: String,
    tab_id: String,
    url: Url,
    staged: PathBuf,
) {
    let name = staged
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    let pending = Pending {
        tab_id,
        url,
        name,
        staged,
        decision: None,
        finished: None,
    };
    app.state::<BrowserDownloads>()
        .pending
        .lock()
        .unwrap()
        .insert(id.clone(), pending);
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        tokio::time::sleep(ANSWER_TIMEOUT).await;
        let expired = {
            let state = app.state::<BrowserDownloads>();
            let mut pending = state.pending.lock().unwrap();
            match pending.get_mut(&id) {
                Some(entry) if entry.decision.is_none() => {
                    entry.decision = Some(Decision::Deny);
                    true
                }
                _ => false,
            }
        };
        if expired {
            settle(&app, &id);
        }
    });
}

/// Windows and macOS file systems ignore case, so `Report.pdf` and `report.pdf` are one file.
pub(crate) fn same_path(a: &Path, b: &Path) -> bool {
    if cfg!(any(windows, target_os = "macos")) {
        // Compared upper-cased as Windows does (`m\u{17f}i` is `MSI`); on Windows `\\?\C:\x` is
        // `C:\x` and `\\?\UNC\h\s` is `\\h\s`.
        let plain = |p: &Path| {
            let text = p.to_string_lossy().to_uppercase();
            if !cfg!(windows) {
                text
            } else if let Some(share) = text.strip_prefix(r"\\?\UNC\") {
                format!(r"\\{share}")
            } else if let Some(local) = text.strip_prefix(r"\\?\") {
                local.to_string()
            } else {
                text
            }
        };
        plain(a) == plain(b)
    } else {
        a == b
    }
}

fn pending_for<'a>(
    pending: &'a mut HashMap<String, Pending>,
    candidates: &[Option<&Path>],
) -> Option<(&'a String, &'a mut Pending)> {
    pending.iter_mut().find(|(_, entry)| {
        candidates
            .iter()
            .flatten()
            .any(|candidate| same_path(&entry.staged, candidate))
    })
}

/// `candidates`: the path reserved for the download, then the one the engine reported.
pub(crate) fn finished<R: Runtime>(
    app: &AppHandle<R>,
    candidates: [Option<&Path>; 2],
    success: bool,
) {
    let id = {
        let state = app.state::<BrowserDownloads>();
        let mut pending = state.pending.lock().unwrap();
        let Some((id, entry)) = pending_for(&mut pending, &candidates) else {
            return;
        };
        entry.finished = Some(success);
        id.clone()
    };
    settle(app, &id);
}

#[tauri::command]
pub fn browser_download_decide(
    webview: tauri::Webview,
    app: AppHandle,
    id: String,
    allow: bool,
    ask: bool,
) -> Result<(), String> {
    crate::native_intents::ensure_main_window(&webview)?;
    let decision = if allow {
        Decision::Allow { ask }
    } else {
        Decision::Deny
    };
    decide(
        &mut app.state::<BrowserDownloads>().pending.lock().unwrap(),
        &id,
        decision,
    )?;
    settle(&app, &id);
    Ok(())
}

/// A second answer is refused: after ANSWER_TIMEOUT denied it, a late Download click must not allow it.
fn decide(
    pending: &mut HashMap<String, Pending>,
    id: &str,
    decision: Decision,
) -> Result<(), String> {
    let entry = pending
        .get_mut(id)
        .ok_or_else(|| "Unknown download.".to_string())?;
    if entry.decision.is_some() {
        return Err("This download was already answered.".to_string());
    }
    entry.decision = Some(decision);
    Ok(())
}

/// Once answered and finished, move the download out of staging or drop it.
fn settle<R: Runtime>(app: &AppHandle<R>, id: &str) {
    let entry = {
        let state = app.state::<BrowserDownloads>();
        let mut pending = state.pending.lock().unwrap();
        match pending.get(id) {
            Some(entry) if entry.decision.is_some() && entry.finished.is_some() => {
                pending.remove(id)
            }
            _ => None,
        }
    };
    let Some(entry) = entry else {
        return;
    };
    let app = app.clone();
    let id = id.to_string();
    tauri::async_runtime::spawn(async move {
        let staging = entry.staged.parent().map(Path::to_path_buf);
        let result = match (entry.decision, entry.finished) {
            (Some(Decision::Allow { ask }), Some(true)) => deliver(&app, &entry, ask).await,
            (Some(Decision::Allow { .. }), _) => Err("failed".to_string()),
            _ => Ok(None),
        };
        if let Some(staging) = staging {
            let _ = fs::remove_dir_all(staging);
        }
        match result {
            Ok(Some((path, download_id, marked))) => crate::browser_webview::emit_download_done(
                &app,
                &entry.tab_id,
                &entry.url,
                &path,
                Some(download_id),
                marked,
                &id,
            ),
            Ok(None) if matches!(entry.decision, Some(Decision::Allow { .. })) => {
                crate::browser_webview::emit_download_cancelled(
                    &app,
                    &entry.tab_id,
                    &entry.url,
                    &id,
                )
            }
            Ok(None) => {}
            Err(_) => crate::browser_webview::emit_download_failed(
                &app,
                &entry.tab_id,
                &entry.url,
                &entry.name,
                Some(&id),
            ),
        }
    });
}

/// None if the save dialog was cancelled.
async fn deliver<R: Runtime>(
    app: &AppHandle<R>,
    entry: &Pending,
    ask: bool,
) -> Result<Option<(PathBuf, String, Option<bool>)>, String> {
    let folder = download_folder(app)?;
    let staged = entry.staged.clone();
    let target = if ask {
        let (tx, rx) = tokio::sync::oneshot::channel();
        app.dialog()
            .file()
            .set_title("Save download")
            .set_directory(&folder)
            .set_file_name(&entry.name)
            .save_file(move |path| {
                let _ = tx.send(path);
            });
        let Some(picked) = rx.await.map_err(|_| "Dialog closed".to_string())? else {
            return Ok(None);
        };
        let target = picked
            .into_path()
            .map_err(|_| "Only local paths are supported.".to_string())?;
        let to = target.clone();
        blocking(move || move_file(&staged, &to)).await?;
        target
    } else {
        let app = app.clone();
        let name = entry.name.clone();
        blocking(move || {
            let state = app.state::<BrowserDownloads>();
            let _naming = state.naming.lock().unwrap();
            // A name another program took since it was picked is never replaced: pick again.
            for _ in 0..3 {
                let target = crate::native_file_dialogs::unique_destination(&folder, &name)?;
                match place_new(&staged, &target) {
                    Ok(()) => return Ok(target),
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(error) => {
                        return Err(format!("Failed to save {}: {error}", target.display()))
                    }
                }
            }
            Err(format!("Failed to save {name}: its name keeps being taken"))
        })
        .await?
    };
    let marked = crate::browser_webview::mark_downloaded(&target, &entry.url);
    let id = record(app, target.clone());
    Ok(Some((target, id, marked)))
}

/// A cross-volume copy can take minutes; keep it off the workers IPC and timers share.
async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, String> {
    tauri::async_runtime::spawn_blocking(work)
        .await
        .map_err(|error| format!("Failed to save the download: {error}"))?
}

/// Put a staged download at a new name without ever replacing a file there: a hard link where the
/// volume allows it (no copy, and AlreadyExists if the name was taken), else a copy that only lands
/// while the name is still free.
fn place_new(from: &Path, to: &Path) -> std::io::Result<()> {
    match fs::hard_link(from, to) {
        Ok(()) => {
            let _ = fs::remove_file(from);
            return Ok(());
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => return Err(error),
        // Another volume (a chosen folder elsewhere) or no hard links (FAT, exFAT): copy.
        Err(_) => {}
    }
    // Created like any new file (0666 less the umask on Unix), not a tempfile's 0600.
    let mut temporary =
        crate::native_file_dialogs::staged_temp_file(to).map_err(std::io::Error::other)?;
    let mut source = fs::File::open(from)?;
    std::io::copy(&mut source, temporary.as_file_mut())?;
    temporary.as_file().sync_all()?;
    temporary
        .persist_noclobber(to)
        .map(|_| ())
        .map_err(|error| error.error)
}

fn move_file(from: &Path, to: &Path) -> Result<(), String> {
    if fs::rename(from, to).is_ok() {
        return Ok(());
    }
    copy_into_place(from, to)
}

/// Copied beside `to` and renamed over it once complete, so a failed copy leaves no half file and the target intact.
fn copy_into_place(from: &Path, to: &Path) -> Result<(), String> {
    let failed = |error: std::io::Error| format!("Failed to save {}: {error}", to.display());
    let mut temporary = crate::native_file_dialogs::staged_temp_file(to)?;
    let mut source = fs::File::open(from).map_err(failed)?;
    std::io::copy(&mut source, temporary.as_file_mut())
        .and_then(|_| temporary.as_file().sync_all())
        .map_err(failed)?;
    temporary
        .persist(to)
        .map(|_| ())
        .map_err(|error| failed(error.error))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn programs_and_scripts_are_not_opened() {
        for name in [
            "setup.exe",
            "Install.CMD",
            "run.sh",
            "tool.command",
            "x.jar",
            "Foo.app",
            "a.exe. ",
            "installer.m\u{17f}i",
        ] {
            assert!(runs_code(Path::new(name)), "{name}");
        }
        for name in [
            "report.pdf",
            "data.csv",
            "clip.mp4",
            "noext",
            ".exe-notes.txt",
        ] {
            assert!(!runs_code(Path::new(name)), "{name}");
        }
    }

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
    fn a_copy_replaces_its_target_only_once_complete() {
        let dir = tempfile::tempdir().unwrap();
        let to = dir.path().join("file.zip");
        fs::write(&to, b"kept").unwrap();
        assert!(copy_into_place(&dir.path().join("missing.zip"), &to).is_err());
        assert_eq!(fs::read(&to).unwrap(), b"kept");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
        let from = dir.path().join("staged.zip");
        fs::write(&from, b"new").unwrap();
        copy_into_place(&from, &to).unwrap();
        assert_eq!(fs::read(&to).unwrap(), b"new");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 2);
    }

    #[test]
    fn a_failed_location_write_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("blocker");
        fs::write(&blocker, b"x").unwrap();
        let folder = FolderFile {
            path: dir.path().to_path_buf(),
        };
        assert!(write_json(&blocker.join("folder.json"), &folder).is_err());
        let file = dir.path().join("folder.json");
        write_json(&file, &folder).unwrap();
        let read: FolderFile = serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
        assert_eq!(read.path, dir.path());
    }

    #[test]
    fn a_download_is_answered_once() {
        let mut pending = HashMap::new();
        pending.insert(
            "a".to_string(),
            Pending {
                tab_id: "t".into(),
                url: Url::parse("https://example.com/f.zip").unwrap(),
                name: "f.zip".into(),
                staged: PathBuf::from("f.zip"),
                decision: None,
                finished: None,
            },
        );
        // A late Download click after the timeout's Deny changes nothing.
        decide(&mut pending, "a", Decision::Deny).unwrap();
        assert!(decide(&mut pending, "a", Decision::Allow { ask: false }).is_err());
        assert_eq!(pending["a"].decision, Some(Decision::Deny));
        assert!(decide(&mut pending, "b", Decision::Deny).is_err());
    }

    #[test]
    fn a_staged_download_moves_to_its_target() {
        let dir = tempfile::tempdir().unwrap();
        let from = dir.path().join("staged.zip");
        let to = dir.path().join("out").join("file.zip");
        fs::write(&from, b"zip").unwrap();
        fs::create_dir_all(to.parent().unwrap()).unwrap();
        move_file(&from, &to).unwrap();
        assert_eq!(fs::read(&to).unwrap(), b"zip");
        assert!(!from.exists());
    }

    #[test]
    fn placing_a_download_never_replaces_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let from = dir.path().join("staged.exe");
        let taken = dir.path().join("setup.exe");
        fs::write(&from, b"new").unwrap();
        fs::write(&taken, b"mine").unwrap();
        let error = place_new(&from, &taken).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(fs::read(&taken).unwrap(), b"mine");
        assert!(from.exists());
        let free = dir.path().join("setup (1).exe");
        place_new(&from, &free).unwrap();
        assert_eq!(fs::read(&free).unwrap(), b"new");
        assert!(!from.exists());
    }

    #[cfg(windows)]
    #[test]
    fn a_verbatim_path_is_the_same_file() {
        assert!(same_path(
            Path::new(r"\\?\C:\Users\A\x.zip"),
            Path::new(r"c:\users\a\X.zip")
        ));
        assert!(same_path(
            Path::new(r"\\?\UNC\host\share\x.zip"),
            Path::new(r"\\HOST\share\X.zip")
        ));
        assert!(same_path(
            Path::new("C:\\d\\payload.m\u{17f}i"),
            Path::new(r"C:\d\PAYLOAD.MSI")
        ));
    }

    #[test]
    fn a_finished_download_is_found_by_its_reserved_path() {
        let mut pending = HashMap::new();
        let staged = PathBuf::from("/cache/staging/a/Report.pdf");
        pending.insert(
            "a".to_string(),
            Pending {
                tab_id: "t".into(),
                url: Url::parse("https://example.com/Report.pdf").unwrap(),
                name: "Report.pdf".into(),
                staged: staged.clone(),
                decision: None,
                finished: None,
            },
        );
        // The engine reported no path (macOS) or another one: the reserved path still finds it.
        let other = PathBuf::from("/elsewhere/x.pdf");
        assert!(pending_for(&mut pending, &[Some(&staged), Some(&other)]).is_some());
        assert!(pending_for(&mut pending, &[None, Some(&other)]).is_none());
        let respelled = PathBuf::from("/cache/staging/a/report.pdf");
        assert_eq!(
            pending_for(&mut pending, &[None, Some(&respelled)]).is_some(),
            cfg!(any(windows, target_os = "macos"))
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
