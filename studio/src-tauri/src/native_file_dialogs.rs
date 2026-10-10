use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};

use serde::Serialize;
use std::borrow::Cow;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Manager, State};
use tauri_plugin_dialog::DialogExt;

const MAX_TRAINING_CONFIG_BYTES: u64 = 1024 * 1024;
/// Per redeemed range, not per file: each piece of a large import must fit one IPC response.
const MAX_CHAT_IMPORT_CHUNK_BYTES: usize = 8 * 1024 * 1024;
const NATIVE_FILE_NAME_HEADER: &str = "x-unsloth-default-name";
const NATIVE_FILE_SAVE_TOKEN_HEADER: &str = "x-unsloth-save-token";
const MAX_NATIVE_FILE_SAVE_CHUNK_BYTES: usize = 8 * 1024 * 1024;
const CHAT_IMPORT_EXTENSIONS: &[&str] = &["json", "jsonl", "ndjson", "csv", "md", "markdown"];
const CHAT_IMPORT_TYPE_ERROR: &str =
    "Chat import must be a .json, .jsonl, .ndjson, .csv, or .md file.";
const TRAINING_CONFIG_EXTENSIONS: &[&str] = &["yaml", "yml"];

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeImportedFile {
    name: String,
    content: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeChatImport {
    name: String,
    size: u64,
    token: String,
}

const CHAT_IMPORT_HANDLE_LIMIT: usize = 8;

#[derive(Default)]
pub struct ChatImportRegistry {
    /// Oldest first, so eviction drops the least recent pick.
    files: Mutex<Vec<(String, PathBuf, Arc<Mutex<File>>)>>,
}

impl ChatImportRegistry {
    /// Holds the opened file, not its path, so a file replaced mid-import cannot splice in bytes.
    fn register(&self, path: PathBuf, file: File) -> String {
        let token: String = (0..4)
            .map(|_| format!("{:016x}", rand::random::<u64>()))
            .collect();
        let mut files = self.files.lock().expect("chat import registry poisoned");
        files.push((token.clone(), path, Arc::new(Mutex::new(file))));
        while files.len() > CHAT_IMPORT_HANDLE_LIMIT {
            files.remove(0);
        }
        token
    }

    fn resolve(&self, token: &str) -> Option<(PathBuf, Arc<Mutex<File>>)> {
        let files = self.files.lock().expect("chat import registry poisoned");
        files
            .iter()
            .find(|(candidate, _, _)| candidate == token)
            .map(|(_, path, file)| (path.clone(), Arc::clone(file)))
    }
}

struct NativeSave {
    destination: PathBuf,
    temporary: tempfile::NamedTempFile,
}

const NATIVE_SAVE_HANDLE_LIMIT: usize = 8;

#[derive(Default)]
pub struct NativeSaveRegistry {
    files: Mutex<Vec<(String, Arc<Mutex<NativeSave>>)>>,
}

impl NativeSaveRegistry {
    fn register(&self, save: NativeSave) -> String {
        let token: String = (0..4)
            .map(|_| format!("{:016x}", rand::random::<u64>()))
            .collect();
        let mut files = self.files.lock().expect("native save registry poisoned");
        files.push((token.clone(), Arc::new(Mutex::new(save))));
        while files.len() > NATIVE_SAVE_HANDLE_LIMIT {
            files.remove(0);
        }
        token
    }

    fn resolve(&self, token: &str) -> Option<Arc<Mutex<NativeSave>>> {
        let files = self.files.lock().expect("native save registry poisoned");
        files
            .iter()
            .find(|(candidate, _)| candidate == token)
            .map(|(_, save)| Arc::clone(save))
    }

    fn take(&self, token: &str) -> Option<Arc<Mutex<NativeSave>>> {
        let mut files = self.files.lock().expect("native save registry poisoned");
        let index = files.iter().position(|(candidate, _)| candidate == token)?;
        Some(files.remove(index).1)
    }

    fn cancel(&self, token: &str) {
        let _ = self.take(token);
    }
}

fn default_file_name(suggested_name: &str) -> String {
    Path::new(suggested_name)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty() && *name != "." && *name != "..")
        .unwrap_or("unsloth-export.json")
        .to_string()
}
fn decode_default_file_name(encoded_name: &str) -> Result<String, String> {
    let bytes = BASE64
        .decode(encoded_name)
        .map_err(|_| "Invalid native export filename.".to_string())?;
    let name =
        String::from_utf8(bytes).map_err(|_| "Invalid native export filename.".to_string())?;
    Ok(default_file_name(&name))
}

fn filter_extensions<const N: usize>(values: [&str; N]) -> Vec<String> {
    values.into_iter().map(str::to_string).collect()
}

fn is_safe_filter_extension(extension: &str) -> bool {
    !extension.is_empty()
        && extension.len() <= 32
        && extension
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn save_filter(file_name: &str) -> (&'static str, Vec<String>) {
    match Path::new(file_name)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("json") => ("JSON", filter_extensions(["json"])),
        Some("jsonl") | Some("ndjson") => ("JSON Lines", filter_extensions(["jsonl", "ndjson"])),
        Some("csv") => ("CSV", filter_extensions(["csv"])),
        Some("md") | Some("markdown") => ("Markdown", filter_extensions(["md", "markdown"])),
        Some("html") | Some("htm") => ("HTML", filter_extensions(["html", "htm"])),
        Some("yaml") | Some("yml") => ("YAML", filter_extensions(["yaml", "yml"])),
        Some("py") => ("Python", filter_extensions(["py"])),
        Some("sh") => ("Shell script", filter_extensions(["sh"])),
        Some("js") | Some("jsx") => ("JavaScript", filter_extensions(["js", "jsx"])),
        Some("ts") | Some("tsx") => ("TypeScript", filter_extensions(["ts", "tsx"])),
        Some("sql") => ("SQL", filter_extensions(["sql"])),
        Some("zip") => ("ZIP archive", filter_extensions(["zip"])),
        // A name outside the active filter can be rejected or re-extensioned by the OS dialog.
        Some("txt") | Some("log") => ("Text", filter_extensions(["txt", "log"])),
        Some("png") => ("PNG image", filter_extensions(["png"])),
        Some("jpg") | Some("jpeg") => ("JPEG image", filter_extensions(["jpg", "jpeg"])),
        Some("webp") => ("WebP image", filter_extensions(["webp"])),
        Some("gif") => ("GIF image", filter_extensions(["gif"])),
        Some("svg") => ("SVG image", filter_extensions(["svg"])),
        Some("wav") => ("WAV audio", filter_extensions(["wav"])),
        Some("mp3") => ("MP3 audio", filter_extensions(["mp3"])),
        // Named for both tracks: the gallery saves .mp4 through this dialog too.
        Some("m4a") | Some("mp4") => ("MPEG-4 video or audio", filter_extensions(["m4a", "mp4"])),
        Some("ogg") | Some("oga") => ("Ogg audio", filter_extensions(["ogg", "oga"])),
        Some("flac") => ("FLAC audio", filter_extensions(["flac"])),
        Some("webm") => ("WebM video or audio", filter_extensions(["webm"])),
        Some(extension) if is_safe_filter_extension(extension) => {
            ("Export file", vec![extension.to_string()])
        }
        _ => (
            "Export files",
            filter_extensions([
                "json", "jsonl", "ndjson", "csv", "md", "markdown", "html", "htm", "yaml", "yml",
                "py", "sh", "js", "jsx", "ts", "tsx", "sql", "zip", "txt", "log", "png", "jpg",
                "jpeg", "webp", "gif", "svg", "wav", "mp3", "m4a", "mp4", "ogg", "oga", "flac",
                "webm",
            ]),
        ),
    }
}

fn invoke_body_bytes(body: &tauri::ipc::InvokeBody) -> Option<Cow<'_, [u8]>> {
    match body {
        tauri::ipc::InvokeBody::Raw(content) => Some(Cow::Borrowed(content)),
        tauri::ipc::InvokeBody::Json(value) => value
            .as_array()?
            .iter()
            .map(|item| u8::try_from(item.as_u64()?).ok())
            .collect::<Option<Vec<_>>>()
            .map(Cow::Owned),
    }
}

fn local_dialog_path(path: tauri_plugin_dialog::FilePath) -> Result<PathBuf, String> {
    path.into_path()
        .map_err(|_| "Only local filesystem paths are supported.".to_string())
}

/// Stage the write beside the destination so a partial file never replaces a real one.
pub(crate) fn staged_temp_file(path: &Path) -> Result<tempfile::NamedTempFile, String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut builder = tempfile::Builder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let permissions = fs::metadata(path)
            .map(|metadata| metadata.permissions())
            .unwrap_or_else(|_| fs::Permissions::from_mode(0o666));
        builder.permissions(permissions);
    }
    builder
        .prefix(".unsloth-export-")
        .tempfile_in(parent)
        .map_err(|error| format!("Failed to prepare {}: {error}", path.display()))
}

fn saved_file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("export")
        .to_string()
}

fn save_selected_file(
    selected_path: Option<PathBuf>,
    content: &[u8],
) -> Result<Option<PathBuf>, String> {
    let Some(path) = selected_path else {
        return Ok(None);
    };
    let mut temporary = staged_temp_file(&path)?;
    temporary
        .write_all(content)
        .and_then(|()| temporary.as_file().sync_all())
        .map_err(|error| format!("Failed to save {}: {error}", path.display()))?;
    temporary
        .persist(&path)
        .map_err(|error| format!("Failed to save {}: {}", path.display(), error.error))?;
    Ok(Some(path))
}

fn native_save_chunk(body: &tauri::ipc::InvokeBody) -> Result<Cow<'_, [u8]>, String> {
    let content = invoke_body_bytes(body)
        .ok_or_else(|| "Native export content must be binary.".to_string())?;
    if content.len() > MAX_NATIVE_FILE_SAVE_CHUNK_BYTES {
        return Err("Native export chunk is too large.".to_string());
    }
    Ok(content)
}

fn append_native_save(save: &Arc<Mutex<NativeSave>>, content: &[u8]) -> Result<(), String> {
    let mut save = save
        .lock()
        .map_err(|_| "Native export handle is unavailable.".to_string())?;
    let destination = save.destination.clone();
    save.temporary
        .write_all(content)
        .map_err(|error| format!("Failed to save {}: {error}", destination.display()))
}

fn finish_native_save(save: Arc<Mutex<NativeSave>>) -> Result<String, String> {
    let save = Arc::try_unwrap(save)
        .map_err(|_| "Native export is still being written.".to_string())?
        .into_inner()
        .map_err(|_| "Native export handle is unavailable.".to_string())?;
    let NativeSave {
        destination,
        mut temporary,
    } = save;
    temporary
        .flush()
        .and_then(|()| temporary.as_file().sync_all())
        .map_err(|error| format!("Failed to save {}: {error}", destination.display()))?;
    let name = saved_file_name(&destination);
    temporary
        .persist(&destination)
        .map_err(|error| format!("Failed to save {}: {}", destination.display(), error.error))?;
    Ok(name)
}

/// Only the local backend. Parsed, not sliced: in `http://127.0.0.1:8888@evil.test/clip` the
/// loopback-looking part is userinfo.
fn require_loopback_url(url: &str) -> Result<(), String> {
    const REJECT: &str = "Only local http URLs can be saved.";
    let parsed = reqwest::Url::parse(url).map_err(|_| REJECT.to_string())?;
    if parsed.scheme() != "http" || !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(REJECT.to_string());
    }
    let host = parsed.host_str().ok_or_else(|| REJECT.to_string())?;
    // host_str keeps the brackets on an IPv6 literal.
    let bare = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host);
    let loopback = bare
        .parse::<std::net::IpAddr>()
        .map(|ip| ip.is_loopback())
        .unwrap_or(host == "localhost");
    if loopback {
        Ok(())
    } else {
        Err(REJECT.to_string())
    }
}

fn read_selected_text_import(
    selected_path: Option<PathBuf>,
    label: &str,
    extensions: &[&str],
    extension_description: &str,
    fallback_name: &str,
    max_bytes: u64,
) -> Result<Option<NativeImportedFile>, String> {
    let Some(path) = selected_path else {
        return Ok(None);
    };
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| format!("{label} must be a {extension_description} file."))?;
    if !extensions.contains(&extension.as_str()) {
        return Err(format!("{label} must be a {extension_description} file."));
    }

    let metadata = fs::metadata(&path)
        .map_err(|error| format!("Failed to inspect {}: {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!("{label} is not a file: {}", path.display()));
    }
    if metadata.len() > max_bytes {
        return Err(format!(
            "{label} is too large (maximum {} MiB).",
            max_bytes / 1024 / 1024
        ));
    }

    // Limit the read too, so a file that grows after the stat cannot force unbounded allocation.
    let file =
        File::open(&path).map_err(|error| format!("Failed to open {}: {error}", path.display()))?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(max_bytes + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
    if bytes.len() as u64 > max_bytes {
        return Err(format!(
            "{label} is too large (maximum {} MiB).",
            max_bytes / 1024 / 1024
        ));
    }
    let content = String::from_utf8(bytes)
        .map_err(|_| format!("{label} is not valid UTF-8: {}", path.display()))?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| format!("{fallback_name}.{extension}"));
    Ok(Some(NativeImportedFile { name, content }))
}

fn open_selected_import(
    registry: &ChatImportRegistry,
    selected_path: Option<PathBuf>,
) -> Result<Option<NativeChatImport>, String> {
    let Some(path) = selected_path else {
        return Ok(None);
    };
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| CHAT_IMPORT_TYPE_ERROR.to_string())?;
    if !CHAT_IMPORT_EXTENSIONS.contains(&extension.as_str()) {
        return Err(CHAT_IMPORT_TYPE_ERROR.to_string());
    }

    let metadata = fs::metadata(&path)
        .map_err(|error| format!("Failed to inspect {}: {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!("Chat import is not a file: {}", path.display()));
    }

    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| format!("chat-import.{extension}"));
    let file =
        File::open(&path).map_err(|error| format!("Failed to open {}: {error}", path.display()))?;
    // Size from the token's handle, not the path, since this bounds the streamed read.
    let opened = file
        .metadata()
        .map_err(|error| format!("Failed to inspect {}: {error}", path.display()))?;
    if !opened.is_file() {
        return Err(format!("Chat import is not a file: {}", path.display()));
    }
    let size = opened.len();
    let token = registry.register(path, file);
    Ok(Some(NativeChatImport { name, size, token }))
}

fn read_selected_training_config(
    selected_path: Option<PathBuf>,
) -> Result<Option<NativeImportedFile>, String> {
    read_selected_text_import(
        selected_path,
        "Training config",
        TRAINING_CONFIG_EXTENSIONS,
        ".yaml or .yml",
        "training-config",
        MAX_TRAINING_CONFIG_BYTES,
    )
}

#[tauri::command]
pub async fn save_native_file(
    webview: tauri::Webview,
    app: AppHandle,
    request: tauri::ipc::Request<'_>,
) -> Result<Option<String>, String> {
    crate::native_intents::ensure_main_window(&webview)?;
    Ok(save_request_with_dialog(&app, &request, None)
        .await?
        .map(|path| saved_file_name(&path)))
}

fn request_file<'a>(
    request: &'a tauri::ipc::Request<'_>,
) -> Result<(String, Cow<'a, [u8]>), String> {
    let encoded_name = request
        .headers()
        .get(NATIVE_FILE_NAME_HEADER)
        .ok_or_else(|| "Native export filename is missing.".to_string())?
        .to_str()
        .map_err(|_| "Invalid native export filename.".to_string())?;
    let file_name = decode_default_file_name(encoded_name)?;
    let content = invoke_body_bytes(request.body())
        .ok_or_else(|| "Native export content must be binary.".to_string())?;
    Ok((file_name, content))
}

/// Save the request body where the user picks, starting in `directory`; None if cancelled.
pub(crate) async fn save_request_with_dialog(
    app: &AppHandle,
    request: &tauri::ipc::Request<'_>,
    directory: Option<&Path>,
) -> Result<Option<PathBuf>, String> {
    let (file_name, content) = request_file(request)?;
    let (filter_name, extensions) = save_filter(&file_name);
    let extension_refs = extensions.iter().map(String::as_str).collect::<Vec<_>>();
    let (tx, rx) = tokio::sync::oneshot::channel();
    let mut dialog = app
        .dialog()
        .file()
        .set_title("Save Unsloth export")
        .set_file_name(file_name)
        .add_filter(filter_name, &extension_refs);
    if let Some(directory) = directory {
        dialog = dialog.set_directory(directory);
    }
    dialog.save_file(move |path| {
        let _ = tx.send(path);
    });
    let selected_path = rx
        .await
        .map_err(|_| "Save dialog closed unexpectedly.".to_string())?
        .map(local_dialog_path)
        .transpose()?;
    save_selected_file(selected_path, content.as_ref())
}

#[tauri::command]
pub async fn begin_native_file_save(
    webview: tauri::Webview,
    app: AppHandle,
    registry: State<'_, NativeSaveRegistry>,
    file_name: String,
) -> Result<Option<String>, String> {
    crate::native_intents::ensure_main_window(&webview)?;
    let file_name = default_file_name(&file_name);
    let (filter_name, extensions) = save_filter(&file_name);
    let extension_refs = extensions.iter().map(String::as_str).collect::<Vec<_>>();
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog()
        .file()
        .set_title("Save Unsloth export")
        .set_file_name(file_name)
        .add_filter(filter_name, &extension_refs)
        .save_file(move |path| {
            let _ = tx.send(path);
        });
    let selected_path = rx
        .await
        .map_err(|_| "Save dialog closed unexpectedly.".to_string())?
        .map(local_dialog_path)
        .transpose()?;
    let Some(destination) = selected_path else {
        return Ok(None);
    };
    let temporary = staged_temp_file(&destination)?;
    Ok(Some(registry.register(NativeSave {
        destination,
        temporary,
    })))
}

#[tauri::command]
pub async fn append_native_file_save_chunk(
    webview: tauri::Webview,
    registry: State<'_, NativeSaveRegistry>,
    request: tauri::ipc::Request<'_>,
) -> Result<(), String> {
    crate::native_intents::ensure_main_window(&webview)?;
    let token = request
        .headers()
        .get(NATIVE_FILE_SAVE_TOKEN_HEADER)
        .ok_or_else(|| "Native export token is missing.".to_string())?
        .to_str()
        .map_err(|_| "Invalid native export token.".to_string())?;
    let save = registry
        .resolve(token)
        .ok_or_else(|| "That native export is no longer available.".to_string())?;
    let content = native_save_chunk(request.body())?.into_owned();
    tokio::task::spawn_blocking(move || append_native_save(&save, &content))
        .await
        .map_err(|error| format!("Failed to write the native export: {error}"))?
}

#[tauri::command]
pub async fn finish_native_file_save(
    webview: tauri::Webview,
    registry: State<'_, NativeSaveRegistry>,
    token: String,
) -> Result<String, String> {
    crate::native_intents::ensure_main_window(&webview)?;
    let save = registry
        .take(&token)
        .ok_or_else(|| "That native export is no longer available.".to_string())?;
    tokio::task::spawn_blocking(move || finish_native_save(save))
        .await
        .map_err(|error| format!("Failed to finish the native export: {error}"))?
}

#[tauri::command]
pub fn cancel_native_file_save(
    webview: tauri::Webview,
    registry: State<'_, NativeSaveRegistry>,
    token: String,
) -> Result<(), String> {
    crate::native_intents::ensure_main_window(&webview)?;
    registry.cancel(&token);
    Ok(())
}

/// Room under the usual 255-byte name limit for the " (999)" a taken name gets.
const MAX_DOWNLOAD_NAME_BYTES: usize = 240;
const WINDOWS_DEVICE_NAMES: [&str; 22] = [
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
    "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

fn is_bidi_control(c: char) -> bool {
    matches!(
        c,
        '\u{061c}' | '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}'
    )
}

/// A website's file name made safe everywhere: no separators, control or reserved characters, trailing dots/spaces or device names; length capped.
pub(crate) fn safe_download_name(name: &str) -> String {
    let mut name: String = name
        .chars()
        .map(|c| {
            // Bidi controls too: `invoice\u{202e}fdp.exe` must not read as `invoiceexe.pdf`.
            if c.is_control() || is_bidi_control(c) || "/\\:*?\"<>|".contains(c) {
                '_'
            } else {
                c
            }
        })
        .collect();
    let kept = name.trim_end_matches(['.', ' ']).len();
    name.truncate(kept);
    if name.trim_matches(['.', ' ']).is_empty() {
        return "download".into();
    }
    let device = name.split('.').next().unwrap_or("").trim_end();
    if WINDOWS_DEVICE_NAMES
        .iter()
        .any(|reserved| reserved.eq_ignore_ascii_case(device))
    {
        name.insert(0, '_');
    }
    if name.len() > MAX_DOWNLOAD_NAME_BYTES {
        let extension = name
            .rfind('.')
            .filter(|&at| at > 0 && name.len() - at <= 32)
            .map_or("", |at| &name[at..]);
        let mut end = MAX_DOWNLOAD_NAME_BYTES - extension.len();
        while !name.is_char_boundary(end) {
            end -= 1;
        }
        name = format!("{}{extension}", &name[..end]);
    }
    name
}

pub(crate) fn save_request_in(
    request: &tauri::ipc::Request<'_>,
    directory: &Path,
) -> Result<PathBuf, String> {
    let (file_name, content) = request_file(request)?;
    let path = unique_destination(directory, &safe_download_name(&file_name))?;
    save_selected_file(Some(path), content.as_ref())?
        .ok_or_else(|| "Failed to save the file.".to_string())
}

/// Save a backend URL by streaming it to the chosen path.
///
/// `save_native_file` carries the bytes through IPC, so the caller buffers the whole body
/// and the chooser waits on it. A clip is capped at 2048x2048 by 1024 frames, so this opens
/// the chooser first and writes the response chunk by chunk, leaving nothing resident.
#[tauri::command]
pub async fn save_native_file_from_url(
    webview: tauri::Webview,
    app: AppHandle,
    url: String,
    file_name: String,
) -> Result<Option<String>, String> {
    crate::native_intents::ensure_main_window(&webview)?;
    require_loopback_url(&url)?;
    let file_name = default_file_name(&file_name);
    let (filter_name, extensions) = save_filter(&file_name);
    let extension_refs = extensions.iter().map(String::as_str).collect::<Vec<_>>();
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog()
        .file()
        .set_title("Save Unsloth export")
        .set_file_name(file_name)
        .add_filter(filter_name, &extension_refs)
        .save_file(move |path| {
            let _ = tx.send(path);
        });
    let selected_path = rx
        .await
        .map_err(|_| "Save dialog closed unexpectedly.".to_string())?
        .map(local_dialog_path)
        .transpose()?;
    let Some(path) = selected_path else {
        return Ok(None);
    };
    stream_url_to_path(&url, &path, DOWNLOAD_READ_TIMEOUT, None).await?;
    Ok(Some(saved_file_name(&path)))
}

/// Resets per chunk, so a large save is not cut short.
const DOWNLOAD_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// `bearer` is always minted by the command, never passed over IPC, so the webview cannot choose
/// what this client authenticates as; see `download_logs_to_downloads`.
async fn stream_url_to_path(
    url: &str,
    path: &Path,
    read_timeout: Duration,
    bearer: Option<&str>,
) -> Result<(), String> {
    let request = crate::loopback_http::streaming_client(Duration::from_secs(10), read_timeout)
        .map_err(|error| format!("Download failed: {error}"))?
        .get(url);
    let request = match bearer {
        Some(token) => request.bearer_auth(token),
        None => request,
    };
    let mut response = request
        .send()
        .await
        .map_err(|error| format!("Download failed: {error}"))?;
    // Redirects are refused, so a 3xx is a rejection here, not a hop.
    if !response.status().is_success() {
        return Err(format!(
            "Download failed with status {}.",
            response.status().as_u16()
        ));
    }
    let mut temporary = staged_temp_file(path)?;
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| format!("Download failed: {error}"))?
    {
        temporary
            .write_all(&chunk)
            .map_err(|error| format!("Failed to save {}: {error}", path.display()))?;
    }
    temporary
        .as_file()
        .sync_all()
        .map_err(|error| format!("Failed to save {}: {error}", path.display()))?;
    temporary
        .persist(path)
        .map_err(|error| format!("Failed to save {}: {}", path.display(), error.error))?;
    Ok(())
}

/// The only route `download_logs_to_downloads` may fetch: its bearer is a full UI session.
/// Suffix match so a base path still resolves.
const LOG_EXPORT_ROUTE: &str = "/api/settings/debug/logs/export";
const LOG_ARCHIVE_FALLBACK_NAME: &str = "unsloth-logs.zip";
const LOG_ARCHIVE_MAX_COPIES: u32 = 999;

/// No chooser guards this destination, and on Unix `..\..\x` survives `Path::file_name`.
fn log_archive_file_name(suggested: &str) -> String {
    let base = suggested
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or_default()
        // A drive-relative name like `C:evil.zip` would still resolve off this directory.
        .rsplit(':')
        .next()
        .unwrap_or_default();
    if base.is_empty() || base == "." || base == ".." || base.contains('\0') {
        return LOG_ARCHIVE_FALLBACK_NAME.to_string();
    }
    // The destination may fall back to $HOME and the name comes from the webview: refuse dotfiles
    // and non-.zip names (the real caller sends `unsloth-logs-<stamp>.zip`).
    if base.starts_with('.') || !base.ends_with(".zip") {
        return LOG_ARCHIVE_FALLBACK_NAME.to_string();
    }
    base.to_string()
}

fn log_archive_directory() -> Result<PathBuf, String> {
    let directory = dirs::download_dir()
        .or_else(dirs::home_dir)
        .ok_or_else(|| "Could not determine home directory".to_string())?;
    // XDG can name a Downloads folder that was never created.
    fs::create_dir_all(&directory)
        .map_err(|error| format!("Failed to prepare {}: {error}", directory.display()))?;
    Ok(directory)
}

/// `name.zip`, then `name (2).zip`, the way a browser download uniquifies.
///
/// Exporting twice must not silently replace the archive the user is still attaching to
/// an issue. Best effort by nature -- another process can take the name between the check
/// and the rename -- but it removes the case that actually happens, which is the same
/// user pressing the button again.
pub(crate) fn unique_destination(directory: &Path, file_name: &str) -> Result<PathBuf, String> {
    let candidate = directory.join(file_name);
    if !candidate.exists() {
        return Ok(candidate);
    }
    let name = Path::new(file_name);
    let stem = name
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(file_name);
    let extension = name.extension().and_then(|extension| extension.to_str());
    for copy in 2..=LOG_ARCHIVE_MAX_COPIES {
        let candidate = directory.join(match extension {
            Some(extension) => format!("{stem} ({copy}).{extension}"),
            None => format!("{stem} ({copy})"),
        });
        if !candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(format!(
        "Could not find a free name for {file_name} in {}.",
        directory.display()
    ))
}

const NOT_THE_LOG_EXPORT: &str = "Only the local log export endpoint can be downloaded.";

/// Matched by `DESKTOP_LOGIN_REQUIRED` in features/settings/api/debug-logs.ts; keep the two in step.
const LOGIN_REQUIRED: &str = "Log export requires a signed-in Unsloth session.";

/// A minted desktop session where one exists, otherwise the tab's own token: desktop-login
/// refuses on multi-account installs and can fail under a custom home. A minting error wins
/// when there is no fallback.
fn select_export_session(
    minted: Result<crate::desktop_auth::DesktopAuthResponse, String>,
    ui_token: Option<String>,
) -> Result<String, String> {
    let fallback = ui_token.filter(|token| !token.trim().is_empty());
    match minted {
        Ok(crate::desktop_auth::DesktopAuthResponse::Tokens { access_token, .. }) => {
            Ok(access_token)
        }
        Ok(crate::desktop_auth::DesktopAuthResponse::LoginRequired { .. }) => {
            fallback.ok_or_else(|| LOGIN_REQUIRED.to_string())
        }
        Err(error) => fallback.ok_or(error),
    }
}

/// `Url::parse` normalises `..` segments, so a path walking out of the route fails here.
fn require_log_export_route(url: &str) -> Result<reqwest::Url, String> {
    let parsed = reqwest::Url::parse(url).map_err(|_| NOT_THE_LOG_EXPORT.to_string())?;
    if !parsed.path().ends_with(LOG_EXPORT_ROUTE) {
        return Err(NOT_THE_LOG_EXPORT.to_string());
    }
    Ok(parsed)
}

/// Rebase onto the live port so the token cannot be aimed at another loopback listener; this also
/// fixes a `127.0.0.1:0` placeholder base.
fn pin_to_backend(mut url: reqwest::Url, port: u16) -> Result<String, String> {
    url.set_host(Some("127.0.0.1"))
        .map_err(|_| NOT_THE_LOG_EXPORT.to_string())?;
    url.set_port(Some(port))
        .map_err(|_| NOT_THE_LOG_EXPORT.to_string())?;
    // The whole path: the bearer is a full UI session and the backend has catch-all routes.
    url.set_path(LOG_EXPORT_ROUTE);
    // The export takes no parameters, so anything here came from the caller.
    url.set_query(None);
    url.set_fragment(None);
    Ok(url.to_string())
}

/// Read *after* minting, so any port discovery minting did is reflected.
fn live_backend_port(state: &State<'_, crate::process::BackendState>) -> Result<u16, String> {
    let backend = state.lock().map_err(|error| error.to_string())?;
    if let Some(port) = backend.owned_backend_port() {
        return Ok(port);
    }
    // An owned backend with no known port is starting; the cached field may be stale.
    if backend.has_owned_backend() {
        return Err("Backend is not ready".to_string());
    }
    backend
        .port
        .ok_or_else(|| "Backend is not ready".to_string())
}

/// Compiled on every platform so the UNC branch is tested in ordinary CI. The prefix compare is
/// case-insensitive so a lowercase `unc` does not come out as a relative-looking path.
// Off Windows only the test uses it; kept compiled so the UNC case is tested.
#[cfg_attr(not(windows), allow(dead_code))]
fn strip_verbatim_prefix_inner(text: String) -> String {
    const UNC: &str = r"\\?\UNC\";
    // `get`, not slicing: a non-ASCII first component can put byte 8 inside a character.
    if text
        .get(..UNC.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(UNC))
    {
        return format!(r"\\{}", &text[UNC.len()..]);
    }
    match text.strip_prefix(r"\\?\") {
        Some(rest) => rest.to_string(),
        None => text,
    }
}

/// Strip the `\\?\` verbatim form for display. Identity elsewhere.
#[cfg(windows)]
fn strip_verbatim_prefix(text: String) -> String {
    strip_verbatim_prefix_inner(text)
}

#[cfg(not(windows))]
fn strip_verbatim_prefix(text: String) -> String {
    text
}

/// The absolute, symlink-resolved path to show the user.
pub(crate) fn display_path(path: &Path) -> String {
    let resolved = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    strip_verbatim_prefix(resolved.display().to_string())
}

/// Save the backend's log archive straight into Downloads, returning the real path. The bearer is
/// minted here and the port and route are pinned, so the webview cannot aim auth elsewhere.
/// `filename` is one word on purpose: Tauri camelCases params over IPC (`ui_token` arrives as
/// `uiToken`) and the Logs tab sends `filename`.
#[tauri::command]
pub async fn download_logs_to_downloads(
    webview: tauri::Webview,
    state: State<'_, crate::process::BackendState>,
    diagnostics: State<'_, crate::diagnostics::DiagnosticsState>,
    url: String,
    filename: String,
    ui_token: Option<String>,
) -> Result<String, String> {
    crate::native_intents::ensure_main_window(&webview)?;
    // Both guards run before anything is minted.
    require_loopback_url(&url)?;
    let target = require_log_export_route(&url)?;

    // Mint either way: it also resolves and caches the live port. The tab-token fallback grants
    // nothing new, since host, port and path are pinned.
    let minted = crate::desktop_auth::desktop_auth(state.clone(), diagnostics).await;
    let session = select_export_session(minted, ui_token)?;
    let pinned_url = pin_to_backend(target, live_backend_port(&state)?)?;

    let directory = log_archive_directory()?;
    let destination = unique_destination(&directory, &log_archive_file_name(&filename))?;
    // Staged and renamed, so an interrupted download never leaves a truncated archive.
    stream_url_to_path(
        &pinned_url,
        &destination,
        DOWNLOAD_READ_TIMEOUT,
        Some(&session),
    )
    .await?;
    Ok(display_path(&destination))
}

fn read_range(
    handle: &Arc<Mutex<File>>,
    path: &Path,
    offset: u64,
    length: usize,
) -> Result<Vec<u8>, String> {
    // Clamped here so the read itself is bounded for every caller, not just the allocation.
    let length = length.min(MAX_CHAT_IMPORT_CHUNK_BYTES);
    let mut file = handle
        .lock()
        .map_err(|_| format!("Failed to read {}: handle poisoned", path.display()))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
    let mut bytes = Vec::with_capacity(length);
    Read::by_ref(&mut *file)
        .take(length as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
    Ok(bytes)
}

/// Raw bytes, so the frontend can decode UTF-8 across range boundaries.
#[tauri::command]
pub async fn read_native_chat_import_chunk(
    app: AppHandle,
    token: String,
    offset: u64,
    length: usize,
) -> Result<tauri::ipc::Response, String> {
    let registry = app.state::<ChatImportRegistry>();
    let (path, handle) = registry
        .resolve(&token)
        .ok_or_else(|| "That import is no longer available -- pick the file again.".to_string())?;

    let length = length.min(MAX_CHAT_IMPORT_CHUNK_BYTES);
    // Off the async runtime: an import is tens of these reads back to back.
    let bytes = tokio::task::spawn_blocking(move || read_range(&handle, &path, offset, length))
        .await
        .map_err(|error| format!("Failed to read the import: {error}"))??;
    Ok(tauri::ipc::Response::new(bytes))
}

#[tauri::command]
pub async fn pick_native_chat_import(
    webview: tauri::Webview,
    app: AppHandle,
    registry: State<'_, ChatImportRegistry>,
) -> Result<Option<NativeChatImport>, String> {
    crate::native_intents::ensure_main_window(&webview)?;
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog()
        .file()
        .set_title("Import chats")
        .add_filter("Chat exports", CHAT_IMPORT_EXTENSIONS)
        .pick_file(move |path| {
            let _ = tx.send(path);
        });
    let selected_path = rx
        .await
        .map_err(|_| "Import dialog closed unexpectedly.".to_string())?
        .map(local_dialog_path)
        .transpose()?;
    open_selected_import(&registry, selected_path)
}

#[tauri::command]
pub async fn pick_native_training_config(
    webview: tauri::Webview,
    app: AppHandle,
) -> Result<Option<NativeImportedFile>, String> {
    crate::native_intents::ensure_main_window(&webview)?;
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog()
        .file()
        .set_title("Load training config")
        .add_filter("YAML", TRAINING_CONFIG_EXTENSIONS)
        .pick_file(move |path| {
            let _ = tx.send(path);
        });
    let selected_path = rx
        .await
        .map_err(|_| "Import dialog closed unexpectedly.".to_string())?
        .map(local_dialog_path)
        .transpose()?;
    read_selected_training_config(selected_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn download_names_are_safe_on_every_platform() {
        assert_eq!(safe_download_name("report:2026.pdf"), "report_2026.pdf");
        assert_eq!(
            safe_download_name("a<b>c|d?e*f\"g.txt"),
            "a_b_c_d_e_f_g.txt"
        );
        assert_eq!(safe_download_name("evil\u{7}name.sh"), "evil_name.sh");
        assert_eq!(
            safe_download_name("invoice\u{202e}fdp.exe"),
            "invoice_fdp.exe"
        );
        assert_eq!(safe_download_name("a\u{2066}b\u{061c}.txt"), "a_b_.txt");
        assert_eq!(safe_download_name("CON"), "_CON");
        assert_eq!(safe_download_name("nul.tar.gz"), "_nul.tar.gz");
        assert_eq!(safe_download_name("console.log"), "console.log");
        assert_eq!(safe_download_name("notes. . "), "notes");
        assert_eq!(safe_download_name(".."), "download");
        assert_eq!(safe_download_name(" "), "download");
        let long = format!("{}.pdf", "\u{3042}".repeat(255));
        let safe = safe_download_name(&long);
        assert!(safe.len() <= MAX_DOWNLOAD_NAME_BYTES, "{}", safe.len());
        assert!(safe.ends_with("\u{3042}.pdf"));
    }
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> PathBuf {
        // Parallel tests plus macOS's microsecond clock can collide on names; the counter keeps
        // them unique.
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "unsloth-native-files-{name}-{}-{nanos}-{seq}",
            std::process::id()
        ))
    }

    fn assert_save_filter(file_name: &str, name: &str, expected: &[&str]) {
        let (actual_name, actual_extensions) = save_filter(file_name);
        assert_eq!(actual_name, name);
        assert_eq!(
            actual_extensions
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            expected
        );
    }

    #[test]
    fn cancellation_is_quiet_for_save_and_import() {
        assert!(save_selected_file(None, b"x").unwrap().is_none());
        let registry = ChatImportRegistry::default();
        assert!(open_selected_import(&registry, None).unwrap().is_none());
    }

    #[test]
    fn accepts_raw_and_json_byte_bodies() {
        let raw = tauri::ipc::InvokeBody::Raw(vec![1, 2, 250]);
        assert_eq!(
            invoke_body_bytes(&raw).as_deref(),
            Some([1, 2, 250].as_slice())
        );

        let json = tauri::ipc::InvokeBody::Json(serde_json::json!([1, 2, 250]));
        assert_eq!(
            invoke_body_bytes(&json).as_deref(),
            Some([1, 2, 250].as_slice())
        );

        for value in [
            serde_json::json!({"content": "hi"}),
            serde_json::json!([1, 256]),
            serde_json::json!([1, -2]),
        ] {
            let body = tauri::ipc::InvokeBody::Json(value);
            assert!(invoke_body_bytes(&body).is_none());
        }
    }

    #[test]
    fn writes_text_and_binary_exactly() {
        let text_path = temp_path("text").with_extension("json");
        let binary_path = temp_path("binary").with_extension("zip");

        fs::write(&text_path, b"previous export").unwrap();
        save_selected_file(Some(text_path.clone()), b"{\"ok\":true}").unwrap();
        save_selected_file(Some(binary_path.clone()), &[0, 1, 2, 255]).unwrap();
        assert_eq!(fs::read(&text_path).unwrap(), b"{\"ok\":true}");
        assert_eq!(fs::read(&binary_path).unwrap(), [0, 1, 2, 255]);
        let _ = fs::remove_file(text_path);
        let _ = fs::remove_file(binary_path);
    }

    #[test]
    fn chunked_save_keeps_the_destination_atomic() {
        let destination = temp_path("chunked").with_extension("zip");
        fs::write(&destination, b"previous export").unwrap();
        let registry = NativeSaveRegistry::default();
        let token = registry.register(NativeSave {
            temporary: staged_temp_file(&destination).unwrap(),
            destination: destination.clone(),
        });
        let save = registry.resolve(&token).unwrap();

        append_native_save(&save, b"first ").unwrap();
        append_native_save(&save, b"second").unwrap();
        assert_eq!(fs::read(&destination).unwrap(), b"previous export");
        drop(save);

        let completed = registry.take(&token).unwrap();
        assert_eq!(
            finish_native_save(completed).unwrap(),
            saved_file_name(&destination)
        );
        assert_eq!(fs::read(&destination).unwrap(), b"first second");
        assert!(registry.resolve(&token).is_none());
        let _ = fs::remove_file(destination);
    }

    #[test]
    fn cancelled_chunked_save_preserves_the_destination() {
        let destination = temp_path("chunked-cancel").with_extension("zip");
        fs::write(&destination, b"keep this").unwrap();
        let registry = NativeSaveRegistry::default();
        let token = registry.register(NativeSave {
            temporary: staged_temp_file(&destination).unwrap(),
            destination: destination.clone(),
        });
        append_native_save(&registry.resolve(&token).unwrap(), b"discard this").unwrap();

        registry.cancel(&token);
        assert_eq!(fs::read(&destination).unwrap(), b"keep this");
        assert!(registry.resolve(&token).is_none());
        let _ = fs::remove_file(destination);
    }

    #[test]
    fn native_save_chunks_are_bounded() {
        let accepted = tauri::ipc::InvokeBody::Raw(vec![0; MAX_NATIVE_FILE_SAVE_CHUNK_BYTES]);
        assert_eq!(
            native_save_chunk(&accepted).unwrap().len(),
            MAX_NATIVE_FILE_SAVE_CHUNK_BYTES
        );

        let refused = tauri::ipc::InvokeBody::Raw(vec![0; MAX_NATIVE_FILE_SAVE_CHUNK_BYTES + 1]);
        assert_eq!(
            native_save_chunk(&refused).unwrap_err(),
            "Native export chunk is too large."
        );
    }

    #[test]
    fn markdown_exports_use_a_markdown_save_filter() {
        assert_save_filter("message.md", "Markdown", &["md", "markdown"]);
    }

    #[test]
    fn training_configs_use_a_yaml_save_filter() {
        assert_save_filter("training.yaml", "YAML", &["yaml", "yml"]);
        assert_save_filter("training.YML", "YAML", &["yaml", "yml"]);
    }

    #[test]
    fn html_canvas_exports_use_an_html_save_filter() {
        assert_save_filter("canvas.html", "HTML", &["html", "htm"]);
        assert_save_filter("canvas.HTM", "HTML", &["html", "htm"]);
    }

    #[test]
    fn python_scripts_use_a_python_save_filter() {
        assert_save_filter("script.py", "Python", &["py"]);
        assert_save_filter("script.PY", "Python", &["py"]);
    }

    #[test]
    fn shell_commands_use_a_shell_save_filter() {
        assert_save_filter("command.sh", "Shell script", &["sh"]);
        assert_save_filter("command.SH", "Shell script", &["sh"]);
    }

    #[test]
    fn browser_generated_exports_keep_their_extension() {
        assert_save_filter("training.yaml", "YAML", &["yaml", "yml"]);
        assert_save_filter("snippet.TSX", "TypeScript", &["ts", "tsx"]);
        assert_save_filter("diagram.svg", "SVG image", &["svg"]);
        assert_save_filter("snippet.rs", "Export file", &["rs"]);

        let (name, extensions) = save_filter("snippet.bad!");
        assert_eq!(name, "Export files");
        assert!(!extensions.iter().any(|extension| extension == "bad!"));
    }

    #[test]
    fn saved_chat_attachments_keep_their_own_extension() {
        assert_save_filter("report.txt", "Text", &["txt", "log"]);
        assert_save_filter("photo.PNG", "PNG image", &["png"]);
        assert_save_filter("shot.jpeg", "JPEG image", &["jpg", "jpeg"]);
        assert_save_filter("clip.wav", "WAV audio", &["wav"]);
        assert_save_filter("voice.webm", "WebM video or audio", &["webm"]);
    }

    fn test_runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    fn serve_once(body: Vec<u8>, status: &'static str) -> (String, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut discard = [0_u8; 1024];
            let _ = std::io::Read::read(&mut stream, &mut discard);
            let header = format!(
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nContent-Type: video/mp4\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(&body);
        });
        (format!("http://127.0.0.1:{port}/clip.mp4"), handle)
    }

    #[test]
    fn streaming_save_writes_the_whole_body_without_buffering_it() {
        // Larger than any single chunk, so the loop is what assembles the file.
        let body: Vec<u8> = (0..3_000_000_u32).map(|i| (i % 251) as u8).collect();
        let (url, server) = serve_once(body.clone(), "200 OK");
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("clip.mp4");
        test_runtime()
            .block_on(stream_url_to_path(&url, &dest, DOWNLOAD_READ_TIMEOUT, None))
            .unwrap();
        server.join().unwrap();
        assert_eq!(fs::read(&dest).unwrap(), body);
        let strays: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name() != std::ffi::OsStr::new("clip.mp4"))
            .collect();
        assert!(strays.is_empty(), "staging file left behind");
    }

    #[test]
    fn a_failed_download_leaves_no_file_behind() {
        let (url, server) = serve_once(b"nope".to_vec(), "401 Unauthorized");
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("clip.mp4");
        let error = test_runtime()
            .block_on(stream_url_to_path(&url, &dest, DOWNLOAD_READ_TIMEOUT, None))
            .unwrap_err();
        server.join().unwrap();
        assert!(error.contains("401"), "{error}");
        assert!(!dest.exists(), "a rejected link must not create the file");
    }

    #[test]
    fn streaming_save_only_accepts_the_local_backend() {
        for url in [
            "http://127.0.0.1:8888/api/inference/video/gallery/abc/file-signed?token=t",
            "http://localhost:8908/api/inference/video/gallery/abc/file-signed?token=t",
            "http://[::1]:8888/api/inference/video/gallery/abc/file",
            "http://127.0.0.1/api/inference/video/gallery/abc/file",
        ] {
            assert!(require_loopback_url(url).is_ok(), "should allow {url}");
        }
        // Everything before the '@' is userinfo, so the real host is what follows it.
        for url in [
            "http://evil.test/x.mp4",
            "https://127.0.0.1:8888/x.mp4",
            "file:///etc/passwd",
            "http://127.0.0.1.evil.test/x.mp4",
            "http://user@evil.test/x.mp4",
            "http://127.0.0.1:8888@evil.test/video",
            "http://127.0.0.1@evil.test/video",
            "http://localhost:8888@evil.test/video",
            "http://[::1]:8888@evil.test/video",
            "http://10.0.0.5/x.mp4",
            "http://169.254.169.254/latest/meta-data",
            "",
        ] {
            assert!(require_loopback_url(url).is_err(), "should reject {url}");
        }
    }

    #[test]
    fn a_backend_that_goes_quiet_mid_body_stops_the_save() {
        // Headers promise more than is sent; without a per-read timeout the invoke never resolves.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (release, stalled) = std::sync::mpsc::channel::<()>();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut discard = [0_u8; 1024];
            let _ = std::io::Read::read(&mut stream, &mut discard);
            let _ = stream.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Length: 1024\r\nContent-Type: video/mp4\r\n\r\nhalf",
            );
            let _ = stalled.recv(); // hold the connection open until the read has given up
        });
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("clip.mp4");
        let error = test_runtime()
            .block_on(stream_url_to_path(
                &format!("http://127.0.0.1:{port}/clip.mp4"),
                &dest,
                Duration::from_millis(250),
                None,
            ))
            .unwrap_err();
        drop(release);
        server.join().unwrap();
        assert!(error.starts_with("Download failed"), "{error}");
        assert!(!dest.exists(), "a stalled download must not leave a file");
        assert_eq!(
            fs::read_dir(dir.path()).unwrap().count(),
            0,
            "the staging file must be cleaned up"
        );
    }

    #[test]
    fn a_redirect_off_loopback_is_refused_not_followed() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut discard = [0_u8; 1024];
            let _ = std::io::Read::read(&mut stream, &mut discard);
            let _ = stream.write_all(
                b"HTTP/1.1 302 Found\r\nLocation: http://evil.test/x.mp4\r\n\
                  Content-Length: 0\r\nConnection: close\r\n\r\n",
            );
        });
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("clip.mp4");
        let error = test_runtime()
            .block_on(stream_url_to_path(
                &format!("http://127.0.0.1:{port}/clip.mp4"),
                &dest,
                DOWNLOAD_READ_TIMEOUT,
                None,
            ))
            .unwrap_err();
        server.join().unwrap();
        assert!(error.contains("302"), "{error}");
        assert!(!dest.exists(), "a redirect must not produce a file");
    }

    #[test]
    fn a_gallery_clip_offers_its_own_container() {
        assert_save_filter(
            "Unsloth_video_20260808-120000_1670009728.mp4",
            "MPEG-4 video or audio",
            &["m4a", "mp4"],
        );
    }

    #[test]
    fn generic_fallback_covers_every_tool_download_name() {
        let (name, extensions) = save_filter("no-extension");
        assert_eq!(name, "Export files");
        for wanted in [
            "py", "sh", "js", "ts", "sql", "yaml", "json", "jsonl", "csv", "md", "html", "zip",
            "txt", "png", "jpg", "svg", "wav",
        ] {
            assert!(
                extensions.iter().any(|extension| extension == wanted),
                "fallback lost {wanted}"
            );
        }
    }

    #[test]
    fn opens_supported_imports_and_rejects_other_extensions() {
        let registry = ChatImportRegistry::default();
        let jsonl_path = temp_path("allowed").with_extension("JSONL");
        fs::write(&jsonl_path, "{\"messages\":[]}").unwrap();
        let opened = open_selected_import(&registry, Some(jsonl_path.clone()))
            .unwrap()
            .unwrap();
        assert_eq!(opened.size, 15);
        assert_eq!(registry.resolve(&opened.token).unwrap().0, jsonl_path);

        let json_path = temp_path("openwebui").with_extension("json");
        fs::write(&json_path, "[{\"chat\":{}}]").unwrap();
        assert!(open_selected_import(&registry, Some(json_path.clone()))
            .unwrap()
            .is_some());

        let txt_path = temp_path("denied").with_extension("txt");
        fs::write(&txt_path, "no").unwrap();
        assert!(open_selected_import(&registry, Some(txt_path.clone()))
            .unwrap_err()
            .contains(".json"));
        let _ = fs::remove_file(jsonl_path);
        let _ = fs::remove_file(json_path);
        let _ = fs::remove_file(txt_path);
    }

    #[test]
    fn reads_bounded_yaml_training_configs() {
        let yaml_path = temp_path("training-config").with_extension("YAML");
        fs::write(&yaml_path, "model_name: unsloth/test\n").unwrap();
        let imported = read_selected_training_config(Some(yaml_path.clone()))
            .unwrap()
            .unwrap();
        assert_eq!(imported.content, "model_name: unsloth/test\n");

        let json_path = temp_path("training-config-invalid").with_extension("json");
        fs::write(&json_path, "{}").unwrap();
        assert!(read_selected_training_config(Some(json_path.clone())).is_err());
        let directory = temp_path("training-config-directory").with_extension("yaml");
        fs::create_dir(&directory).unwrap();
        assert!(read_selected_training_config(Some(directory.clone()))
            .unwrap_err()
            .starts_with("Training config is not a file:"));
        let _ = fs::remove_file(yaml_path);
        let _ = fs::remove_file(json_path);
        let _ = fs::remove_dir(directory);
    }

    #[test]
    fn a_large_import_is_opened_rather_than_read_into_memory() {
        let registry = ChatImportRegistry::default();
        let huge = temp_path("huge").with_extension("json");
        let file = File::create(&huge).unwrap();
        file.set_len(600 * 1024 * 1024).unwrap();
        let opened = open_selected_import(&registry, Some(huge.clone()))
            .unwrap()
            .unwrap();
        assert_eq!(opened.size, 600 * 1024 * 1024);

        let directory = temp_path("import-directory").with_extension("json");
        fs::create_dir(&directory).unwrap();
        assert!(open_selected_import(&registry, Some(directory.clone()))
            .unwrap_err()
            .starts_with("Chat import is not a file:"));
        let _ = fs::remove_file(huge);
        let _ = fs::remove_dir(directory);
    }

    fn register_for_test(registry: &ChatImportRegistry, path: &Path) -> String {
        registry.register(path.to_path_buf(), File::open(path).unwrap())
    }

    #[test]
    fn ranges_are_readable_only_through_a_token_the_picker_issued() {
        let registry = ChatImportRegistry::default();
        let path = temp_path("ranges").with_extension("jsonl");
        fs::write(&path, "0123456789").unwrap();

        let token = register_for_test(&registry, &path);
        let (resolved, handle) = registry.resolve(&token).unwrap();
        assert_eq!(resolved, path);
        assert!(registry.resolve("not-a-token").is_none());

        assert_eq!(read_range(&handle, &path, 0, 4).unwrap(), b"0123");
        assert_eq!(read_range(&handle, &path, 4, 4).unwrap(), b"4567");
        assert_eq!(read_range(&handle, &path, 8, 4).unwrap(), b"89");
        assert!(read_range(&handle, &path, 10, 4).unwrap().is_empty());

        // Picking again must not invalidate an earlier handle that may still be streaming.
        let second = temp_path("ranges-2").with_extension("jsonl");
        fs::write(&second, "abc").unwrap();
        let newer = register_for_test(&registry, &second);
        assert_eq!(registry.resolve(&token).unwrap().0, path);
        assert_eq!(registry.resolve(&newer).unwrap().0, second);
        assert_ne!(token, newer);

        let big = temp_path("ranges-clamp").with_extension("jsonl");
        fs::write(&big, vec![b'x'; MAX_CHAT_IMPORT_CHUNK_BYTES + 4096]).unwrap();
        let big_token = register_for_test(&registry, &big);
        let (big_path, big_handle) = registry.resolve(&big_token).unwrap();
        assert_eq!(
            read_range(&big_handle, &big_path, 0, usize::MAX)
                .unwrap()
                .len(),
            MAX_CHAT_IMPORT_CHUNK_BYTES
        );
        let _ = fs::remove_file(big);

        for index in 0..CHAT_IMPORT_HANDLE_LIMIT {
            let filler = temp_path(&format!("ranges-fill-{index}")).with_extension("jsonl");
            fs::write(&filler, "x").unwrap();
            register_for_test(&registry, &filler);
            let _ = fs::remove_file(filler);
        }
        assert!(registry.resolve(&token).is_none());

        let _ = fs::remove_file(path);
        let _ = fs::remove_file(second);
    }

    #[cfg(unix)]
    #[test]
    fn a_file_swapped_under_the_path_mid_import_cannot_reach_the_stream() {
        // A same-size swap leaves no short read to catch, so the handle must be reused.
        let registry = ChatImportRegistry::default();
        let path = temp_path("swapped").with_extension("jsonl");
        fs::write(&path, "original--").unwrap();
        let token = register_for_test(&registry, &path);

        let replacement = temp_path("swapped-other").with_extension("jsonl");
        fs::write(&replacement, "replaced--").unwrap();
        fs::rename(&replacement, &path).unwrap();

        let (resolved, handle) = registry.resolve(&token).unwrap();
        assert_eq!(
            read_range(&handle, &resolved, 0, 10).unwrap(),
            b"original--"
        );
        assert_eq!(fs::read(&path).unwrap(), b"replaced--");

        let _ = fs::remove_file(path);
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_import_name_preserves_csv_extension() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let path = std::env::temp_dir().join(OsString::from_vec(vec![
            b'u', b'n', b's', b'l', b'o', b't', b'h', 0xff, b'.', b'c', b's', b'v',
        ]));
        // macOS rejects non-UTF-8 filenames, so skip where such a file cannot exist.
        if fs::write(&path, "role,content\nuser,hello\n").is_err() {
            return;
        }
        let registry = ChatImportRegistry::default();
        let opened = open_selected_import(&registry, Some(path.clone()))
            .unwrap()
            .unwrap();
        assert_eq!(opened.name, "chat-import.csv");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn a_suggested_log_archive_name_cannot_escape_the_download_folder() {
        let directory = tempfile::tempdir().unwrap();
        for suggested in [
            "../../.bashrc",
            "../unsloth-logs.zip",
            "/etc/cron.d/unsloth",
            "..\\..\\Startup\\unsloth.zip",
            "C:evil.zip",
            r"C:\Windows\System32\evil.zip",
            "sub/dir/unsloth-logs.zip",
            ".",
            "..",
            "",
            "\0",
        ] {
            let name = log_archive_file_name(suggested);
            assert!(
                !name.contains('/') && !name.contains('\\') && !name.contains(':'),
                "{suggested} produced {name}"
            );
            let resolved = directory.path().join(&name);
            assert_eq!(
                resolved.parent(),
                Some(directory.path()),
                "{suggested} escaped to {}",
                resolved.display()
            );
        }

        assert_eq!(
            log_archive_file_name("unsloth-logs-20260910-101500.zip"),
            "unsloth-logs-20260910-101500.zip"
        );
        assert_eq!(log_archive_file_name("../.."), LOG_ARCHIVE_FALLBACK_NAME);
        assert_eq!(log_archive_file_name(".bashrc"), LOG_ARCHIVE_FALLBACK_NAME);
        assert_eq!(log_archive_file_name(".ssh"), LOG_ARCHIVE_FALLBACK_NAME);
        assert_eq!(
            log_archive_file_name("unsloth-logs.zip.exe"),
            LOG_ARCHIVE_FALLBACK_NAME
        );
        assert_eq!(log_archive_file_name("notes.txt"), LOG_ARCHIVE_FALLBACK_NAME);
        assert_eq!(
            log_archive_file_name("unsloth-logs-20260910-101112.zip"),
            "unsloth-logs-20260910-101112.zip"
        );
    }

    #[test]
    fn a_second_export_gets_its_own_name_instead_of_replacing_the_first() {
        let directory = tempfile::tempdir().unwrap();
        let first = unique_destination(directory.path(), "unsloth-logs.zip").unwrap();
        assert_eq!(first, directory.path().join("unsloth-logs.zip"));

        fs::write(&first, b"first").unwrap();
        let second = unique_destination(directory.path(), "unsloth-logs.zip").unwrap();
        assert_eq!(second, directory.path().join("unsloth-logs (2).zip"));

        fs::write(&second, b"second").unwrap();
        assert_eq!(
            unique_destination(directory.path(), "unsloth-logs.zip").unwrap(),
            directory.path().join("unsloth-logs (3).zip")
        );
        assert_eq!(fs::read(&first).unwrap(), b"first");

        let plain = directory.path().join("logs");
        fs::write(&plain, b"x").unwrap();
        assert_eq!(
            unique_destination(directory.path(), "logs").unwrap(),
            directory.path().join("logs (2)")
        );

        for copy in 2..=LOG_ARCHIVE_MAX_COPIES {
            fs::write(directory.path().join(format!("full ({copy}).zip")), b"x").unwrap();
        }
        fs::write(directory.path().join("full.zip"), b"x").unwrap();
        assert!(unique_destination(directory.path(), "full.zip")
            .unwrap_err()
            .contains("free name"));
    }

    #[test]
    fn the_log_export_download_is_pinned_to_the_live_backend_and_its_own_route() {
        fn resolve(url: &str, port: u16) -> Result<String, String> {
            pin_to_backend(require_log_export_route(url)?, port)
        }

        assert_eq!(
            resolve("http://127.0.0.1:0/api/settings/debug/logs/export", 8890).unwrap(),
            "http://127.0.0.1:8890/api/settings/debug/logs/export"
        );
        assert_eq!(
            resolve(
                "http://localhost:9/base/api/settings/debug/logs/export",
                8890
            )
            .unwrap(),
            "http://127.0.0.1:8890/api/settings/debug/logs/export"
        );

        assert_eq!(
            resolve(
                "http://127.0.0.1:0/api/settings/debug/logs/export?x=1#frag",
                8890
            )
            .unwrap(),
            "http://127.0.0.1:8890/api/settings/debug/logs/export"
        );
        assert_eq!(
            resolve("http://127.0.0.1:0/api/v1/api/settings/debug/logs/export", 8890).unwrap(),
            "http://127.0.0.1:8890/api/settings/debug/logs/export"
        );

        for url in [
            "http://127.0.0.1:8888/api/settings",
            "http://127.0.0.1:8888/api/settings/debug/logs",
            "http://127.0.0.1:8888/api/settings/debug/logs/export/../../secrets",
            "http://127.0.0.1:8888/api/settings/debug/logs/export%2fmore",
            "http://127.0.0.1:8888/api/auth/api-keys",
            "not a url",
        ] {
            assert!(resolve(url, 8890).is_err(), "should reject {url}");
        }

        for url in [
            "http://evil.test/api/settings/debug/logs/export",
            "https://127.0.0.1:8888/api/settings/debug/logs/export",
            "http://127.0.0.1:8888@evil.test/api/settings/debug/logs/export",
            "http://169.254.169.254/api/settings/debug/logs/export",
            "file:///etc/passwd",
        ] {
            assert!(require_loopback_url(url).is_err(), "should reject {url}");
        }
        assert!(
            require_loopback_url("http://127.0.0.1:8888/api/settings/debug/logs/export").is_ok()
        );
    }

    #[test]
    fn the_tab_token_stands_in_whenever_minting_cannot_produce_a_session() {
        use crate::desktop_auth::{DesktopAuthResponse, LoginRequired, MultiLoginMode};

        let minted = || {
            Ok(DesktopAuthResponse::Tokens {
                access_token: "minted".to_string(),
                refresh_token: "unused".to_string(),
            })
        };
        let login_required = || {
            Ok(DesktopAuthResponse::LoginRequired {
                login_required: LoginRequired,
                login_mode: MultiLoginMode::Multi,
            })
        };
        let failed = || Err("Desktop auth provisioning failed: no such file".to_string());

        assert_eq!(
            select_export_session(minted(), Some("ui".to_string())).unwrap(),
            "minted"
        );
        assert_eq!(select_export_session(minted(), None).unwrap(), "minted");

        assert_eq!(
            select_export_session(login_required(), Some("ui".to_string())).unwrap(),
            "ui"
        );
        assert_eq!(
            select_export_session(failed(), Some("ui".to_string())).unwrap(),
            "ui"
        );

        assert_eq!(
            select_export_session(login_required(), Some("   ".to_string())).unwrap_err(),
            LOGIN_REQUIRED
        );
        assert_eq!(
            select_export_session(login_required(), None).unwrap_err(),
            LOGIN_REQUIRED
        );
        assert!(
            select_export_session(failed(), None)
                .unwrap_err()
                .starts_with("Desktop auth provisioning failed")
        );
    }

    /// Multi-account installs never mint, so the tab must keep sending `uiToken` or export dies.
    #[test]
    fn the_tab_sends_a_ui_token_for_the_multi_account_fallback() {
        let frontend = include_str!("../../frontend/src/features/settings/api/debug-logs.ts");
        // Pin only that this command's payload names `uiToken`; how the frontend spells the
        // value is its own business.
        let payload = frontend
            .split_once("\"download_logs_to_downloads\"")
            .map(|(_, rest)| rest.chars().take(200).collect::<String>());
        assert!(
            payload.as_deref().is_some_and(|p| p.contains("uiToken")),
            "debug-logs.ts no longer sends uiToken ({payload:?}), so a \
             multi-account desktop install can never export logs"
        );
        assert!(
            frontend.contains("getAuthToken("),
            "debug-logs.ts no longer reads a token to send"
        );
    }

    /// The TypeScript side matches the sentinel's exact text; fail if the two drift.
    #[test]
    fn the_login_required_sentinel_matches_what_the_frontend_looks_for() {
        let frontend = include_str!("../../frontend/src/features/settings/api/debug-logs.ts");
        assert!(
            frontend.contains(&format!("\"{LOGIN_REQUIRED}\"")),
            "debug-logs.ts no longer matches LOGIN_REQUIRED ({LOGIN_REQUIRED:?})"
        );
    }

    /// Drives the inner rewrite, compiled everywhere, so the UNC branch is tested off Windows.
    #[test]
    fn a_verbatim_windows_path_is_shown_the_way_a_user_writes_it() {
        assert_eq!(
            strip_verbatim_prefix_inner(r"\\?\C:\Users\u\Downloads\a.zip".to_string()),
            r"C:\Users\u\Downloads\a.zip"
        );
        // `\\?\UNC\server\share` names `\\server\share`, not `UNC\server\share`.
        assert_eq!(
            strip_verbatim_prefix_inner(r"\\?\UNC\server\share\a.zip".to_string()),
            r"\\server\share\a.zip"
        );
        assert_eq!(
            strip_verbatim_prefix_inner(r"\\?\unc\server\share\a.zip".to_string()),
            r"\\server\share\a.zip"
        );
        // Byte 8 lands inside a multibyte character, so a plain `text[..8]` would panic.
        assert_eq!(
            strip_verbatim_prefix_inner(r"\\?\C:\下载\a.zip".to_string()),
            r"C:\下载\a.zip"
        );
        assert_eq!(
            strip_verbatim_prefix_inner(r"\\?\C:\ダウンロード\a.zip".to_string()),
            r"C:\ダウンロード\a.zip"
        );
        assert_eq!(
            strip_verbatim_prefix_inner(r"C:\Users\u\a.zip".to_string()),
            r"C:\Users\u\a.zip"
        );
        assert_eq!(strip_verbatim_prefix_inner("/home/u/a.zip".to_string()), "/home/u/a.zip");
        assert_eq!(strip_verbatim_prefix_inner(r"\\".to_string()), r"\\");
        assert_eq!(strip_verbatim_prefix_inner(String::new()), "");
    }

    #[test]
    fn the_log_export_sends_the_minted_session_and_lands_a_real_path() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 2048];
            let read = std::io::Read::read(&mut stream, &mut request).unwrap();
            let _ = stream.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Type: application/zip\r\n\r\nPK",
            );
            String::from_utf8_lossy(&request[..read]).to_string()
        });

        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("unsloth-logs.zip");
        test_runtime()
            .block_on(stream_url_to_path(
                &format!("http://127.0.0.1:{port}/api/settings/debug/logs/export"),
                &destination,
                DOWNLOAD_READ_TIMEOUT,
                Some("minted-session-token"),
            ))
            .unwrap();
        let request = server.join().unwrap();
        assert!(
            request
                .to_ascii_lowercase()
                .contains("authorization: bearer minted-session-token"),
            "{request}"
        );
        assert_eq!(fs::read(&destination).unwrap(), b"PK");

        let shown = display_path(&destination);
        assert!(Path::new(&shown).is_absolute(), "{shown}");
        assert!(shown.ends_with("unsloth-logs.zip"), "{shown}");
        assert_eq!(fs::read(&shown).unwrap(), b"PK");
    }

    #[test]
    fn an_old_backend_and_a_refused_session_stay_distinguishable_in_the_error() {
        // The Logs tab parses the status out of this string, so the wording is a contract.
        for status in ["404 Not Found", "403 Forbidden"] {
            let (url, server) = serve_once(b"nope".to_vec(), status);
            let directory = tempfile::tempdir().unwrap();
            let destination = directory.path().join("unsloth-logs.zip");
            let error = test_runtime()
                .block_on(stream_url_to_path(
                    &url,
                    &destination,
                    DOWNLOAD_READ_TIMEOUT,
                    Some("minted-session-token"),
                ))
                .unwrap_err();
            server.join().unwrap();
            assert_eq!(
                error,
                format!("Download failed with status {}.", &status[..3])
            );
            assert!(!destination.exists(), "{status} must not create the file");
        }
    }

    #[test]
    fn strips_directories_from_suggested_default_name() {
        assert_eq!(default_file_name("../../chat.jsonl"), "chat.jsonl");
        assert_eq!(default_file_name(""), "unsloth-export.json");

        assert_eq!(
            decode_default_file_name("Y2hhdC5qc29ubA==").unwrap(),
            "chat.jsonl"
        );
    }
}
