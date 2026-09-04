use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};

use serde::Serialize;
use std::borrow::Cow;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Manager, State, WebviewWindow};
use tauri_plugin_dialog::DialogExt;

const MAX_TRAINING_CONFIG_BYTES: u64 = 1024 * 1024;
/// Per redeemed range, not per file: the frontend streams a large import in
/// pieces, and a piece has to fit in one IPC response.
const MAX_CHAT_IMPORT_CHUNK_BYTES: usize = 8 * 1024 * 1024;
const NATIVE_FILE_NAME_HEADER: &str = "x-unsloth-default-name";
const CHAT_IMPORT_EXTENSIONS: &[&str] = &["json", "jsonl", "ndjson", "csv"];
const CHAT_IMPORT_TYPE_ERROR: &str = "Chat import must be a .json, .jsonl, .ndjson, or .csv file.";
const TRAINING_CONFIG_EXTENSIONS: &[&str] = &["yaml", "yml"];

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeImportedFile {
    name: String,
    content: String,
}

/// A picked chat import addressed by an opaque token instead of a path.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeChatImport {
    name: String,
    size: u64,
    token: String,
}

/// Keep active handles valid while bounding registry growth.
const CHAT_IMPORT_HANDLE_LIMIT: usize = 8;

#[derive(Default)]
pub struct ChatImportRegistry {
    /// Oldest first, so the eviction below drops the least recent pick.
    files: Mutex<Vec<(String, PathBuf, Arc<Mutex<File>>)>>,
}

impl ChatImportRegistry {
    /// Holds the file the picker opened, not just its path. Resolving a path
    /// again per range would read whatever occupies it by then, so a file
    /// replaced mid-import could splice its bytes into the stream.
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
        // Saved chat attachments, not just exports: a name outside the active
        // filter can be rejected or silently re-extensioned by the OS dialog.
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
fn staged_temp_file(path: &Path) -> Result<tempfile::NamedTempFile, String> {
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
) -> Result<Option<String>, String> {
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
    Ok(Some(saved_file_name(&path)))
}

/// Only the local backend, or the webview could write any host's reply to disk. Parsed
/// rather than sliced: in `http://127.0.0.1:8888@evil.test/clip` the loopback-looking part
/// is userinfo and the client would connect to `evil.test`.
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

fn rebase_loopback_url(url: &str, port: u16) -> Result<String, String> {
    const REJECT: &str = "Only local http URLs can be saved.";
    require_loopback_url(url)?;
    let mut parsed = reqwest::Url::parse(url).map_err(|_| REJECT.to_string())?;
    parsed
        .set_host(Some("127.0.0.1"))
        .map_err(|_| REJECT.to_string())?;
    parsed
        .set_port(Some(port))
        .map_err(|_| REJECT.to_string())?;
    Ok(parsed.to_string())
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

    // Limit the read too, so a file that grows after metadata inspection cannot
    // make the command allocate without bound.
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

/// Register a picked file for bounded range reads through an opaque token.
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
    // Size from the handle the token keeps, not from the path: a file swapped in
    // between the two describes a different object, and this number is what
    // bounds the streamed read.
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
    window: WebviewWindow,
    app: AppHandle,
    request: tauri::ipc::Request<'_>,
) -> Result<Option<String>, String> {
    crate::native_intents::ensure_main_window(&window)?;
    let encoded_name = request
        .headers()
        .get(NATIVE_FILE_NAME_HEADER)
        .ok_or_else(|| "Native export filename is missing.".to_string())?
        .to_str()
        .map_err(|_| "Invalid native export filename.".to_string())?;
    let file_name = decode_default_file_name(encoded_name)?;
    let content = invoke_body_bytes(request.body())
        .ok_or_else(|| "Native export content must be binary.".to_string())?;
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
    save_selected_file(selected_path, content.as_ref())
}

/// Save a backend URL by streaming it to the chosen path.
///
/// `save_native_file` carries the bytes through IPC, so the caller buffers the whole body
/// and the chooser waits on it. A clip is capped at 2048x2048 by 1024 frames, so this opens
/// the chooser first and writes the response chunk by chunk, leaving nothing resident.
#[tauri::command]
pub async fn save_native_file_from_url(
    window: WebviewWindow,
    app: AppHandle,
    backend_state: State<'_, crate::process::BackendState>,
    diagnostics: State<'_, crate::diagnostics::DiagnosticsState>,
    url: String,
    file_name: String,
    bearer_token: Option<String>,
    refresh_desktop_auth: Option<bool>,
    slow_first_chunk: Option<bool>,
) -> Result<Option<String>, String> {
    crate::native_intents::ensure_main_window(&window)?;
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
    let (url, bearer_token) = if refresh_desktop_auth.unwrap_or(false) {
        let auth = crate::desktop_auth::desktop_access(backend_state, diagnostics).await?;
        (
            rebase_loopback_url(&url, auth.backend_port)?,
            Some(auth.access_token),
        )
    } else {
        (url, bearer_token)
    };
    stream_url_to_path(
        &url,
        &path,
        DOWNLOAD_READ_TIMEOUT,
        bearer_token.as_deref(),
        slow_first_chunk.unwrap_or(false),
    )
    .await?;
    Ok(Some(saved_file_name(&path)))
}

/// A local backend that has accepted the request should never go quiet for this long, and a
/// clip that is still arriving resets it, so a large save is not cut short.
const DOWNLOAD_READ_TIMEOUT: Duration = Duration::from_secs(30);
// Archive construction starts only after the response headers are sent. Give a
// large retained history time to compress, but never leave the save dialog in
// an exporting state forever if the backend stalls.
const LOG_ARCHIVE_FIRST_CHUNK_TIMEOUT: Duration = Duration::from_secs(15 * 60);

async fn stream_url_to_path(
    url: &str,
    path: &Path,
    read_timeout: Duration,
    bearer_token: Option<&str>,
    slow_first_chunk: bool,
) -> Result<(), String> {
    let client = crate::loopback_http::streaming_client(
        Duration::from_secs(10),
        (!slow_first_chunk).then_some(read_timeout),
    )
    .map_err(|error| format!("Download failed: {error}"))?;
    let mut request = client.get(url);
    if let Some(token) = bearer_token {
        request = request.bearer_auth(token);
    }
    let mut response = if slow_first_chunk {
        tokio::time::timeout(read_timeout, request.send())
            .await
            .map_err(|_| "Download failed: response headers timed out.".to_string())?
    } else {
        request.send().await
    }
    .map_err(|error| format!("Download failed: {error}"))?;
    // Redirects are refused rather than followed, so a 3xx is a rejection here, not a hop.
    if !response.status().is_success() {
        return Err(format!(
            "Download failed with status {}.",
            response.status().as_u16()
        ));
    }
    let mut temporary = staged_temp_file(path)?;
    let mut first_chunk = true;
    loop {
        let chunk = if slow_first_chunk {
            let timeout = if first_chunk {
                LOG_ARCHIVE_FIRST_CHUNK_TIMEOUT
            } else {
                read_timeout
            };
            tokio::time::timeout(timeout, response.chunk())
                .await
                .map_err(|_| "Download failed: response body timed out.".to_string())?
        } else {
            response.chunk().await
        }
        .map_err(|error| format!("Download failed: {error}"))?;
        let Some(chunk) = chunk else { break };
        first_chunk = false;
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

fn read_range(
    handle: &Arc<Mutex<File>>,
    path: &Path,
    offset: u64,
    length: usize,
) -> Result<Vec<u8>, String> {
    // Clamped here rather than only at the call site, so the bound holds for
    // every caller: `take` was reading past it while only the allocation was
    // capped, which left the read itself unbounded.
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

/// Read raw bytes so the frontend can decode UTF-8 across range boundaries.
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
    window: WebviewWindow,
    app: AppHandle,
    registry: State<'_, ChatImportRegistry>,
) -> Result<Option<NativeChatImport>, String> {
    crate::native_intents::ensure_main_window(&window)?;
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
    window: WebviewWindow,
    app: AppHandle,
) -> Result<Option<NativeImportedFile>, String> {
    crate::native_intents::ensure_main_window(&window)?;
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
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "unsloth-native-files-{name}-{}-{nanos}",
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
        // Overwriting must stage the new content before replacing the destination.
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
        // The terminal card downloads command.sh through the same cell.
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
        // Settings > Data saves attachments here; a name outside the filter can
        // be rejected or re-extensioned by the OS dialog.
        assert_save_filter("report.txt", "Text", &["txt", "log"]);
        assert_save_filter("photo.PNG", "PNG image", &["png"]);
        assert_save_filter("shot.jpeg", "JPEG image", &["jpg", "jpeg"]);
        assert_save_filter("clip.wav", "WAV audio", &["wav"]);
        assert_save_filter("voice.webm", "WebM video or audio", &["webm"]);
    }

    /// A current-thread runtime: these tests need one task, not a worker per core.
    fn test_runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    /// A one-shot loopback server, so the streaming save is exercised over real HTTP.
    fn serve_once(
        body: Vec<u8>,
        status: &'static str,
        expected_bearer: Option<&'static str>,
    ) -> (String, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 2048];
            let read = std::io::Read::read(&mut stream, &mut request).unwrap();
            if let Some(token) = expected_bearer {
                let request = String::from_utf8_lossy(&request[..read]);
                assert!(
                    request.contains(&format!("authorization: Bearer {token}")),
                    "bearer header missing from {request}"
                );
            }
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
        let (url, server) = serve_once(body.clone(), "200 OK", None);
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("clip.mp4");
        test_runtime()
            .block_on(stream_url_to_path(
                &url,
                &dest,
                DOWNLOAD_READ_TIMEOUT,
                None,
                false,
            ))
            .unwrap();
        server.join().unwrap();
        assert_eq!(fs::read(&dest).unwrap(), body);
        // Nothing partial left beside it.
        let strays: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name() != std::ffi::OsStr::new("clip.mp4"))
            .collect();
        assert!(strays.is_empty(), "staging file left behind");
    }

    #[test]
    fn streaming_save_can_authenticate_a_protected_backend_download() {
        let token = "desktop-access-token";
        let (url, server) = serve_once(b"zip".to_vec(), "200 OK", Some(token));
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("logs.zip");
        test_runtime()
            .block_on(stream_url_to_path(
                &url,
                &dest,
                DOWNLOAD_READ_TIMEOUT,
                Some(token),
                false,
            ))
            .unwrap();
        server.join().unwrap();
        assert_eq!(fs::read(&dest).unwrap(), b"zip");
    }

    #[test]
    fn log_archive_can_wait_for_its_expensive_first_body_chunk() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut discard = [0_u8; 1024];
            let _ = std::io::Read::read(&mut stream, &mut discard);
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\n")
                .unwrap();
            stream.flush().unwrap();
            std::thread::sleep(Duration::from_millis(150));
            stream.write_all(b"zip").unwrap();
        });
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("logs.zip");
        test_runtime()
            .block_on(stream_url_to_path(
                &format!("http://127.0.0.1:{port}/logs.zip"),
                &dest,
                Duration::from_millis(50),
                None,
                true,
            ))
            .unwrap();
        server.join().unwrap();
        assert_eq!(fs::read(dest).unwrap(), b"zip");
    }

    #[test]
    fn refreshed_download_rebases_the_stale_backend_port() {
        let rebased = rebase_loopback_url(
            "http://127.0.0.1:8000/api/settings/debug/logs/export?all=true",
            9000,
        )
        .unwrap();
        assert_eq!(
            rebased,
            "http://127.0.0.1:9000/api/settings/debug/logs/export?all=true"
        );
        assert!(rebase_loopback_url("https://example.com/archive.zip", 9000).is_err());
    }

    #[test]
    fn a_failed_download_leaves_no_file_behind() {
        let (url, server) = serve_once(b"nope".to_vec(), "401 Unauthorized", None);
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("clip.mp4");
        let error = test_runtime()
            .block_on(stream_url_to_path(
                &url,
                &dest,
                DOWNLOAD_READ_TIMEOUT,
                None,
                false,
            ))
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
        // The userinfo forms are the ones a naive authority split accepts: everything
        // before the '@' is credentials, so the real host is what follows it.
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
        // Headers promise more than is sent, then the server holds the socket open. Without a
        // per-read timeout the invoke never resolves and the staging file stays behind.
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
                false,
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
                false,
            ))
            .unwrap_err();
        server.join().unwrap();
        assert!(error.contains("302"), "{error}");
        assert!(!dest.exists(), "a redirect must not produce a file");
    }

    #[test]
    fn a_gallery_clip_offers_its_own_container() {
        // The gallery's MP4 is the only export that reaches this dialog; WebM and GIF
        // save from a blob. Both mp4 arms are named for video since #8173.
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

        // An Open WebUI export is a .json array, so that extension has to be accepted now.
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
        // The 64 MiB read cap is what turned a real Open WebUI export away; a
        // file past it must now open, because the frontend streams it in ranges.
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

        // Ranges are cut on byte boundaries, so a character split across two of
        // them is the frontend decoder's problem, not a truncated read here.
        assert_eq!(read_range(&handle, &path, 0, 4).unwrap(), b"0123");
        assert_eq!(read_range(&handle, &path, 4, 4).unwrap(), b"4567");
        assert_eq!(read_range(&handle, &path, 8, 4).unwrap(), b"89");
        assert!(read_range(&handle, &path, 10, 4).unwrap().is_empty());

        // Picking again must not invalidate the earlier handle: that import can
        // still be streaming, and its next range read would fail mid-history.
        let second = temp_path("ranges-2").with_extension("jsonl");
        fs::write(&second, "abc").unwrap();
        let newer = register_for_test(&registry, &second);
        assert_eq!(registry.resolve(&token).unwrap().0, path);
        assert_eq!(registry.resolve(&newer).unwrap().0, second);
        assert_ne!(token, newer);

        // A hostile length cannot read past one chunk, whatever the caller passes.
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

        // The map is bounded, so a long session cannot accumulate handles.
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
        // Reopening the path per range would splice a replacement file into the
        // export, and a same-size swap leaves no short read to catch it.
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
        // Linux happily stores arbitrary bytes in a filename, but macOS enforces
        // UTF-8 on APFS/HFS+ and rejects this name outright. The name-recovery
        // path being asserted here is only reachable where such a file can
        // exist, so skip rather than fail on filesystems that forbid it.
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
    fn strips_directories_from_suggested_default_name() {
        assert_eq!(default_file_name("../../chat.jsonl"), "chat.jsonl");
        assert_eq!(default_file_name(""), "unsloth-export.json");

        assert_eq!(
            decode_default_file_name("Y2hhdC5qc29ubA==").unwrap(),
            "chat.jsonl"
        );
    }
}
