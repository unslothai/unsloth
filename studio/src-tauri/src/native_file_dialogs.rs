use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};

use serde::Serialize;
use std::borrow::Cow;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use tauri::{AppHandle, WebviewWindow};
use tauri_plugin_dialog::DialogExt;

const MAX_CHAT_IMPORT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TRAINING_CONFIG_BYTES: u64 = 1024 * 1024;
const NATIVE_FILE_NAME_HEADER: &str = "x-unsloth-default-name";
const CHAT_IMPORT_EXTENSIONS: &[&str] = &["jsonl", "ndjson", "csv"];
const TRAINING_CONFIG_EXTENSIONS: &[&str] = &["yaml", "yml"];

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeImportedFile {
    name: String,
    content: String,
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
        // Named for both tracks: the video gallery saves .mp4 through this dialog too.
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

fn save_selected_file(
    selected_path: Option<PathBuf>,
    content: &[u8],
) -> Result<Option<String>, String> {
    let Some(path) = selected_path else {
        return Ok(None);
    };
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut builder = tempfile::Builder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let permissions = fs::metadata(&path)
            .map(|metadata| metadata.permissions())
            .unwrap_or_else(|_| fs::Permissions::from_mode(0o666));
        builder.permissions(permissions);
    }
    let mut temporary = builder
        .prefix(".unsloth-export-")
        .tempfile_in(parent)
        .map_err(|error| format!("Failed to prepare {}: {error}", path.display()))?;
    temporary
        .write_all(content)
        .and_then(|()| temporary.as_file().sync_all())
        .map_err(|error| format!("Failed to save {}: {error}", path.display()))?;
    temporary
        .persist(&path)
        .map_err(|error| format!("Failed to save {}: {}", path.display(), error.error))?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("export")
        .to_string();
    Ok(Some(file_name))
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

fn read_selected_import(
    selected_path: Option<PathBuf>,
) -> Result<Option<NativeImportedFile>, String> {
    read_selected_text_import(
        selected_path,
        "Chat import",
        CHAT_IMPORT_EXTENSIONS,
        ".jsonl, .ndjson, or .csv",
        "chat-import",
        MAX_CHAT_IMPORT_BYTES,
    )
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

#[tauri::command]
pub async fn pick_native_chat_import(
    window: WebviewWindow,
    app: AppHandle,
) -> Result<Option<NativeImportedFile>, String> {
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
    read_selected_import(selected_path)
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
        assert!(read_selected_import(None).unwrap().is_none());
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

    #[test]
    fn gallery_video_exports_offer_their_own_container() {
        // The three Download menu formats, which reach this dialog via downloadUrl
        // (MP4) and downloadFile (WebM / GIF).
        assert_save_filter(
            "Unsloth_video_20260808-120000_1670009728.mp4",
            "MPEG-4 video or audio",
            &["m4a", "mp4"],
        );
        assert_save_filter("clip.webm", "WebM video or audio", &["webm"]);
        assert_save_filter("clip.gif", "GIF image", &["gif"]);
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
    fn reads_supported_import_and_rejects_other_extensions() {
        let jsonl_path = temp_path("allowed").with_extension("JSONL");
        fs::write(&jsonl_path, "{\"messages\":[]}").unwrap();
        let imported = read_selected_import(Some(jsonl_path.clone()))
            .unwrap()
            .unwrap();
        assert_eq!(imported.content, "{\"messages\":[]}");

        let json_path = temp_path("unsupported").with_extension("json");
        fs::write(&json_path, "{}").unwrap();
        assert!(read_selected_import(Some(json_path.clone())).is_err());
        let txt_path = temp_path("denied").with_extension("txt");
        fs::write(&txt_path, "no").unwrap();
        assert!(read_selected_import(Some(txt_path.clone()))
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
    fn read_limit_and_utf8_errors_are_concrete() {
        let oversized = temp_path("oversized").with_extension("csv");
        let file = File::create(&oversized).unwrap();
        file.set_len(MAX_CHAT_IMPORT_BYTES + 1).unwrap();
        assert!(read_selected_import(Some(oversized.clone()))
            .unwrap_err()
            .contains("too large"));

        let invalid = temp_path("invalid-utf8").with_extension("jsonl");
        fs::write(&invalid, [0xff]).unwrap();
        assert!(read_selected_import(Some(invalid.clone()))
            .unwrap_err()
            .contains("UTF-8"));
        let _ = fs::remove_file(oversized);
        let _ = fs::remove_file(invalid);
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
        let imported = read_selected_import(Some(path.clone())).unwrap().unwrap();
        assert_eq!(imported.name, "chat-import.csv");
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
