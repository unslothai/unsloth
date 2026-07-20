use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};

use serde::Serialize;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use tauri::{AppHandle, WebviewWindow};
use tauri_plugin_dialog::DialogExt;

const MAX_CHAT_IMPORT_BYTES: u64 = 64 * 1024 * 1024;
const NATIVE_FILE_NAME_HEADER: &str = "x-unsloth-default-name";
const CHAT_IMPORT_EXTENSIONS: &[&str] = &["jsonl", "ndjson", "csv"];

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

fn save_filter(file_name: &str) -> (&'static str, Vec<&'static str>) {
    match Path::new(file_name)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("json") => ("JSON", vec!["json"]),
        Some("jsonl") | Some("ndjson") => ("JSON Lines", vec!["jsonl", "ndjson"]),
        Some("csv") => ("CSV", vec!["csv"]),
        Some("md") | Some("markdown") => ("Markdown", vec!["md", "markdown"]),
        Some("zip") => ("ZIP archive", vec!["zip"]),
        _ => (
            "Export files",
            vec!["json", "jsonl", "ndjson", "csv", "md", "markdown", "zip"],
        ),
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
    fs::write(&path, content)
        .map_err(|error| format!("Failed to save {}: {error}", path.display()))?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("export")
        .to_string();
    Ok(Some(file_name))
}

fn read_selected_import(
    selected_path: Option<PathBuf>,
) -> Result<Option<NativeImportedFile>, String> {
    let Some(path) = selected_path else {
        return Ok(None);
    };
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| "Chat import must be a .jsonl, .ndjson, or .csv file.".to_string())?;
    if !CHAT_IMPORT_EXTENSIONS.contains(&extension.as_str()) {
        return Err("Chat import must be a .jsonl, .ndjson, or .csv file.".to_string());
    }

    let metadata = fs::metadata(&path)
        .map_err(|error| format!("Failed to inspect {}: {error}", path.display()))?;
    if !metadata.is_file() {
        return Err(format!("Selected import is not a file: {}", path.display()));
    }
    if metadata.len() > MAX_CHAT_IMPORT_BYTES {
        return Err(format!(
            "Chat import is too large (maximum {} MiB).",
            MAX_CHAT_IMPORT_BYTES / 1024 / 1024
        ));
    }

    // Limit the read too, so a file that grows after metadata inspection cannot
    // make the command allocate without bound.
    let file =
        File::open(&path).map_err(|error| format!("Failed to open {}: {error}", path.display()))?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_CHAT_IMPORT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("Failed to read {}: {error}", path.display()))?;
    if bytes.len() as u64 > MAX_CHAT_IMPORT_BYTES {
        return Err(format!(
            "Chat import is too large (maximum {} MiB).",
            MAX_CHAT_IMPORT_BYTES / 1024 / 1024
        ));
    }
    let content = String::from_utf8(bytes)
        .map_err(|_| format!("Chat import is not valid UTF-8: {}", path.display()))?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("chat-import")
        .to_string();
    Ok(Some(NativeImportedFile { name, content }))
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
    let content = match request.body() {
        tauri::ipc::InvokeBody::Raw(content) => content,
        _ => return Err("Native export content must be binary.".to_string()),
    };
    let (filter_name, extensions) = save_filter(&file_name);
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog()
        .file()
        .set_title("Save Unsloth export")
        .set_file_name(file_name)
        .add_filter(filter_name, &extensions)
        .save_file(move |path| {
            let _ = tx.send(path);
        });
    let selected_path = rx
        .await
        .map_err(|_| "Save dialog closed unexpectedly.".to_string())?
        .map(local_dialog_path)
        .transpose()?;
    save_selected_file(selected_path, content)
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

    #[test]
    fn cancellation_is_quiet_for_save_and_import() {
        assert!(save_selected_file(None, b"x").unwrap().is_none());
        assert!(read_selected_import(None).unwrap().is_none());
    }

    #[test]
    fn writes_text_and_binary_exactly() {
        let text_path = temp_path("text").with_extension("json");
        let binary_path = temp_path("binary").with_extension("zip");
        save_selected_file(Some(text_path.clone()), b"{\"ok\":true}").unwrap();
        save_selected_file(Some(binary_path.clone()), &[0, 1, 2, 255]).unwrap();
        assert_eq!(fs::read(&text_path).unwrap(), b"{\"ok\":true}");
        assert_eq!(fs::read(&binary_path).unwrap(), [0, 1, 2, 255]);
        let _ = fs::remove_file(text_path);
        let _ = fs::remove_file(binary_path);
    }

    #[test]
    fn markdown_exports_use_a_markdown_save_filter() {
        assert_eq!(
            save_filter("message.md"),
            ("Markdown", vec!["md", "markdown"])
        );
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
