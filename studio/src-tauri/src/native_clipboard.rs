// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use serde::Serialize;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

const MAX_CLIPBOARD_IMAGE_DIMENSION: i32 = 8192;
const MAX_CLIPBOARD_RGBA_BYTES: u64 = 64 * 1024 * 1024;
const MAX_CLIPBOARD_PNG_BYTES: usize = 20 * 1024 * 1024;
const MAX_CLIPBOARD_SOURCE_BYTES: u64 = 20 * 1024 * 1024;
const MAX_CLIPBOARD_AUDIO_BYTES: u64 = 25 * 1024 * 1024;
const MAX_CLIPBOARD_TOTAL_BYTES: u64 = MAX_CLIPBOARD_AUDIO_BYTES;
const MAX_CLIPBOARD_FILES: usize = 8;
const MAX_CLIPBOARD_CANDIDATES: usize = 32;
#[cfg(target_os = "linux")]
const MAX_CLIPBOARD_URI_BYTES: usize = 64 * 1024;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeClipboardFile {
    name: String,
    mime_type: String,
    base64: String,
}

fn validate_dimensions(width: i32, height: i32) -> Result<(), String> {
    if width <= 0
        || height <= 0
        || width > MAX_CLIPBOARD_IMAGE_DIMENSION
        || height > MAX_CLIPBOARD_IMAGE_DIMENSION
    {
        return Err("Clipboard image dimensions are invalid or too large.".to_string());
    }
    let rgba_bytes = (width as u64)
        .checked_mul(height as u64)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| "Clipboard image dimensions overflow.".to_string())?;
    if rgba_bytes > MAX_CLIPBOARD_RGBA_BYTES {
        return Err("Clipboard image pixel data is too large.".to_string());
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn validate_png_bytes(png: &[u8]) -> Result<(), String> {
    const SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
    if png.len() < 24 || png.len() > MAX_CLIPBOARD_PNG_BYTES || &png[..8] != SIGNATURE {
        return Err("Clipboard PNG data is invalid or too large.".to_string());
    }
    if &png[12..16] != b"IHDR" {
        return Err("Clipboard PNG header is invalid.".to_string());
    }
    let width = u32::from_be_bytes(png[16..20].try_into().unwrap());
    let height = u32::from_be_bytes(png[20..24].try_into().unwrap());
    let width = i32::try_from(width).map_err(|_| "Clipboard PNG width is invalid.".to_string())?;
    let height =
        i32::try_from(height).map_err(|_| "Clipboard PNG height is invalid.".to_string())?;
    validate_dimensions(width, height)
}

fn clipboard_file_mime_type(path: &Path) -> Option<&'static str> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let mime_type = match extension.as_str() {
        "json" | "jsonl" | "ndjson" => "application/json",
        "md" | "markdown" | "mdx" => "text/markdown",
        "csv" => "text/csv",
        "html" | "htm" => "text/html",
        "xml" => "application/xml",
        "svg" => "image/svg+xml",
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "pdf" => "application/pdf",
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "odt" => "application/vnd.oasis.opendocument.text",
        "ods" => "application/vnd.oasis.opendocument.spreadsheet",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "m4a" => "audio/mp4",
        "ogg" | "oga" => "audio/ogg",
        "flac" => "audio/flac",
        "aac" => "audio/aac",
        "txt" | "text" | "log" | "rst" | "tsv" | "yaml" | "yml" | "toml" | "ini" | "cfg"
        | "conf" | "env" | "properties" | "css" | "scss" | "sass" | "less" | "js" | "jsx"
        | "mjs" | "cjs" | "ts" | "tsx" | "py" | "pyi" | "ipynb" | "rb" | "php" | "go" | "rs"
        | "java" | "kt" | "kts" | "scala" | "swift" | "c" | "h" | "cc" | "cpp" | "hpp" | "cxx"
        | "cs" | "m" | "mm" | "sh" | "bash" | "zsh" | "fish" | "ps1" | "bat" | "lua" | "pl"
        | "pm" | "r" | "jl" | "dart" | "vue" | "svelte" | "astro" | "sql" | "graphql" | "gql"
        | "proto" | "tf" | "tfvars" | "gradle" | "dockerfile" | "makefile" | "cmake" | "diff"
        | "patch" => "text/plain",
        _ => return None,
    };
    Some(mime_type)
}

fn clipboard_file_max_bytes(mime_type: &str) -> u64 {
    if mime_type.starts_with("audio/") {
        MAX_CLIPBOARD_AUDIO_BYTES
    } else {
        MAX_CLIPBOARD_SOURCE_BYTES
    }
}

fn open_regular_clipboard_file(path: &Path) -> Option<File> {
    let metadata = std::fs::symlink_metadata(path).ok()?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return None;
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW)
            .open(path)
            .ok()
    }
    #[cfg(not(unix))]
    {
        File::open(path).ok()
    }
}

fn read_clipboard_files(paths: Vec<PathBuf>) -> Result<Vec<NativeClipboardFile>, String> {
    let mut remaining = MAX_CLIPBOARD_TOTAL_BYTES;
    let mut files = Vec::new();
    for path in paths.into_iter().take(MAX_CLIPBOARD_CANDIDATES) {
        if remaining == 0 || files.len() >= MAX_CLIPBOARD_FILES {
            break;
        }
        let Some(name) = path
            .file_name()
            .map(|value| value.to_string_lossy().into_owned())
        else {
            continue;
        };
        let Some(mime_type) = clipboard_file_mime_type(&path) else {
            continue;
        };
        let Some(source) = open_regular_clipboard_file(&path) else {
            continue;
        };
        let Ok(metadata) = source.metadata() else {
            continue;
        };
        let limit = clipboard_file_max_bytes(mime_type).min(remaining);
        if !metadata.is_file() || metadata.len() == 0 || metadata.len() > limit {
            continue;
        }
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        if source.take(limit + 1).read_to_end(&mut bytes).is_err()
            || bytes.is_empty()
            || bytes.len() as u64 > limit
        {
            continue;
        }
        remaining -= bytes.len() as u64;
        files.push(NativeClipboardFile {
            name,
            mime_type: mime_type.to_string(),
            base64: BASE64.encode(bytes),
        });
    }
    if files.is_empty() {
        return Err("Clipboard does not contain readable local files.".to_string());
    }
    Ok(files)
}

#[cfg(target_os = "linux")]
fn encode_clipboard_pixbuf(image: &gdk_pixbuf::Pixbuf) -> Result<Vec<u8>, String> {
    validate_dimensions(image.width(), image.height())?;
    let png = image
        .save_to_bufferv("png", &[])
        .map_err(|error| format!("Could not encode clipboard image: {error}"))?;
    if png.is_empty() || png.len() > MAX_CLIPBOARD_PNG_BYTES {
        return Err("Clipboard image encoding is empty or too large.".to_string());
    }
    Ok(png)
}

#[cfg(target_os = "linux")]
fn local_clipboard_path(uri: &str) -> Option<PathBuf> {
    let (path, hostname) = glib::filename_from_uri(uri).ok()?;
    hostname.is_none().then_some(path)
}

#[cfg(target_os = "linux")]
fn local_clipboard_paths_from_bytes(data: &[u8]) -> Vec<PathBuf> {
    if data.len() > MAX_CLIPBOARD_URI_BYTES {
        return Vec::new();
    }
    let Ok(text) = std::str::from_utf8(data) else {
        return Vec::new();
    };
    text.lines()
        .map(|line| {
            line.trim_matches(|character: char| character.is_whitespace() || character == '\0')
        })
        .filter_map(local_clipboard_path)
        .take(MAX_CLIPBOARD_CANDIDATES)
        .collect()
}

#[cfg(target_os = "linux")]
fn read_gtk_clipboard_paths() -> Vec<PathBuf> {
    let clipboard = gtk::Clipboard::get(&gdk::SELECTION_CLIPBOARD);
    let mut paths: Vec<PathBuf> = clipboard
        .wait_for_uris()
        .into_iter()
        .filter_map(|uri| local_clipboard_path(uri.as_str()))
        .take(MAX_CLIPBOARD_CANDIDATES)
        .collect();

    for target in clipboard.wait_for_targets().unwrap_or_default() {
        if paths.len() >= MAX_CLIPBOARD_CANDIDATES {
            break;
        }
        if !target.name().to_ascii_lowercase().contains("copied-files") {
            continue;
        }
        let Some(data) = clipboard.wait_for_contents(&target) else {
            continue;
        };
        let length = data.length();
        if length <= 0 || length as usize > MAX_CLIPBOARD_URI_BYTES {
            continue;
        }
        for path in local_clipboard_paths_from_bytes(&data.data()) {
            if !paths.contains(&path) {
                paths.push(path);
            }
        }
    }
    paths.truncate(MAX_CLIPBOARD_CANDIDATES);
    paths
}

#[cfg(target_os = "linux")]
async fn native_clipboard_paths() -> Result<Vec<PathBuf>, String> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    glib::MainContext::default().invoke(move || {
        let _ = tx.send(read_gtk_clipboard_paths());
    });
    rx.await
        .map_err(|_| "Clipboard file reader stopped unexpectedly.".to_string())
}

#[cfg(not(target_os = "linux"))]
async fn native_clipboard_paths() -> Result<Vec<PathBuf>, String> {
    tokio::task::spawn_blocking(|| {
        let mut clipboard = arboard::Clipboard::new().map_err(|error| error.to_string())?;
        let mut paths = clipboard
            .get()
            .file_list()
            .map_err(|error| error.to_string())?;
        paths.truncate(MAX_CLIPBOARD_CANDIDATES);
        Ok(paths)
    })
    .await
    .map_err(|_| "Clipboard file reader stopped unexpectedly.".to_string())?
}

#[tauri::command]
pub async fn read_native_clipboard_files(
    window: tauri::WebviewWindow,
) -> Result<Vec<NativeClipboardFile>, String> {
    crate::native_intents::ensure_main_window(&window)?;
    let paths = native_clipboard_paths().await?;
    tokio::task::spawn_blocking(move || read_clipboard_files(paths))
        .await
        .map_err(|_| "Clipboard file loader stopped unexpectedly.".to_string())?
}

#[cfg(target_os = "linux")]
fn read_gtk_clipboard_file_image() -> Result<gdk_pixbuf::Pixbuf, String> {
    use std::os::fd::AsRawFd;

    for path in read_gtk_clipboard_paths() {
        let Some(source) = open_regular_clipboard_file(&path) else {
            continue;
        };
        let Ok(metadata) = source.metadata() else {
            continue;
        };
        if !metadata.is_file() || metadata.len() > MAX_CLIPBOARD_SOURCE_BYTES {
            continue;
        }
        let descriptor_path = PathBuf::from(format!("/proc/self/fd/{}", source.as_raw_fd()));
        let Some((_, width, height)) = gdk_pixbuf::Pixbuf::file_info(&descriptor_path) else {
            continue;
        };
        if validate_dimensions(width, height).is_err() {
            continue;
        }
        let Ok(image) = gdk_pixbuf::Pixbuf::from_file(&descriptor_path) else {
            continue;
        };
        if validate_dimensions(image.width(), image.height()).is_ok() {
            return Ok(image);
        }
    }
    Err("Clipboard does not contain a readable image or local image file.".to_string())
}

#[cfg(target_os = "linux")]
fn read_gtk_clipboard_png() -> Result<Vec<u8>, String> {
    let clipboard = gtk::Clipboard::get(&gdk::SELECTION_CLIPBOARD);
    let png_target = gdk::Atom::intern("image/png");
    if let Some(data) = clipboard.wait_for_contents(&png_target) {
        let length = data.length();
        if length <= 0 || length as usize > MAX_CLIPBOARD_PNG_BYTES {
            return Err("Clipboard PNG data is empty or too large.".to_string());
        }
        let png = data.data();
        validate_png_bytes(&png)?;
        return Ok(png);
    }

    let image = match clipboard.wait_for_image() {
        Some(image) => image,
        None => read_gtk_clipboard_file_image()?,
    };
    encode_clipboard_pixbuf(&image)
}

#[cfg(target_os = "linux")]
#[tauri::command]
pub async fn read_native_clipboard_png(
    window: tauri::WebviewWindow,
) -> Result<tauri::ipc::Response, String> {
    crate::native_intents::ensure_main_window(&window)?;
    let (tx, rx) = tokio::sync::oneshot::channel();
    glib::MainContext::default().invoke(move || {
        let _ = tx.send(read_gtk_clipboard_png());
    });
    let png = rx
        .await
        .map_err(|_| "Clipboard image reader stopped unexpectedly.".to_string())??;
    Ok(tauri::ipc::Response::new(png))
}

#[cfg(not(target_os = "linux"))]
#[tauri::command]
pub async fn read_native_clipboard_png(
    window: tauri::WebviewWindow,
) -> Result<tauri::ipc::Response, String> {
    crate::native_intents::ensure_main_window(&window)?;
    Err("Native PNG clipboard fallback is only available on Linux.".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipboard_dimensions_are_bounded() {
        assert!(validate_dimensions(3840, 2160).is_ok());
        assert!(validate_dimensions(0, 100).is_err());
        assert!(validate_dimensions(8193, 100).is_err());
        assert!(validate_dimensions(8192, 8192).is_err());
    }

    #[test]
    fn clipboard_file_mime_types_cover_text_attachments() {
        assert_eq!(
            clipboard_file_mime_type(Path::new("data.json")),
            Some("application/json")
        );
        assert_eq!(
            clipboard_file_mime_type(Path::new("notes.md")),
            Some("text/markdown")
        );
        assert_eq!(clipboard_file_mime_type(Path::new("unknown.bin")), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn clipboard_png_headers_are_bounded_before_decode() {
        let mut png = vec![0; 24];
        png[..8].copy_from_slice(b"\x89PNG\r\n\x1a\n");
        png[12..16].copy_from_slice(b"IHDR");
        png[16..20].copy_from_slice(&1920_u32.to_be_bytes());
        png[20..24].copy_from_slice(&1080_u32.to_be_bytes());
        assert!(validate_png_bytes(&png).is_ok());
        png[16..20].copy_from_slice(&9000_u32.to_be_bytes());
        assert!(validate_png_bytes(&png).is_err());
    }

    #[test]
    fn clipboard_file_reads_are_bounded() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("notes.md");
        std::fs::write(&path, b"clipboard text").unwrap();

        let empty = directory.path().join("empty.md");
        File::create(&empty).unwrap();
        let oversized = directory.path().join("oversized.md");
        File::create(&oversized)
            .unwrap()
            .set_len(MAX_CLIPBOARD_SOURCE_BYTES + 1)
            .unwrap();

        let files =
            read_clipboard_files(vec![directory.path().to_path_buf(), empty, oversized, path])
                .unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].name, "notes.md");
        assert_eq!(files[0].mime_type, "text/markdown");

        assert_eq!(BASE64.decode(&files[0].base64).unwrap(), b"clipboard text");
    }

    #[test]
    fn clipboard_audio_uses_the_chat_upload_boundary() {
        assert_eq!(clipboard_file_max_bytes("audio/mpeg"), 25 * 1024 * 1024);
        assert_eq!(
            clipboard_file_max_bytes("text/markdown"),
            MAX_CLIPBOARD_SOURCE_BYTES
        );

        let directory = tempfile::tempdir().unwrap();
        let accepted = directory.path().join("accepted.mp3");
        File::create(&accepted)
            .unwrap()
            .set_len(MAX_CLIPBOARD_AUDIO_BYTES)
            .unwrap();
        let oversized = directory.path().join("oversized.mp3");
        File::create(&oversized)
            .unwrap()
            .set_len(MAX_CLIPBOARD_AUDIO_BYTES + 1)
            .unwrap();

        let files = read_clipboard_files(vec![oversized, accepted]).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].name, "accepted.mp3");
        assert_eq!(
            BASE64.decode(&files[0].base64).unwrap().len() as u64,
            MAX_CLIPBOARD_AUDIO_BYTES
        );
    }

    #[cfg(unix)]
    #[test]
    fn clipboard_file_reads_reject_symlinks() {
        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("target.md");
        let link = directory.path().join("link.md");
        std::fs::write(&target, b"clipboard text").unwrap();
        std::os::unix::fs::symlink(&target, &link).unwrap();

        assert!(open_regular_clipboard_file(&link).is_none());
        assert!(read_clipboard_files(vec![link]).is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn copied_file_targets_parse_local_uris() {
        let paths = local_clipboard_paths_from_bytes(
            b"copy\nfile:///tmp/pasted%20notes.md\nhttps://example.com/ignored.md\0",
        );
        assert_eq!(paths, vec![PathBuf::from("/tmp/pasted notes.md")]);
        assert!(
            local_clipboard_paths_from_bytes(&vec![b'x'; MAX_CLIPBOARD_URI_BYTES + 1]).is_empty()
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn clipboard_file_uris_must_be_local() {
        assert_eq!(
            local_clipboard_path("file:///tmp/pasted%20image.png"),
            Some(PathBuf::from("/tmp/pasted image.png"))
        );
        assert!(local_clipboard_path("file://remote/tmp/image.png").is_none());
        assert!(local_clipboard_path("https://example.com/image.png").is_none());
    }
}
