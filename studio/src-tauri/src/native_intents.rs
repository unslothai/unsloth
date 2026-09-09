use crate::native_backend_lease::{
    encode_secret_env, now_ms, random_token, sign_path_lease, NativePathKind,
    NativePathLeaseRequest, NativePathLeaseResponse, NativePathOperation, NativePathSourceKind,
    NativePathType,
};
use crate::native_path_policy::{
    classify_artifact_path, classify_native_attachment_path, classify_native_dataset_path,
    classify_native_document_folder, classify_native_model_path, is_audio_only_3gp,
    is_binary_property_list, is_binary_tracker_mod, is_binary_vobsub, is_binary_office_template, is_compiled_fortran_mod, is_text_attachment_name,
    reveal_target, ClassifiedPath, NativeArtifactKind,
};
use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;
use tauri::{AppHandle, WebviewWindow};
use tauri_plugin_dialog::DialogExt;

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};

const TOKEN_TTL: Duration = Duration::from_secs(15 * 60);
// How long after the OS reports a drop the renderer may still register those paths.
const DROP_GRACE: Duration = Duration::from_secs(2 * 60);

#[cfg(any(windows, test))]
fn normalize_windows_verbatim_path(path: String) -> String {
    if let Some(rest) = path.strip_prefix(r"\\?\UNC\") {
        return format!(r"\\{rest}");
    }
    path.strip_prefix(r"\\?\").unwrap_or(&path).to_string()
}

fn portable_path_string(path: &Path) -> String {
    let value = path.to_string_lossy().to_string();
    #[cfg(windows)]
    {
        return normalize_windows_verbatim_path(value);
    }
    #[cfg(not(windows))]
    {
        value
    }
}

#[derive(Clone, Debug)]
struct NativePathEntry {
    token: String,
    canonical_path: PathBuf,
    validation_policy: NativePathValidationPolicy,
    path_kind: NativePathKind,
    path_type: NativePathType,
    source_kind: NativePathSourceKind,
    allowed_operations: Vec<NativePathOperation>,
    display_label: String,
    expires_at_ms: u64,
    size_bytes: Option<u64>,
    modified_ms: Option<u64>,
    device_id: String,
    file_id: String,
}

#[derive(Clone, Copy, Debug)]
enum NativePathValidationPolicy {
    Model,
    Dataset,
    Attachment,
    Artifact(NativeArtifactKind),
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativePathRef {
    token: String,
    kind: NativePathKind,
    display_label: String,
    allowed_operations: Vec<NativePathOperation>,
    expires_at_ms: u64,
    // The frontend dedups uploads on these the way it does on a File's size/mtime.
    size_bytes: Option<u64>,
    modified_ms: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeIntent {
    id: String,
    kind: NativePathKind,
    source_kind: NativePathSourceKind,
    path: NativePathRef,
    display_label: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeDocumentFolderSelection {
    token: String,
    display_name: String,
}

#[derive(Default)]
struct NativeIntakeInner {
    tokens: HashMap<String, NativePathEntry>,
    queued_intents: VecDeque<NativeIntent>,
    // Paths Rust itself saw land on the window, so the renderer can only register
    // what the user actually dropped. Expiry keeps a stale drop from being spent later.
    recent_drops: HashMap<PathBuf, u64>,
}

pub struct NativeIntakeState {
    inner: Mutex<NativeIntakeInner>,
    lease_secret: Vec<u8>,
}

/// Per process, and deliberately not persisted: a key on disk outlives every
/// backend restart, and spent nonces are only remembered in memory, so a
/// consumed lease could be replayed against a replacement inside the TTL. The
/// adopted-survivor case is answered by `native_path_leases_usable` instead.
pub fn new_native_intake_state() -> NativeIntakeState {
    NativeIntakeState {
        inner: Mutex::new(NativeIntakeInner::default()),
        lease_secret: crate::native_backend_lease::new_lease_secret(),
    }
}

impl NativeIntakeState {
    pub fn lease_secret_env(&self) -> String {
        encode_secret_env(&self.lease_secret)
    }

    #[allow(dead_code)]
    pub fn enqueue_model_path(
        &self,
        path: impl AsRef<Path>,
        source_kind: NativePathSourceKind,
    ) -> Result<NativeIntent, String> {
        let intent = self.register_model_path(path, source_kind)?;
        let mut inner = self.inner.lock().map_err(|e| e.to_string())?;
        inner.queued_intents.push_back(intent.clone());
        Ok(intent)
    }

    fn register_model_path(
        &self,
        path: impl AsRef<Path>,
        source_kind: NativePathSourceKind,
    ) -> Result<NativeIntent, String> {
        let classified = classify_native_model_path(path.as_ref())?;
        self.register_classified_path(classified, source_kind, NativePathValidationPolicy::Model)
    }

    /// Record what the OS dropped on the window. Called from the window event handler,
    /// never from the renderer.
    pub fn note_dropped_paths(&self, paths: &[PathBuf]) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        let now = now_ms();
        inner.recent_drops.retain(|_, expires| *expires > now);
        let expires_at = now + DROP_GRACE.as_millis() as u64;
        for path in paths {
            if let Ok(canonical) = path.canonicalize() {
                inner.recent_drops.insert(canonical, expires_at);
            }
        }
    }

    fn was_recently_dropped(&self, canonical: &Path) -> Result<bool, String> {
        let mut inner = self.inner.lock().map_err(|e| e.to_string())?;
        let now = now_ms();
        inner.recent_drops.retain(|_, expires| *expires > now);
        Ok(inner.recent_drops.contains_key(canonical))
    }

    fn register_attachment_path(
        &self,
        path: impl AsRef<Path>,
        source_kind: NativePathSourceKind,
    ) -> Result<NativeIntent, String> {
        let classified = classify_native_attachment_path(path.as_ref())?;
        // The renderer hands us a path string, so a script in the webview could name any
        // readable document. Only paths the user actually dropped can be registered.
        if !self.was_recently_dropped(&classified.canonical_path)? {
            return Err("Attachments must come from a file dropped on the window.".to_string());
        }
        self.register_classified_path(
            classified,
            source_kind,
            NativePathValidationPolicy::Attachment,
        )
    }

    fn register_dataset_path(
        &self,
        path: impl AsRef<Path>,
        source_kind: NativePathSourceKind,
    ) -> Result<NativeIntent, String> {
        let classified = classify_native_dataset_path(path.as_ref())?;
        if !self.was_recently_dropped(&classified.canonical_path)? {
            return Err("Datasets must come from a file dropped on the window.".to_string());
        }
        self.register_classified_path(classified, source_kind, NativePathValidationPolicy::Dataset)
    }

    fn register_artifact(
        &self,
        kind: NativeArtifactKind,
        path: impl AsRef<Path>,
    ) -> Result<NativePathRef, String> {
        let classified = classify_artifact_path(kind, path.as_ref())?;
        let entry = self.insert_entry(
            classified,
            NativePathSourceKind::Artifact,
            NativePathValidationPolicy::Artifact(kind),
        )?;
        Ok(entry.to_ref())
    }

    fn sign_document_folder_path(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<NativeDocumentFolderSelection, String> {
        let classified = classify_native_document_folder(path.as_ref())?;
        let token = random_token("path_");
        let lease = sign_path_lease(
            &self.lease_secret,
            NativePathLeaseRequest {
                operation: NativePathOperation::LinkDocuments,
                canonical_path: portable_path_string(&classified.canonical_path),
                path_kind: classified.path_kind,
                path_type: classified.path_type,
                source_kind: NativePathSourceKind::Dialog,
                token,
                display_label: classified.display_label,
                size_bytes: classified.size_bytes,
                modified_ms: None,
                device_id: Some(classified.device_id),
                file_id: Some(classified.file_id),
            },
        )?;
        Ok(NativeDocumentFolderSelection {
            token: lease.native_path_lease,
            display_name: lease.display_label,
        })
    }

    fn register_classified_path(
        &self,
        classified: ClassifiedPath,
        source_kind: NativePathSourceKind,
        validation_policy: NativePathValidationPolicy,
    ) -> Result<NativeIntent, String> {
        let entry = self.insert_entry(classified, source_kind, validation_policy)?;
        Ok(NativeIntent {
            id: random_token("intent_"),
            kind: entry.path_kind,
            source_kind,
            path: entry.to_ref(),
            display_label: entry.display_label.clone(),
        })
    }

    fn insert_entry(
        &self,
        classified: ClassifiedPath,
        source_kind: NativePathSourceKind,
        validation_policy: NativePathValidationPolicy,
    ) -> Result<NativePathEntry, String> {
        let token = random_token("path_");
        let expires_at_ms = now_ms() + TOKEN_TTL.as_millis() as u64;
        let entry = NativePathEntry {
            token: token.clone(),
            canonical_path: classified.canonical_path,
            validation_policy,
            path_kind: classified.path_kind,
            path_type: classified.path_type,
            source_kind,
            allowed_operations: classified.allowed_operations,
            display_label: classified.display_label,
            expires_at_ms,
            size_bytes: classified.size_bytes,
            modified_ms: classified.modified_ms,
            device_id: classified.device_id,
            file_id: classified.file_id,
        };
        let mut inner = self.inner.lock().map_err(|e| e.to_string())?;
        inner.tokens.insert(token, entry.clone());
        Ok(entry)
    }

    fn drain_intents(&self) -> Result<Vec<NativeIntent>, String> {
        let mut inner = self.inner.lock().map_err(|e| e.to_string())?;
        prune_expired(&mut inner);
        Ok(inner.queued_intents.drain(..).collect())
    }

    fn entry_for_operation(
        &self,
        token: &str,
        operation: NativePathOperation,
    ) -> Result<NativePathEntry, String> {
        let mut inner = self.inner.lock().map_err(|e| e.to_string())?;
        prune_expired(&mut inner);
        let entry = inner
            .tokens
            .get(token)
            .ok_or_else(|| "Native path token is unavailable or expired.".to_string())?;
        if !entry.allowed_operations.contains(&operation) {
            return Err("Native path token does not allow this operation.".to_string());
        }
        Ok(entry.clone())
    }

    fn sign_grant(
        &self,
        token: &str,
        operation: NativePathOperation,
    ) -> Result<NativePathLeaseResponse, String> {
        let entry = self.entry_for_operation(token, operation)?;
        validate_entry_path(&entry, operation)?;
        sign_path_lease(
            &self.lease_secret,
            NativePathLeaseRequest {
                operation,
                canonical_path: entry.canonical_path.to_string_lossy().to_string(),
                path_kind: entry.path_kind,
                path_type: entry.path_type,
                source_kind: entry.source_kind,
                token: entry.token,
                display_label: entry.display_label,
                size_bytes: entry.size_bytes,
                modified_ms: entry.modified_ms,
                device_id: Some(entry.device_id),
                file_id: Some(entry.file_id),
            },
        )
    }

    fn path_for_operation(
        &self,
        token: &str,
        operation: NativePathOperation,
    ) -> Result<NativePathEntry, String> {
        let entry = self.entry_for_operation(token, operation)?;
        validate_entry_path(&entry, operation)?;
        Ok(entry)
    }
}

fn validate_entry_path(
    entry: &NativePathEntry,
    operation: NativePathOperation,
) -> Result<(), String> {
    let classified = match entry.validation_policy {
        NativePathValidationPolicy::Model => classify_native_model_path(&entry.canonical_path)?,
        NativePathValidationPolicy::Dataset => classify_native_dataset_path(&entry.canonical_path)?,
        NativePathValidationPolicy::Attachment => {
            classify_native_attachment_path(&entry.canonical_path)?
        }
        NativePathValidationPolicy::Artifact(kind) => {
            classify_artifact_path(kind, &entry.canonical_path)?
        }
    };
    let check_fingerprint = !matches!(
        operation,
        NativePathOperation::Reveal | NativePathOperation::Open
    );
    if classified.canonical_path != entry.canonical_path
        || classified.path_kind != entry.path_kind
        || classified.path_type != entry.path_type
        || !classified.allowed_operations.contains(&operation)
        || (check_fingerprint && classified.size_bytes != entry.size_bytes)
        || (check_fingerprint && classified.modified_ms != entry.modified_ms)
        || (check_fingerprint && classified.device_id != entry.device_id)
        || (check_fingerprint && classified.file_id != entry.file_id)
    {
        return Err("Native path changed after it was selected.".to_string());
    }
    Ok(())
}

impl NativePathEntry {
    fn to_ref(&self) -> NativePathRef {
        NativePathRef {
            token: self.token.clone(),
            kind: self.path_kind,
            display_label: self.display_label.clone(),
            allowed_operations: self.allowed_operations.clone(),
            expires_at_ms: self.expires_at_ms,
            size_bytes: self.size_bytes,
            modified_ms: self.modified_ms,
        }
    }
}

fn prune_expired(inner: &mut NativeIntakeInner) {
    let now = now_ms();
    inner.tokens.retain(|_, entry| entry.expires_at_ms > now);
    inner
        .queued_intents
        .retain(|intent| intent.path.expires_at_ms > now);
}

pub(crate) fn ensure_main_window(window: &WebviewWindow) -> Result<(), String> {
    if window.label() == "main" {
        Ok(())
    } else {
        Err("Native path commands are only available to the main window.".to_string())
    }
}

#[tauri::command]
pub fn drain_native_intents(
    window: WebviewWindow,
    state: tauri::State<'_, NativeIntakeState>,
) -> Result<Vec<NativeIntent>, String> {
    ensure_main_window(&window)?;
    state.drain_intents()
}

#[tauri::command]
pub fn register_native_model_path(
    window: WebviewWindow,
    state: tauri::State<'_, NativeIntakeState>,
    path: String,
) -> Result<NativeIntent, String> {
    ensure_main_window(&window)?;
    state.register_model_path(path, NativePathSourceKind::Drop)
}

#[tauri::command]
pub fn register_native_attachment_path(
    window: WebviewWindow,
    state: tauri::State<'_, NativeIntakeState>,
    path: String,
) -> Result<NativeIntent, String> {
    ensure_main_window(&window)?;
    state.register_attachment_path(path, NativePathSourceKind::Drop)
}

#[tauri::command]
pub fn register_native_dataset_path(
    window: WebviewWindow,
    state: tauri::State<'_, NativeIntakeState>,
    path: String,
) -> Result<NativeIntent, String> {
    ensure_main_window(&window)?;
    state.register_dataset_path(path, NativePathSourceKind::Drop)
}

#[tauri::command]
pub async fn pick_native_model(
    window: WebviewWindow,
    app: AppHandle,
    state: tauri::State<'_, NativeIntakeState>,
) -> Result<Option<NativeIntent>, String> {
    ensure_main_window(&window)?;
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog()
        .file()
        .set_title("Choose a GGUF model")
        .add_filter("GGUF models", &["gguf"])
        .pick_file(move |path| {
            let _ = tx.send(path);
        });
    let Some(file_path) = rx.await.map_err(|_| "Dialog closed".to_string())? else {
        return Ok(None);
    };
    let path = file_path
        .into_path()
        .map_err(|_| "Only local filesystem model paths are supported.".to_string())?;
    state
        .register_model_path(path, NativePathSourceKind::Dialog)
        .map(Some)
}

#[tauri::command]
pub async fn pick_hugging_face_cache_dir(
    window: WebviewWindow,
    app: AppHandle,
) -> Result<Option<String>, String> {
    ensure_main_window(&window)?;
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog()
        .file()
        .set_title("Choose model download location")
        .pick_folder(move |path| {
            let _ = tx.send(path);
        });
    let Some(folder_path) = rx.await.map_err(|_| "Dialog closed".to_string())? else {
        return Ok(None);
    };
    let path = folder_path
        .into_path()
        .map_err(|_| "Only local filesystem folders are supported.".to_string())?;
    let canonical = path
        .canonicalize()
        .map_err(|e| format!("Could not use the selected folder: {e}"))?;
    if !canonical.is_dir() {
        return Err("The selected location is not a folder.".to_string());
    }
    Ok(Some(portable_path_string(&canonical)))
}

#[tauri::command]
pub async fn pick_native_document_folder(
    window: WebviewWindow,
    app: AppHandle,
    state: tauri::State<'_, NativeIntakeState>,
) -> Result<Option<NativeDocumentFolderSelection>, String> {
    ensure_main_window(&window)?;
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog()
        .file()
        .set_title("Link a document folder")
        .pick_folder(move |path| {
            let _ = tx.send(path);
        });
    let Some(folder_path) = rx.await.map_err(|_| "Dialog closed".to_string())? else {
        return Ok(None);
    };
    let path = folder_path
        .into_path()
        .map_err(|_| "Only local filesystem folders are supported.".to_string())?;
    state.sign_document_folder_path(path).map(Some)
}

#[tauri::command]
pub fn consume_native_path_token(
    window: WebviewWindow,
    state: tauri::State<'_, NativeIntakeState>,
    token: String,
    operation: NativePathOperation,
) -> Result<NativePathLeaseResponse, String> {
    ensure_main_window(&window)?;
    match operation {
        NativePathOperation::Reveal | NativePathOperation::Open => {
            Err("Reveal/Open do not use backend path grants.".to_string())
        }
        _ => state.sign_grant(&token, operation),
    }
}

#[tauri::command]
pub fn register_artifact_path(
    window: WebviewWindow,
    state: tauri::State<'_, NativeIntakeState>,
    kind: NativeArtifactKind,
    path: String,
) -> Result<NativePathRef, String> {
    ensure_main_window(&window)?;
    state.register_artifact(kind, path)
}

#[tauri::command]
pub fn reveal_path_token(
    window: WebviewWindow,
    state: tauri::State<'_, NativeIntakeState>,
    token: String,
) -> Result<(), String> {
    ensure_main_window(&window)?;
    let entry = state.path_for_operation(&token, NativePathOperation::Reveal)?;
    #[cfg(target_os = "macos")]
    {
        if entry.canonical_path.is_file() {
            return std::process::Command::new("open")
                .arg("-R")
                .arg(&entry.canonical_path)
                .spawn()
                .map(|_| ())
                .map_err(|e| format!("Failed to reveal path: {e}"));
        }
    }
    #[cfg(target_os = "windows")]
    {
        if entry.canonical_path.is_file() {
            let mut select_arg = std::ffi::OsString::from("/select,");
            select_arg.push(entry.canonical_path.as_os_str());
            return std::process::Command::new("explorer")
                .arg(select_arg)
                .spawn()
                .map(|_| ())
                .map_err(|e| format!("Failed to reveal path: {e}"));
        }
    }
    let target = reveal_target(&entry.canonical_path);
    crate::process::open_detached(target).map_err(|e| format!("Failed to reveal path: {e}"))
}

#[tauri::command]
pub fn open_path_token(
    window: WebviewWindow,
    state: tauri::State<'_, NativeIntakeState>,
    token: String,
) -> Result<(), String> {
    ensure_main_window(&window)?;
    let entry = state.path_for_operation(&token, NativePathOperation::Open)?;
    crate::process::open_detached(entry.canonical_path)
        .map_err(|e| format!("Failed to open path: {e}"))
}

// Covers the generic client-side limit (audio, 25 MB).
const MAX_NATIVE_ATTACHMENT_BYTES: u64 = 25 * 1024 * 1024;

// Matches the clipboard reader, so a dropped source file and a pasted one
// accept the same sizes.
const MAX_NATIVE_TEXT_BYTES: u64 = 20 * 1024 * 1024;
// OpenDocument archives use the composer's larger archive limit.
const MAX_NATIVE_OPEN_DOCUMENT_BYTES: u64 = 50 * 1024 * 1024;
// Images stop lower: the composer throws over 20 MB without a toast and the
// drain swallows it, so a larger read loses them silently.
const MAX_NATIVE_IMAGE_BYTES: u64 = 20 * 1024 * 1024;
// The largest client-side video limit: a reference clip, whose 96 MiB cap
// bounds the data URL, not the file. Mirrors rawLimitFor in reference-budget.ts
// so we don't read and encode 96 MiB the caller is about to reject. Each caller
// still enforces its own tighter limit.
const MAX_NATIVE_VIDEO_BYTES: u64 = 75_497_280;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeAttachmentFile {
    name: String,
    mime_type: String,
    base64: String,
}

fn attachment_mime_type(path: &Path) -> Option<&'static str> {
    if is_text_attachment_name(path) {
        return Some("text/plain");
    }
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "jpg" | "jpeg" => Some("image/jpeg"),
        "png" => Some("image/png"),
        "webp" => Some("image/webp"),
        "gif" => Some("image/gif"),
        "wav" => Some("audio/wav"),
        "mp3" | "mp2" => Some("audio/mpeg"),
        "m4a" => Some("audio/mp4"),
        "ogg" | "oga" => Some("audio/ogg"),
        "opus" => Some("audio/opus"),
        "flac" => Some("audio/flac"),
        "aac" => Some("audio/aac"),
        "aiff" | "aif" | "aifc" => Some("audio/aiff"),
        "caf" => Some("audio/x-caf"),
        "wma" => Some("audio/x-ms-wma"),
        "amr" => Some("audio/amr"),
        "mp4" => Some("video/mp4"),
        "m4v" => Some("video/x-m4v"),
        "mov" => Some("video/quicktime"),
        "webm" => Some("video/webm"),
        "mkv" => Some("video/x-matroska"),
        "avi" => Some("video/x-msvideo"),
        "mpg" | "mpeg" => Some("video/mpeg"),
        "wmv" => Some("video/x-ms-wmv"),
        "flv" => Some("video/x-flv"),
        "3gp" => Some("video/3gpp"),
        "ogv" => Some("video/ogg"),
        "ods" => Some("application/vnd.oasis.opendocument.spreadsheet"),
        "odt" => Some("application/vnd.oasis.opendocument.text"),
        // Stamped like native_clipboard.rs.
        "json" | "jsonl" | "ndjson" | "jsonc" | "json5" | "geojson" | "har" | "avsc"
        | "tfstate" => Some("application/json"),
        "mdx" | "rmd" | "qmd" => Some("text/markdown"),
        "csv" => Some("text/csv"),
        "tsv" => Some("text/tab-separated-values"),
        "xml" | "plist" | "resx" | "xliff" | "xlf" | "csproj" | "vbproj" | "fsproj" | "props"
        | "targets" => Some("application/xml"),
        "vtt" => Some("text/vtt"),
        "srt" => Some("application/x-subrip"),
        "ics" => Some("text/calendar"),
        "vcf" => Some("text/vcard"),
        "eml" => Some("message/rfc822"),
        other if crate::native_path_policy::TEXT_ATTACHMENT_EXTS.contains(&other) => {
            Some("text/plain")
        }
        _ => None,
    }
}

fn attachment_payload_mime_type(path: &Path, raw: &[u8]) -> Option<&'static str> {
    if path
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("3gp"))
        && is_audio_only_3gp(raw)
    {
        return Some("audio/3gpp");
    }
    attachment_mime_type(path)
}

// Same shape as the clipboard reader: never traverse a link swapped in after
// the path was validated, and never block the caller on a FIFO.
fn open_attachment_file(path: &Path) -> Result<fs::File, String> {
    let unavailable = || "Path is no longer available.".to_string();
    let metadata = fs::symlink_metadata(path).map_err(|_| unavailable())?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(unavailable());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW)
            .open(path)
            .map_err(|_| unavailable())
    }
    // Windows analogue: open the reparse point itself, then refuse it. Literals
    // because windows-sys is not built with Win32_Storage_FileSystem here.
    #[cfg(windows)]
    {
        use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
        let file = fs::OpenOptions::new()
            .read(true)
            .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
            .open(path)
            .map_err(|_| unavailable())?;
        let opened = file.metadata().map_err(|_| unavailable())?;
        if opened.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 || !opened.is_file() {
            return Err(unavailable());
        }
        Ok(file)
    }
    #[cfg(not(any(unix, windows)))]
    {
        fs::File::open(path).map_err(|_| unavailable())
    }
}

fn read_attachment_payload(entry: &NativePathEntry) -> Result<NativeAttachmentFile, String> {
    let path = &entry.canonical_path;
    let mime_type = attachment_mime_type(path)
        .ok_or_else(|| "Only chat attachments can be read inline.".to_string())?;
    let is_text_attachment = is_text_attachment_name(path)
        || path
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| value.to_ascii_lowercase())
            .is_some_and(|ext| {
                crate::native_path_policy::TEXT_ATTACHMENT_EXTS.contains(&ext.as_str())
            });
    let max_bytes = if is_text_attachment {
        MAX_NATIVE_TEXT_BYTES
    } else if mime_type.starts_with("image/") {
        MAX_NATIVE_IMAGE_BYTES
    } else if mime_type.starts_with("video/") {
        MAX_NATIVE_VIDEO_BYTES
    } else if mime_type.starts_with("application/vnd.oasis.opendocument.") {
        MAX_NATIVE_OPEN_DOCUMENT_BYTES
    } else {
        MAX_NATIVE_ATTACHMENT_BYTES
    };
    let file = open_attachment_file(path)?;
    let metadata = file
        .metadata()
        .map_err(|e| format!("Path is no longer available: {e}"))?;
    if !metadata.is_file() || metadata.len() > max_bytes {
        return Err("Attachment is unavailable or too large.".to_string());
    }
    // path_for_operation validated a fingerprint against the path; bind the
    // handle we are about to read to that same one, or a swap in between wins.
    let modified_ms = metadata
        .modified()
        .ok()
        .and_then(|value| value.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|value| value.as_millis() as u64);
    if Some(metadata.len()) != entry.size_bytes || modified_ms != entry.modified_ms {
        return Err("Native path changed after it was selected.".to_string());
    }
    // The file can grow between the stat and the read, so cap the reader itself.
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(max_bytes + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("Could not read attachment: {e}"))?;
    if bytes.len() as u64 > max_bytes {
        return Err("Attachment is unavailable or too large.".to_string());
    }
    if is_binary_property_list(path, &bytes) {
        return Err(
            "Binary property-list files are not supported. Convert the file to text first."
                .to_string(),
        );
    }
    if is_binary_vobsub(path, &bytes) {
        return Err("VobSub .sub files are not supported as text attachments.".to_string());
    }
    if is_binary_tracker_mod(path, &bytes) {
        return Err("Tracker .mod audio files are not supported as text attachments.".to_string());
    }
    if is_compiled_fortran_mod(path, &bytes) {
        return Err(
            "Compiled Fortran .mod modules are not supported as text attachments.".to_string(),
        );
    }
    if is_binary_office_template(path, &bytes) {
        return Err(
            "Legacy Word and PowerPoint templates are not supported as text attachments."
                .to_string(),
        );
    }
    let mime_type = attachment_payload_mime_type(path, &bytes)
        .ok_or_else(|| "Only chat attachments can be read inline.".to_string())?;
    // A 3GP path is provisionally video until its track handlers are available.
    // Reapply the audio cap after an audio-only recording is identified.
    if mime_type.starts_with("audio/") && bytes.len() as u64 > MAX_NATIVE_ATTACHMENT_BYTES {
        return Err("Attachment is unavailable or too large.".to_string());
    }
    let name = path
        .file_name()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| entry.display_label.clone());
    Ok(NativeAttachmentFile {
        name,
        mime_type: mime_type.to_string(),
        base64: BASE64.encode(bytes),
    })
}

// Async: a sync command would base64 up to 20 MiB on the main thread. Only the
// token lookup stays here; State is not 'static and validation hits the disk.
#[tauri::command]
pub async fn read_native_attachment_file(
    window: WebviewWindow,
    state: tauri::State<'_, NativeIntakeState>,
    token: String,
) -> Result<NativeAttachmentFile, String> {
    ensure_main_window(&window)?;
    let entry = state.entry_for_operation(&token, NativePathOperation::Attach)?;
    tokio::task::spawn_blocking(move || {
        validate_entry_path(&entry, NativePathOperation::Attach)?;
        read_attachment_payload(&entry)
    })
    .await
    .map_err(|_| "Image attachment reader stopped unexpectedly.".to_string())?
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        crate::native_path_policy::scratch_root().join(format!(
            "unsloth-native-intents-{name}-{}-{nanos}",
            std::process::id()
        ))
    }

    fn attachment_entry(path: &Path) -> (NativeIntakeState, NativePathEntry) {
        let state = new_native_intake_state();
        state.note_dropped_paths(std::slice::from_ref(&path.to_path_buf()));
        let intent = state
            .register_attachment_path(path, NativePathSourceKind::Drop)
            .unwrap();
        let entry = state
            .path_for_operation(&intent.path.token, NativePathOperation::Attach)
            .unwrap();
        (state, entry)
    }

    // The reader maps its own mime types; an unmapped one refuses an accepted file.
    #[test]
    fn audio_read_round_trips_with_its_mime_type() {
        for (ext, mime) in [
            ("wav", "audio/wav"),
            ("mp3", "audio/mpeg"),
            ("m4a", "audio/mp4"),
            ("ogg", "audio/ogg"),
            ("oga", "audio/ogg"),
            ("flac", "audio/flac"),
        ] {
            let path = temp_path("clip").with_extension(ext);
            fs::write(&path, b"ID3AUDIO").unwrap();
            let (_state, entry) = attachment_entry(&path);
            let payload = read_attachment_payload(&entry)
                .unwrap_or_else(|error| panic!(".{ext} was unreadable: {error}"));
            assert_eq!(payload.mime_type, mime);
            assert_eq!(BASE64.decode(payload.base64).unwrap(), b"ID3AUDIO");
            assert!(payload.name.ends_with(&format!(".{ext}")));
            let _ = fs::remove_file(path);
        }
    }

    fn bmff_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let size = u32::try_from(8 + payload.len()).unwrap();
        let mut boxed = Vec::with_capacity(size as usize);
        boxed.extend_from_slice(&size.to_be_bytes());
        boxed.extend_from_slice(kind);
        boxed.extend_from_slice(payload);
        boxed
    }

    fn three_gp_with_tracks(handlers: &[[u8; 4]]) -> Vec<u8> {
        let mut moov_payload = Vec::new();
        for handler in handlers {
            let mut hdlr_payload = vec![0; 8];
            hdlr_payload.extend_from_slice(handler);
            let mdia = bmff_box(b"mdia", &bmff_box(b"hdlr", &hdlr_payload));
            moov_payload.extend_from_slice(&bmff_box(b"trak", &mdia));
        }
        bmff_box(b"moov", &moov_payload)
    }

    #[test]
    fn audio_only_3gp_read_is_stamped_as_audio() {
        let path = temp_path("recording").with_extension("3gp");
        let raw = three_gp_with_tracks(&[*b"soun"]);
        fs::write(&path, &raw).unwrap();
        let (_state, entry) = attachment_entry(&path);
        let payload = read_attachment_payload(&entry).unwrap();
        assert_eq!(payload.mime_type, "audio/3gpp");
        assert_eq!(BASE64.decode(payload.base64).unwrap(), raw);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn audio_only_3gp_read_reapplies_the_audio_cap() {
        let path = temp_path("oversized-recording").with_extension("3gp");
        let mut raw = three_gp_with_tracks(&[*b"soun"]);
        raw.resize(MAX_NATIVE_ATTACHMENT_BYTES as usize + 1, 0);
        fs::write(&path, raw).unwrap();
        let (_state, entry) = attachment_entry(&path);
        let Err(error) = read_attachment_payload(&entry) else {
            panic!("expected oversized audio-only 3GP read to fail");
        };
        assert!(error.contains("too large"), "unexpected error: {error}");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn any_video_track_keeps_3gp_on_the_video_adapter() {
        for handlers in [&[*b"vide"][..], &[*b"soun", *b"vide"][..]] {
            let raw = three_gp_with_tracks(handlers);
            assert!(!is_audio_only_3gp(&raw));
            assert_eq!(
                attachment_payload_mime_type(Path::new("clip.3gp"), &raw),
                Some("video/3gpp")
            );
        }
    }

    #[test]
    fn binary_property_list_read_is_rejected() {
        for extension in ["plist", "strings"] {
            let path = temp_path("settings").with_extension(extension);
            fs::write(&path, b"bplist00payload").unwrap();
            let (_state, entry) = attachment_entry(&path);
            let Err(error) = read_attachment_payload(&entry) else {
                panic!("expected binary .{extension} read to fail");
            };
            assert!(error.contains("Binary property-list"));
            let _ = fs::remove_file(path);
        }
    }

    #[test]
    fn binary_vobsub_read_is_rejected() {
        let path = temp_path("movie").with_extension("sub");
        fs::write(&path, b"\x00\x00\x01\xbapayload").unwrap();
        let (_state, entry) = attachment_entry(&path);
        let Err(error) = read_attachment_payload(&entry) else {
            panic!("expected binary VobSub read to fail");
        };
        assert!(error.contains("VobSub"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn tracker_mod_read_is_rejected() {
        let mut tracker = vec![0u8; 1084];
        tracker[1080..].copy_from_slice(b"M.K.");
        let mut soundtracker = vec![0u8; 600 + 1024 + 8];
        soundtracker[43] = 4;
        soundtracker[45] = 64;
        soundtracker[470] = 1;
        soundtracker[471] = 120;

        for (name, bytes) in [("track", tracker), ("classic", soundtracker)] {
            let path = temp_path(name).with_extension("mod");
            fs::write(&path, bytes).unwrap();
            let (_state, entry) = attachment_entry(&path);
            let Err(error) = read_attachment_payload(&entry) else {
                panic!("expected tracker MOD read to fail");
            };
            assert!(error.contains("Tracker .mod"));
            let _ = fs::remove_file(path);
        }
    }

    #[test]
    fn extensionless_containerfile_reads_as_text() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("Containerfile");
        fs::write(&path, b"FROM scratch").unwrap();
        let (_state, entry) = attachment_entry(&path);
        let payload = read_attachment_payload(&entry).unwrap();
        assert_eq!(payload.mime_type, "text/plain");
        assert_eq!(BASE64.decode(payload.base64).unwrap(), b"FROM scratch");
    }

    #[test]
    fn image_read_round_trips_and_names_the_file() {
        let path = temp_path("photo").with_extension("png");
        fs::write(&path, b"\x89PNG\r\n\x1a\n").unwrap();
        let (_state, entry) = attachment_entry(&path);
        let payload = read_attachment_payload(&entry).unwrap();
        assert_eq!(payload.mime_type, "image/png");
        assert_eq!(BASE64.decode(payload.base64).unwrap(), b"\x89PNG\r\n\x1a\n");
        assert!(payload.name.ends_with(".png"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn open_document_read_round_trips_with_its_mime_type() {
        for (ext, mime) in [
            ("ods", "application/vnd.oasis.opendocument.spreadsheet"),
            ("odt", "application/vnd.oasis.opendocument.text"),
        ] {
            let path = temp_path("open-document").with_extension(ext);
            fs::write(&path, b"open-document").unwrap();
            let (_state, entry) = attachment_entry(&path);
            let payload = read_attachment_payload(&entry)
                .unwrap_or_else(|error| panic!(".{ext} was unreadable: {error}"));
            assert_eq!(payload.mime_type, mime);
            assert_eq!(BASE64.decode(payload.base64).unwrap(), b"open-document");
            let _ = fs::remove_file(path);
        }
    }

    #[test]
    fn document_token_is_refused_by_the_image_reader() {
        let path = temp_path("note").with_extension("pdf");
        fs::write(&path, b"%PDF-1.4").unwrap();
        let (_state, entry) = attachment_entry(&path);
        let Err(err) = read_attachment_payload(&entry) else {
            panic!("expected the read to be refused");
        };
        assert!(err.contains("Only chat attachments can be read inline"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn text_attachments_read_inline_with_their_own_cap() {
        for (ext, mime) in [
            ("cs", "text/plain"),
            ("php", "text/plain"),
            ("js", "text/plain"),
            ("json", "application/json"),
            ("csv", "text/csv"),
        ] {
            let path = temp_path("source").with_extension(ext);
            fs::write(&path, b"sample").unwrap();
            let (_state, entry) = attachment_entry(&path);
            let payload = read_attachment_payload(&entry).unwrap();
            assert_eq!(payload.mime_type, mime, "{ext}");
            let _ = fs::remove_file(path);
        }

        let path = temp_path("huge").with_extension("cs");
        fs::write(&path, vec![b'x'; MAX_NATIVE_TEXT_BYTES as usize + 1]).unwrap();
        let (_state, entry) = attachment_entry(&path);
        assert!(read_attachment_payload(&entry).is_err());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn every_text_extension_the_drop_accepts_has_a_mime_type() {
        for ext in crate::native_path_policy::TEXT_ATTACHMENT_EXTS {
            let path = PathBuf::from(format!("sample.{ext}"));
            assert!(attachment_mime_type(&path).is_some(), "{ext}");
        }
    }

    #[test]
    fn image_read_rejects_a_file_swapped_in_after_validation() {
        let path = temp_path("swap").with_extension("png");
        fs::write(&path, b"the dropped image").unwrap();
        let (_state, entry) = attachment_entry(&path);
        fs::write(&path, b"a different file entirely").unwrap();
        let Err(err) = read_attachment_payload(&entry) else {
            panic!("expected the read to be refused");
        };
        assert!(err.contains("changed"), "unexpected error: {err}");
        let _ = fs::remove_file(path);
    }

    // Reading past the image cap would make the file disappear instead of
    // reporting it.
    #[test]
    fn image_read_refuses_more_than_the_image_cap() {
        let path = temp_path("huge").with_extension("png");
        fs::write(&path, vec![0u8; MAX_NATIVE_IMAGE_BYTES as usize + 1]).unwrap();
        let (_state, entry) = attachment_entry(&path);
        let Err(err) = read_attachment_payload(&entry) else {
            panic!("expected the read to be refused");
        };
        assert!(err.contains("too large"), "unexpected error: {err}");
        let _ = fs::remove_file(path);
    }

    // Audio keeps the larger cap: the caps are per kind, not one shared ceiling.
    #[test]
    fn audio_read_allows_more_than_the_image_cap() {
        let path = temp_path("clip").with_extension("wav");
        fs::write(&path, vec![0u8; MAX_NATIVE_IMAGE_BYTES as usize + 1]).unwrap();
        let (_state, entry) = attachment_entry(&path);
        let payload = read_attachment_payload(&entry).expect("audio under the audio cap reads");
        assert_eq!(payload.mime_type, "audio/wav");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn audio_read_refuses_more_than_the_audio_cap() {
        let path = temp_path("huge").with_extension("wav");
        fs::write(&path, vec![0u8; MAX_NATIVE_ATTACHMENT_BYTES as usize + 1]).unwrap();
        let (_state, entry) = attachment_entry(&path);
        let Err(err) = read_attachment_payload(&entry) else {
            panic!("expected the read to be refused");
        };
        assert!(err.contains("too large"), "unexpected error: {err}");
        let _ = fs::remove_file(path);
    }

    #[test]
    fn open_document_read_allows_more_than_the_generic_cap() {
        let path = temp_path("spreadsheet").with_extension("ods");
        fs::write(&path, vec![0u8; MAX_NATIVE_ATTACHMENT_BYTES as usize + 1]).unwrap();
        let (_state, entry) = attachment_entry(&path);
        let payload =
            read_attachment_payload(&entry).expect("OpenDocument archive under 50 MiB reads");
        assert_eq!(
            payload.mime_type,
            "application/vnd.oasis.opendocument.spreadsheet"
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn model_token_issues_distinct_validate_and_load_grants() {
        let state = new_native_intake_state();
        let path = temp_path("model").with_extension("gguf");
        fs::write(&path, b"gguf").unwrap();
        let intent = state
            .register_model_path(&path, NativePathSourceKind::Dialog)
            .unwrap();
        let validate = state
            .sign_grant(&intent.path.token, NativePathOperation::ValidateModel)
            .unwrap();
        let load = state
            .sign_grant(&intent.path.token, NativePathOperation::LoadModel)
            .unwrap();
        assert_ne!(validate.native_path_lease, load.native_path_lease);
        assert!(validate.display_label.ends_with(".gguf"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn model_token_rejects_dataset_operation() {
        let state = new_native_intake_state();
        let path = temp_path("model").with_extension("gguf");
        fs::write(&path, b"gguf").unwrap();
        let intent = state
            .register_model_path(&path, NativePathSourceKind::Dialog)
            .unwrap();
        let err = state
            .sign_grant(&intent.path.token, NativePathOperation::DatasetImport)
            .unwrap_err();
        assert!(err.contains("does not allow"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn dataset_token_issues_an_import_grant() {
        let state = new_native_intake_state();
        let path = temp_path("dataset").with_extension("jsonl");
        fs::write(&path, b"{\"text\":\"hello\"}\n").unwrap();
        state.note_dropped_paths(std::slice::from_ref(&path));
        let intent = state
            .register_dataset_path(&path, NativePathSourceKind::Drop)
            .unwrap();
        let grant = state
            .sign_grant(&intent.path.token, NativePathOperation::DatasetImport)
            .unwrap();
        assert!(grant.display_label.ends_with(".jsonl"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn model_token_revalidates_path_changes() {
        let state = new_native_intake_state();
        let path = temp_path("model").with_extension("gguf");
        fs::write(&path, b"gguf").unwrap();
        let intent = state
            .register_model_path(&path, NativePathSourceKind::Dialog)
            .unwrap();
        fs::write(&path, b"changed").unwrap();
        let err = state
            .sign_grant(&intent.path.token, NativePathOperation::ValidateModel)
            .unwrap_err();
        assert!(err.contains("changed"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn attachment_registration_needs_a_real_drop() {
        let state = new_native_intake_state();
        let path = temp_path("attachment").with_extension("txt");
        fs::write(&path, b"notes").unwrap();

        // A renderer naming a path we never saw dropped gets nothing.
        let err = state
            .register_attachment_path(&path, NativePathSourceKind::Drop)
            .unwrap_err();
        assert!(err.contains("dropped on the window"));

        state.note_dropped_paths(std::slice::from_ref(&path));
        let intent = state
            .register_attachment_path(&path, NativePathSourceKind::Drop)
            .unwrap();
        assert_eq!(intent.kind, NativePathKind::Attachment);
        assert!(intent
            .path
            .allowed_operations
            .contains(&NativePathOperation::Attach));
        // The fingerprint the frontend dedups on comes from the stat, not the label.
        assert_eq!(intent.path.size_bytes, Some(5));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn a_drop_does_not_unlock_its_neighbours() {
        let state = new_native_intake_state();
        let dropped = temp_path("dropped").with_extension("txt");
        let sibling = temp_path("sibling").with_extension("txt");
        fs::write(&dropped, b"dropped").unwrap();
        fs::write(&sibling, b"sibling").unwrap();

        state.note_dropped_paths(std::slice::from_ref(&dropped));
        assert!(state
            .register_attachment_path(&sibling, NativePathSourceKind::Drop)
            .is_err());
        let _ = fs::remove_file(dropped);
        let _ = fs::remove_file(sibling);
    }

    #[test]
    fn windows_verbatim_paths_are_portable() {
        assert_eq!(
            normalize_windows_verbatim_path(r"\\?\C:\models\cache".to_string()),
            r"C:\models\cache"
        );
        assert_eq!(
            normalize_windows_verbatim_path(r"\\?\UNC\server\share\cache".to_string()),
            r"\\server\share\cache"
        );
    }

    #[test]
    fn document_folder_picker_grant_is_not_a_reusable_path_token() {
        let state = new_native_intake_state();
        let path = temp_path("document-folder");
        fs::create_dir(&path).unwrap();

        let lease = state.sign_document_folder_path(&path).unwrap();
        assert!(lease.token.contains('.'));
        let payload = lease.token.split('.').next().unwrap();
        let payload: serde_json::Value =
            serde_json::from_slice(&URL_SAFE_NO_PAD.decode(payload).unwrap()).unwrap();
        assert!(payload["modified_ms"].is_null());
        assert!(payload["device_id"]
            .as_str()
            .is_some_and(|value| !value.is_empty()));
        assert!(payload["file_id"]
            .as_str()
            .is_some_and(|value| !value.is_empty()));
        assert_eq!(
            lease.display_name,
            path.file_name().unwrap().to_string_lossy()
        );
        let response = serde_json::to_value(&lease).unwrap();
        assert_eq!(response["token"], lease.token);
        assert_eq!(response["displayName"], lease.display_name);
        assert!(response.get("path").is_none());
        assert!(state
            .entry_for_operation("path_token", NativePathOperation::LinkDocuments)
            .is_err());
        let _ = fs::remove_dir(path);
    }

    #[cfg(unix)]
    #[test]
    fn reveal_rejects_symlink_replacement() {
        use std::os::unix::fs::symlink;

        let state = new_native_intake_state();
        let path = temp_path("model").with_extension("gguf");
        let target = temp_path("replacement").with_extension("gguf");
        fs::write(&path, b"gguf").unwrap();
        fs::write(&target, b"gguf").unwrap();
        let intent = state
            .register_model_path(&path, NativePathSourceKind::Dialog)
            .unwrap();
        fs::remove_file(&path).unwrap();
        symlink(&target, &path).unwrap();
        let err = state
            .path_for_operation(&intent.path.token, NativePathOperation::Reveal)
            .unwrap_err();
        assert!(err.contains("Symlink") || err.contains("changed"));
        let _ = fs::remove_file(path);
        let _ = fs::remove_file(target);
    }
}
