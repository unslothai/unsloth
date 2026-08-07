use crate::native_backend_lease::{
    encode_secret_env, now_ms, random_token, sign_path_lease, NativePathKind,
    NativePathLeaseResponse, NativePathOperation, NativePathSourceKind, NativePathType,
};
use crate::native_path_policy::{
    classify_artifact_path, classify_native_attachment_path, classify_native_dataset_path,
    classify_native_model_path, reveal_target, ClassifiedPath, NativeArtifactKind,
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
            operation,
            entry.canonical_path.to_string_lossy().to_string(),
            entry.path_kind,
            entry.path_type,
            entry.source_kind,
            &entry.token,
            entry.display_label,
            entry.size_bytes,
            entry.modified_ms,
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
    open::that_detached(target).map_err(|e| format!("Failed to reveal path: {e}"))
}

#[tauri::command]
pub fn open_path_token(
    window: WebviewWindow,
    state: tauri::State<'_, NativeIntakeState>,
    token: String,
) -> Result<(), String> {
    ensure_main_window(&window)?;
    let entry = state.path_for_operation(&token, NativePathOperation::Open)?;
    open::that_detached(entry.canonical_path).map_err(|e| format!("Failed to open path: {e}"))
}

const MAX_NATIVE_ATTACHMENT_BYTES: u64 = 20 * 1024 * 1024;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeAttachmentFile {
    name: String,
    mime_type: String,
    base64: String,
}

fn attachment_mime_type(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "jpg" | "jpeg" => Some("image/jpeg"),
        "png" => Some("image/png"),
        "webp" => Some("image/webp"),
        "gif" => Some("image/gif"),
        _ => None,
    }
}

#[tauri::command]
pub fn read_native_attachment_file(
    window: WebviewWindow,
    state: tauri::State<'_, NativeIntakeState>,
    token: String,
) -> Result<NativeAttachmentFile, String> {
    ensure_main_window(&window)?;
    let entry = state.path_for_operation(&token, NativePathOperation::Attach)?;
    let path = &entry.canonical_path;
    let mime_type = attachment_mime_type(path).ok_or_else(|| {
        "Only chat image attachments can be read for vision input.".to_string()
    })?;
    let file = fs::File::open(path).map_err(|e| format!("Path is no longer available: {e}"))?;
    let metadata = file
        .metadata()
        .map_err(|e| format!("Path is no longer available: {e}"))?;
    if !metadata.is_file() || metadata.len() > MAX_NATIVE_ATTACHMENT_BYTES {
        return Err("Image attachment is unavailable or too large.".to_string());
    }
    // The file can still grow between the stat and the read, so cap the reader
    // itself rather than trusting the size we just measured.
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_NATIVE_ATTACHMENT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("Could not read image attachment: {e}"))?;
    if bytes.len() as u64 > MAX_NATIVE_ATTACHMENT_BYTES {
        return Err("Image attachment is unavailable or too large.".to_string());
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "unsloth-native-intents-{name}-{}-{nanos}",
            std::process::id()
        ))
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

        state.note_dropped_paths(&[path.clone()]);
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

        state.note_dropped_paths(&[dropped.clone()]);
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
