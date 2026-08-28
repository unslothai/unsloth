use crate::native_backend_lease::{NativePathKind, NativePathOperation, NativePathType};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NativeArtifactKind {
    TrainingOutput,
    Export,
    DatasetUpload,
    RecipeArtifact,
    DiagnosticLog,
}

#[derive(Clone, Debug)]
pub struct ClassifiedPath {
    pub canonical_path: PathBuf,
    pub path_kind: NativePathKind,
    pub path_type: NativePathType,
    pub allowed_operations: Vec<NativePathOperation>,
    pub display_label: String,
    pub size_bytes: Option<u64>,
    pub modified_ms: Option<u64>,
    pub device_id: String,
    pub file_id: String,
}

pub fn classify_native_model_path(path: &Path) -> Result<ClassifiedPath, String> {
    let classified = classify_existing_path(path)?;
    if classified.path_type != NativePathType::File {
        return Err("Only GGUF model files are supported for native model intake.".to_string());
    }
    if !has_extension(&classified.canonical_path, "gguf") {
        return Err("Only .gguf model files are supported for native model intake.".to_string());
    }
    Ok(ClassifiedPath {
        path_kind: NativePathKind::Model,
        allowed_operations: vec![
            NativePathOperation::ValidateModel,
            NativePathOperation::LoadModel,
            NativePathOperation::Reveal,
        ],
        ..classified
    })
}

/// Document types the RAG ingest accepts; keep in sync with `config.UPLOAD_EXTS`.
pub const ATTACHMENT_EXTS: &[&str] = &["pdf", "txt", "md", "markdown", "docx", "html", "htm"];
/// OpenDocument files the chat composer parses directly rather than indexing as RAG sources.
pub const OPEN_DOCUMENT_ATTACHMENT_EXTS: &[&str] = &["ods", "odt"];
pub const TRAINING_DATASET_EXTS: &[&str] = &["csv", "json", "jsonl", "parquet"];

/// Keep in sync with `text-attachment-accept.ts`. RAG types are absent so a
/// dropped .txt/.md keeps being indexed.
pub const TEXT_ATTACHMENT_EXTS: &[&str] = &[
    "text",
    "log",
    "mdx",
    "rst",
    "csv",
    "tsv",
    "json",
    "jsonl",
    "ndjson",
    "xml",
    "yaml",
    "yml",
    "toml",
    "ini",
    "cfg",
    "conf",
    "env",
    "properties",
    "css",
    "scss",
    "sass",
    "less",
    "svg",
    "js",
    "jsx",
    "mjs",
    "cjs",
    "ts",
    "tsx",
    "py",
    "pyi",
    "ipynb",
    "rb",
    "php",
    "go",
    "rs",
    "java",
    "kt",
    "kts",
    "scala",
    "swift",
    "c",
    "h",
    "cc",
    "cpp",
    "hpp",
    "cxx",
    "cs",
    "m",
    "mm",
    "sh",
    "bash",
    "zsh",
    "fish",
    "ps1",
    "bat",
    "lua",
    "pl",
    "pm",
    "r",
    "jl",
    "dart",
    "vue",
    "svelte",
    "astro",
    "sql",
    "graphql",
    "gql",
    "proto",
    "tf",
    "tfvars",
    "gradle",
    "dockerfile",
    "makefile",
    "cmake",
    "diff",
    "patch",
];

/// Vision chat image attachments; keep in sync with `drop-paths.ts` `CHAT_IMAGE_DROP_ACCEPT`.
pub const IMAGE_ATTACHMENT_EXTS: &[&str] = &["jpg", "jpeg", "png", "webp", "gif"];

/// Chat audio attachments; keep in sync with `audio-attachment-adapter.ts` `accept`.
pub const AUDIO_ATTACHMENT_EXTS: &[&str] = &["wav", "mp3", "m4a", "ogg", "oga", "flac"];

/// Chat video attachments; keep in sync with `drop-paths.ts`
/// `CHAT_VIDEO_DROP_ACCEPT`. llama-server decodes with ffmpeg, so this is what
/// ffmpeg reads, not what the webview can play.
pub const VIDEO_ATTACHMENT_EXTS: &[&str] = &["mp4", "mov", "webm", "mkv", "avi"];

fn accepted_attachment_exts() -> impl Iterator<Item = &'static &'static str> {
    ATTACHMENT_EXTS
        .iter()
        .chain(OPEN_DOCUMENT_ATTACHMENT_EXTS.iter())
        .chain(TEXT_ATTACHMENT_EXTS.iter())
        .chain(IMAGE_ATTACHMENT_EXTS.iter())
        .chain(AUDIO_ATTACHMENT_EXTS.iter())
        .chain(VIDEO_ATTACHMENT_EXTS.iter())
}

pub fn classify_native_attachment_path(path: &Path) -> Result<ClassifiedPath, String> {
    let classified = classify_existing_path(path)?;
    if classified.path_type != NativePathType::File {
        return Err("Only files can be attached to a chat.".to_string());
    }
    let supported =
        accepted_attachment_exts().any(|ext| has_extension(&classified.canonical_path, ext));
    if !supported {
        return Err(format!(
            "Unsupported attachment type. Supported: {}",
            accepted_attachment_exts()
                .map(|ext| format!(".{ext}"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    Ok(ClassifiedPath {
        path_kind: NativePathKind::Attachment,
        allowed_operations: vec![NativePathOperation::Attach, NativePathOperation::Reveal],
        ..classified
    })
}

pub fn classify_native_document_folder(path: &Path) -> Result<ClassifiedPath, String> {
    let classified = classify_existing_path(path)?;
    if classified.path_type != NativePathType::Directory {
        return Err("Only folders can be linked as document sources.".to_string());
    }
    reject_document_folder_root(&classified.canonical_path)?;
    reject_sensitive_document_folder(&classified.canonical_path)?;
    Ok(ClassifiedPath {
        path_kind: NativePathKind::DocumentFolder,
        allowed_operations: vec![NativePathOperation::LinkDocuments],
        ..classified
    })
}

pub fn classify_native_project_workspace(path: &Path) -> Result<ClassifiedPath, String> {
    let classified = classify_existing_path(path)?;
    if classified.path_type != NativePathType::Directory {
        return Err("Only folders can be used as project workspaces.".to_string());
    }
    if classified.canonical_path.parent().is_none() {
        return Err("A filesystem root cannot be used as a project workspace.".to_string());
    }
    reject_sensitive_document_folder(&classified.canonical_path).map_err(|_| {
        "Sensitive system or application folders cannot be used as project workspaces.".to_string()
    })?;
    Ok(ClassifiedPath {
        path_kind: NativePathKind::ProjectWorkspace,
        allowed_operations: vec![NativePathOperation::SetProjectWorkspace],
        ..classified
    })
}

pub fn classify_native_dataset_path(path: &Path) -> Result<ClassifiedPath, String> {
    let classified = classify_existing_path(path)?;
    if classified.path_type != NativePathType::File {
        return Err("Only files can be imported as training datasets.".to_string());
    }
    if !TRAINING_DATASET_EXTS
        .iter()
        .any(|ext| has_extension(&classified.canonical_path, ext))
    {
        return Err(format!(
            "Unsupported training dataset type. Supported: {}",
            TRAINING_DATASET_EXTS
                .iter()
                .map(|ext| format!(".{ext}"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    Ok(ClassifiedPath {
        path_kind: NativePathKind::Dataset,
        allowed_operations: vec![
            NativePathOperation::DatasetImport,
            NativePathOperation::Reveal,
        ],
        ..classified
    })
}

pub fn classify_artifact_path(
    kind: NativeArtifactKind,
    path: &Path,
) -> Result<ClassifiedPath, String> {
    let classified = classify_existing_path(path)?;
    ensure_artifact_root(kind, &classified.canonical_path)?;
    reject_sensitive_artifact(&classified.canonical_path)?;

    let mut allowed_operations = vec![NativePathOperation::Reveal];
    if is_open_safe_artifact(&classified.canonical_path, classified.path_type) {
        allowed_operations.push(NativePathOperation::Open);
    }

    Ok(ClassifiedPath {
        path_kind: NativePathKind::Artifact,
        allowed_operations,
        ..classified
    })
}

pub fn refresh_path_fingerprint(
    path: &Path,
) -> Result<(NativePathType, Option<u64>, Option<u64>), String> {
    let metadata = fs::metadata(path).map_err(|e| format!("Path is no longer available: {e}"))?;
    let path_type = if metadata.is_file() {
        NativePathType::File
    } else if metadata.is_dir() {
        NativePathType::Directory
    } else {
        return Err("Special files are not supported.".to_string());
    };
    let size_bytes = metadata.is_file().then_some(metadata.len());
    let modified_ms = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_millis() as u64);
    Ok((path_type, size_bytes, modified_ms))
}

#[cfg(unix)]
fn stable_path_identity(path: &Path) -> Result<(String, String), String> {
    use std::os::unix::fs::MetadataExt;

    let metadata = fs::metadata(path).map_err(|e| format!("Path is no longer available: {e}"))?;
    Ok((
        format!("{:x}", metadata.dev()),
        format!("{:x}", metadata.ino()),
    ))
}

#[cfg(windows)]
fn stable_path_identity(path: &Path) -> Result<(String, String), String> {
    use std::mem::zeroed;
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        FileIdInfo, GetFileInformationByHandle, GetFileInformationByHandleEx,
        BY_HANDLE_FILE_INFORMATION, FILE_ID_INFO,
    };

    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
    let file = fs::OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)
        .map_err(|e| format!("Path is no longer available: {e}"))?;
    let mut info: BY_HANDLE_FILE_INFORMATION = unsafe { zeroed() };
    let ok = unsafe {
        GetFileInformationByHandle(file.as_raw_handle() as _, std::ptr::addr_of_mut!(info))
    };
    if ok == 0 || info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err("Path is no longer available.".to_string());
    }
    let legacy_device_id = info.dwVolumeSerialNumber as u64;
    let legacy_file_id = ((info.nFileIndexHigh as u64) << 32) | info.nFileIndexLow as u64;
    let mut identity: FILE_ID_INFO = unsafe { zeroed() };
    let extended_ok = unsafe {
        GetFileInformationByHandleEx(
            file.as_raw_handle() as _,
            FileIdInfo,
            std::ptr::addr_of_mut!(identity).cast(),
            std::mem::size_of::<FILE_ID_INFO>() as u32,
        )
    };
    if extended_ok != 0 {
        let file_id = u128::from_le_bytes(identity.FileId.Identifier);
        if identity.VolumeSerialNumber == legacy_device_id && file_id == legacy_file_id as u128 {
            return Ok((
                format!("{legacy_device_id:x}"),
                format!("{legacy_file_id:x}"),
            ));
        }
        // Python <=3.11 exposes the legacy pair; Python >=3.12 exposes FILE_ID_INFO.
        return Ok((
            format!("{legacy_device_id:x}:{:x}", identity.VolumeSerialNumber),
            format!("{legacy_file_id:x}:{file_id:x}"),
        ));
    }
    Ok((
        format!("{legacy_device_id:x}"),
        format!("{legacy_file_id:x}"),
    ))
}

pub fn reveal_target(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent().unwrap_or(path).to_path_buf()
    }
}

fn sanitize_display_label(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|ch| if ch.is_control() { ' ' } else { ch })
        .collect();
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        "Selected path".to_string()
    } else {
        trimmed.chars().take(160).collect()
    }
}

pub fn is_open_safe_artifact(path: &Path, path_type: NativePathType) -> bool {
    if path_type == NativePathType::Directory {
        return false;
    }
    let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
        return false;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "txt" | "log" | "json" | "jsonl" | "csv" | "tsv" | "parquet" | "md"
    )
}

fn classify_existing_path(path: &Path) -> Result<ClassifiedPath, String> {
    reject_network_or_device_path(path)?;
    let symlink_metadata =
        fs::symlink_metadata(path).map_err(|e| format!("Path is not available: {e}"))?;
    if symlink_metadata.file_type().is_symlink() {
        return Err("Symlink paths are not supported for native intake.".to_string());
    }

    let canonical_path = path
        .canonicalize()
        .map_err(|e| format!("Path could not be resolved: {e}"))?;
    reject_network_or_device_path(&canonical_path)?;
    let canonical_symlink_metadata =
        fs::symlink_metadata(&canonical_path).map_err(|e| format!("Path is not available: {e}"))?;
    if canonical_symlink_metadata.file_type().is_symlink() {
        return Err("Symlink paths are not supported for native intake.".to_string());
    }
    let (path_type, size_bytes, modified_ms) = refresh_path_fingerprint(&canonical_path)?;
    let (device_id, file_id) = stable_path_identity(&canonical_path)?;
    let display_label = sanitize_display_label(
        canonical_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("Selected path"),
    );

    Ok(ClassifiedPath {
        canonical_path,
        path_kind: NativePathKind::Artifact,
        path_type,
        allowed_operations: vec![NativePathOperation::Reveal],
        display_label,
        size_bytes,
        modified_ms,
        device_id,
        file_id,
    })
}

fn ensure_artifact_root(kind: NativeArtifactKind, canonical_path: &Path) -> Result<(), String> {
    let Some(home) = dirs::home_dir() else {
        return Err("Could not determine home directory.".to_string());
    };
    let studio = home.join(".unsloth").join("studio");
    let allowed_root = match kind {
        NativeArtifactKind::TrainingOutput => studio.join("outputs"),
        NativeArtifactKind::Export => studio.join("exports"),
        NativeArtifactKind::DatasetUpload => studio.join("assets").join("datasets").join("uploads"),
        NativeArtifactKind::RecipeArtifact => {
            studio.join("assets").join("datasets").join("recipes")
        }
        NativeArtifactKind::DiagnosticLog => studio.join("logs"),
    };
    let root = allowed_root
        .canonicalize()
        .map_err(|_| "Artifact root is not available.".to_string())?;
    if canonical_path == root || canonical_path.starts_with(&root) {
        Ok(())
    } else {
        Err("Artifact path is outside the allowed artifact root.".to_string())
    }
}

fn reject_sensitive_artifact(path: &Path) -> Result<(), String> {
    let has_sensitive_segment = path.components().any(|component| {
        let segment = component.as_os_str().to_string_lossy().to_ascii_lowercase();
        segment == "auth"
            || segment == "pid"
            || segment.starts_with("auth.db")
            || segment.starts_with("studio.db")
    });
    if has_sensitive_segment {
        return Err("Sensitive Unsloth state cannot be registered as an artifact.".to_string());
    }
    if let Some(ext) = path.extension().and_then(|ext| ext.to_str()) {
        if matches!(
            ext.to_ascii_lowercase().as_str(),
            "exe" | "dll" | "dylib" | "so" | "sh" | "bash" | "zsh" | "ps1" | "bat" | "cmd"
        ) {
            return Err(
                "Executable artifacts cannot be registered for native open/reveal.".to_string(),
            );
        }
    }
    Ok(())
}

fn reject_document_folder_root(path: &Path) -> Result<(), String> {
    if path.parent().is_none() {
        Err("A filesystem root cannot be linked as a document folder.".to_string())
    } else {
        Ok(())
    }
}

pub(crate) fn reject_sensitive_document_folder(path: &Path) -> Result<(), String> {
    let has_sensitive_segment = path.components().any(|component| {
        matches!(
            component
                .as_os_str()
                .to_string_lossy()
                .to_ascii_lowercase()
                .as_str(),
            ".1password"
                | ".aws"
                | ".azure"
                | ".bitwarden"
                | ".config"
                | ".docker"
                | ".gcloud"
                | ".gnupg"
                | ".huggingface"
                | ".kaggle"
                | ".kube"
                | ".local"
                | ".modelscope"
                | ".mozilla"
                | ".ngc"
                | ".password-store"
                | ".pki"
                | ".ssh"
                | ".thunderbird"
                | "1password"
                | "bitwarden"
                | "keychains"
                | "keyrings"
                | "mozilla"
                | "thunderbird"
        )
    });
    if has_sensitive_segment {
        return Err(
            "Sensitive system or application folders cannot be linked as document sources."
                .to_string(),
        );
    }

    let mut sensitive_roots = Vec::new();
    if let Some(home) = dirs::home_dir() {
        if same_native_path(path, &home) {
            return Err(
                "The entire home folder cannot be linked as a document source.".to_string(),
            );
        }
        for relative in [".unsloth"] {
            sensitive_roots.push(home.join(relative));
        }
    }
    if let Some(config) = dirs::config_dir() {
        sensitive_roots.push(config);
    }
    if let Some(data) = dirs::data_local_dir() {
        sensitive_roots.push(data);
    }

    #[cfg(unix)]
    sensitive_roots.extend(
        [
            "/boot", "/etc", "/root", "/run", "/usr", "/var/lib", "/var/run",
        ]
        .into_iter()
        .map(PathBuf::from),
    );
    #[cfg(target_os = "macos")]
    sensitive_roots.extend(
        ["/Library", "/System", "/private"]
            .into_iter()
            .map(PathBuf::from),
    );
    #[cfg(windows)]
    for variable in ["WINDIR", "ProgramFiles", "ProgramFiles(x86)", "ProgramData"] {
        if let Some(value) = std::env::var_os(variable) {
            sensitive_roots.push(PathBuf::from(value));
        }
    }

    if sensitive_roots.iter().any(|sensitive| {
        same_path_or_descendant(path, sensitive)
            && !(is_linux_removable_media_path(path)
                && same_native_path(sensitive, Path::new("/run")))
    }) {
        Err(
            "Sensitive system or application folders cannot be linked as document sources."
                .to_string(),
        )
    } else {
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn is_linux_removable_media_path(path: &Path) -> bool {
    let media_root = Path::new("/run/media");
    path != media_root && path.starts_with(media_root)
}

#[cfg(not(target_os = "linux"))]
fn is_linux_removable_media_path(_path: &Path) -> bool {
    false
}

/// One spelling for both sides of a Windows path comparison.
///
/// `Path::canonicalize` returns a verbatim `\\?\C:\...` path, while `dirs::home_dir` and
/// `%WINDIR%` come back plain. Every caller here classifies a path first, so the guard was
/// comparing `\\?\C:\Users\me` against `C:\Users\me` and missing on all of them: the whole
/// sensitive-folder check was inert. Strip the prefix instead of fixing it per call site,
/// because the callers keep multiplying.
///
/// The trailing separator has to go too, or a drive root stored as `C:\` fails the descendant
/// test against `C:\Users` for want of a doubled backslash.
#[cfg(windows)]
fn comparable_native_path(path: &Path) -> String {
    let text = path.to_string_lossy().replace('/', "\\");
    // `\\?\UNC\server\share` is the verbatim spelling of `\\server\share`.
    let stripped = match text.strip_prefix("\\\\?\\UNC\\") {
        Some(rest) => format!("\\\\{rest}"),
        None => text.strip_prefix("\\\\?\\").unwrap_or(&text).to_string(),
    };
    let trimmed = stripped.trim_end_matches('\\');
    if trimmed.is_empty() { stripped } else { trimmed.to_ascii_lowercase() }
}

fn same_native_path(left: &Path, right: &Path) -> bool {
    #[cfg(windows)]
    {
        return comparable_native_path(left) == comparable_native_path(right);
    }
    #[cfg(not(windows))]
    {
        left == right
    }
}

fn same_path_or_descendant(path: &Path, root: &Path) -> bool {
    #[cfg(windows)]
    {
        let path = comparable_native_path(path);
        let root = comparable_native_path(root);
        return path == root
            || path
                .strip_prefix(&root)
                .is_some_and(|rest| rest.starts_with('\\'));
    }
    #[cfg(not(windows))]
    {
        path == root || path.starts_with(root)
    }
}

fn reject_network_or_device_path(path: &Path) -> Result<(), String> {
    let text = path.to_string_lossy();
    #[cfg(windows)]
    {
        let normalized = text.replace('/', "\\").to_ascii_lowercase();
        if let Some(rest) = normalized.strip_prefix("\\\\?\\") {
            let bytes = rest.as_bytes();
            if !(bytes.len() >= 3
                && bytes[0].is_ascii_alphabetic()
                && bytes[1] == b':'
                && bytes[2] == b'\\')
            {
                return Err("Network paths are not supported for native intake.".to_string());
            }
        } else if normalized.starts_with("\\\\") {
            return Err("Network paths are not supported for native intake.".to_string());
        }
    }
    #[cfg(unix)]
    {
        for root in ["/dev", "/proc", "/sys"] {
            if path.starts_with(root) {
                return Err("Device and virtual filesystem paths are not supported.".to_string());
            }
        }
    }
    if text.contains('\0') {
        return Err("Path contains invalid NUL characters.".to_string());
    }
    Ok(())
}

fn has_extension(path: &Path, expected: &str) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case(expected))
        .unwrap_or(false)
}

/// Guards process-global environment variables for tests.
///
/// `std::env::set_var` is process-wide while the harness runs tests in
/// parallel, and the policy below reads `XDG_DATA_HOME` through
/// `dirs::data_local_dir`. Anything that sets or depends on those variables
/// takes this, `main.rs`'s `with_xdg_data_home` included.
#[cfg(test)]
pub(crate) static PROCESS_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// A directory that scratch paths for the path-policy tests can be built in:
/// writable, and accepted by the very policy those tests exercise.
///
/// Picking one is not obvious. `std::env::temp_dir()` is out on macOS, where it
/// is /var/folders/..., whose real path is under /private and which the policy
/// rejects on purpose, so every scratch path built from it would fail for that
/// reason rather than the one under test. The test binary's directory is the
/// usual answer but is not safe either: a checkout under /root or /usr/src puts
/// it inside a sensitive root, a target directory on /dev/shm or a UNC share is
/// refused as a device path, and a read-only build tree cannot be written to at
/// all.
///
/// So each candidate is tried for real: a uniquely named child is created in
/// it, a file is written inside that child, and the child is put through
/// `classify_native_document_folder`, which is the whole policy rather than one
/// clause of it. The first candidate that survives all three wins. Chosen once
/// per test binary, under `PROCESS_ENV_LOCK`, so a concurrent
/// `with_xdg_data_home` cannot change the answer midway.
#[cfg(test)]
pub(crate) fn scratch_root() -> PathBuf {
    static ROOT: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    ROOT.get_or_init(|| {
        let _guard = PROCESS_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // The temp directory first, since the OS clears it, then the build
        // tree, then home. On macOS the first is rejected by the policy and the
        // second is taken, which is the case that started all this.
        let candidates = [
            Some(std::env::temp_dir()),
            std::env::current_exe()
                .ok()
                .and_then(|exe| exe.parent().map(Path::to_path_buf)),
            dirs::home_dir(),
        ];
        for candidate in candidates.into_iter().flatten() {
            // tempfile, not a name derived from the PID: the first candidate is
            // a shared temp directory, where another local user can pre-create
            // a predictable path and leave a symlink for the probe write below
            // to follow. tempdir_in creates an unguessable directory with an
            // exclusive operation, or fails.
            let Ok(child) = tempfile::Builder::new()
                .prefix("unsloth-test-scratch-")
                .tempdir_in(&candidate)
            else {
                continue;
            };
            // Existing is not writable: a directory that is already there is
            // accepted without a byte being written.
            if fs::write(child.path().join("writable"), b"1").is_err() {
                continue;
            }
            if classify_native_document_folder(child.path()).is_ok() {
                // Kept rather than dropped, which would delete it out from
                // under every test that follows.
                return child.keep();
            }
        }
        // Nothing qualifies. Fall back rather than skip, so the tests fail
        // loudly here instead of quietly not running.
        std::env::temp_dir()
    })
    .clone()
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
        scratch_root().join(format!(
            "unsloth-native-policy-{name}-{}-{nanos}",
            std::process::id()
        ))
    }

    #[test]
    fn gguf_model_allows_validate_load_reveal() {
        let path = temp_path("model").with_extension("gguf");
        fs::write(&path, b"gguf").unwrap();
        let classified = classify_native_model_path(&path).unwrap();
        assert_eq!(classified.path_kind, NativePathKind::Model);
        assert!(classified
            .allowed_operations
            .contains(&NativePathOperation::ValidateModel));
        assert!(classified
            .allowed_operations
            .contains(&NativePathOperation::LoadModel));
        assert!(classified
            .allowed_operations
            .contains(&NativePathOperation::Reveal));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn non_gguf_model_is_rejected() {
        let path = temp_path("model").with_extension("txt");
        fs::write(&path, b"not gguf").unwrap();
        assert!(classify_native_model_path(&path).is_err());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn document_allows_attach_and_reveal_only() {
        let path = temp_path("doc").with_extension("pdf");
        fs::write(&path, b"%PDF-1.4").unwrap();
        let classified = classify_native_attachment_path(&path).unwrap();
        assert_eq!(classified.path_kind, NativePathKind::Attachment);
        assert!(classified
            .allowed_operations
            .contains(&NativePathOperation::Attach));
        assert!(!classified
            .allowed_operations
            .contains(&NativePathOperation::LoadModel));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn image_allows_attach_and_reveal_only() {
        let path = temp_path("photo").with_extension("png");
        fs::write(&path, b"\x89PNG\r\n").unwrap();
        let classified = classify_native_attachment_path(&path).unwrap();
        assert_eq!(classified.path_kind, NativePathKind::Attachment);
        assert!(classified
            .allowed_operations
            .contains(&NativePathOperation::Attach));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn unsupported_attachment_type_is_rejected() {
        let path = temp_path("doc").with_extension("exe");
        fs::write(&path, b"MZ").unwrap();
        assert!(classify_native_attachment_path(&path).is_err());
        let _ = fs::remove_file(path);
    }

    // The composer takes audio uploads, so a dropped one has to classify too.
    #[test]
    fn audio_attachments_are_accepted() {
        for ext in AUDIO_ATTACHMENT_EXTS {
            let path = temp_path("clip").with_extension(ext);
            fs::write(&path, b"ID3").unwrap();
            let classified = classify_native_attachment_path(&path)
                .unwrap_or_else(|error| panic!(".{ext} was rejected: {error}"));
            assert_eq!(classified.path_kind, NativePathKind::Attachment);
            assert!(classified
                .allowed_operations
                .contains(&NativePathOperation::Attach));
            let _ = fs::remove_file(path);
        }
    }

    #[test]
    fn document_folder_is_directory_with_link_only_capability() {
        let path = temp_path("documents");
        fs::create_dir(&path).unwrap();
        let classified = classify_native_document_folder(&path).unwrap();
        assert_eq!(classified.path_kind, NativePathKind::DocumentFolder);
        assert_eq!(classified.path_type, NativePathType::Directory);
        assert_eq!(
            classified.allowed_operations,
            vec![NativePathOperation::LinkDocuments]
        );
        let _ = fs::remove_dir(path);
    }

    #[test]
    fn document_folder_rejects_files_and_filesystem_root() {
        let file = temp_path("documents-file");
        fs::write(&file, b"not a folder").unwrap();
        assert!(classify_native_document_folder(&file).is_err());
        assert!(classify_native_document_folder(Path::new(std::path::MAIN_SEPARATOR_STR)).is_err());
        let _ = fs::remove_file(file);
    }

    #[test]
    fn project_workspace_is_directory_with_workspace_only_capability() {
        let path = temp_path("project-workspace");
        fs::create_dir(&path).unwrap();
        let classified = classify_native_project_workspace(&path).unwrap();
        assert_eq!(classified.path_kind, NativePathKind::ProjectWorkspace);
        assert_eq!(classified.path_type, NativePathType::Directory);
        assert_eq!(
            classified.allowed_operations,
            vec![NativePathOperation::SetProjectWorkspace]
        );
        let _ = fs::remove_dir(path);
    }

    #[test]
    fn project_workspace_rejects_files_and_sensitive_folders() {
        let file = temp_path("project-workspace-file");
        fs::write(&file, b"not a folder").unwrap();
        assert!(classify_native_project_workspace(&file).is_err());
        assert!(
            classify_native_project_workspace(Path::new(std::path::MAIN_SEPARATOR_STR)).is_err()
        );
        if let Some(home) = dirs::home_dir() {
            assert!(classify_native_project_workspace(&home).is_err());
        }
        let _ = fs::remove_file(file);
    }

    /// The guard sees whatever `classify_existing_path` canonicalized, so it has to match a
    /// verbatim path against the plain one every source of a sensitive root hands back. On
    /// Windows those two spellings are different strings and the comparison used to miss,
    /// which let the entire home folder through as a workspace.
    #[test]
    fn sensitive_roots_match_across_path_spellings() {
        let Some(home) = dirs::home_dir() else { return };
        assert!(reject_sensitive_document_folder(&home).is_err());
        let canonical = home.canonicalize().expect("home resolves");
        assert!(
            reject_sensitive_document_folder(&canonical).is_err(),
            "canonical {} slipped past the guard that rejects {}",
            canonical.display(),
            home.display(),
        );
        assert!(same_native_path(&canonical, &home));
        assert!(same_path_or_descendant(&canonical.join("Documents"), &home));
    }

    #[cfg(windows)]
    #[test]
    fn windows_system_roots_are_rejected_after_canonicalizing() {
        for variable in ["WINDIR", "ProgramFiles", "ProgramData"] {
            let Some(value) = std::env::var_os(variable) else {
                continue;
            };
            let plain = PathBuf::from(value);
            let Ok(canonical) = plain.canonicalize() else {
                continue;
            };
            assert!(
                reject_sensitive_document_folder(&canonical).is_err(),
                "%{variable}% as {} was accepted",
                canonical.display(),
            );
            assert!(classify_native_project_workspace(&plain).is_err());
        }
    }

    #[cfg(windows)]
    #[test]
    fn a_drive_root_still_contains_its_children() {
        assert!(same_path_or_descendant(
            Path::new("C:\\Users"),
            Path::new("C:\\"),
        ));
        assert!(!same_path_or_descendant(
            Path::new("C:\\Users"),
            Path::new("D:\\"),
        ));
    }

    #[test]
    fn document_folder_policy_allows_normal_home_subfolders_but_not_credentials() {
        let Some(home) = dirs::home_dir() else { return };
        assert!(reject_sensitive_document_folder(&home.join("Documents")).is_ok());
        assert!(reject_sensitive_document_folder(&home.join(".ssh")).is_err());
        assert!(reject_sensitive_document_folder(&home.join(".huggingface")).is_err());
        assert!(reject_sensitive_document_folder(&home.join("work").join(".local")).is_err());
        assert!(reject_sensitive_document_folder(&home.join("work").join("keyrings")).is_err());
        assert!(reject_sensitive_document_folder(&home.join(".unsloth").join("studio")).is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn document_folder_policy_allows_linux_removable_media_under_run() {
        assert!(
            reject_sensitive_document_folder(Path::new("/run/media/user/USB/Documents")).is_ok()
        );
        assert!(reject_sensitive_document_folder(Path::new("/run/media")).is_err());
        assert!(reject_sensitive_document_folder(Path::new("/run/secrets")).is_err());
    }

    #[cfg(windows)]
    #[test]
    fn windows_unc_is_rejected_but_mapped_drive_spelling_is_allowed() {
        assert!(reject_network_or_device_path(Path::new(r"\\server\share\documents")).is_err());
        assert!(reject_network_or_device_path(Path::new(r"Z:\documents")).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn document_folder_rejects_symlinks_and_sensitive_directories() {
        use std::os::unix::fs::symlink;

        let target = temp_path("documents-target");
        let link = temp_path("documents-link");
        fs::create_dir(&target).unwrap();
        symlink(&target, &link).unwrap();
        assert!(classify_native_document_folder(&link).is_err());
        assert!(classify_native_document_folder(Path::new("/etc")).is_err());
        let _ = fs::remove_file(link);
        let _ = fs::remove_dir(target);
    }

    #[test]
    fn training_dataset_allows_import_and_reveal_only() {
        let path = temp_path("dataset").with_extension("jsonl");
        fs::write(&path, b"{\"text\":\"hello\"}\n").unwrap();
        let classified = classify_native_dataset_path(&path).unwrap();
        assert_eq!(classified.path_kind, NativePathKind::Dataset);
        assert!(classified
            .allowed_operations
            .contains(&NativePathOperation::DatasetImport));
        assert!(!classified
            .allowed_operations
            .contains(&NativePathOperation::Attach));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn unsupported_training_dataset_type_is_rejected() {
        let path = temp_path("dataset").with_extension("exe");
        fs::write(&path, b"MZ").unwrap();
        assert!(classify_native_dataset_path(&path).is_err());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn sensitive_artifact_names_require_an_exact_path_segment() {
        for path in [
            Path::new("/outputs/pid/metrics.json"),
            Path::new("/outputs/auth/credentials.json"),
            Path::new("/outputs/studio.db"),
            Path::new("/outputs/auth.db.backup"),
        ] {
            assert!(reject_sensitive_artifact(path).is_err());
        }
        for path in [
            Path::new("/outputs/pid_sweep_3/metrics.json"),
            Path::new("/outputs/auth_results/metrics.json"),
            Path::new("/outputs/studio_database/metrics.json"),
        ] {
            assert!(reject_sensitive_artifact(path).is_ok());
        }
    }
}
