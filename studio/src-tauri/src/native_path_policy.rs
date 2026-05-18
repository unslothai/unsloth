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
    let canonical_symlink_metadata = fs::symlink_metadata(&canonical_path)
        .map_err(|e| format!("Path is not available: {e}"))?;
    if canonical_symlink_metadata.file_type().is_symlink() {
        return Err("Symlink paths are not supported for native intake.".to_string());
    }
    let (path_type, size_bytes, modified_ms) = refresh_path_fingerprint(&canonical_path)?;
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
    let lowered = path.to_string_lossy().to_ascii_lowercase();
    for needle in [
        "/auth/",
        "\\auth\\",
        "/auth.db",
        "\\auth.db",
        "/studio.db",
        "\\studio.db",
        "/pid",
        "\\pid",
    ] {
        if lowered.contains(needle) {
            return Err("Sensitive Studio state cannot be registered as an artifact.".to_string());
        }
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
}
