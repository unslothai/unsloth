use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

pub const DEFAULT_HOTKEY: &str = "ctrl+super+u";
pub const DEFAULT_ASK_HOTKEY: &str = "alt+space";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct PillConfig {
    pub enabled: bool,
    pub hotkey: String,
    pub excluded_apps: Vec<String>,
    pub ask_enabled: bool,
    pub ask_hotkey: String,
}

impl Default for PillConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            hotkey: DEFAULT_HOTKEY.to_string(),
            excluded_apps: Vec::new(),
            ask_enabled: true,
            ask_hotkey: DEFAULT_ASK_HOTKEY.to_string(),
        }
    }
}

impl PillConfig {
    pub fn is_app_excluded(&self, bundle_id: &str) -> bool {
        let bundle_id = bundle_id.trim();
        !bundle_id.is_empty()
            && self
                .excluded_apps
                .iter()
                .any(|excluded| excluded.eq_ignore_ascii_case(bundle_id))
    }
}

pub fn config_path(app_config_dir: &Path) -> PathBuf {
    app_config_dir.join("selection-pill.json")
}

pub fn load_config(app_config_dir: &Path) -> PillConfig {
    let path = config_path(app_config_dir);
    match fs::read_to_string(&path) {
        Ok(raw) => serde_json::from_str(&raw).unwrap_or_default(),
        Err(_) => PillConfig::default(),
    }
}

pub fn save_config(app_config_dir: &Path, config: &PillConfig) -> Result<(), String> {
    fs::create_dir_all(app_config_dir)
        .map_err(|e| format!("Failed to create config dir: {e}"))?;
    let raw = serde_json::to_string_pretty(config)
        .map_err(|e| format!("Failed to serialize pill config: {e}"))?;
    // Staged through a sibling temp file and renamed over the destination, the
    // same way the backend owner file is written. fs::write truncates first, so
    // a full disk or an interrupted write leaves a partial file, and
    // load_config reads unparseable JSON as the default: a half-written save
    // would silently discard the persisted configuration on the next launch.
    let tmp = app_config_dir.join(format!(".selection-pill.{}.tmp", std::process::id()));
    fs::write(&tmp, raw).map_err(|e| format!("Failed to write pill config: {e}"))?;
    fs::rename(&tmp, config_path(app_config_dir)).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        format!("Failed to replace pill config: {e}")
    })
}

pub fn save_for_app(app: &tauri::AppHandle, config: &PillConfig) -> Result<(), String> {
    use tauri::Manager;
    let dir = app
        .path()
        .app_config_dir()
        .map_err(|e| format!("No app config dir: {e}"))?;
    save_config(&dir, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_disabled_with_default_hotkey() {
        let config = PillConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.hotkey, DEFAULT_HOTKEY);
        assert!(config.excluded_apps.is_empty());
    }

    #[test]
    fn roundtrip_persists_fields() {
        let dir = std::env::temp_dir().join(format!(
            "pill-config-test-{}",
            std::process::id()
        ));
        let config = PillConfig {
            enabled: true,
            hotkey: "alt+shift+u".to_string(),
            excluded_apps: vec!["com.apple.Passwords".to_string()],
            ask_enabled: false,
            ask_hotkey: "alt+shift+space".to_string(),
        };
        save_config(&dir, &config).unwrap();
        assert_eq!(load_config(&dir), config);
        // The save stages through a temp file; a successful one must leave none
        // behind, or the config dir accumulates a file per process.
        let strays: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name())
            .filter(|name| name.to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(strays.is_empty(), "temp files left behind: {strays:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_or_missing_config_falls_back_to_default() {
        let dir = std::env::temp_dir().join(format!(
            "pill-config-corrupt-{}",
            std::process::id()
        ));
        assert_eq!(load_config(&dir), PillConfig::default());
        fs::create_dir_all(&dir).unwrap();
        fs::write(config_path(&dir), "{not json").unwrap();
        assert_eq!(load_config(&dir), PillConfig::default());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn app_exclusion_is_case_insensitive_and_ignores_empty() {
        let config = PillConfig {
            excluded_apps: vec!["com.apple.Passwords".to_string()],
            ..Default::default()
        };
        assert!(config.is_app_excluded("com.apple.passwords"));
        assert!(!config.is_app_excluded("com.apple.Safari"));
        assert!(!config.is_app_excluded(""));
    }
}
