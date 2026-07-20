use std::fs;
use std::path::{Path, PathBuf};
use tauri::{Manager, WebviewWindow};

use tauri_plugin_window_state::AppHandleExt;

const APP_LAYOUT_MARKER_FILE: &str = "app-layout-initialized-v1";

const SETUP_WINDOW_WIDTH: f64 = 760.0;
const SETUP_WINDOW_HEIGHT: f64 = 560.0;
const MIN_REASONABLE_WINDOW_WIDTH: u64 = 320;
const MIN_REASONABLE_WINDOW_HEIGHT: u64 = 240;

fn marker_path(config_dir: &Path) -> PathBuf {
    config_dir.join(APP_LAYOUT_MARKER_FILE)
}

fn is_initialized(config_dir: &Path) -> bool {
    marker_path(config_dir).is_file()
}
fn has_legacy_full_app_state(
    config_dir: &Path,
    state_file_name: &str,
    current_scale_factor: f64,
) -> bool {
    let Ok(contents) = fs::read(config_dir.join(state_file_name)) else {
        return false;
    };
    let Ok(states) = serde_json::from_slice::<serde_json::Value>(&contents) else {
        return false;
    };
    let Some(main) = states.get("main") else {
        return false;
    };
    let Some(width) = main.get("width").and_then(serde_json::Value::as_u64) else {
        return false;
    };
    let Some(height) = main.get("height").and_then(serde_json::Value::as_u64) else {
        return false;
    };
    if width < MIN_REASONABLE_WINDOW_WIDTH || height < MIN_REASONABLE_WINDOW_HEIGHT {
        return false;
    }
    if main
        .get("maximized")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        return true;
    }

    let expected_setup_width = SETUP_WINDOW_WIDTH * current_scale_factor;
    let expected_setup_height = SETUP_WINDOW_HEIGHT * current_scale_factor;
    let is_setup_size = (width as f64 - expected_setup_width).abs() <= 2.0
        && (height as f64 - expected_setup_height).abs() <= 2.0;
    !is_setup_size
}

fn mark_initialized(config_dir: &Path) -> Result<(), String> {
    fs::create_dir_all(config_dir).map_err(|error| {
        format!(
            "Failed to create app configuration directory {}: {error}",
            config_dir.display()
        )
    })?;
    let path = marker_path(config_dir);
    fs::write(&path, b"initialized\n").map_err(|error| {
        format!(
            "Failed to write app layout marker {}: {error}",
            path.display()
        )
    })
}

fn app_config_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_config_dir()
        .map_err(|error| format!("Could not determine app configuration directory: {error}"))
}

/// Returns whether a full-app layout has previously completed. Legacy state is
/// migrated unless it matches the fixed setup-window size at the current scale;
/// maximized state always came from full-app mode.
#[tauri::command]
pub fn has_initialized_app_window_layout(
    window: WebviewWindow,
    app: tauri::AppHandle,
) -> Result<bool, String> {
    crate::native_intents::ensure_main_window(&window)?;
    let config_dir = app_config_dir(&app)?;
    let scale_factor = window.scale_factor().unwrap_or(1.0);
    Ok(is_initialized(&config_dir)
        || has_legacy_full_app_state(&config_dir, &app.filename(), scale_factor))
}

/// Persist only after the caller has successfully sized/centered or restored,
/// shown, constrained, and minimum-sized the full application window.
#[tauri::command]
pub fn mark_app_window_layout_initialized(
    window: WebviewWindow,
    app: tauri::AppHandle,
) -> Result<(), String> {
    crate::native_intents::ensure_main_window(&window)?;
    mark_initialized(&app_config_dir(&app)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "unsloth-app-layout-{name}-{}-{nanos}",
            std::process::id()
        ))
    }

    #[test]
    fn marker_is_absent_until_full_layout_is_marked() {
        let dir = temp_dir("transition");
        assert!(!is_initialized(&dir));
        mark_initialized(&dir).unwrap();
        assert!(is_initialized(&dir));
        assert_eq!(fs::read(marker_path(&dir)).unwrap(), b"initialized\n");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn marker_write_is_idempotent() {
        let dir = temp_dir("idempotent");
        mark_initialized(&dir).unwrap();
        mark_initialized(&dir).unwrap();
        assert!(is_initialized(&dir));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn legacy_full_app_state_is_migrated_but_setup_state_is_not() {
        let dir = temp_dir("legacy-plugin-state");
        fs::create_dir_all(&dir).unwrap();
        let state_file = ".window-state.json";
        let state_path = dir.join(state_file);

        fs::write(&state_path, r#"{"main":{"width":1200,"height":800}}"#).unwrap();
        assert!(has_legacy_full_app_state(&dir, state_file, 1.0));

        fs::write(&state_path, r#"{"main":{"width":760,"height":560}}"#).unwrap();
        assert!(!has_legacy_full_app_state(&dir, state_file, 1.0));

        fs::write(&state_path, r#"{"main":{"width":1520,"height":1120}}"#).unwrap();
        assert!(!has_legacy_full_app_state(&dir, state_file, 2.0));

        fs::write(
            &state_path,
            r#"{"main":{"width":1520,"height":1120,"maximized":true}}"#,
        )
        .unwrap();
        assert!(has_legacy_full_app_state(&dir, state_file, 2.0));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn corrupt_plugin_state_does_not_count_as_app_layout_marker() {
        let dir = temp_dir("plugin-state");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(".window-state.json"), b"{}").unwrap();
        assert!(!is_initialized(&dir));
        assert!(!has_legacy_full_app_state(&dir, ".window-state.json", 1.0));
        let _ = fs::remove_dir_all(dir);
    }
}
