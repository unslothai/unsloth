use std::fs;
use std::path::{Path, PathBuf};
use tauri::{Manager, WebviewWindow};

use tauri_plugin_window_state::AppHandleExt;

const APP_LAYOUT_MARKER_FILE: &str = "app-layout-initialized-v1";
const INITIALIZED_MARKER: &[u8] = b"initialized\n";
const RESET_MARKER: &[u8] = b"reset\n";

const SETUP_WINDOW_WIDTH: f64 = 760.0;
const SETUP_WINDOW_HEIGHT: f64 = 560.0;
const MIN_REASONABLE_WINDOW_WIDTH: u64 = 320;
const MIN_REASONABLE_WINDOW_HEIGHT: u64 = 240;

const SETUP_SIZE_TOLERANCE_PX: f64 = 2.0;

#[derive(serde::Deserialize)]
struct PersistedWindowState {
    width: u32,
    height: u32,
    #[serde(rename = "x")]
    _x: i32,
    #[serde(rename = "y")]
    _y: i32,
    #[serde(rename = "prev_x")]
    _prev_x: i32,
    #[serde(rename = "prev_y")]
    _prev_y: i32,
    #[serde(rename = "maximized")]
    _maximized: bool,
    #[serde(rename = "visible")]
    _visible: bool,
    #[serde(rename = "decorated")]
    _decorated: bool,
    #[serde(rename = "fullscreen")]
    _fullscreen: bool,
}

fn marker_path(config_dir: &Path) -> PathBuf {
    config_dir.join(APP_LAYOUT_MARKER_FILE)
}

fn marker_matches(config_dir: &Path, expected: &[u8]) -> bool {
    fs::read(marker_path(config_dir))
        .map(|contents| contents == expected)
        .unwrap_or(false)
}

fn is_initialized(config_dir: &Path) -> bool {
    marker_matches(config_dir, INITIALIZED_MARKER)
}

fn is_reset_pending(config_dir: &Path) -> bool {
    marker_matches(config_dir, RESET_MARKER)
}

fn write_marker(config_dir: &Path, contents: &[u8]) -> Result<(), String> {
    fs::create_dir_all(config_dir).map_err(|error| {
        format!(
            "Failed to create app configuration directory {}: {error}",
            config_dir.display()
        )
    })?;
    let path = marker_path(config_dir);
    fs::write(&path, contents).map_err(|error| {
        format!(
            "Failed to write app layout marker {}: {error}",
            path.display()
        )
    })
}

fn reset_initialized(config_dir: &Path) -> Result<(), String> {
    write_marker(config_dir, RESET_MARKER)
}

fn is_setup_window_size(width: u64, height: u64) -> bool {
    // Window-state persists physical dimensions but not their source scale.
    let width_scale = width as f64 / SETUP_WINDOW_WIDTH;
    let height_scale = height as f64 / SETUP_WINDOW_HEIGHT;
    let scale_tolerance = SETUP_SIZE_TOLERANCE_PX / SETUP_WINDOW_WIDTH
        + SETUP_SIZE_TOLERANCE_PX / SETUP_WINDOW_HEIGHT;
    (width_scale - height_scale).abs() <= scale_tolerance
}

fn saved_main_window_size(config_dir: &Path, state_file_name: &str) -> Option<(u64, u64)> {
    let contents = fs::read(config_dir.join(state_file_name)).ok()?;
    let states = serde_json::from_slice::<std::collections::HashMap<String, PersistedWindowState>>(
        &contents,
    )
    .ok()?;
    let main = states.get("main")?;
    let width = u64::from(main.width);
    let height = u64::from(main.height);
    if width < MIN_REASONABLE_WINDOW_WIDTH || height < MIN_REASONABLE_WINDOW_HEIGHT {
        return None;
    }
    Some((width, height))
}

fn should_restore_saved_layout(config_dir: &Path, state_file_name: &str) -> bool {
    if is_reset_pending(config_dir) {
        return false;
    }
    let Some((width, height)) = saved_main_window_size(config_dir, state_file_name) else {
        return false;
    };
    is_initialized(config_dir) || !is_setup_window_size(width, height)
}

fn mark_initialized(config_dir: &Path) -> Result<(), String> {
    write_marker(config_dir, INITIALIZED_MARKER)
}

fn app_config_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_config_dir()
        .map_err(|error| format!("Could not determine app configuration directory: {error}"))
}

/// Returns whether a full-app layout has previously completed. Legacy state is
/// migrated unless it matches the fixed setup-window size at any display scale.
#[tauri::command]
pub fn has_initialized_app_window_layout(
    window: WebviewWindow,
    app: tauri::AppHandle,
) -> Result<bool, String> {
    crate::native_intents::ensure_main_window(&window)?;
    let config_dir = app_config_dir(&app)?;
    Ok(should_restore_saved_layout(&config_dir, &app.filename()))
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

/// Force the next full-app transition to use a monitor-safe layout. Setup can
/// overwrite the plugin's saved full-app dimensions before the process exits.
#[tauri::command]
pub fn reset_app_window_layout_initialized(
    window: WebviewWindow,
    app: tauri::AppHandle,
) -> Result<(), String> {
    crate::native_intents::ensure_main_window(&window)?;
    reset_initialized(&app_config_dir(&app)?)
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

    fn window_state(width: u32, height: u32, maximized: bool) -> String {
        serde_json::json!({
            "main": {
                "width": width,
                "height": height,
                "x": 0,
                "y": 0,
                "prev_x": 0,
                "prev_y": 0,
                "maximized": maximized,
                "visible": true,
                "decorated": true,
                "fullscreen": false
            }
        })
        .to_string()
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
    fn marker_write_and_reset_are_idempotent() {
        let dir = temp_dir("idempotent");
        mark_initialized(&dir).unwrap();
        mark_initialized(&dir).unwrap();
        assert!(is_initialized(&dir));
        assert!(!is_reset_pending(&dir));

        reset_initialized(&dir).unwrap();
        reset_initialized(&dir).unwrap();
        assert!(!is_initialized(&dir));
        assert!(is_reset_pending(&dir));

        mark_initialized(&dir).unwrap();
        assert!(is_initialized(&dir));
        assert!(!is_reset_pending(&dir));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn legacy_full_app_state_is_migrated_but_setup_state_at_any_scale_is_not() {
        let dir = temp_dir("legacy-plugin-state");
        fs::create_dir_all(&dir).unwrap();
        let state_file = ".window-state.json";
        let state_path = dir.join(state_file);

        fs::write(&state_path, window_state(1200, 800, false)).unwrap();
        assert!(should_restore_saved_layout(&dir, state_file));

        fs::write(&state_path, window_state(760, 560, false)).unwrap();
        assert!(!should_restore_saved_layout(&dir, state_file));

        fs::write(&state_path, window_state(950, 700, false)).unwrap();
        assert!(!should_restore_saved_layout(&dir, state_file));

        fs::write(&state_path, window_state(1520, 1120, false)).unwrap();
        assert!(!should_restore_saved_layout(&dir, state_file));

        fs::write(&state_path, window_state(1520, 1120, true)).unwrap();
        assert!(!should_restore_saved_layout(&dir, state_file));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn explicit_reset_blocks_legacy_migration_until_app_layout_completes() {
        let dir = temp_dir("reset-migration");
        fs::create_dir_all(&dir).unwrap();
        let state_file = ".window-state.json";
        fs::write(dir.join(state_file), window_state(1200, 800, false)).unwrap();

        mark_initialized(&dir).unwrap();
        assert!(should_restore_saved_layout(&dir, state_file));
        reset_initialized(&dir).unwrap();
        assert!(!should_restore_saved_layout(&dir, state_file));
        mark_initialized(&dir).unwrap();
        assert!(should_restore_saved_layout(&dir, state_file));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn corrupt_plugin_state_does_not_restore_even_with_initialized_marker() {
        let dir = temp_dir("plugin-state");
        fs::create_dir_all(&dir).unwrap();
        let state_file = ".window-state.json";
        fs::write(
            dir.join(state_file),
            r#"{"main":{"width":1200,"height":800}}"#,
        )
        .unwrap();
        mark_initialized(&dir).unwrap();
        assert!(is_initialized(&dir));
        assert!(!should_restore_saved_layout(&dir, state_file));
        let _ = fs::remove_dir_all(dir);
    }
}
