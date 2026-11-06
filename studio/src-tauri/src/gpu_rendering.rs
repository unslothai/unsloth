// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! The "GPU rendering" desktop setting (Auto / On / Off). It is read before the webview starts,
//! so a change applies on the next launch.
//!
//! Linux also keeps a crash flag: when a GPU launch dies with Wayland Error 71 before the window
//! is ready (WebKit bug 280210), the supervisor writes it and Auto/On take the CPU path until the
//! host changes or the user picks a setting again.

use log::warn;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use tauri::Manager;

const SETTING_FILE: &str = "gpu-rendering-v1";
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const UNSAFE_FLAG_FILE: &str = "gpu-rendering-unsafe-v1";
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const UNSAFE_REASON: &str = "error71";
/// Below this WebKitGTK the GPU path gave no measured speedup and can crash on NVIDIA Wayland
/// (Error 71 under explicit sync until 2.54.2).
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const RECOMMENDED_WEBKIT: (u32, u32, u32) = (2, 54, 2);
/// wry's default WebView2 arguments, which any explicit value replaces wholesale.
#[cfg_attr(not(windows), allow(dead_code))]
const WEBVIEW2_DEFAULT_ARGS: &str =
    "--disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection";
#[cfg_attr(not(windows), allow(dead_code))]
const WEBVIEW2_DISABLE_GPU: &str = "--disable-gpu";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuRendering {
    #[default]
    Auto,
    On,
    Off,
}

impl GpuRendering {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "on" => Some(Self::On),
            "off" => Some(Self::Off),
            _ => None,
        }
    }
}

/// Only Windows and Linux can change it: WKWebView has no public switch for GPU compositing.
pub const CONFIGURABLE: bool = cfg!(any(target_os = "windows", target_os = "linux"));

fn setting_path(config_dir: &Path) -> PathBuf {
    config_dir.join(SETTING_FILE)
}

/// Missing, unreadable or invalid reads as Auto.
pub fn read_setting(config_dir: &Path) -> GpuRendering {
    let path = setting_path(config_dir);
    match fs::read_to_string(&path) {
        Ok(text) => GpuRendering::parse(&text).unwrap_or_else(|| {
            warn!("Ignoring invalid GPU rendering setting {}", path.display());
            GpuRendering::Auto
        }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => GpuRendering::Auto,
        Err(error) => {
            warn!(
                "Could not read GPU rendering setting {}: {error}",
                path.display()
            );
            GpuRendering::Auto
        }
    }
}

/// The setting the next launch applies; macOS has nothing to apply.
pub fn launch_setting(config_dir: Option<&Path>) -> GpuRendering {
    match config_dir {
        Some(dir) if CONFIGURABLE => read_setting(dir),
        _ => GpuRendering::Auto,
    }
}

/// Written to a temp file and renamed, so a crash never leaves half a file.
fn write_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| std::io::Error::other("no parent folder"))?;
    fs::create_dir_all(parent)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(bytes)?;
    temporary
        .persist(path)
        .map(|_| ())
        .map_err(|error| error.error)
}

pub fn write_setting(config_dir: &Path, setting: GpuRendering) -> Result<(), String> {
    let path = setting_path(config_dir);
    write_atomically(&path, format!("{}\n", setting.as_str()).as_bytes()).map_err(|error| {
        format!(
            "Failed to save GPU rendering setting {}: {error}",
            path.display()
        )
    })
}

/// What the crash flag is keyed by: a change to any of these may have fixed the crash.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub struct HostKey {
    pub webkit_version: (u32, u32, u32),
    /// First line of /proc/driver/nvidia/version.
    pub nvidia_driver: Option<String>,
    pub wayland_session: bool,
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
impl HostKey {
    fn render(&self) -> String {
        let (major, minor, micro) = self.webkit_version;
        format!(
            "webkit={major}.{minor}.{micro}\nnvidia={}\nsession={}\nreason={UNSAFE_REASON}\n",
            self.nvidia_driver.as_deref().unwrap_or("none"),
            if self.wayland_session {
                "wayland"
            } else {
                "x11"
            },
        )
    }
}

#[derive(Debug, PartialEq, Eq)]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub enum FlagState {
    Absent,
    Honoured,
    /// Deleted because it no longer describes this host; holds a log line.
    Cleared(String),
}

fn unsafe_flag_path(config_dir: &Path) -> PathBuf {
    config_dir.join(UNSAFE_FLAG_FILE)
}

/// Honours a flag written for this exact host and deletes any other, so it never outlives a
/// WebKit, driver or session change.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub fn check_unsafe_flag(config_dir: &Path, key: &HostKey) -> FlagState {
    let path = unsafe_flag_path(config_dir);
    let Ok(stored) = fs::read_to_string(&path) else {
        return FlagState::Absent;
    };
    let expected = key.render();
    if stored == expected {
        return FlagState::Honoured;
    }
    let _ = fs::remove_file(&path);
    let changed = stored
        .lines()
        .zip(expected.lines())
        .filter(|(old, new)| old != new)
        .map(|(old, new)| format!("{old} -> {new}"))
        .collect::<Vec<_>>();
    let detail = if changed.is_empty() {
        "unreadable".to_string()
    } else {
        changed.join(", ")
    };
    FlagState::Cleared(format!("GPU rendering fallback flag cleared: {detail}"))
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub fn write_unsafe_flag(config_dir: &Path, key: &HostKey) -> std::io::Result<()> {
    write_atomically(&unsafe_flag_path(config_dir), key.render().as_bytes())
}

/// True when a flag was there to remove.
pub fn clear_unsafe_flag(config_dir: &Path) -> bool {
    fs::remove_file(unsafe_flag_path(config_dir)).is_ok()
}

/// Shown next to On: the GPU path did not help on older WebKitGTK and can crash on NVIDIA.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
pub fn old_webkit_warning(
    webkit_version: (u32, u32, u32),
    wayland_session: bool,
) -> Option<String> {
    let (major, minor, micro) = webkit_version;
    (wayland_session && webkit_version < RECOMMENDED_WEBKIT)
        .then(|| format!("{major}.{minor}.{micro}"))
}

/// WebView2 arguments for the main window: None keeps today's (wry's defaults).
#[cfg_attr(not(windows), allow(dead_code))]
pub fn windows_main_browser_args(setting: GpuRendering) -> Option<String> {
    (setting == GpuRendering::Off)
        .then(|| format!("{WEBVIEW2_DEFAULT_ARGS} {WEBVIEW2_DISABLE_GPU}"))
}

static PAGES_DISABLE_GPU: AtomicBool = AtomicBool::new(false);

/// Whether browser-panel pages should also start WebView2 without the GPU (Windows Off).
#[cfg_attr(not(windows), allow(dead_code))]
pub fn pages_disable_gpu() -> bool {
    PAGES_DISABLE_GPU.load(Ordering::SeqCst)
}

#[cfg_attr(not(windows), allow(dead_code))]
pub fn set_pages_disable_gpu(disabled: bool) {
    PAGES_DISABLE_GPU.store(disabled, Ordering::SeqCst);
}

/// What this launch applied, so the UI can tell a pending change from the running one.
pub struct LaunchState {
    pub setting: GpuRendering,
    pub fallback_active: bool,
    pub env_override: Option<&'static str>,
    pub old_webkit_warning: Option<String>,
    pub old_driver_warning: Option<String>,
    flag_cleared: AtomicBool,
}

impl LaunchState {
    pub fn new(
        setting: GpuRendering,
        fallback_active: bool,
        env_override: Option<&'static str>,
        old_webkit_warning: Option<String>,
        old_driver_warning: Option<String>,
    ) -> Self {
        Self {
            setting,
            fallback_active,
            env_override,
            old_webkit_warning,
            old_driver_warning,
            flag_cleared: AtomicBool::new(false),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuRenderingStatus {
    pub setting: GpuRendering,
    pub configurable: bool,
    pub restart_required: bool,
    pub fallback_active: bool,
    pub env_override: Option<String>,
    pub old_webkit_warning: Option<String>,
    pub old_driver_warning: Option<String>,
}

fn status(launch: &LaunchState, stored: GpuRendering) -> GpuRenderingStatus {
    if !CONFIGURABLE {
        return GpuRenderingStatus {
            setting: GpuRendering::On,
            configurable: false,
            restart_required: false,
            fallback_active: false,
            env_override: None,
            old_webkit_warning: None,
            old_driver_warning: None,
        };
    }
    let flag_cleared = launch.flag_cleared.load(Ordering::SeqCst);
    GpuRenderingStatus {
        setting: stored,
        configurable: true,
        restart_required: stored != launch.setting || (launch.fallback_active && flag_cleared),
        fallback_active: launch.fallback_active && !flag_cleared,
        env_override: launch.env_override.map(str::to_string),
        old_webkit_warning: launch.old_webkit_warning.clone(),
        old_driver_warning: launch.old_driver_warning.clone(),
    }
}

fn config_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_config_dir()
        .map_err(|error| format!("Could not determine app configuration directory: {error}"))
}

#[tauri::command]
pub fn get_gpu_rendering(
    app: tauri::AppHandle,
    launch: tauri::State<'_, LaunchState>,
) -> Result<GpuRenderingStatus, String> {
    let stored = if CONFIGURABLE {
        read_setting(&config_dir(&app)?)
    } else {
        GpuRendering::On
    };
    Ok(status(&launch, stored))
}

/// Picking any value, the current one included, is also how the user retries the GPU path.
#[tauri::command]
pub fn set_gpu_rendering(
    app: tauri::AppHandle,
    launch: tauri::State<'_, LaunchState>,
    value: GpuRendering,
) -> Result<GpuRenderingStatus, String> {
    if !CONFIGURABLE {
        return Err("GPU rendering cannot be changed on macOS".to_string());
    }
    let dir = config_dir(&app)?;
    write_setting(&dir, value)?;
    if clear_unsafe_flag(&dir) {
        log::info!(
            "GPU rendering fallback flag cleared: setting set to {}",
            value.as_str()
        );
    }
    if launch.fallback_active {
        launch.flag_cleared.store(true, Ordering::SeqCst);
    }
    log::info!("GPU rendering setting saved: {}", value.as_str());
    Ok(status(&launch, value))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> HostKey {
        HostKey {
            webkit_version: (2, 54, 2),
            nvidia_driver: Some("NVRM version: NVIDIA UNIX Open Kernel Module  615.78.08".into()),
            wayland_session: true,
        }
    }

    #[test]
    fn the_setting_defaults_to_auto_and_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_setting(dir.path()), GpuRendering::Auto);
        for value in [GpuRendering::On, GpuRendering::Off, GpuRendering::Auto] {
            write_setting(dir.path(), value).unwrap();
            assert_eq!(read_setting(dir.path()), value);
        }
        assert_eq!(
            fs::read_to_string(setting_path(dir.path())).unwrap(),
            "auto\n"
        );
    }

    #[test]
    fn an_invalid_setting_reads_as_auto() {
        let dir = tempfile::tempdir().unwrap();
        for text in ["maybe\n", "", "1", "gpu"] {
            fs::write(setting_path(dir.path()), text).unwrap();
            assert_eq!(read_setting(dir.path()), GpuRendering::Auto, "{text:?}");
        }
        fs::write(setting_path(dir.path()), " OFF \n").unwrap();
        assert_eq!(read_setting(dir.path()), GpuRendering::Off);
    }

    #[test]
    fn a_flag_for_this_host_is_honoured_and_kept() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(check_unsafe_flag(dir.path(), &key()), FlagState::Absent);
        write_unsafe_flag(dir.path(), &key()).unwrap();
        assert_eq!(check_unsafe_flag(dir.path(), &key()), FlagState::Honoured);
        assert_eq!(check_unsafe_flag(dir.path(), &key()), FlagState::Honoured);
    }

    #[test]
    fn a_flag_for_another_webkit_driver_or_session_is_deleted() {
        let mut webkit = key();
        webkit.webkit_version = (2, 54, 3);
        let mut driver = key();
        driver.nvidia_driver = Some("NVRM version: NVIDIA UNIX Open Kernel Module  620.10".into());
        let mut session = key();
        session.wayland_session = false;
        for (changed, needle) in [
            (webkit, "webkit=2.54.2 -> webkit=2.54.3"),
            (driver, "620.10"),
            (session, "session=wayland -> session=x11"),
        ] {
            let dir = tempfile::tempdir().unwrap();
            write_unsafe_flag(dir.path(), &key()).unwrap();
            match check_unsafe_flag(dir.path(), &changed) {
                FlagState::Cleared(line) => assert!(line.contains(needle), "{line}"),
                other => panic!("{other:?}"),
            }
            assert!(!unsafe_flag_path(dir.path()).exists());
            assert_eq!(check_unsafe_flag(dir.path(), &key()), FlagState::Absent);
        }
    }

    #[test]
    fn a_garbled_or_partial_flag_is_deleted_not_honoured() {
        let full = key().render();
        let partial = full.lines().take(3).collect::<Vec<_>>().join("\n") + "\n";
        let wrong_reason = full.replace("reason=error71", "reason=whatever");
        for text in ["", "garbage", partial.as_str(), wrong_reason.as_str()] {
            let dir = tempfile::tempdir().unwrap();
            fs::write(unsafe_flag_path(dir.path()), text).unwrap();
            assert!(
                matches!(check_unsafe_flag(dir.path(), &key()), FlagState::Cleared(_)),
                "{text:?}"
            );
            assert!(!unsafe_flag_path(dir.path()).exists());
        }
    }

    #[test]
    fn clearing_reports_whether_a_flag_was_there() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!clear_unsafe_flag(dir.path()));
        write_unsafe_flag(dir.path(), &key()).unwrap();
        assert!(clear_unsafe_flag(dir.path()));
        assert_eq!(check_unsafe_flag(dir.path(), &key()), FlagState::Absent);
    }

    #[test]
    fn the_old_webkit_warning_is_for_wayland_below_2_54_2() {
        assert_eq!(
            old_webkit_warning((2, 50, 4), true).as_deref(),
            Some("2.50.4")
        );
        assert_eq!(
            old_webkit_warning((2, 53, 99), true).as_deref(),
            Some("2.53.99")
        );
        // 2.54.0 and 2.54.1 still crash with Error 71 under explicit sync.
        assert_eq!(
            old_webkit_warning((2, 54, 1), true).as_deref(),
            Some("2.54.1")
        );
        assert_eq!(old_webkit_warning((2, 54, 2), true), None);
        assert_eq!(old_webkit_warning((2, 50, 4), false), None);
    }

    #[test]
    fn only_off_changes_the_windows_arguments_and_keeps_wrys_defaults() {
        assert_eq!(windows_main_browser_args(GpuRendering::Auto), None);
        assert_eq!(windows_main_browser_args(GpuRendering::On), None);
        let off = windows_main_browser_args(GpuRendering::Off).unwrap();
        assert!(off.contains("--disable-gpu"), "{off}");
        assert!(off.starts_with(WEBVIEW2_DEFAULT_ARGS), "{off}");
    }

    #[cfg(any(target_os = "windows", target_os = "linux"))]
    #[test]
    fn status_reports_a_pending_change_and_a_retried_fallback() {
        let launch = LaunchState::new(GpuRendering::Auto, true, None, Some("2.50.4".into()), None);
        let before = status(&launch, GpuRendering::Auto);
        assert!(before.fallback_active && !before.restart_required);
        // "Try GPU again" re-selects Auto and clears the flag.
        launch.flag_cleared.store(true, Ordering::SeqCst);
        let after = status(&launch, GpuRendering::Auto);
        assert!(!after.fallback_active && after.restart_required);
        let quiet = LaunchState::new(
            GpuRendering::Auto,
            false,
            Some("WEBKIT_DISABLE_DMABUF_RENDERER"),
            None,
            Some("580".into()),
        );
        let changed = status(&quiet, GpuRendering::Off);
        assert!(changed.restart_required && changed.configurable);
        assert_eq!(
            changed.env_override.as_deref(),
            Some("WEBKIT_DISABLE_DMABUF_RENDERER")
        );
        assert_eq!(
            serde_json::to_value(&changed).unwrap()["oldWebkitWarning"],
            serde_json::Value::Null
        );
        assert_eq!(
            serde_json::to_value(&changed).unwrap()["oldDriverWarning"],
            serde_json::Value::from("580")
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_is_fixed_on_and_ignores_a_stored_value() {
        let dir = tempfile::tempdir().unwrap();
        write_setting(dir.path(), GpuRendering::Off).unwrap();
        assert_eq!(launch_setting(Some(dir.path())), GpuRendering::Auto);
        let launch = LaunchState::new(GpuRendering::Auto, false, None, None, None);
        let reported = status(&launch, GpuRendering::Off);
        assert_eq!(reported.setting, GpuRendering::On);
        assert!(!reported.configurable && !reported.restart_required);
    }
}
