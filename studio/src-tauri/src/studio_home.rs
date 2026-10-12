// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! Where the desktop keeps its Studio install: `~/.unsloth/studio`, or the folder picked on the
//! Windows first-run screen, recorded in `~/.unsloth/desktop_studio_home` and passed to managed
//! children as UNSLOTH_STUDIO_HOME. No pointer file means the default root.

use std::fs;
use std::path::{Path, PathBuf};

const POINTER_FILE: &str = "desktop_studio_home";

pub(crate) const STUDIO_HOME_ENV: &str = "UNSLOTH_STUDIO_HOME";

/// Marks UNSLOTH_STUDIO_HOME as the desktop's choice; install.ps1 --tauri refuses any other.
pub(crate) const DESKTOP_CHOSEN_HOME_ENV: &str = "UNSLOTH_DESKTOP_CHOSEN_STUDIO_HOME";

pub(crate) fn default_root_for_home(home: &Path) -> PathBuf {
    home.join(".unsloth").join("studio")
}

fn pointer_path(home: &Path) -> PathBuf {
    home.join(".unsloth").join(POINTER_FILE)
}

/// A missing, empty or relative pointer falls back to the default root.
pub(crate) fn chosen_root_for_home(home: &Path) -> Option<PathBuf> {
    let text = fs::read_to_string(pointer_path(home)).ok()?;
    let root = PathBuf::from(text.trim());
    if root.as_os_str().is_empty() || !root.is_absolute() {
        return None;
    }
    if same_path(&root, &default_root_for_home(home)) {
        return None;
    }
    Some(root)
}

pub(crate) fn studio_root_for_home(home: &Path) -> PathBuf {
    chosen_root_for_home(home).unwrap_or_else(|| default_root_for_home(home))
}

pub(crate) fn studio_root() -> Option<PathBuf> {
    dirs::home_dir().map(|home| studio_root_for_home(&home))
}

pub(crate) fn chosen_root() -> Option<PathBuf> {
    dirs::home_dir().and_then(|home| chosen_root_for_home(&home))
}

fn same_path(left: &Path, right: &Path) -> bool {
    let normalize = |path: &Path| {
        let text = path.to_string_lossy().replace('/', "\\");
        let text = text.trim_end_matches('\\').to_string();
        if cfg!(windows) {
            text.to_lowercase()
        } else {
            text
        }
    };
    normalize(left) == normalize(right)
}

/// `<picked>\studio` for a folder already named Unsloth, else `<picked>\Unsloth\studio`.
pub(crate) fn root_for_picked_folder(picked: &Path) -> PathBuf {
    let named_unsloth = picked
        .file_name()
        .map(|name| {
            let name = name.to_string_lossy();
            name.eq_ignore_ascii_case("unsloth") || name.eq_ignore_ascii_case(".unsloth")
        })
        .unwrap_or(false);
    if named_unsloth {
        picked.join("studio")
    } else {
        picked.join("Unsloth").join("studio")
    }
}

/// Network shares are refused: the venv, llama.cpp and memory-mapped models live there.
pub(crate) fn validate_picked_folder(picked: &Path) -> Result<(), String> {
    if !picked.is_absolute() {
        return Err("Choose a folder on a local drive.".to_string());
    }
    let text = picked.to_string_lossy();
    if text.starts_with("\\\\") && !text.starts_with("\\\\?\\") || text.starts_with("\\\\?\\UNC\\")
    {
        return Err(
            "Network folders are not supported. Choose a folder on a local drive.".to_string(),
        );
    }
    // create_new: never truncates or deletes an existing entry (or link) at the probe name.
    let probe = picked.join(format!(".unsloth-write-probe-{}", std::process::id()));
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
        .map_err(|_| "Unsloth cannot write to this folder. Choose another folder.".to_string())?;
    let _ = fs::remove_file(&probe);
    Ok(())
}

/// Written to a temporary file and renamed; the default root removes the pointer instead.
pub(crate) fn write_choice_for_home(home: &Path, root: &Path) -> Result<(), String> {
    let pointer = pointer_path(home);
    if same_path(root, &default_root_for_home(home)) {
        return match fs::remove_file(&pointer) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(format!("Could not reset the install location: {error}")),
        };
    }
    let parent = pointer
        .parent()
        .ok_or("Could not resolve the Unsloth folder.")?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("Could not save the install location: {error}"))?;
    let temp = parent.join(format!("{POINTER_FILE}.tmp-{}", std::process::id()));
    fs::write(&temp, root.to_string_lossy().as_bytes())
        .and_then(|()| fs::rename(&temp, &pointer))
        .map_err(|error| {
            let _ = fs::remove_file(&temp);
            format!("Could not save the install location: {error}")
        })
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct StudioInstallLocation {
    pub path: String,
    pub is_default: bool,
    pub can_change: bool,
}

fn location_for_home(home: &Path) -> StudioInstallLocation {
    let chosen = chosen_root_for_home(home);
    StudioInstallLocation {
        is_default: chosen.is_none(),
        path: chosen
            .unwrap_or_else(|| default_root_for_home(home))
            .to_string_lossy()
            .into_owned(),
        can_change: cfg!(windows),
    }
}

fn home_dir() -> Result<PathBuf, String> {
    dirs::home_dir().ok_or_else(|| "Could not determine home directory".to_string())
}

fn ensure_changeable(install: &crate::install::InstallState) -> Result<(), String> {
    if !cfg!(windows) {
        return Err("Choosing an install location is only available on Windows.".to_string());
    }
    if crate::install::is_install_running(install) {
        return Err("Installation is already running.".to_string());
    }
    if crate::process::find_unsloth_binary().is_some() {
        return Err("Unsloth is already installed in this location.".to_string());
    }
    Ok(())
}

/// Under the install lock: an install that already has the old root never sees it move.
fn commit_choice(
    install: &crate::install::InstallState,
    home: &Path,
    root: &Path,
) -> Result<(), String> {
    crate::install::while_install_idle(install, || {
        if crate::process::find_unsloth_binary().is_some() {
            return Err("Unsloth is already installed in this location.".to_string());
        }
        write_choice_for_home(home, root)
    })
}

#[tauri::command]
pub fn studio_install_location() -> Result<StudioInstallLocation, String> {
    Ok(location_for_home(&home_dir()?))
}

/// The webview never names a path: the folder comes from the system dialog.
#[tauri::command]
pub async fn choose_studio_install_location(
    webview: tauri::Webview,
    app: tauri::AppHandle,
    install: tauri::State<'_, crate::install::InstallState>,
) -> Result<Option<StudioInstallLocation>, String> {
    use tauri_plugin_dialog::DialogExt;

    crate::native_intents::ensure_main_window(&webview)?;
    ensure_changeable(install.inner())?;
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog()
        .file()
        .set_title("Choose where to install Unsloth")
        .pick_folder(move |path| {
            let _ = tx.send(path);
        });
    let Some(picked) = rx.await.map_err(|_| "Dialog closed".to_string())? else {
        return Ok(None);
    };
    let picked = picked
        .into_path()
        .map_err(|_| "Choose a folder on a local drive.".to_string())?;
    validate_picked_folder(&picked)?;
    let home = home_dir()?;
    commit_choice(install.inner(), &home, &root_for_picked_folder(&picked))?;
    Ok(Some(location_for_home(&home)))
}

#[tauri::command]
pub fn reset_studio_install_location(
    webview: tauri::Webview,
    install: tauri::State<'_, crate::install::InstallState>,
) -> Result<StudioInstallLocation, String> {
    crate::native_intents::ensure_main_window(&webview)?;
    ensure_changeable(install.inner())?;
    let home = home_dir()?;
    commit_choice(install.inner(), &home, &default_root_for_home(&home))?;
    Ok(location_for_home(&home))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_home(name: &str) -> PathBuf {
        let home = std::env::temp_dir().join(format!(
            "unsloth-studio-home-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&home).unwrap();
        home
    }

    #[test]
    fn no_pointer_keeps_the_default_root() {
        let home = temp_home("default");
        assert_eq!(
            studio_root_for_home(&home),
            home.join(".unsloth").join("studio")
        );
        assert_eq!(chosen_root_for_home(&home), None);
        let location = location_for_home(&home);
        assert!(location.is_default);
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn a_written_choice_is_read_back_and_reset_removes_it() {
        let home = temp_home("choice");
        let custom = home.join("other-drive").join("Unsloth").join("studio");
        write_choice_for_home(&home, &custom).unwrap();
        assert_eq!(studio_root_for_home(&home), custom);
        assert_eq!(chosen_root_for_home(&home), Some(custom.clone()));
        write_choice_for_home(&home, &custom).unwrap();
        assert_eq!(chosen_root_for_home(&home), Some(custom));

        write_choice_for_home(&home, &default_root_for_home(&home)).unwrap();
        assert!(!pointer_path(&home).exists());
        assert_eq!(chosen_root_for_home(&home), None);
        write_choice_for_home(&home, &default_root_for_home(&home)).unwrap();
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn damaged_pointers_fall_back_to_the_default_root() {
        let home = temp_home("damaged");
        fs::create_dir_all(home.join(".unsloth")).unwrap();
        for text in ["", "   \n", "relative\\studio"] {
            fs::write(pointer_path(&home), text).unwrap();
            assert_eq!(chosen_root_for_home(&home), None, "{text:?}");
            assert_eq!(studio_root_for_home(&home), default_root_for_home(&home));
        }
        let default_text = default_root_for_home(&home).to_string_lossy().into_owned();
        fs::write(pointer_path(&home), format!("{default_text}\n")).unwrap();
        assert_eq!(chosen_root_for_home(&home), None);
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn picked_folders_map_to_a_studio_folder_inside_them() {
        let base = PathBuf::from(if cfg!(windows) {
            "D:\\Apps"
        } else {
            "/mnt/d/Apps"
        });
        assert_eq!(
            root_for_picked_folder(&base),
            base.join("Unsloth").join("studio")
        );
        assert_eq!(
            root_for_picked_folder(&base.join("Unsloth")),
            base.join("Unsloth").join("studio")
        );
        assert_eq!(
            root_for_picked_folder(&base.join("unsloth")),
            base.join("unsloth").join("studio")
        );
    }

    #[test]
    fn unwritable_or_relative_folders_are_refused() {
        assert!(validate_picked_folder(Path::new("relative")).is_err());
        let home = temp_home("writable");
        assert!(validate_picked_folder(&home).is_ok());
        assert!(validate_picked_folder(&home.join("missing")).is_err());
        fs::remove_dir_all(&home).unwrap();
    }

    #[test]
    fn the_write_probe_never_touches_an_existing_file() {
        let home = temp_home("probe");
        let existing = home.join(format!(".unsloth-write-probe-{}", std::process::id()));
        fs::write(&existing, b"keep").unwrap();
        assert!(validate_picked_folder(&home).is_err());
        assert_eq!(fs::read(&existing).unwrap(), b"keep");
        fs::remove_dir_all(&home).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn network_folders_are_refused() {
        assert!(validate_picked_folder(Path::new(r"\\server\share\models")).is_err());
        assert!(validate_picked_folder(Path::new(r"\\?\UNC\server\share")).is_err());
    }
}
