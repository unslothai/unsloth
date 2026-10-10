// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! Open a browser file tab in another app, as ChatGPT's Open menu does. A tab holds its bytes in
//! the webview, and another app can only open a file on disk, so each is copied out first.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use serde::Serialize;
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Manager, Url};

const OPEN_DIR: &str = "open-with";
// Finder's Open With lists a dozen or so; past that the menu stops being a choice.
#[cfg(target_os = "macos")]
const MAX_APPS: usize = 12;

static CLEARED: AtomicBool = AtomicBool::new(false);
static PARTIAL: AtomicU64 = AtomicU64::new(0);

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LocalCopy {
    id: String,
    path: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OpenWithApp {
    path: String,
    name: String,
    /// A PNG data URL of the app's icon.
    icon: Option<String>,
    default: bool,
}

fn root(app: &AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_cache_dir()
        .map(|dir| dir.join(OPEN_DIR))
        .map_err(|error| error.to_string())
}

fn copy_path(app: &AppHandle, id: &str) -> Result<PathBuf, String> {
    // The id names a folder under the root, so nothing else on disk can be reached through it.
    if id.len() != 32
        || !id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("Unknown file.".to_string());
    }
    fs::read_dir(root(app)?.join(id))
        .ok()
        .and_then(|entries| {
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .find(|path| path.is_file())
        })
        .ok_or_else(|| "The file is no longer there.".to_string())
}

/// Write a tab's file to the app cache and return its id and path. The same bytes and name reuse
/// one copy; copies from an earlier run are cleared on the first use after launch.
#[tauri::command]
pub async fn browser_file_local_copy(
    webview: tauri::Webview,
    app: AppHandle,
    request: tauri::ipc::Request<'_>,
) -> Result<LocalCopy, String> {
    crate::native_intents::ensure_main_window(&webview)?;
    let (name, content) = crate::native_file_dialogs::request_file(&request)?;
    let name = crate::native_file_dialogs::safe_download_name(&name);
    let root = root(&app)?;
    if !CLEARED.swap(true, Ordering::SeqCst) {
        let _ = fs::remove_dir_all(&root);
    }
    let mut hasher = Sha256::new();
    hasher.update(name.as_bytes());
    hasher.update([0]);
    hasher.update(content.as_ref());
    let id: String = hasher.finalize()[..16]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let dir = root.join(&id);
    let path = dir.join(&name);
    if !path.is_file() {
        fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
        // Written outside the folder, then moved in, so a half-written copy is never opened.
        let partial = root.join(format!(
            "{id}.{}.partial",
            PARTIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&partial, content.as_ref()).map_err(|error| error.to_string())?;
        fs::rename(&partial, &path).map_err(|error| {
            let _ = fs::remove_file(&partial);
            error.to_string()
        })?;
        // A tab's file may come from a website or a model, so the system checks it as a download.
        if let Ok(blank) = Url::parse("about:blank") {
            let _ = crate::browser_webview::mark_downloaded(&path, &blank);
        }
    }
    Ok(LocalCopy {
        id,
        path: path.to_string_lossy().into_owned(),
    })
}

/// The apps that can open a copy, its default first. Empty where the system can't list them.
#[tauri::command]
pub fn browser_file_apps(
    webview: tauri::Webview,
    app: AppHandle,
    id: String,
) -> Result<Vec<OpenWithApp>, String> {
    crate::native_intents::ensure_main_window(&webview)?;
    let path = copy_path(&app, &id)?;
    if crate::browser_downloads::runs_code(&path) {
        return Ok(Vec::new());
    }
    Ok(platform_apps(&path))
}

/// Open a copy with its default app, or with `with`, one of the apps listed for it.
#[tauri::command]
pub fn browser_file_open(
    webview: tauri::Webview,
    app: AppHandle,
    id: String,
    with: Option<String>,
) -> Result<(), String> {
    crate::native_intents::ensure_main_window(&webview)?;
    let path = copy_path(&app, &id)?;
    // As for downloads: programs and scripts stay a Show in folder away.
    if crate::browser_downloads::runs_code(&path) {
        return Err("Programs and scripts open from their folder.".to_string());
    }
    match with {
        None => {
            tauri_plugin_opener::open_path(&path, None::<&str>).map_err(|error| error.to_string())
        }
        Some(with) => open_with(&path, &with),
    }
}

#[tauri::command]
pub fn browser_file_reveal(
    webview: tauri::Webview,
    app: AppHandle,
    id: String,
) -> Result<(), String> {
    crate::native_intents::ensure_main_window(&webview)?;
    crate::native_intents::reveal_in_file_manager(&copy_path(&app, &id)?)
}

#[cfg(target_os = "macos")]
fn app_paths(path: &Path) -> (Vec<String>, Option<String>) {
    use objc2_app_kit::NSWorkspace;
    use objc2_foundation::{NSString, NSURL};

    let workspace = NSWorkspace::sharedWorkspace();
    let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
    let default = workspace
        .URLForApplicationToOpenURL(&url)
        .and_then(|app| app.path())
        .map(|app| app.to_string());
    let urls = workspace.URLsForApplicationsToOpenURL(&url);
    let mut apps: Vec<String> = Vec::new();
    for index in 0..urls.count() {
        if let Some(app) = urls.objectAtIndex(index).path().map(|app| app.to_string()) {
            if !apps.contains(&app) {
                apps.push(app);
            }
        }
    }
    (apps, default)
}

#[cfg(target_os = "macos")]
fn platform_apps(path: &Path) -> Vec<OpenWithApp> {
    use objc2_app_kit::NSWorkspace;

    let (mut apps, default) = app_paths(path);
    // The default first, as Finder's Open With lists it.
    if let Some(default) = &default {
        if let Some(at) = apps.iter().position(|app| app == default) {
            let app = apps.remove(at);
            apps.insert(0, app);
        }
    }
    apps.truncate(MAX_APPS);
    let workspace = NSWorkspace::sharedWorkspace();
    apps.into_iter()
        .map(|app| {
            let name = Path::new(&app)
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_else(|| app.clone());
            OpenWithApp {
                icon: app_icon(&workspace, &app),
                default: default.as_deref() == Some(app.as_str()),
                name,
                path: app,
            }
        })
        .collect()
}

/// The app's icon drawn at menu size, as a PNG data URL.
#[cfg(target_os = "macos")]
fn app_icon(workspace: &objc2_app_kit::NSWorkspace, app: &str) -> Option<String> {
    use base64::Engine;
    use objc2::{AllocAnyThread, MainThreadMarker};
    use objc2_app_kit::{NSBitmapImageFileType, NSBitmapImageRep, NSCompositingOperation, NSImage};
    use objc2_foundation::{NSDictionary, NSPoint, NSRect, NSSize, NSString};

    // Drawing into an image is AppKit's, so main thread only; a sync command runs there.
    MainThreadMarker::new()?;
    let icon = workspace.iconForFile(&NSString::from_str(app));
    let size = NSSize::new(32.0, 32.0);
    let image = NSImage::initWithSize(NSImage::alloc(), size);
    // An icon holds every size up to 1024 px; drawing one at 32 pt keeps just that (2x on Retina).
    #[allow(deprecated)]
    image.lockFocus();
    icon.drawInRect_fromRect_operation_fraction(
        NSRect::new(NSPoint::new(0.0, 0.0), size),
        NSRect::ZERO,
        NSCompositingOperation::SourceOver,
        1.0,
    );
    #[allow(deprecated)]
    image.unlockFocus();
    let png = image
        .TIFFRepresentation()
        .and_then(|tiff| NSBitmapImageRep::imageRepWithData(&tiff))
        // Safety: an empty properties dictionary, as browser_capture.rs passes.
        .and_then(|bitmap| unsafe {
            bitmap.representationUsingType_properties(
                NSBitmapImageFileType::PNG,
                &NSDictionary::new(),
            )
        })?
        .to_vec();
    Some(format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(png)
    ))
}

#[cfg(target_os = "macos")]
fn open_with(path: &Path, with: &str) -> Result<(), String> {
    // Only an app the system lists for this file, never a path the page made up.
    if !app_paths(path).0.iter().any(|app| app == with) {
        return Err("That app can't open this file.".to_string());
    }
    std::process::Command::new("/usr/bin/open")
        .arg("-a")
        .arg(with)
        .arg(path)
        .spawn()
        .map(|_| ())
        .map_err(|error| format!("Couldn't open the file: {error}"))
}

// Windows and Linux open with the default app only for now.
#[cfg(not(target_os = "macos"))]
fn platform_apps(_path: &Path) -> Vec<OpenWithApp> {
    Vec::new()
}

#[cfg(not(target_os = "macos"))]
fn open_with(_path: &Path, _with: &str) -> Result<(), String> {
    Err("Choosing an app isn't supported here.".to_string())
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn a_png_lists_preview_with_a_default() {
        let dir = std::env::temp_dir().join(format!("open-with-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let png = dir.join("pixel.png");
        // A real 1x1 PNG, so Launch Services types it by content as well as by extension.
        fs::write(
            &png,
            base64::Engine::decode(
                &base64::engine::general_purpose::STANDARD,
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            )
            .unwrap(),
        )
        .unwrap();
        let (apps, default) = app_paths(&png);
        let refused = open_with(&png, "/System/Applications/Calculator.app");
        let _ = fs::remove_dir_all(&dir);
        assert!(
            apps.iter().any(|app| app.ends_with("/Preview.app")),
            "{apps:?}"
        );
        assert!(default.is_some_and(|app| apps.contains(&app)));
        assert!(refused.is_err());
    }
}
