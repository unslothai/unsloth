// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! Print and screenshot for the browser panel. Native views use their own engine; everything else
//! is captured from the app's webview, cropped to the page area.

use crate::browser_webview::{require_main, view, ViewBounds};
use std::sync::Mutex;
use std::time::Duration;
use tauri::ipc::Response;
use tauri::{Manager, Runtime, Webview};

const CAPTURE_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_PNG_BYTES: usize = 64 * 1024 * 1024;

/// A rect in the calling webview's CSS pixels, plus its width in CSS pixels to scale by its zoom.
#[derive(Clone, Copy)]
struct Clip {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    viewport_width: f64,
}

type Done = Box<dyn FnOnce(Result<Vec<u8>, String>) + Send>;

#[tauri::command]
pub async fn browser_capture<R: Runtime>(
    webview: Webview<R>,
    tab_id: Option<String>,
    bounds: Option<ViewBounds>,
) -> Result<Response, String> {
    require_main(&webview)?;
    let (target, clip) = match (tab_id, bounds) {
        (Some(tab_id), _) => (view(webview.app_handle(), &tab_id)?, None),
        (None, Some(bounds)) => {
            let (x, y, width, height, viewport_width) = bounds.parts();
            if !(width >= 1.0 && height >= 1.0 && viewport_width >= 1.0) {
                return Err("nothing to capture".into());
            }
            let clip = Clip {
                x: x.max(0.0),
                y: y.max(0.0),
                width,
                height,
                viewport_width,
            };
            (webview, Some(clip))
        }
        (None, None) => return Err("nothing to capture".into()),
    };
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let sender = Mutex::new(Some(sender));
    let done: Done = Box::new(move |result| {
        if let Some(sender) = sender.lock().unwrap().take() {
            let _ = sender.send(result);
        }
    });
    target
        .with_webview(move |platform| platform_capture(platform, clip, done))
        .map_err(|error| error.to_string())?;
    let png = match tokio::time::timeout(CAPTURE_TIMEOUT, receiver).await {
        Ok(Ok(result)) => result?,
        _ => return Err("The screenshot didn't finish".into()),
    };
    if png.is_empty() || png.len() > MAX_PNG_BYTES {
        return Err("The screenshot came back empty".into());
    }
    Ok(Response::new(png))
}

/// Async: `with_webview` runs on the main thread, which a sync command would be holding.
#[tauri::command]
pub async fn browser_view_print<R: Runtime>(
    webview: Webview<R>,
    tab_id: String,
) -> Result<(), String> {
    require_main(&webview)?;
    let page = view(webview.app_handle(), &tab_id)?;
    let (sender, receiver) = tokio::sync::oneshot::channel();
    page.with_webview(move |platform| {
        let _ = sender.send(platform_print(platform));
    })
    .map_err(|error| error.to_string())?;
    // Only until the dialog is up.
    match tokio::time::timeout(Duration::from_secs(5), receiver).await {
        Ok(Ok(result)) => result,
        _ => Err("The print dialog didn't open".into()),
    }
}

#[cfg(target_os = "macos")]
fn platform_capture(platform: tauri::webview::PlatformWebview, clip: Option<Clip>, done: Done) {
    use block2::RcBlock;
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSBitmapImageFileType, NSBitmapImageRep, NSImage};
    use objc2_foundation::{NSDictionary, NSError, NSPoint, NSRect, NSSize};
    use objc2_web_kit::{WKSnapshotConfiguration, WKWebView};

    let Some(mtm) = MainThreadMarker::new() else {
        return done(Err("Not on the main thread".into()));
    };
    // Safety: wry's live view, used on the main thread.
    unsafe {
        let view = &*(platform.inner() as *const WKWebView);
        let config = WKSnapshotConfiguration::new(mtm);
        if let Some(clip) = clip {
            // WKWebView is flipped, so its points read like CSS pixels, scaled by the app's zoom.
            let width = view.frame().size.width;
            let scale = if width > 0.0 { width / clip.viewport_width } else { 1.0 };
            config.setRect(NSRect::new(
                NSPoint::new(clip.x * scale, clip.y * scale),
                NSSize::new(clip.width * scale, clip.height * scale),
            ));
        }
        let done = Mutex::new(Some(done));
        let handler = RcBlock::new(move |image: *mut NSImage, _error: *mut NSError| {
            let Some(done) = done.lock().unwrap().take() else {
                return;
            };
            let png = image
                .as_ref()
                .and_then(|image| image.TIFFRepresentation())
                .and_then(|tiff| NSBitmapImageRep::imageRepWithData(&tiff))
                .and_then(|bitmap| {
                    bitmap.representationUsingType_properties(
                        NSBitmapImageFileType::PNG,
                        &NSDictionary::new(),
                    )
                })
                .map(|data| data.to_vec());
            done(png.ok_or_else(|| "The page couldn't be captured".to_string()));
        });
        view.takeSnapshotWithConfiguration_completionHandler(Some(&config), &handler);
    }
}

#[cfg(windows)]
fn platform_capture(platform: tauri::webview::PlatformWebview, clip: Option<Clip>, done: Done) {
    use base64::Engine;
    use webview2_com::CallDevToolsProtocolMethodCompletedHandler;
    use windows_core::HSTRING;

    let mut params = serde_json::json!({ "format": "png", "fromSurface": true });
    if let Some(clip) = clip {
        params["clip"] = serde_json::json!({
            "x": clip.x,
            "y": clip.y,
            "width": clip.width,
            "height": clip.height,
            "scale": 1,
        });
    }
    let done = std::sync::Arc::new(Mutex::new(Some(done)));
    let finish = {
        let done = done.clone();
        move |result: Result<Vec<u8>, String>| {
            if let Some(done) = done.lock().unwrap().take() {
                done(result);
            }
        }
    };
    let answer = finish.clone();
    let handler = CallDevToolsProtocolMethodCompletedHandler::create(Box::new(
        move |result, json: String| {
            answer(
                result
                    .map_err(|error| error.to_string())
                    .and_then(|()| {
                        let value: serde_json::Value =
                            serde_json::from_str(&json).map_err(|error| error.to_string())?;
                        let data = value
                            .get("data")
                            .and_then(|data| data.as_str())
                            .ok_or("The page couldn't be captured")?;
                        base64::engine::general_purpose::STANDARD
                            .decode(data)
                            .map_err(|error| error.to_string())
                    }),
            );
            Ok(())
        },
    ));
    // Bound, not inlined: the call reads both strings while it runs.
    let method = HSTRING::from("Page.captureScreenshot");
    let params = HSTRING::from(params.to_string());
    let started = unsafe {
        platform
            .controller()
            .CoreWebView2()
            .and_then(|webview| webview.CallDevToolsProtocolMethod(&method, &params, &handler))
    };
    if let Err(error) = started {
        finish(Err(error.to_string()));
    }
}

/// No native views on Linux: crop the app's snapshot.
#[cfg(target_os = "linux")]
fn platform_capture(platform: tauri::webview::PlatformWebview, clip: Option<Clip>, done: Done) {
    use webkit2gtk::{SnapshotOptions, SnapshotRegion, WebViewExt};

    let Some(clip) = clip else {
        return done(Err("No native pages on Linux".into()));
    };
    let view = platform.inner();
    let scale = {
        use gtk::prelude::WidgetExt;
        let width = f64::from(view.allocated_width());
        if width > 0.0 { width / clip.viewport_width } else { 1.0 }
    };
    view.snapshot(
        SnapshotRegion::Visible,
        SnapshotOptions::NONE,
        None::<&webkit2gtk::gio::Cancellable>,
        move |result| {
            done(
                result
                    .map_err(|error| error.to_string())
                    .and_then(|surface| crop_png(&surface, clip, scale)),
            )
        },
    );
}

#[cfg(target_os = "linux")]
fn crop_png(source: &gtk::cairo::Surface, clip: Clip, scale: f64) -> Result<Vec<u8>, String> {
    use gtk::cairo::{Context, Format, ImageSurface};

    let (device_x, _) = source.device_scale();
    let device = if device_x > 0.0 { device_x } else { 1.0 };
    let (x, y) = (clip.x * scale, clip.y * scale);
    let (width, height) = (clip.width * scale, clip.height * scale);
    let out = ImageSurface::create(
        Format::ARgb32,
        (width * device).round().max(1.0) as i32,
        (height * device).round().max(1.0) as i32,
    )
    .map_err(|error| error.to_string())?;
    out.set_device_scale(device, device);
    {
        let context = Context::new(&out).map_err(|error| error.to_string())?;
        context
            .set_source_surface(source, -x, -y)
            .map_err(|error| error.to_string())?;
        context.paint().map_err(|error| error.to_string())?;
    }
    let mut png = Vec::new();
    out.write_to_png(&mut png)
        .map_err(|error| error.to_string())?;
    Ok(png)
}

#[cfg(not(any(target_os = "macos", windows, target_os = "linux")))]
fn platform_capture(_platform: tauri::webview::PlatformWebview, _clip: Option<Clip>, done: Done) {
    done(Err("Screenshots aren't supported here".into()));
}

/// Not wry's `print()`, which zeroes margins on the app's shared print info.
#[cfg(target_os = "macos")]
fn platform_print(platform: tauri::webview::PlatformWebview) -> Result<(), String> {
    use objc2::MainThreadMarker;
    use objc2_app_kit::NSPrintInfo;
    use objc2_foundation::NSCopying;
    use objc2_web_kit::WKWebView;

    if MainThreadMarker::new().is_none() {
        return Err("Not on the main thread".into());
    }
    // Safety: wry's live view, used on the main thread.
    unsafe {
        let view = &*(platform.inner() as *const WKWebView);
        let window = view.window().ok_or("The page isn't in a window")?;
        let operation = view.printOperationWithPrintInfo(&NSPrintInfo::sharedPrintInfo().copy());
        operation.setShowsPrintPanel(true);
        operation.setShowsProgressPanel(true);
        if let Some(title) = view.title().filter(|title| title.length() > 0) {
            operation.setJobTitle(Some(&title));
        }
        operation.runOperationModalForWindow_delegate_didRunSelector_contextInfo(
            &window,
            None,
            None,
            std::ptr::null_mut(),
        );
    }
    Ok(())
}

/// Not `window.print()`, which the page can replace.
#[cfg(windows)]
fn platform_print(platform: tauri::webview::PlatformWebview) -> Result<(), String> {
    use webview2_com::Microsoft::Web::WebView2::Win32::{
        ICoreWebView2_16, COREWEBVIEW2_PRINT_DIALOG_KIND_BROWSER,
    };
    use windows_core::Interface;
    unsafe {
        platform
            .controller()
            .CoreWebView2()
            .and_then(|webview| webview.cast::<ICoreWebView2_16>())
            .and_then(|webview| webview.ShowPrintUI(COREWEBVIEW2_PRINT_DIALOG_KIND_BROWSER))
            .map_err(|error| error.to_string())
    }
}

#[cfg(not(any(target_os = "macos", windows)))]
fn platform_print(_platform: tauri::webview::PlatformWebview) -> Result<(), String> {
    Err("No native pages here".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_bounds_match_the_panel() {
        // The shape capture.ts sends.
        let bounds: ViewBounds = serde_json::from_value(serde_json::json!({
            "x": 640.5, "y": 96, "width": 579, "height": 799, "viewportWidth": 1440
        }))
        .unwrap();
        assert_eq!(bounds.parts(), (640.5, 96.0, 579.0, 799.0, 1440.0));
    }
}
