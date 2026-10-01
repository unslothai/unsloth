// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

use crate::browser_webview::BrowserRect;
use tauri::webview::PlatformWebview;

#[derive(Default)]
pub struct NativePageState {
    pub can_go_back: bool,
    pub can_go_forward: bool,
    pub actual_bounds: Option<BrowserRect>,
}

#[cfg(target_os = "linux")]
pub fn read(platform: &PlatformWebview) -> Result<NativePageState, String> {
    use webkit2gtk::WebViewExt;
    let page = platform.inner();
    Ok(NativePageState {
        can_go_back: page.can_go_back(),
        can_go_forward: page.can_go_forward(),
        actual_bounds: Some(crate::browser_linux::allocated_bounds(&page)),
    })
}
#[cfg(target_os = "linux")]
pub fn action(platform: &PlatformWebview, action: &str) -> Result<(), String> {
    use webkit2gtk::WebViewExt;
    let page = platform.inner();
    match action {
        "back" => page.go_back(),
        "forward" => page.go_forward(),
        "stop" => page.stop_loading(),
        _ => return Err("Unknown native history action".into()),
    }
    Ok(())
}

#[cfg(windows)]
pub fn read(platform: &PlatformWebview) -> Result<NativePageState, String> {
    unsafe {
        let core = platform
            .controller()
            .CoreWebView2()
            .map_err(|e| e.to_string())?;
        let mut back = windows_core::BOOL::default();
        let mut forward = windows_core::BOOL::default();
        core.CanGoBack(&mut back).map_err(|e| e.to_string())?;
        core.CanGoForward(&mut forward).map_err(|e| e.to_string())?;
        Ok(NativePageState {
            can_go_back: back.as_bool(),
            can_go_forward: forward.as_bool(),
            actual_bounds: None,
        })
    }
}
#[cfg(windows)]
pub fn action(platform: &PlatformWebview, action: &str) -> Result<(), String> {
    unsafe {
        let core = platform
            .controller()
            .CoreWebView2()
            .map_err(|e| e.to_string())?;
        match action {
            "back" => core.GoBack(),
            "forward" => core.GoForward(),
            "stop" => core.Stop(),
            _ => return Err("Unknown native history action".into()),
        }
        .map_err(|e| e.to_string())
    }
}

#[cfg(target_os = "macos")]
pub fn read(platform: &PlatformWebview) -> Result<NativePageState, String> {
    use objc2::{msg_send, runtime::AnyObject};
    unsafe {
        let page = platform.inner() as *mut AnyObject;
        let back: bool = msg_send![page, canGoBack];
        let forward: bool = msg_send![page, canGoForward];
        Ok(NativePageState {
            can_go_back: back,
            can_go_forward: forward,
            actual_bounds: None,
        })
    }
}
#[cfg(target_os = "macos")]
pub fn action(platform: &PlatformWebview, action: &str) -> Result<(), String> {
    use objc2::{msg_send, runtime::AnyObject};
    unsafe {
        let page = platform.inner() as *mut AnyObject;
        match action {
            "back" => {
                let _: *mut AnyObject = msg_send![page, goBack];
            }
            "forward" => {
                let _: *mut AnyObject = msg_send![page, goForward];
            }
            "stop" => {
                let _: () = msg_send![page, stopLoading];
            }
            _ => return Err("Unknown native history action".into()),
        }
    }
    Ok(())
}
