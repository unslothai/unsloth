// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

//! macOS system window material; the page clears its own backgrounds only after this returns true.

use tauri::{Runtime, Webview};

/// True when the material is on (or off, when `enabled` is false); false where there is none.
#[tauri::command]
pub fn set_native_vibrancy<R: Runtime>(webview: Webview<R>, enabled: bool) -> Result<bool, String> {
    #[cfg(target_os = "macos")]
    {
        use tauri::window::{Effect, EffectState, EffectsBuilder};
        let window = webview.window();
        if !enabled {
            window
                .set_effects(None)
                .map_err(|error| error.to_string())?;
            return Ok(true);
        }
        // Left transparent when off: the page paints its own fill and the browser layer needs it transparent.
        webview
            .with_webview(|platform| {
                use objc2_app_kit::NSView;
                use objc2_foundation::{ns_string, NSNumber, NSObjectNSKeyValueCoding};
                // Safety: wry's live WKWebView, on the main thread. Private KVC key, as wry's `transparent` option uses.
                unsafe {
                    let view = &*(platform.inner() as *const NSView);
                    view.setValue_forKey(
                        Some(&NSNumber::numberWithBool(false)),
                        ns_string!("drawsBackground"),
                    );
                }
            })
            .map_err(|error| error.to_string())?;
        window
            .set_effects(
                EffectsBuilder::new()
                    .effect(Effect::Sidebar)
                    .state(EffectState::FollowsWindowActiveState)
                    .build(),
            )
            .map_err(|error| error.to_string())?;
        Ok(true)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (webview, enabled);
        Ok(false)
    }
}
