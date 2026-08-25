// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function getCurrentWebview() {
  return {
    setZoom: (scaleFactor) => {
      const control = globalThis.__TAURI_WEBVIEW_STUB__;
      if (control.setZoom) {
        return control.setZoom(scaleFactor);
      }
      control.zooms.push(scaleFactor);
      return Promise.resolve();
    },
  };
}
