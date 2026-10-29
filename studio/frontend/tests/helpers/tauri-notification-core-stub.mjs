// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Stands in for @tauri-apps/api/core beside tauri-notification-stub.mjs, sharing its
// control object. `backend` is what the plugin's desktop backend answers to
// is_permission_granted; left unset, invoke throws, as it does outside a Tauri webview.
const control = (globalThis.__TAURI_NOTIFICATION_STUB__ ??= {
  sent: [],
  granted: false,
  mode: "ok",
  requests: 0,
});

export async function invoke(command) {
  if (command === "plugin:notification|is_permission_granted" && control.backend !== undefined) {
    return control.backend;
  }
  throw new Error(`invoke ${command}: no Tauri backend`);
}
