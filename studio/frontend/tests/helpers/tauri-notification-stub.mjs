// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Stands in for @tauri-apps/plugin-notification, which only resolves inside a
// Tauri webview. State lives on globalThis so it survives the re-evaluation
// notification-resolver.mjs forces with its "?bust=N" key.
//
//   mode "ok"             -> permission follows `granted`, notifications send
//   mode "send-fails"     -> sendNotification throws (IPC error)
//   mode "module-missing" -> the module throws on evaluation, as `await import()`
//                            does when the capability is absent from src-tauri

const control = (globalThis.__TAURI_NOTIFICATION_STUB__ ??= {
  sent: [],
  granted: false,
  mode: "ok",
  requests: 0,
});

if (control.mode === "module-missing") {
  throw new Error("Cannot find module '@tauri-apps/plugin-notification'");
}

export async function isPermissionGranted() {
  return control.granted;
}

export async function requestPermission() {
  control.requests += 1;
  return control.granted ? "granted" : "denied";
}

export function sendNotification(payload) {
  if (control.mode === "send-fails") {
    throw new Error("notification: forbidden, capability not granted");
  }
  control.sent.push(payload);
}
