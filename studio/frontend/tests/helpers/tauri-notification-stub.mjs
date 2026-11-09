// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Tauri notification stub. State lives on globalThis to survive "?bust=N" re-evaluation.

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
