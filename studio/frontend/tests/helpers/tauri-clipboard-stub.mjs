// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Tauri clipboard stub. State lives on globalThis to survive "?bust=N" re-evaluation.

const control = (globalThis.__TAURI_CLIPBOARD_STUB__ ??= { calls: [], mode: "ok" });

if (control.mode === "module-missing") {
  throw new Error("Cannot find module '@tauri-apps/plugin-clipboard-manager'");
}

export async function writeText(text) {
  control.calls.push(text);
  if (control.mode === "write-fails") {
    throw new Error("clipboard-manager: forbidden, capability not granted");
  }
}
