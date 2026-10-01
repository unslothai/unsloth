// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Stands in for @tauri-apps/plugin-clipboard-manager, which only resolves inside a
// Tauri webview. State lives on globalThis, not in module scope, so it survives the
// re-evaluation clipboard-resolver.mjs forces with its "?bust=N" key.
//
//   mode "ok"             -> writeText resolves (native copy succeeded)
//   mode "write-fails"    -> writeText rejects (capability missing / IPC error)
//   mode "module-missing" -> the module itself throws on evaluation, which is what
//                            `await import(...)` does on an install without the plugin

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
