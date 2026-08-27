// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Stands in for @tauri-apps/api/core, which only resolves inside a Tauri webview.
// State lives on globalThis so it survives the "?bust=N" re-evaluation.
//
//   mode "ok"      -> invoke resolves
//   mode "rejects" -> invoke rejects, as an older WebView2 runtime would

const control = (globalThis.__TAURI_CORE_STUB__ ??= { calls: [], mode: "ok" });

export async function invoke(command, args) {
  control.calls.push({ command, args });
  if (control.mode === "rejects") {
    throw new Error("command reset_microphone_permission not found");
  }
}
