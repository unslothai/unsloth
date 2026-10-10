// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Tauri core stub. State lives on globalThis to survive "?bust=N" re-evaluation.

const control = (globalThis.__TAURI_CORE_STUB__ ??= { calls: [], mode: "ok" });

export async function invoke(command, args) {
  control.calls.push({ command, args });
  if (control.mode === "rejects") {
    throw new Error("command reset_microphone_permission not found");
  }
}
