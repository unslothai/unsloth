// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Stands in for @tauri-apps/api/window, which only resolves inside a Tauri
// webview. Control lives on globalThis so a test can hold the drag-drop install
// unresolved and inspect the window before it completes.
const control = (globalThis.__TAURI_WINDOW_STUB__ ??= {});

export function getCurrentWindow() {
  return {
    onDragDropEvent: (handler) =>
      new Promise((resolve) => {
        control.deliver = handler;
        control.installed = () => resolve(() => undefined);
      }),
    onScaleChanged: async () => () => undefined,
    scaleFactor: async () => 1,
  };
}
