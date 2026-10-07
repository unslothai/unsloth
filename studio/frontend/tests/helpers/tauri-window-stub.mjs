// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Tauri window stub. Control lives on globalThis so a test can hold the install open.
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
