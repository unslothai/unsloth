// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Keep webview drop installation pending while window scale queries stay independent.
const control = (globalThis.__TAURI_WINDOW_STUB__ ??= {});

export function getCurrentWebview() {
  return {
    onDragDropEvent: (handler) =>
      new Promise((resolve) => {
        control.deliver = handler;
        control.installed = () => resolve(() => undefined);
      }),
  };
}

export function getCurrentWindow() {
  return {
    onScaleChanged: async () => () => undefined,
    scaleFactor: async () => 1,
  };
}
