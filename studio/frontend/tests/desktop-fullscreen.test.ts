// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./helpers/tauri-window-resolver.mjs", import.meta.url);

test("F11 enters and leaves the desktop window's fullscreen mode", async () => {
  const control = {
    fullscreen: false,
    fullscreenChanges: [] as boolean[],
  };
  Object.assign(globalThis, { __TAURI_WINDOW_STUB__: control });
  const { watchDesktopFullscreen } =
    await import("../src/lib/desktop-fullscreen.ts");
  const target = new EventTarget();
  watchDesktopFullscreen(target as unknown as Window);

  const press = (code: string, options = {}) => {
    const event = Object.assign(new Event("keydown", { cancelable: true }), {
      code,
      repeat: false,
      altKey: false,
      ctrlKey: false,
      metaKey: false,
      shiftKey: false,
      ...options,
    });
    target.dispatchEvent(event);
    return event;
  };
  const settle = () => new Promise((resolve) => setImmediate(resolve));

  assert.equal(press("F11").defaultPrevented, true);
  await settle();
  assert.deepEqual(control.fullscreenChanges, [true]);
  press("F11", { repeat: true });
  press("F11", { ctrlKey: true });
  press("F10");
  await settle();
  assert.deepEqual(control.fullscreenChanges, [true]);
  press("F11");
  await settle();
  assert.deepEqual(control.fullscreenChanges, [true, false]);
});
