// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";
import { setTimeout as delay } from "node:timers/promises";

register("./helpers/tauri-window-resolver.mjs", import.meta.url);

test("F11 enters and leaves the desktop window's fullscreen mode", async () => {
  const control: {
    fullscreen: boolean;
    fullscreenChanges: boolean[];
    fullscreenGate?: Promise<void>;
  } = {
    fullscreen: false,
    fullscreenChanges: [],
  };
  Object.assign(globalThis, { __TAURI_WINDOW_STUB__: control });
  const { getDesktopFullscreen, setDesktopFullscreen, watchDesktopFullscreen } =
    await import("../src/lib/desktop-fullscreen.ts");
  const target = new EventTarget();
  let keyTaken = false;
  watchDesktopFullscreen(target as unknown as Window, () => keyTaken);

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
  // Wait for the native call rather than assuming the lazy import takes one tick.
  const waitForChanges = async (expected: number) => {
    const deadline = Date.now() + 10_000;
    while (
      control.fullscreenChanges.length < expected &&
      Date.now() < deadline
    ) {
      await delay(10);
    }
    assert.equal(control.fullscreenChanges.length, expected);
  };

  assert.equal(press("F11").defaultPrevented, true);
  await waitForChanges(1);
  assert.deepEqual(control.fullscreenChanges, [true]);
  // The titlebar reads this to hide itself while fullscreen.
  assert.equal(getDesktopFullscreen(), true);
  assert.equal(press("F11", { repeat: true }).defaultPrevented, false);
  assert.equal(press("F11", { ctrlKey: true }).defaultPrevented, false);
  assert.equal(press("F10").defaultPrevented, false);
  assert.deepEqual(control.fullscreenChanges, [true]);
  press("F11");
  await waitForChanges(2);
  assert.deepEqual(control.fullscreenChanges, [true, false]);
  assert.equal(getDesktopFullscreen(), false);

  let releaseGate: (() => void) | undefined;
  control.fullscreenGate = new Promise<void>((resolve) => {
    releaseGate = resolve;
  });
  press("F11");
  press("F11");
  assert.ok(releaseGate);
  releaseGate();
  await waitForChanges(3);
  assert.deepEqual(control.fullscreenChanges, [true, false, true]);

  // The fullscreen titlebar's restore button leaves fullscreen directly.
  await setDesktopFullscreen(false);
  assert.deepEqual(control.fullscreenChanges, [true, false, true, false]);
  assert.equal(getDesktopFullscreen(), false);

  keyTaken = true;
  assert.equal(press("F11").defaultPrevented, false);
  await delay(50);
  assert.deepEqual(control.fullscreenChanges, [true, false, true, false]);
});

test("a bare F11 bound in Settings has an owner the watcher can defer to", async () => {
  const { shortcutOwningBinding, useKeyboardShortcutsStore } =
    await import("../src/features/settings/stores/keyboard-shortcuts-store.ts");
  const owner = () =>
    shortcutOwningBinding(
      useKeyboardShortcutsStore.getState().overrides,
      "F11",
    );
  assert.equal(owner(), null);
  useKeyboardShortcutsStore.getState().setBinding("newChat", "primary", "F11");
  assert.equal(owner(), "newChat");
  useKeyboardShortcutsStore.getState().resetAll();
});
