// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

class FakeElement {
  parentElement: FakeElement | null = null;
}

const zone = new FakeElement();
// The stub fills these: `installed` settles the listener install, `deliver` is its callback.
const control: {
  installed?: () => void;
  deliver?: (event: {
    payload: { type: string; position: { x: number; y: number }; paths: string[] };
  }) => void;
} = {};

Object.assign(globalThis, {
  __TAURI_WINDOW_STUB__: control,
  HTMLElement: FakeElement,
  // lib/api-base reads this to decide it is running inside the desktop app.
  window: {
    __TAURI_INTERNALS__: {},
    devicePixelRatio: 1,
    location: { protocol: "http:" },
  },
  document: { elementFromPoint: () => zone },
});

register("./helpers/tauri-window-resolver.mjs", import.meta.url);

const { nativeDropTargetAt, registerNativeDropTarget } = await import(
  "../src/features/native-intents/native-drop-targets.ts"
);

/** The install is a dynamic import, so wait on the condition, not a tick count. */
async function until(condition: () => boolean, what: string) {
  for (let i = 0; i < 500 && !condition(); i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  assert.ok(condition(), `timed out waiting for ${what}`);
}

const dropped: string[][] = [];
registerNativeDropTarget(zone as unknown as HTMLElement, {
  onDrop: (paths) => dropped.push(paths),
});

// Claim only after the async listener is installed, or drops land nowhere.
test("a target is not claimed until its listener is installed", async () => {
  await until(() => control.installed !== undefined, "the drag-drop install");
  assert.equal(nativeDropTargetAt({ x: 10, y: 10 }), null);
});

test("the same target is claimed once the listener is installed", async () => {
  control.installed?.();
  await until(
    () => nativeDropTargetAt({ x: 10, y: 10 }) !== null,
    "the target to be claimed",
  );
  assert.equal(nativeDropTargetAt({ x: 10, y: 10 }), zone as unknown as HTMLElement);
  control.deliver?.({
    payload: { type: "drop", position: { x: 10, y: 10 }, paths: ["/tmp/a.pdf"] },
  });
  assert.deepEqual(dropped, [["/tmp/a.pdf"]]);
});
