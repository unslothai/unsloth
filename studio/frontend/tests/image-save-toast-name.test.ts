// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Issue 15621: the save dialog can rename a file, and `save_native_file` answers with the path it
// wrote. The image save toast reported the generated name instead, naming a file that is not on
// disk, so `downloadFile` has to hand the chosen path back to its caller.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./helpers/tauri-core-resolver.mjs", import.meta.url);

const MODULE = new URL("../src/lib/native-files.ts", import.meta.url).href;

type StubControl = {
  calls: { command: string }[];
  mode: "ok" | "rejects";
  result?: unknown;
};

type NativeFiles = {
  downloadFile: (
    content: string | Blob | Uint8Array,
    filename: string,
    mimeType?: string,
  ) => Promise<string | null>;
  savedFileName: (savedPath: string | null) => string | null;
};

let generation = 0;

async function load(options: { tauri: boolean; result?: unknown }) {
  const control: StubControl = {
    calls: [],
    mode: "ok",
    result: options.result,
  };
  const define = (name: string, value: unknown) =>
    Object.defineProperty(globalThis, name, {
      value,
      configurable: true,
      writable: true,
    });
  define("__TAURI_CORE_STUB__", control);
  define("window", {
    location: { protocol: options.tauri ? "tauri:" : "https:" },
    ...(options.tauri ? { __TAURI_INTERNALS__: {} } : {}),
  });

  // `isTauri` is read once, at import time, so each load needs its own evaluation.
  generation += 1;
  const module = (await import(`${MODULE}?bust=${generation}`)) as NativeFiles;
  return { module, control };
}

const suggested = "Unsloth_20261027-224500_42.png";
const typed = "C:\\Users\\yaksh\\Pictures\\my sloth.png";

test("a renamed desktop save hands back the path the dialog wrote", async () => {
  const { module, control } = await load({ tauri: true, result: typed });
  const saved = await module.downloadFile(new Blob(["bytes"]), suggested, "image/png");
  assert.equal(saved, typed);
  assert.equal(module.savedFileName(saved) ?? suggested, "my sloth.png");
  assert.deepEqual(
    control.calls.map((call) => call.command),
    ["save_native_file"],
  );
});

test("cancelling the dialog still throws instead of reporting a save", async () => {
  const { module } = await load({ tauri: true, result: null });
  await assert.rejects(
    () => module.downloadFile(new Blob(["bytes"]), suggested, "image/png"),
    /Save cancelled/,
  );
});

test("savedFileName takes the last segment of either separator", async () => {
  const { module } = await load({ tauri: false });
  assert.equal(module.savedFileName("C:\\Users\\me\\Pictures\\shot.png"), "shot.png");
  assert.equal(module.savedFileName("/home/me/Pictures/shot.png"), "shot.png");
  assert.equal(module.savedFileName("shot.png"), "shot.png");
  // A path still ending in a separator names no file, and the web has no dialog of ours.
  assert.equal(module.savedFileName("C:\\Users\\me\\Pictures\\"), null);
  assert.equal(module.savedFileName(null), null);
  assert.equal(module.savedFileName(""), null);
});
