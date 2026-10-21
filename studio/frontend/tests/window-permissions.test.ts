// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import test from "node:test";

const SRC = new URL("../src/", import.meta.url);
const CAPABILITIES = new URL(
  "../../src-tauri/capabilities/default.json",
  import.meta.url,
);
const WINDOW_SETTER =
  /\b(?:win|appWindow|getCurrentWindow\(\))\.(set\w+|maximize|unmaximize|minimize|unminimize|toggleMaximize|show|hide|center|close|start\w+)\(/g;
const SOURCE_FILE = /\.tsx?$/;
const toPermission = (method: string) =>
  `core:window:allow-${method.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}`;

// a denied call rejects at runtime, which once sent setup through its resizable fallback
test("every window setter the frontend calls is granted", async () => {
  const granted = JSON.parse(await readFile(CAPABILITIES, "utf8")).permissions;
  const files = await readdir(SRC, { recursive: true });
  const used = new Set<string>();
  for (const file of files.filter((f) => SOURCE_FILE.test(f))) {
    const source = await readFile(new URL(file, SRC), "utf8");
    for (const match of source.matchAll(WINDOW_SETTER)) {
      used.add(match[1]);
    }
  }
  assert.ok(used.has("unmaximize"));
  const missing = [...used]
    .map(toPermission)
    .filter((p) => !granted.includes(p));
  assert.deepEqual(missing, []);
});
