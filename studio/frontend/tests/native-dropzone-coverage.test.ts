// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

// Tauri suppresses webview drop events, so every file zone must claim or defer native drops.
const NATIVE_MARKERS = [
  "useNativeFileDrop",
  "useNativeDropTarget",
  "nativeDropTargetAt",
  "isTauri",
];

// getData/types alone is an in-app drag, which the webview delivers itself.
const FILE_DROP_MARKERS = [
  "dataTransfer.files",
  "dataTransfer.items",
  "filesFromDataTransfer",
];

const SRC = new URL("../src/", import.meta.url);

async function sourceFiles(dir: URL): Promise<URL[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  const found: URL[] = [];
  for (const entry of entries) {
    if (entry.name === "node_modules") continue;
    if (entry.isDirectory()) {
      found.push(...(await sourceFiles(new URL(`${entry.name}/`, dir))));
    } else if (/\.tsx?$/.test(entry.name)) {
      found.push(new URL(entry.name, dir));
    }
  }
  return found;
}

test("every file drop zone is reachable from the desktop app", async () => {
  const files = await sourceFiles(SRC);
  const dead: string[] = [];
  for (const file of files) {
    const source = await readFile(file, "utf8");
    if (!FILE_DROP_MARKERS.some((marker) => source.includes(marker))) continue;
    if (/export (async )?function filesFromDataTransfer/.test(source)) continue;
    if (NATIVE_MARKERS.some((marker) => source.includes(marker))) continue;
    dead.push(path.relative(new URL(".", SRC).pathname, file.pathname));
  }
  assert.deepEqual(
    dead,
    [],
    `These read files from a drag payload but neither claim the native drop nor ` +
      `defer to the window handler, so they do nothing on the desktop app: ${dead.join(", ")}`,
  );
});

test("the project sources panel shows a drag-over state", async () => {
  const source = await readFile(
    new URL("features/rag/components/project-sources-panel.tsx", SRC),
    "utf8",
  );
  assert.match(source, /useNativeFileDrop\(\{/);
  assert.match(source, /ref=\{dropRef\}/);
  assert.match(source, /\{\.\.\.dragHandlers\}/);
  assert.match(source, /dragging && "border-primary\/60/);
  // The native reader only serves media inline, so documents upload by lease.
  assert.match(source, /onNativeIntents: handleNativeIntents/);
});

// A disabled zone still owns the drop, so it must say why it refuses.
test("a claimed drop zone that refuses a drop says so", async () => {
  const source = await readFile(
    new URL("features/rag/components/project-source-dropzone.tsx", SRC),
    "utf8",
  );
  const onDrop = source.slice(source.indexOf("const nativeDropRef"));
  assert.match(onDrop.slice(0, 600), /if \(disabled\) \{\s*toast\.error\(/);
});

test("compare mode refuses drops out loud", async () => {
  const source = await readFile(
    new URL("features/chat/chat-page.tsx", SRC),
    "utf8",
  );
  assert.match(source, /dropsUnsupportedReason:/);
  assert.doesNotMatch(source, /enabled: active && view\.mode === "single"/);
});

test("a refusing view loads no model either", async () => {
  const source = await readFile(
    new URL("features/native-intents/use-native-drop.ts", SRC),
    "utf8",
  );
  const guard = source.indexOf("dropsUnsupportedReason && isActionableKind");
  const modelBranch = source.indexOf("registerNativeModelPath(dropped.path)");
  assert.ok(guard > 0 && modelBranch > guard);
  assert.match(
    source,
    /function isActionableKind[\s\S]*?dropped\.kind !== "none" && dropped\.kind !== "unsupported"/,
  );
});

// Hit testing skips pointer-events-none, which would effectively unregister the zone.
test("a native drop zone stays hit-testable while disabled", async () => {
  const files = await sourceFiles(SRC);
  const hidden: string[] = [];
  for (const file of files) {
    const source = await readFile(file, "utf8");
    if (!source.includes("useNativeFileDrop(")) continue;
    if (!/disabled\s*[,:]/.test(source)) continue;
    if (/\$\{\s*disabled\s*\?[^}]*pointer-events-none/.test(source)) {
      hidden.push(path.relative(new URL(".", SRC).pathname, file.pathname));
    }
  }
  assert.deepEqual(hidden, []);
});
