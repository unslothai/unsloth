// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import test from "node:test";

// Tauri delivers OS file drops window-wide and suppresses the webview's own drop
// events, so a zone wired only to `onDrop` is dead on the desktop app: no
// drag-over border, and the file is silently ignored (#9036). Every file drop
// zone therefore has to do one of two things, and this test is what stops the
// next one from quietly doing neither.
const NATIVE_MARKERS = [
  // Claims the OS drop for its own element.
  "useNativeFileDrop",
  "useNativeDropTarget",
  "nativeDropTargetAt",
  // Or explicitly stands aside for the window-wide chat handler.
  "isTauri",
];

// Reading files out of a drag payload. `getData`/`types` alone is an in-app
// drag (block reordering, pin reordering), which the webview delivers itself.
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
    // The shared readers themselves (the DataTransfer walker, the hook) are not
    // drop zones; their callers are the ones that have to be reachable.
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

// The panel had no drag-over styling at all, on either surface, so a drop that
// did nothing looked the same as a drop that worked.
test("the project sources panel shows a drag-over state", async () => {
  const source = await readFile(
    new URL("features/rag/components/project-sources-panel.tsx", SRC),
    "utf8",
  );
  assert.match(source, /useNativeFileDrop\(\{/);
  assert.match(source, /ref=\{dropRef\}/);
  assert.match(source, /\{\.\.\.dragHandlers\}/);
  assert.match(source, /dragging && "border-primary\/60/);
  // Documents upload by lease: the native reader only serves media inline, so
  // reading a PDF back through the webview would be refused.
  assert.match(source, /onNativeIntents: handleNativeIntents/);
});

// A zone that stays registered while disabled owns the drop, so returning
// quietly is the same silent failure the issue reports.
test("a claimed drop zone that refuses a drop says so", async () => {
  const source = await readFile(
    new URL("features/rag/components/project-source-dropzone.tsx", SRC),
    "utf8",
  );
  const onDrop = source.slice(source.indexOf("const nativeDropRef"));
  assert.match(onDrop.slice(0, 600), /if \(disabled\) \{\s*toast\.error\(/);
});

// Compare mode disabled the window-wide handler outright, so a file dropped on
// a compare view produced no overlay, no toast and no attachment.
test("compare mode refuses drops out loud", async () => {
  const source = await readFile(
    new URL("features/chat/chat-page.tsx", SRC),
    "utf8",
  );
  assert.match(source, /dropsUnsupportedReason:/);
  assert.doesNotMatch(source, /enabled: active && view\.mode === "single"/);
});

// Keeping the listener on in compare must not start loading models there:
// before, nothing happened; auto-loading would replace the model behind it.
test("a refusing view loads no model either", async () => {
  const source = await readFile(
    new URL("features/native-intents/use-native-drop.ts", SRC),
    "utf8",
  );
  // The guard has to sit above the model branch, not just the attachment ones.
  const guard = source.indexOf("dropsUnsupportedReason && isActionableKind");
  const modelBranch = source.indexOf("registerNativeModelPath(dropped.path)");
  assert.ok(guard > 0 && modelBranch > guard);
  assert.match(
    source,
    /function isActionableKind[\s\S]*?dropped\.kind !== "none" && dropped\.kind !== "unsupported"/,
  );
});

// A registered target is found by hit testing document.elementFromPoint, which
// skips pointer-events-none. Disabling a zone that way un-registers it in
// practice: nativeDropTargetAt misses it, so the drop falls through to the
// window handler instead of reaching the zone's own disabled message.
test("a native drop zone stays hit-testable while disabled", async () => {
  const files = await sourceFiles(SRC);
  const hidden: string[] = [];
  for (const file of files) {
    const source = await readFile(file, "utf8");
    if (!source.includes("useNativeFileDrop(")) continue;
    // Only where it gates on the same flag the hook was told to refuse on.
    if (!/disabled\s*[,:]/.test(source)) continue;
    if (/\$\{\s*disabled\s*\?[^}]*pointer-events-none/.test(source)) {
      hidden.push(path.relative(new URL(".", SRC).pathname, file.pathname));
    }
  }
  assert.deepEqual(hidden, []);
});
