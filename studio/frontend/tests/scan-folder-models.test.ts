// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { countModelsPerFolder, folderDisplayName } = await import(
  "../src/features/hub/lib/scan-folder-models.ts"
);
const { scanFolderStatusCopy } = await import(
  "../src/features/hub/lib/scan-folder-status.ts"
);

test("models count toward the folder that holds them", () => {
  const counts = countModelsPerFolder(
    ["/models", "/data/lms"],
    [
      "/models/a.gguf",
      "/models/sub/b",
      "/models-old/c.gguf",
      "/data/lms/x/y.gguf",
      "/elsewhere/z.gguf",
    ],
  );
  assert.equal(counts.get("/models"), 2);
  assert.equal(counts.get("/data/lms"), 1);
});

test("a nested folder takes its own models from its parent", () => {
  const counts = countModelsPerFolder(
    ["/models", "/models/comfy"],
    ["/models/a.gguf", "/models/comfy/unet/b.safetensors"],
  );
  assert.equal(counts.get("/models"), 1);
  assert.equal(counts.get("/models/comfy"), 1);
});

test("Windows paths compare without case or slash direction", () => {
  const counts = countModelsPerFolder(
    ["D:\\Models\\"],
    ["d:/models/q4.gguf", "D:\\Models\\sub\\m"],
  );
  assert.equal(counts.get("D:\\Models\\"), 2);
});

test("a folder name is the last path segment", () => {
  assert.equal(folderDisplayName("/Users/me/LM Studio/models"), "models");
  assert.equal(folderDisplayName("C:\\AI\\gguf\\"), "gguf");
});

test("a folder skipped for taking too long says so", () => {
  const copy = scanFolderStatusCopy("slow", "Mozilla/5.0 (Macintosh)");
  assert.ok(copy);
  assert.match(copy.title, /too long/);
  assert.match(copy.hint, /network drive/);
});

test("every folder surface shares one store and the hub API", async () => {
  const [pickers, dialog, chip, chatIndex] = await Promise.all([
    readSrcAsync("features/model-picker/components/model-selector/pickers.tsx"),
    readSrcAsync("features/hub/catalog/on-device-folders-dialog.tsx"),
    readSrcAsync("features/hub/catalog/download-location-picker.tsx"),
    readSrcAsync("features/chat/index.ts"),
  ]);
  for (const source of [pickers, dialog, chip]) {
    assert.match(source, /useScanFolders\(/);
    assert.doesNotMatch(source, /\blistScanFolders\(/);
  }
  // The /api/models copy never bumped the inventory version, so the rest of the app went stale.
  assert.doesNotMatch(chatIndex, /ScanFolder/);
  // Re-adding from the dialog must not send an explicit flag that resets a folder's sub-folder setting.
  assert.match(dialog, /addScanFolderToStore\(nextPath\)/);
});
