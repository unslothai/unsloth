// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Stop loading cancels only the run in activeLoadRunRef, so an NPU load must register one.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const runtime = readFileSync(
  fileURLToPath(new URL("../src/features/chat/hooks/use-chat-model-runtime.ts", import.meta.url)),
  "utf8",
);
const start = runtime.indexOf("const loadNpuModel = useCallback(");
const end = runtime.indexOf("const ejectModel = useCallback(", start);
assert.ok(start !== -1 && end > start, "loadNpuModel not found");
const npuLoad = runtime.slice(start, end);

test("Stop loading reaches an NPU load: it owns the shared load slot", () => {
  assert.match(npuLoad, /activeLoadRunRef\.current = loadRun;/);
  assert.match(npuLoad, /loadRun.loadAttemptPath = modelPath;/);
  assert.match(npuLoad, /load_request_id: loadRun\.requestId,/);
});

test("an NPU pick supersedes a local pick still in its preflight", () => {
  assert.match(npuLoad, /const loadIntentId = \+\+modelSelectionIntentEpoch;/);
});

test("a cancel during the status read does not adopt the cancelled model", () => {
  const afterStatus = npuLoad.slice(npuLoad.indexOf("await getInferenceStatus();"));
  assert.ok(
    afterStatus.indexOf("if (signal.aborted) return;") <
      afterStatus.indexOf("setCheckpoint(modelPath, null)"),
  );
});

test("the run releases the slot through its owner and settles last", () => {
  const cleanup = npuLoad.slice(npuLoad.lastIndexOf("} finally {"));
  assert.match(cleanup, /resetLoadingUiForRun\(loadRun\)/);
  assert.doesNotMatch(npuLoad, /\bresetLoadingUi\(\)/);
  assert.ok(cleanup.indexOf("resetLoadingUiForRun") < cleanup.indexOf("markLoadRunSettled();"));
});

test("a finished NPU download does not replace a model picked meanwhile", () => {
  const pickers = readFileSync(
    fileURLToPath(
      new URL("../src/features/model-picker/components/model-selector/pickers.tsx", import.meta.url),
    ),
    "utf8",
  );
  const flow = pickers.slice(pickers.indexOf("void npuCatalog.download(model).then("));
  const guard = flow.indexOf("now.params.checkpoint === chosen");
  assert.ok(guard !== -1 && guard < flow.indexOf("pick();"));
});
