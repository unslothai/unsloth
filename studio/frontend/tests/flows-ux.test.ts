// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  categoriesOf,
  downloadBytes,
  failedSteps,
  modelSignature,
  modelsToDownload,
  needsLine,
  planCheck,
  visibleCategories,
} = await import("../src/features/flows/ux-rules.ts");

test("examples land in every category their kind, tags or models say", () => {
  assert.deepEqual(categoriesOf({ kind: "world", tags: ["npc"], needs: ["chat"] }), ["games"]);
  assert.deepEqual(
    categoriesOf({ kind: "pipeline", tags: ["video", "image"], needs: ["chat", "image", "video"] }),
    ["images", "video"],
  );
  assert.deepEqual(
    categoriesOf({ kind: "pipeline", tags: ["notebook", "training", "audio", "tts"], needs: ["training"] }),
    ["audio", "training"],
  );
  assert.deepEqual(categoriesOf({ kind: "pipeline", tags: ["data", "json"], needs: [] }), ["data"]);
  assert.deepEqual(categoriesOf({ kind: "pipeline", tags: ["misc"], needs: [] }), []);
  assert.deepEqual(categoriesOf({ kind: "pipeline" }), []);
});

test("only categories with examples get a chip, with their counts", () => {
  const chips = visibleCategories([
    { kind: "story", tags: [] },
    { kind: "pipeline", tags: ["image"] },
    { kind: "pipeline", tags: ["image", "audio"] },
  ]);
  assert.deepEqual(
    chips.map((c) => [c.id, c.count]),
    [
      ["games", 1],
      ["images", 2],
      ["audio", 1],
    ],
  );
});

test("the card says which models an example needs, or none", () => {
  assert.equal(needsLine([]), "No models needed");
  assert.equal(needsLine(["chat", "image"]), "Needs: chat model, image model");
  assert.equal(needsLine(["tts", "clone"]), "Needs: voice model");
  assert.equal(needsLine(["training"]), "Needs: training (GPU)");
  assert.equal(needsLine(undefined), null);
});

test("a run waits only for models that are neither loaded nor downloaded", () => {
  const models = [
    { kind: "chat", repo: "a/chat", downloaded: true },
    { kind: "image", repo: "b/image", downloaded: false, size_bytes: 8e9 },
    { kind: "video", repo: "c/video", downloaded: false, loaded: true },
    { kind: "tts", repo: "d/tts", downloaded: false, size_bytes: 3e8 },
    { kind: "chat", repo: "e/unknown" },
  ];
  const missing = modelsToDownload(models);
  assert.deepEqual(
    missing.map((m) => m.repo),
    ["b/image", "d/tts"],
  );
  assert.equal(downloadBytes(missing), 8.3e9);
  assert.equal(downloadBytes([{ kind: "x", repo: "y", downloaded: false }]), null);
  assert.deepEqual(modelsToDownload(null), []);
});

test("the model signature changes only when a block's model pick changes", () => {
  const a = modelSignature({
    n1: { params: { model: { repo: "Org/Chat", variant: "Q4" }, prompt: "hi" } },
    n2: { params: { prompt: "x" } },
  });
  const b = modelSignature({
    n2: { params: { prompt: "changed" } },
    n1: { params: { prompt: "hi", model: { repo: "org/chat", variant: "q4" } } },
  });
  assert.equal(a, b);
  assert.notEqual(
    a,
    modelSignature({ n1: { params: { model: { repo: "org/chat", variant: "Q8" } } } }),
  );
  assert.equal(modelSignature({}), "");
});

test("failed blocks are named the way the canvas names them, in graph order", () => {
  const failed = failedSteps(
    ["a", "b", "c"],
    {
      a: { type: "llm.generate", label: "Writer" },
      b: { type: "image.generate" },
      c: { type: "output.preview" },
    },
    {
      a: { status: "done" },
      b: { status: "failed", error: "Out of memory" },
      c: { status: "failed", error: "" },
    },
    (type) => ({ "image.generate": "Generate Image" })[type],
  );
  assert.deepEqual(failed, [
    { id: "b", title: "Generate Image", error: "Out of memory" },
    { id: "c", title: "output.preview", error: "It stopped with an error" },
  ]);
});

test("Run asks the plan before starting and holds the run when models must download", () => {
  const src = readSrc("features/flows/editor/run-controller.ts");
  assert.match(src, /if \(await heldForDownloads\(flowId, options\)\) return;/);
  assert.match(src, /options\.downloadsConfirmed/);
  const dialogs = readSrc("features/flows/editor/dialogs.tsx");
  assert.match(dialogs, /pending\?\.models ===\s*modelSignature\(useDocStore\.getState\(\)\.graph\.nodes\)/);
  assert.match(dialogs, /downloadsConfirmed: same/);
  assert.match(src, /setPendingRun\(\{ options: \{ \.\.\.options \}, plan, models: signature \}\)/);
  assert.match(readSrc("features/flows/editor/run-bar.tsx"), /<DownloadRunDialog \/>/);
});

test("enabling a disabled block changes the model signature", () => {
  const on = modelSignature({ n1: { params: { model: { repo: "a/chat" } } } });
  const off = modelSignature({ n1: { params: { model: { repo: "a/chat" } }, disabled: true } });
  assert.notEqual(on, off);
});

test("a download check is remembered only when it proves nothing must download", () => {
  const missing = [{ kind: "image", repo: "b/image", loaded: false, downloaded: false }];
  const issue = [{ node_id: "n1", message: "Prompt is not connected" }];
  const asked = { flowId: "f1", models: "a/chat" };
  assert.equal(planCheck({ models: missing, errors: [] }, asked, asked), "download");
  assert.equal(planCheck({ models: [], errors: [] }, asked, asked), "clear");
  assert.equal(planCheck({ models: [], errors: issue }, asked, asked), "unchecked");
  assert.equal(planCheck({ models: missing, errors: [] }, asked, { flowId: "f2", models: "a/chat" }), "stale");
  assert.equal(planCheck({ models: [], errors: [] }, asked, { flowId: null, models: "a/chat" }), "stale");
  assert.equal(planCheck({ models: missing, errors: [] }, asked, { flowId: "f1", models: "b/image" }), "stale");
});
