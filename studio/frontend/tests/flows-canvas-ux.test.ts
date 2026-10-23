// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { dragHasFiles, inputForFile } = await import(
  "../src/features/flows/editor/canvas-rules.ts"
);
const { outputDownload, quickModelParam, toolbarNodeId } = await import(
  "../src/features/flows/editor/canvas-rules.ts"
);

test("a dropped file becomes the matching input block", () => {
  assert.deepEqual(inputForFile("cat.png", "image/png"), {
    type: "input.image",
    via: "asset",
  });
  assert.equal(inputForFile("song.mp3", "")?.type, "input.audio");
  assert.equal(inputForFile("clip.webm", "video/webm")?.type, "input.video");
  assert.deepEqual(inputForFile("notes.md", ""), {
    type: "input.text",
    via: "text",
  });
  assert.equal(
    inputForFile("data.json", "application/json")?.type,
    "input.text",
  );
  // A flow file is imported from the gallery, not turned into a block.
  assert.equal(inputForFile("story.unslothflow", "application/zip"), null);
  assert.equal(inputForFile("tool.exe", "application/octet-stream"), null);
});

test("only drags carrying files are taken as file drops", () => {
  assert.equal(dragHasFiles(["Files"]), true);
  assert.equal(dragHasFiles(["application/x-unsloth-flow-node"]), false);
  assert.equal(dragHasFiles(["text/plain"]), false);
});

test("the toolbar shows for one selected block in an editable flow", () => {
  const isBlock = (id: string) => id.startsWith("n");
  assert.equal(toolbarNodeId(["n1"], isBlock, false), "n1");
  assert.equal(toolbarNodeId(["n1"], isBlock, true), null);
  assert.equal(toolbarNodeId(["n1", "n2"], isBlock, false), null);
  assert.equal(toolbarNodeId([], isBlock, false), null);
  // A frame or note is selected by its key, which is not a block.
  assert.equal(toolbarNodeId(["group:g1"], isBlock, false), null);
});

test("the model switch appears only for blocks with a model setting", () => {
  const spec = (kinds: string[]) =>
    ({
      params: kinds.map((kind, i) => ({ id: `p${i}`, kind })),
    }) as never;
  assert.equal(quickModelParam(spec(["text", "model"]))?.id, "p1");
  assert.equal(quickModelParam(spec(["text"])), null);
  assert.equal(quickModelParam(undefined), null);
});

test("Download gives the first media output, else the outputs as JSON", () => {
  assert.equal(outputDownload(null, "x"), null);
  assert.equal(outputDownload({}, "x"), null);
  const media = outputDownload(
    {
      images: [{ kind: "image", ref: "sha256:ab", mime: "image/webp" }],
      text: "hi",
    },
    "My Picture!",
  );
  assert.equal(media?.kind, "media");
  assert.equal(media?.filename, "my-picture.webp");
  const clip = outputDownload({ audio: { kind: "audio", ref: "audio:1" } }, "");
  assert.equal(clip?.filename, "output.wav");
  const json = outputDownload({ rows: [1, 2] }, "Table");
  assert.equal(json?.kind, "json");
  assert.equal(json?.filename, "table.json");
  assert.deepEqual(JSON.parse(json?.kind === "json" ? json.text : "{}"), {
    rows: [1, 2],
  });
});
