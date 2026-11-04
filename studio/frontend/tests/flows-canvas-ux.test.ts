// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { dragHasFiles, inputForAssetKind, inputForFile } = await import(
  "../src/features/flows/editor/canvas-rules.ts"
);
const { outputDownload, toolbarNodeId } = await import(
  "../src/features/flows/editor/canvas-rules.ts"
);

test("a dropped file becomes the matching input block", () => {
  assert.deepEqual(inputForFile("cat.png", "image/png"), {
    type: "input.image",
    via: "asset",
  });
  assert.equal(inputForFile("song.mp3", "")?.type, "input.audio");
  assert.equal(inputForFile("memo.m4a", "")?.type, "input.audio");
  assert.equal(inputForFile("clip.webm", "video/webm")?.type, "input.video");
  assert.deepEqual(inputForFile("notes.md", ""), {
    type: "input.text",
    via: "text",
  });
  assert.equal(
    inputForFile("data.json", "application/json")?.type,
    "input.text",
  );
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
  assert.equal(toolbarNodeId(["group:g1"], isBlock, false), null);
});

test("the card toolbar holds actions only: no model picker or other setting", () => {
  const bar = readFileSync(new URL("../src/features/flows/editor/node-toolbar.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(bar, /ParamField|node-toolbar-model|Use the loaded/);
  for (const id of ["run", "reroll", "download", "delete"]) assert.match(bar, new RegExp(`data-testid="node-toolbar-${id}"`));
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
  const memo = outputDownload(
    { audio: { kind: "audio", ref: "sha256:cd", mime: "audio/mp4" } },
    "Memo",
  );
  assert.equal(memo?.filename, "memo.m4a");
  const voice = outputDownload(
    { audio: { kind: "audio", ref: "sha256:ef", mime: "audio/webm" } },
    "Voice",
  );
  assert.equal(voice?.filename, "voice.weba");
  const json = outputDownload({ rows: [1, 2] }, "Table");
  assert.equal(json?.kind, "json");
  assert.equal(json?.filename, "table.json");
  assert.deepEqual(JSON.parse(json?.kind === "json" ? json.text : "{}"), {
    rows: [1, 2],
  });
});

test("generated gallery media downloads through its authenticated /api/ route", () => {
  // api.ts pulls in the auth transport, so check the mapping in its source.
  const src = readSrc("features/flows/api.ts");
  const body = src.slice(
    src.indexOf("export function mediaPath"),
    src.indexOf("export async function fetchMediaBlob"),
  );
  assert.match(
    src,
    /GALLERY_ROUTES: Record<string, string> = \{\s*image: "images",\s*audio: "audio",\s*video: "video",/,
  );
  assert.match(
    body,
    /return `\/api\/inference\/\$\{gallery\}\/gallery\/\$\{encodeURIComponent\(id\)\}\/file`;/,
  );
});


test("PDF and Word drops become Document blocks, and the uploaded kind picks the block", () => {
  assert.equal(inputForFile("report.pdf", "application/pdf")?.type, "input.file");
  assert.equal(inputForFile("notes.docx", "")?.type, "input.file");
  assert.equal(inputForFile("notes.md", "")?.type, "input.text");
  assert.equal(inputForAssetKind("audio"), "input.audio");
  assert.equal(inputForAssetKind("document"), "input.file");
  assert.equal(inputForAssetKind("file"), null);
  const src = readSrc("features/flows/editor/file-drop.ts");
  assert.match(src, /const fits = inputForAssetKind\(asset\.kind\);\s*if \(fits && fits !== target\.type\) node = createNode\(fits, position\);/);
});
