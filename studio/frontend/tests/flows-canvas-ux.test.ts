// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { doc } from "./helpers/flows.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";
import { loadWithStubs, stubEveryImport } from "./helpers/module-stubs.ts";

registerBundlerResolver();

const { dragHasFiles, inputForAssetKind, inputForFile } = await import(
  "../src/features/flows/editor/canvas-rules.ts"
);
const { outputDownload, toolbarNodeId } = await import(
  "../src/features/flows/editor/canvas-rules.ts"
);
const { isStudioMediaUrl } = await import("../src/features/flows/media-url.ts");

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
  // api.ts pulls in the auth transport, so run it with its imports stubbed.
  const api = loadWithStubs<{ mediaPath: (m: { kind: string; ref: string; url?: string }) => string | null }>(
    new URL("../src/features/flows/api.ts", import.meta.url),
    stubEveryImport({ "./media-url": { isStudioMediaUrl } }),
  );
  const cases: [string, string | null][] = [
    ["image:7", "/api/inference/images/gallery/7/file"],
    ["audio:a b", "/api/inference/audio/gallery/a%20b/file"],
    ["video:v1", "/api/inference/video/gallery/v1/file"],
    ["sha256:ab", "/api/flows/assets/sha256%3Aab"],
    ["flowasset:x", "/api/flows/assets/flowasset%3Ax"],
    ["model:x", null],
    ["image:", null],
  ];
  for (const [ref, path] of cases) assert.equal(api.mediaPath({ kind: "image", ref }), path, ref);
  assert.equal(
    api.mediaPath({ kind: "image", ref: "x:y", url: "/api/inference/images/gallery/1/file" }),
    "/api/inference/images/gallery/1/file",
  );
  assert.equal(api.mediaPath({ kind: "image", ref: "x:y", url: "https://evil.example/x.png" }), null);
});

test("PDF and Word drops become Document blocks, and the uploaded kind picks the block", () => {
  assert.equal(inputForFile("report.pdf", "application/pdf")?.type, "input.file");
  assert.equal(inputForFile("notes.docx", "")?.type, "input.file");
  assert.equal(inputForFile("notes.md", "")?.type, "input.text");
  assert.equal(inputForAssetKind("audio"), "input.audio");
  assert.equal(inputForAssetKind("document"), "input.file");
  assert.equal(inputForAssetKind("file"), null);
});

test("an uploaded file becomes the block its uploaded kind picks", async () => {
  const docStore = await import("../src/features/flows/stores/doc-store.ts");
  const canvasRules = await import("../src/features/flows/editor/canvas-rules.ts");
  let uploaded = "image";
  const drop = loadWithStubs<{ addFileNodes: (files: unknown[], at: { x: number; y: number }) => Promise<string[]> }>(
    new URL("../src/features/flows/editor/file-drop.ts", import.meta.url),
    stubEveryImport({
      "../api": { uploadAsset: async () => ({ kind: uploaded, ref: "sha256:aa" }) },
      "../stores/doc-store": docStore,
      "../stores/ui-store": await import("../src/features/flows/stores/ui-store.ts"),
      "./canvas-rules": canvasRules,
      "./node-factory": {
        createNode: (type: string, pos: [number, number]) => ({ id: `n-${type}`, type, v: 1, params: {}, pos }),
      },
    }),
  );
  docStore.useDocStore.getState().load(doc([]));
  const file = (name: string) => ({ name, type: "", size: 10, text: async () => "" });
  const types = () => Object.values(docStore.useDocStore.getState().graph.nodes).map((n) => n.type).sort();
  await drop.addFileNodes([file("cat.png")], { x: 0, y: 0 });
  uploaded = "audio";
  await drop.addFileNodes([file("clip.mp4")], { x: 0, y: 0 });
  assert.deepEqual(types(), ["input.audio", "input.image"], "an audio-only clip becomes an Audio block");
});

test("only Studio's own media paths are fetched with the user's token", () => {
  assert.equal(isStudioMediaUrl("/api/inference/images/gallery/1/file"), true);
  assert.equal(isStudioMediaUrl("https://evil.example/x.png"), false);
  assert.equal(isStudioMediaUrl("//evil.example/x.png"), false);
  assert.equal(isStudioMediaUrl("/api//evil.example"), false);
  assert.equal(isStudioMediaUrl(null), false);
});
