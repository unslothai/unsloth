// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";
import {
  prepareQueuedPromptFiles,
  snapshotQueuedTextPrompt,
} from "../src/features/chat/utils/queued-text-attachments.ts";
import { readTextAttachmentOnce } from "../src/features/chat/text-attachment-accept.ts";

const accept = await import("../src/features/chat/open-document-accept.ts");
const { TOOL_ONLY_ATTACHMENT_EXTENSIONS } = accept;
const uploads: string[] = [];
const originals = loadWithStubs<typeof import("../src/features/chat/attachment-originals.ts")>(
  new URL("../src/features/chat/attachment-originals.ts", import.meta.url),
  {
    "./api/chat-api": {
      uploadChatAttachmentOriginal: async (file: File) => {
        uploads.push(file.name);
        return { sha256: SHA, sizeBytes: 3 };
      },
    },
  },
);
const { persistAttachmentOriginals, reuseStagedUpload, withAttachmentOriginal } = originals;
const { sandboxAttachmentPath, sandboxReader, withSandboxAttachmentPaths } =
  loadWithStubs<typeof import("../src/features/chat/sandbox-attachments.ts")>(
    new URL("../src/features/chat/sandbox-attachments.ts", import.meta.url),
    { "./attachment-originals": originals, "./open-document-accept": accept },
  );

const SHA = "ab".repeat(32);

// Same table as test_sandbox_attachment_paths_match_the_frontend in test_sandbox_files_and_storage_roots.py.
const PATHS: [string, string][] = [
  ["data.csv", "data.csv"],
  ["../deck?.pptx", "_deck_.pptx"],
  [" .hidden. ", "hidden"],
  ["季度".repeat(45) + ".xlsx", "季度".repeat(12) + "季.xlsx"],
  ["a".repeat(79) + " ." + "x".repeat(20), "a".repeat(79)],
  ["b".repeat(10) + "." + "x".repeat(100), "b".repeat(10)],
  ["..", "attachment"],
  ["e".repeat(100), "e".repeat(80)],
  ["CON.csv", "_CON.csv"],
  ["nul.tar.gz", "_nul.tar.gz"],
  ["com1", "_com1"],
  ["CONSOLE.txt", "CONSOLE.txt"],
];

test("the sandbox path is the server's, and its basename derives it again", () => {
  for (const [name, base] of PATHS) {
    const path = sandboxAttachmentPath(SHA, name);
    assert.equal(path, `.unsloth_attachments/abababababab/${base}`, name);
    assert.equal(sandboxAttachmentPath(SHA, base), path, name);
  }
});

type Part = { type: string; text: string };
const kept = (name: string, sha256: string) => ({
  name,
  content: [{ type: "text", text: name }] as Part[],
  original: { sha256, sizeBytes: 1 },
});

test("kept files across the history name their sandbox copy once each", () => {
  const [a, b, c] = ["a", "b", "c"].map((x) => x.repeat(64));
  const book = kept("book.csv", a!);
  const deck = kept("deck?.pptx", b!);
  const table = kept("t.parquet", c!);
  const notes = { name: "notes.txt", content: [] as Part[] };
  const messages = [
    { attachments: [notes, book] },
    { attachments: [] },
    { attachments: [deck, book, table] },
  ];
  const { messages: rendered, sandboxAttachments } =
    withSandboxAttachmentPaths(messages);
  assert.deepEqual(sandboxAttachments, [
    { sha256: b, name: "deck_.pptx" },
    { sha256: a, name: "book.csv" },
    { sha256: c, name: "t.parquet" },
  ]);
  assert.equal(rendered[0]!.attachments[0], notes);
  assert.equal(rendered[1], messages[1]);
  const text = (m: number, i: number) =>
    (rendered[m]!.attachments[i] as { content: Part[] }).content[0]!.text;
  assert.equal(
    text(2, 0),
    '[deck?.pptx: its text is below, so answer from it. For calculations, the python tool has the file at path = ".unsloth_attachments/bbbbbbbbbbbb/deck_.pptx"; fitz.open(path)]',
  );
  assert.equal(
    text(2, 2),
    '[t.parquet is saved at .unsloth_attachments/cccccccccccc/t.parquet in the python tool\'s working directory; open it with pandas.read_parquet(path), where path = ".unsloth_attachments/cccccccccccc/t.parquet"]',
  );
  assert.equal(book.content.length, 1);
});

test("a thread past the request's limit carries its most recent files, not none", () => {
  const one = (i: number) => kept(`f${i}.parquet`, i.toString(16).padStart(64, "0"));
  const all = Array.from({ length: 70 }, (_, i) => ({ attachments: [one(i)] }));
  all.push({ attachments: [one(0)] });
  const { messages, sandboxAttachments } = withSandboxAttachmentPaths(all);
  assert.equal(sandboxAttachments.length, 64);
  assert.deepEqual(
    [sandboxAttachments[0]!.name, sandboxAttachments.at(-1)!.name],
    ["f7.parquet", "f0.parquet"],
  );
  assert.equal(messages[6]!.attachments[0], all[6]!.attachments[0]);
  assert.notEqual(messages[7]!.attachments[0], all[7]!.attachments[0]);
});

test("a document outside the kept types is uploaded only for the python tool", async () => {
  const send = (name: string, python: boolean, temporary = false, extra = {}) => {
    const file = new File(["a,b"], name);
    const pending = { id: "1", type: "document", name, file } as never;
    const complete = { id: "1", type: "document", name, content: [], ...extra } as never;
    return withAttachmentOriginal(pending, complete, temporary, 0, python);
  };
  const original = { sha256: SHA, sizeBytes: 3 };
  assert.deepEqual(await send("data.csv", false), { id: "1", type: "document", name: "data.csv", content: [] });
  assert.deepEqual((await send("data.csv", true) as { original?: unknown }).original, original);
  assert.ok("file" in (await send("a.pdf", false, true)));
  assert.ok("file" in (await send("a.pdf", true, true)));
  await send("t.parquet", true, false, { original });
  assert.deepEqual(uploads, ["data.csv"]);
});

test("a queued text file is uploaded before the python sandbox request", async () => {
  const file = new File(["a,b"], "data.csv", { type: "text/csv" });
  await readTextAttachmentOnce(file);
  const pending = {
    id: "queued",
    type: "document",
    name: file.name,
    contentType: file.type,
    file,
    status: { type: "requires-action", reason: "composer-send" },
  } as never;
  const queued = snapshotQueuedTextPrompt("analyze", [pending])!;
  const prepared = await prepareQueuedPromptFiles(
    queued,
    (source, complete) =>
      withAttachmentOriginal({ file: source }, complete, false, 0, true),
  );
  const { sandboxAttachments } = withSandboxAttachmentPaths([
    { attachments: prepared.attachments },
  ]);
  assert.deepEqual(sandboxAttachments, [
    { sha256: SHA, name: "data.csv" },
  ]);
});

test("only a python turn asks for copies, and every tool-only file has a reader", () => {
  const read = (path: string) =>
    readFileSync(new URL(`../src/features/chat/${path}`, import.meta.url), "utf8");
  assert.match(
    read("runtime-provider.tsx"),
    /withAttachmentOriginal\(\s*attachment,\s*await this\.delegate\.send\(attachment\),\s*incognito,\s*epoch,\s*pythonToolRunsInStudio\(\),\s*\)/,
  );
  const adapter = read("api/chat-adapter.ts");
  assert.match(
    adapter,
    /supportsStudioToolsForThisTurn &&\s*studioLocalCodeTools\.includes\("python"\)\s*\? withSandboxAttachmentPaths\(survivingMessages\)[^;]*;\s*(\/\/[^\n]*\n\s*)*(?:const|let) outboundMessages = renderedMessages/,
  );
  assert.equal(adapter.split("{ sandbox_attachments: sandboxAttachments }").length, 3);
  for (const extension of TOOL_ONLY_ATTACHMENT_EXTENSIONS.split(",")) {
    assert.ok(sandboxReader(`Data${extension.toUpperCase()}`), extension);
  }
  assert.equal(sandboxReader("logs.tar.gz"), "tarfile.open(path)");
  assert.equal(sandboxReader("report.odt"), null);
});

test("a malformed stored hash is left out instead of failing the turn", () => {
  const message = { attachments: [{ name: "a.csv", original: { sha256: "NOT-HEX", sizeBytes: 1 } }] };
  assert.deepEqual(withSandboxAttachmentPaths([message]).sandboxAttachments, []);
});

test("saving a temporary chat stores a document it kept for the python tool", async () => {
  const file = new File(["a,b"], "data.csv");
  const [saved] = await persistAttachmentOriginals([{ id: "1", type: "document", name: "data.csv", content: [], file } as never], 0);
  assert.deepEqual((saved as { original?: unknown }).original, { sha256: SHA, sizeBytes: 3 });
  assert.ok(!("file" in (saved as object)));
});

test("a staged upload is sent again once it nears the originals sweep", () => {
  const now = 10 * 60 * 60 * 1000;
  assert.equal(reuseStagedUpload(now - 60_000, now), true);
  assert.equal(reuseStagedUpload(now - 31 * 60_000, now), false);
  assert.equal(reuseStagedUpload(undefined, now), false);
});
