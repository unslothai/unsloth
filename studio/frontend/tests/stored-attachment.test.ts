// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();

const { uploadAttachmentFile, withSandboxAttachmentPaths, sandboxReader } =
  await import("../src/features/chat/stored-attachment.ts");
const { TOOL_ONLY_ATTACHMENT_EXTENSIONS } =
  await import("../src/features/chat/open-document-accept.ts");

type Part = { type: string; text: string };
test("stored attachments across the history name their sandbox copy once each", () => {
  const stored = (name: string, id: string, file: string) => ({
    name,
    content: [{ type: "text", text: name }] as Part[],
    storedFile: { id, sandboxPath: `.unsloth_attachments/x/${file}` },
  });
  const book = stored("book.xlsx", "a", "book.xlsx");
  const deck = stored("deck?.pptx", "b", "deck_.pptx");
  const table = stored("t.parquet", "c", "t.parquet");
  const notes = { name: "notes.txt", content: [] as Part[] };
  const messages = [
    { attachments: [notes, book] },
    { attachments: [] },
    { attachments: [deck, book, table] },
  ];
  const { messages: rendered, sandboxAttachments } =
    withSandboxAttachmentPaths(messages);
  assert.deepEqual(sandboxAttachments, [
    { id: "a", name: "book.xlsx" },
    { id: "b", name: "deck_.pptx" },
    { id: "c", name: "t.parquet" },
  ]);
  assert.equal(rendered[0].attachments[0], notes);
  assert.equal(rendered[1], messages[1]);
  assert.deepEqual(rendered[2].attachments[0].content, [
    {
      type: "text",
      text: '[deck?.pptx: its text is below, so answer from it. For calculations, the python tool has the file at path = ".unsloth_attachments/x/deck_.pptx"; fitz.open(path)]',
    },
    { type: "text", text: "deck?.pptx" },
  ]);
  assert.equal(
    rendered[2].attachments[1].content[0].text,
    '[book.xlsx: its text is below, so answer from it. For calculations, the python tool has the file at path = ".unsloth_attachments/x/book.xlsx"]',
  );
  assert.equal(
    rendered[2].attachments[2].content[0].text,
    '[t.parquet is saved at .unsloth_attachments/x/t.parquet in the python tool\'s working directory; open it with pandas.read_parquet(path), where path = ".unsloth_attachments/x/t.parquet"]',
  );
  assert.equal(book.content.length, 1);
});

test("a thread past the request's limit carries its most recent files, not none", () => {
  const one = (i: number) => ({
    name: `f${i}.parquet`,
    storedFile: { id: `${i}`, sandboxPath: `x/${i}` },
  });
  const all = Array.from({ length: 70 }, (_, i) => ({ attachments: [one(i)] }));
  all.push({ attachments: [one(0)] }); // Attached again late, so it counts as recent.
  const { messages: m, sandboxAttachments: s } =
    withSandboxAttachmentPaths(all);
  assert.deepEqual([s.length, s[0].id, s[1].id], [64, "0", "7"]);
  assert.equal(m[6].attachments[0], all[6].attachments[0]);
  assert.notEqual(m[7].attachments[0], all[7].attachments[0]);
});

test("an upload that fails or answers oddly leaves the attachment inline-only", async () => {
  const file = new File(["a,b"], "data.csv");
  const answers = [
    () => Response.json({ id: "c", sandboxPath: "p/data.csv" }),
    () => Response.json({ id: 7 }, { status: 413 }),
    () => Promise.reject(new TypeError("offline")),
  ];
  const results = [];
  for (const answer of answers) {
    globalThis.fetch = (async (_url: unknown, init?: RequestInit) => {
      assert.equal((init?.body as FormData).get("file"), file);
      return answer();
    }) as typeof fetch;
    results.push(await uploadAttachmentFile(file));
  }
  assert.deepEqual(results, [
    { id: "c", sandboxPath: "p/data.csv" },
    null,
    null,
  ]);
});

test("only document adapters keep their file, and only a python turn asks for copies", () => {
  const read = (path: string) =>
    readFileSync(
      new URL(`../src/features/chat/${path}`, import.meta.url),
      "utf8",
    );
  const provider = read("runtime-provider.tsx");
  assert.match(
    provider,
    /new VisionImageAdapter\(\),\s*new AudioAttachmentAdapter\(\),[^[]*new VideoAttachmentAdapter\(\),\s*\.\.\.\[\s*new TextAttachmentAdapter\(\),\s*new HtmlAttachmentAdapter\(\),\s*new PDFAttachmentAdapter\(\),\s*new DocxAttachmentAdapter\(\),\s*new OpenDocumentAttachmentAdapter\(\),\s*new OfficeOpenXmlAttachmentAdapter\(\),\s*new RtfAttachmentAdapter\(\),\s*new IworkAttachmentAdapter\(\),\s*\]\.map\(\(adapter\) => new StoredFileAttachmentAdapter\(adapter\)\)/,
  );
  assert.match(
    provider,
    /uploadAttachmentFile\(attachment\.file\),\s*\]\);[^}]*return storedFile\s*\?\s*\(\{ \.\.\.complete, storedFile \}/,
  );
  const adapter = read("api/chat-adapter.ts");
  assert.match(
    adapter,
    /supportsStudioToolsForThisTurn &&\s*studioLocalCodeTools\.includes\("python"\)\s*\? withSandboxAttachmentPaths\(survivingMessages\)[^;]*;\s*(\/\/[^\n]*\n\s*)*(?:const|let) outboundMessages = renderedMessages/,
  );
  assert.equal(
    adapter.split("{ sandbox_attachments: sandboxAttachments }").length,
    3,
  );
});

test("every file only the python tool reads comes with a reader", () => {
  for (const extension of TOOL_ONLY_ATTACHMENT_EXTENSIONS.split(",")) {
    assert.ok(sandboxReader(`Data${extension.toUpperCase()}`), extension);
  }
  assert.equal(sandboxReader("logs.tar.gz"), "tarfile.open(path)");
  assert.equal(sandboxReader("Deck.KEY"), null);
  assert.equal(sandboxReader("report.odt"), null);
  assert.equal(sandboxReader("app.APK"), "zipfile.ZipFile(path)");
  assert.equal(sandboxReader("notes.txt"), null);
});
