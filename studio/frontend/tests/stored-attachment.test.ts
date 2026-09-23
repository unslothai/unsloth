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

const {
  storedAttachmentFile,
  toolOnlyAttachmentContent,
  uploadAttachmentFile,
  withSandboxAttachmentPaths,
  sandboxReader,
} = await import("../src/features/chat/stored-attachment.ts");
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
  const stored = { id: "c", sandboxPath: "p/data.csv" };
  const outline = { kind: "outline", text: "2 columns" };
  const answers = [
    () => Response.json(stored),
    () => Response.json({ ...stored, preview: outline }),
    // A preview missing a field of its own kind, or of no kind at all, is not one.
    () => Response.json({ ...stored, preview: { kind: "text", text: "a,b" } }),
    () => Response.json({ ...stored, preview: { kind: "image", image: "d" } }),
    () => Response.json({ ...stored, preview: { text: "a,b" } }),
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
  const none = { ...stored, preview: undefined };
  const read = { ...stored, preview: outline };
  assert.deepEqual(results, [none, read, none, none, none, null, null]);
});

test("the preview is what the model sees, and the message keeps only the note it needs", () => {
  const parts = toolOnlyAttachmentContent;
  const said = (text: string) => [{ type: "text", text }];
  const image = { kind: "image" as const, description: "8x8", image: "d:,A" };
  const text = { kind: "text" as const, label: "DOCM", text: "Hi" };
  const img = { type: "image", image: "d:,A" };
  assert.deepEqual(parts("a.psd", image, true), [img, ...said("[a.psd: 8x8]")]);
  // A model that takes no image still learns what the file holds.
  assert.deepEqual(parts("a.psd", image, false), said("[a.psd: 8x8]"));
  assert.deepEqual(parts("m.docm", text, true), said("[DOCM: m.docm]\nHi"));
  assert.deepEqual(
    parts("t.parquet", { kind: "outline", text: "3 rows" }, true),
    said("[Outline of t.parquet]\n3 rows"),
  );
  assert.deepEqual(
    parts("t.parquet", undefined, true),
    said("[t.parquet: only the python tool can read this file]"),
  );
  const upload = { id: "a", sandboxPath: "p/m.docm" };
  const inline = { ...upload, inlineText: true };
  assert.deepEqual(storedAttachmentFile({ ...upload, preview: image }), upload);
  assert.deepEqual(storedAttachmentFile({ ...upload, preview: text }), inline);
  // Its text is in the message, so the note stops telling the model to open the file for it.
  const noted = (storedFile: object) =>
    withSandboxAttachmentPaths([
      { attachments: [{ name: "m.docm", content: [] as Part[], storedFile }] },
    ]).messages[0].attachments[0].content[0].text;
  assert.match(noted(inline), /its text is below/);
  assert.match(noted(upload), /is saved at p\/m\.docm/);
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
  // The tool-only adapter sends the preview, and its image only where the vision adapter would.
  assert.match(
    provider,
    /content: upload\s*\? toolOnlyAttachmentContent\(\s*attachment\.name,\s*upload\.preview,\s*!imageInputUnavailableReason\(\),\s*\)/,
  );
  assert.equal(provider.split("imageInputUnavailableReason").length, 4);
  // Both adapters persist what the helper leaves, not the preview bytes the upload carried.
  assert.equal(provider.split("storedAttachmentFile(upload)").length, 3);
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
