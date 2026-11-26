// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import type { PendingAttachment } from "@assistant-ui/react";
import {
  cachedTextAttachment,
  readTextAttachment,
  readTextAttachmentOnce,
} from "../src/features/chat/text-attachment-accept.ts";
import {
  createPastedTextFile,
  pastedTextOf,
} from "../src/features/chat/utils/pasted-text.ts";
import { snapshotQueuedTextAttachments } from "../src/features/chat/utils/queued-text-attachments.ts";

function adapterRead(file: File): Promise<string> {
  return readTextAttachmentOnce(file, pastedTextOf(file));
}

const XML_LATIN1 = '<?xml version="1.0" encoding="ISO-8859-1"?><a>café</a>';

function pending(file: File): PendingAttachment {
  return {
    id: crypto.randomUUID(),
    type: "document",
    name: file.name,
    contentType: file.type,
    file,
    status: { type: "requires-action", reason: "composer-send" },
  } as PendingAttachment;
}

test("a pasted XML declaration does not re-decode the pasted text", async () => {
  const file = createPastedTextFile(XML_LATIN1);
  assert.equal(await adapterRead(file), XML_LATIN1);
  assert.equal(cachedTextAttachment(file), XML_LATIN1);
});

test("a paste declaring a charset the decoder refuses is still accepted", async () => {
  const text = '<?xml version="1.0" encoding="x-unknown-charset"?><a>naïve</a>';
  const file = createPastedTextFile(text);
  assert.equal(await adapterRead(file), text);
});

test("queue snapshots of a mixed send carry the exact pasted text", async () => {
  const pasted = createPastedTextFile(XML_LATIN1);
  const uploaded = new File(["plain"], "notes.txt", { type: "text/plain" });
  await adapterRead(pasted);
  await adapterRead(uploaded);
  const snapshot = snapshotQueuedTextAttachments([
    pending(pasted),
    pending(uploaded),
  ]);
  assert.ok(snapshot);
  const content = snapshot[0]!.content[0] as { text: string };
  assert.ok(content.text.includes(`\n${XML_LATIN1}\n`));
});

for (const [label, text] of [
  ["plain UTF-8", "héllo wörld 日本語 \u{1F600}\r\nline two"],
  ["leading U+FEFF", "\uFEFFstarts with a BOM \uFEFF inside"],
  ["lone surrogates", "a\uD800b\uDC00c😀d\uDBFF"],
  ["only a BOM", "\uFEFF"],
  ["trailing high surrogate", "end\uD83D"],
] as const) {
  test(`a ${label} paste reads exactly as its UTF-8 bytes decode`, async () => {
    const file = createPastedTextFile(text);
    const fromBytes = await readTextAttachment(file);
    assert.equal(await adapterRead(file), fromBytes);
  });
}

test("an uploaded XML file still honours its declared charset", async () => {
  const bytes = new Uint8Array(
    [...XML_LATIN1].map((char) => char.charCodeAt(0)),
  );
  const file = new File([bytes], "feed.xml", { type: "text/xml" });
  const text = await adapterRead(file);
  assert.equal(text, XML_LATIN1);
  assert.equal(text, await readTextAttachment(file));
});

test("an uploaded UTF-8 file declaring Latin-1 is decoded as before", async () => {
  const file = new File([XML_LATIN1], "feed.xml", { type: "text/xml" });
  const text = await adapterRead(file);
  assert.equal(text, await readTextAttachment(file));
  assert.ok(text.includes("cafÃ©"));
});

test("the text adapter hands the pasted string to both add and send", () => {
  const source = readFileSync(
    new URL("../src/features/chat/runtime-provider.tsx", import.meta.url),
    "utf8",
  );
  const calls = source.match(/readTextAttachmentOnce\([^)]*\)?\)/g) ?? [];
  assert.deepEqual(calls, [
    "readTextAttachmentOnce(file, pastedTextOf(file))",
    "readTextAttachmentOnce(attachment.file, pastedTextOf(attachment.file))",
  ]);
});
