// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type { PendingAttachment } from "@assistant-ui/react";
import { readTextAttachmentOnce } from "../src/features/chat/text-attachment-accept.ts";
import { createPastedTextFile } from "../src/features/chat/utils/pasted-text.ts";
import {
  canQueueTextAttachment,
  completeTextAttachment,
  snapshotQueuedTextAttachments,
  normalizeQueuedPrompt,
  queuedPromptHasContent,
  queuedPromptMessage,
} from "../src/features/chat/utils/queued-text-attachments.ts";
import { reorderPromptQueueItems } from "../src/features/chat/utils/prompt-queue-reorder.ts";

function pending(file: File): PendingAttachment {
  return {
    id: crypto.randomUUID(),
    type: "document",
    name: file.name,
    file,
    contentType: file.type,
    status: { type: "requires-action", reason: "composer-send" },
  };
}

async function ready(
  file = new File(["file contents"], "notes.txt", { type: "text/plain" }),
) {
  await readTextAttachmentOnce(file);
  return pending(file);
}

test("uploaded text reaches dispatch with the same filename and content as a normal send", async () => {
  const attachment = await ready();
  const snapshot = snapshotQueuedTextAttachments([attachment])!;
  const message = queuedPromptMessage(
    normalizeQueuedPrompt({ prompt: "  summarize  ", attachments: snapshot }),
  );
  assert.deepEqual(message.content, [{ type: "text", text: "summarize" }]);
  assert.deepEqual(message.attachments, [
    completeTextAttachment(attachment, "file contents"),
  ]);
  assert.equal(message.attachments[0].name, "notes.txt");
  assert.deepEqual(message.attachments[0].content, [
    {
      type: "text",
      text: "<attachment name=notes.txt>\nfile contents\n</attachment>",
    },
  ]);
  assert.equal("file" in message.attachments[0], false);
});

test("queueing uses decoded bytes without another asynchronous file read", async () => {
  const file = new File(
    [new Uint8Array([0xff, 0xfe, 0x48, 0, 0x69, 0])],
    "windows.txt",
  );
  const attachment = await ready(file);
  file.arrayBuffer = () => {
    throw new Error("must not reread");
  };
  file.text = () => {
    throw new Error("must not decode with File.text");
  };
  const snapshot = snapshotQueuedTextAttachments([attachment])!;
  assert.deepEqual(snapshot[0].content, [
    { type: "text", text: "<attachment name=windows.txt>\nHi\n</attachment>" },
  ]);
});

test("attachment-only messages, including an empty file, survive queue filtering", async () => {
  const attachments = snapshotQueuedTextAttachments([
    await ready(new File([""], "empty.txt")),
  ])!;
  const items = [" ", { prompt: "  ", attachments }, "next"]
    .map(normalizeQueuedPrompt)
    .filter(queuedPromptHasContent);
  assert.equal(items.length, 2);
  assert.deepEqual(queuedPromptMessage(items[0]).content, []);
  assert.equal(queuedPromptMessage(items[0]).attachments[0].name, "empty.txt");
});

test("undecoded, running, failed and unsupported attachments cannot slip into a queue", async () => {
  const text = await ready();
  assert.equal(
    canQueueTextAttachment(pending(new File(["text"], "new.txt"))),
    false,
  );
  assert.equal(snapshotQueuedTextAttachments([]), null);
  const unsupported = pending(
    new File(["PDF bytes"], "report.pdf", { type: "application/pdf" }),
  );
  assert.equal(snapshotQueuedTextAttachments([text, unsupported]), null);
  assert.equal(canQueueTextAttachment({ ...text, type: "image" }), false);
  assert.equal(
    canQueueTextAttachment({
      ...text,
      status: { type: "running", reason: "uploading", progress: 0 },
    }),
    false,
  );
  assert.equal(
    canQueueTextAttachment({
      ...text,
      status: { type: "incomplete", reason: "error" },
    }),
    false,
  );
});

test("mixed uploads and pastes preserve file order and paste metadata", async () => {
  const paste = createPastedTextFile("pasted body");
  const attachments = snapshotQueuedTextAttachments([
    await ready(),
    await ready(paste),
  ])!;
  assert.deepEqual(
    attachments.map((a) => a.name),
    ["notes.txt", paste.name],
  );
  assert.match(
    (attachments[1].content[0] as { text: string }).text,
    /^<pasted_text name=.*bytes=11>\npasted body\n<\/pasted_text>$/,
  );
});

test("editing, reordering and removing prompts keeps each file with its own message", async () => {
  const source = await ready();
  const attachments = snapshotQueuedTextAttachments([source])!;
  // Removing or mutating the original composer attachment cannot change the queued snapshot.
  source.name = "changed.txt";
  source.content = [{ type: "text", text: "changed" }];
  const first = { id: "first", prompt: "original", attachments };
  const second = { id: "second", prompt: "next" };
  first.prompt = "edited";
  const moved = reorderPromptQueueItems([first, second], 0, 1, 0)!;
  assert.equal(moved[1], first);
  assert.equal(queuedPromptMessage(moved[1]).attachments[0].name, "notes.txt");
  assert.deepEqual(queuedPromptMessage(moved[1]).content, [
    { type: "text", text: "edited" },
  ]);
  assert.equal(queuedPromptHasContent({ ...first, prompt: "" }), true);
  assert.equal(queuedPromptHasContent({ prompt: "" }), false);
  moved.splice(1, 1);
  assert.equal(queuedPromptMessage(moved[0]).attachments.length, 0);
});
