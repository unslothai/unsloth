// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// useAuiState reads through useSyncExternalStore, so its selector runs on every
// store notification and on every render, and useShallow gates the re-render,
// not the selector. Building a "data:audio/wav;base64,..." string in there
// therefore copies the whole clip per token of a streamed reply, twice over:
// AttachmentThumb and AttachmentPreviewDialog each mount the hook.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { selectAttachmentSource } = await import(
  "../src/components/assistant-ui/attachment-selection.ts"
);
const { assertDocumentAttachmentSize } = await import(
  "../src/features/chat/attachment-content.ts"
);

function source(path: string): string {
  return readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");
}

function audioAttachment(payload: string) {
  return {
    attachment: {
      type: "document",
      name: "clip.wav",
      contentType: "audio/wav",
      content: [
        { type: "audio" as const, audio: { data: payload, format: "wav" } },
      ],
    },
  };
}

// The selected value is what useShallow compares with Object.is. A fresh string
// each run makes that comparison walk the payload; the part itself is a stable
// reference off the store, so it compares in constant time.
test("the attachment selector does not rebuild the audio payload per run", () => {
  const state = audioAttachment("A".repeat(4 * 1024 * 1024));

  const first = selectAttachmentSource(state);
  const second = selectAttachmentSource(state);

  assert.equal(first.kind, "audio");
  for (const [key, value] of Object.entries(first)) {
    assert.equal(
      Object.is(value, second[key as keyof typeof second]),
      true,
      `selector rebuilt "${key}" on a second run over unchanged state`,
    );
  }
  assert.equal(Object.is(first.audio, state.attachment.content[0].audio), true);
});

// A plain text/document attachment must keep working: the control that has to
// pass with and without the fix.
test("the attachment selector still resolves text and image attachments", () => {
  const text = selectAttachmentSource({
    attachment: {
      type: "document",
      name: "notes.txt",
      content: [{ type: "text", text: "line one" }],
    },
  });
  assert.equal(text.kind, "text");
  assert.equal(text.text, "line one");
  assert.equal(text.audio, undefined);

  const image = selectAttachmentSource({
    attachment: {
      type: "image",
      name: "shot.png",
      content: [{ type: "image", image: "data:image/png;base64,AAA" }],
    },
  });
  assert.equal(image.kind, "image");
  assert.equal(image.image, "data:image/png;base64,AAA");
  assert.equal(image.audio, undefined);
});

// The composer runs _emptyTextAndAttachments() before it awaits adapter.send(),
// so a ceiling that only fires at send drops the typed message with the file.
// The refusal also has to be visible: the file picker calls addAttachment
// without awaiting it and nothing subscribes to attachmentAddError, so a bare
// throw leaves the user with no file and no reason, as the audio adapter's
// toast avoids.
test("the pdf and docx adapters refuse an oversized file at add, with a toast", () => {
  const provider = source("features/chat/runtime-provider.tsx");

  for (const [adapter, label] of [
    ["PDFAttachmentAdapter", "PDF"],
    ["DocxAttachmentAdapter", "DOCX"],
  ]) {
    const start = provider.indexOf(`class ${adapter}`);
    assert.notEqual(start, -1, `${adapter} not found`);
    const body = provider.slice(start, provider.indexOf("\n}", start));
    const guard = body.indexOf(
      `getDocumentAttachmentSizeError(file, "${label}")`,
    );
    assert.notEqual(
      guard,
      -1,
      `${adapter} never applies the document size ceiling`,
    );
    assert.equal(
      guard > body.indexOf("add({ file }") &&
        guard < body.indexOf("async send("),
      true,
      `${adapter} accepts a file past the ceiling and only fails at send`,
    );
    const toasted = body.indexOf("toast.error(sizeError)");
    assert.equal(
      toasted > guard && toasted < body.indexOf("async send("),
      true,
      `${adapter} refuses an oversized file without telling the user`,
    );
  }
});

test("the shared ceiling refuses an oversized document and passes a normal one", () => {
  const oversized = { name: "huge.pdf", size: 60 * 1024 * 1024 } as File;
  assert.throws(
    () => assertDocumentAttachmentSize(oversized, "PDF"),
    /PDF file is too large: huge\.pdf/,
  );
  assert.doesNotThrow(() =>
    assertDocumentAttachmentSize(
      { name: "ok.docx", size: 64 * 1024 } as File,
      "DOCX",
    ),
  );
});
