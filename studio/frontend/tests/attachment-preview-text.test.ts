// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  attachmentAudioSrc,
  countAttachmentTextLines,
  isAudioAttachment,
  parseAttachmentText,
  readAttachmentText,
  truncateAttachmentPreviewText,
} = await import("../src/features/chat/attachment-content.ts");

// The preview reads a sent attachment back out of the text the adapter built,
// so every wrapper the adapters write has to round-trip.
test("parseAttachmentText unwraps a labelled document header", () => {
  const parsed = parseAttachmentText("[PDF: report.pdf]\nline one\nline two");
  assert.deepEqual(parsed, {
    label: "PDF",
    text: "line one\nline two",
    truncated: false,
  });
});

test("parseAttachmentText unwraps the plain text attachment tag", () => {
  const parsed = parseAttachmentText(
    "<attachment name=notes.txt>\nline one\nline two\n</attachment>",
  );
  assert.deepEqual(parsed, {
    label: null,
    text: "line one\nline two",
    truncated: false,
  });
});

test("parseAttachmentText keeps text that carries no wrapper", () => {
  const parsed = parseAttachmentText("[not a label] still content");
  assert.deepEqual(parsed, {
    label: null,
    text: "[not a label] still content",
    truncated: false,
  });
});

test("parseAttachmentText keeps a header-like first line inside the body", () => {
  const parsed = parseAttachmentText("[PDF: a.pdf]\n[DOCX: b.docx]\nbody");
  assert.deepEqual(parsed, {
    label: "PDF",
    text: "[DOCX: b.docx]\nbody",
    truncated: false,
  });
});

test("truncateAttachmentPreviewText caps very long attachments", () => {
  const short = truncateAttachmentPreviewText("abc");
  assert.deepEqual(short, { text: "abc", truncated: false });

  const long = truncateAttachmentPreviewText("a".repeat(200_001));
  assert.equal(long.truncated, true);
  assert.equal(long.text.length, 200_000);
});

test("countAttachmentTextLines counts empty and single-line text", () => {
  assert.equal(countAttachmentTextLines(""), 0);
  assert.equal(countAttachmentTextLines("one line"), 1);
  assert.equal(countAttachmentTextLines("one\ntwo\n"), 3);
});

// The sent audio part only carries "mp3" or "wav", so an OGG or FLAC upload
// would be mislabelled without the attachment's own content type.
test("attachmentAudioSrc keeps the uploaded audio MIME", () => {
  const part = { data: "AAA", format: "wav" };
  assert.equal(
    attachmentAudioSrc(part, "audio/ogg", "clip.ogg"),
    "data:audio/ogg;base64,AAA",
  );
  assert.equal(
    attachmentAudioSrc({ data: "AAA", format: "mp3" }, undefined, "clip.mp3"),
    "data:audio/mpeg;base64,AAA",
  );
  assert.equal(
    attachmentAudioSrc(part, "", "clip.wav"),
    "data:audio/wav;base64,AAA",
  );
});

// An extension-only upload reaches the sent preview with an empty content type
// and format "wav", so the filename is what identifies the container.
test("attachmentAudioSrc falls back to the extension for untyped uploads", () => {
  const part = { data: "AAA", format: "wav" };
  assert.equal(
    attachmentAudioSrc(part, "", "clip.m4a"),
    "data:audio/mp4;base64,AAA",
  );
  assert.equal(
    attachmentAudioSrc(part, "application/octet-stream", "clip.flac"),
    "data:audio/flac;base64,AAA",
  );
  assert.equal(
    attachmentAudioSrc(part, undefined, "clip"),
    "data:audio/wav;base64,AAA",
  );
});

// The text and HTML adapters accept uploads with no size limit, so opening a
// preview must not materialize the whole file.
test("readAttachmentText reads a bounded slice of a large text file", async () => {
  const oversized = new File(["a".repeat(2_000_000)], "huge.txt", {
    type: "text/plain",
  });
  const { label, text, truncated } = await readAttachmentText(
    oversized,
    oversized.name,
    oversized.type,
  );
  assert.equal(label, null);
  assert.equal(truncated, true);
  assert.equal(text.length, 1_000_000);
  assert.equal(truncateAttachmentPreviewText(text).truncated, true);
});

test("readAttachmentText reads a bounded slice of a large html file", async () => {
  const parsed: number[] = [];
  const original = (globalThis as { DOMParser?: unknown }).DOMParser;
  (globalThis as { DOMParser?: unknown }).DOMParser = class {
    parseFromString(source: string) {
      parsed.push(source.length);
      return {
        querySelectorAll: () => [],
        body: { textContent: source },
      };
    }
  };

  try {
    const oversized = new File(
      [`<p>${"b".repeat(2_000_000)}</p>`],
      "huge.html",
      {
        type: "text/html",
      },
    );
    const { label, text, truncated } = await readAttachmentText(
      oversized,
      oversized.name,
      oversized.type,
    );
    assert.equal(label, "HTML");
    assert.equal(truncated, true);
    assert.deepEqual(parsed, [1_000_000]);
    assert.equal(text.length <= 1_000_000, true);
  } finally {
    (globalThis as { DOMParser?: unknown }).DOMParser = original;
  }
});

test("isAudioAttachment matches by MIME and by extension", () => {
  assert.equal(isAudioAttachment("clip.m4a", ""), true);
  assert.equal(isAudioAttachment("clip", "audio/webm"), true);
  assert.equal(isAudioAttachment("notes.txt", "text/plain"), false);
  assert.equal(isAudioAttachment(undefined, undefined), false);
});

// A bounded HTML read can extract almost nothing when the slice ends inside a
// script block, so the flag, not the text length, is what discloses the cut.
test("readAttachmentText reports truncation even when the slice extracts no text", async () => {
  const original = (globalThis as { DOMParser?: unknown }).DOMParser;
  (globalThis as { DOMParser?: unknown }).DOMParser = class {
    parseFromString() {
      return { querySelectorAll: () => [], body: { textContent: "" } };
    }
  };

  try {
    const oversized = new File(
      [`<script>${"c".repeat(2_000_000)}`],
      "big.html",
      {
        type: "text/html",
      },
    );
    const { text, truncated } = await readAttachmentText(
      oversized,
      oversized.name,
      oversized.type,
    );
    assert.equal(text, "");
    assert.equal(truncated, true);
    assert.equal(truncateAttachmentPreviewText(text).truncated, false);
  } finally {
    (globalThis as { DOMParser?: unknown }).DOMParser = original;
  }
});

// Stored payloads have no size limit, so unwrapping must copy at most the
// capped body rather than the whole attachment.
test("parseAttachmentText caps the body it copies out of a wrapper", () => {
  const body = "d".repeat(300_000);
  const tagged = parseAttachmentText(
    `<attachment name=huge.txt>\n${body}\n</attachment>`,
  );
  assert.equal(tagged.label, null);
  assert.equal(tagged.text.length, 200_000);
  assert.equal(tagged.truncated, true);

  const labelled = parseAttachmentText(`[PDF: huge.pdf]\n${body}`);
  assert.equal(labelled.label, "PDF");
  assert.equal(labelled.text.length, 200_000);
  assert.equal(labelled.truncated, true);

  const bare = parseAttachmentText(body);
  assert.equal(bare.text.length, 200_000);
  assert.equal(bare.truncated, true);
});

test("parseAttachmentText keeps an unterminated tag as plain text", () => {
  const parsed = parseAttachmentText("<attachment name=notes.txt>\nbody");
  assert.deepEqual(parsed, {
    label: null,
    text: "<attachment name=notes.txt>\nbody",
    truncated: false,
  });
});
