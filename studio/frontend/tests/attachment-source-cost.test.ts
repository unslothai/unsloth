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

const ATTACHMENT_ID_SELECTOR_RE =
  /const attachmentId = useAuiState\(\(\{ attachment \}\) => attachment\.id\)/;
const ATTACHMENT_ID_ROOT_KEY_RE =
  /<AttachmentPrimitive\.Root\s+key=\{attachmentId\}/;
const PASTED_TEXT_ID_KEY_RE =
  /<PastedTextAttachmentUI\s+key=\{attachmentId\}/;

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

/**
 * The selector hands the part through, so the hook must not join it either.
 *
 * Every tile in a transcript mounts `useAttachmentSource`, and the dialog
 * mounts it again, so a data URL built there copies MAX_AUDIO_SIZE of base64
 * twice per sent clip with the dialog still closed. Radix only renders
 * DialogContent once the dialog opens, which is where the join belongs.
 */
test("the audio data URL is built in the dialog, not on every attachment tile", () => {
  const hook = source("components/assistant-ui/use-attachment-source.ts");
  assert.doesNotMatch(
    hook,
    /attachmentAudioSrc/,
    "useAttachmentSource joins the audio payload before the preview opens",
  );
  assert.match(hook, /audio: source\.audio/);

  const preview = source("components/assistant-ui/attachment-preview.tsx");
  const body = preview.indexOf("const AttachmentAudioBody");
  assert.notEqual(body, -1, "no component owns the audio data URL");
  assert.equal(
    preview.indexOf("attachmentAudioSrc(") > body,
    true,
    "the audio data URL is built outside AttachmentAudioBody",
  );
  assert.match(preview, /<DialogContent[\s\S]*<AttachmentAudioBody/);
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

// ComposerPrimitive.Attachments keys its providers by list index. Removing the
// first attachment therefore reuses AttachmentUI for the next attachment, so
// every stateful preview/source branch needs the attachment identity as its own
// reset boundary.
test("attachment previews reset when an index is reused for another attachment", () => {
  const attachment = source("components/assistant-ui/attachment.tsx");
  const ui = attachment.slice(
    attachment.indexOf("const AttachmentUI"),
    attachment.indexOf("const AttachmentRemove"),
  );

  assert.match(ui, ATTACHMENT_ID_SELECTOR_RE);
  assert.match(
    ui,
    ATTACHMENT_ID_ROOT_KEY_RE,
    "index reuse can carry the previous dialog and object URL state into the successor tile",
  );
  assert.match(
    ui,
    PASTED_TEXT_ID_KEY_RE,
    "index reuse can carry an in-progress pasted-text conversion into the successor tile",
  );
});

// The composer runs _emptyTextAndAttachments() before it awaits adapter.send(),
// so a ceiling that only fires at send drops the typed message with the file.
// The refusal also has to be visible: the file picker calls addAttachment
// without awaiting it and nothing subscribes to attachmentAddError, so a bare
// throw leaves the user with no file and no reason, as the audio adapter's
// toast avoids.
test("the pdf and docx adapters refuse an oversized file at add, with a toast", () => {
  const provider = source("features/chat/runtime-provider.tsx");

  // DOCX also has to clear its per-part bound at add: a small archive can
  // still declare a part mammoth would inflate past the cap.
  for (const [adapter, call, toast] of [
    [
      "PDFAttachmentAdapter",
      'getDocumentAttachmentSizeError(file, "PDF")',
      "toast.error(sizeError)",
    ],
    [
      "DocxAttachmentAdapter",
      "await getDocxAttachmentError(file)",
      "toast.error(error)",
    ],
  ]) {
    const start = provider.indexOf(`class ${adapter}`);
    assert.notEqual(start, -1, `${adapter} not found`);
    const body = provider.slice(start, provider.indexOf("\n}", start));
    const guard = body.indexOf(call);
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
    const toasted = body.indexOf(toast);
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
