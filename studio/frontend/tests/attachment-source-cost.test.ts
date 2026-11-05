// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// useAuiState selectors run on every store notification, so building a data URL there copies
// the whole clip per streamed token.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

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

/** Radix renders DialogContent only once open, so the audio data URL is built there. */
test("the audio data URL is built in the dialog, not on every attachment tile", () => {
  const hook = readSrc("components/assistant-ui/use-attachment-source.ts");
  assert.doesNotMatch(
    hook,
    /attachmentAudioSrc/,
    "useAttachmentSource joins the audio payload before the preview opens",
  );
  assert.match(hook, /audio: source\.audio/);

  const preview = readSrc("components/assistant-ui/attachment-preview.tsx");
  const body = preview.indexOf("const AttachmentAudioBody");
  assert.notEqual(body, -1, "no component owns the audio data URL");
  assert.equal(
    preview.indexOf("attachmentAudioSrc(") > body,
    true,
    "the audio data URL is built outside AttachmentAudioBody",
  );
  assert.match(preview, /<AttachmentViewer[\s\S]*<AttachmentAudioBody/);
  const viewer = readSrc("components/assistant-ui/attachment-document-dialog.tsx");
  assert.match(viewer, /\{open && children\}/);
});

test("the attachment selector resolves a video from the composer and from a sent message", () => {
  const part = { type: "file", filename: "clip.mp4", data: "A".repeat(1024), mimeType: "video/mp4" };
  const state = {
    attachment: { type: "file", name: "clip.mp4", contentType: "video/mp4", content: [part] },
  };
  const sent = selectAttachmentSource(state);
  assert.equal(sent.kind, "video");
  assert.equal(Object.is(sent.video, part), true, "the selector copied the clip's payload");
  assert.equal(sent.audio, undefined);

  const file = new File(["x"], "clip.webm", { type: "video/webm" });
  const unsent = { attachment: { type: "file", name: "clip.webm", contentType: "video/webm", file } };
  const composer = selectAttachmentSource(unsent);
  assert.equal(composer.kind, "video");
  assert.equal(composer.file, file);

  const listen = selectAttachmentSource({
    attachment: {
      type: "file",
      name: "talk.mp4",
      contentType: "audio/mp4",
      content: [{ type: "audio" as const, audio: { data: "AAA", format: "wav" } }],
    },
  });
  assert.equal(listen.kind, "audio");
  assert.equal(listen.video, undefined);
});

test("the video data URL is built in the viewer, not on every attachment tile", () => {
  const hook = readSrc("components/assistant-ui/use-attachment-source.ts");
  assert.match(hook, /video: source\.video/);
  assert.doesNotMatch(hook, /attachmentVideoSrc/);
  const preview = readSrc("components/assistant-ui/attachment-preview.tsx");
  const body = preview.slice(preview.indexOf("const AttachmentVideoBody"));
  assert.match(body, /attachmentVideoSrc\(source\.video\)/);
  assert.match(preview, /<AttachmentViewer[\s\S]*<AttachmentVideoBody/);
  assert.match(preview, /source\.kind === "video"[\s\S]*?<AttachmentVideoDialog/);
});

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

// ComposerPrimitive.Attachments keys providers by index, so previews must reset on identity.
test("attachment previews reset when an index is reused for another attachment", () => {
  const attachment = readSrc("components/assistant-ui/attachment.tsx");
  const ui = attachment.slice(
    attachment.indexOf("const ComposerAttachmentCard: FC"),
    attachment.indexOf("const SentAttachmentLayoutContext"),
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

// The composer clears itself before awaiting send(), and nothing surfaces attachmentAddError,
// so refusal must happen at add with a toast.
test("the pdf and docx adapters refuse an oversized file at add, with a toast", () => {
  const provider = readSrc("features/chat/runtime-provider.tsx");

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
