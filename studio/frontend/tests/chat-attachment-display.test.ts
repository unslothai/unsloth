// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Attachments show their kind: a colored icon per file type in the composer's cards and in a
// sent message's list or chips. Files can wait in the composer before any model is loaded, and
// the model that answers is checked when the message is sent.

import assert from "node:assert/strict";
import test from "node:test";

import { attachedMediaUnavailableReason } from "../src/features/chat/lib/attached-media-gate.ts";
import {
  ATTACHMENT_KIND_ICON_CLASS,
  attachmentFileKind,
  attachmentKindLabel,
} from "../src/features/chat/lib/attachment-file-kind.ts";
import {
  COMPOSER_ATTACHMENT_MAX_ROWS,
  composerAttachmentsOverflow,
  SENT_ATTACHMENT_LIST_MAX,
  sentAttachmentLayout,
} from "../src/features/chat/lib/attachment-layout.ts";
import {
  DEFAULT_CUSTOMIZATION,
  sanitizeCustomization,
} from "../src/features/settings/stores/appearance-custom-store.ts";
import { readSrcAsync } from "./helpers/kit.ts";

const ATTACHMENT = await readSrcAsync("components/assistant-ui/attachment.tsx");

test("each file kind is read off the name, with a decisive MIME type first", () => {
  const cases: Array<[string, string, string]> = [
    ["invoice_INV-6.pdf", "", "pdf"],
    ["scan", "application/pdf", "pdf"],
    ["Rain 2.6-1.mp3", "audio/mpeg", "audio"],
    ["Book.m4a", "", "audio"],
    ["clip.mov", "", "video"],
    ["Untitled document.docx", "", "document"],
    ["results.csv", "text/csv", "spreadsheet"],
    ["deck.pptx", "", "presentation"],
    ["flappy-bird(1)(1).html", "", "web"],
    ["glm-5-next-sgl.patch", "", "text"],
    ["README.md", "text/markdown", "text"],
    ["train.py", "text/x-python", "code"],
    ["Dockerfile", "", "code"],
    ["weights.zip", "application/zip", "archive"],
    ["photo.HEIC", "", "image"],
    ["mystery.bin", "application/octet-stream", "file"],
    ["notes", "text/plain", "text"],
  ];
  for (const [name, mime, kind] of cases) {
    assert.equal(attachmentFileKind(name, mime), kind, `${name} (${mime || "no type"})`);
  }
  // A misleading extension does not beat a MIME type the browser is sure of.
  assert.equal(attachmentFileKind("recording.txt", "audio/wav"), "audio");
  // Nor can a name reach into Object.prototype.
  assert.equal(attachmentFileKind("x.constructor", ""), "file");
});

test("the common kinds carry the colors people recognize", () => {
  assert.match(ATTACHMENT_KIND_ICON_CLASS.pdf, /red/);
  assert.match(ATTACHMENT_KIND_ICON_CLASS.audio, /violet/);
  assert.match(ATTACHMENT_KIND_ICON_CLASS.document, /blue/);
  assert.match(ATTACHMENT_KIND_ICON_CLASS.spreadsheet, /emerald/);
  assert.match(ATTACHMENT_KIND_ICON_CLASS.presentation, /orange/);
});

test("a sent file is labeled by its kind, or by its extension when unknown", () => {
  assert.equal(attachmentKindLabel("pdf", "a.pdf"), "PDF");
  assert.equal(attachmentKindLabel("audio", "a.mp3"), "Audio");
  assert.equal(attachmentKindLabel("document", "a.docx"), "Document");
  assert.equal(attachmentKindLabel("file", "model.safetensors"), "File");
  assert.equal(attachmentKindLabel("file", "blob.bin"), "BIN");
  assert.equal(attachmentKindLabel("file", "noextension"), "File");
});

test("Auto lists up to six sent files, then collapses them to chips", () => {
  assert.equal(SENT_ATTACHMENT_LIST_MAX, 6);
  assert.equal(sentAttachmentLayout("auto", 1), "list");
  assert.equal(sentAttachmentLayout("auto", 6), "list");
  assert.equal(sentAttachmentLayout("auto", 7), "chips");
  assert.equal(sentAttachmentLayout("list", 40), "list");
  assert.equal(sentAttachmentLayout("chips", 1), "chips");
});

test("composer cards wrap for two rows, then become one scrolling strip", () => {
  assert.equal(COMPOSER_ATTACHMENT_MAX_ROWS, 2);
  // 724px holds four 144px cards with 8px gaps.
  assert.equal(composerAttachmentsOverflow(8, 724, 144, 8), false);
  assert.equal(composerAttachmentsOverflow(9, 724, 144, 8), true);
  // A narrow composer reaches two rows sooner.
  assert.equal(composerAttachmentsOverflow(5, 300, 144, 8), true);
  assert.equal(composerAttachmentsOverflow(0, 724, 144, 8), false);
  // Before a card has laid out there is nothing to measure, so nothing changes.
  assert.equal(composerAttachmentsOverflow(20, 724, 0, 8), false);
});

test("the strip is laid out through the DOM, never through React state", () => {
  const strip = ATTACHMENT.slice(
    ATTACHMENT.indexOf("const ComposerAttachmentCards: FC"),
    ATTACHMENT.indexOf("export const ComposerAttachments: FC"),
  );
  assert.match(strip, /el\.dataset\.layout = next/);
  assert.doesNotMatch(strip, /useState/);
  // Only the attachment count re-runs the effect; widths come from a ResizeObserver.
  assert.match(strip, /\}, \[count\]\);/);
  assert.match(strip, /new ResizeObserver\(layout\)/);
});

test("sent attachments split images from files by type, with stable component maps", () => {
  assert.match(
    ATTACHMENT,
    /const SENT_IMAGE_COMPONENTS = \{\n\s*Image: SentImageTile,\n\s*Document: NoAttachment,\n\s*File: NoAttachment,\n\};/,
  );
  assert.match(
    ATTACHMENT,
    /const SENT_FILE_COMPONENTS = \{\n\s*Image: NoAttachment,\n\s*Document: SentFileItem,\n\s*File: SentFileItem,\n\};/,
  );
  // Every row and chip still opens the attachment's preview.
  const sent = ATTACHMENT.slice(ATTACHMENT.indexOf("const SentFileItem: FC"));
  assert.match(sent, /<AttachmentPreviewDialog redactFromReload=\{false\}>/);
});

test("both attachment settings default to the new look and reject anything else", () => {
  assert.equal(DEFAULT_CUSTOMIZATION.composerAttachments, "cards");
  assert.equal(DEFAULT_CUSTOMIZATION.sentAttachments, "auto");
  const saved = sanitizeCustomization({
    composerAttachments: "compact",
    sentAttachments: "chips",
  });
  assert.equal(saved.composerAttachments, "compact");
  assert.equal(saved.sentAttachments, "chips");
  const junk = sanitizeCustomization({
    composerAttachments: "huge",
    sentAttachments: 7,
  });
  assert.equal(junk.composerAttachments, "cards");
  assert.equal(junk.sentAttachments, "auto");
});

test("audio or video attached before a model loaded is checked when sent", () => {
  const listener = { hasAudioInput: true, hasVideoInput: false };
  const textOnly = { hasAudioInput: false, hasVideoInput: false };
  const base = { checkpoint: "unsloth/model", modelLabel: "Model" };
  assert.equal(
    attachedMediaUnavailableReason({ ...base, activeModel: listener, audio: true, video: false }),
    null,
  );
  assert.match(
    attachedMediaUnavailableReason({ ...base, activeModel: textOnly, audio: true, video: false }) ?? "",
    /^Model cannot accept audio\./,
  );
  assert.match(
    attachedMediaUnavailableReason({ ...base, activeModel: listener, audio: false, video: true }) ?? "",
    /^Model cannot accept video\./,
  );
  // A connected model has no row in `models`, and takes neither.
  assert.ok(
    attachedMediaUnavailableReason({ ...base, activeModel: undefined, audio: true, video: false }),
  );
  // Nothing attached, or nothing loaded to send to: not this gate's call.
  assert.equal(
    attachedMediaUnavailableReason({ ...base, activeModel: textOnly, audio: false, video: false }),
    null,
  );
  assert.equal(
    attachedMediaUnavailableReason({
      ...base,
      checkpoint: null,
      activeModel: textOnly,
      audio: true,
      video: false,
    }),
    null,
  );
});

test("attaching no longer needs a loaded model, only a capable one when one is loaded", async () => {
  const audio = await readSrcAsync("features/chat/audio-attachment-adapter.ts");
  const video = await readSrcAsync("features/chat/video-attachment-adapter.ts");
  const provider = await readSrcAsync("features/chat/runtime-provider.tsx");
  assert.ok(!audio.includes("Load a model before adding audio files."));
  assert.ok(!video.includes("Load a model before adding video."));
  assert.match(provider, /const unavailableReason = !modelLoaded\n\s*\? null/);
});

test("every attachment opens in the Library's viewer, from the composer and from a message", async () => {
  const preview = await readSrcAsync("components/assistant-ui/attachment-preview.tsx");
  const viewer = await readSrcAsync("components/assistant-ui/attachment-document-dialog.tsx");
  // Images, source and text files, clips and documents all go through one frame.
  for (const dialog of ["AttachmentImageDialog", "AttachmentTextDialog", "AttachmentAudioDialog"]) {
    const body = preview.slice(preview.indexOf(`const ${dialog}: FC`));
    assert.match(body, /^[\s\S]*?<AttachmentViewer\n/, dialog);
  }
  assert.match(viewer, /const DocumentDialog[\s\S]*?<AttachmentViewer\n/);
  assert.match(viewer, /export const AttachmentViewer[\s\S]*?<MediaViewer\n/);
  // No bare lightbox or plain dialog is left behind.
  assert.doesNotMatch(preview, /<DialogContent/);
  // "Chat about this" is the Library's, and only for a sent file: an unsent one is already here.
  assert.match(viewer, /load && !redactFromReload/);
  assert.match(viewer, /startLibraryChat\(navigate, \{ files: \[file\] \}\)/);
  // A page renders as the Library shows it, with its code a click away.
  assert.match(preview, /<ArtifactHtmlFrame code=\{preview\.text\}/);
});

test("files handed to a new chat wait until it is on screen", async () => {
  const start = await readSrcAsync("features/library/start-chat.ts");
  const fn = start.slice(start.indexOf("export function startLibraryChat("));
  const navigate = fn.indexOf('navigate({ to: "/chat"');
  const offer = fn.indexOf(".offer(");
  assert.ok(navigate !== -1 && offer > navigate, "the files are offered before the new chat opens");
  assert.match(fn, /\.then\(\(\) => \{\n\s*requestAnimationFrame\(/);
});
