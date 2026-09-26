// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { documentKind } from "@/components/file-viewer/kind";
import { isAudioAttachment } from "@/features/chat/attachment-content";

/** "document": shown as it looks (pages, a grid, slides or rendered markdown), not as text. */
export type AttachmentPreviewKind = "image" | "audio" | "text" | "document";

export type AttachmentAudioPart = { data: string; format: string };

type AttachmentContentPart = {
  type: string;
  text?: string;
  image?: string;
  audio?: AttachmentAudioPart;
};

export type AttachmentSelection = {
  kind: AttachmentPreviewKind;
  name: string;
  contentType: string | undefined;
  file: File | undefined;
  image: string | undefined;
  audio: AttachmentAudioPart | undefined;
  text: string | undefined;
  /** A sent document whose original file the server kept (features/chat/attachment-originals). */
  hasOriginal: boolean;
};

const MARKDOWN_NAME = /\.(md|markdown|mdx)$/i;
const DELIMITED_NAME = /\.(csv|tsv)$/i;

/** Whether the viewer has content: the file, a kept original, or full text (markdown, CSV). */
function isViewableDocument(
  name: string,
  contentType: string | undefined,
  file: File | undefined,
  hasOriginal: boolean,
  hasText: boolean,
): boolean {
  if (MARKDOWN_NAME.test(name)) return Boolean(file) || hasText;
  if (!documentKind(name, contentType)) return false;
  return Boolean(file) || hasOriginal || (DELIMITED_NAME.test(name) && hasText);
}

/**
 * Picks the parts a preview can show out of the attachment in scope.
 *
 * useAuiState reads through useSyncExternalStore, so this runs on every store
 * notification and on every render, and useShallow gates the re-render rather
 * than the call. It therefore only selects; the audio data URL is derived from
 * `audio` afterwards, because concatenating the base64 payload here and then
 * comparing that string in useShallow costs the whole payload every time.
 */
export const selectAttachmentSource = ({
  attachment,
}: {
  attachment: {
    type?: string;
    name: string;
    content?: AttachmentContentPart[];
  };
}): AttachmentSelection => {
  const parts = attachment.content ?? [];
  const file = (attachment as { file?: File }).file;
  const contentType =
    file?.type ||
    (attachment as { contentType?: string }).contentType ||
    undefined;
  const audio = parts.find((part) => part.type === "audio")?.audio;
  const isImage = attachment.type === "image";
  const isAudio =
    !isImage && (!!audio || isAudioAttachment(attachment.name, contentType));
  const text = parts
    .filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("\n");
  const hasOriginal = Boolean((attachment as { original?: unknown }).original);
  const isDocument =
    !isImage &&
    !isAudio &&
    isViewableDocument(attachment.name, contentType, file, hasOriginal, Boolean(text));

  return {
    kind: isImage ? "image" : isAudio ? "audio" : isDocument ? "document" : "text",
    name: attachment.name,
    contentType,
    file,
    image: isImage
      ? parts.find((part) => part.type === "image")?.image
      : undefined,
    audio: isImage ? undefined : audio,
    text: text || undefined,
    hasOriginal,
  };
};
