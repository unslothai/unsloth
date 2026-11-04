// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { documentKind } from "@/components/file-viewer";
import { mediaKind, textFileKind } from "../file-kind";
import type { BrowserEntry } from "../store";
import { type FileTabKind, MAX_EDIT_BYTES } from "./file-tab";

type FileEntry = Extract<BrowserEntry, { kind: "file" }>;

const TEXT_UNDERNEATH = /\.(svg|csv|tsv)$/i;
const TEXT_UNDERNEATH_TYPE = /^(image\/svg\+xml|text\/csv|text\/tab-separated-values)\b/i;

export function fileTabKind(entry: FileEntry): FileTabKind {
  const { name, contentType, plainText } = entry;
  const text = textFileKind(name, contentType, plainText);
  if (text === "html" || text === "react") return "preview";
  if (text) return "text";
  const media = mediaKind(name, contentType);
  if (media === "image") return "image";
  if (media) return "media";
  return documentKind(name, contentType) ? "document" : "other";
}

export function editable(entry: FileEntry, kind: FileTabKind): boolean {
  return kind === "preview" || kind === "text" || TEXT_UNDERNEATH.test(entry.name) || TEXT_UNDERNEATH_TYPE.test(entry.contentType);
}

export async function fileText(entry: FileEntry, blob: Blob, kind: FileTabKind): Promise<{ text: string; cut: boolean }> {
  if (editable(entry, kind)) {
    const cut = blob.size > MAX_EDIT_BYTES;
    return { text: await (cut ? blob.slice(0, MAX_EDIT_BYTES) : blob).text(), cut };
  }
  const { readAttachmentText } = await import("@/features/chat/attachment-content");
  const file = new File([blob], entry.name, { type: entry.contentType });
  const { text, truncated } = await readAttachmentText(file, entry.name, entry.contentType || undefined);
  return { text, cut: truncated };
}
