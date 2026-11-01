// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import type { MediaValue, NodeSpec, ParamSpec } from "../types";

function isMedia(value: unknown): value is MediaValue {
  if (!value || typeof value !== "object") return false;
  const v = value as { kind?: unknown; ref?: unknown };
  return (
    (v.kind === "image" || v.kind === "audio" || v.kind === "video") &&
    typeof v.ref === "string"
  );
}

export const MAX_TEXT_BYTES = 200_000;
const TEXT_EXT = /\.(txt|md|markdown|json|csv|tsv|yaml|yml|xml|html?)$/i;
const MEDIA_EXT: Record<string, "image" | "audio" | "video"> = {
  png: "image",
  jpg: "image",
  jpeg: "image",
  webp: "image",
  gif: "image",
  wav: "audio",
  mp3: "audio",
  ogg: "audio",
  flac: "audio",
  m4a: "audio",
  m4b: "audio",
  mp4: "video",
  webm: "video",
};

const DOCUMENT_MIMES = new Set([
  "application/pdf",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
]);
const DOCUMENT_EXT = new Set(["pdf", "docx"]);

export function inputForAssetKind(kind: string): FileTarget["type"] | null {
  if (kind === "image" || kind === "audio" || kind === "video")
    return `input.${kind}`;
  return kind === "document" ? "input.file" : null;
}

export interface FileTarget {
  type:
    | "input.image"
    | "input.audio"
    | "input.video"
    | "input.file"
    | "input.text";
  via: "asset" | "text";
}

export function inputForFile(name: string, mime: string): FileTarget | null {
  const kind = mime.split("/", 1)[0];
  const ext = (name.split(".").pop() ?? "").toLowerCase();
  const media =
    kind === "image" || kind === "audio" || kind === "video"
      ? kind
      : MEDIA_EXT[ext];
  if (media) return { type: `input.${media}`, via: "asset" };
  if (DOCUMENT_MIMES.has(mime) || DOCUMENT_EXT.has(ext))
    return { type: "input.file", via: "asset" };
  if (kind === "text" || mime === "application/json" || TEXT_EXT.test(name))
    return { type: "input.text", via: "text" };
  return null;
}

export function dragHasFiles(types: readonly string[]): boolean {
  return types.includes("Files");
}

/** Fit never zooms out past where card text hides (0.45): wide flows pan instead. */
export const FIT_MIN_ZOOM = 0.5;

export function toolbarNodeId(
  selected: readonly string[],
  isBlock: (id: string) => boolean,
  readOnly: boolean,
): string | null {
  if (readOnly || selected.length !== 1) return null;
  return isBlock(selected[0]) ? selected[0] : null;
}

export type OutputDownload =
  | { kind: "media"; media: MediaValue; filename: string }
  | { kind: "json"; text: string; filename: string };

const EXT: Record<string, string> = {
  "image/png": "png",
  "image/jpeg": "jpg",
  "image/webp": "webp",
  "image/gif": "gif",
  "audio/wav": "wav",
  "audio/mpeg": "mp3",
  "audio/ogg": "ogg",
  "audio/flac": "flac",
  "audio/mp4": "m4a",
  "audio/webm": "weba",
  "video/mp4": "mp4",
  "video/webm": "webm",
};

function firstMedia(value: unknown, depth = 0): MediaValue | null {
  if (isMedia(value)) return value;
  if (depth > 3 || !value || typeof value !== "object") return null;
  for (const item of Array.isArray(value) ? value : Object.values(value)) {
    const found = firstMedia(item, depth + 1);
    if (found) return found;
  }
  return null;
}

export function outputDownload(
  outputs: Record<string, unknown> | null,
  baseName: string,
): OutputDownload | null {
  if (!outputs || Object.keys(outputs).length === 0) return null;
  const stem =
    baseName
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "") || "output";
  const media = firstMedia(outputs);
  if (media) {
    const fallback =
      media.kind === "image" ? "png" : media.kind === "audio" ? "wav" : "mp4";
    const ext = (media.mime && EXT[media.mime]) || fallback;
    return { kind: "media", media, filename: `${stem}.${ext}` };
  }
  return {
    kind: "json",
    text: JSON.stringify(outputs, null, 2),
    filename: `${stem}.json`,
  };
}
