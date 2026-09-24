// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AudioWave01Icon,
  File02Icon,
  FlimSlateIcon,
  Globe02Icon,
  Image02Icon,
  Pdf01Icon,
  Presentation01Icon,
  SourceCodeIcon,
} from "@hugeicons/core-free-icons";
import { SheetIcon, TestTubeOutlineIcon } from "@/lib/hugeicons-derived";
import type { IconSvgElement } from "@hugeicons/react";
import type { LibraryItem } from "./api";

export type LibraryFileKind =
  | "image"
  | "web"
  | "document"
  | "spreadsheet"
  | "presentation"
  | "pdf"
  | "code"
  | "audio"
  | "video"
  | "model";

/** The buckets the File type filter offers. */
export type LibraryTypeFilter =
  | "images"
  | "videos"
  | "audio"
  | "documents"
  | "spreadsheets"
  | "presentations"
  | "pdfs";

const MODEL_CONTENT_TYPE = "application/x-unsloth-model";

const EXTENSION_KINDS: Record<string, LibraryFileKind> = {
  png: "image",
  jpg: "image",
  jpeg: "image",
  gif: "image",
  webp: "image",
  avif: "image",
  bmp: "image",
  svg: "image",
  html: "web",
  htm: "web",
  csv: "spreadsheet",
  tsv: "spreadsheet",
  xls: "spreadsheet",
  xlsx: "spreadsheet",
  ods: "spreadsheet",
  ppt: "presentation",
  pptx: "presentation",
  odp: "presentation",
  key: "presentation",
  pdf: "pdf",
  py: "code",
  js: "code",
  ts: "code",
  tsx: "code",
  jsx: "code",
  json: "code",
  sh: "code",
  css: "code",
  sql: "code",
  yaml: "code",
  yml: "code",
  mp3: "audio",
  wav: "audio",
  ogg: "audio",
  flac: "audio",
  m4a: "audio",
  mp4: "video",
  mov: "video",
  webm: "video",
  mkv: "video",
};

export function fileExtension(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(dot + 1).toLowerCase() : "";
}

export function isModelItem(item: LibraryItem): boolean {
  return item.model != null;
}

/** Only files download; a model is a directory. */
export function isFileItem(item: LibraryItem): boolean {
  return !isModelItem(item);
}

/** GGUF exports are deleted one quant at a time, from the model picker. */
export function isDeletable(item: LibraryItem): boolean {
  return item.model?.exportType !== "gguf";
}

const MODEL_LABELS: Record<string, string> = {
  "training:lora": "LoRA",
  "training:merged": "Full fine-tune",
  "exported:lora": "LoRA export",
  "exported:merged": "Merged export",
  "exported:gguf": "GGUF export",
};

export function modelLabel(item: LibraryItem): string {
  const model = item.model;
  return model ? (MODEL_LABELS[`${model.origin}:${model.exportType}`] ?? "Model") : "";
}

export function fileKind(item: Pick<LibraryItem, "name" | "contentType">): LibraryFileKind {
  if (item.contentType === MODEL_CONTENT_TYPE) return "model";
  const byExtension = EXTENSION_KINDS[fileExtension(item.name)];
  if (byExtension) return byExtension;
  const type = item.contentType.toLowerCase();
  if (type.startsWith("image/")) return "image";
  if (type.startsWith("audio/")) return "audio";
  if (type.startsWith("video/")) return "video";
  if (type === "application/pdf") return "pdf";
  if (type === "text/html") return "web";
  return "document";
}

/** Raster images only: an svg is served as an opaque download and never rendered inline. */
export function hasImagePreview(item: LibraryItem): boolean {
  return (
    !item.textOnly &&
    fileKind(item) === "image" &&
    fileExtension(item.name) !== "svg"
  );
}

/** Cards show a picture for raster images, and for videos their first frame. */
export function hasThumbnail(item: LibraryItem): boolean {
  return hasImagePreview(item) || (!item.textOnly && fileKind(item) === "video");
}

export const TYPE_FILTER_KINDS: Record<LibraryTypeFilter, LibraryFileKind[]> = {
  images: ["image"],
  videos: ["video"],
  audio: ["audio"],
  documents: ["document", "web", "code"],
  spreadsheets: ["spreadsheet"],
  presentations: ["presentation"],
  pdfs: ["pdf"],
};

export const KIND_ICONS: Record<LibraryFileKind, IconSvgElement> = {
  image: Image02Icon,
  web: Globe02Icon,
  document: File02Icon,
  spreadsheet: SheetIcon,
  presentation: Presentation01Icon,
  pdf: Pdf01Icon,
  code: SourceCodeIcon,
  audio: AudioWave01Icon,
  video: FlimSlateIcon,
  model: TestTubeOutlineIcon,
};

/** Tints for a few recognizable kinds; the rest use the foreground color. */
export const KIND_ICON_CLASS: Partial<Record<LibraryFileKind, string>> = {
  spreadsheet: "text-emerald-500",
  pdf: "text-red-500",
  presentation: "text-orange-500",
};

/** Kinds whose content reads as text, so the preview can show it. */
export function isTextPreviewable(item: LibraryItem): boolean {
  if (item.textOnly) return true;
  const kind = fileKind(item);
  if (kind === "code" || kind === "web") return true;
  const ext = fileExtension(item.name);
  return (
    item.contentType.startsWith("text/") ||
    ["md", "txt", "csv", "tsv", "log", "xml"].includes(ext)
  );
}
