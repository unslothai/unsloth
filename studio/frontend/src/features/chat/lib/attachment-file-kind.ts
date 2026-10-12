// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Name and MIME type only: the bytes can be megabytes and this runs on every tile and row.

import {
  Doc01Icon,
  FileEmpty02Icon,
  FlimSlateIcon,
  Image02Icon,
  Pdf01Icon,
  Presentation01Icon,
  SourceCodeIcon,
  Zip02Icon,
} from "@hugeicons/core-free-icons";
// Relative so the node tests can load this file.
import { AiSpeechIcon, SheetIcon } from "../../../lib/hugeicons-derived.ts";

export type AttachmentFileKind =
  | "image"
  | "pdf"
  | "audio"
  | "video"
  | "word"
  | "document"
  | "spreadsheet"
  | "presentation"
  | "web"
  | "code"
  | "text"
  | "archive"
  | "file";

const EXTENSION_KINDS: Record<string, AttachmentFileKind> = {};
function register(kind: AttachmentFileKind, extensions: string) {
  for (const extension of extensions.split(" ")) EXTENSION_KINDS[extension] = kind;
}
register("image", "png jpg jpeg gif webp avif bmp svg heic heif tif tiff ico");
register("pdf", "pdf");
register(
  "audio",
  "mp3 mp2 wav m4a ogg oga opus flac aac aiff aif aifc caf wma amr",
);
register("video", "mp4 m4v mov mkv avi webm 3gp 3g2 mpg mpeg wmv");
register("word", "doc docx docm dot dotx dotm odt ott rtf gdoc");
register("document", "pages epub");
register("spreadsheet", "csv tsv xls xlsx xlsm ods numbers");
register("presentation", "ppt pptx odp key");
register("web", "html htm xhtml mhtml");
register(
  "code",
  "py pyi ipynb js mjs cjs ts mts cts tsx jsx json jsonl c h cc cpp cxx hpp cs java kt kts go rs rb php swift scala sh bash zsh fish ps1 bat cmd css scss sass less sql yaml yml toml ini cfg conf xml r lua dart vue svelte gradle cmake proto graphql gql pl hs ex exs erl clj zig nim jl m mm",
);
register("text", "txt md markdown mdx rst log patch diff tex srt vtt");
register("archive", "zip tar gz tgz bz2 xz zst 7z rar");

// Word, Google Docs and OpenDocument text, for files that arrive without an extension.
const WORD_TYPES = new Set([
  "application/msword",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.template",
  "application/vnd.ms-word.document.macroenabled.12",
  "application/vnd.ms-word.template.macroenabled.12",
  "application/vnd.oasis.opendocument.text",
  "application/vnd.oasis.opendocument.text-template",
  "application/vnd.google-apps.document",
  "application/rtf",
  "text/rtf",
]);

const CODE_BASENAMES = new Set(["dockerfile", "makefile", "gemfile", "rakefile"]);

function extensionOf(name: string): string {
  const base = name.toLowerCase().split(/[\\/]/).pop() ?? "";
  const dot = base.lastIndexOf(".");
  return dot > 0 ? base.slice(dot + 1) : "";
}

export function attachmentFileKind(
  name: string | undefined,
  contentType: string | undefined,
): AttachmentFileKind {
  const mime = (contentType ?? "").toLowerCase();
  if (mime.startsWith("image/")) return "image";
  if (mime.startsWith("audio/")) return "audio";
  if (mime.startsWith("video/")) return "video";
  if (mime === "application/pdf") return "pdf";
  if (mime === "text/html") return "web";
  const extension = extensionOf(name ?? "");
  if (extension && Object.hasOwn(EXTENSION_KINDS, extension)) {
    return EXTENSION_KINDS[extension];
  }
  if (WORD_TYPES.has(mime.split(";", 1)[0]!.trim())) return "word";
  const base = (name ?? "").toLowerCase().split(/[\\/]/).pop() ?? "";
  if (CODE_BASENAMES.has(base)) return "code";
  if (mime.startsWith("text/")) return "text";
  return "file";
}

// Same icon shapes as the Library (features/library/file-kind.ts).
export const ATTACHMENT_KIND_ICONS = {
  image: Image02Icon,
  pdf: Pdf01Icon,
  audio: AiSpeechIcon,
  video: FlimSlateIcon,
  word: Doc01Icon,
  document: FileEmpty02Icon,
  spreadsheet: SheetIcon,
  presentation: Presentation01Icon,
  web: SourceCodeIcon,
  code: SourceCodeIcon,
  text: FileEmpty02Icon,
  archive: Zip02Icon,
  file: FileEmpty02Icon,
} as const satisfies Record<AttachmentFileKind, unknown>;

export const ATTACHMENT_KIND_ICON_CLASS: Record<AttachmentFileKind, string> = {
  image: "text-sky-500",
  pdf: "text-red-500",
  audio: "text-violet-500",
  video: "text-pink-400 scale-90",
  // Google Docs' blue, close to Word's lighter blue. Only docs are blue.
  word: "text-[#4285F4]",
  document: "text-foreground",
  spreadsheet: "text-emerald-500",
  presentation: "text-orange-500",
  web: "text-foreground",
  code: "text-foreground",
  text: "text-foreground",
  archive: "text-amber-500",
  file: "text-foreground",
};

const KIND_LABELS: Record<Exclude<AttachmentFileKind, "file">, string> = {
  image: "Image",
  pdf: "PDF",
  audio: "Audio",
  video: "Video",
  word: "Document",
  document: "Document",
  spreadsheet: "Spreadsheet",
  presentation: "Presentation",
  web: "HTML",
  code: "Code",
  text: "Text",
  archive: "Archive",
};

export function attachmentKindLabel(
  kind: AttachmentFileKind,
  name: string | undefined,
): string {
  if (kind !== "file") return KIND_LABELS[kind];
  const extension = extensionOf(name ?? "");
  return extension && extension.length <= 6 ? extension.toUpperCase() : "File";
}
