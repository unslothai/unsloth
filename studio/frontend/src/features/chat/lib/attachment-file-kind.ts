// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What kind of file an attachment is, for its icon, color and label. Read off the name and the
// MIME type only: an attachment's bytes can be megabytes and this runs on every tile and row.

import {
  AudioWave01Icon,
  Doc02Icon,
  File02Icon,
  Globe02Icon,
  Image02Icon,
  Pdf02Icon,
  Presentation01Icon,
  SourceCodeIcon,
  Video01Icon,
  Xls02Icon,
  Zip02Icon,
} from "@hugeicons/core-free-icons";

export type AttachmentFileKind =
  | "image"
  | "pdf"
  | "audio"
  | "video"
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
register("document", "doc docx odt rtf pages epub");
register("spreadsheet", "csv tsv xls xlsx xlsm ods numbers");
register("presentation", "ppt pptx odp key");
register("web", "html htm xhtml mhtml");
register(
  "code",
  "py pyi ipynb js mjs cjs ts mts cts tsx jsx json jsonl c h cc cpp cxx hpp cs java kt kts go rs rb php swift scala sh bash zsh fish ps1 bat cmd css scss sass less sql yaml yml toml ini cfg conf xml r lua dart vue svelte gradle cmake proto graphql gql pl hs ex exs erl clj zig nim jl m mm",
);
register("text", "txt md markdown mdx rst log patch diff tex srt vtt");
register("archive", "zip tar gz tgz bz2 xz zst 7z rar");

// Extensionless names that are still source.
const CODE_BASENAMES = new Set(["dockerfile", "makefile", "gemfile", "rakefile"]);

function extensionOf(name: string): string {
  const base = name.toLowerCase().split(/[\\/]/).pop() ?? "";
  const dot = base.lastIndexOf(".");
  return dot > 0 ? base.slice(dot + 1) : "";
}

/** The kind of an attachment, from its name and MIME type. The MIME type wins where it is
 *  decisive, since a browser types media and PDFs reliably; the extension decides the rest,
 *  which is also where an empty or generic MIME type lands. */
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
  const base = (name ?? "").toLowerCase().split(/[\\/]/).pop() ?? "";
  if (CODE_BASENAMES.has(base)) return "code";
  if (mime.startsWith("text/")) return "text";
  return "file";
}

export const ATTACHMENT_KIND_ICONS = {
  image: Image02Icon,
  pdf: Pdf02Icon,
  audio: AudioWave01Icon,
  video: Video01Icon,
  document: Doc02Icon,
  spreadsheet: Xls02Icon,
  presentation: Presentation01Icon,
  web: Globe02Icon,
  code: SourceCodeIcon,
  text: File02Icon,
  archive: Zip02Icon,
  file: File02Icon,
} as const satisfies Record<AttachmentFileKind, unknown>;

/** Each kind's color, as the Library and most file pickers tell them apart at a glance. The web
 *  page globe and the generic file stay the text color. */
export const ATTACHMENT_KIND_ICON_CLASS: Record<AttachmentFileKind, string> = {
  image: "text-sky-500",
  pdf: "text-red-500",
  audio: "text-violet-500",
  video: "text-pink-500",
  document: "text-blue-500",
  spreadsheet: "text-emerald-500",
  presentation: "text-orange-500",
  web: "text-foreground",
  code: "text-blue-500",
  text: "text-blue-500",
  archive: "text-amber-500",
  file: "text-muted-foreground",
};

const KIND_LABELS: Record<Exclude<AttachmentFileKind, "file">, string> = {
  image: "Image",
  pdf: "PDF",
  audio: "Audio",
  video: "Video",
  document: "Document",
  spreadsheet: "Spreadsheet",
  presentation: "Presentation",
  web: "HTML",
  code: "Code",
  text: "Text",
  archive: "Archive",
};

/** The line under a sent file's name: its kind, or for an unknown kind its extension. */
export function attachmentKindLabel(
  kind: AttachmentFileKind,
  name: string | undefined,
): string {
  if (kind !== "file") return KIND_LABELS[kind];
  const extension = extensionOf(name ?? "");
  return extension && extension.length <= 6 ? extension.toUpperCase() : "File";
}
