// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { unzipSync } from "fflate";

import {
  MAX_OPEN_DOCUMENT_ARCHIVE_BYTES,
  MAX_OPEN_DOCUMENT_XML_BYTES,
  OPEN_DOCUMENT_SPREADSHEET_MIME,
  OPEN_DOCUMENT_TEXT_MIME,
  readOpenDocumentAttachmentContent,
} from "./open-document";

export type AttachmentTextLabel = "PDF" | "DOCX" | "HTML" | "ODS" | "ODT";

export type AttachmentText = {
  label: AttachmentTextLabel | null;
  text: string;
  // True when the file was only read up to the preview cap, so the dialog can
  // say so even if the extracted text ends up short.
  truncated: boolean;
};

const AUDIO_ATTACHMENT_RE = /\.(wav|mp3|m4a|ogg|oga|flac|webm|mp4|aac)$/i;
const AUDIO_MIME_RE = /^audio\//i;
const PDF_ATTACHMENT_RE = /\.pdf$/i;
const DOCX_ATTACHMENT_RE = /\.docx$/i;
const HTML_ATTACHMENT_RE = /\.x?html?$/i;
const OPEN_DOCUMENT_ATTACHMENT_RE = /\.(ods|odt)$/i;
const LABELLED_ATTACHMENT_TEXT_RE = /^\[(PDF|DOCX|HTML|ODS|ODT): [^\n]*\]\n/;
const ATTACHMENT_TAG_OPEN_RE = /^<attachment name=[^\n]*>\n/;
const ATTACHMENT_TAG_CLOSE = "\n</attachment>";
// Both wrappers start on the first line, so only a prefix is matched against.
const MAX_ATTACHMENT_WRAPPER_LENGTH = 4096;
const DOCX_MIME =
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
const AUDIO_EXTENSION_MIMES: Record<string, string> = {
  wav: "audio/wav",
  mp3: "audio/mpeg",
  m4a: "audio/mp4",
  mp4: "audio/mp4",
  ogg: "audio/ogg",
  oga: "audio/ogg",
  flac: "audio/flac",
  aac: "audio/aac",
  webm: "audio/webm",
};
// Long attachments still have to render inside a dialog, so the preview stops
// well before the point where a single <pre> stalls the webview.
const MAX_PREVIEW_TEXT_LENGTH = 200_000;
// Text and HTML have no size limit at upload, so a preview reads a bounded slice
// instead of the whole file. Five bytes per character keeps the slice past the
// character cap for any UTF-8 input, so truncation is still detected.
const MAX_PREVIEW_TEXT_BYTES = MAX_PREVIEW_TEXT_LENGTH * 5;

export function isAudioAttachment(
  name: string | undefined,
  contentType: string | undefined,
): boolean {
  return (
    AUDIO_MIME_RE.test(contentType ?? "") ||
    AUDIO_ATTACHMENT_RE.test(name ?? "")
  );
}

// The audio part keeps only the coarse format the backend needs ("mp3" or
// "wav"), so the content type wins, then the extension for uploads the browser
// typed as empty, and the part format only as a last resort.
export function attachmentAudioSrc(
  audio: { data: string; format: string },
  contentType: string | undefined,
  name: string | undefined,
): string {
  const extension = name?.toLowerCase().split(".").pop() ?? "";
  const mime = AUDIO_MIME_RE.test(contentType ?? "")
    ? (contentType as string)
    : (AUDIO_EXTENSION_MIMES[extension] ??
      (audio.format === "mp3" ? "audio/mpeg" : "audio/wav"));
  return `data:${mime};base64,${audio.data}`;
}

export function isPdfAttachment(
  name: string | undefined,
  contentType: string | undefined,
): boolean {
  return (
    contentType === "application/pdf" || PDF_ATTACHMENT_RE.test(name ?? "")
  );
}

export function isDocxAttachment(
  name: string | undefined,
  contentType: string | undefined,
): boolean {
  return contentType === DOCX_MIME || DOCX_ATTACHMENT_RE.test(name ?? "");
}

export function isHtmlAttachment(
  name: string | undefined,
  contentType: string | undefined,
): boolean {
  return contentType === "text/html" || HTML_ATTACHMENT_RE.test(name ?? "");
}

export function isOpenDocumentAttachment(
  name: string | undefined,
  contentType: string | undefined,
): boolean {
  return (
    contentType === OPEN_DOCUMENT_SPREADSHEET_MIME ||
    contentType === OPEN_DOCUMENT_TEXT_MIME ||
    OPEN_DOCUMENT_ATTACHMENT_RE.test(name ?? "")
  );
}

// unpdf and mammoth decode the whole file on the main thread, so both refuse a
// document past the ceiling the OpenDocument path already enforces, and refuse
// it before the read rather than after.
function assertDocumentAttachmentSize(file: File, label: "PDF" | "DOCX"): void {
  if (file.size > MAX_OPEN_DOCUMENT_ARCHIVE_BYTES) {
    throw new Error(`${label} file is too large: ${file.name}`);
  }
}

// mammoth takes no entry filter, so the sizes the archive declares are checked
// first. This pass decompresses nothing.
function assertDocxXmlSizes(filename: string, bytes: Uint8Array): void {
  const oversized: string[] = [];
  try {
    unzipSync(bytes, {
      filter: (entry) => {
        if (
          entry.name.startsWith("word/") &&
          entry.name.endsWith(".xml") &&
          entry.originalSize > MAX_OPEN_DOCUMENT_XML_BYTES
        ) {
          oversized.push(entry.name);
        }
        return false;
      },
    });
  } catch {
    // A malformed archive is mammoth's to report, not the size guard's.
    return;
  }
  if (oversized.length > 0) {
    throw new Error(`DOCX XML file is too large: ${filename}:${oversized[0]}`);
  }
}

export async function extractPdfAttachmentText(file: File): Promise<string> {
  assertDocumentAttachmentSize(file, "PDF");
  const [{ extractText, getDocumentProxy }, buffer] = await Promise.all([
    import("unpdf"),
    file.arrayBuffer().then((bytes) => new Uint8Array(bytes)),
  ]);
  const pdf = await getDocumentProxy(buffer);
  const { text } = await extractText(pdf, { mergePages: true });
  return text;
}

export async function extractDocxAttachmentText(file: File): Promise<string> {
  assertDocumentAttachmentSize(file, "DOCX");
  const [{ default: mammoth }, arrayBuffer] = await Promise.all([
    import("mammoth"),
    file.arrayBuffer(),
  ]);
  assertDocxXmlSizes(file.name, new Uint8Array(arrayBuffer));
  const { value } = await mammoth.extractRawText({ arrayBuffer });
  return value;
}

export function extractHtmlAttachmentText(html: string): string {
  const doc = new DOMParser().parseFromString(html, "text/html");
  for (const el of doc.querySelectorAll("script, style")) {
    el.remove();
  }
  return (doc.body.textContent ?? "").replace(/\s+/g, " ").trim();
}

// Reads the same text the matching adapter would send, so a composer preview
// shows what the model will receive.
export async function readAttachmentText(
  file: File,
  name: string,
  contentType: string | undefined,
): Promise<AttachmentText> {
  if (isPdfAttachment(name, contentType)) {
    return {
      label: "PDF",
      text: await extractPdfAttachmentText(file),
      truncated: false,
    };
  }
  if (isDocxAttachment(name, contentType)) {
    return {
      label: "DOCX",
      text: await extractDocxAttachmentText(file),
      truncated: false,
    };
  }
  if (isHtmlAttachment(name, contentType)) {
    const { text, truncated } = await readBoundedText(file);
    return {
      label: "HTML",
      text: extractHtmlAttachmentText(text),
      truncated,
    };
  }
  if (isOpenDocumentAttachment(name, contentType)) {
    const { label, text } = await readOpenDocumentAttachmentContent(
      file,
      name,
      contentType ?? "",
    );
    return { label, text, truncated: false };
  }
  return { label: null, ...(await readBoundedText(file)) };
}

async function readBoundedText(
  file: File,
): Promise<{ text: string; truncated: boolean }> {
  const truncated = file.size > MAX_PREVIEW_TEXT_BYTES;
  const slice = truncated ? file.slice(0, MAX_PREVIEW_TEXT_BYTES) : file;
  return { text: await slice.text(), truncated };
}

// A sent attachment keeps only the text its adapter produced, so the preview
// unwraps the adapter's header or tag rather than showing it to the user. The
// stored payload has no size limit, so the wrapper is matched on a prefix and
// only the capped body is copied out.
export function parseAttachmentText(raw: string): AttachmentText {
  const head = raw.slice(0, MAX_ATTACHMENT_WRAPPER_LENGTH);

  const labelled = head.match(LABELLED_ATTACHMENT_TEXT_RE);
  if (labelled) {
    return {
      label: labelled[1] as AttachmentTextLabel,
      ...sliceAttachmentBody(raw, labelled[0].length, raw.length),
    };
  }

  const tagOpen = head.match(ATTACHMENT_TAG_OPEN_RE);
  if (tagOpen && raw.endsWith(ATTACHMENT_TAG_CLOSE)) {
    return {
      label: null,
      ...sliceAttachmentBody(
        raw,
        tagOpen[0].length,
        raw.length - ATTACHMENT_TAG_CLOSE.length,
      ),
    };
  }

  return { label: null, ...sliceAttachmentBody(raw, 0, raw.length) };
}

function sliceAttachmentBody(
  raw: string,
  start: number,
  end: number,
): { text: string; truncated: boolean } {
  const bodyEnd = Math.max(start, end);
  const cappedEnd = Math.min(bodyEnd, start + MAX_PREVIEW_TEXT_LENGTH);
  return {
    text: raw.slice(start, cappedEnd),
    truncated: cappedEnd < bodyEnd,
  };
}

export function truncateAttachmentPreviewText(text: string): {
  text: string;
  truncated: boolean;
} {
  if (text.length <= MAX_PREVIEW_TEXT_LENGTH) {
    return { text, truncated: false };
  }
  return { text: text.slice(0, MAX_PREVIEW_TEXT_LENGTH), truncated: true };
}

export function countAttachmentTextLines(text: string): number {
  if (!text) {
    return 0;
  }
  return text.split("\n").length;
}
