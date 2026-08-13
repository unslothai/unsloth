// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  OPEN_DOCUMENT_SPREADSHEET_MIME,
  OPEN_DOCUMENT_TEXT_MIME,
  readOpenDocumentAttachmentContent,
} from "./open-document";

export type AttachmentTextLabel = "PDF" | "DOCX" | "HTML" | "ODS" | "ODT";

export type AttachmentText = {
  label: AttachmentTextLabel | null;
  text: string;
};

const AUDIO_ATTACHMENT_RE = /\.(wav|mp3|m4a|ogg|oga|flac|webm|mp4|aac)$/i;
const AUDIO_MIME_RE = /^audio\//i;
const PDF_ATTACHMENT_RE = /\.pdf$/i;
const DOCX_ATTACHMENT_RE = /\.docx$/i;
const HTML_ATTACHMENT_RE = /\.x?html?$/i;
const OPEN_DOCUMENT_ATTACHMENT_RE = /\.(ods|odt)$/i;
const LABELLED_ATTACHMENT_TEXT_RE = /^\[(PDF|DOCX|HTML|ODS|ODT): [^\n]*\]\n/;
const TAGGED_ATTACHMENT_TEXT_RE =
  /^<attachment name=[^\n]*>\n([\s\S]*)\n<\/attachment>$/;
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
// Plain text has no size limit at upload, so a preview reads a bounded slice
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

export async function extractPdfAttachmentText(file: File): Promise<string> {
  const [{ extractText, getDocumentProxy }, buffer] = await Promise.all([
    import("unpdf"),
    file.arrayBuffer().then((bytes) => new Uint8Array(bytes)),
  ]);
  const pdf = await getDocumentProxy(buffer);
  const { text } = await extractText(pdf, { mergePages: true });
  return text;
}

export async function extractDocxAttachmentText(file: File): Promise<string> {
  const [{ default: mammoth }, arrayBuffer] = await Promise.all([
    import("mammoth"),
    file.arrayBuffer(),
  ]);
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
    return { label: "PDF", text: await extractPdfAttachmentText(file) };
  }
  if (isDocxAttachment(name, contentType)) {
    return { label: "DOCX", text: await extractDocxAttachmentText(file) };
  }
  if (isHtmlAttachment(name, contentType)) {
    return {
      label: "HTML",
      text: extractHtmlAttachmentText(await file.text()),
    };
  }
  if (isOpenDocumentAttachment(name, contentType)) {
    const { label, text } = await readOpenDocumentAttachmentContent(
      file,
      name,
      contentType ?? "",
    );
    return { label, text };
  }
  return { label: null, text: await readBoundedText(file) };
}

function readBoundedText(file: File): Promise<string> {
  return (
    file.size > MAX_PREVIEW_TEXT_BYTES
      ? file.slice(0, MAX_PREVIEW_TEXT_BYTES)
      : file
  ).text();
}

// A sent attachment keeps only the text its adapter produced, so the preview
// unwraps the adapter's header or tag rather than showing it to the user.
export function parseAttachmentText(raw: string): AttachmentText {
  const labelled = raw.match(LABELLED_ATTACHMENT_TEXT_RE);
  if (labelled) {
    return {
      label: labelled[1] as AttachmentTextLabel,
      text: raw.slice(labelled[0].length),
    };
  }
  const tagged = raw.match(TAGGED_ATTACHMENT_TEXT_RE);
  if (tagged) {
    return { label: null, text: tagged[1] };
  }
  return { label: null, text: raw };
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
