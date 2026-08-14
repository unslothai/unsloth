// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { strFromU8, unzipSync } from "fflate";

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
// mammoth picks the parts it parses out of the relationships, not out of the
// filenames, so a target may be called anything ("payload.bin") and still be
// inflated and parsed as XML, while an .xml part nothing points at (customXml,
// the glossary document) is never opened at all. The bound therefore follows
// docx-reader.js: the two package parts it always reads, the main document and
// its relationships, and the five parts resolved out of those relationships,
// each with mammoth's own "word/<name>.xml" fallback. Image targets stay lazy
// and are never read by extractRawText.
const DOCX_CONTENT_TYPES_PART = "[Content_Types].xml";
const DOCX_PACKAGE_RELATIONSHIPS = "_rels/.rels";
const DOCX_RELATIONSHIP_NAMESPACE =
  "http://schemas.openxmlformats.org/officeDocument/2006/relationships/";
const DOCX_MAIN_DOCUMENT_TYPE = `${DOCX_RELATIONSHIP_NAMESPACE}officeDocument`;
const DOCX_RELATED_PART_NAMES = [
  "comments",
  "endnotes",
  "footnotes",
  "numbering",
  "styles",
];
// readXmlFileWithBody opens the relationships of every part it reads a body
// from, so these three carry a .rels part of their own.
const DOCX_BODY_PART_NAMES = new Set(["comments", "endnotes", "footnotes"]);
const DOCX_MAIN_DOCUMENT_FALLBACK = "word/document.xml";
// A <Relationship> tag, with the element prefix mammoth's namespace mapping
// accepts, and skipping any ">" that sits inside an attribute value.
const DOCX_RELATIONSHIP_TAG_RE =
  /<(?:[^\s/>"'=]+:)?Relationship(?=[\s/>])(?:"[^"]*"|'[^']*'|[^"'>])*>/g;
// Attribute names are matched by consuming whole name="value" pairs, so a
// value that itself contains Target= cannot be mistaken for an attribute.
// mammoth reads child.attributes.Target, which a prefixed r:Target never
// populates, so prefixed names are deliberately not accepted here.
const XML_ATTRIBUTE_RE = /([^\s/>"'=]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g;
const XML_ENTITY_RE = /&(?:#(\d+)|#[xX]([\da-fA-F]+)|([a-zA-Z]+));/g;
const XML_NAMED_ENTITIES: Record<string, string> = {
  amp: "&",
  lt: "<",
  gt: ">",
  quot: '"',
  apos: "'",
};
// add() unzips the archive to check its parts, so send() and the preview reuse
// that verdict instead of unzipping the same File a second time.
const validatedDocxFiles = new WeakSet<File>();
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
// it before the read rather than after. The adapters call this from add() as
// well: the composer clears its text and attachments before it awaits send(),
// so a throw there loses the typed message along with the file.
export function getDocumentAttachmentSizeError(
  file: File,
  label: "PDF" | "DOCX",
): string | null {
  return file.size > MAX_OPEN_DOCUMENT_ARCHIVE_BYTES
    ? `${label} file is too large: ${file.name}`
    : null;
}

export function assertDocumentAttachmentSize(
  file: File,
  label: "PDF" | "DOCX",
): void {
  const error = getDocumentAttachmentSizeError(file, label);
  if (error) {
    throw new Error(error);
  }
}

// mammoth's joinPath: an absolute target drops the base path.
function joinDocxPath(basePath: string, target: string): string {
  const joined = target.startsWith("/")
    ? target
    : [basePath, target].filter(Boolean).join("/");
  return joined.startsWith("/") ? joined.slice(1) : joined;
}

// XML attribute values are entity-decoded by the parser mammoth hands the
// relationships to, so a target only matches the archive once decoded.
function decodeXmlEntities(value: string): string {
  return value.replace(XML_ENTITY_RE, (match, decimal, hex, name) => {
    const code = decimal
      ? Number.parseInt(decimal, 10)
      : hex
        ? Number.parseInt(hex, 16)
        : Number.NaN;
    if (Number.isNaN(code)) {
      return XML_NAMED_ENTITIES[name] ?? match;
    }
    return code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : match;
  });
}

// The relationship parts are XML, but only their targets are needed, so they
// are scanned rather than parsed: DOMParser is not available where this also
// runs under test, and a malformed rels file is mammoth's to report. The scan
// accepts every attribute form mammoth's parser resolves (both quote styles,
// entity-encoded values, a prefixed element name, ">" inside a value), so a
// crafted rels file cannot hide a target from the bound below.
function readDocxXmlTargets(
  rels: Uint8Array | undefined,
  basePath: string,
): Map<string, string[]> {
  const targets = new Map<string, string[]>();
  if (!rels) {
    return targets;
  }
  for (const tag of strFromU8(rels).match(DOCX_RELATIONSHIP_TAG_RE) ?? []) {
    let type: string | undefined;
    let target: string | undefined;
    for (const [, name, quoted, apostrophed] of tag.matchAll(
      XML_ATTRIBUTE_RE,
    )) {
      const value = quoted ?? apostrophed ?? "";
      if (name === "Type") {
        type = decodeXmlEntities(value);
      } else if (name === "Target") {
        target = decodeXmlEntities(value);
      }
    }
    if (type && target) {
      const resolved = targets.get(type) ?? [];
      resolved.push(joinDocxPath(basePath, target));
      targets.set(type, resolved);
    }
  }
  return targets;
}

// The .rels part that names the XML parts of the part at `path`.
function docxRelationshipsPath(path: string): string {
  const cut = path.lastIndexOf("/");
  const dirname = cut === -1 ? "" : path.slice(0, cut);
  const basename = path.slice(cut + 1);
  return joinDocxPath(dirname, `_rels/${basename}.rels`);
}

// mammoth takes no entry filter, so the sizes the archive declares are checked
// first, for exactly the parts it goes on to inflate: findPartPaths resolves
// the main document out of "_rels/.rels" and the styles, numbering and note
// parts out of the document's own .rels, falling back to a fixed name when no
// target resolves. Only the relationship parts are decompressed here, and only
// after their own declared size has passed.
function assertDocxXmlSizes(filename: string, bytes: Uint8Array): void {
  const sizes = new Map<string, number>();
  const oversized: string[] = [];
  const bound = (path: string) => {
    const size = sizes.get(path);
    if (size !== undefined && size > MAX_OPEN_DOCUMENT_XML_BYTES) {
      oversized.push(path);
    }
  };
  const read = (path: string): Uint8Array | undefined => {
    const size = sizes.get(path);
    if (size === undefined || size > MAX_OPEN_DOCUMENT_XML_BYTES) {
      return undefined;
    }
    return unzipSync(bytes, { filter: (entry) => entry.name === path })[path];
  };
  // findPartPath keeps the first target that exists, and every target it
  // discards is one mammoth never opens.
  const resolve = (targets: string[] | undefined, fallback: string) =>
    targets?.find((path) => sizes.has(path)) ?? fallback;

  try {
    unzipSync(bytes, {
      filter: (entry) => {
        sizes.set(entry.name, entry.originalSize);
        return false;
      },
    });

    bound(DOCX_CONTENT_TYPES_PART);
    bound(DOCX_PACKAGE_RELATIONSHIPS);
    const mainDocument = resolve(
      readDocxXmlTargets(read(DOCX_PACKAGE_RELATIONSHIPS), "").get(
        DOCX_MAIN_DOCUMENT_TYPE,
      ),
      DOCX_MAIN_DOCUMENT_FALLBACK,
    );
    bound(mainDocument);
    const mainDocumentRels = docxRelationshipsPath(mainDocument);
    bound(mainDocumentRels);

    const cut = mainDocument.lastIndexOf("/");
    const documentTargets = readDocxXmlTargets(
      read(mainDocumentRels),
      cut === -1 ? "" : mainDocument.slice(0, cut),
    );
    for (const name of DOCX_RELATED_PART_NAMES) {
      const path = resolve(
        documentTargets.get(`${DOCX_RELATIONSHIP_NAMESPACE}${name}`),
        `word/${name}.xml`,
      );
      bound(path);
      if (DOCX_BODY_PART_NAMES.has(name)) {
        bound(docxRelationshipsPath(path));
      }
    }
  } catch {
    // A malformed archive is mammoth's to report, not the size guard's.
    if (oversized.length === 0) {
      return;
    }
  }
  if (oversized.length > 0) {
    throw new Error(`DOCX XML file is too large: ${filename}:${oversized[0]}`);
  }
}

// The composer clears its text and attachments before it awaits send(), so a
// DOCX whose parts only fail there discards the typed message along with the
// file. add() calls this instead, the way the audio adapter checks its own
// ceiling, and the verdict is cached so send() does not unzip the file again.
export async function getDocxAttachmentError(
  file: File,
): Promise<string | null> {
  const sizeError = getDocumentAttachmentSizeError(file, "DOCX");
  if (sizeError) {
    return sizeError;
  }
  try {
    assertDocxXmlSizes(file.name, new Uint8Array(await file.arrayBuffer()));
  } catch (error) {
    return error instanceof Error ? error.message : String(error);
  }
  validatedDocxFiles.add(file);
  return null;
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
  if (!validatedDocxFiles.has(file)) {
    assertDocxXmlSizes(file.name, new Uint8Array(arrayBuffer));
    validatedDocxFiles.add(file);
  }
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
