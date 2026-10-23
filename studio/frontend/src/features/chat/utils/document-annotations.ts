// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type DocumentAnnotation = { quote: string; request: string };

/** A file's annotations, or a web page's: `file` is then the page's title and `url` its address. */
export type DocumentAnnotations = { file: string; url?: string; items: DocumentAnnotation[] };

const TAG = "document_annotations";
const MIME = "text/plain";

// Like pasted text: the File identity marks it while it sits in the composer.
const annotationsByFile = new WeakMap<File, DocumentAnnotations>();

export function createAnnotationsFile(annotations: DocumentAnnotations): File {
  const text = annotationsContentText(annotations);
  const file = new File([text], "annotations.txt", {
    type: MIME,
    lastModified: Date.now(),
  });
  annotationsByFile.set(file, annotations);
  return file;
}

export function annotationsOfFile(
  file: File | undefined,
): DocumentAnnotations | undefined {
  return file === undefined ? undefined : annotationsByFile.get(file);
}

// One line with no quotes or tags: a page picks its title, and must not end the block early.
const quoteAttr = (value: string) => value.replace(/[\r\n"<>]/g, " ");

/** What the model reads: JSON items, so the chip can parse them back from a stored message. */
export function annotationsContentText({
  file,
  url,
  items,
}: DocumentAnnotations): string {
  return [
    url
      ? `<${TAG} file="${quoteAttr(file)}" url="${quoteAttr(url)}">`
      : `<${TAG} file="${quoteAttr(file)}">`,
    url
      ? `The user selected parts of the web page ${quoteAttr(file)} (${quoteAttr(url)}) and asked a question or made a request about each. Answer each request about its selection.`
      : `The user selected parts of ${quoteAttr(file)} and asked for a change or a question on each. Apply each request to its selection.`,
    JSON.stringify(
      items.map((item) => ({ selection: item.quote, request: item.request })),
      null,
      2,
    ),
    `</${TAG}>`,
  ].join("\n");
}

export function isAnnotationsContent(text: string | undefined): boolean {
  return text?.startsWith(`<${TAG} file="`) === true;
}

export function parseAnnotationsContent(
  text: string | undefined,
): DocumentAnnotations | null {
  if (!text || !isAnnotationsContent(text)) return null;
  // The header is the first line: its attributes never hold a newline, but a page's address can be long.
  const header = /^<document_annotations file="([^"]*)"(?: url="([^"]*)")?>/.exec(
    text.split("\n", 1)[0],
  );
  const file = header?.[1] ?? "";
  const source = header?.[2] ? { file, url: header[2] } : { file };
  const start = text.indexOf("\n[");
  const end = text.lastIndexOf("]");
  if (start === -1 || end < start) return { ...source, items: [] };
  try {
    const parsed: unknown = JSON.parse(text.slice(start + 1, end + 1));
    if (!Array.isArray(parsed)) return { ...source, items: [] };
    const items = parsed.flatMap((entry) => {
      if (typeof entry !== "object" || entry === null) return [];
      const { selection, request } = entry as {
        selection?: unknown;
        request?: unknown;
      };
      return typeof selection === "string" && typeof request === "string"
        ? [{ quote: selection, request }]
        : [];
    });
    return { ...source, items };
  } catch {
    return { ...source, items: [] };
  }
}
