// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A part of a document the user marked in the browser, and what they asked for there. */
export type DocumentAnnotation = { quote: string; request: string };

export type DocumentAnnotations = { file: string; items: DocumentAnnotation[] };

const TAG = "document_annotations";
const MIME = "text/plain";
// The header, file name included, always fits in this; parsing never reads the whole body to decide.
const HEADER_SCAN_CHARS = 1024;

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

const quoteAttr = (value: string) => value.replace(/[\n"]/g, " ");

/** What the model reads: JSON items, so the chip can parse them back from a stored message. */
export function annotationsContentText({
  file,
  items,
}: DocumentAnnotations): string {
  return [
    `<${TAG} file="${quoteAttr(file)}">`,
    `The user selected parts of ${file} and asked for a change or a question on each. Apply each request to its selection.`,
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

/** A stored message's annotations, or null when the text is not one. */
export function parseAnnotationsContent(
  text: string | undefined,
): DocumentAnnotations | null {
  if (!text || !isAnnotationsContent(text)) return null;
  const file =
    /^<document_annotations file="([^"]*)">/.exec(
      text.slice(0, HEADER_SCAN_CHARS),
    )?.[1] ?? "";
  const start = text.indexOf("\n[");
  const end = text.lastIndexOf("]");
  if (start === -1 || end < start) return { file, items: [] };
  try {
    const parsed: unknown = JSON.parse(text.slice(start + 1, end + 1));
    if (!Array.isArray(parsed)) return { file, items: [] };
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
    return { file, items };
  } catch {
    return { file, items: [] };
  }
}
