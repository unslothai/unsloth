// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { strFromU8, unzipSync } from "fflate";
import { OPEN_DOCUMENT_SPREADSHEET_MIME } from "./open-document-accept";
const OFFICE_NAMESPACE = "urn:oasis:names:tc:opendocument:xmlns:office:1.0";
const TABLE_NAMESPACE = "urn:oasis:names:tc:opendocument:xmlns:table:1.0";
const TEXT_NAMESPACE = "urn:oasis:names:tc:opendocument:xmlns:text:1.0";
const OPEN_DOCUMENT_CELL_VALUE_ATTRIBUTES = [
  "string-value",
  "value",
  "boolean-value",
  "date-value",
  "time-value",
] as const;
const OPEN_DOCUMENT_TEXT_BLOCK_NAMES = ["h", "p"] as const;
export const MAX_OPEN_DOCUMENT_ARCHIVE_BYTES = 50 * 1024 * 1024;
export const MAX_OPEN_DOCUMENT_XML_BYTES = 10 * 1024 * 1024;
const MAX_REPEATED_OPEN_DOCUMENT_ROWS = 100;
const MAX_REPEATED_OPEN_DOCUMENT_COLUMNS = 100;

export type OpenDocumentAttachmentContent = {
  label: "ODS" | "ODT";
  text: string;
};

type OpenDocumentHiddenState = "hidden" | "visible" | "unset";

export async function readOpenDocumentAttachmentContent(
  file: File,
  filename: string,
  contentType: string,
): Promise<OpenDocumentAttachmentContent> {
  const doc = parseOpenDocumentXml(
    await readOpenDocumentContentXml(file),
    filename,
  );
  const isSpreadsheet =
    contentType === OPEN_DOCUMENT_SPREADSHEET_MIME ||
    filename.toLowerCase().endsWith(".ods");

  return {
    label: isSpreadsheet ? "ODS" : "ODT",
    text: isSpreadsheet
      ? extractOpenDocumentSpreadsheetText(doc)
      : extractOpenDocumentText(doc),
  };
}

export async function readActiveOpenDocumentAttachmentContent(
  file: File,
  filename: string,
  contentType: string,
  isActive: () => boolean,
): Promise<OpenDocumentAttachmentContent | null> {
  try {
    const content = await readOpenDocumentAttachmentContent(
      file,
      filename,
      contentType,
    );
    return isActive() ? content : null;
  } catch (error) {
    if (!isActive()) {
      return null;
    }
    throw error;
  }
}

async function readOpenDocumentContentXml(file: File): Promise<string> {
  assertOpenDocumentArchiveSize(file);

  let files: Record<string, Uint8Array>;
  try {
    files = unzipSync(new Uint8Array(await file.arrayBuffer()), {
      filter: (entry) => {
        const shouldRead = entry.name === "content.xml";
        if (shouldRead) {
          assertOpenDocumentXmlSize(file.name, entry.name, entry.originalSize);
        }
        return shouldRead;
      },
    });
  } catch (error) {
    if (isOpenDocumentSizeError(error)) {
      throw error;
    }
    throw new Error(`Failed to read OpenDocument archive: ${file.name}`, {
      cause: error,
    });
  }

  const content = files["content.xml"];
  if (!content) {
    throw new Error(`OpenDocument file is missing content.xml: ${file.name}`);
  }

  assertOpenDocumentXmlSize(file.name, "content.xml", content.length);
  return strFromU8(content);
}

function assertOpenDocumentArchiveSize(file: File): void {
  if (file.size > MAX_OPEN_DOCUMENT_ARCHIVE_BYTES) {
    throw new Error(`OpenDocument archive is too large: ${file.name}`);
  }
}

function assertOpenDocumentXmlSize(
  filename: string,
  entryName: string,
  bytes: number,
): void {
  if (bytes > MAX_OPEN_DOCUMENT_XML_BYTES) {
    throw new Error(
      `OpenDocument XML file is too large: ${filename}:${entryName}`,
    );
  }
}

function isOpenDocumentSizeError(error: unknown): boolean {
  return (
    error instanceof Error &&
    (error.message.startsWith("OpenDocument archive is too large:") ||
      error.message.startsWith("OpenDocument XML file is too large:"))
  );
}

function parseOpenDocumentXml(xml: string, filename: string): XMLDocument {
  const doc = new DOMParser().parseFromString(xml, "application/xml");
  if (doc.getElementsByTagName("parsererror").length > 0) {
    throw new Error(`Failed to parse OpenDocument content.xml: ${filename}`);
  }
  return doc;
}

function extractOpenDocumentText(doc: XMLDocument): string {
  const body =
    doc.getElementsByTagNameNS(OFFICE_NAMESPACE, "body")[0] ??
    doc.documentElement;
  const blocks = collectVisibleOpenDocumentTextBlocks(body);

  return blocks
    .map((block) =>
      normalizeOpenDocumentText(extractOpenDocumentInlineText(block)),
    )
    .filter(Boolean)
    .join("\n\n");
}

function extractOpenDocumentSpreadsheetText(doc: XMLDocument): string {
  const body =
    doc.getElementsByTagNameNS(OFFICE_NAMESPACE, "body")[0] ??
    doc.documentElement;
  const tables = getOpenDocumentChildElements(body, OFFICE_NAMESPACE, [
    "spreadsheet",
  ]).flatMap((spreadsheet) =>
    getOpenDocumentChildElements(spreadsheet, TABLE_NAMESPACE, ["table"]),
  );

  return tables.map(extractOpenDocumentTableText).filter(Boolean).join("\n\n");
}

function extractOpenDocumentTableText(table: Element): string {
  const rows = collectOpenDocumentTableRows(table).flatMap(
    extractOpenDocumentRowText,
  );

  if (rows.length === 0) {
    return "";
  }

  const name = getOpenDocumentAttribute(table, TABLE_NAMESPACE, "name");
  return name ? `[Sheet: ${name}]\n${rows.join("\n")}` : rows.join("\n");
}

function extractOpenDocumentRowText(row: Element): string[] {
  const cells = getOpenDocumentChildElements(row, TABLE_NAMESPACE, [
    "table-cell",
    "covered-table-cell",
  ]);
  const rowCells = cells.flatMap((cell) =>
    repeatOpenDocumentValue(
      cell.localName === "covered-table-cell"
        ? ""
        : extractOpenDocumentCellText(cell),
      getOpenDocumentRepeatCount(
        cell,
        "number-columns-repeated",
        MAX_REPEATED_OPEN_DOCUMENT_COLUMNS,
      ),
    ),
  );

  const line = rowCells.join("\t").replace(/\t+$/g, "");

  if (!line.trim()) {
    return [];
  }

  return repeatOpenDocumentValue(
    line,
    getOpenDocumentRepeatCount(
      row,
      "number-rows-repeated",
      MAX_REPEATED_OPEN_DOCUMENT_ROWS,
    ),
  );
}

function repeatOpenDocumentValue<T>(value: T, count: number): T[] {
  return Array.from({ length: count }, () => value);
}

function extractOpenDocumentCellText(cell: Element): string {
  const blocks = collectVisibleOpenDocumentTextBlocks(cell);
  const text = blocks
    .map((block) =>
      normalizeOpenDocumentText(extractOpenDocumentInlineText(block)),
    )
    .filter(Boolean)
    .join("\n");

  if (text) {
    return text;
  }

  return getOpenDocumentCellValueText(cell);
}

function getOpenDocumentCellValueText(cell: Element): string {
  for (const attributeName of OPEN_DOCUMENT_CELL_VALUE_ATTRIBUTES) {
    const value = getOpenDocumentAttribute(
      cell,
      OFFICE_NAMESPACE,
      attributeName,
    );
    if (value !== null) {
      return value;
    }
  }

  return "";
}

function extractOpenDocumentInlineText(node: Node): string {
  if (node.nodeType === Node.TEXT_NODE) {
    return node.nodeValue ?? "";
  }
  if (node.nodeType !== Node.ELEMENT_NODE) {
    return "";
  }

  const element = node as Element;
  if (isHiddenOpenDocumentElement(element)) {
    return "";
  }

  if (element.namespaceURI === TEXT_NAMESPACE) {
    if (element.localName === "hidden-text") {
      return (
        getOpenDocumentAttribute(element, TEXT_NAMESPACE, "string-value") ??
        Array.from(element.childNodes)
          .map(extractOpenDocumentInlineText)
          .join("")
      );
    }
    if (element.localName === "tab") {
      return "\t";
    }
    if (element.localName === "line-break") {
      return "\n";
    }
    if (element.localName === "s") {
      return " ".repeat(
        getOpenDocumentRepeatCount(
          element,
          "c",
          MAX_REPEATED_OPEN_DOCUMENT_COLUMNS,
          TEXT_NAMESPACE,
        ),
      );
    }
  }

  return Array.from(element.childNodes)
    .map(extractOpenDocumentInlineText)
    .join("");
}

function normalizeOpenDocumentText(text: string): string {
  return text.replace(/[^\S\r\n\t]+/g, " ").trim();
}

function collectVisibleOpenDocumentTextBlocks(root: Element): Element[] {
  const matches: Element[] = [];

  for (const child of getOpenDocumentChildElementNodes(root)) {
    if (isHiddenOpenDocumentElement(child)) {
      continue;
    }

    if (
      child.namespaceURI === TEXT_NAMESPACE &&
      OPEN_DOCUMENT_TEXT_BLOCK_NAMES.includes(
        child.localName as (typeof OPEN_DOCUMENT_TEXT_BLOCK_NAMES)[number],
      ) &&
      !isOpenDocumentParagraphHidden(child)
    ) {
      matches.push(child);
    } else {
      matches.push(...collectVisibleOpenDocumentTextBlocks(child));
    }
  }

  return matches;
}

function isHiddenOpenDocumentElement(element: Element): boolean {
  if (element.namespaceURI === TABLE_NAMESPACE) {
    const visibility = getOpenDocumentAttribute(
      element,
      TABLE_NAMESPACE,
      "visibility",
    );
    return (
      visibility === "collapse" ||
      visibility === "filter" ||
      ((element.localName === "table" ||
        element.localName === "table-row-group" ||
        element.localName === "table-column-group") &&
        getOpenDocumentAttribute(element, TABLE_NAMESPACE, "display") ===
          "false")
    );
  }

  if (element.namespaceURI === OFFICE_NAMESPACE) {
    return (
      element.localName === "annotation" || element.localName === "change-info"
    );
  }

  if (element.namespaceURI === TEXT_NAMESPACE) {
    return (
      (element.localName === "section" &&
        isHiddenOpenDocumentSection(element)) ||
      (element.localName === "hidden-text" &&
        getOpenDocumentHiddenState(element) === "hidden") ||
      (element.localName === "hidden-paragraph" &&
        getOpenDocumentHiddenState(element) === "hidden") ||
      element.localName === "tracked-changes" ||
      element.localName === "changed-region" ||
      element.localName === "deletion" ||
      element.localName === "insertion" ||
      element.localName === "format-change"
    );
  }

  return false;
}

function isHiddenOpenDocumentSection(element: Element): boolean {
  const display = getOpenDocumentAttribute(element, TEXT_NAMESPACE, "display");
  return (
    display === "none" ||
    (display === "condition" &&
      getOpenDocumentAttribute(element, TEXT_NAMESPACE, "condition") !== null)
  );
}

function isOpenDocumentParagraphHidden(element: Element): boolean {
  const visibility = getOpenDocumentParagraphVisibility(element);
  return visibility.hidden && !visibility.visible;
}

function getOpenDocumentParagraphVisibility(element: Element): {
  hidden: boolean;
  visible: boolean;
} {
  let hidden = false;
  let visible = false;

  for (const child of getOpenDocumentChildElementNodes(element)) {
    const isHiddenParagraph =
      child.namespaceURI === TEXT_NAMESPACE &&
      child.localName === "hidden-paragraph";
    if (isHiddenParagraph) {
      const hiddenState = getOpenDocumentHiddenState(child);
      hidden ||= hiddenState === "hidden";
      visible ||= hiddenState === "visible";
    }
    if (isHiddenParagraph || isHiddenOpenDocumentElement(child)) {
      continue;
    }

    const childVisibility = getOpenDocumentParagraphVisibility(child);
    hidden ||= childVisibility.hidden;
    visible ||= childVisibility.visible;
  }

  return { hidden, visible };
}

function getOpenDocumentHiddenState(element: Element): OpenDocumentHiddenState {
  const isHidden = getOpenDocumentAttribute(
    element,
    TEXT_NAMESPACE,
    "is-hidden",
  );
  if (isHidden === "true") {
    return "hidden";
  }
  if (isHidden === "false") {
    return "visible";
  }
  return getOpenDocumentAttribute(element, TEXT_NAMESPACE, "condition") !== null
    ? "hidden"
    : "unset";
}

function collectOpenDocumentTableRows(root: Element): Element[] {
  const rows: Element[] = [];

  for (const child of getOpenDocumentChildElementNodes(root)) {
    if (
      child.namespaceURI === TABLE_NAMESPACE &&
      child.localName === "table-row"
    ) {
      rows.push(child);
    } else if (
      child.namespaceURI !== TABLE_NAMESPACE ||
      ["table-row-group", "table-rows", "table-header-rows"].includes(
        child.localName,
      )
    ) {
      rows.push(...collectOpenDocumentTableRows(child));
    }
  }

  return rows;
}

function getOpenDocumentChildElements(
  root: Element,
  namespaceUri: string,
  localNames: string[],
): Element[] {
  return getOpenDocumentChildElementNodes(root).filter(
    (child) =>
      child.namespaceURI === namespaceUri &&
      localNames.includes(child.localName),
  );
}

function getOpenDocumentChildElementNodes(root: Element): Element[] {
  return Array.from(root.childNodes).filter(
    (child): child is Element => child.nodeType === Node.ELEMENT_NODE,
  );
}

function getOpenDocumentRepeatCount(
  element: Element,
  name: string,
  max: number,
  namespaceUri = TABLE_NAMESPACE,
): number {
  const value = getOpenDocumentAttribute(element, namespaceUri, name);
  if (!value) {
    return 1;
  }

  const count = Number.parseInt(value, 10);
  if (!Number.isFinite(count) || count < 1) {
    return 1;
  }
  return Math.min(count, max);
}

function getOpenDocumentAttribute(
  element: Element,
  namespaceUri: string,
  name: string,
): string | null {
  const value = element.getAttributeNS(namespaceUri, name);
  return value === "" && !element.hasAttributeNS(namespaceUri, name)
    ? null
    : value;
}
