// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { strFromU8, unzipSync } from "fflate";
import { OPEN_DOCUMENT_SPREADSHEET_MIME } from "./open-document-accept";
const OFFICE_NAMESPACE = "urn:oasis:names:tc:opendocument:xmlns:office:1.0";
const STYLE_NAMESPACE = "urn:oasis:names:tc:opendocument:xmlns:style:1.0";
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
const MAX_OPEN_DOCUMENT_COLUMN_INDEX = Number.MAX_SAFE_INTEGER;

export type OpenDocumentAttachmentContent = {
  label: "ODS" | "ODT";
  text: string;
};

type HiddenOpenDocumentColumnRange = {
  start: number;
  end: number;
};

type HiddenOpenDocumentColumnRanges = {
  ranges: HiddenOpenDocumentColumnRange[];
  nextColumn: number;
};

type OpenDocumentXmlFiles = {
  contentXml: string;
  stylesXml?: string;
};

type OpenDocumentHiddenState = "hidden" | "visible" | "unset";

export async function readOpenDocumentAttachmentContent(
  file: File,
  filename: string,
  contentType: string,
): Promise<OpenDocumentAttachmentContent> {
  const { contentXml, stylesXml } = await readOpenDocumentXmlFiles(file);
  const doc = parseOpenDocumentXml(contentXml, filename);
  const stylesDoc = stylesXml
    ? parseOpenDocumentXml(stylesXml, `${filename}:styles.xml`)
    : undefined;
  const isSpreadsheet =
    contentType === OPEN_DOCUMENT_SPREADSHEET_MIME ||
    filename.toLowerCase().endsWith(".ods");

  return {
    label: isSpreadsheet ? "ODS" : "ODT",
    text: isSpreadsheet
      ? extractOpenDocumentSpreadsheetText(doc, stylesDoc)
      : extractOpenDocumentText(doc),
  };
}

async function readOpenDocumentXmlFiles(
  file: File,
): Promise<OpenDocumentXmlFiles> {
  assertOpenDocumentArchiveSize(file);

  let files: Record<string, Uint8Array>;
  try {
    files = unzipSync(new Uint8Array(await file.arrayBuffer()), {
      filter: (entry) => {
        const shouldRead =
          entry.name === "content.xml" || entry.name === "styles.xml";
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

  const styles = files["styles.xml"];
  assertOpenDocumentXmlSize(file.name, "content.xml", content.length);
  if (styles) {
    assertOpenDocumentXmlSize(file.name, "styles.xml", styles.length);
  }
  return {
    contentXml: strFromU8(content),
    stylesXml: styles ? strFromU8(styles) : undefined,
  };
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

function extractOpenDocumentSpreadsheetText(
  doc: XMLDocument,
  stylesDoc?: XMLDocument,
): string {
  const hiddenTableStyles = collectHiddenOpenDocumentTableStyles(
    doc,
    stylesDoc,
  );
  const body =
    doc.getElementsByTagNameNS(OFFICE_NAMESPACE, "body")[0] ??
    doc.documentElement;
  const tables = getOpenDocumentChildElements(body, OFFICE_NAMESPACE, [
    "spreadsheet",
  ])
    .flatMap((spreadsheet) =>
      getOpenDocumentChildElements(spreadsheet, TABLE_NAMESPACE, ["table"]),
    )
    .filter(
      (table) =>
        !isHiddenOpenDocumentElement(table) &&
        !hasHiddenOpenDocumentTableStyle(table, hiddenTableStyles),
    );

  return tables.map(extractOpenDocumentTableText).filter(Boolean).join("\n\n");
}

function extractOpenDocumentTableText(table: Element): string {
  const hiddenColumns = collectHiddenOpenDocumentColumns(table).ranges;
  const rows = collectOpenDocumentTableRows(table).flatMap((row) =>
    extractOpenDocumentRowText(row, hiddenColumns),
  );

  if (rows.length === 0) {
    return "";
  }

  const name = getOpenDocumentAttribute(table, TABLE_NAMESPACE, "name");
  return name ? `[Sheet: ${name}]\n${rows.join("\n")}` : rows.join("\n");
}

function extractOpenDocumentRowText(
  row: Element,
  hiddenColumns: HiddenOpenDocumentColumnRange[],
): string[] {
  const cells = getOpenDocumentChildElements(row, TABLE_NAMESPACE, [
    "table-cell",
    "covered-table-cell",
  ]);
  const rowCells: string[] = [];
  let columnIndex = 0;

  for (const cell of cells) {
    const isCoveredCell = cell.localName === "covered-table-cell";
    const repeat = getOpenDocumentRepeatCount(
      cell,
      "number-columns-repeated",
      MAX_OPEN_DOCUMENT_COLUMN_INDEX,
    );
    appendOpenDocumentVisibleCells(
      rowCells,
      hiddenColumns,
      columnIndex,
      repeat,
      isCoveredCell ? "" : extractOpenDocumentCellText(cell),
    );
    columnIndex = advanceOpenDocumentColumnIndex(columnIndex, repeat);
  }

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

function appendOpenDocumentVisibleCells(
  rowCells: string[],
  hiddenColumns: HiddenOpenDocumentColumnRange[],
  columnIndex: number,
  repeat: number,
  text: string,
): void {
  let emitted = 0;
  for (
    let i = 0;
    i < repeat && emitted < MAX_REPEATED_OPEN_DOCUMENT_COLUMNS;
    i++
  ) {
    const hiddenEnd = getHiddenOpenDocumentColumnEnd(
      hiddenColumns,
      columnIndex + i,
    );
    if (hiddenEnd === null) {
      rowCells.push(text);
      emitted++;
    } else {
      i += hiddenEnd - columnIndex - i - 1;
    }
  }
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
    if (isHiddenOpenDocumentElement(child)) {
      continue;
    }
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

function collectHiddenOpenDocumentColumns(
  root: Element,
  hidden = false,
  startColumn = 0,
): HiddenOpenDocumentColumnRanges {
  const ranges: HiddenOpenDocumentColumnRange[] = [];
  let column = startColumn;

  for (const child of getOpenDocumentChildElementNodes(root)) {
    if (child.namespaceURI !== TABLE_NAMESPACE) {
      continue;
    }

    const childHidden = hidden || isHiddenOpenDocumentElement(child);
    if (child.localName === "table-column") {
      const repeat = getOpenDocumentRepeatCount(
        child,
        "number-columns-repeated",
        MAX_OPEN_DOCUMENT_COLUMN_INDEX,
      );
      const nextColumn = advanceOpenDocumentColumnIndex(column, repeat);
      if (childHidden) {
        ranges.push({ start: column, end: nextColumn });
      }
      column = nextColumn;
    } else if (
      ["table-column-group", "table-columns", "table-header-columns"].includes(
        child.localName,
      )
    ) {
      const childRanges = collectHiddenOpenDocumentColumns(
        child,
        childHidden,
        column,
      );
      ranges.push(...childRanges.ranges);
      column = childRanges.nextColumn;
    }
  }

  return { ranges, nextColumn: column };
}

function getHiddenOpenDocumentColumnEnd(
  ranges: HiddenOpenDocumentColumnRange[],
  column: number,
): number | null {
  for (const range of ranges) {
    if (column < range.start) {
      return null;
    }
    if (column < range.end) {
      return range.end;
    }
  }

  return null;
}

function advanceOpenDocumentColumnIndex(
  column: number,
  repeat: number,
): number {
  return Math.min(column + repeat, MAX_OPEN_DOCUMENT_COLUMN_INDEX);
}

function collectOpenDocumentElements(
  root: Element,
  namespaceUri: string,
  localNames: string[],
): Element[] {
  const matches: Element[] = [];

  for (const child of getOpenDocumentChildElementNodes(root)) {
    if (isHiddenOpenDocumentElement(child)) {
      continue;
    }

    if (
      child.namespaceURI === namespaceUri &&
      localNames.includes(child.localName)
    ) {
      matches.push(child);
    } else {
      matches.push(
        ...collectOpenDocumentElements(child, namespaceUri, localNames),
      );
    }
  }

  return matches;
}

function collectHiddenOpenDocumentTableStyles(
  doc: XMLDocument,
  stylesDoc?: XMLDocument,
): Set<string> {
  const hidden = new Set<string>();
  const styles = [
    ...collectOpenDocumentElements(doc.documentElement, STYLE_NAMESPACE, [
      "style",
    ]),
    ...(stylesDoc
      ? collectOpenDocumentElements(
          stylesDoc.documentElement,
          STYLE_NAMESPACE,
          ["style"],
        )
      : []),
  ];

  for (const style of styles) {
    const name = getOpenDocumentAttribute(style, STYLE_NAMESPACE, "name");
    if (
      !name ||
      getOpenDocumentAttribute(style, STYLE_NAMESPACE, "family") !== "table"
    ) {
      continue;
    }

    const hidesTable =
      getOpenDocumentAttribute(style, TABLE_NAMESPACE, "display") === "false" ||
      getOpenDocumentChildElements(style, STYLE_NAMESPACE, [
        "table-properties",
      ]).some(
        (properties) =>
          getOpenDocumentAttribute(properties, TABLE_NAMESPACE, "display") ===
          "false",
      );
    if (hidesTable) {
      hidden.add(name);
    }
  }

  return hidden;
}

function hasHiddenOpenDocumentTableStyle(
  table: Element,
  hiddenTableStyles: Set<string>,
): boolean {
  const styleName = getOpenDocumentAttribute(
    table,
    TABLE_NAMESPACE,
    "style-name",
  );
  return styleName !== null && hiddenTableStyles.has(styleName);
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

// Elements are matched by local name: strict OOXML moves every part to other namespaces.
const ELEMENT_NODE = 1;
const MAX_UNPACKED_BYTES = 2 * MAX_OPEN_DOCUMENT_ARCHIVE_BYTES;
const MAX_TEXT_LENGTH = MAX_OPEN_DOCUMENT_XML_BYTES;
// A sparse row whose only cell sits at column XFD would otherwise become 16,383 tabs.
const MAX_SPREADSHEET_COLUMNS = 1024;
const BUILTIN_TIME_FORMAT_IDS = new Set([
  18, 19, 20, 21, 32, 33, 34, 35, 45, 47, 55, 56,
]);
const BUILTIN_DATE_FORMAT_IDS = new Set([
  14, 15, 16, 17, 27, 28, 29, 30, 31, 36, 50, 51, 52, 53, 54, 57, 58,
]);
const MAX_DATE_SERIAL = 2_958_465;

export type OfficeOpenXmlAttachmentContent = {
  label: "XLSX" | "PPTX";
  text: string;
};

type Relationship = { type: string; target: string };

type DateFormatKind = "date" | "time" | "datetime" | "duration" | null;

class OfficeOpenXmlSizeError extends Error {}

export async function readOfficeOpenXmlAttachmentContent(
  file: File,
  filename: string,
): Promise<OfficeOpenXmlAttachmentContent> {
  const parts = await readPackageParts(file);
  const main = [...packageRelationships(parts, "").values()].find((rel) =>
    rel.type.endsWith("/officeDocument"),
  );
  const mainXml = main && parts.get(main.target);
  if (!main || !mainXml) {
    throw new Error(`Office file has no main document: ${filename}`);
  }
  const root = parseXml(mainXml, filename).documentElement;
  if (root.localName === "workbook") {
    return {
      label: "XLSX",
      text: extractWorkbookText(parts, main.target, root, filename),
    };
  }
  if (root.localName === "presentation") {
    return {
      label: "PPTX",
      text: extractPresentationText(parts, main.target, root, filename),
    };
  }
  throw new Error(`Unsupported Office document: ${filename}`);
}

async function readPackageParts(file: File): Promise<Map<string, string>> {
  if (file.size > MAX_OPEN_DOCUMENT_ARCHIVE_BYTES) {
    throw new OfficeOpenXmlSizeError(`Office file is too large: ${file.name}`);
  }
  let unpacked = 0;
  let files: Record<string, Uint8Array>;
  try {
    files = unzipSync(new Uint8Array(await file.arrayBuffer()), {
      filter: (entry) => {
        if (!/\.(xml|rels)$/i.test(entry.name)) {
          return false;
        }
        // fflate uses the declared size, or the real one for a stored entry, so charge the larger.
        const bytes = Math.max(entry.size, entry.originalSize);
        unpacked += bytes;
        if (
          bytes > MAX_OPEN_DOCUMENT_XML_BYTES ||
          unpacked > MAX_UNPACKED_BYTES
        ) {
          throw new OfficeOpenXmlSizeError(
            `Office file unpacks too large: ${file.name}`,
          );
        }
        return true;
      },
    });
  } catch (error) {
    if (error instanceof OfficeOpenXmlSizeError) {
      throw error;
    }
    throw new Error(`Failed to read Office archive: ${file.name}`, {
      cause: error,
    });
  }
  return new Map(
    Object.entries(files).map(([name, bytes]) => [name, strFromU8(bytes)]),
  );
}

function packageRelationships(
  parts: Map<string, string>,
  partPath: string,
): Map<string, Relationship> {
  const slash = partPath.lastIndexOf("/");
  const dir = partPath.slice(0, slash + 1);
  const xml = parts.get(`${dir}_rels/${partPath.slice(slash + 1)}.rels`);
  const relationships = new Map<string, Relationship>();
  if (!xml) {
    return relationships;
  }
  for (const rel of descendants(parseXml(xml, partPath).documentElement, [
    "Relationship",
  ])) {
    const id = attribute(rel, "Id");
    const target = attribute(rel, "Target");
    if (id && target && attribute(rel, "TargetMode") !== "External") {
      relationships.set(id, {
        type: attribute(rel, "Type") ?? "",
        target: resolvePartPath(dir, target),
      });
    }
  }
  return relationships;
}

function resolvePartPath(dir: string, target: string): string {
  let decoded = target;
  try {
    decoded = decodeURIComponent(target);
  } catch {
    // An unescaped "%" is a literal part name character.
  }
  const segments: string[] = [];
  const joined = decoded.startsWith("/") ? decoded : `${dir}${decoded}`;
  for (const segment of joined.split("/")) {
    if (segment === "..") {
      segments.pop();
    } else if (segment && segment !== ".") {
      segments.push(segment);
    }
  }
  return segments.join("/");
}

function relatedPart(
  parts: Map<string, string>,
  relationships: Map<string, Relationship>,
  typeSuffix: string,
  filename: string,
): Element | null {
  const rel = [...relationships.values()].find((candidate) =>
    candidate.type.endsWith(typeSuffix),
  );
  const xml = rel && parts.get(rel.target);
  return xml ? parseXml(xml, filename).documentElement : null;
}

function extractWorkbookText(
  parts: Map<string, string>,
  workbookPath: string,
  workbook: Element,
  filename: string,
): string {
  const relationships = packageRelationships(parts, workbookPath);
  const sharedStringsRoot = relatedPart(
    parts,
    relationships,
    "/sharedStrings",
    filename,
  );
  const sharedStrings = sharedStringsRoot
    ? children(sharedStringsRoot, "si").map(richText)
    : [];
  const stylesRoot = relatedPart(parts, relationships, "/styles", filename);
  const dateStyles = stylesRoot ? collectDateFormatKinds(stylesRoot) : [];
  const workbookProperties = descendants(workbook, ["workbookPr"])[0];
  const date1904 = ["1", "true"].includes(
    (workbookProperties && attribute(workbookProperties, "date1904")) ?? "",
  );
  const cells: CellContext = { sharedStrings, dateStyles, date1904 };

  const budget = { remaining: MAX_TEXT_LENGTH };
  const sheets: string[] = [];
  for (const sheet of descendants(workbook, ["sheet"])) {
    const state = attribute(sheet, "state");
    const target = relationships.get(relationshipId(sheet))?.target;
    const xml = target && parts.get(target);
    if (state === "hidden" || state === "veryHidden" || !xml) {
      continue;
    }
    const rows = extractSheetRows(
      parseXml(xml, filename).documentElement,
      cells,
      budget,
    );
    if (rows.length > 0) {
      sheets.push(
        `[Sheet: ${attribute(sheet, "name") ?? ""}]\n${rows.join("\n")}`,
      );
    }
    if (budget.remaining <= 0) {
      sheets.push(
        "[Truncated: the workbook has more text than one attachment carries]",
      );
      break;
    }
  }
  return sheets.join("\n\n");
}

function extractPresentationText(
  parts: Map<string, string>,
  presentationPath: string,
  presentation: Element,
  filename: string,
): string {
  const relationships = packageRelationships(parts, presentationPath);
  const budget = { remaining: MAX_TEXT_LENGTH };
  const slides: string[] = [];
  // Numbered by position in the deck, hidden slides included, so [Slide N] matches what PowerPoint shows.
  for (const [index, slide] of descendants(presentation, ["sldId"]).entries()) {
    const target = relationships.get(relationshipId(slide))?.target;
    const xml = target && parts.get(target);
    if (!xml) {
      continue;
    }
    const root = parseXml(xml, filename).documentElement;
    const show = attribute(root, "show");
    if (show !== null && !isTrue(show)) {
      continue;
    }
    const lines = shapeTreeLines(root, budget);
    if (lines.length > 0) {
      slides.push(`[Slide ${index + 1}]\n${lines.join("\n")}`);
    }
    if (budget.remaining <= 0) {
      slides.push(
        "[Truncated: the presentation has more text than one attachment carries]",
      );
      break;
    }
  }
  return slides.join("\n\n");
}

/** Text frames and tables in document order, skipping hidden shapes and the Fallback copy of mc:AlternateContent. */
function shapeTreeLines(
  root: Element,
  budget: { remaining: number },
): string[] {
  const lines: string[] = [];
  const push = (line: string) => {
    if (line.trim() && budget.remaining > 0) {
      lines.push(line.slice(0, budget.remaining));
      budget.remaining -= line.length + 1;
    }
  };
  const stack = [root];
  while (stack.length > 0) {
    const element = stack.pop() as Element;
    if (element.localName === "txBody") {
      children(element, "p").forEach((paragraph) =>
        push(paragraphText(paragraph)),
      );
    } else if (element.localName === "tbl") {
      for (const row of children(element, "tr")) {
        push(
          children(row, "tc")
            .map((cell) =>
              descendants(cell, ["p"]).map(paragraphText).join(" ").trim(),
            )
            .join("\t"),
        );
      }
    } else if (element.localName !== "Fallback" && !isHiddenShape(element)) {
      const nested = elementChildren(element);
      for (let index = nested.length - 1; index >= 0; index--) {
        stack.push(nested[index]);
      }
    }
  }
  return lines;
}

function isHiddenShape(element: Element): boolean {
  const properties = elementChildren(element).find((child) =>
    child.localName.startsWith("nv"),
  );
  const shape = properties && children(properties, "cNvPr")[0];
  return Boolean(shape && isTrue(attribute(shape, "hidden")));
}

function paragraphText(paragraph: Element): string {
  let text = "";
  for (const child of elementChildren(paragraph)) {
    if (child.localName === "t") {
      text += child.textContent ?? "";
    } else if (child.localName === "br") {
      text += "\n";
    } else if (child.localName !== "Fallback") {
      text += paragraphText(child);
    }
  }
  return text;
}

type CellContext = {
  sharedStrings: string[];
  dateStyles: DateFormatKind[];
  date1904: boolean;
};

function extractSheetRows(
  sheet: Element,
  cells: CellContext,
  budget: { remaining: number },
): string[] {
  const hiddenColumns = new Set<number>();
  for (const column of descendants(sheet, ["col"])) {
    if (!isTrue(attribute(column, "hidden"))) {
      continue;
    }
    const min = Math.max(1, Number(attribute(column, "min")));
    const max = Math.min(
      Number(attribute(column, "max")),
      MAX_SPREADSHEET_COLUMNS,
    );
    for (let index = min; index <= max; index++) {
      hiddenColumns.add(index);
    }
  }

  const lines: string[] = [];
  for (const row of descendants(sheet, ["row"])) {
    if (isTrue(attribute(row, "hidden"))) {
      continue;
    }
    const values = new Map<number, string>();
    let column = 0;
    let rowLength = 0;
    for (const cell of children(row, "c")) {
      const reference = attribute(cell, "r");
      column = reference ? columnIndex(reference) : column + 1;
      if (column > MAX_SPREADSHEET_COLUMNS || hiddenColumns.has(column)) {
        continue;
      }
      // Charged per cell: one shared string repeated across a row would otherwise build a gigabyte line.
      const text = cellText(cell, cells);
      rowLength += text.length + 1;
      if (rowLength > budget.remaining) {
        budget.remaining = 0;
        break;
      }
      values.set(column, text);
    }
    const lastColumn = Math.max(0, ...values.keys());
    const line: string[] = [];
    for (let index = 1; index <= lastColumn; index++) {
      if (!hiddenColumns.has(index)) {
        line.push(values.get(index) ?? "");
      }
    }
    const text = line.join("\t").replace(/\t+$/g, "");
    if (text.trim()) {
      lines.push(text);
      budget.remaining -= text.length + 1;
    }
    if (budget.remaining <= 0) {
      break;
    }
  }
  return lines;
}

function columnIndex(reference: string): number {
  let index = 0;
  for (const char of reference.toUpperCase()) {
    const code = char.charCodeAt(0);
    if (code < 65 || code > 90) {
      break;
    }
    index = index * 26 + code - 64;
  }
  return index;
}

function cellText(cell: Element, context: CellContext): string {
  const type = attribute(cell, "t");
  const value = children(cell, "v")[0]?.textContent ?? "";
  switch (type) {
    case "s":
      return context.sharedStrings[Number(value)] ?? "";
    case "inlineStr": {
      const inline = children(cell, "is")[0];
      return inline ? richText(inline) : "";
    }
    case "b":
      return value === "1" ? "TRUE" : "FALSE";
    case "str":
    case "e":
    case "d":
      return value;
  }
  const number = Number(value);
  if (value === "" || !Number.isFinite(number)) {
    return value;
  }
  const kind = context.dateStyles[Number(attribute(cell, "s") ?? 0)];
  return kind
    ? formatDateSerial(number, kind, context.date1904)
    : formatNumber(number);
}

// Excel shows 15 significant digits, so 0.1 + 0.2 stored as 0.30000000000000004 reads 0.3.
function formatNumber(number: number): string {
  return String(Number(number.toPrecision(15)));
}

function formatDateSerial(
  serial: number,
  kind: Exclude<DateFormatKind, null>,
  date1904: boolean,
): string {
  if (serial < 0 || serial > MAX_DATE_SERIAL) {
    return formatNumber(serial);
  }
  const seconds = Math.round(serial * 86_400);
  if (kind === "duration") {
    const pad = (value: number) => String(value).padStart(2, "0");
    return `${Math.floor(seconds / 3600)}:${pad(Math.floor(seconds / 60) % 60)}:${pad(seconds % 60)}`;
  }
  // The 1900 system counts a nonexistent 1900-02-29 (serial 60), so earlier serials start a day later.
  const epoch = date1904
    ? Date.UTC(1904, 0, 1)
    : Date.UTC(1899, 11, serial < 60 ? 31 : 30);
  const iso = new Date(epoch + seconds * 1000).toISOString();
  const date = iso.slice(0, 10);
  const time = iso.slice(11, 19);
  return kind === "date" ? date : kind === "time" ? time : `${date} ${time}`;
}

function collectDateFormatKinds(styles: Element): DateFormatKind[] {
  const customFormats = new Map<number, string>();
  for (const format of descendants(styles, ["numFmt"])) {
    customFormats.set(
      Number(attribute(format, "numFmtId")),
      attribute(format, "formatCode") ?? "",
    );
  }
  const cellFormats = descendants(styles, ["cellXfs"])[0];
  return (cellFormats ? children(cellFormats, "xf") : []).map((format) => {
    const id = Number(attribute(format, "numFmtId") ?? 0);
    const code = customFormats.get(id);
    if (code !== undefined) {
      return dateFormatCodeKind(code);
    }
    if (id === 22) {
      return "datetime";
    }
    if (id === 46) {
      return "duration";
    }
    return BUILTIN_DATE_FORMAT_IDS.has(id)
      ? "date"
      : BUILTIN_TIME_FORMAT_IDS.has(id)
        ? "time"
        : null;
  });
}

function dateFormatCodeKind(code: string): DateFormatKind {
  const unquoted = code.replace(/"[^"]*"/g, "").replace(/\\./g, "");
  // [h], [mm] and [ss] count elapsed time, which may pass 24 hours.
  if (/\[(h+|m+|s+)\]/i.test(unquoted)) {
    return "duration";
  }
  // Padding, [colour]/[$currency]/[condition] sections and AM/PM markers are not date tokens.
  const tokens = unquoted
    .replace(/[_*]./g, "")
    .replace(/\[[^\]]*\]/g, "")
    .replace(/am\/pm|a\/p/gi, "h");
  const hasDate = /[dy]/i.test(tokens);
  const hasTime = /[hs]/i.test(tokens);
  if (hasDate && hasTime) {
    return "datetime";
  }
  if (hasTime) {
    return "time";
  }
  // A lone "m" (e.g. "mmm") is a month.
  return hasDate || /m/i.test(tokens) ? "date" : null;
}

// Phonetic guides (rPh) repeat the base text as kana, so only the runs are read.
function richText(element: Element): string {
  let text = "";
  for (const child of elementChildren(element)) {
    if (child.localName === "t") {
      text += child.textContent ?? "";
    } else if (child.localName !== "rPh" && child.localName !== "phoneticPr") {
      text += richText(child);
    }
  }
  return text;
}

function isTrue(value: string | null): boolean {
  return value === "1" || value === "true";
}

function parseXml(xml: string, filename: string): XMLDocument {
  const doc = new DOMParser().parseFromString(xml, "application/xml");
  if (doc.getElementsByTagName("parsererror").length > 0) {
    throw new Error(`Failed to parse Office XML: ${filename}`);
  }
  return doc;
}

function elementChildren(element: Element): Element[] {
  return Array.from(element.childNodes).filter(
    (child): child is Element => child.nodeType === ELEMENT_NODE,
  );
}

function children(element: Element, localName: string): Element[] {
  return elementChildren(element).filter(
    (child) => child.localName === localName,
  );
}

function descendants(element: Element, localNames: string[]): Element[] {
  const matches: Element[] = [];
  const stack = elementChildren(element).reverse();
  while (stack.length > 0) {
    const next = stack.pop() as Element;
    if (localNames.includes(next.localName)) {
      matches.push(next);
    }
    const nested = elementChildren(next);
    for (let index = nested.length - 1; index >= 0; index--) {
      stack.push(nested[index]);
    }
  }
  return matches;
}

// Matched by namespace: <p:sldId> also carries a plain numeric id.
function relationshipId(element: Element): string {
  for (const attr of Array.from(element.attributes)) {
    if (
      attr.localName === "id" &&
      attr.namespaceURI?.endsWith("/relationships")
    ) {
      return attr.value;
    }
  }
  return "";
}

function attribute(element: Element, localName: string): string | null {
  for (const attr of Array.from(element.attributes)) {
    if ((attr.localName ?? attr.name) === localName) {
      return attr.value;
    }
  }
  return null;
}
