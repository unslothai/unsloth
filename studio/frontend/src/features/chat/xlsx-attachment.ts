// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { unzipSync } from "fflate";

export const XLSX_WORKBOOK_MIME =
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";

const MAX_XLSX_ARCHIVE_BYTES = 50 * 1024 * 1024;

export type XlsxArchiveLimits = {
  maxEntries: number;
  maxXmlEntryBytes: number;
  maxXmlBytes: number;
};

const DEFAULT_XLSX_ARCHIVE_LIMITS: XlsxArchiveLimits = {
  maxEntries: 2_000,
  maxXmlEntryBytes: 25 * 1024 * 1024,
  maxXmlBytes: 100 * 1024 * 1024,
};

export type XlsxFormatLimits = {
  maxSheets: number;
  maxRows: number;
  maxColumns: number;
  maxCharacters: number;
};

const DEFAULT_XLSX_FORMAT_LIMITS: XlsxFormatLimits = {
  maxSheets: 100,
  maxRows: 20_000,
  maxColumns: 1_000,
  maxCharacters: 250_000,
};

export type ChatXlsxSheet = {
  sheet: string;
  data: unknown[][];
};

const TRUNCATION_NOTICE = "[Workbook content truncated for chat.]";

export async function readXlsxAttachmentText(file: File): Promise<string> {
  if (file.size > MAX_XLSX_ARCHIVE_BYTES) {
    throw new Error(`XLSX archive is too large: ${file.name}`);
  }

  const [{ default: readXlsxFile }, buffer] = await Promise.all([
    import("read-excel-file/browser"),
    file.arrayBuffer(),
  ]);
  assertXlsxArchiveSafety(new Uint8Array(buffer), file.name);

  let sheets: ChatXlsxSheet[];
  try {
    sheets = (await readXlsxFile(buffer)) as ChatXlsxSheet[];
  } catch (error) {
    const detail = error instanceof Error ? `: ${error.message}` : "";
    throw new Error(`Failed to parse XLSX workbook: ${file.name}${detail}`, {
      cause: error,
    });
  }

  return formatXlsxWorkbookForChat(sheets);
}

export async function readActiveXlsxAttachmentText(
  file: File,
  isActive: () => boolean,
): Promise<string | null> {
  try {
    const text = await readXlsxAttachmentText(file);
    return isActive() ? text : null;
  } catch (error) {
    if (!isActive()) return null;
    throw error;
  }
}

export function assertXlsxArchiveSafety(
  bytes: Uint8Array,
  filename: string,
  limits: XlsxArchiveLimits = DEFAULT_XLSX_ARCHIVE_LIMITS,
): void {
  let entries = 0;
  let xmlBytes = 0;
  let hasWorkbook = false;

  try {
    unzipSync(bytes, {
      filter: (entry) => {
        entries++;
        if (entries > limits.maxEntries) {
          throw new Error(`XLSX archive has too many entries: ${filename}`);
        }

        if (entry.name === "xl/workbook.xml") hasWorkbook = true;
        if (!isXlsxXmlEntry(entry.name)) return false;

        if (entry.originalSize > limits.maxXmlEntryBytes) {
          throw new Error(
            `XLSX XML entry is too large: ${filename}:${entry.name}`,
          );
        }
        xmlBytes += entry.originalSize;
        if (xmlBytes > limits.maxXmlBytes) {
          throw new Error(`XLSX XML content is too large: ${filename}`);
        }
        return false;
      },
    });
  } catch (error) {
    if (isXlsxSafetyError(error)) throw error;
    throw new Error(`Failed to read XLSX archive: ${filename}`, {
      cause: error,
    });
  }

  if (!hasWorkbook) {
    throw new Error(`XLSX archive is missing xl/workbook.xml: ${filename}`);
  }
}

export function formatXlsxWorkbookForChat(
  sheets: ChatXlsxSheet[],
  limits: XlsxFormatLimits = DEFAULT_XLSX_FORMAT_LIMITS,
): string {
  if (sheets.length === 0) return "(Workbook contains no worksheets.)";

  const lines: string[] = [];
  let characters = 0;
  let rows = 0;
  let truncated = false;

  const appendLine = (line: string): boolean => {
    const addedLength = line.length + (lines.length > 0 ? 1 : 0);
    if (characters + addedLength > limits.maxCharacters) {
      const remaining = Math.max(
        0,
        limits.maxCharacters - characters - (lines.length > 0 ? 1 : 0),
      );
      if (remaining > 0) lines.push(line.slice(0, remaining).trimEnd());
      truncated = true;
      return false;
    }
    lines.push(line);
    characters += addedLength;
    return true;
  };

  sheetLoop: for (const [sheetIndex, sheet] of sheets.entries()) {
    if (sheetIndex >= limits.maxSheets) {
      truncated = true;
      break;
    }

    if (lines.length > 0 && !appendLine("")) break;
    const sheetName =
      normalizeXlsxCellText(sheet.sheet) || `Sheet ${sheetIndex + 1}`;
    if (!appendLine(`[Sheet: ${sheetName}]`)) break;

    let sheetHasRows = false;
    for (const row of sheet.data) {
      const cells = row.slice(0, limits.maxColumns).map(formatXlsxCellForChat);
      while (cells.length > 0 && cells.at(-1) === "") cells.pop();
      if (cells.length === 0) continue;

      if (rows >= limits.maxRows) {
        truncated = true;
        break sheetLoop;
      }
      if (row.length > limits.maxColumns) truncated = true;
      if (!appendLine(cells.join("\t"))) break sheetLoop;
      rows++;
      sheetHasRows = true;
    }

    if (!sheetHasRows && !appendLine("(empty sheet)")) break;
  }

  if (truncated) {
    if (lines.at(-1) !== "") lines.push("");
    lines.push(TRUNCATION_NOTICE);
  }
  return lines.join("\n");
}

function isXlsxXmlEntry(name: string): boolean {
  const lower = name.toLowerCase();
  return lower.endsWith(".xml") || lower.endsWith(".xml.rels");
}

function isXlsxSafetyError(error: unknown): boolean {
  return (
    error instanceof Error &&
    (error.message.startsWith("XLSX archive has too many entries:") ||
      error.message.startsWith("XLSX XML entry is too large:") ||
      error.message.startsWith("XLSX XML content is too large:"))
  );
}

function formatXlsxCellForChat(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? "" : value.toISOString();
  }
  return normalizeXlsxCellText(String(value));
}

function normalizeXlsxCellText(value: string): string {
  return value
    .replace(/\r\n?/g, "\n")
    .replace(/\t/g, "    ")
    .replace(/\n/g, " ⏎ ")
    .trim();
}
