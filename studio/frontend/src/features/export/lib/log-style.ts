// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ExportLogEntry } from "../api/export-api";

type ExportLogTone = "stdout" | "stderr" | "status" | "warning";

const WARNING_LINE_PATTERNS = [
  /Skipping import of cpp extensions due to incompatible torch version/i,
  /Please see GitHub issue #2919 for more info/i,
  /torch_dtype is deprecated!\s*Use dtype instead!/i,
] as const;

function isWarningLine(line: string): boolean {
  return WARNING_LINE_PATTERNS.some((pattern) => pattern.test(line));
}

export function getExportLogTone(entry: ExportLogEntry): ExportLogTone {
  if (entry.stream === "status") {
    return "status";
  }
  if (isWarningLine(entry.line)) {
    return "warning";
  }
  return entry.stream === "stderr" ? "stderr" : "stdout";
}

export function getExportLogLineClass(entry: ExportLogEntry): string {
  const tone = getExportLogTone(entry);
  if (tone === "stderr") {
    return "text-rose-300/90";
  }
  if (tone === "status") {
    return "text-sky-300/90";
  }
  if (tone === "warning") {
    return "text-status-warning";
  }
  return "";
}

/** Lines per rendered block; a block holds absolute lines [k * size, (k + 1) * size). */
export const EXPORT_LOG_GROUP_SIZE = 100;

export interface ExportLogGroup {
  key: number;
  start: number;
  entries: ExportLogEntry[];
}

/** Fixed absolute blocks: a trim touches only the first, an append only the last. */
export function groupExportLogLines(
  lines: readonly ExportLogEntry[],
  base: number,
  size: number = EXPORT_LOG_GROUP_SIZE,
): ExportLogGroup[] {
  const groups: ExportLogGroup[] = [];
  let i = 0;
  while (i < lines.length) {
    const start = base + i;
    const end = Math.min(lines.length, i + size - (start % size));
    groups.push({
      key: Math.floor(start / size),
      start,
      entries: lines.slice(i, end),
    });
    i = end;
  }
  return groups;
}
