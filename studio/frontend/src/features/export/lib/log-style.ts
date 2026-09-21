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
