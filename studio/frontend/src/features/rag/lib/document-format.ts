// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Display and ordering helpers shared by the settings uploaded-files list and
 * the project sources panel. Kept free of React so both can import them and the
 * node:test suite (which cannot load .tsx) can exercise them directly. */

export function formatUploadedAt(
  value: string | number | null | undefined,
): string {
  if (value === null || value === undefined || value === "") return "-";
  // Chat attachments carry ms epoch numbers; RAG documents carry SQLite
  // ISO-ish strings (no timezone). Unparseable strings fall through raw.
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleDateString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

export function formatSize(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return "-";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes;
  let unit = "B";
  for (const next of units) {
    if (value < 1024) break;
    value /= 1024;
    unit = next;
  }
  return `${value >= 10 ? Math.round(value) : value.toFixed(1)} ${unit}`;
}

/** Epoch ms for sorting; rows with unknown or unparseable dates sort last. */
export function toSortTime(value: string | number | null | undefined): number {
  if (value === null || value === undefined || value === "") return 0;
  const parsed = new Date(value).getTime();
  return Number.isNaN(parsed) ? 0 : parsed;
}

/** Short uppercase file-type label from the filename extension, falling back
 *  to the content-type subtype (e.g. "image/webp" gives WEBP). */
export function fileTypeLabel(
  name: string,
  contentType?: string | null,
): string | null {
  const dot = name.lastIndexOf(".");
  const ext = dot > 0 ? name.slice(dot + 1).trim() : "";
  if (ext && ext.length <= 5) return ext.toUpperCase();
  const subtype = contentType?.split("/")[1]?.split("+")[0]?.trim();
  return subtype && subtype.length <= 10 ? subtype.toUpperCase() : null;
}
