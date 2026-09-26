// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function csvEscape(val: string): string {
  return `"${val.replace(/"/g, '""')}"`;
}

export const CSV_MIME = "text/csv;charset=utf-8";

// Excel opens .csv files using the system code page (e.g. Windows-1252) unless the file starts with a
// UTF-8 byte-order mark, so every non-Latin1 character we write (accents, Cyrillic, CJK, emoji) turns
// into mojibake without it. Prefixing the BOM is what tells Excel to decode the file as UTF-8.
export function csvDocument(lines: string[]): string {
  return "\ufeff" + lines.join("\n");
}
