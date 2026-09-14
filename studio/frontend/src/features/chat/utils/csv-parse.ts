// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// RFC 4180 CSV parser: handles quoted fields with embedded newlines/commas.
export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let i = 0;

  function finishRow() {
    rows.push(row);
    row = [];
  }

  while (i < text.length) {
    if (text[i] === '"') {
      i++;
      let cell = "";
      while (i < text.length) {
        if (text[i] === '"') {
          if (text[i + 1] === '"') {
            cell += '"';
            i += 2;
          } else {
            i++;
            break;
          }
        } else {
          cell += text[i++];
        }
      }
      row.push(cell);
      if (text[i] === ",") { i++; }
      else if (text[i] === "\r") { i++; if (text[i] === "\n") i++; finishRow(); }
      else if (text[i] === "\n") { i++; finishRow(); }
    } else if (text[i] === "\r") {
      i++;
      if (text[i] === "\n") i++;
      row.push("");
      finishRow();
    } else if (text[i] === "\n") {
      i++;
      row.push("");
      finishRow();
    } else {
      let cell = "";
      while (i < text.length && text[i] !== "," && text[i] !== "\r" && text[i] !== "\n") {
        cell += text[i++];
      }
      row.push(cell);
      if (text[i] === ",") { i++; }
      else if (text[i] === "\r") { i++; if (text[i] === "\n") i++; finishRow(); }
      else if (text[i] === "\n") { i++; finishRow(); }
    }
  }
  if (row.length > 0) rows.push(row);
  return rows;
}
