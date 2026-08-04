// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { strToU8, zipSync } from "fflate";
import {
  XLSX_WORKBOOK_MIME,
  assertXlsxArchiveSafety,
  formatXlsxWorkbookForChat,
  readXlsxAttachmentText,
} from "../src/features/chat/xlsx-attachment.ts";

function minimalXlsxArchive(): Uint8Array {
  return zipSync({
    "xl/workbook.xml": strToU8(
      [
        '<?xml version="1.0"?>',
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"',
        ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
        '<sheets><sheet name="Data" sheetId="1" r:id="rId1"/></sheets>',
        "</workbook>",
      ].join(""),
    ),
    "xl/_rels/workbook.xml.rels": strToU8(
      [
        '<?xml version="1.0"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        '<Relationship Id="rId1"',
        ' Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"',
        ' Target="worksheets/sheet1.xml"/>',
        "</Relationships>",
      ].join(""),
    ),
    "xl/worksheets/sheet1.xml": strToU8(
      [
        '<?xml version="1.0"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        '<sheetData><row r="1">',
        '<c r="A1" t="inlineStr"><is><t>value</t></is></c>',
        '<c r="B1"><v>42</v></c>',
        "</row></sheetData></worksheet>",
      ].join(""),
    ),
    "xl/styles.xml": strToU8(
      [
        '<?xml version="1.0"?>',
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        '<cellXfs count="1"><xf numFmtId="0"/></cellXfs>',
        "</styleSheet>",
      ].join(""),
    ),
  });
}

test("formats every worksheet and common cell value", () => {
  const text = formatXlsxWorkbookForChat([
    {
      sheet: "Revenue",
      data: [
        ["Name", "Amount", "Active", "When"],
        ["Alice", 12.5, true, new Date("2026-07-31T12:30:00.000Z")],
        [],
      ],
    },
    {
      sheet: "Notes\n2026",
      data: [["Line 1\nLine 2", "tab\tvalue"]],
    },
  ]);

  assert.equal(
    text,
    [
      "[Sheet: Revenue]",
      "Name\tAmount\tActive\tWhen",
      "Alice\t12.5\ttrue\t2026-07-31T12:30:00.000Z",
      "",
      "[Sheet: Notes ⏎ 2026]",
      "Line 1 ⏎ Line 2\ttab    value",
    ].join("\n"),
  );
});

test("marks workbook text when chat limits truncate it", () => {
  const text = formatXlsxWorkbookForChat(
    [{ sheet: "Sheet 1", data: [["first"], ["second"]] }],
    { maxSheets: 10, maxRows: 1, maxColumns: 10, maxCharacters: 1_000 },
  );

  assert.match(text, /^\[Sheet: Sheet 1\]\nfirst/);
  assert.match(text, /\[Workbook content truncated for chat\.\]$/);
  assert.doesNotMatch(text, /second/);
});

test("preflights XLSX archives before parsing", () => {
  const archive = minimalXlsxArchive();
  assert.doesNotThrow(() => assertXlsxArchiveSafety(archive, "workbook.xlsx"));
  assert.throws(
    () =>
      assertXlsxArchiveSafety(archive, "workbook.xlsx", {
        maxEntries: 100,
        maxXmlEntryBytes: 10,
        maxXmlBytes: 1_000,
      }),
    /XLSX XML entry is too large/,
  );
  assert.throws(
    () =>
      assertXlsxArchiveSafety(
        zipSync({ "xl/worksheets/sheet1.xml": strToU8("<worksheet/>") }),
        "not-a-workbook.xlsx",
      ),
    /missing xl\/workbook\.xml/,
  );
});

test("parses and formats a real XLSX archive", async () => {
  const archive = minimalXlsxArchive();
  const file = new File([archive.buffer as ArrayBuffer], "fixture.xlsx", {
    type: XLSX_WORKBOOK_MIME,
  });

  const text = await readXlsxAttachmentText(file);
  assert.equal(text, "[Sheet: Data]\nvalue\t42");
});
