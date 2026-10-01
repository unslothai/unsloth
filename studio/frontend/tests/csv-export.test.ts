// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { csvDocument, csvEscape } from "../src/features/chat/utils/csv-export.ts";
import { parseCsv } from "../src/features/chat/utils/csv-parse.ts";

test("csvDocument prefixes a UTF-8 byte-order mark", () => {
  const doc = csvDocument(["a,b", "1,2"]);
  assert.equal(doc.charCodeAt(0), 0xfeff);
  assert.equal(doc, "\ufeffa,b\n1,2");
});

test("escaped rows survive a csvDocument/parseCsv round trip with non-Latin1 and multiline text", () => {
  const header = ["name", "text"];
  const rows = [
    ["comma", "a,b"],
    ["quotes", 'she said "hi"'],
    ["crlf", "line one\r\nline two"],
    ["lf", "line one\nline two"],
    ["czech", "není potřeba"],
    ["russian", "это"],
    ["cjk", "日本語"],
    ["emoji", "👍🏽"],
  ];

  const doc = csvDocument([
    header.map(csvEscape).join(","),
    ...rows.map((cells) => cells.map(csvEscape).join(",")),
  ]);

  assert.equal(doc.charCodeAt(0), 0xfeff);

  const parsed = parseCsv(doc);
  assert.deepEqual(parsed[0], header);
  assert.deepEqual(parsed.slice(1), rows);
});

test("parseCsv returns identical rows with and without a leading BOM", () => {
  const body = 'name,text\nczech,není potřeba\ncjk,日本語\nemoji,👍🏽';
  const withBom = "\ufeff" + body;

  assert.deepEqual(parseCsv(withBom), parseCsv(body));
});
