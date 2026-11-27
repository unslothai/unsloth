// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import vm from "node:vm";
import ts from "@typescript/typescript6";
import * as csvExport from "../src/features/chat/utils/csv-export.ts";
import { parseCsv } from "../src/features/chat/utils/csv-parse.ts";

import { readSrc } from "./helpers/kit.ts";

type List = { id: string; name: string; items: string[] };

const SOURCE = readSrc(
  "features/chat/prompt-storage/prompt-storage-dialog.tsx",
);

function slice(start: string, end: string): string {
  const from = SOURCE.indexOf(start);
  const to = SOURCE.indexOf(end);
  assert.ok(from !== -1 && to > from, `${start} not found`);
  return SOURCE.slice(from, to);
}

// The real exporter and importer, with the download and the store replaced by spies.
function listCsv() {
  const javascript = ts.transpileModule(
    [
      slice("function exportAllListsCsv(", "function contentBlocksToText("),
      slice(
        "async function importListsFromText(",
        "async function importCollectionFromText(",
      ),
      "globalThis.__lists = { exportAllListsCsv, importListsFromText };",
    ].join("\n"),
    {
      compilerOptions: {
        module: ts.ModuleKind.CommonJS,
        target: ts.ScriptTarget.ES2022,
      },
    },
  ).outputText;
  const downloads: string[] = [];
  const saved: List[] = [];
  let ids = 0;
  const context: Record<string, unknown> = {
    exports: {},
    ...csvExport,
    parseCsv,
    newId: () => `new-${++ids}`,
    now: () => 0,
    downloadBlob: async (body: string) => {
      downloads.push(body);
    },
    bulkSavePromptLists: async (lists: List[]) => {
      saved.push(...lists);
    },
  };
  vm.runInNewContext(javascript, context);
  const { exportAllListsCsv, importListsFromText } = context.__lists as {
    exportAllListsCsv: (lists: List[]) => Promise<void>;
    importListsFromText: (
      text: string,
      isCsv: boolean,
    ) => Promise<{ count: number }>;
  };
  return {
    async exportCsv(lists: List[]): Promise<string> {
      downloads.length = 0;
      await exportAllListsCsv(lists);
      return downloads[0];
    },
    async importCsv(text: string): Promise<List[]> {
      saved.length = 0;
      await importListsFromText(text, true);
      return [...saved];
    },
  };
}

// Copy out of the vm realm: deepStrictEqual compares prototypes.
const lists = (entries: List[]) =>
  entries.map(({ name, items }) => ({ name, items: [...items] }));

test("a CSV export of several lists imports back as the same lists, even when they share a name", async () => {
  const { exportCsv, importCsv } = listCsv();
  const original: List[] = [
    { id: "a", name: "Review", items: ["A1", "A2"] },
    { id: "b", name: "Review", items: ["B1", 'Say "hi", then\nwave'] },
    { id: "c", name: "Code", items: ["C1"] },
  ];

  const body = await exportCsv(original);
  assert.ok(body.startsWith("﻿list,list_name,order,prompt_text\n"), body);

  const imported = await importCsv(body);
  assert.deepEqual(lists(imported), lists(original));
  assert.deepEqual(
    imported.map(({ id }) => id),
    ["new-1", "new-2", "new-3"],
  );
});

test("a CSV export of one list keeps the list_name,order,prompt_text layout", async () => {
  const { exportCsv, importCsv } = listCsv();
  const original: List[] = [{ id: "a", name: "Review", items: ["A1", "A2"] }];

  const body = await exportCsv(original);
  assert.ok(body.startsWith("﻿list_name,order,prompt_text\n"), body);
  assert.deepEqual(lists(await importCsv(body)), lists(original));
});

test("a list_name,order,prompt_text CSV still imports one list per name, in order", async () => {
  const { importCsv } = listCsv();
  const row = (...cells: string[]) => cells.map(csvExport.csvEscape).join(",");
  const doc = csvExport.csvDocument([
    "list_name,order,prompt_text",
    row("Review", "2", "second"),
    row("Other", "1", "only"),
    row("Review", "1", "first"),
  ]);

  assert.deepEqual(lists(await importCsv(doc)), [
    { name: "Review", items: ["first", "second"] },
    { name: "Other", items: ["only"] },
  ]);
});
