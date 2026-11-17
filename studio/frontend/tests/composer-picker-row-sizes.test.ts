// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Opt-in GB and params on composer picker rows, from the same data the header picker used.

import assert from "node:assert/strict";
import test from "node:test";

import {
  rowOwnBytes,
  rowSizeMeta,
} from "../src/components/assistant-ui/composer-model-picker/picker-model.ts";
import { readSrc } from "./helpers/kit.ts";

const GB = 1e9;

test("an on-device row shows its bytes on disk and the name's param count", () => {
  assert.deepEqual(
    rowSizeMeta({
      kind: "gguf",
      id: "unsloth/Qwen3-4B-GGUF",
      format: "gguf",
      sizeBytes: 2.5 * GB,
    }),
    { params: "4B", size: "2.5 GB" },
  );
  assert.deepEqual(
    rowSizeMeta({ kind: "cached", id: "unsloth/Llama-3.1-8B-Instruct", sizeBytes: 16.1 * GB }),
    { params: "8B", size: "16 GB" },
  );
});

test("a trained row takes its params from the base model", () => {
  assert.deepEqual(
    rowSizeMeta({
      kind: "local",
      id: "outputs/my-run/checkpoint-60",
      baseModel: "unsloth/Qwen3-14B",
      sizeBytes: 120e6,
    }),
    { params: "14B", size: "120 MB" },
  );
});

test("unknown stays blank instead of guessing", () => {
  assert.deepEqual(rowSizeMeta({ kind: "cached", id: "someone/mystery-model" }), {
    params: null,
    size: null,
  });
  assert.deepEqual(
    rowSizeMeta({ kind: "cached", id: "someone/mystery-model", sizeBytes: 0 }),
    { params: null, size: null },
  );
  assert.deepEqual(
    rowSizeMeta({ kind: "cached", id: "x/y", sizeBytes: Number.NaN }),
    { params: null, size: null },
  );
});

test("a partial download shows no size: its bytes are what arrived", () => {
  assert.equal(
    rowSizeMeta({
      kind: "gguf",
      id: "unsloth/Qwen3-32B-GGUF",
      partial: true,
      sizeBytes: 3 * GB,
    }).size,
    null,
  );
});

test("Discover rows: the Hub estimate reads as approximate, never for a GGUF repo", () => {
  assert.deepEqual(
    rowSizeMeta({
      kind: "hub",
      id: "unsloth/Qwen3-8B",
      format: "checkpoint",
      estimatedBytes: 16.4 * GB,
      hubParams: "8.2B",
    }),
    { params: "8.2B", size: "~16 GB" },
  );
  // The search's GGUF size is the unquantized weights, so no quant on the row would match it.
  assert.deepEqual(
    rowSizeMeta({
      kind: "hub",
      id: "unsloth/Qwen3-8B-GGUF",
      format: "gguf",
      estimatedBytes: 16.4 * GB,
      hubParams: "8B",
    }),
    { params: "8B", size: null },
  );
  // Discover takes params from the search result only, not from the row's bytes on disk.
  assert.deepEqual(
    rowSizeMeta({ kind: "hub", id: "a/b", sizeBytes: 4 * GB }),
    { params: null, size: null },
  );
});

test("a favorite quant row sizes that quant, and only that quant", () => {
  assert.equal(rowOwnBytes("Q4_K_M", 2 * GB, null, 9 * GB), 2 * GB);
  // Pinned quant deleted while a sibling stays: not the sibling's bytes, not the repo's.
  assert.equal(rowOwnBytes("Q4_K_M", null, 3 * GB, 3 * GB), null);
  assert.equal(rowOwnBytes("Q4_K_M", null, null, 9 * GB), null);
  assert.equal(rowOwnBytes(undefined, null, 3 * GB, 9 * GB), 3 * GB);
  assert.equal(rowOwnBytes(undefined, null, null, 9 * GB), 9 * GB);
  assert.equal(rowOwnBytes(undefined, null, null, undefined), null);
});

test("connected API models get neither", () => {
  assert.deepEqual(
    rowSizeMeta({ kind: "external", id: "meta-llama/llama-3.1-70b", sizeBytes: GB }),
    { params: null, size: null },
  );
});

test("the setting is off by default and Reset All clears it", () => {
  const store = readSrc("features/chat/stores/chat-runtime-store.ts");
  const key = store.match(/CHAT_SHOW_MODEL_SIZES_KEY\s*=\s*"([^"]+)"/);
  assert.ok(key, "no CHAT_SHOW_MODEL_SIZES_KEY");
  assert.match(
    store,
    /showModelSizes:\s*loadBool\(\s*CHAT_SHOW_MODEL_SIZES_KEY\s*,\s*false\s*\)/,
  );
  assert.match(
    readSrc("features/settings/tabs/general-tab.tsx"),
    new RegExp(`PREFS_KEYS[\\s\\S]*?"${key[1]}"`),
  );
});

test("Settings has the toggle beside the other model selection rows", () => {
  const tab = readSrc("features/settings/tabs/chat-tab.tsx");
  const section = tab.indexOf('t("settings.chat.modelSelection.title")');
  const toggle = tab.indexOf("checked={showModelSizes}");
  assert.ok(section > 0 && toggle > section, "toggle is not under Model selection");
  assert.match(tab, /onCheckedChange=\{setShowModelSizes\}/);
});

test("rows only draw the columns when the setting is on", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(picker, /const sizeMeta = showModelSizes\s*\?\s*rowSizeMeta\(/);
  assert.match(picker, /\{sizeMeta && row\.kind !== "external" \? \(\s*<RowSizeColumns/);
  // Off: Discover keeps its param chip exactly as before; on, the column carries it instead.
  assert.match(picker, /\{paramLabel && !sizeMeta \? \(/);
  // Discover reserves the size column only when some listed row can fill it (GGUF search rows never do).
  assert.match(picker, /sizeSlot=\{row\.kind !== "hub" \|\| hubHasSizes\}/);
  // Judged on the rows that survive the format and fit filters, not the raw search results.
  assert.match(
    picker,
    /const hubHasSizes = useMemo\(\s*\(\) =>\s*shownRows\.some\(\s*\(row\) =>\s*row\.kind === "hub" &&\s*rowSizeMeta\(/,
  );
  assert.match(picker, /\[shownRows, hubEstimates\]/);
  assert.match(
    picker,
    /sizeBytes: rowOwnBytes\(\s*row\.favoriteQuant,\s*savedVariant \? quantBytes\(savedVariant\) : null,\s*onDisk\.length === 1 \? quantBytes\(onDisk\[0\]\) : null,\s*row\.sizeBytes,\s*\)/,
  );
});
