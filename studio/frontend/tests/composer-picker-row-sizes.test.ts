// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Opt-in GB and params on composer picker rows, from the same data the header picker used.

import assert from "node:assert/strict";
import test from "node:test";

import {
  rowOwnBytes,
  rowSizeMeta,
} from "../src/components/assistant-ui/composer-model-picker/picker-model.ts";
import { fitMemoryText } from "../src/features/model-picker/components/model-selector/fit-verdict.ts";
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

test("a folder model whose baseModel is its store falls back to the id", () => {
  assert.equal(
    rowSizeMeta({ kind: "local", id: "/models/lmstudio/Qwen3-8B", baseModel: "LM Studio" }).params,
    "8B",
  );
  assert.equal(
    rowSizeMeta({ kind: "local", id: "/models/custom/mystery", baseModel: "Custom Folders" }).params,
    null,
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
  // A GGUF repo has no dtype-summed estimate, so none shows.
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

test("an NPU row reads params from its lemonade id", () => {
  assert.equal(
    rowSizeMeta({ kind: "npu", id: "lemonade:qwen3-0.6b-FLM", sizeBytes: 0.9 * GB }).params,
    "0.6B",
  );
});

test("connected API models get neither", () => {
  assert.deepEqual(
    rowSizeMeta({ kind: "external", id: "meta-llama/llama-3.1-70b", sizeBytes: GB }),
    { params: null, size: null },
  );
});

test("folder models carry the inventory's bytes to the picker", () => {
  assert.match(
    readSrc("features/model-picker/inventory/use-chat-picker-inventory.ts"),
    /function toLocalModelInfo[\s\S]*?size_bytes: row\.bytes,/,
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
  assert.match(picker, /const rowSize = rowSizeMeta\(/);
  assert.match(picker, /const sizeMeta = showModelSizes \? rowSize : null;/);
  assert.match(picker, /\{sizeMeta && row\.kind !== "external" \? \(\s*<RowSizeColumns/);
  // Off: Discover keeps its param chip exactly as before; on, the column carries it instead.
  assert.match(picker, /\{paramLabel && !sizeMeta \? \(/);
  // Judged on the rows that survive the format and fit filters, and not at all with the setting off.
  assert.match(
    picker,
    /const hubHasSizes = useMemo\(\s*\(\) =>\s*showModelSizes &&\s*shownRows\.some\(\s*\(row\) =>\s*row\.kind === "hub" &&\s*rowSizeMeta\(/,
  );
  assert.match(picker, /\[showModelSizes, shownRows, hubSizes\]/);
  // A favorited quant is looked up on disk once, for every row kind, local GGUF folders too.
  assert.match(
    picker,
    /const favoriteOnDisk = row\.favoriteQuant\s*\?\s*onDisk\.find\(/,
  );
  assert.match(picker, /row\.favoriteQuant\s*\?\s*favoriteOnDisk\s*:/);
  assert.match(
    picker,
    /sizeBytes: rowOwnBytes\(\s*row\.favoriteQuant,\s*favoriteOnDisk \? quantBytes\(favoriteOnDisk\) : null,\s*onDisk\.length === 1 \? quantBytes\(onDisk\[0\]\) : null,[\s\S]*?row\.sizeBytes \?\?\s*\(row\.lora\?\.source === "local" \? row\.lora\.sizeBytes : undefined\),\s*\)/,
  );
});

test("Discover alone hides the size column when no Hub row can fill it", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  // A pinned Hub row in Favorites keeps the slot so it lines up with its neighbours.
  assert.match(
    picker,
    /sizeSlot=\{\s*row\.kind !== "hub" \|\|\s*viewed !== RECOMMENDED_TAB \|\|\s*hubHasSizes\s*\}/,
  );
});

test("the Details menu shows the row's own size and params", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  const facts = picker.slice(
    picker.indexOf("function modelFacts("),
    picker.indexOf("const task =", picker.indexOf("function modelFacts(")),
  );
  assert.match(facts, /sized: ReturnType<typeof rowSizeMeta>/);
  assert.match(facts, /facts\.push\(\["Disk space", sized\.size\]\)/);
  assert.match(facts, /facts\.push\(\["Parameters", sized\.params\]\)/);
  // No second derivation that could drift from the row.
  assert.doesNotMatch(facts, /quantBytes|extractParamLabel|row\.sizeBytes/);
  assert.match(
    picker,
    /<ModelFacts\s+facts=\{modelFacts\([\s\S]*?\),\s*rowSize,\s*\)\}/,
  );
  // Computed whether or not the setting is on, so Details matches the row either way.
  assert.doesNotMatch(picker, /showModelSizes\s*\?\s*rowSizeMeta\(/);
});

test("Details and the row agree for a favorited quant and a folder model", () => {
  // Favorite Q4 with Q8 also on disk: the quant's bytes, not the repo total.
  const fav = rowSizeMeta({
    kind: "gguf",
    id: "unsloth/Qwen3-4B-GGUF",
    format: "gguf",
    sizeBytes: rowOwnBytes("Q4_K_M", 2.5 * GB, null, 6.8 * GB),
  });
  assert.deepEqual(fav, { params: "4B", size: "2.5 GB" });
  // LM Studio folder: inventory bytes, params from the id.
  assert.deepEqual(
    rowSizeMeta({
      kind: "local",
      id: "/models/lmstudio/Qwen3-8B",
      baseModel: "LM Studio",
      sizeBytes: rowOwnBytes(undefined, null, null, 5 * GB),
    }),
    { params: "8B", size: "5.0 GB" },
  );
});

test("a fit mark's tooltip gives the memory it was judged with", () => {
  assert.equal(fitMemoryText("oom", 40, 24), "Needs ~40 GiB memory (GPU: 24 GiB)");
  assert.equal(fitMemoryText("partial", 30.4, 24), "Needs ~30 GiB memory (GPU: 24 GiB)");
  assert.equal(
    fitMemoryText("offload", 9.2, 8),
    "~9.2 GiB memory. Might be slower with RAM offloading.",
  );
  assert.equal(fitMemoryText("marginal", 22, 24), "~22 GiB VRAM (tight fit on 24 GiB)");
  // No GPU or no estimate: the generic hint stays.
  assert.equal(fitMemoryText("oom", 40, 0), null);
  assert.equal(fitMemoryText("oom", undefined, 24), null);
  assert.equal(fitMemoryText(null, 40, 24), null);
});

test("composer fit marks carry the estimate on Discover rows and quant rows", () => {
  const picker = readSrc(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(picker, /\{fitMemoryText\(status, needGb, gpuGb\) \?\? verdict\.hint\}/);
  // The same smallest-quant bytes the verdict was reached with.
  assert.match(picker, /map\.set\(id, fitOf\(smallestQuantBytes\(r\)\)\);/);
  assert.match(picker, /map\.set\(r\.id\.toLowerCase\(\), requiredGgufMemoryGb\(bytes\)\);/);
  assert.match(picker, /<FitBadge\s+status=\{fit\}\s+needGb=\{needGb\}\s+gpuGb=\{inferenceGpu\.memoryTotalGb\}/);
  assert.match(
    picker,
    /<FitBadge\s+status=\{fit\}\s+needGb=\{requiredGgufMemoryGb\(quantBytes\(variant\)\)\}\s+gpuGb=\{inferenceGpu\.memoryTotalGb\}/,
  );
  // The header picker's rows read the same text.
  assert.match(
    readSrc("features/model-picker/components/model-selector/pickers.tsx"),
    /: fitMemoryText\(vramStatus, vramEst, gpuGb\)/,
  );
});

