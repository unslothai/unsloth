// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  DEDICATED_TRANSLATE_MODELS,
  offeredTranslateModels,
} from "../src/features/translate/models.ts";
import { quantOnDisk } from "../src/features/translate/quant-on-disk.ts";
import type * as Runner from "../src/features/translate/translate-runner.ts";
import { readText } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Row = { repo_id: string; partial?: boolean };
let cachedCalls = 0;
let listing: () => Promise<Row[]> = async () => [];

const runner = loadWithStubs<typeof Runner>(
  new URL("../src/features/translate/translate-runner.ts", import.meta.url),
  {
    "@/features/chat/api/chat-api": {
      listCachedModels: (_token: unknown, signal?: AbortSignal) => {
        cachedCalls += 1;
        signal?.throwIfAborted();
        return listing();
      },
      listGgufVariants: async () => ({ variants: [] }),
    },
    "@/features/chat/external-providers": {},
    "@/features/chat/stores/chat-runtime-store": {},
    "@/features/chat/utils/chat-title": {},
    "@/features/hf-auth": {},
    "@/features/hub/download-manager/constants": {},
    "@/features/hub/download-manager/download-manager-controller": {},
    "@/features/hub/lib/model-identity": {},
    "@/features/hub/stores/hf-token-store": {},
    "@/lib/navigation-nonce": {},
    "@/lib/translation-models": {},
    "./detect-language": {},
    "./models": {},
    "./quant-on-disk": { quantOnDisk },
    "./translate-names": {},
    "./translate-subtitles": {},
    "./translate-text": {},
  },
);

const fp8Rows = offeredTranslateModels(["fp8"]).filter((model) => model.fp8);
const onDiskRows: Row[] = [
  { repo_id: fp8Rows[0].repo },
  { repo_id: fp8Rows[1].repo.toUpperCase(), partial: true },
];

async function probePass(shared: boolean): Promise<(boolean | null)[]> {
  const signal = new AbortController().signal;
  const cachedModels = shared
    ? runner.sharedCachedModelsListing(signal)
    : undefined;
  return Promise.all(
    fp8Rows.map((model) =>
      runner.translateModelDownloaded(model, signal, {
        remote: true,
        cachedModels,
      }),
    ),
  );
}

test("the premise: an FP8 GPU offers three FP8 rows, each read from the cache listing", () => {
  assert.equal(fp8Rows.length, 3);
  assert.ok(fp8Rows.every((model) => model.variant === null));
});

test("one probe pass reads the cache listing once for every FP8 row", async () => {
  listing = async () => onDiskRows;
  cachedCalls = 0;
  const perRow = await probePass(false);
  assert.equal(cachedCalls, 3, "without a shared listing each row asks again");
  cachedCalls = 0;
  const shared = await probePass(true);
  assert.equal(cachedCalls, 1);
  assert.deepEqual(shared, perRow);
  assert.deepEqual(shared, [true, false, false]);
});

test("a failed listing leaves every FP8 row unknown, as a per-row read does", async () => {
  listing = async () => {
    throw new Error("HTTP 500");
  };
  cachedCalls = 0;
  assert.deepEqual(await probePass(false), [null, null, null]);
  cachedCalls = 0;
  assert.deepEqual(await probePass(true), [null, null, null]);
  assert.equal(cachedCalls, 1);
});

test("an aborted pass leaves every FP8 row unknown", async () => {
  listing = async () => onDiskRows;
  const abort = new AbortController();
  abort.abort();
  const cachedModels = runner.sharedCachedModelsListing(abort.signal);
  const results = await Promise.all(
    fp8Rows.map((model) =>
      runner.translateModelDownloaded(model, abort.signal, { cachedModels }),
    ),
  );
  assert.deepEqual(results, [null, null, null]);
});

test("a pass with no FP8 row never reads the cache listing", async () => {
  cachedCalls = 0;
  runner.sharedCachedModelsListing();
  assert.equal(cachedCalls, 0);
});

test("callers outside the page pass keep their own read", async () => {
  listing = async () => onDiskRows;
  cachedCalls = 0;
  assert.equal(await runner.translateModelDownloaded(fp8Rows[0]), true);
  assert.equal(await runner.translateModelDownloaded(fp8Rows[0]), true);
  assert.equal(cachedCalls, 2, "no listing is kept between separate calls");
  const gguf = DEDICATED_TRANSLATE_MODELS.find((model) => !model.fp8);
  assert.ok(gguf?.variant);
});

test("the page shares one listing across the rows of each probe pass", () => {
  const page = readText("../src/features/translate/translate-page.tsx");
  assert.match(
    page,
    /const cachedModels = sharedCachedModelsListing\(abort\.signal\);\s*(?:\/\/[^\n]*\n\s*)*for \(const model of offeredModels\) \{[\s\S]*?translateModelDownloaded\(model, abort\.signal, \{\s*remote: downloadedRef\.current\[model\.id\] !== false,\s*cachedModels,/,
  );
});

test("the page re-probes the disk when the model cache changes elsewhere", () => {
  // RAM-only polls skip the pass, so an external download must still flip the row to downloaded.
  const page = readText("../src/features/translate/translate-page.tsx");
  assert.match(page, /const inventoryVersion = useInventoryVersion\(\);/);
  assert.match(
    page,
    /for \(const model of offeredModels\) \{[\s\S]*?return \(\) => abort\.abort\(\);\s*\}, \[busy, offeredModels, inventoryVersion\]\);/,
  );
});
