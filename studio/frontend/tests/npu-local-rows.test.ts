// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as NpuApi from "../src/features/npu/api.ts";
import * as displayName from "../src/features/npu/display-name.ts";
import * as formatFastApiError from "../src/lib/format-fastapi-error.ts";
import * as sseJsonEvents from "../src/lib/sse-json-events.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const api = loadWithStubs<typeof NpuApi>(
  new URL("../src/features/npu/api.ts", import.meta.url),
  {
    "@/features/auth": { authFetch: async () => new Response(null) },
    "@/features/auth/api": { authFetch: async () => new Response(null) },
    "@/lib/format-fastapi-error": formatFastApiError,
    "@/lib/sse-json-events": sseJsonEvents,
    "./display-name": displayName,
  },
);

function model(
  id: string,
  local: NpuApi.NpuLocalModel | null = null,
): NpuApi.NpuModel {
  return {
    id,
    model_path: `lemonade:${id}`,
    checkpoint: id.replace("-FLM", ""),
    size_gb: 0.66,
    downloaded: true,
    labels: ["chat"],
    supports_vision: false,
    supports_reasoning: false,
    supports_tools: false,
    max_context_length: 40960,
    resume_percent: null,
    local,
  };
}

const converted = model("local-qwen3-0.6b-ft-1a2b3c4d-q4nx-FLM", {
  folder: "q4nx/Qwen3-0.6B-ft.Q4_0-q4nx",
  base: "qwen3:0.6b",
  problem: null,
});
const broken = model("local-half-5e6f7a8b-q4nx-FLM", {
  folder: "q4nx/half",
  base: null,
  problem: "The folder is incomplete: tokenizer.json is missing.",
});
const catalog = model("qwen3-0.6b-FLM");

test("a converted folder is named after the folder and marked Converted", () => {
  assert.equal(displayName.npuDisplayName(converted), "Qwen3-0.6B-ft.Q4_0-q4nx");
  assert.deepEqual(api.npuRowDetails(converted), [
    "NPU",
    "Converted",
    "660 MB",
  ]);
  assert.equal(
    api.npuRowTooltip(converted),
    "Converted on the Export page, from exports/q4nx/Qwen3-0.6B-ft.Q4_0-q4nx",
  );
});

test("catalog rows keep their id and details", () => {
  assert.equal(displayName.npuDisplayName(catalog), "qwen3-0.6b-FLM");
  assert.deepEqual(api.npuRowDetails(catalog), ["NPU", "660 MB"]);
  assert.equal(api.npuRowTooltip(catalog), null);
});

test("a row too large for the NPU memory window keeps its warning", () => {
  assert.deepEqual(api.npuRowDetails({ ...catalog, fits_memory: false }), [
    "NPU",
    "660 MB",
    "May not fit NPU memory",
  ]);
});

test("a conversion saved outside the exports folder shows its full path", () => {
  for (const folder of ["/mnt/models/Qwen3-q4nx", "D:/models/Qwen3-q4nx"]) {
    const external = model("local-qwen3-q4nx-9c8d7e6f-q4nx-FLM", {
      folder,
      base: "qwen3:0.6b",
      problem: null,
    });
    assert.equal(displayName.npuDisplayName(external), "Qwen3-q4nx");
    assert.equal(
      api.npuRowTooltip(external),
      `Converted on the Export page, from ${folder}`,
    );
  }
});

test("a GGUF export's Q4NX folder is named after its save folder", () => {
  for (const folder of [
    "Qwen3-0.6B-ft/npu-q4nx",
    "/mnt/models/Qwen3-0.6B-ft/npu-q4nx",
  ]) {
    const exported = model("local-npu-q4nx-0a1b2c3d-q4nx-FLM", {
      folder,
      base: "qwen3:0.6b",
      problem: null,
    });
    assert.equal(displayName.npuDisplayName(exported), "Qwen3-0.6B-ft");
  }
});

test("an unloadable folder says why on hover", () => {
  assert.equal(
    api.npuRowTooltip(broken),
    "Can't load q4nx/half: The folder is incomplete: tokenizer.json is missing.",
  );
});

test("converted folders are on device and found by their folder name", () => {
  const rows = [catalog, converted, broken];
  const ids = (query: string) =>
    api.npuRowsFor(rows, { onDevice: true, query }).map((m) => m.id);
  assert.deepEqual(ids(""), [catalog.id, converted.id, broken.id]);
  assert.deepEqual(ids("ft.q4_0"), [converted.id]);
});

test("deleting a converted row says the export stays", () => {
  assert.match(api.npuDeleteDescription(converted), /exported folder stays/);
  assert.match(api.npuDeleteDescription(catalog), /NPU files/);
});

test("the composer picker re-reads the NPU list each time it opens on a ready NPU", async () => {
  const { readSrcAsync } = await import("./helpers/kit.ts");
  const picker = await readSrcAsync(
    "components/assistant-ui/composer-model-picker/composer-model-picker.tsx",
  );
  assert.match(
    picker,
    /useEffect\(\(\) => \{\s*if \(open && npuReady\) void refreshNpuModels\(\);\s*\}, \[open, npuReady\]\);/,
  );
  assert.match(picker, /npuStatus\?\.supported === true && npuStatus\.ready === true/);
});
