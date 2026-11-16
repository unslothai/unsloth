// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type * as NpuRows from "../src/components/assistant-ui/composer-model-picker/npu-rows.ts";
import * as pickerModel from "../src/components/assistant-ui/composer-model-picker/picker-model.ts";
import type * as NpuApi from "../src/features/npu/api.ts";
import * as displayName from "../src/features/npu/display-name.ts";
import * as runtimeVersions from "../src/features/npu/runtime-versions.ts";
import * as formatFastApiError from "../src/lib/format-fastapi-error.ts";
import * as sseJsonEvents from "../src/lib/sse-json-events.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const api = loadWithStubs<typeof NpuApi>(
  new URL("../src/features/npu/api.ts", import.meta.url),
  {
    "@/features/auth/api": { authFetch: async () => new Response(null) },
    "@/lib/format-fastapi-error": formatFastApiError,
    "./display-name": displayName,
    "@/lib/sse-json-events": sseJsonEvents,
    "./runtime-versions": runtimeVersions,
  },
);
const { npuPickerRows } = loadWithStubs<typeof NpuRows>(
  new URL(
    "../src/components/assistant-ui/composer-model-picker/npu-rows.ts",
    import.meta.url,
  ),
  {
    "@/features/npu/api": api,
    "@/features/npu/display-name": displayName,
    "./picker-model": pickerModel,
  },
);

// FastFlowLM lists by id, alphabetically.
const backend = [
  ["d3-edge-FLM", "d3:edge", null],
  ["gemma3-4b-FLM", "gemma3:4b", "FastFlowLM/Gemma3-4B-IT-NPU2"],
  ["lfm3-1.7b-FLM", "lfm3:1.7b", "FastFlowLM/LFM3-1.7B-NPU2"],
  ["qwen3.5-9b-FLM", "qwen3.5:9b", "FastFlowLM/Qwen3.5-9B-NPU2"],
  [
    "qwen3.6-moe-35b-a3b-FLM",
    "qwen3.6-moe:35b-a3b",
    "FastFlowLM/Qwen3.6-35B-A3B-NPU2",
  ],
  ["qwen3.8-mtp-27b-FLM", "qwen3.8-mtp:27b", "FastFlowLM/Qwen3.8-27B-NPU2"],
].map(([id, checkpoint, hf_repo], i) => ({
  id: id as string,
  model_path: `lemonade:${id}`,
  checkpoint: checkpoint as string,
  hf_repo,
  size_gb: i + 1,
  downloaded: i % 2 === 1,
  labels: [],
  supports_vision: false,
  supports_reasoning: false,
  supports_tools: false,
  max_context_length: null,
  resume_percent: null,
}));

test("the composer's NPU rows take the Hub's names and order and load by the -FLM path", () => {
  const all = [true, false].flatMap((onDevice) =>
    npuPickerRows(backend, onDevice),
  );
  const hub = api.npuRowsFor(backend, { onDevice: false, query: "" });
  const shown = (onDevice: boolean) =>
    npuPickerRows(backend, onDevice).map((row) => [row.label, row.id]);

  assert.deepEqual(shown(false), [
    ["Qwen3.6-35B-A3B-NPU2", "lemonade:qwen3.6-moe-35b-a3b-FLM"],
    ["LFM3-1.7B-NPU2", "lemonade:lfm3-1.7b-FLM"],
    ["d3-edge-FLM", "lemonade:d3-edge-FLM"],
  ]);
  assert.deepEqual(shown(true), [
    ["Qwen3.8-27B-NPU2", "lemonade:qwen3.8-mtp-27b-FLM"],
    ["Qwen3.5-9B-NPU2", "lemonade:qwen3.5-9b-FLM"],
    ["Gemma3-4B-IT-NPU2", "lemonade:gemma3-4b-FLM"],
  ]);
  for (const row of all) {
    assert.equal(row.label, displayName.npuDisplayName(row.npu));
    assert.equal(row.tab, row.npu.downloaded ? "device" : "recommended");
    assert.equal(row.section, "npu");
  }
  // Both lists keep the Hub's relative order.
  for (const onDevice of [true, false]) {
    assert.deepEqual(
      npuPickerRows(backend, onDevice).map((row) => row.npu.id),
      hub.filter((m) => m.downloaded === onDevice).map((m) => m.id),
    );
  }
  assert.equal(npuPickerRows(backend, true)[0].sizeBytes, 6e9);
  assert.equal(npuPickerRows(backend, false)[0].sizeBytes, undefined);
  assert.deepEqual(npuPickerRows(null, false), []);
});
