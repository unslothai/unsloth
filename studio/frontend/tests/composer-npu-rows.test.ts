// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type * as NpuApi from "../src/features/npu/api.ts";
import type * as NpuRows from "../src/components/assistant-ui/composer-model-picker/npu-rows.ts";
import * as pickerModel from "../src/components/assistant-ui/composer-model-picker/picker-model.ts";
import * as displayName from "../src/features/npu/display-name.ts";
import * as runtimeVersions from "../src/features/npu/runtime-versions.ts";
import * as formatFastApiError from "../src/lib/format-fastapi-error.ts";
import * as sseJsonEvents from "../src/lib/sse-json-events.ts";
import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const api = loadWithStubs<typeof NpuApi>(new URL("../src/features/npu/api.ts", import.meta.url), {
  "@/features/auth/api": { authFetch: async () => new Response(null) },
  "@/lib/format-fastapi-error": formatFastApiError,
  "@/lib/sse-json-events": sseJsonEvents,
  "./display-name": displayName,
  "./runtime-versions": runtimeVersions,
});
const loadRows = () =>
  loadWithStubs<typeof NpuRows>(
    new URL("../src/components/assistant-ui/composer-model-picker/npu-rows.ts", import.meta.url),
    {
      "@/features/npu": { npuDisplayName: displayName.npuDisplayName, npuRowsFor: api.npuRowsFor },
      "./picker-model": pickerModel,
    },
  ).npuPickerRows;

const model = (id: string, checkpoint: string, hf_repo: string, downloaded = false) =>
  ({
    id,
    model_path: `lemonade:${id}`,
    checkpoint,
    hf_repo,
    size_gb: 1.5,
    downloaded,
    labels: [],
    supports_vision: false,
    supports_reasoning: false,
    supports_tools: false,
    max_context_length: null,
    resume_percent: null,
  }) as NpuApi.NpuModel;

// The backend's alphabetical order, as run 43900429635 showed it in the composer picker.
const catalog = [
  model("d3-edge-FLM", "d3:edge", "FastFlowLM/D3-Edge-NPU2"),
  model("lfm3-1.7b-FLM", "lfm3:1.7b", "FastFlowLM/LFM3-1.7B-NPU2"),
  model("qwen3-0.6b-FLM", "qwen3:0.6b", "FastFlowLM/Qwen3-0.6B-NPU2", true),
  model("qwen3.6-moe-35b-a3b-FLM", "qwen3.6-moe:35b-a3b", "FastFlowLM/Qwen3.6-35B-A3B-NPU2"),
  model("qwen3.8-mtp-27b-FLM", "qwen3.8-mtp:27b", "FastFlowLM/Qwen3.8-27B-NPU2"),
];

test("the composer picker's NPU rows carry the Hub's names and popular order", () => {
  const npuPickerRows = loadRows();
  const { device, offers } = npuPickerRows(catalog);
  assert.deepEqual(
    offers.map((row) => row.label),
    ["Qwen3.6-35B-A3B-NPU2", "Qwen3.8-27B-NPU2", "LFM3-1.7B-NPU2", "D3-Edge-NPU2"],
  );
  assert.deepEqual(
    offers.map((row) => [row.key, row.id, row.tab, row.kind, row.section]),
    offers.map((row) => [
      `npu-offer:${row.npu.id}`,
      row.npu.model_path,
      pickerModel.RECOMMENDED_TAB,
      "npu",
      "npu",
    ]),
  );
  assert.deepEqual(
    device.map((row) => [row.key, row.label, row.tab, row.sizeBytes]),
    [["npu:qwen3-0.6b-FLM", "Qwen3-0.6B-NPU2", pickerModel.DEVICE_TAB, 1.5e9]],
  );
  assert.deepEqual(npuPickerRows(null), { device: [], offers: [] });
});

test("the composer picker builds its NPU rows through the shared helper, not by model id", () => {
  const src = readSrc("components/assistant-ui/composer-model-picker/composer-model-picker.tsx");
  assert.match(src, /npuPickerRows\(npuModels\)/);
  assert.doesNotMatch(src, /label: model\.id/);
});
