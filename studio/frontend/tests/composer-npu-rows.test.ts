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
const { npuMarkOwner, npuNoticeShown, npuPickerRows } = loadWithStubs<
  typeof NpuRows
>(
  new URL(
    "../src/components/assistant-ui/composer-model-picker/npu-rows.ts",
    import.meta.url,
  ),
  {
    "@/features/npu": { ...api, ...displayName },
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

test("On Device's Recent sort keeps never-loaded NPU rows in the Hub's order", () => {
  const rows = npuPickerRows(backend, true);
  const recent = pickerModel
    .sortDeviceRows(rows, "recent", () => 0)
    .map((row) => row.label);
  assert.deepEqual(
    recent,
    rows.map((row) => row.label),
  );
  // Mixed with undated rows that sort between them by name, the NPU rows still keep their order.
  const others = ["A", "H", "Qwen3.6", "Z"].map((label) => ({
    key: label,
    id: label,
    label,
    tab: "device",
    kind: "local" as const,
  }));
  const mixed = pickerModel
    .sortDeviceRows([...others, ...rows], "recent", () => 0)
    .filter((row) => row.kind === "npu")
    .map((row) => row.label);
  assert.deepEqual(
    mixed,
    rows.map((row) => row.label),
  );
  assert.deepEqual(
    pickerModel.sortDeviceRows(rows, "name", () => 0).map((row) => row.label),
    ["Gemma3-4B-IT-NPU2", "Qwen3.5-9B-NPU2", "Qwen3.8-27B-NPU2"],
  );
  const loaded = rows[2].id;
  assert.equal(
    pickerModel.sortDeviceRows(rows, "recent", (row) =>
      row.id === loaded ? 1 : 0,
    )[0].id,
    loaded,
  );
});

test("an NPU row is marked by its repo's Hugging Face org, not a letter", () => {
  const rows = npuPickerRows(backend, true);
  const named = rows.find((row) => row.npu.hf_repo);
  assert.equal(npuMarkOwner(named), "FastFlowLM");
  assert.equal(npuMarkOwner({ kind: "npu", npu: { hf_repo: null } }), "FastFlowLM");
  assert.equal(
    npuMarkOwner({ kind: "npu", npu: { hf_repo: "amd/Some-Model-NPU2" } }),
    "amd",
  );
  // The chip's loaded model has only its `lemonade:` path.
  assert.equal(
    npuMarkOwner({ kind: "loaded", id: "lemonade:qwen3.6-moe-35b-a3b-FLM" }),
    "FastFlowLM",
  );
  assert.equal(npuMarkOwner({ kind: "hub", id: "unsloth/Qwen3-4B-GGUF" }), null);
});

test("Discover keeps the NPU notice while a ready runtime lists its models or when listing fails", () => {
  assert.equal(npuNoticeShown(null), false);
  assert.equal(
    npuNoticeShown({ ready: false, listError: null, models: null }),
    true,
  );
  assert.equal(
    npuNoticeShown({ ready: true, listError: null, models: null }),
    true,
  );
  assert.equal(
    npuNoticeShown({ ready: true, listError: null, models: [] }),
    false,
  );
  assert.equal(
    npuNoticeShown({
      ready: true,
      listError: "Listing NPU models failed",
      models: null,
    }),
    true,
  );
});
