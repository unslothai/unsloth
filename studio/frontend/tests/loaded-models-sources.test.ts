// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

// The indicator .tsx pulls in the router, motion and hugeicons, so it cannot be
// imported here. The status to row mapping lives in a plain module, driven directly.
import { modelIdsMatch } from "../src/features/hub/lib/model-identity.ts";
import {
  type LoadedModelEntry,
  describeDiffusionStatus,
  describeInferenceStatus,
  describeSttStatus,
  describeVideoStatus,
  mergeLoadedModels,
  shortModelLabel,
  verifyResident,
} from "../src/features/loaded-models/loaded-models-sources.ts";

// Only the fields the mapping reads; the real responses carry dozens more.
function inferenceStatus(
  overrides: Record<string, unknown> = {},
): Parameters<typeof describeInferenceStatus>[0] {
  return {
    active_model: null,
    is_vision: false,
    loading: [],
    loaded: [],
    ...overrides,
  } as Parameters<typeof describeInferenceStatus>[0];
}

test("no runtime loaded produces no rows", () => {
  assert.deepEqual(describeInferenceStatus(inferenceStatus()), []);
  assert.deepEqual(describeDiffusionStatus({ loaded: false } as never), []);
  assert.deepEqual(describeVideoStatus({ loaded: false } as never), []);
  assert.deepEqual(describeSttStatus({}), []);
});

test("an unreachable runtime yields no rows rather than throwing", () => {
  assert.deepEqual(describeInferenceStatus(null), []);
  assert.deepEqual(describeDiffusionStatus(null), []);
  assert.deepEqual(describeVideoStatus(null), []);
  assert.deepEqual(describeSttStatus(null), []);
});

test("a GGUF chat model reports its variant", () => {
  const [row] = describeInferenceStatus(
    inferenceStatus({
      active_model: "unsloth/gemma-3-4b-it-GGUF",
      is_gguf: true,
      gguf_variant: "Q4_K_M",
      loaded: ["unsloth/gemma-3-4b-it-GGUF"],
    }),
  );
  assert.equal(row.kind, "text");
  assert.equal(row.source, "chat");
  assert.equal(row.detail, "GGUF · Q4_K_M");
});

// Same picker, same memory, but only one of them answers prompts.
test("an audio model is a speech row, and a whisper one is dictation", () => {
  const [tts] = describeInferenceStatus(
    inferenceStatus({
      active_model: "unsloth/orpheus-3b-0.1-ft",
      is_audio: true,
      audio_type: "tts",
    }),
  );
  assert.equal(tts.kind, "tts");
  const [stt] = describeInferenceStatus(
    inferenceStatus({
      active_model: "openai/whisper-large-v3",
      is_audio: true,
      audio_type: "whisper",
    }),
  );
  assert.equal(stt.kind, "stt");
  // Still the chat runtime's, so it ejects through /api/inference/unload.
  assert.equal(stt.source, "chat");
});

test("a model the runtime still holds besides the active one gets its own row", () => {
  const rows = describeInferenceStatus(
    inferenceStatus({
      active_model: "unsloth/Llama-3.2-3B",
      loaded: ["unsloth/Llama-3.2-3B", "unsloth/Qwen3-4B"],
    }),
  );
  assert.equal(rows.length, 2);
  assert.equal(rows[0].inactive, undefined);
  assert.equal(rows[1].name, "unsloth/Qwen3-4B");
  assert.equal(rows[1].inactive, true);
});

// A server predating the engine split reports only the top-level fields.
test("a legacy STT status still shows its resident Transformers model", () => {
  const rows = describeSttStatus({
    loaded_model: "openai/whisper-large-v3",
    device: "cuda",
  });
  assert.deepEqual(
    rows.map((row) => [row.sttEngine, row.name, row.detail]),
    [["transformers", "openai/whisper-large-v3", "Transformers · cuda"]],
  );
});

test("an engine block wins over the legacy fields, and never doubles a row", () => {
  const rows = describeSttStatus({
    loaded_model: "openai/whisper-large-v3",
    device: "cuda",
    transformers: { loaded_model: null },
  });
  assert.deepEqual(rows, []);
});

test("each STT engine that has a model resident gets a row naming its engine", () => {
  const rows = describeSttStatus({
    transformers: { loaded_model: null },
    mtmd: { loaded_model: "unsloth/voxtral-mini", device: "cuda" },
    gguf: { loaded_model: "ggml-base.en", device: "metal" },
  });
  assert.deepEqual(
    rows.map((row) => [row.sttEngine, row.name]),
    [
      ["mtmd", "unsloth/voxtral-mini"],
      ["gguf", "ggml-base.en"],
    ],
  );
  assert.equal(rows[0].detail, "llama.cpp · cuda");
});

// Those two sidecars report their engine name as the device, so the label and
// the device are the same string and must not print twice.
test("an engine that reports itself as its device is named once", () => {
  const rows = describeSttStatus({
    mtmd: { loaded_model: "qwen3-asr-0.6b", device: "llama.cpp" },
    gguf: { loaded_model: "ggml-base.en", device: "whisper.cpp" },
  });
  assert.deepEqual(
    rows.map((row) => row.detail),
    ["llama.cpp", "whisper.cpp"],
  );
});

test("a real device is still reported next to its engine", () => {
  const [row] = describeSttStatus({
    transformers: { loaded_model: "openai/whisper-large-v3", device: "cuda" },
  });
  assert.equal(row.detail, "Transformers · cuda");
});

test("image and video rows omit the parts the backend did not report", () => {
  const [image] = describeDiffusionStatus({
    loaded: true,
    repo_id: "unsloth/FLUX.1-dev",
    family: "flux",
    device: null,
  } as never);
  assert.equal(image.detail, "flux");
  const [video] = describeVideoStatus({
    loaded: true,
    repo_id: "unsloth/Wan2.2-T2V-A14B",
    family: "wan",
    model_kind: "gguf",
    device: "cuda",
  } as never);
  assert.equal(video.detail, "wan · GGUF · cuda");
});

test("every runtime's rows appear together, in a fixed order", () => {
  const merged = mergeLoadedModels([
    describeInferenceStatus(
      inferenceStatus({
        active_model: "unsloth/orpheus-3b-0.1-ft",
        is_audio: true,
      }),
    ),
    describeDiffusionStatus({
      loaded: true,
      repo_id: "unsloth/FLUX.1-dev",
    } as never),
    describeVideoStatus(null),
    describeSttStatus({ gguf: { loaded_model: "ggml-base.en" } }),
  ]);
  assert.deepEqual(
    merged.map((row) => row.kind),
    ["tts", "image", "stt"],
  );
});

test("one runtime naming the same model twice is still one row", () => {
  const duplicated: LoadedModelEntry[] = [
    {
      id: "chat:unsloth/Qwen3-4B",
      kind: "text",
      source: "chat",
      name: "unsloth/Qwen3-4B",
      detail: "GGUF",
    },
  ];
  assert.equal(mergeLoadedModels([duplicated, duplicated]).length, 1);
});

// /images/unload, /video/unload and the STT unload carry no model id, so a row
// up to one poll old must be checked against the runtime before either fires.
test("a runtime holding the row's model is safe to unload", () => {
  assert.equal(
    verifyResident("unsloth/FLUX.1-dev", "unsloth/FLUX.1-dev", modelIdsMatch),
    "match",
  );
});

test("a runtime holding something else must not be unloaded", () => {
  assert.equal(
    verifyResident("unsloth/FLUX.1-dev", "unsloth/Qwen-Image", modelIdsMatch),
    "replaced",
  );
});

test("an idle runtime is already free, so there is nothing to unload", () => {
  assert.equal(
    verifyResident("unsloth/FLUX.1-dev", null, modelIdsMatch),
    "gone",
  );
  assert.equal(
    verifyResident("unsloth/FLUX.1-dev", undefined, modelIdsMatch),
    "gone",
  );
});

// These runtimes report repo_id / loaded_model, the same fields the rows were
// built from, so matching is exact bar the tolerance modelIdsMatch already has.
// A spurious "replaced" would refuse a legitimate eject, so pin that too.
test("a trailing separator or casing difference is not a replacement", () => {
  assert.equal(
    verifyResident("/models/flux", "/models/flux/", modelIdsMatch),
    "match",
  );
  assert.equal(
    verifyResident("unsloth/FLUX.1-dev", "unsloth/flux.1-dev", modelIdsMatch),
    "match",
  );
});

test("a local load shows its model folder rather than leading directories", () => {
  assert.equal(
    shortModelLabel("unsloth/gemma-3-4b-it"),
    "unsloth/gemma-3-4b-it",
  );
  assert.equal(
    shortModelLabel("/Users/me/models/hub/gemma-3-4b-it"),
    "hub/gemma-3-4b-it",
  );
  // Windows path, trailing separator: still the last two segments.
  assert.equal(shortModelLabel("C:\\models\\hub\\gemma\\"), "hub/gemma");
});
