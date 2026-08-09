// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The browser SPA is served by the same process that answers /api, so it can
// never be older than its backend. The desktop app can: it ships its own
// frontend bundle, is versioned separately from the pip wheel, and adopts an
// already-running server (commands.rs, start_managed_server). Against that, the
// indicator is the widest reader in the app -- it touches four /status
// endpoints, and /video/status plus the STT mtmd block are only days old.
//
// So every one of these is a real desktop-app-newer-than-backend shape, plus the
// forward direction: a backend that grows a field must not break a frontend that
// has never heard of it.

import assert from "node:assert/strict";
import test from "node:test";

import {
  type SttStatusResponse,
  describeDiffusionStatus,
  describeInferenceStatus,
  describeSttStatus,
  describeVideoStatus,
  mergeLoadedModels,
  sttEngineStatus,
} from "../src/features/loaded-models/loaded-models-sources.ts";

// A read that failed for any reason -- 404 on a route that did not exist yet,
// 401/403 on an expired token, a 500, or the 10s timeout -- reaches the mappers
// as null, because settled() collapses them all.
const UNREACHABLE = null;

test("a backend with no video route at all still lists the other runtimes", () => {
  // /api/inference/video/status landed 2026-08-04. Before that it 404s, which
  // parseJson throws on and settled() turns into null.
  const rows = mergeLoadedModels([
    describeInferenceStatus({
      active_model: "unsloth/Qwen3-4B-GGUF",
      loaded: ["unsloth/Qwen3-4B-GGUF"],
      is_gguf: true,
      gguf_variant: "Q4_K_M",
    } as never),
    describeDiffusionStatus(UNREACHABLE),
    describeVideoStatus(UNREACHABLE),
    describeSttStatus(UNREACHABLE),
  ]);
  assert.equal(rows.length, 1, "one dead runtime must not blank the others");
  assert.equal(rows[0].name, "unsloth/Qwen3-4B-GGUF");
});

test("every runtime unreachable is an empty list, never a crash", () => {
  assert.deepEqual(
    mergeLoadedModels([
      describeInferenceStatus(UNREACHABLE),
      describeDiffusionStatus(UNREACHABLE),
      describeVideoStatus(UNREACHABLE),
      describeSttStatus(UNREACHABLE),
    ]),
    [],
  );
});

test("a pre-split dictation backend reports through the legacy fields", () => {
  // Before 2026-07-23 there were no per-engine blocks: the resident Transformers
  // model appeared only at the top level.
  const rows = describeSttStatus({
    loaded_model: "large-v3",
    device: "cuda",
  } as SttStatusResponse);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].name, "large-v3");
  assert.equal(rows[0].sttEngine, "transformers");
  assert.equal(rows[0].detail, "Transformers · cuda");
});

test("a current backend does not double the dictation row", () => {
  // Both the legacy top level and the engine block are present on every current
  // server, and they hold the same model. The block must win.
  const rows = describeSttStatus({
    loaded_model: "large-v3",
    device: "cuda",
    transformers: { loaded_model: "large-v3", device: "cuda" },
  } as SttStatusResponse);
  assert.equal(rows.length, 1);
});

test("the legacy fallback is transformers-only", () => {
  // The top-level fields are the Transformers sidecar's, character for
  // character -- not a "last engine used" -- so they must not stand in for the
  // llama.cpp or whisper.cpp sidecars.
  const status = { loaded_model: "large-v3", device: "cuda" } as SttStatusResponse;
  assert.equal(sttEngineStatus(status, "transformers")?.loaded_model, "large-v3");
  assert.equal(sttEngineStatus(status, "mtmd"), null);
  assert.equal(sttEngineStatus(status, "gguf"), null);
});

test("a dictation backend without the mtmd engine skips it", () => {
  // The mtmd (Qwen3-ASR) block arrived 2026-08-04, after the other two.
  const rows = describeSttStatus({
    transformers: { loaded_model: null, device: null },
    gguf: { loaded_model: "ggml-base.en", device: "whisper.cpp" },
  } as SttStatusResponse);
  assert.deepEqual(
    rows.map((row) => row.sttEngine),
    ["gguf"],
  );
});

test("an engine block explicitly nulled is skipped, not read as legacy", () => {
  const rows = describeSttStatus({
    loaded_model: "large-v3",
    device: "cuda",
    transformers: null,
  } as SttStatusResponse);
  // transformers: null means "no such block", so the legacy fallback applies.
  assert.equal(rows.length, 1);
  assert.equal(rows[0].name, "large-v3");
});

test("a chat payload missing every optional field still renders", () => {
  // The oldest shape this has to survive: a name and nothing else.
  const rows = describeInferenceStatus({
    active_model: "unsloth/Qwen3-4B",
  } as never);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].detail, "Transformers", "the ladder needs no flags");
  assert.equal(rows[0].kind, "text");
});

test("a diffusion payload missing dtype, device and family still renders", () => {
  const rows = describeDiffusionStatus({
    loaded: true,
    repo_id: "black-forest-labs/FLUX.1-dev",
  } as never);
  assert.equal(rows.length, 1);
  assert.equal(rows[0].detail, "", "no parts is an empty line, not a stray dot");
  assert.equal(rows[0].name, "black-forest-labs/FLUX.1-dev");
});

test("undefined and null are the same absence", () => {
  const withNulls = describeVideoStatus({
    loaded: true,
    repo_id: "Wan-AI/Wan2.2-T2V-A14B",
    family: null,
    device: null,
    dtype: null,
    transformer_quant: null,
  } as never);
  const withUndefined = describeVideoStatus({
    loaded: true,
    repo_id: "Wan-AI/Wan2.2-T2V-A14B",
  } as never);
  assert.deepEqual(withNulls, withUndefined);
});

test("empty strings are dropped rather than printed as separators", () => {
  const rows = describeDiffusionStatus({
    loaded: true,
    repo_id: "x/y",
    family: "",
    device: "cuda",
    dtype: "",
  } as never);
  assert.equal(rows[0].detail, "cuda");
});

test("fields a future backend adds are ignored, not rendered", () => {
  // Forward compatibility: an old desktop bundle against a newer wheel.
  const rows = describeDiffusionStatus({
    loaded: true,
    repo_id: "x/y",
    family: "flux",
    device: "cuda",
    dtype: "bfloat16",
    some_future_field: "should not appear",
    nested: { also: "ignored" },
  } as never);
  assert.equal(rows[0].detail, "flux · BF16 · cuda");
});

test("an unrecognised precision is passed through rather than dropped", () => {
  // precisionLabel upper-cases anything it does not know, so a quant added
  // later still tells the user something instead of vanishing.
  const rows = describeVideoStatus({
    loaded: true,
    repo_id: "x/y",
    family: "wan",
    device: "cuda",
    transformer_quant: "nvfp4",
  } as never);
  assert.equal(rows[0].detail, "wan · NVFP4 · cuda");
});

test("a chat runtime caching past the active model marks the extras inactive", () => {
  // Only the Transformers backend can do this, and only the active model is
  // ejectable by the normal path -- the rest need naming directly.
  const rows = describeInferenceStatus({
    active_model: "unsloth/Qwen3-4B",
    loaded: ["unsloth/Qwen3-4B", "unsloth/Llama-3.2-3B"],
  } as never);
  assert.equal(rows.length, 2);
  assert.equal(rows[0].inactive, undefined);
  assert.equal(rows[1].inactive, true);
  assert.equal(rows[1].detail, "Still in memory");
});

test("a duplicate in the loaded list is not listed twice", () => {
  const rows = describeInferenceStatus({
    active_model: "unsloth/Qwen3-4B",
    loaded: ["unsloth/Qwen3-4B", "unsloth/Llama-3.2-3B", "unsloth/Llama-3.2-3B"],
  } as never);
  assert.equal(rows.length, 2);
});

test("the same row arriving from two sources is merged once", () => {
  const row = describeInferenceStatus({
    active_model: "unsloth/Qwen3-4B",
  } as never);
  assert.equal(mergeLoadedModels([row, row]).length, 1);
});
