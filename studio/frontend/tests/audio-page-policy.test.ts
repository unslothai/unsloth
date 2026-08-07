// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  canTransitionAudioMode,
  exactGgufLoadSelector,
  expectedGgufDownloadBytes,
  macTtsPickAction,
  micStreamRequestIsCurrent,
  newlyPersistedClip,
  reconcileSttSelection,
  resolveSttLoadedModel,
  resolveSttResidency,
  selectAutoGgufVariant,
  stagedTtsLoadIsOwned,
  sttSelectionReady,
} from "../src/features/audio/audio-page-policy.ts";

const audioPageSource = readFileSync(
  new URL("../src/features/audio/audio-page.tsx", import.meta.url),
  "utf8",
);

test("mode transitions cancel generation but wait for non-cancellable work", () => {
  assert.equal(canTransitionAudioMode(null), true);
  assert.equal(canTransitionAudioMode("generating"), true);
  assert.equal(canTransitionAudioMode("loading"), false);
  assert.equal(canTransitionAudioMode("unloading"), false);
  assert.equal(canTransitionAudioMode("transcribing"), false);
});

test("staged TTS completion requires the same Speak ownership generation", () => {
  assert.equal(stagedTtsLoadIsOwned(4, 4, "speak"), true);
  assert.equal(stagedTtsLoadIsOwned(4, 5, "speak"), false);
  assert.equal(stagedTtsLoadIsOwned(4, 4, "transcribe"), false);
  assert.equal(stagedTtsLoadIsOwned(null, 4, "speak"), false);
});

test("cached GGUF quant labels remain exact when no filename is present", () => {
  assert.equal(exactGgufLoadSelector({ ggufVariant: "Q4_K_M" }), "Q4_K_M");
  assert.equal(
    exactGgufLoadSelector({
      ggufVariant: "Q4_K_M",
      ggufFilename: "model-q4_k_m.gguf",
    }),
    "model-q4_k_m.gguf",
  );
});

test("Mac rejects safetensors-only TTS and redirects sibling families", () => {
  assert.equal(
    macTtsPickAction({ isMac: true, isGguf: false, ggufSibling: null }),
    "reject",
  );
  assert.equal(
    macTtsPickAction({
      isMac: true,
      isGguf: false,
      ggufSibling: "org/model-GGUF",
    }),
    "use-gguf-sibling",
  );
  assert.equal(
    macTtsPickAction({ isMac: true, isGguf: true, ggufSibling: null }),
    "allow",
  );
});

test("Mac sibling resolution returns an exact managed-download file", () => {
  const variants = [
    {
      filename: "model-q4.gguf",
      quant: "Q4_K_M",
      size_bytes: 4_000,
      download_size_bytes: 3_500,
      downloaded: false,
    },
    {
      filename: "model-q8.gguf",
      quant: "Q8_0",
      size_bytes: 8_000,
      downloaded: true,
    },
  ];
  const cached = selectAutoGgufVariant(variants, "Q4_K_M");
  assert.equal(cached?.filename, "model-q8.gguf");

  const remote = selectAutoGgufVariant(
    variants.map((variant) => ({ ...variant, downloaded: false })),
    "Q4_K_M",
  );
  assert.equal(remote?.filename, "model-q4.gguf");
  assert.equal(remote && expectedGgufDownloadBytes(remote), 3_500);
});

test("old gallery history never impersonates the new generated clip", () => {
  const before = new Set(["old"]);
  assert.equal(newlyPersistedClip(before, [{ id: "old" }]), null);
  assert.deepEqual(newlyPersistedClip(before, [{ id: "new" }, { id: "old" }]), {
    id: "new",
  });
});

test("STT controls require the selected sidecar to be resident", () => {
  const keyFor = (repo: string) => (repo === "org/asr" ? "asr" : repo);
  assert.equal(sttSelectionReady("org/asr", "asr", keyFor), true);
  assert.equal(sttSelectionReady("org/asr", null, keyFor), false);
  assert.equal(sttSelectionReady("org/asr", "other", keyFor), false);
  assert.equal(
    sttSelectionReady("org/asr", "asr", keyFor, "transformers", "gguf"),
    false,
  );
});

test("STT activation adopts resident sidecars and preserves only pending selections", () => {
  const sidecarKeyFor = (repo: string) =>
    repo === "org/asr-a" ? "asr-a" : repo === "org/asr-b" ? "asr-b" : repo;
  const repoIdForSidecarKey = (key: string) =>
    key === "asr-a" ? "org/asr-a" : key === "asr-b" ? "org/asr-b" : key;

  assert.equal(
    reconcileSttSelection({
      selectedRepo: null,
      loadedModel: "asr-b",
      preservePending: false,
      sidecarKeyFor,
      repoIdForSidecarKey,
    }),
    "org/asr-b",
  );
  assert.equal(
    reconcileSttSelection({
      selectedRepo: "org/asr-a",
      loadedModel: "asr-b",
      preservePending: false,
      sidecarKeyFor,
      repoIdForSidecarKey,
    }),
    "org/asr-b",
  );
  assert.equal(
    reconcileSttSelection({
      selectedRepo: "org/asr-a",
      loadedModel: null,
      preservePending: true,
      sidecarKeyFor,
      repoIdForSidecarKey,
    }),
    "org/asr-a",
  );
  assert.equal(
    reconcileSttSelection({
      selectedRepo: "org/asr-a",
      loadedModel: null,
      preservePending: false,
      sidecarKeyFor,
      repoIdForSidecarKey,
    }),
    null,
  );

  assert.equal(
    reconcileSttSelection({
      selectedRepo: "unsloth/whisper-small",
      loadedModel: "small",
      loadedEngine: "gguf",
      preservePending: false,
      sidecarKeyFor: () => "small",
      repoIdForSidecarKey: (_key, engine) =>
        engine === "gguf"
          ? "unslothai/whisper-small-GGUF"
          : "unsloth/whisper-small",
      engineForRepo: (repo) =>
        repo.endsWith("-GGUF") ? "gguf" : "transformers",
    }),
    "unslothai/whisper-small-GGUF",
  );
});

test("STT residency reads the selected engine instead of Transformers-only legacy fields", () => {
  const qwenStatus = {
    loaded_model: null,
    loading: false,
    transformers: { loaded_model: null, loading: false },
    gguf: { loaded_model: null, loading: false },
    mtmd: { loaded_model: "qwen3-asr-0.6b", loading: false },
  };
  assert.equal(
    resolveSttLoadedModel(qwenStatus, "mtmd", false),
    "qwen3-asr-0.6b",
  );

  const whisperStatus = {
    ...qwenStatus,
    mtmd: { loaded_model: null, loading: false },
    gguf: { loaded_model: "small", loading: false },
  };
  assert.equal(resolveSttLoadedModel(whisperStatus, null, false), "small");
  assert.deepEqual(resolveSttResidency(whisperStatus, "gguf", false), {
    model: "small",
    engine: "gguf",
  });
});

test("a pending engine selection is not replaced by an older resident sidecar", () => {
  const status = {
    loaded_model: "small",
    loading: false,
    transformers: { loaded_model: "small", loading: false },
    mtmd: { loaded_model: null, loading: false },
  };
  assert.equal(resolveSttLoadedModel(status, "mtmd", true), null);
  assert.equal(resolveSttLoadedModel(status, "mtmd", false), "small");
});

test("a resolved microphone permission stream is accepted only by its live request", () => {
  assert.equal(micStreamRequestIsCurrent(4, 4, true), true);
  assert.equal(micStreamRequestIsCurrent(4, 5, true), false);
  assert.equal(micStreamRequestIsCurrent(4, 4, false), false);
});

test("MediaRecorder setup failures release the acquired microphone stream", () => {
  assert.match(
    audioPageSource,
    /recorder\.start\(\);[\s\S]*?catch \{[\s\S]*?stopRecordStream\(\);/,
  );
});
