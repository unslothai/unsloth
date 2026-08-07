// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Each mode advertises only checkpoints it can execute, while the curated STT
// set still covers every dictation sidecar id.

import assert from "node:assert/strict";
import test from "node:test";

import {
  AUDIO_CATALOG,
  catalogToModelOptions,
  groupForRepoId,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";

type AudioTask = "tts" | "stt";

const taskFor = (repoId: string): AudioTask | null =>
  (groupForRepoId(repoId, AUDIO_CATALOG)?.task as AudioTask | undefined) ??
  null;

// Mirrors audioModelsForTask in src/features/audio/catalog.ts, which cannot be
// imported here: it resolves through the "@/" alias.
const modelsForTask = (task: AudioTask) => {
  const all = catalogToModelOptions(AUDIO_CATALOG);
  const matches = (id: string) => taskFor(id) === task;
  return all.filter((o) => matches(o.id)).map((o) => o.id);
};

test("Transcribe offers only STT, with Qwen3-ASR above the fold", () => {
  const rows = modelsForTask("stt");
  assert.ok(rows.every((id) => taskFor(id) === "stt"));
  // The regression: both unslothai models sat at 8 and 9 behind every TTS row.
  const qwen = rows.filter((id) => id.startsWith("unslothai/Qwen3-ASR"));
  assert.equal(qwen.length, 2);
  for (const id of qwen) assert.ok(rows.indexOf(id) < 3, `${id} buried`);
});

test("Qwen3-ASR rows retain the fixed sidecar quant", () => {
  const qwen = catalogToModelOptions(AUDIO_CATALOG).filter((option) =>
    option.id.startsWith("unslothai/Qwen3-ASR"),
  );
  assert.equal(qwen.length, 2);
  assert.ok(qwen.every((option) => option.deviceQuant === "Q8_0"));
});

test("Generate offers only TTS", () => {
  const rows = modelsForTask("tts");
  assert.ok(rows.every((id) => taskFor(id) === "tts"));
  assert.ok(rows.indexOf("unsloth/orpheus-3b-0.1-ft") < 2);
});

test("the two modes partition every curated model", () => {
  const all = catalogToModelOptions(AUDIO_CATALOG).map((o) => o.id);
  const tts = modelsForTask("tts");
  const stt = modelsForTask("stt");
  assert.equal(
    tts.some((id) => stt.includes(id)),
    false,
  );
  assert.deepEqual([...tts, ...stt].sort(), [...all].sort());
});

test("the curated STT rows cover every dictation sidecar model", () => {
  // GGML_STT_REPOS / STT_MODEL_REPOS carry tiny and base too; the picker used to
  // stop at small, so two supported sizes were unreachable from this page.
  const stt = modelsForTask("stt").filter((id) => taskFor(id) === "stt");
  for (const size of ["tiny", "base", "small", "large-v3", "large-v3-turbo"]) {
    assert.ok(
      stt.includes(`unsloth/whisper-${size}`),
      `whisper-${size} missing from the audio picker`,
    );
  }
});

test("a TTS group's GGUF sibling is what resolves for a safetensors pick", () => {
  // On a Mac the safetensors build loads through MLX, which has no TTS decoder,
  // so the page swaps in the GGUF build rather than letting generation 501.
  const ggufSibling = (repoId: string) => {
    const group = groupForRepoId(repoId, AUDIO_CATALOG);
    if (!group || group.task !== "tts") return null;
    const gguf = group.artifacts.find((a) => a.format === "gguf");
    return gguf && gguf.repoId !== repoId ? gguf.repoId : null;
  };
  assert.equal(
    ggufSibling("unsloth/orpheus-3b-0.1-ft"),
    "unsloth/orpheus-3b-0.1-ft-GGUF",
  );
  // Already GGUF: nothing to swap.
  assert.equal(ggufSibling("unsloth/orpheus-3b-0.1-ft-GGUF"), null);
  // No GGUF published, so the pick stands and the 501 explains why.
  assert.equal(ggufSibling("unsloth/csm-1b"), null);
  // STT never swaps: it runs on the sidecar, not the main slot.
  assert.equal(ggufSibling("unsloth/whisper-small"), null);
});
