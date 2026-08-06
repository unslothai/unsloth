// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Recommended seeds curated rows in the order the picker is handed them, and the
// list scrolls: whichever task trails is the one nobody sees. These pin that the
// active mode's task leads, and that the curated STT set still covers every
// dictation sidecar id.

import assert from "node:assert/strict";
import test from "node:test";

import {
  AUDIO_CATALOG,
  catalogToModelOptions,
  groupForRepoId,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";

type AudioTask = "tts" | "stt";

const taskFor = (repoId: string): AudioTask | null =>
  (groupForRepoId(repoId, AUDIO_CATALOG)?.task as AudioTask | undefined) ?? null;

// Mirrors audioModelsForTask in src/features/audio/catalog.ts, which cannot be
// imported here: it resolves through the "@/" alias.
const modelsForTask = (task: AudioTask) => {
  const all = catalogToModelOptions(AUDIO_CATALOG);
  const matches = (id: string) => taskFor(id) === task;
  return [
    ...all.filter((o) => matches(o.id)),
    ...all.filter((o) => !matches(o.id)),
  ].map((o) => o.id);
};

test("Transcribe leads with STT, so Qwen3-ASR is not below the fold", () => {
  const rows = modelsForTask("stt");
  assert.equal(taskFor(rows[0]), "stt");
  // The regression: both unslothai models sat at 8 and 9 behind every TTS row.
  const qwen = rows.filter((id) => id.startsWith("unslothai/Qwen3-ASR"));
  assert.equal(qwen.length, 2);
  for (const id of qwen) assert.ok(rows.indexOf(id) < 3, `${id} buried`);
});

test("Generate leads with TTS", () => {
  const rows = modelsForTask("tts");
  assert.equal(taskFor(rows[0]), "tts");
  assert.ok(rows.indexOf("unsloth/orpheus-3b-0.1-ft") < 2);
});

test("both modes offer every curated model, only reordered", () => {
  const all = catalogToModelOptions(AUDIO_CATALOG).map((o) => o.id);
  for (const task of ["tts", "stt"] as const) {
    assert.deepEqual([...modelsForTask(task)].sort(), [...all].sort());
  }
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
