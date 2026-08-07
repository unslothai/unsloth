// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Enabling community models let the picker return audio repos with no catalog
// entry. These pin the classification and filtering that has to hold for them.

import assert from "node:assert/strict";
import test from "node:test";

import {
  AUDIO_CATALOG,
  groupForRepoId,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";
import { resolveAudioPickTask } from "../src/features/audio/audio-page-policy.ts";

type Task = "tts" | "stt" | null;

const routeFor = (repoId: string, pipelineTag?: string | null): Task =>
  resolveAudioPickTask(
    (groupForRepoId(repoId, AUDIO_CATALOG)?.task as Task) ?? null,
    pipelineTag,
  );

test("an uncurated ASR repo routes to the STT sidecar, not the TTS slot", () => {
  // The regression: audioTaskFor returns null off-catalog, so this fell through
  // to TTS and loaded a Whisper checkpoint into the main inference slot.
  assert.equal(
    routeFor("openai/whisper-large-v3", "automatic-speech-recognition"),
    "stt",
  );
  assert.equal(
    routeFor("nvidia/parakeet-tdt", "automatic-speech-recognition"),
    "stt",
  );
});

test("an uncurated TTS repo still takes the main slot", () => {
  assert.equal(routeFor("hexgrad/Kokoro-82M", "text-to-speech"), null);
  // No tag at all (a pasted repo id) stays on the TTS path, which /load validates.
  assert.equal(routeFor("someone/mystery-model", null), null);
});

test("the catalog still wins over the tag", () => {
  // A curated STT row routes by catalog even with no tag on the pick.
  assert.equal(routeFor("unsloth/whisper-small", null), "stt");
  assert.equal(routeFor("unslothai/Qwen3-ASR-0.6B-GGUF", null), "stt");
  assert.equal(routeFor("unsloth/orpheus-3b-0.1-ft", null), "tts");
});

// Mirrors keepCommunity vs keep: community rows have no catalog artifact by
// definition, so requiring one dropped every third-party safetensors model.
const curatedArtifactGate = (repoId: string, isGguf: boolean) =>
  isGguf || Boolean(groupForRepoId(repoId, AUDIO_CATALOG));

test("the curated-artifact gate would hide community safetensors", () => {
  assert.equal(curatedArtifactGate("hexgrad/Kokoro-82M", false), false);
  assert.equal(curatedArtifactGate("openai/whisper-large-v3", false), false);
  // Which is why community rows skip that clause; GGUF ones passed either way.
  assert.equal(curatedArtifactGate("some/community-GGUF", true), true);
});
