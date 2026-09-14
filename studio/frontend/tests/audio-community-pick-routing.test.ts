// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
  assert.equal(routeFor("someone/mystery-model", null), null);
});

test("the catalog still wins over the tag", () => {
  assert.equal(routeFor("unsloth/whisper-small", null), "stt");
  assert.equal(routeFor("unslothai/Qwen3-ASR-0.6B-GGUF", null), "stt");
  assert.equal(routeFor("unsloth/orpheus-3b-0.1-ft", null), "tts");
});

const curatedArtifactGate = (repoId: string, isGguf: boolean) =>
  isGguf || Boolean(groupForRepoId(repoId, AUDIO_CATALOG));

test("the curated-artifact gate would hide community safetensors", () => {
  assert.equal(curatedArtifactGate("hexgrad/Kokoro-82M", false), false);
  assert.equal(curatedArtifactGate("openai/whisper-large-v3", false), false);
  assert.equal(curatedArtifactGate("some/community-GGUF", true), true);
});
