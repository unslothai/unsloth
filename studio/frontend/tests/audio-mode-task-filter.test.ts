// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  sttEngineForRepoId,
  sttRepoIdForSidecarKey,
  sttSidecarKeyFor,
} from "../src/features/audio/stt-artifacts.ts";
import { detectCapabilities } from "../src/features/model-picker/components/model-selector/model-capabilities.ts";

import { readSrc } from "./helpers/kit.ts";
import { readAudioWorkspaceSource } from "./helpers/audio-workspace.ts";

const pageSource = readAudioWorkspaceSource();
const catalogSource = readSrc("features/audio/catalog.ts");

test("Hub discovery follows the active audio mode", () => {
  assert.match(
    pageSource,
    /speak: \["text-to-speech", "text-to-audio"\],[\s\S]*transcribe: \["automatic-speech-recognition"\]/,
  );
  assert.match(pageSource, /task=\{HUB_TASKS_BY_MODE\[mode\]\}/);
});

test("resident sidecar keys restore canonical repo casing", () => {
  assert.equal(
    sttRepoIdForSidecarKey("qwen3-asr-0.6b", "mtmd"),
    "unslothai/Qwen3-ASR-0.6B-GGUF",
  );
});

test("Whisper artifacts retain distinct runtime identities", () => {
  assert.equal(
    sttRepoIdForSidecarKey("small", "transformers"),
    "unsloth/whisper-small",
  );
  assert.equal(
    sttRepoIdForSidecarKey("small", "gguf"),
    "unslothai/whisper-small-GGUF",
  );
  assert.equal(sttEngineForRepoId("unsloth/whisper-small"), "transformers");
  assert.equal(sttEngineForRepoId("unslothai/whisper-small-GGUF"), "gguf");
  assert.equal(sttEngineForRepoId("unslothai/Qwen3-ASR-0.6B-GGUF"), "mtmd");
  assert.equal(sttSidecarKeyFor("unslothai/whisper-small-GGUF"), "small");
  assert.match(catalogSource, /isKnownSttArtifactRepoId\(repoId\)/);
});

test("ASR names carry the audio capability badge", () => {
  assert.equal(
    detectCapabilities({ id: "unslothai/Qwen3-ASR-0.6B-GGUF" }).audio,
    true,
  );
});

test("Search Hub from an audio picker carries the page's pipeline tasks", () => {
  const selector = readSrc(
    "features/model-picker/components/model-selector.tsx",
  );
  assert.match(
    selector,
    /search: \{\s*tab: "discover",\s*capability: hubCapability,\s*audioTask: hubCapability === "audio" && task \? \[task\]\.flat\(\) : undefined,\s*\}/,
  );
  // The picker task is a tag or a tag list; the hub param is always a list.
  const asParam = (task: string | readonly string[]) => [task].flat();
  assert.deepEqual(asParam(["text-to-speech", "text-to-audio"]), [
    "text-to-speech",
    "text-to-audio",
  ]);
  assert.deepEqual(asParam("automatic-speech-recognition"), [
    "automatic-speech-recognition",
  ]);
});
