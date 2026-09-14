// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { sttDownloadedArtifacts } from "../src/features/audio/audio-page-policy.ts";
import {
  sttEngineForRepoId,
  sttRepoIdForSidecarKey,
} from "../src/features/audio/stt-artifacts.ts";

const repoIdForSidecarKey = (
  key: string,
  engine: "transformers" | "gguf" | "mtmd",
) => {
  const repos: Record<string, string> =
    engine === "gguf"
      ? {
          tiny: "unslothai/whisper-tiny-GGUF",
          base: "unslothai/whisper-base-GGUF",
          small: "unslothai/whisper-small-GGUF",
        }
      : {
          tiny: "unsloth/whisper-tiny",
          base: "unsloth/whisper-base",
          small: "unsloth/whisper-small",
          "qwen3-asr-0.6b": "unslothai/Qwen3-ASR-0.6B-GGUF",
        };
  return repos[key] ?? key;
};

test("STT On Device inventory follows all sidecar download engines", () => {
  assert.deepEqual(
    sttDownloadedArtifacts(
      {
        transformers: {
          downloaded_models: ["small", "org/custom-whisper"],
        },
        gguf: { downloaded_models: ["small"] },
        mtmd: { downloaded_models: ["qwen3-asr-0.6b"] },
      },
      repoIdForSidecarKey,
    ),
    [
      {
        repoId: "unsloth/whisper-small",
        sidecarKey: "small",
        engine: "transformers",
      },
      {
        repoId: "org/custom-whisper",
        sidecarKey: "org/custom-whisper",
        engine: "transformers",
      },
      {
        repoId: "unslothai/whisper-small-GGUF",
        sidecarKey: "small",
        engine: "gguf",
      },
      {
        repoId: "unslothai/Qwen3-ASR-0.6B-GGUF",
        sidecarKey: "qwen3-asr-0.6b",
        engine: "mtmd",
      },
    ],
  );
});

test("legacy top-level downloads are retained without duplicate rows", () => {
  assert.deepEqual(
    sttDownloadedArtifacts(
      {
        downloaded_models: ["tiny"],
        transformers: { downloaded_models: ["tiny", "base"] },
      },
      repoIdForSidecarKey,
    ),
    [
      {
        repoId: "unsloth/whisper-tiny",
        sidecarKey: "tiny",
        engine: "transformers",
      },
      {
        repoId: "unsloth/whisper-base",
        sidecarKey: "base",
        engine: "transformers",
      },
    ],
  );
});

test("only exact curated Qwen artifacts use the finite MTMD runtime", () => {
  assert.equal(sttEngineForRepoId("unslothai/Qwen3-ASR-0.6B-GGUF"), "mtmd");
  assert.equal(sttEngineForRepoId("Qwen/Qwen3-ASR-0.6B"), "transformers");
  assert.equal(sttEngineForRepoId("community/Qwen3-ASR-finetune"), "transformers");
  assert.equal(
    sttRepoIdForSidecarKey("qwen3-asr-0.6b", "mtmd"),
    "unslothai/Qwen3-ASR-0.6B-GGUF",
  );
});
