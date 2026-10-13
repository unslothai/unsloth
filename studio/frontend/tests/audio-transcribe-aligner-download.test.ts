// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  alignerDownloadEntry,
  type SttCapabilities,
} from "../src/features/audio/transcribe-capabilities.ts";
import { readSrc } from "./helpers/kit.ts";

const ALIGNER_ID = "audio-cpp/audio.cpp-gguf/Qwen3-ForcedAligner-0.6B-GGUF";
const QWEN3: SttCapabilities = {
  engine: "audiocpp",
  family: "qwen3_asr",
  timestamps: "on_request",
  speakers: false,
  aligner: {
    downloaded: false,
    size_bytes: 1_129_966_496,
    id: ALIGNER_ID,
    variant: "Q8_0",
  },
  cpu_only: false,
};

test("a timestamped run stages a missing aligner as a normal quant download", () => {
  assert.deepEqual(alignerDownloadEntry(QWEN3, true), {
    repoId: ALIGNER_ID,
    files: [],
    bytes: 1_129_966_496,
    ggufVariant: "Q8_0",
  });
});

test("nothing is staged when the run can start now", () => {
  assert.equal(alignerDownloadEntry(QWEN3, false), null);
  assert.equal(
    alignerDownloadEntry(
      { ...QWEN3, aligner: { ...QWEN3.aligner!, downloaded: true } },
      true,
    ),
    null,
  );
  assert.equal(alignerDownloadEntry({ ...QWEN3, aligner: null }, true), null);
  assert.equal(alignerDownloadEntry(null, true), null);
  // An older backend that does not name the aligner keeps its own download.
  assert.equal(
    alignerDownloadEntry(
      { ...QWEN3, aligner: { downloaded: false, size_bytes: 1 } },
      true,
    ),
    null,
  );
});

test("the aligner download runs through the shared manager; cancel, Timestamps off or an untimestamped run drops the pending run", () => {
  const hook = readSrc("features/audio/hooks/use-aligner-download.ts");
  assert.match(hook, /useStagedDownload\(\{\s*scopeId: "audio-aligner"/);
  assert.match(
    hook,
    /onReady: \(\) => \{[\s\S]*pendingRun\.current = null;[\s\S]*refreshCaps\(\);[\s\S]*run\?\.\(\);/,
  );
  assert.match(
    hook,
    /useEffect\(\(\) => \{[^}]*if \(!timestamps\) pendingRun\.current = null;\s*\}, \[timestamps\]\);/,
  );
  // Only a dropped run explains itself; a plan with nothing waiting stays quiet.
  assert.match(
    hook,
    /onCancelled: \(\) => \{\s*if \(!pendingRun\.current\) return;\s*pendingRun\.current = null;\s*toast\.info\("Transcription not started"/,
  );
  assert.match(
    hook,
    /if \(!entry\) \{[^}]*pendingRun\.current = null;\s*run\(\);\s*return;\s*\}\s*pendingRun\.current = run;\s*stage\(\[entry\]\);/,
  );
});

test("only a timestamped Transcribe waits for the aligner download", () => {
  const page = readSrc("features/audio/audio-page.tsx");
  assert.match(
    page,
    /useAlignerDownload\(\s*transcribeCaps\.caps,\s*transcribeCaps\.refresh,\s*transcribeOptions\.request\.timestamps,\s*\)/,
  );
  assert.match(
    page,
    /const alignerBlocks =\s*alignerDownload\.staging && transcribeOptions\.request\.timestamps;\s*const transcribeBlocker: GenerateBlocker \| null = alignerBlocks\s*\?\s*\{ reason: "Downloading the timing aligner…" \}/,
  );
  // The run reads the page as it is when the download ends, and only while it is still showing.
  assert.match(
    page,
    /const startTranscriptionRef = useRef\(startTranscription\);\s*useEffect\(\(\) => \{\s*startTranscriptionRef\.current = startTranscription;\s*\}\);/,
  );
  assert.match(
    page,
    /alignerDownload\.runWithAligner\(transcribeOptions\.request\.timestamps, \(\) => \{[\s\S]*?!activeRef\.current \|\|\s*modeRef\.current !== "transcribe" \|\|\s*useAudioTranscribeStore\.getState\(\)\.source\?\.id !== source\.id\s*\)\s*return;\s*startTranscriptionRef\.current\(source\);/,
  );
});

test("a session reset drops a staged plan so Transcribe is not left blocked", () => {
  const staged = readSrc(
    "features/hub/download-manager/use-staged-download.ts",
  );
  assert.match(
    staged,
    /useStagedDownloadQueues\.subscribe\(\(state, previous\) => \{\s*if \(state\.resets === previous\.resets \|\| queueRef\.current === null\) \{\s*return;\s*\}[\s\S]*?setQueue\(null\);\s*onCancelledRef\.current\?\.\(\);/,
  );
});
