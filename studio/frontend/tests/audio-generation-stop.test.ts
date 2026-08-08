// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL("../src/features/audio/audio-page.tsx", import.meta.url),
  "utf8",
);

test("generation exposes a clickable Stop action wired to the request abort", () => {
  assert.match(
    source,
    /const handleStopGeneration[\s\S]*generateAbort\.current\?\.abort\(\)/,
  );
  assert.match(
    source,
    /onClick=\{\s*busy === "generating"\s*\?\s*handleStopGeneration\s*:\s*handleGenerate\s*\}/,
  );
  assert.match(
    source,
    /busy === "generating"[\s\S]*icon=\{StopIcon\}[\s\S]*Stop/,
  );
});

test("leaving the audio page aborts an in-flight generation", () => {
  assert.match(
    source,
    /useEffect\(\(\) => \{\s*if \(!active\) generateAbort\.current\?\.abort\(\);\s*\}, \[active\]\)/,
  );
});

test("leaving the audio page aborts an in-flight transcription", () => {
  assert.match(
    source,
    /const transcriptionAbort = useRef<AbortController \| null>\(null\)/,
  );
  assert.match(
    source,
    /transcribeAudioBlob\(blob, \{[\s\S]*signal: controller\.signal/,
  );
  assert.match(
    source,
    /if \(!active\) \{[\s\S]*transcriptionAbort\.current\?\.abort\(\)/,
  );
  assert.match(
    source,
    /if \(controller\.signal\.aborted \|\| !activeRef\.current\) return/,
  );
});

test("a non-abort generation failure refreshes authoritative residency", () => {
  assert.match(
    source,
    /catch \(error\) \{[\s\S]*if \(!controller\.signal\.aborted\)[\s\S]*await refreshStatus\(\)/,
  );
  assert.match(
    source,
    /const refreshStatus[\s\S]*catch \{[\s\S]*setStatus\(null\)/,
  );
});
