// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

const source = readSrc("features/audio/audio-page.tsx");

function section(start: string, end: string): string {
  return source.slice(
    source.indexOf(start),
    source.indexOf(end, source.indexOf(start)),
  );
}

test("changing model residency preserves the transcript and its recorded origin", () => {
  const refresh = section("const refreshSttStatus", "const sttSelected");
  const release = section(
    "const releaseTranscribeSelection",
    "const ensureClipSrc",
  );
  assert.doesNotMatch(refresh, /clearTranscript\(/);
  assert.doesNotMatch(release, /clearTranscript\(/);
  assert.match(source, /setTranscriptModel\(result.model\)/);
  assert.match(source, /setTranscriptModel\(record.model\)/);
});

test("the previous result is replaced only after its replacement model is ready", () => {
  const run = section("const runTranscription", "const handleRecordToggle");
  assert.match(run, /confirmTranscriptReplacement\(\)/);
  assert.ok(
    run.indexOf("await prepareTranscriptionModel()") <
      run.indexOf("clearTranscript()"),
  );
  assert.ok(
    run.indexOf("clearTranscript()") <
      run.indexOf("await transcribeWithProgress"),
  );
});

test("leaving the page stops microphone capture while transcription can finish into history", () => {
  const lifecycle = section(
    "// Release the microphone",
    "const handleTranscribeFile",
  );
  assert.match(
    lifecycle,
    /if \(!active\) \{\s*stopAndDiscardRecording\(\);\s*\}/,
  );
  assert.match(
    lifecycle,
    /useEffect\(\(\) => \(\) => transcriptionAbort.current\?\.abort\(\), \[\]\)/,
  );
  const run = section("const runTranscription", "const handleRecordToggle");
  assert.doesNotMatch(
    run.slice(run.indexOf("await transcribeWithProgress")),
    /!activeRef.current/,
  );
});
