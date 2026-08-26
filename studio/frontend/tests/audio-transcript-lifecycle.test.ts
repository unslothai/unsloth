// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL("../src/features/audio/audio-page.tsx", import.meta.url),
  "utf8",
);

test("a transcription run clears the previous result before it awaits", () => {
  // A failed run used to leave the previous transcript on screen while transcribedName
  // already named the new file, so Download .txt saved A's words as B.txt.
  assert.match(
    source,
    /setBusy\("transcribing"\);[\s\S]*?clearTranscript\(\);\s*setTranscribedName\(name\);\s*try \{/,
  );
});

test("a failed transcription is reported in the pane, not only in a toast", () => {
  assert.match(
    source,
    /if \(activeRef\.current\) setTranscriptError\(message\);\s*toast\.error\(message\);/,
  );
  assert.match(
    source,
    /\) : transcriptError \? \([\s\S]*?Could not transcribe \{transcribedName \?\? "that audio"\}/,
  );
});

test("clearing a transcript drops its name and error together", () => {
  assert.match(
    source,
    /const clearTranscript = useCallback\(\(\) => \{\s*setTranscript\(""\);\s*setTranscribedName\(null\);\s*setTranscriptError\(null\);\s*\}, \[\]\);/,
  );
});

test("a transcript never outlives the selection that produced it", () => {
  assert.match(source, /const forget = \(\) => \{[\s\S]*?clearTranscript\(\);/);
  assert.match(
    source,
    /if \(selectedSttRepoRef\.current !== id\) clearTranscript\(\);/,
  );
});

test("a status resync that drops or swaps the sidecar model clears it too", () => {
  // The sidecar is shared with chat dictation, so reconcileSttSelection can drop the
  // pick or adopt another surface's model with none of the explicit clear paths run.
  // Copy and Download .txt stayed live over a transcript whose model was already gone.
  assert.match(
    source,
    /const reconciled = reconcileSttSelection\(\{[\s\S]*?\}\);[\s\S]*?if \(reconciled !== selectedSttRepoRef\.current\) clearTranscript\(\);\s*selectedSttRepoRef\.current = reconciled;/,
  );
});
