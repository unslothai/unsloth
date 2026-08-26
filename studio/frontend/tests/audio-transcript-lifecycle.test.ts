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

test("a resync that adopts another surface's model clears the transcript too", () => {
  // The sidecar is shared with chat dictation, so reconcileSttSelection can adopt a
  // model this page never picked with none of the explicit clear paths run, leaving
  // one model's transcript on screen under another model's name in the picker.
  assert.match(
    source,
    /const reconciled = reconcileSttSelection\(\{[\s\S]*?\}\);[\s\S]*?if \(reconciled !== null && reconciled !== selectedSttRepoRef\.current\)\s*clearTranscript\(\);\s*selectedSttRepoRef\.current = reconciled;/,
  );
});

test("an idle sidecar unload leaves the transcript the user came back for", () => {
  // STT_KEEP_ALIVE_SECONDS is 5 minutes and the unload fires while Audio is hidden, so
  // the activation resync reconciles to null through no action of the user's. Clearing
  // there deleted text the page tells them to copy or download to keep, and nothing is
  // misattributed by a transcript outliving a selection that is simply gone.
  assert.doesNotMatch(
    source,
    /if \(reconciled !== selectedSttRepoRef\.current\) clearTranscript\(\);/,
  );
});
