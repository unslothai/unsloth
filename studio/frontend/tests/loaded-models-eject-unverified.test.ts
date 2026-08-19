// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The dictation unload answers {loaded_model: null} whatever it did, and the
// backend serves `gguf` from the transformers engine when whisper-server is
// absent, so the eject re-reads the status to find out what actually happened.
// That re-read returns null on any non-2xx as well as on an empty runtime, and
// reading those two as the same thing defeats the check: a transient 404 on the
// verification would toast "Ejected" and drop a row whose model is still
// holding memory.
//
// The eject path is bound to authFetch and the chat store, so this asserts the
// shape of the decision rather than driving it: the three branches must be
// distinguishable, and the unreadable one must not be spelled `null`.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const SOURCE = readFileSync(
  fileURLToPath(
    new URL("../src/features/loaded-models/loaded-models-api.ts", import.meta.url),
  ),
  "utf8",
);

test("an unreadable verification is its own outcome, not `ejected`", () => {
  assert.match(
    SOURCE,
    /\|\s*\{ status: "unverified" \}/,
    "EjectOutcome must be able to say the unload was not confirmed",
  );
  assert.match(
    SOURCE,
    /if \(stillResident === UNVERIFIED\) return \{ status: "unverified" \};/,
    "the runtime eject must map the sentinel before the null check",
  );
});

test("the STT re-read returns the sentinel rather than null on a failed read", () => {
  const stt = SOURCE.slice(SOURCE.indexOf('case "stt": {'), SOURCE.length);
  assert.match(stt, /const after = await bounded\(readSttStatus\);/);
  assert.match(
    stt,
    /if \(!after\) return UNVERIFIED;/,
    "a null status read is unreadable, not proof the engine is empty",
  );
  assert.doesNotMatch(
    stt.slice(stt.indexOf("const after")),
    /if \(!after\) return null;/,
  );
});

test("the sentinel is checked before the truthiness test that would hide it", () => {
  // A Symbol is truthy, so an UNVERIFIED reaching the `stillResident` branch
  // would report the runtime as still holding something called "Symbol()".
  const fn = SOURCE.slice(
    SOURCE.indexOf("const stillResident = await unload();"),
    SOURCE.indexOf("/** Release one row"),
  );
  assert.ok(
    fn.indexOf("UNVERIFIED") < fn.indexOf("stillResident\n"),
    "the sentinel check must precede the resident check",
  );
});

test("the user is told it was not confirmed, not that it worked", () => {
  const HOOK = readFileSync(
    fileURLToPath(
      new URL("../src/features/loaded-models/use-loaded-models.ts", import.meta.url),
    ),
    "utf8",
  );
  const branch = HOOK.slice(
    HOOK.indexOf('outcome.status === "unverified"'),
    HOOK.indexOf('outcome.status === "replaced"'),
  );
  assert.match(branch, /toast\.warning/, "neither success nor failure");
  assert.doesNotMatch(branch, /toast\.success/);
});
