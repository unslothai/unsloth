// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The status re-read returns null on both errors and empty runtime; unreadable must differ.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const SOURCE = readSrc("features/loaded-models/loaded-models-api.ts");

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
  // A Symbol is truthy, so UNVERIFIED must not reach the stillResident branch.
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
  const HOOK = readSrc("features/loaded-models/use-loaded-models.ts");
  const branch = HOOK.slice(
    HOOK.indexOf('outcome.status === "unverified"'),
    HOOK.indexOf('outcome.status === "replaced"'),
  );
  assert.match(branch, /toast\.warning/, "neither success nor failure");
  assert.doesNotMatch(branch, /toast\.success/);
});
