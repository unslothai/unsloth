// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A paused turn that cannot be resumed (its tool calls cannot be replayed) still says why. */

import assert from "node:assert/strict";
import test from "node:test";

import {
  incompleteLabel,
  incompleteRemedy,
  incompleteRemedyFor,
} from "../src/features/chat/utils/continuation.ts";

import { readSrc } from "./helpers/kit.ts";

test("an unresumable paused turn gets a way out instead of nothing", () => {
  // No remedy and no Resume made the bar render nothing at all.
  assert.equal(incompleteRemedy("paused"), null);
  const remedy = incompleteRemedyFor("paused", false);
  assert.ok(remedy, "the paused turn must still explain itself");
  assert.match(incompleteLabel("paused"), /paused/i);
});

test("a resumable paused turn keeps its Resume button", () => {
  assert.equal(incompleteRemedyFor("paused", true), null);
});

test("other reasons are unchanged either way", () => {
  for (const reason of ["length", "cancelled", "interrupted", "context_window", "empty"] as const) {
    assert.equal(incompleteRemedyFor(reason, false), incompleteRemedy(reason));
    assert.equal(incompleteRemedyFor(reason, true), incompleteRemedy(reason));
  }
});

test("the continue bar asks with whether the turn can resume", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, /const remedy = reason \? incompleteRemedyFor\(reason, canResume\) : null;/);
  // And still renders whenever a remedy is owed.
  assert.match(thread, /if \(!reason \|\| \(!remedy && !resumable\)\) \{\s+return null;/);
});
