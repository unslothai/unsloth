// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { applySentTextGuard } from "../src/features/chat/utils/composer-send-guard.ts";

const PROMPT = "Simulate a Tesla coil in a workshop.";

test("with no send in flight every write is applied", () => {
  assert.deepEqual(applySentTextGuard(null, PROMPT), {
    accept: true,
    sentText: null,
  });
});

// The reported bug: the prompt sends, the text stays in the box.
test("a write carrying the just-sent text is refused", () => {
  assert.deepEqual(applySentTextGuard(PROMPT, PROMPT), {
    accept: false,
    sentText: PROMPT,
  });
});

test("the guard survives a refusal, since an engine can queue several", () => {
  const first = applySentTextGuard(PROMPT, PROMPT);
  assert.deepEqual(applySentTextGuard(first.sentText, PROMPT), {
    accept: false,
    sentText: PROMPT,
  });
});

test("typing after a send is applied and retires the guard", () => {
  const typed = applySentTextGuard(PROMPT, "a follow-up");
  assert.deepEqual(typed, { accept: true, sentText: null });
  // Once retired, the sent text itself goes in again.
  assert.deepEqual(applySentTextGuard(typed.sentText, PROMPT), {
    accept: true,
    sentText: null,
  });
});

// An attachment-only send depends on the clear going through.
test("clearing the composer is never refused", () => {
  assert.deepEqual(applySentTextGuard(PROMPT, ""), {
    accept: true,
    sentText: null,
  });
});

// Retyping passes through shorter values first, each retiring the guard.
test("retyping the same prompt is not held up", () => {
  let guard: string | null = PROMPT;
  for (const value of ["S", "Si", "Sim"]) {
    const step = applySentTextGuard(guard, value);
    assert.equal(step.accept, true);
    guard = step.sentText;
  }
  assert.deepEqual(applySentTextGuard(guard, PROMPT), {
    accept: true,
    sentText: null,
  });
});

// An autocorrect commit mutates the text, so it is a genuine edit.
test("a mutated late write is applied", () => {
  assert.deepEqual(applySentTextGuard(PROMPT, `${PROMPT} Extra.`), {
    accept: true,
    sentText: null,
  });
});
