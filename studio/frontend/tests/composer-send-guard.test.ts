// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  applySentTextGuard,
  armSentTextGuard,
  SENT_TEXT_GUARD_MS,
  sentTextGuardBlocksDraft,
  sentTextGuardLive,
} from "../src/features/chat/utils/composer-send-guard.ts";

const PROMPT = "Simulate a Tesla coil in a workshop.";
const KEY = "chat-draft:thread-1";
const T0 = 1_000_000;

const typed = (value: string) => ({
  value,
  replacesText: false,
  composerIsEmpty: false,
});
const replacement = (value: string, composerIsEmpty = true) => ({
  value,
  replacesText: true,
  composerIsEmpty,
});
const armed = (texts: string[] = [PROMPT], key: string | null = KEY) =>
  armSentTextGuard(texts, key, T0);

test("with nothing armed every write is applied", () => {
  const result = applySentTextGuard(null, typed(PROMPT), T0);
  assert.deepEqual(result, { accept: true, guard: null });
});

// The reported bug: the prompt sends, the text stays in the box.
test("a write carrying the just-sent text is refused", () => {
  const result = applySentTextGuard(armed(), typed(PROMPT), T0);
  assert.equal(result.accept, false);
});

test("the guard survives a refusal, since an engine can queue several", () => {
  const first = applySentTextGuard(armed(), typed(PROMPT), T0);
  const second = applySentTextGuard(first.guard, typed(PROMPT), T0 + 1);
  assert.equal(second.accept, false);
});

test("typing after a send is applied and retires the guard", () => {
  const result = applySentTextGuard(armed(), typed("a follow-up"), T0);
  assert.deepEqual(result, { accept: true, guard: null });
});

// An attachment-only send depends on the clear going through.
test("empty texts are never armed, so clearing is never refused", () => {
  const guard = armSentTextGuard(["", ""], KEY, T0);
  assert.deepEqual(guard.texts, []);
  assert.equal(applySentTextGuard(guard, typed(""), T0).accept, true);
});

// An autocorrect commit mutates the text, so equality alone would let it back
// in. It cannot originate from an empty composer, which is what marks it stale.
test("a mutated replacement into an emptied composer is refused", () => {
  const result = applySentTextGuard(armed(), replacement(`${PROMPT}!`), T0);
  assert.equal(result.accept, false);
});

test("a replacement once the user has typed again is applied", () => {
  const result = applySentTextGuard(
    armed(),
    replacement("teh cat", false),
    T0,
  );
  assert.equal(result.accept, true);
});

// Without a lifetime, a deliberate re-paste of the same prompt is refused
// forever, since the refusal keeps the guard and nothing retires it.
test("an identical re-paste after the window is applied", () => {
  const result = applySentTextGuard(
    armed(),
    typed(PROMPT),
    T0 + SENT_TEXT_GUARD_MS,
  );
  assert.deepEqual(result, { accept: true, guard: null });
});

test("the guard stops being live once the window passes", () => {
  const guard = armed();
  assert.equal(sentTextGuardLive(guard, T0 + SENT_TEXT_GUARD_MS - 1), true);
  assert.equal(sentTextGuardLive(guard, T0 + SENT_TEXT_GUARD_MS), false);
  assert.equal(sentTextGuardLive(null, T0), false);
});

// The image-edit send replaces what the user typed with a wrapper, so a late
// write carrying either one has to be refused.
test("both the wrapper and the visible text are guarded", () => {
  const guard = armed([`Apply this edit: ${PROMPT}`, PROMPT]);
  assert.equal(applySentTextGuard(guard, typed(PROMPT), T0).accept, false);
  assert.equal(
    applySentTextGuard(guard, typed(`Apply this edit: ${PROMPT}`), T0).accept,
    false,
  );
});

test("a draft holding the sent text is not restored", () => {
  assert.equal(sentTextGuardBlocksDraft(armed(), PROMPT, KEY, T0), true);
});

// The composer outlives a thread switch, so text alone would hide an unrelated
// thread's identical draft.
test("another thread's identical draft still restores", () => {
  assert.equal(
    sentTextGuardBlocksDraft(armed(), PROMPT, "chat-draft:thread-2", T0),
    false,
  );
});

test("a draft restore past the window is allowed", () => {
  assert.equal(
    sentTextGuardBlocksDraft(armed(), PROMPT, KEY, T0 + SENT_TEXT_GUARD_MS),
    false,
  );
  assert.equal(sentTextGuardBlocksDraft(null, PROMPT, KEY, T0), false);
});

test("an unrelated draft is restored", () => {
  assert.equal(
    sentTextGuardBlocksDraft(armed(), "something else", KEY, T0),
    false,
  );
});
