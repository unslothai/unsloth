// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  applySentTextGuard,
  armSentTextGuard,
  isGuardRetiringKey,
  markSentTextGuardUserInput,
  sentTextGuardBlocksDraft,
} from "../src/features/chat/utils/composer-send-guard.ts";

const PROMPT = "Simulate a Tesla coil in a workshop.";
const KEY = "chat-draft:thread-1";

const typed = (value: string) => ({
  value,
  replacesText: false,
  isDeliberate: false,
  isComposition: false,
  composerIsEmpty: false,
});
const replacement = (value: string, composerIsEmpty = true) => ({
  value,
  replacesText: true,
  isDeliberate: false,
  isComposition: false,
  composerIsEmpty,
});
const deliberate = (value: string) => ({
  value,
  replacesText: false,
  isDeliberate: true,
  isComposition: false,
  composerIsEmpty: true,
});
const composing = (value: string) => ({
  value,
  replacesText: false,
  isDeliberate: false,
  isComposition: true,
  composerIsEmpty: true,
});
const ARMS_WITH_IME_SESSION =
  /armSentTextGuard\(\s*texts,\s*draftKeyRef\.current,\s*imeSessionOpenRef\.current,?\s*\)/;
const armed = (texts: string[] = [PROMPT], key: string | null = KEY) =>
  armSentTextGuard(texts, key);

test("with nothing armed every write is applied", () => {
  assert.deepEqual(applySentTextGuard(null, typed(PROMPT)), {
    accept: true,
    guard: null,
  });
});

test("a write carrying the just-sent text is refused", () => {
  assert.equal(applySentTextGuard(armed(), typed(PROMPT)).accept, false);
});

test("the guard survives a refusal, since an engine can queue several", () => {
  const first = applySentTextGuard(armed(), typed(PROMPT));
  assert.equal(applySentTextGuard(first.guard, typed(PROMPT)).accept, false);
});

// Event queue latency is unbounded, so nothing here is time based.
test("a stale write is refused however late it arrives", () => {
  let guard = armed();
  for (let i = 0; i < 100; i += 1) {
    const result = applySentTextGuard(guard, typed(PROMPT));
    assert.equal(result.accept, false);
    guard = result.guard as ReturnType<typeof armed>;
  }
});

test("typing after a send is applied and retires the guard", () => {
  assert.deepEqual(applySentTextGuard(armed(), typed("a follow-up")), {
    accept: true,
    guard: null,
  });
});

test("empty texts are never armed, so clearing is never refused", () => {
  const guard = armSentTextGuard(["", ""], KEY);
  assert.deepEqual(guard.texts, []);
  assert.equal(applySentTextGuard(guard, typed("")).accept, true);
});

test("a mutated replacement into an emptied composer is refused", () => {
  assert.equal(applySentTextGuard(armed(), replacement(`${PROMPT}!`)).accept, false);
});

test("a replacement once the user has typed again is applied", () => {
  assert.equal(
    applySentTextGuard(armed(), replacement("teh cat", false)).accept,
    true,
  );
});

test("an identical re-paste is applied once the paste has retired the guard", () => {
  assert.deepEqual(applySentTextGuard(null, typed(PROMPT)), {
    accept: true,
    guard: null,
  });
});

test("both the wrapper and the visible text are guarded", () => {
  const guard = armed([`Apply this edit: ${PROMPT}`, PROMPT]);
  assert.equal(applySentTextGuard(guard, typed(PROMPT)).accept, false);
  assert.equal(
    applySentTextGuard(guard, typed(`Apply this edit: ${PROMPT}`)).accept,
    false,
  );
});

test("a draft holding the sent text is not restored", () => {
  assert.equal(sentTextGuardBlocksDraft(armed(), PROMPT, KEY), true);
});

test("another thread's identical draft still restores", () => {
  assert.equal(
    sentTextGuardBlocksDraft(armed(), PROMPT, "chat-draft:thread-2"),
    false,
  );
});

test("an unrelated draft is restored", () => {
  assert.equal(sentTextGuardBlocksDraft(armed(), "something else", KEY), false);
  assert.equal(sentTextGuardBlocksDraft(null, PROMPT, KEY), false);
});

test("a deliberate undo restores the sent prompt and retires the guard", () => {
  assert.deepEqual(applySentTextGuard(armed(), deliberate(PROMPT)), {
    accept: true,
    guard: null,
  });
});

test("undo wins even over an armed wrapper pair", () => {
  const guard = armed([`Apply this edit: ${PROMPT}`, PROMPT]);
  assert.equal(applySentTextGuard(guard, deliberate(PROMPT)).accept, true);
});

test("an autocorrect commit is still refused after the undo carve-out", () => {
  assert.equal(
    applySentTextGuard(armed(), replacement(`${PROMPT}!`)).accept,
    false,
  );
});

test("the raw pre-send value is guarded, not just its trimmed form", () => {
  const raw = "  make it brighter  ";
  const guard = armSentTextGuard(
    [`Apply this edit: ${raw.trim()}`, raw, raw.trim()],
    KEY,
  );
  assert.equal(applySentTextGuard(guard, typed(raw)).accept, false);
  assert.equal(applySentTextGuard(guard, typed(raw.trim())).accept, false);
});

test("re-typing a one-character prompt is applied", () => {
  const guard = markSentTextGuardUserInput(armed(["?"]));
  assert.deepEqual(applySentTextGuard(guard, typed("?")), {
    accept: true,
    guard: null,
  });
});

test("a stale write with no keystroke behind it is still refused", () => {
  assert.equal(applySentTextGuard(armed(["?"]), typed("?")).accept, false);
});

test("a keystroke does not let an autocorrect commit through", () => {
  const guard = markSentTextGuardUserInput(armed());
  assert.equal(
    applySentTextGuard(guard, replacement(`${PROMPT}!`)).accept,
    false,
  );
});

test("a keystroke does not unblock the raced draft", () => {
  const guard = markSentTextGuardUserInput(armed());
  assert.equal(sentTextGuardBlocksDraft(guard, PROMPT, KEY), true);
});

test("a composition started after the send lets its commit through", () => {
  const guard = markSentTextGuardUserInput(armed(["\u{1F642}"]));
  assert.deepEqual(applySentTextGuard(guard, typed("\u{1F642}")), {
    accept: true,
    guard: null,
  });
});

test("marking an unarmed guard is a no-op", () => {
  assert.equal(markSentTextGuardUserInput(null), null);
});

const key = (k: string, mods: { metaKey?: boolean; ctrlKey?: boolean } = {}) =>
  isGuardRetiringKey({
    key: k,
    metaKey: mods.metaKey ?? false,
    ctrlKey: mods.ctrlKey ?? false,
  });

test("the sending Enter is not a keystroke boundary", () => {
  assert.equal(key("Enter"), false);
  assert.equal(key("Enter", { metaKey: true }), false);
});

test("characters and IME keys are keystroke boundaries", () => {
  assert.equal(key("?"), true);
  assert.equal(key("a"), true);
  assert.equal(key("Process"), true);
  assert.equal(key("Backspace"), true);
});

test("chords and bare modifiers are not keystroke boundaries", () => {
  assert.equal(key("v", { metaKey: true }), false);
  assert.equal(key("z", { ctrlKey: true }), false);
  assert.equal(key("Shift"), false);
  assert.equal(key("Meta"), false);
  assert.equal(key("Escape"), false);
  assert.equal(key("Tab"), false);
});

// Drop and yank fire no paste event, so the paste carve-out never sees them.
test("a drop or a yank of the sent text is applied", () => {
  assert.deepEqual(applySentTextGuard(armed(), deliberate(PROMPT)), {
    accept: true,
    guard: null,
  });
});

test("a deliberate write wins over an armed wrapper pair", () => {
  const guard = armed([`Apply this edit: ${PROMPT}`, PROMPT]);
  assert.equal(applySentTextGuard(guard, deliberate(PROMPT)).accept, true);
});

test("a composition begun before the send is refused", () => {
  const guard = armed();
  assert.deepEqual(applySentTextGuard(guard, composing("\u65e5\u672c\u8a9e")), {
    accept: false,
    guard,
  });
});

test("a composition begun after the send is applied", () => {
  const guard = markSentTextGuardUserInput(armed());
  assert.deepEqual(applySentTextGuard(guard, composing("\u65e5\u672c\u8a9e")), {
    accept: true,
    guard: null,
  });
});

test("a composition write with none open at the send is applied", () => {
  const guard = armSentTextGuard([PROMPT], KEY, false);
  for (const value of ["h", "\u65e5\u672c\u8a9e"]) {
    assert.deepEqual(applySentTextGuard(guard, composing(value)), {
      accept: true,
      guard: null,
    });
  }
});

test("retyping a one-character prompt applies with no composition open at the send", () => {
  const guard = armSentTextGuard(["?"], KEY, false);
  assert.deepEqual(applySentTextGuard(guard, composing("?")), {
    accept: true,
    guard: null,
  });
});

test("the sent text is still refused when typed by plain keys", () => {
  const guard = armSentTextGuard([PROMPT], KEY, false);
  assert.equal(applySentTextGuard(guard, typed(PROMPT)).accept, false);
});

test("the composer arms the guard with whether an IME session was open", () => {
  const thread = readFileSync(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  assert.match(thread, ARMS_WITH_IME_SESSION);
});

// AltGr is how a lot of layouts reach @, so a one-character prompt typed with
// it must retire the equality guard. Windows reports it as Ctrl+Alt, and some
// builds set those flags even while AltGraph reads true, so both forms count.
test("an AltGr character is a keystroke boundary", () => {
  assert.equal(
    isGuardRetiringKey({
      key: "@",
      metaKey: false,
      ctrlKey: true,
      altKey: true,
      getModifierState: (k: "AltGraph") => k === "AltGraph",
    }),
    true,
  );
  assert.equal(
    isGuardRetiringKey({ key: "@", metaKey: false, ctrlKey: true, altKey: true }),
    true,
  );
});

test("a real Ctrl chord is still not a keystroke boundary", () => {
  assert.equal(
    isGuardRetiringKey({
      key: "a",
      metaKey: false,
      ctrlKey: true,
      altKey: false,
      getModifierState: () => false,
    }),
    false,
  );
  assert.equal(
    isGuardRetiringKey({ key: "Delete", metaKey: false, ctrlKey: true, altKey: true }),
    false,
  );
});
