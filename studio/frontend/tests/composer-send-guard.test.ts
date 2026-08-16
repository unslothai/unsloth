// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
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
// Undo, redo, paste, drop and yank all arrive this way.
const deliberate = (value: string) => ({
  value,
  replacesText: false,
  isDeliberate: true,
  isComposition: false,
  composerIsEmpty: true,
});
// An IME write. Finalisation converts the text, so equality never matches it.
const composing = (value: string) => ({
  value,
  replacesText: false,
  isDeliberate: false,
  isComposition: true,
  composerIsEmpty: true,
});
const armed = (texts: string[] = [PROMPT], key: string | null = KEY) =>
  armSentTextGuard(texts, key);

test("with nothing armed every write is applied", () => {
  assert.deepEqual(applySentTextGuard(null, typed(PROMPT)), {
    accept: true,
    guard: null,
  });
});

// The reported bug: the prompt sends, the text stays in the box.
test("a write carrying the just-sent text is refused", () => {
  assert.equal(applySentTextGuard(armed(), typed(PROMPT)).accept, false);
});

test("the guard survives a refusal, since an engine can queue several", () => {
  const first = applySentTextGuard(armed(), typed(PROMPT));
  assert.equal(applySentTextGuard(first.guard, typed(PROMPT)).accept, false);
});

// Event queue latency has no upper bound, so a stale write arriving long after
// the send must still be refused. Nothing here is time based.
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

// An attachment-only send depends on the clear going through.
test("empty texts are never armed, so clearing is never refused", () => {
  const guard = armSentTextGuard(["", ""], KEY);
  assert.deepEqual(guard.texts, []);
  assert.equal(applySentTextGuard(guard, typed("")).accept, true);
});

// An autocorrect commit mutates the text, so equality alone would let it back
// in. It cannot originate from an empty composer, which is what marks it stale.
test("a mutated replacement into an emptied composer is refused", () => {
  assert.equal(applySentTextGuard(armed(), replacement(`${PROMPT}!`)).accept, false);
});

test("a replacement once the user has typed again is applied", () => {
  assert.equal(
    applySentTextGuard(armed(), replacement("teh cat", false)).accept,
    true,
  );
});

// A deliberate re-paste is byte-identical to what was sent, so only the paste
// gesture itself can tell the two apart. The composer nulls the guard on paste,
// which is this state.
test("an identical re-paste is applied once the paste has retired the guard", () => {
  assert.deepEqual(applySentTextGuard(null, typed(PROMPT)), {
    accept: true,
    guard: null,
  });
});

// The image-edit send replaces what the user typed with a wrapper, so a late
// write carrying either one has to be refused.
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

// The composer outlives a thread switch, so text alone would hide an unrelated
// thread's identical draft.
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

// Undo after sending is how a user recovers a prompt to edit and resend. It
// restores exactly what was sent, so only the gesture tells it from a stale
// write, and the gesture is deliberate.
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

// Autocorrect is the engine, not the user, so it stays refused.
test("an autocorrect commit is still refused after the undo carve-out", () => {
  assert.equal(
    applySentTextGuard(armed(), replacement(`${PROMPT}!`)).accept,
    false,
  );
});

// The image-edit send wraps what the user typed, and the wrapper is built from
// the trimmed text. A late DOM write carries the raw textarea value, so arming
// the trimmed form alone misses it whenever the instruction had whitespace.
test("the raw pre-send value is guarded, not just its trimmed form", () => {
  const raw = "  make it brighter  ";
  const guard = armSentTextGuard(
    [`Apply this edit: ${raw.trim()}`, raw, raw.trim()],
    KEY,
  );
  assert.equal(applySentTextGuard(guard, typed(raw)).accept, false);
  assert.equal(applySentTextGuard(guard, typed(raw.trim())).accept, false);
});

// Re-typing the sent prompt is normally several writes, and the first one
// already differs. A one-character prompt arrives whole in a single write, so
// equality alone swallowed it and kept the guard, blocking every retry.
test("re-typing a one-character prompt is applied", () => {
  const guard = markSentTextGuardUserInput(armed(["?"]));
  assert.deepEqual(applySentTextGuard(guard, typed("?")), {
    accept: true,
    guard: null,
  });
});

// The stale write is queued before the send, so it is delivered before any
// later keydown. Without one, equality still refuses it.
test("a stale write with no keystroke behind it is still refused", () => {
  assert.equal(applySentTextGuard(armed(["?"]), typed("?")).accept, false);
});

// The keystroke relaxes equality only. An autocorrect commit into an empty
// composer stays stale whatever the user pressed.
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

// Dictation, handwriting and IMEs insert without a keydown, so the composition
// they start is the boundary instead. A one-emoji prompt is the case that needs
// it, since the whole value arrives in a single committed write.
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

// Enter is the send itself: the guard is armed from inside that keydown, so
// counting it would retire the guard before a single stale write could land.
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

// A chord is a command, and paste has its own carve-out already.
test("chords and bare modifiers are not keystroke boundaries", () => {
  assert.equal(key("v", { metaKey: true }), false);
  assert.equal(key("z", { ctrlKey: true }), false);
  assert.equal(key("Shift"), false);
  assert.equal(key("Meta"), false);
  assert.equal(key("Escape"), false);
  assert.equal(key("Tab"), false);
});

// Dragging the sent prompt back in, or yanking it back, carries exactly what
// was sent and fires no paste event, so the paste carve-out never sees it.
// Both were measured swallowed in Chromium, Firefox and WebKit alike.
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

// A composition still open when the send happened commits a converted value
// that matches no armed text. compositionstart is what separates it from one
// the user began afterwards.
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
  // Ctrl+Alt on a named key is a shortcut, not a character.
  assert.equal(
    isGuardRetiringKey({ key: "Delete", metaKey: false, ctrlKey: true, altKey: true }),
    false,
  );
});
