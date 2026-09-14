// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

// The expanded writing area is a frame around a textarea that stays mounted, so
// nothing here can be asserted from a pure module: the invariants live in where
// the component puts one state setter. These read the source for the same
// reason the plain-paste chord test does.

test("expansion is reset wherever the composer is emptied, not only on send", async () => {
  const thread = await readSrcAsync("components/assistant-ui/thread.tsx");

  // handleSubmit returns before send() on three separate queueing paths: a run
  // already in flight, a prompt queue already active, and the Cmd/Ctrl+Enter
  // chord with nothing running. All of them empty the composer. Resetting next
  // to send() alone left a queued prompt behind a tall empty box.
  const at = thread.indexOf("const armJustSent = useCallback(");
  assert.notEqual(at, -1, "armJustSent should still be the shared clear hook");
  const body = thread.slice(at, thread.indexOf("\n  }, [", at));
  assert.match(body, /setIsWritingExpanded\(false\)/);

  // And nowhere else, so the reset cannot drift back out to one caller.
  const resets = thread.match(/setIsWritingExpanded\(false\)/g) ?? [];
  assert.equal(
    resets.length,
    3,
    "expected exactly three resets: armJustSent, the chat switch effect, and Escape",
  );

  // Every path that empties the composer goes through armJustSent; if a new one
  // appears that clears the text itself, it has to route through here too.
  for (const clearer of [
    "const queueComposerText = useCallback(",
    "const queuePastedTextPrompt = useCallback(",
  ]) {
    const start = thread.indexOf(clearer);
    assert.notEqual(start, -1, `${clearer} should still exist`);
    const region = thread.slice(start, start + 2500);
    assert.match(region, /armJustSent\(/);
  }
});

test("Escape collapses from the wrapper, leaving the input's capture slot alone", async () => {
  const thread = await readSrcAsync("components/assistant-ui/thread.tsx");

  // The input's onKeyDownCapture belongs to the plain-paste chord (see
  // pasted-text-attachment.test.ts). Escape therefore rides on the editor
  // wrapper, whose capture handler React dispatches first anyway.
  const at = thread.indexOf('className="unsloth-composer-editor"');
  assert.notEqual(at, -1);
  const wrapper = thread.slice(at, thread.indexOf("<ComposerPrimitive.Input", at));
  assert.match(wrapper, /onKeyDownCapture=\{\(event\) => \{/);
  assert.match(wrapper, /event\.key === "Escape"/);
  assert.match(wrapper, /!event\.nativeEvent\.isComposing/);

  // assistant-ui's cancelOnEscape listens on the document with capture:true, so
  // it runs before any React handler. Calling stopPropagation here would look
  // like it suppressed the cancel while doing nothing at all, so it must stay
  // out of the source.
  assert.doesNotMatch(wrapper, /stopPropagation/);
});
