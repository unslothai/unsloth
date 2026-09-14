// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

// Source-reading, like the plain-paste chord test: the invariant is where the
// component puts one state setter.

test("expansion is reset wherever the composer is emptied, not only on send", async () => {
  const thread = await readSrcAsync("components/assistant-ui/thread.tsx");

  // handleSubmit returns before send() on three queueing paths that also clear.
  const at = thread.indexOf("const armJustSent = useCallback(");
  assert.notEqual(at, -1, "armJustSent should still be the shared clear hook");
  const body = thread.slice(at, thread.indexOf("\n  }, [", at));
  assert.match(body, /setIsWritingExpanded\(false\)/);

  const resets = thread.match(/setIsWritingExpanded\(false\)/g) ?? [];
  assert.equal(
    resets.length,
    3,
    "expected exactly three resets: armJustSent, the chat switch effect, and Escape",
  );

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

  // The input's capture slot is the plain-paste chord's (pasted-text-attachment.test.ts).
  const at = thread.indexOf('className="unsloth-composer-editor"');
  assert.notEqual(at, -1);
  const wrapper = thread.slice(at, thread.indexOf("<ComposerPrimitive.Input", at));
  assert.match(wrapper, /onKeyDownCapture=\{\(event\) => \{/);
  assert.match(wrapper, /event\.key === "Escape"/);
  assert.match(wrapper, /!event\.nativeEvent\.isComposing/);
  // The @-mention popover consumes Escape from a document capture:true listener
  // that runs first, so an already-defaulted Escape must not collapse as well.
  assert.match(wrapper, /!event\.nativeEvent\.defaultPrevented/);

  // cancelOnEscape is a document capture:true listener, so stopPropagation here
  // would read as suppressing a cancel that already ran.
  assert.doesNotMatch(wrapper, /stopPropagation/);
});
