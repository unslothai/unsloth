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

test("Escape collapses unless it closes the mention popover", async () => {
  const thread = await readSrcAsync("components/assistant-ui/thread.tsx");
  const mentions = await readSrcAsync(
    "components/assistant-ui/skill-mentions.tsx",
  );

  // cancelOnEscape preventDefaults every Escape on the document, so a
  // defaultPrevented gate never collapses. Decide from the window, before it.
  const at = thread.indexOf("const collapseOnEscape = (");
  assert.notEqual(at, -1);
  const body = thread.slice(at, thread.indexOf("\n  }, []);", at));
  assert.match(body, /event\.key === "Escape"/);
  assert.match(body, /!event\.isComposing/);
  assert.match(body, /!mentionOpenRef\.current/);
  assert.match(body, /editorRef\.current\?\.contains\(event\.target\)/);
  assert.doesNotMatch(body, /defaultPrevented/);
  assert.match(
    body,
    /window\.addEventListener\("keydown", collapseOnEscape, true\)/,
  );
  assert.match(thread, /onOpenChange=\{setMentionOpen\}/);

  // The same open flag the popover's own Escape handling checks.
  assert.match(
    mentions,
    /const \{ open \} = unstable_useTriggerPopoverScopeContext\(\);/,
  );
  assert.match(mentions, /<MentionOpenSignal onChange=\{onOpenChange\} \/>/);

  // The input's capture slot stays the plain-paste chord's (pasted-text-attachment.test.ts).
  const editor = thread.indexOf('className="unsloth-composer-editor"');
  const wrapper = thread.slice(
    editor,
    thread.indexOf("<ComposerPrimitive.Input", editor),
  );
  assert.doesNotMatch(wrapper, /onKeyDownCapture/);
});
