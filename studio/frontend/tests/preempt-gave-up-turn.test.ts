// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A turn the backend gave up on can be EMPTY, and an empty turn used to render as nothing
 * at all.
 *
 * Measured in the GUI on 2026-09-05, four chats on a 35B at -c 8192 with the GPU shared:
 * one chat was evicted while still prefilling, waited for room, and the backend finished
 * its turn without it ever producing a token. The stream carried no error, so the row
 * mounted with no text, no notice, no Continue and no explanation. `isContinuableContent`
 * and the bar's own `partial.trim()` check both require text, which is right for every
 * other reason a turn stops early and wrong for this one, since this is the only reason
 * the backend can raise BEFORE the first token.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  PREEMPT_GAVE_UP_REASON,
  incompleteLabel,
  isContinuableContent,
  isPreemptGaveUp,
  resumesWithoutText,
} from "../src/features/chat/utils/continuation.ts";

test("the give-up signal is read off the truncation event", () => {
  assert.equal(
    isPreemptGaveUp({ reason: PREEMPT_GAVE_UP_REASON }),
    true,
    "this is the whole contract with the backend's _preempt_gave_up_event",
  );
});

test("an ordinary fit is not a give-up", () => {
  // The same event carries real fits, several per turn on a compacting thread. Reading one
  // of those as a give-up would relabel a healthy turn as paused.
  assert.equal(isPreemptGaveUp({ fits: false, dropped_messages: 4 }), false);
  assert.equal(isPreemptGaveUp({}), false);
  assert.equal(isPreemptGaveUp(null), false);
  assert.equal(isPreemptGaveUp(undefined), false);
});

test("only a paused turn may be continued with no text", () => {
  assert.equal(resumesWithoutText("paused"), true);
  // Every other reason has text by construction, and offering Continue on an empty one
  // would resume a turn that never started.
  assert.equal(resumesWithoutText("length"), false);
  assert.equal(resumesWithoutText("cancelled"), false);
  assert.equal(resumesWithoutText("interrupted"), false);
  assert.equal(resumesWithoutText(null), false);
});

test("an empty assistant turn is continuable only when it is allowed to be", () => {
  const empty = [{ type: "text", text: "" }];
  assert.equal(
    isContinuableContent(empty),
    false,
    "the default must not change: an empty turn is normally nothing to resume",
  );
  assert.equal(isContinuableContent(empty, { allowEmpty: true }), true);
  assert.equal(isContinuableContent([], { allowEmpty: true }), true);
});

test("allowEmpty does not reopen the tool-call rule", () => {
  // The reason `isContinuableContent` exists. A continuation runs as a sibling, so the
  // call and its result are missing from the outbound history whatever stopped the turn.
  const calledATool = [
    { type: "text", text: "" },
    { type: "tool-call", toolName: "web_search" },
  ];
  assert.equal(isContinuableContent(calledATool, { allowEmpty: true }), false);
  assert.equal(
    isContinuableContent([{ type: "tool-call", toolName: "web_search" }], {
      allowEmpty: true,
    }),
    false,
  );
});

test("reasoning and citations still neither block nor enable", () => {
  const thoughtOnly = [
    { type: "reasoning", text: "thinking" },
    { type: "source", sourceType: "url", id: "1", url: "https://example.com" },
  ];
  assert.equal(isContinuableContent(thoughtOnly), false);
  assert.equal(isContinuableContent(thoughtOnly, { allowEmpty: true }), true);
});

test("the paused label does not promise text that may not exist", () => {
  const label = incompleteLabel("paused");
  assert.ok(label.length > 0);
  assert.ok(
    !/text so far/i.test(label),
    "a turn given up on before its first token has no text so far",
  );
});
