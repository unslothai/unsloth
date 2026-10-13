// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// /rewind: the command match, the turn ids sent to POST /rewind, and the turn id every
// completion carries so the backend can key a turn's file checkpoints to its prompt.

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";
import {
  collectRewindTurnIds,
  defaultRewindMode,
  isRewindCommand,
  rewindPickerEntries,
  userMessageIdRequestField,
} from "../src/features/chat/utils/rewind.ts";

test("only the bare /rewind command matches", () => {
  for (const text of ["/rewind", "  /rewind\n", "/REWIND", "/Rewind "]) {
    assert.equal(isRewindCommand(text), true, JSON.stringify(text));
  }
  for (const text of ["", "/rewind now", "rewind", "/rewinds", "please /rewind", "/ rewind"]) {
    assert.equal(isRewindCommand(text), false, JSON.stringify(text));
  }
});

const branch = [
  { id: "u1", role: "user", content: [{ type: "text", text: "first" }] },
  { id: "a1", role: "assistant", content: [] },
  { id: "u2", role: "user", content: [{ type: "text", text: "second\n  line" }] },
  { id: "a2", role: "assistant", content: [] },
  { id: "u3", role: "user", content: [{ type: "image", image: "x" }] },
  { id: "a3", role: "assistant", content: [] },
];

test("turn ids run from the chosen prompt to the end of the branch, in order", () => {
  assert.deepEqual(collectRewindTurnIds(branch, "u1"), ["u1", "u2", "u3"]);
  assert.deepEqual(collectRewindTurnIds(branch, "u2"), ["u2", "u3"]);
  assert.deepEqual(collectRewindTurnIds(branch, "u3"), ["u3"]);
});

test("an assistant id or a message off the branch yields no turns", () => {
  assert.deepEqual(collectRewindTurnIds(branch, "a1"), []);
  assert.deepEqual(collectRewindTurnIds(branch, "missing"), []);
});

test("the picker lists prompts newest first on one line", () => {
  assert.deepEqual(rewindPickerEntries(branch), [
    { id: "u3", preview: "" },
    { id: "u2", preview: "second line" },
    { id: "u1", preview: "first" },
  ]);
  const long = rewindPickerEntries([{ id: "u", role: "user", content: "x".repeat(200) }]);
  assert.ok(long[0].preview.length <= 80);
  assert.ok(long[0].preview.endsWith("…"));
});

test("files are offered by default only when the turns changed files", () => {
  assert.equal(defaultRewindMode(true), "both");
  assert.equal(defaultRewindMode(false), "conversation");
});

test("user_message_id is omitted without a user message", () => {
  assert.deepEqual(userMessageIdRequestField("m1"), { user_message_id: "m1" });
  assert.deepEqual(userMessageIdRequestField(undefined), {});
  assert.deepEqual(userMessageIdRequestField(""), {});
});

test("the completion payload carries user_message_id on the local and external tool paths", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(
    adapter,
    /const userMessageIdField = userMessageIdRequestField\(\s*generationUserMessage\?\.id,?\s*\);/,
  );
  // Local body (also the durable run's requestPayload): beside thread_id.
  assert.match(
    adapter,
    /\.\.\.\(resolvedThreadId \? \{ thread_id: resolvedThreadId \} : \{\}\),\s*\.\.\.userMessageIdField,\s*\.\.\.\(useAdapter === undefined/,
  );
  // External provider with Studio-run tools: beside session_id.
  assert.match(
    adapter,
    /\? \{ session_id: sandboxSessionId \}\s*: \{\}\),\s*\.\.\.userMessageIdField,/,
  );
  // The durable run posts the same requestPayload, so the field rides inside it, not beside it.
  assert.match(
    adapter,
    /userMessageId: generationUserMessage!\.id,\s*assistantMessageId: unstable_assistantMessageId!,\s*requestPayload,/,
  );
});

test("the composer opens the rewind picker instead of sending /rewind", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  const submit = thread.slice(thread.indexOf("const handleSubmit = useCallback("));
  const intercept = submit.indexOf("isRewindCommand(composerText)");
  assert.ok(intercept > 0);
  assert.ok(intercept < submit.indexOf("if (disabled || shouldBlockSend())"));
  assert.ok(intercept < submit.indexOf("interceptSend(event)"));
  assert.match(thread, /<RewindMessageMenuItem \/>\s*<DeleteMessageMenuItem \/>/);
});
