// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
  for (const text of [
    "",
    "/rewind now",
    "rewind",
    "/rewinds",
    "please /rewind",
    "/ rewind",
  ]) {
    assert.equal(isRewindCommand(text), false, JSON.stringify(text));
  }
});

const branch = [
  { id: "u1", role: "user", content: [{ type: "text", text: "first" }] },
  { id: "a1", role: "assistant", content: [] },
  {
    id: "u2",
    role: "user",
    content: [{ type: "text", text: "second\n  line" }],
  },
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
    { id: "u3", text: "", preview: "" },
    { id: "u2", text: "second line", preview: "second line" },
    { id: "u1", text: "first", preview: "first" },
  ]);
  const long = rewindPickerEntries([
    { id: "u", role: "user", content: "x".repeat(200) },
  ]);
  assert.ok(long[0].preview.length <= 80);
  assert.ok(long[0].preview.endsWith("…"));
  assert.equal(long[0].text, "x".repeat(200));
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

test("completion payloads carry turn identity and temporary usage independently of tools", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(
    adapter,
    /const userMessageIdField = userMessageIdRequestField\(\s*temporaryChat \? undefined : generationUserMessage\?\.id,?\s*\);/,
  );
  assert.match(
    adapter,
    /\.\.\.\(resolvedThreadId \? \{ thread_id: resolvedThreadId \} : \{\}\),\s*\.\.\.userMessageIdField,\s*\.\.\.temporaryUsageFields,\s*\.\.\.\(useAdapter === undefined/,
  );
  assert.match(
    adapter,
    /model: externalSelection\.modelId,\s*messages: outboundMessages,\s*\.\.\.temporaryUsageFields,/,
  );
  assert.match(
    adapter,
    /\? \{ session_id: sandboxSessionId \}\s*: \{\}\),\s*\.\.\.userMessageIdField,/,
  );
  assert.match(
    adapter,
    /userMessageId: generationUserMessage!\.id,\s*assistantMessageId: unstable_assistantMessageId!,\s*requestPayload,/,
  );
});

test("the composer opens the rewind picker instead of sending /rewind", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  const submit = thread.slice(
    thread.indexOf("const handleSubmit = useCallback("),
  );
  const intercept = submit.indexOf("isRewindCommand(composerText)");
  assert.ok(intercept > 0);
  assert.ok(intercept < submit.indexOf("if (disabled || shouldBlockSend())"));
  assert.ok(intercept < submit.indexOf("interceptSend(event)"));
  assert.match(
    thread,
    /<RewindMessageButton \/>\s*<ForkCountBadge \/>\s*<ForkMessageButton \/>/,
  );
});

test("the dialog lists only options that can run and skips itself for a conversation-only rewind", () => {
  const dialog = readSrc("components/assistant-ui/rewind-dialog.tsx");
  assert.ok(!dialog.includes("RadioGroupItem"));
  assert.match(dialog, /if \(!hasChanges \|\| result === null\) return null;/);
  assert.match(dialog, /void rewind\("conversation"\);/);
  assert.match(dialog, /t\("rewind\.confirmTitle"\)/);
  assert.match(dialog, /<SelectTrigger id="rewind-restore"/);
});

test("an all-skipped rewind still says which files stay, though it skips the dialog", () => {
  const dialog = readSrc("components/assistant-ui/rewind-dialog.tsx");
  const effect = dialog.slice(dialog.indexOf("const conversationOnly ="));
  const warn = effect.indexOf('toast.warning(t("rewind.filesNotRestored"');
  const go = effect.indexOf('void rewind("conversation");');
  assert.ok(
    warn > 0 && go > 0,
    "the warning and the rewind are both in the conversation-only effect",
  );
  assert.ok(warn < go);
  assert.match(
    effect,
    /result\.conflicts\.length \+ result\.untracked\.length/,
  );
});
