// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Deleting one message used to re-render every message in the thread, which is why the cost of a
// delete grew with thread length: 98ms at 25K characters of content, 472ms at 300K.
//
// What removes that growth is not visible in the rendered output. The DOM is identical either
// way, so nothing that inspects the result can tell the two apart; what differs is how much of
// the tree React walks to produce it. Like research-render-budget.test.ts and
// drag-costs-no-render.test.ts, the wiring is therefore pinned at the source: assert the cheap
// path, assert the expensive one is gone.
//
// Each of the three seams below is silently load-bearing. Undo any one and the thread still
// renders correctly, the unit tests beside this file still pass, and the delete goes back to
// being linear in thread length.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

function source(path: string): string {
  return readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");
}

const thread = source("components/assistant-ui/thread.tsx");

function block(start: string): string {
  const [, rest] = thread.split(start, 2);
  assert.ok(rest !== undefined, `thread.tsx no longer contains ${start}`);
  const [body] = (rest ?? "").split("\n};", 1);
  return body ?? "";
}

test("the message list is rendered through a render prop, not a components map", () => {
  // The map form returns <ThreadMessageComponent components={...} />, whose props object is
  // freshly allocated per render, and assistant-ui only skips a message subtree when the element
  // the render prop returns has no props at all.
  assert.match(
    thread,
    /<ThreadPrimitive\.Messages>\s*\{renderThreadMessage\}\s*<\/ThreadPrimitive\.Messages>/,
  );
  assert.doesNotMatch(thread, /<ThreadPrimitive\.Messages[^>]*\scomponents=/s);
});

test("the render prop is built once, at module scope", () => {
  // An arrow written inline in Thread would be a new function on every Thread render, and
  // ThreadPrimitive.Messages memoizes on children identity: the message array would be rebuilt
  // from scratch each time and there would be nothing left for the bail-out to skip.
  assert.match(
    thread,
    /^const renderThreadMessage = proplessSlot\(ThreadMessage\);$/m,
  );
});

test("ThreadMessage sends each kind to the component that names it", () => {
  const body = block("const ThreadMessage: FC = () => {");
  assert.match(body, /threadMessageKind\(role, isEditing\)/);
  assert.match(body, /case "edit":\s*return <EditComposer \/>;/);
  assert.match(body, /case "user":\s*return <UserMessage \/>;/);
  assert.match(body, /case "assistant":\s*return <AssistantMessage \/>;/);
  assert.match(body, /default:\s*return null;/);
});

test("research-reply ownership is selected as an answer, not as the message list", () => {
  const hook = block("const useOwnsResearchMessage = () => {");
  // Selecting the array is what subscribed every user message's action bar to every thread
  // change, so one delete re-rendered all of them along with their tooltips.
  assert.doesNotMatch(
    hook,
    /useAuiState\(\(\{ thread \}\) => thread\.messages\)/,
  );
  // The revision key has to be the array the store hands out. A copy is a new object on every
  // read, so the memo would never hit and a full repository export would be back inside every
  // getSnapshot, which is worse than what this replaced.
  assert.match(hook, /researchReplyOwners\(\s*thread\.messages,/);
  assert.doesNotMatch(
    hook,
    /researchReplyOwners\(\s*\[\.\.\.thread\.messages\]/,
  );
});
