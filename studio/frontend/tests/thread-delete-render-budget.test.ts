// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The DOM is identical either way, so the cheap render path is pinned at the source.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const thread = readSrc("components/assistant-ui/thread.tsx");

function block(start: string): string {
  const [, rest] = thread.split(start, 2);
  assert.ok(rest !== undefined, `thread.tsx no longer contains ${start}`);
  const [body] = (rest ?? "").split("\n};", 1);
  return body ?? "";
}

test("the message list is rendered through a render prop, not a components map", () => {
  // assistant-ui only skips a message subtree when the render prop's element has no props.
  assert.match(thread, /renderMessage=\{renderThreadMessage\}/);
  assert.doesNotMatch(thread, /<ThreadPrimitive\.Messages\b/);
  assert.doesNotMatch(thread, /<ProgressiveMessages[^>]*\scomponents=/s);
});

test("the render prop is built once, at module scope", () => {
  // Messages memoizes on children identity, so an inline arrow rebuilds the array each render.
  assert.match(
    thread,
    /^const renderThreadMessage = proplessSlot\(ThreadMessage\);$/m,
  );
});

test("ThreadMessage sends each kind to the component that names it", () => {
  const body = block("const ThreadMessage: FC = () => {");
  assert.match(body, /threadMessageKind\(role, isEditing\)/);
  assert.match(body, /case "edit":\s*body = <EditComposer \/>;/);
  assert.match(body, /case "user":\s*body = <UserMessage \/>;/);
  assert.match(body, /case "assistant":\s*body = <AssistantMessage \/>;/);
  assert.match(body, /default:\s*return null;/);
  assert.match(body, /\{body\}\s*<ForkContinuationRule \/>/);
});

test("research-reply ownership is selected as an answer, not as the message list", () => {
  const hook = block("const useOwnsResearchMessage = () => {");
  // Selecting the array subscribed every action bar to every thread change.
  assert.doesNotMatch(
    hook,
    /useAuiState\(\(\{ thread \}\) => thread\.messages\)/,
  );
  // The key must be the store's own array; a copy never hits the memo.
  assert.match(hook, /researchReplyOwners\(\s*thread\.messages,/);
  assert.doesNotMatch(
    hook,
    /researchReplyOwners\(\s*\[\.\.\.thread\.messages\]/,
  );
});
