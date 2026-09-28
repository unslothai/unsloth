// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Rendered hover, focus, touch, layout and streaming coverage lives in
// tests/studio/playwright_user_message_time.py. These checks pin the production
// call sites to those tested components and cover invalid/estimated dates.
import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";
import { messageTimestamp } from "../src/lib/message-timestamp.ts";

const THREAD = readSrc("components/assistant-ui/thread.tsx");
const ACTIONS = readSrc("components/assistant-ui/user-message-actions.tsx");
const TIME = readSrc("components/assistant-ui/user-message-time.tsx");

test("user messages use the tested footer and focus/touch reveal", () => {
  const start = THREAD.indexOf("const UserMessage: FC = () => {");
  assert.notEqual(start, -1);
  const message = THREAD.slice(
    start,
    THREAD.indexOf("const EditComposer:", start),
  );
  assert.match(message, /useActionBarFocusReveal\(\)/);
  assert.match(message, /tabIndex=\{0\}/);
  assert.match(message, /\{\.\.\.focusReveal\}/);
  assert.match(message, /<UserMessageFooter>/);
  assert.match(message, /<UserMessageActionBar>/);
  assert.match(message, /aui-user-message-content-wrapper flex w-full/);
  assert.match(ACTIONS, /autohide="always"/);
  assert.match(ACTIONS, /<UserMessageTime \/>/);
});

test("the timestamp has no periodic clock work", () => {
  assert.doesNotMatch(TIME, /setInterval|setTimeout|useEffect/);
});

test("only known, valid message times are displayed", () => {
  const createdAt = new Date("2025-09-11T23:22:00Z");
  assert.equal(messageTimestamp({ createdAt }), createdAt.getTime());
  assert.equal(messageTimestamp({}), undefined);
  assert.equal(messageTimestamp({ createdAt: new Date(NaN) }), undefined);
  assert.equal(
    messageTimestamp({
      createdAt,
      metadata: { custom: { createdAtEstimated: true } },
    }),
    undefined,
  );
  assert.equal(
    messageTimestamp({
      createdAt,
      metadata: { custom: { createdAtEstimated: false } },
    }),
    createdAt.getTime(),
  );
});
