// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A prompt's send time shows on hover, left of its action bar. It must stay inside that
// autohide bar, which mounts only while the message is hovered, so an idle thread renders
// no timestamps and runs no timers.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const THREAD = readSrc("components/assistant-ui/thread.tsx");
const TIME = readSrc("components/assistant-ui/user-message-time.tsx");

test("the time is the first child of the hover-only user action bar", () => {
  const start = THREAD.indexOf("const UserActionBar: FC = () => {");
  assert.notEqual(
    start,
    -1,
    "UserActionBar is gone; this test needs rewriting",
  );
  const bar = THREAD.slice(start, THREAD.indexOf("\n};", start));
  assert.match(bar, /autohide="always"/);
  assert.match(bar, /<UserMessageTime \/>\s*<CopyButton \/>/);
});

test("the time renders once per hover, with no timer or interval", () => {
  assert.doesNotMatch(TIME, /setInterval|setTimeout|useEffect/);
  assert.match(TIME, /formatMessageDate\(/);
});
