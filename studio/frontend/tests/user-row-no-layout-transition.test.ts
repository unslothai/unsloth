// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

// A bare duration-* transitions `all`, so the distant-row flex-shrink flip collapses the row for a frame.
test("the user message row does not transition its layout", () => {
  const thread = readSrc("components/assistant-ui/thread.tsx");
  const start = thread.indexOf("const UserMessage: FC");
  const root = thread.slice(start, thread.indexOf('data-role="user"', start));
  assert.match(root, /aui-user-message-root/);
  assert.doesNotMatch(root, /\bduration-\d/);
});
