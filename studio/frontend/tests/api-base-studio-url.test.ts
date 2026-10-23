// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

import { isStudioUrl, resetApiBase, setApiBase } from "../src/lib/api-base.ts";

test("only the backend's own URLs count as Studio's, so others never get the sign-in", () => {
  assert.equal(isStudioUrl("/api/chat/attachments/a.png"), true);
  assert.equal(isStudioUrl("https://images.example/a.png"), false);
  assert.equal(isStudioUrl("http://[bad"), false);
  setApiBase(8888);
  assert.equal(isStudioUrl("/api/chat/attachments/a.png"), true);
  assert.equal(isStudioUrl("http://127.0.0.1:8888/api/chat/attachments/a.png"), true);
  assert.equal(isStudioUrl("http://127.0.0.1:9999/a.png"), false);
  assert.equal(isStudioUrl("HTTPS://images.example/a.png"), false);
  assert.equal(isStudioUrl("//images.example/a.png"), false);
  resetApiBase();
});
