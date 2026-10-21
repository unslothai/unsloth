// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

import { backendPathUrl, isStudioUrl, resetApiBase, setApiBase } from "../src/lib/api-base.ts";

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

test("a media tool's origin-relative backend link points at the backend on Desktop", () => {
  const clip = "/v1/audio/gallery/clip-1/file-signed?token=t";
  assert.equal(backendPathUrl(clip), clip);
  setApiBase(8888);
  assert.equal(backendPathUrl(clip), `http://127.0.0.1:8888${clip}`);
  assert.equal(backendPathUrl("/api/inference/images/a.png"), "http://127.0.0.1:8888/api/inference/images/a.png");
  assert.equal(backendPathUrl("/chat/abc"), "/chat/abc");
  assert.equal(backendPathUrl("https://example.com/v1/x"), "https://example.com/v1/x");
  resetApiBase();
});
