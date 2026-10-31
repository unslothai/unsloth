// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  backendPathUrl,
  desktopSignedMediaPath,
  isStudioUrl,
  protectedBackendPath,
  resetApiBase,
  setApiBase,
} from "../src/lib/api-base.ts";

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

test("on Desktop a backend link points at the backend and only signed media is a blob", () => {
  const signed = "/v1/audio/a/file-signed?token=t";
  assert.equal(backendPathUrl(signed), signed);
  assert.equal(desktopSignedMediaPath(signed), null);
  setApiBase(8888);
  for (const [path, onBackend, blob, session] of [
    [signed, true, true, false],
    ["/api/x/file-signed?token=t", true, true, false],
    ["/api/x/file-signed?token=t&x=1", true, false, false],
    ["/v1/audio/voices/v1/file", true, false, true],
    ["/api/gallery/x?archived=1", true, false, true],
    ["/studio/v1/audio/voices/v1/file", false, false, true],
    ["https://evil.example/api/x/file-signed?token=t", false, false, false],
    ["https://example.com/v1/x", false, false, false],
    ["//example.com/v1/x", false, false, false],
    ["/chat", false, false, false],
    ["/v1/", true, false, false],
  ] as const) {
    const url = onBackend ? `http://127.0.0.1:8888${path}` : path;
    assert.equal(backendPathUrl(path), url, path);
    assert.equal(desktopSignedMediaPath(path), blob ? path : null, path);
    assert.equal(protectedBackendPath(path), session ? path : null, path);
  }
  resetApiBase();
});
