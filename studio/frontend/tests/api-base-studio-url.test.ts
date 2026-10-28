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
  assert.equal(
    isStudioUrl("http://127.0.0.1:8888/api/chat/attachments/a.png"),
    true,
  );
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
  assert.equal(
    backendPathUrl("/api/inference/images/a.png"),
    "http://127.0.0.1:8888/api/inference/images/a.png",
  );
  assert.equal(backendPathUrl("/chat/abc"), "/chat/abc");
  assert.equal(
    backendPathUrl("https://example.com/v1/x"),
    "https://example.com/v1/x",
  );
  resetApiBase();
});

test("only a signed backend media URL is fetched into a blob, and only on Desktop", () => {
  const signed = "/api/inference/images/gallery/img-1/file-signed?token=t";
  assert.equal(desktopSignedMediaPath(signed), null);
  setApiBase(8888);
  assert.equal(desktopSignedMediaPath(signed), signed);
  assert.equal(
    desktopSignedMediaPath("/v1/audio/gallery/a/file-signed?token=t"),
    "/v1/audio/gallery/a/file-signed?token=t",
  );
  for (const other of [
    "/api/inference/images/gallery/img-1/file",
    "/api/inference/images/gallery/img-1/file-signed?token=t&x=1",
    "/api/settings/file-signed",
    "http://127.0.0.1:8888/api/x/file-signed?token=t",
    "https://evil.example/api/x/file-signed?token=t",
    "//evil.example/api/x/file-signed?token=t",
  ]) {
    assert.equal(desktopSignedMediaPath(other), null, other);
  }
  resetApiBase();
});

test("an unsigned backend link is fetched with the session, a signed or outside one is not", () => {
  for (const path of [
    "/v1/audio/voices/v1/file",
    "/api/gallery/x?archived=1",
    "/studio/v1/audio/voices/v1/file",
  ]) {
    assert.equal(protectedBackendPath(path), path, path);
  }
  for (const other of [
    "/v1/audio/gallery/a/file-signed?token=t",
    "/studio/api/inference/images/gallery/a/file-signed?token=t",
    "/chat",
    "https://example.com/v1/x",
    "//example.com/v1/x",
    "/v1/",
  ]) {
    assert.equal(protectedBackendPath(other), null, other);
  }
});
