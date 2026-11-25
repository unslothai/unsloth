// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  isBackendRouteUrl,
  isSignedGalleryMediaUrl,
  protectedBackendMediaPath,
  resetApiBase,
  setApiBase,
} from "../src/lib/api-base.ts";

test("a reply image loads only gallery media with the session", () => {
  for (const apiBase of [null, 8888]) {
    if (apiBase !== null) setApiBase(apiBase);
    for (const path of [
      "/api/inference/images/gallery/abc/file",
      "/api/inference/images/gallery/abc/file?thumb=1024",
      "/api/inference/video/gallery/v1/frame",
      "/v1/audio/gallery/c1/file",
      "/studio/api/inference/images/gallery/abc/file",
    ]) {
      assert.equal(protectedBackendMediaPath(path), path, path);
    }
    for (const path of [
      "/api/hub/modelscope/api/models?search=secret",
      "/api/hub/modelscope/api/inference/images/gallery/x/file",
      "/api/export/moe-info?model=attacker/repo",
      "/api/chat/pins",
      "https://evil.example/api/inference/images/gallery/abc/file",
    ]) {
      assert.equal(protectedBackendMediaPath(path), null, path);
    }
    resetApiBase();
  }
});

test("any other backend route in a reply image is recognised so it is never loaded", () => {
  assert.equal(isBackendRouteUrl("/api/hub/modelscope/api/models?search=x"), true);
  assert.equal(isBackendRouteUrl("http://localhost/api/export/moe-info?model=x"), true);
  assert.equal(isBackendRouteUrl("/v1/models"), true);
  assert.equal(isBackendRouteUrl("/assets/logo.png"), false);
  assert.equal(isBackendRouteUrl("https://images.example/api/x.png"), false);
});

test("signed gallery media still loads as a plain image in the browser", () => {
  for (const url of [
    "/api/inference/images/gallery/abc/file-signed?token=t0k",
    "http://localhost/api/inference/images/gallery/abc/file-signed?token=t0k",
    "/api/inference/audio/gallery/c1/file-signed?token=t0k",
  ]) {
    assert.equal(isSignedGalleryMediaUrl(url), true, url);
  }
  for (const url of [
    "/api/inference/images/gallery/abc/file-signed",
    "/api/hub/modelscope/api/inference/images/gallery/x/file-signed?token=t",
    "/api/export/moe-info?model=x&token=t",
    "https://evil.example/api/inference/images/gallery/abc/file-signed?token=t",
  ]) {
    assert.equal(isSignedGalleryMediaUrl(url), false, url);
  }
});
