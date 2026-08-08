// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The gallery clip link is minted by the backend as a path, and its consumers bypass authFetch:
// it goes straight into <video src> and into the download anchor's href. Under Tauri the page
// origin is the webview rather than the backend, so a relative path resolved there returns the
// SPA shell — the clip never plays and Download saves index.html renamed to .mp4. authFetch
// already routes its own requests through apiUrl(); only the URL handed back was left relative.
// Same shape as the RAG document link, which is absolute for exactly this reason.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

// features/video/api reaches authFetch through the auth barrel, which re-exports login-page.tsx.
// See helpers/auth-stub.mjs.
register("./helpers/settings-api-resolver.mjs", import.meta.url);
const { store } = installLocalStorageFake();
// The signed-url route is bearer-gated, so the caller is signed in.
store.set("unsloth_auth_token", "token");

// What the backend replies with: a path, not an absolute URL.
const RELATIVE_URL = "/api/inference/video/gallery/abc123/file?sig=deadbeef";

globalThis.fetch = (async () => {
  const res = {
    ok: true,
    json: async () => ({ url: RELATIVE_URL }),
    clone: () => res,
  };
  return res;
}) as unknown as typeof fetch;

const { setApiBase, resetApiBase } = await import("../src/lib/api-base.ts");
const { fetchGalleryVideoSignedUrl } = await import("../src/features/video/api.ts");

test("the gallery clip link is absolute when the backend is on its own origin", async () => {
  // What the desktop app does once the server port arrives.
  setApiBase(52001);
  try {
    const url = await fetchGalleryVideoSignedUrl("abc123");
    assert.equal(
      url,
      `http://127.0.0.1:52001${RELATIVE_URL}`,
      "a relative link resolves against the webview origin under Tauri, so <video src> and the download href receive the SPA shell instead of the MP4",
    );
  } finally {
    resetApiBase();
  }
});

test("the browser build keeps the same-origin path it already used", async () => {
  // apiBase stays empty in the browser, so the path is already correct there and must not be rewritten.
  resetApiBase();
  const url = await fetchGalleryVideoSignedUrl("abc123");
  assert.equal(url, RELATIVE_URL, "the browser build should be unaffected by this change");
});
