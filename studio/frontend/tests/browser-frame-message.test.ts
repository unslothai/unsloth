// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { parseFrameMessage } from "../src/features/browser/frame-message.ts";

const from = (message: Record<string, unknown>) => parseFrameMessage({ source: "unsloth-browser", ...message });

test("malformed messages are dropped", () => {
  assert.equal(parseFrameMessage(null), null);
  assert.equal(parseFrameMessage("x"), null);
  assert.equal(parseFrameMessage({ type: "reload" }), null);
  assert.equal(from({ type: "navigate", url: 42 }), null);
  assert.equal(from({ type: "navigate", url: { trim: 1 } }), null);
  assert.equal(from({ type: "navigate", url: "x".repeat(9000) }), null);
  assert.equal(from({ type: "navigate", url: "https://a.b/", method: "POST", body: {} }), null);
  assert.equal(from({ type: "url", url: [] }), null);
  assert.equal(from({ type: "external", url: null }), null);
  assert.equal(from({ type: "shortcut", key: {} }), null);
  assert.equal(from({ type: "shortcut", key: "q" }), null);
  assert.equal(from({ type: "unknown" }), null);
});

test("titles are always bounded strings", () => {
  assert.deepEqual(from({ type: "loaded", title: {}, favicon: 7 }), { type: "loaded", title: "", favicon: null });
  assert.deepEqual(from({ type: "title", title: ["a"] }), { type: "title", title: "" });
  const long = from({ type: "title", title: "x".repeat(5000) });
  assert.equal(long?.type === "title" && long.title.length, 1024);
});

test("valid messages keep only known fields", () => {
  assert.deepEqual(from({ type: "navigate", url: "https://a.b/", newTab: "yes", extra: 1 }), {
    type: "navigate",
    url: "https://a.b/",
    newTab: false,
    background: false,
    replace: false,
  });
  assert.deepEqual(from({ type: "navigate", url: "https://a.b/", method: "POST", body: "q=1", newTab: true }), {
    type: "navigate",
    url: "https://a.b/",
    method: "POST",
    body: "q=1",
    newTab: true,
    background: false,
    replace: false,
  });
  assert.deepEqual(from({ type: "shortcut", key: "l", shift: 1 }), { type: "shortcut", key: "l", shift: false });
});

test("an upload notice carries nothing a page could fill", () => {
  assert.deepEqual(from({ type: "upload", url: "https://evil.example/" }), { type: "upload" });
  assert.deepEqual(from({ type: "scriptNavigation", url: "https://evil.example/" }), { type: "scriptNavigation" });
});

test("a page's zoom is a step in, out or back to 100%, and nothing else", () => {
  assert.deepEqual(from({ type: "zoom", direction: 1 }), { type: "zoom", direction: 1, wheel: false });
  assert.deepEqual(from({ type: "zoom", direction: -1, wheel: true }), { type: "zoom", direction: -1, wheel: true });
  assert.deepEqual(from({ type: "zoom", direction: 0, wheel: "yes" }), { type: "zoom", direction: 0, wheel: false });
  assert.equal(from({ type: "zoom", direction: 3 }), null);
  assert.equal(from({ type: "zoom", direction: "1" }), null);
  assert.equal(from({ type: "zoom" }), null);
});

test("find results are counts a page can't stretch", () => {
  assert.deepEqual(from({ type: "findResult", count: 3, active: 1 }), { type: "findResult", count: 3, active: 1 });
  assert.deepEqual(from({ type: "findResult", count: 0, active: -1 }), { type: "findResult", count: 0, active: -1 });
  assert.equal(from({ type: "findResult", count: 2, active: 2 }), null);
  assert.equal(from({ type: "findResult", count: -1, active: -1 }), null);
  assert.equal(from({ type: "findResult", count: 1.5, active: 0 }), null);
  assert.equal(from({ type: "findResult", count: "3", active: 0 }), null);
});

test("a snapshot is markup or nothing", () => {
  assert.deepEqual(from({ type: "snapshot", html: "<p>x</p>" }), { type: "snapshot", html: "<p>x</p>" });
  assert.deepEqual(from({ type: "snapshot", html: 5 }), { type: "snapshot", html: null });
  assert.deepEqual(from({ type: "snapshot", html: "x".repeat(8 * 1024 * 1024 + 1) }), { type: "snapshot", html: null });
});
