// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { type FrameMessage, frameNotice, parseFrameMessage, userActionGate } from "../src/features/browser/frame-message.ts";

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

test("a page the sandbox refused a feature reaches the panel as a bare notice", () => {
  assert.deepEqual(from({ type: "needsBrowser", url: "https://evil.example/" }), { type: "needsBrowser" });
});

test("typing in a page reaches the panel without the text", () => {
  assert.deepEqual(from({ type: "typed", value: "secret" }), { type: "typed" });
  assert.deepEqual(from({ type: "typed", reset: true, value: "secret" }), { type: "typed", reset: true });
  assert.deepEqual(from({ type: "typed", reset: "true" }), { type: "typed" });
});

test("the sandbox notice stays until the page goes, the others time out", () => {
  assert.deepEqual(frameNotice("needsBrowser", "t1"), { id: "browser-needsBrowser-t1", duration: Number.POSITIVE_INFINITY });
  assert.deepEqual(frameNotice("upload", "t1"), { id: "browser-upload-t1" });
  assert.deepEqual(frameNotice("scriptNavigation", "t2"), { id: "browser-scriptNavigation-t2" });
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

test("a page's request is checked field by field, and one the panel won't make still gets an answer", () => {
  const body = new ArrayBuffer(4);
  const request = { url: "https://site.test/api", method: "POST", headers: [["X-A", "1"]], body, redirect: "manual", integrity: "sha256-x" };
  assert.deepEqual(from({ type: "request", id: "q1", ...request }), { type: "request", id: "q1", request });
  // Fetch's default redirect mode when the page names none.
  assert.deepEqual(from({ type: "request", id: "q1", ...request, body: null, redirect: undefined, integrity: undefined }), {
    type: "request",
    id: "q1",
    request: { ...request, body: null, redirect: "follow", integrity: "" },
  });
  // Without an id there is no one to answer.
  assert.equal(from({ type: "request", ...request }), null);
  for (const bad of [
    { body: new ArrayBuffer(1024 * 1024 + 1) },
    { body: "text" },
    { method: "GET POST" },
    { url: 7 },
    { headers: [["X-A"]] },
    { headers: [["X-A", 1]] },
    { headers: Array.from({ length: 65 }, () => ["X-A", "1"]) },
    { redirect: "skip" },
  ]) {
    assert.deepEqual(from({ type: "request", id: "q1", ...request, ...bad }), { type: "request", id: "q1", request: null });
  }
  assert.deepEqual(from({ type: "requestAbort", id: "q1" }), { type: "requestAbort", id: "q1" });
  assert.equal(from({ type: "requestAbort", id: 3 }), null);
});

test("a file a page saves reaches the panel with its bytes and name only", () => {
  const blob = new Blob(["a,b"], { type: "text/csv" });
  const message = from({ type: "download", name: "data.csv", blob, url: "https://evil.example/" });
  assert.deepEqual(message && Object.keys(message), ["type", "name", "blob"]);
  assert.equal(message && "blob" in message && message.blob, blob);
  assert.equal(message?.type === "download" && message.name, "data.csv");
  assert.equal(from({ type: "download", name: "data.csv", blob: "a,b" }), null);
  assert.equal(from({ type: "download", name: "", blob }), null);
  const long = from({ type: "download", name: `${"x".repeat(5000)}.csv`, blob });
  assert.ok(long?.type === "download" && long.name.length <= 1024 && /^x+\.csv$/.test(long.name));
  const unnamed = from({ type: "download", name: "\u{1F600}".repeat(3000), blob });
  assert.ok(unnamed?.type === "download" && unnamed.name.length <= 1024 && !/[\uD800-\uDBFF]$/.test(unnamed.name));
});

test("a file sent without a fresh click, or within a second of the last action, waits for Save instead of being dropped", () => {
  const gate = userActionGate();
  const file: FrameMessage = { type: "download", name: "data.csv", blob: new Blob(["a,b"]) };
  const asks = (message: FrameMessage | null) => (message?.type === "download" ? message.askFirst === true : message);
  assert.equal(asks(gate(file, false, 0)), true);
  assert.equal(asks(gate(file, true, 10)), false);
  assert.equal(asks(gate(file, true, 500)), true);
  assert.equal(gate({ type: "shortcut", key: "t", shift: false }, true, 600), null);
  assert.equal(asks(gate(file, true, 1200)), false);
  assert.equal(gate({ type: "external", url: "https://a.example/" }, false, 5000), null);
  assert.deepEqual(gate({ type: "title", title: "x" }, false, 5000), { type: "title", title: "x" });
});

test("a download link reaches the panel as its address and the link's name, which may be empty", () => {
  const url = "https://a.example/data.txt";
  assert.deepEqual(from({ type: "download", url, name: "" }), { type: "download", name: "", url });
  assert.deepEqual(from({ type: "download", url, name: "data.txt", extra: 1 }), { type: "download", name: "data.txt", url });
  assert.equal(from({ type: "download", name: "data.txt" }), null);
  assert.equal(from({ type: "download", url, name: 3 }), null);
  const gate = userActionGate();
  const link: FrameMessage = { type: "download", name: "", url };
  assert.deepEqual(gate(link, false, 0), { ...link, askFirst: true });
});
