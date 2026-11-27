// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

const data = new Map<string, string>([["unsloth.browser-account.v1", "account:b1:bob"]]);
const localStorage = {
  getItem: (key: string) => data.get(key) ?? null,
  setItem: (key: string, value: string) => void data.set(key, value),
  removeItem: (key: string) => void data.delete(key),
};
Object.assign(globalThis, {
  localStorage,
  window: { localStorage, location: { protocol: "http:" }, addEventListener: () => {} },
});

register("./helpers/browser-store-resolver.mjs", import.meta.url);
const { MAX_CAPTURE_SITES, captureRules, normalizeCaptureOrigin, useCaptureSitesStore, watchCaptureRules } = await import(
  "../src/features/browser/capture-sites-store.ts"
);
const { accountDatabaseName } = await import("../src/lib/account-transition.ts");

const BLOCK_CAMERA = { camera: "block", microphone: "ask" } as const;
const reset = () => useCaptureSitesStore.setState({ camera: "ask", microphone: "ask", sites: {} });

test("a typed site becomes an http(s) origin, or is refused", () => {
  assert.equal(normalizeCaptureOrigin(" Example.com/call?x=1 "), "https://example.com");
  assert.equal(normalizeCaptureOrigin("http://Meet.Example.com.:8080/a"), "http://meet.example.com:8080");
  assert.equal(normalizeCaptureOrigin("https://example.com:443"), "https://example.com");
  assert.equal(normalizeCaptureOrigin("localhost:3000"), "https://localhost:3000");
  for (const bad of ["", "   ", "two words", "file:///etc/passwd", "javascript://x", "ftp://example.com", `${"a".repeat(300)}.example`]) {
    assert.equal(normalizeCaptureOrigin(bad), null, bad);
  }
});

test("defaults and site rules are kept per account, and asking is where both start", () => {
  const store = useCaptureSitesStore.getState();
  assert.deepEqual(captureRules(store), { camera: "ask", microphone: "ask", sites: {} });
  store.setDefault("microphone", "block");
  assert.equal(store.setSite("https://a.example", BLOCK_CAMERA), true);
  store.setSite("https://b.example", { camera: "ask", microphone: "ask" });
  store.setSite("https://b.example", null);
  assert.deepEqual(captureRules(useCaptureSitesStore.getState()), {
    camera: "ask",
    microphone: "block",
    sites: { "https://a.example": BLOCK_CAMERA },
  });
  const name = useCaptureSitesStore.persist.getOptions().name;
  assert.equal(name, accountDatabaseName("unsloth_browser_capture_sites"));
  assert.notEqual(name, "unsloth_browser_capture_sites");
  assert.equal(JSON.parse(data.get(name!)!).state.microphone, "block");
  reset();
});

test("the list stops at the number of sites the desktop app accepts", () => {
  const sites = Object.fromEntries(Array.from({ length: MAX_CAPTURE_SITES }, (_, index) => [`https://s${index}.example`, BLOCK_CAMERA]));
  useCaptureSitesStore.setState({ sites });
  const store = useCaptureSitesStore.getState();
  assert.equal(store.setSite("https://one-more.example", BLOCK_CAMERA), false);
  assert.equal(store.setSite("https://s0.example", { camera: "ask", microphone: "block" }), true);
  assert.equal(Object.keys(useCaptureSitesStore.getState().sites).length, MAX_CAPTURE_SITES);
  reset();
});

test("what is sent never holds an allow, a bad origin or an unknown rule", () => {
  const stored = {
    camera: "allow",
    microphone: "block",
    sites: {
      "https://a.example": { camera: "allow", microphone: "block" },
      "HTTPS://B.example./x": { camera: "block" },
      "file:///etc/passwd": BLOCK_CAMERA,
      "not a site": BLOCK_CAMERA,
    },
  } as unknown as Parameters<typeof captureRules>[0];
  const rules = captureRules(stored);
  assert.deepEqual(rules, {
    camera: "ask",
    microphone: "block",
    sites: {
      "https://a.example": { camera: "ask", microphone: "block" },
      "https://b.example": { camera: "block", microphone: "ask" },
    },
  });
  assert.ok(!JSON.stringify(rules).includes("allow"));
});

test("the rules reach the desktop app at startup and on every change, and a failed send is retried", async () => {
  useCaptureSitesStore.getState().setDefault("camera", "block");
  const sent: unknown[] = [];
  let fail = false;
  const stop = watchCaptureRules(async (rules) => {
    sent.push(structuredClone(rules));
    if (fail) throw new Error("no such command");
  });
  assert.deepEqual(sent, [{ camera: "block", microphone: "ask", sites: {} }]);
  useCaptureSitesStore.getState().setSite("https://a.example", BLOCK_CAMERA);
  assert.deepEqual(sent[1], { camera: "block", microphone: "ask", sites: { "https://a.example": BLOCK_CAMERA } });
  useCaptureSitesStore.getState().setDefault("camera", "block");
  assert.equal(sent.length, 2);
  fail = true;
  useCaptureSitesStore.getState().setDefault("microphone", "block");
  await new Promise((resolve) => setTimeout(resolve, 0));
  fail = false;
  useCaptureSitesStore.setState({});
  assert.equal(sent.length, 4);
  assert.deepEqual(sent[3], sent[2]);
  stop();
  useCaptureSitesStore.getState().setDefault("camera", "ask");
  assert.equal(sent.length, 4);
  reset();
});
