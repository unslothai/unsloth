// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { fileNameFromUrl, resolveAddress, unwrapRedirect, withBaseUrl } from "../src/features/browser/address.ts";

test("the address bar takes URLs as typed and hosts as https", () => {
  assert.equal(resolveAddress("https://unsloth.ai/docs", "duckduckgo"), "https://unsloth.ai/docs");
  assert.equal(resolveAddress("  docs.unsloth.ai/get-started ", "duckduckgo"), "https://docs.unsloth.ai/get-started");
  assert.equal(resolveAddress("192.168.1.10:8080", "duckduckgo"), "https://192.168.1.10:8080");
  assert.equal(resolveAddress("localhost:8888", "duckduckgo"), "http://localhost:8888");
  assert.equal(resolveAddress("[2606:4700:4700::1111]", "duckduckgo"), "https://[2606:4700:4700::1111]");
  assert.equal(resolveAddress("example.xn--p1ai/a", "duckduckgo"), "https://example.xn--p1ai/a");
  assert.equal(resolveAddress("пример.рф", "duckduckgo"), "https://пример.рф");
  assert.equal(resolveAddress("   ", "duckduckgo"), null);
});

test("anything else is a search on the chosen engine", () => {
  assert.equal(
    resolveAddress("unsloth fine-tuning", "duckduckgo"),
    "https://html.duckduckgo.com/html/?q=unsloth%20fine-tuning",
  );
  assert.equal(resolveAddress("what is lora", "google"), "https://www.google.com/search?q=what%20is%20lora");
  assert.match(resolveAddress("node.js tutorial", "bing") ?? "", /^https:\/\/www\.bing\.com\/search\?q=/);
});

test("search engines' click-tracking hops are skipped, only to another web page", () => {
  assert.equal(
    unwrapRedirect("https://duckduckgo.com/l/?uddg=https%3A%2F%2Funsloth.ai%2Fdocs&rut=abc"),
    "https://unsloth.ai/docs",
  );
  assert.equal(unwrapRedirect("https://www.google.com/url?q=https://unsloth.ai/&sa=U"), "https://unsloth.ai/");
  for (const hop of [
    "https://unsloth.ai/l/?uddg=x",
    "https://evilduckduckgo.com/l/?uddg=https%3A%2F%2Funsloth.ai%2F",
    ...["javascript:alert(1)", "file:///etc/passwd"].map(
      (target) => `https://duckduckgo.com/l/?uddg=${encodeURIComponent(target)}`,
    ),
  ]) {
    assert.equal(unwrapRedirect(hop), hop);
  }
});

test("a document without a title is named by its path", () => {
  assert.equal(fileNameFromUrl("https://arxiv.org/pdf/1706.03762"), "1706.03762");
  assert.equal(fileNameFromUrl("https://example.com/files/My%20Report.pdf"), "My Report.pdf");
  assert.equal(fileNameFromUrl("https://example.com/"), "example.com");
});

test("a saved page gets its base URL back", () => {
  assert.equal(
    withBaseUrl("<!doctype html><html><head><title>x</title>", "https://a.example/d/?q=1&r=\"2\""),
    '<!doctype html><html><head><base href="https://a.example/d/?q=1&amp;r=&quot;2&quot;"><title>x</title>',
  );
  assert.equal(withBaseUrl("<!DOCTYPE html><p>x", "https://a.example/"), '<!DOCTYPE html><base href="https://a.example/"><p>x');
  assert.equal(withBaseUrl("<p>x", "https://a.example/"), '<base href="https://a.example/"><p>x');
  const start = performance.now();
  withBaseUrl("<head<!doctype".repeat(10_000), "https://a.example/");
  assert.ok(performance.now() - start < 1000);
});

test("bare host matching stays linear on long input", () => {
  const start = performance.now();
  resolveAddress(`${"a.".repeat(50_000)}!`, "duckduckgo");
  resolveAddress(`${"a".repeat(100_000)}.`, "duckduckgo");
  assert.ok(performance.now() - start < 1000);
});
