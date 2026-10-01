// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { fileNameFromUrl, resolveAddress, unwrapRedirect } from "../src/features/browser/address.ts";

test("the address bar takes URLs as typed and hosts as https", () => {
  assert.equal(resolveAddress("https://unsloth.ai/docs", "duckduckgo"), "https://unsloth.ai/docs");
  assert.equal(resolveAddress("  docs.unsloth.ai/get-started ", "duckduckgo"), "https://docs.unsloth.ai/get-started");
  assert.equal(resolveAddress("192.168.1.10:8080", "duckduckgo"), "https://192.168.1.10:8080");
  assert.equal(resolveAddress("localhost:8888", "duckduckgo"), "http://localhost:8888");
  assert.equal(resolveAddress("   ", "duckduckgo"), null);
});

test("anything else is a search on the chosen engine", () => {
  assert.equal(
    resolveAddress("unsloth fine-tuning", "duckduckgo"),
    "https://html.duckduckgo.com/html/?q=unsloth%20fine-tuning",
  );
  assert.equal(resolveAddress("what is lora", "google"), "https://www.google.com/search?q=what%20is%20lora");
  // A dotted word with spaces around it is a query, not a host.
  assert.match(resolveAddress("node.js tutorial", "bing") ?? "", /^https:\/\/www\.bing\.com\/search\?q=/);
});

test("search engines' click-tracking hops are skipped", () => {
  assert.equal(
    unwrapRedirect("https://duckduckgo.com/l/?uddg=https%3A%2F%2Funsloth.ai%2Fdocs&rut=abc"),
    "https://unsloth.ai/docs",
  );
  assert.equal(unwrapRedirect("https://www.google.com/url?q=https://unsloth.ai/&sa=U"), "https://unsloth.ai/");
  assert.equal(unwrapRedirect("https://unsloth.ai/l/?uddg=x"), "https://unsloth.ai/l/?uddg=x");
});

test("a document without a title is named by its path", () => {
  assert.equal(fileNameFromUrl("https://arxiv.org/pdf/1706.03762"), "1706.03762");
  assert.equal(fileNameFromUrl("https://example.com/files/My%20Report.pdf"), "My Report.pdf");
  assert.equal(fileNameFromUrl("https://example.com/"), "example.com");
});

test("a search engine's redirect hop is only followed to another web page", () => {
  assert.equal(
    unwrapRedirect("https://duckduckgo.com/l/?uddg=https%3A%2F%2Funsloth.ai%2F"),
    "https://unsloth.ai/",
  );
  for (const target of ["javascript:alert(1)", "file:///etc/passwd"]) {
    const hop = `https://duckduckgo.com/l/?uddg=${encodeURIComponent(target)}`;
    assert.equal(unwrapRedirect(hop), hop, target);
  }
  const lookalike = "https://evilduckduckgo.com/l/?uddg=https%3A%2F%2Funsloth.ai%2F";
  assert.equal(unwrapRedirect(lookalike), lookalike);
});
