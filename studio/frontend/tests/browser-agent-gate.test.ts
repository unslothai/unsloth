// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { gateElement, gateNavigation, policyBlocks } from "../src/features/browser/agent/gate.ts";
import type { ElementInfo } from "../src/features/browser/agent/page-agent.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

const open = { allowedDomains: [], blockedDomains: [] };
const navigation = (overrides: Partial<Parameters<typeof gateNavigation>[1]> = {}) => ({
  policy: open,
  knownHosts: new Set<string>(),
  pageLinks: new Set<string>(),
  userText: "",
  ...overrides,
});

const element = (overrides: Partial<ElementInfo> = {}): ElementInfo => ({
  role: "button",
  name: "Next",
  tag: "button",
  type: "button",
  href: "",
  autocomplete: "",
  formMethod: "get",
  formAction: "",
  formHasPassword: false,
  submits: false,
  ...overrides,
});

test("a site the user named, already reached, or linked from the page opens without asking", () => {
  assert.equal(gateNavigation("https://en.wikipedia.org/wiki/GGUF", navigation({ userText: "look up GGUF on wikipedia" })).kind, "allow");
  assert.equal(gateNavigation("https://huggingface.co/unsloth", navigation({ userText: "check huggingface.co" })).kind, "allow");
  assert.equal(gateNavigation("https://github.com/x", navigation({ knownHosts: new Set(["github.com"]) })).kind, "allow");
  assert.equal(gateNavigation("https://other.example/a", navigation({ pageLinks: new Set(["https://other.example/a"]) })).kind, "allow");
});

test("an address the model made up for a new site asks first", () => {
  const gate = gateNavigation("https://collector.example/?q=secret", navigation({ userText: "summarize my notes" }));
  assert.equal(gate.kind, "ask");
  assert.ok(gate.kind === "ask" && gate.ask === "open" && gate.host === "collector.example");
});

test("non-web schemes and the website policy are refused outright", () => {
  assert.equal(gateNavigation("javascript:alert(1)", navigation()).kind, "block");
  assert.equal(gateNavigation("file:///etc/passwd", navigation()).kind, "block");
  const policy = { allowedDomains: ["wikipedia.org"], blockedDomains: [] };
  assert.equal(gateNavigation("https://example.com/", navigation({ policy, userText: "example.com" })).kind, "block");
  assert.equal(policyBlocks("https://de.wikipedia.org/", policy), null);
  assert.match(policyBlocks("https://x.evil.example/", { allowedDomains: [], blockedDomains: ["evil.example"] }) ?? "", /blocks/);
});

test("password, payment and one-time-code fields are never typed into", () => {
  const page = "https://shop.example/checkout";
  for (const info of [
    element({ role: "password", tag: "input", type: "password" }),
    element({ role: "textbox", tag: "input", type: "text", autocomplete: "cc-number" }),
    element({ role: "textbox", tag: "input", type: "text", autocomplete: "one-time-code" }),
    element({ role: "textbox", tag: "input", type: "text", name: "Card number" }),
  ]) {
    assert.equal(gateElement("type", info, page, open).kind, "block", info.name);
  }
  assert.equal(gateElement("type", element({ role: "searchbox", tag: "input", type: "search", name: "Search" }), page, open).kind, "allow");
  assert.equal(gateElement("click", element({ role: "file-input", tag: "input", type: "file" }), page, open).kind, "block");
});

test("buying, posting, signing in and POST forms ask; ordinary clicks and searches don't", () => {
  const page = "https://shop.example/cart";
  assert.equal(gateElement("click", element({ name: "Place your order" }), page, open).kind, "ask");
  assert.equal(gateElement("click", element({ name: "Post reply" }), page, open).kind, "ask");
  assert.equal(gateElement("click", element({ name: "Show more results" }), page, open).kind, "allow");
  const login = element({ name: "Continue", submits: true, formMethod: "post", formAction: "https://shop.example/login", formHasPassword: true });
  const signIn = gateElement("click", login, page, open);
  assert.ok(signIn.kind === "ask" && signIn.ask === "signIn");
  const comment = element({ name: "Save", submits: true, formMethod: "post", formAction: "https://shop.example/c" });
  assert.ok(gateElement("click", comment, page, open).kind === "ask");
  const search = element({ role: "searchbox", tag: "input", name: "Search", formAction: "https://shop.example/s" });
  assert.equal(gateElement("submit", search, page, open).kind, "allow");
  // A link that only opens the sign-in page is just navigation.
  assert.equal(gateElement("click", element({ role: "link", tag: "a", name: "Sign in", href: "https://shop.example/login" }), page, open).kind, "allow");
  assert.equal(gateElement("click", element({ role: "link", tag: "a", name: "Buy now", href: "https://shop.example/buy" }), page, open).kind, "ask");
});

test("the durable gate keeps every browser tool on the live-tab stream", async () => {
  registerBundlerResolver();
  const { BROWSER_AGENT_TOOLS, BROWSER_SCREENSHOT_TOOL } = await import("../src/features/browser/agent/tools.ts");
  const { BROWSER_EXECUTED_TOOLS } = await import("../src/features/chat/api/durable-gate.ts");
  assert.deepEqual([...BROWSER_EXECUTED_TOOLS].sort(), [...BROWSER_AGENT_TOOLS, BROWSER_SCREENSHOT_TOOL].sort());
});

test("a form posting to a blocked site is refused", () => {
  const policy = { allowedDomains: [], blockedDomains: ["evil.example"] };
  const info = element({ submits: true, formAction: "https://evil.example/collect" });
  assert.equal(gateElement("click", info, "https://ok.example/", policy).kind, "block");
});
