// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import type { ElementInfo } from "../src/features/browser/agent/page-agent.ts";
import type { AgentPermissions } from "../src/features/browser/agent/permissions.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { gateElement, gateNavigation, policyBlocks } = await import("../src/features/browser/agent/gate.ts");
const {
  DEFAULT_PERMISSIONS: NONE,
  checkedPermissions,
  normalizeSitePattern,
  ruleFor,
} = await import("../src/features/browser/agent/permissions.ts");

const open = { allowedDomains: [], blockedDomains: [] };
const navigation = (overrides: Partial<Parameters<typeof gateNavigation>[1]> = {}) => ({
  policy: open,
  permissions: NONE as AgentPermissions,
  currentUrl: "",
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
  // Naming a shared host's brand doesn't approve its users' subdomains, nor a host hidden inside a longer one.
  assert.equal(gateNavigation("https://collector.github.io/x", navigation({ userText: "search GitHub for gguf" })).kind, "ask");
  assert.equal(gateNavigation("https://e.com/x", navigation({ userText: "open example.com" })).kind, "ask");
});

test("consequential labels ask in the languages Studio ships in", () => {
  const page = "https://shop.example/cart";
  for (const name of ["Supprimer le compte", "Comprar ahora", "Jetzt überweisen", "立即购买", "注文を確定する", "구매하기", "Удалить"]) {
    assert.equal(gateElement("click", element({ name }), page, open, NONE).kind, "ask", name);
  }
  assert.equal(gateElement("click", element({ name: "Ordinary results" }), page, open, NONE).kind, "allow");
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
    assert.equal(gateElement("type", info, page, open, NONE).kind, "block", info.name);
  }
  assert.equal(gateElement("type", element({ role: "searchbox", tag: "input", type: "search", name: "Search" }), page, open, NONE).kind, "allow");
  assert.equal(gateElement("click", element({ role: "file-input", tag: "input", type: "file" }), page, open, NONE).kind, "block");
});

test("buying, posting, signing in and POST forms ask; ordinary clicks and searches don't", () => {
  const page = "https://shop.example/cart";
  assert.equal(gateElement("click", element({ name: "Place your order" }), page, open, NONE).kind, "ask");
  assert.equal(gateElement("click", element({ name: "Post reply" }), page, open, NONE).kind, "ask");
  assert.equal(gateElement("click", element({ name: "Show more results" }), page, open, NONE).kind, "allow");
  const login = element({ name: "Continue", submits: true, formMethod: "post", formAction: "https://shop.example/login", formHasPassword: true });
  const signIn = gateElement("click", login, page, open, NONE);
  assert.ok(signIn.kind === "ask" && signIn.ask === "signIn");
  const comment = element({ name: "Save", submits: true, formMethod: "post", formAction: "https://shop.example/c" });
  assert.ok(gateElement("click", comment, page, open, NONE).kind === "ask");
  const search = element({ role: "searchbox", tag: "input", name: "Search", formAction: "https://shop.example/s" });
  assert.equal(gateElement("submit", search, page, open, NONE).kind, "allow");
  // A link that only opens the sign-in page is just navigation.
  assert.equal(gateElement("click", element({ role: "link", tag: "a", name: "Sign in", href: "https://shop.example/login" }), page, open, NONE).kind, "allow");
  assert.equal(gateElement("click", element({ role: "link", tag: "a", name: "Buy now", href: "https://shop.example/buy" }), page, open, NONE).kind, "ask");
  // Ticking a box is a click on it; picking a dropdown option is not.
  assert.equal(gateElement("select", element({ role: "checkbox", tag: "input", type: "checkbox", name: "Subscribe" }), page, open, NONE).kind, "ask");
  assert.equal(gateElement("select", element({ role: "combobox", tag: "select", name: "Cancel reason" }), page, open, NONE).kind, "allow");
});

test("the durable gate keeps every browser tool on the live-tab stream", async () => {
  registerBundlerResolver();
  const { BROWSER_AGENT_TOOLS, BROWSER_HISTORY_TOOL, BROWSER_SCREENSHOT_TOOL } = await import(
    "../src/features/browser/agent/tools.ts"
  );
  const { BROWSER_EXECUTED_TOOLS } = await import("../src/features/chat/api/durable-gate.ts");
  assert.deepEqual(
    [...BROWSER_EXECUTED_TOOLS].sort(),
    [...BROWSER_AGENT_TOOLS, BROWSER_SCREENSHOT_TOOL, BROWSER_HISTORY_TOOL].sort(),
  );
});

test("a form posting to a blocked site is refused", () => {
  const policy = { allowedDomains: [], blockedDomains: ["evil.example"] };
  const info = element({ submits: true, formAction: "https://evil.example/collect" });
  assert.equal(gateElement("click", info, "https://ok.example/", policy, NONE).kind, "block");
});

const permissions = (sites: AgentPermissions["sites"], defaults = NONE.defaults): AgentPermissions => ({ defaults, sites });
const rule = (browsing: "allow" | "ask" | "block", actions: "ask" | "allow" = "ask") => ({ browsing, actions, downloads: "ask" as const });

test("site patterns are normalized hosts, and the most specific rule wins", () => {
  assert.equal(normalizeSitePattern("https://www.Example.com/path?q=1"), "example.com");
  assert.equal(normalizeSitePattern("*.Example.com"), "*.example.com");
  assert.equal(normalizeSitePattern("localhost"), null);
  assert.equal(normalizeSitePattern("not a host"), null);
  const rules = permissions({ "example.com": rule("allow"), "admin.example.com": rule("block"), "*.cdn.org": rule("ask") });
  assert.equal(ruleFor("https://docs.example.com/", rules).site, "example.com");
  assert.equal(ruleFor("https://admin.example.com/x", rules).rule.browsing, "block");
  assert.equal(ruleFor("https://cdn.org/", rules).site, null);
  assert.equal(ruleFor("https://a.cdn.org/", rules).site, "*.cdn.org");
});

test("site rules decide navigation before the new-site check", () => {
  const rules = permissions({ "allowed.example": rule("allow"), "careful.example": rule("ask"), "never.example": rule("block") });
  assert.equal(gateNavigation("https://allowed.example/?q=1", navigation({ permissions: rules })).kind, "allow");
  // Asked about even when the user named it, but not while already on it.
  assert.equal(gateNavigation("https://careful.example/", navigation({ permissions: rules, userText: "careful.example" })).kind, "ask");
  assert.equal(gateNavigation("https://careful.example/b", navigation({ permissions: rules, currentUrl: "https://careful.example/a" })).kind, "allow");
  assert.equal(gateNavigation("https://never.example/", navigation({ permissions: rules, userText: "never.example" })).kind, "block");
  // Default Blocked: only the sites with their own rule open.
  const allowlist = permissions({ "allowed.example": rule("allow") }, { browsing: "block", actions: "ask", downloads: "ask" });
  assert.equal(gateNavigation("https://other.example/", navigation({ permissions: allowlist, userText: "other.example" })).kind, "block");
  assert.equal(gateNavigation("https://allowed.example/", navigation({ permissions: allowlist })).kind, "allow");
  const open = permissions({}, { browsing: "allow", actions: "ask", downloads: "ask" });
  assert.equal(gateNavigation("https://made-up.example/?data=1", navigation({ permissions: open })).kind, "allow");
});

test("Always allow for actions skips the prompt, never the password and file rules", () => {
  const rules = permissions({ "shop.example": rule("allow", "allow") });
  const page = "https://shop.example/cart";
  assert.equal(gateElement("click", element({ name: "Place your order" }), page, open, rules).kind, "allow");
  assert.equal(gateElement("type", element({ role: "password", tag: "input", type: "password" }), page, open, rules).kind, "block");
  assert.equal(gateElement("click", element({ role: "file-input", tag: "input", type: "file" }), page, open, rules).kind, "block");
  assert.equal(gateElement("click", element({ name: "Place your order" }), "https://other.example/", open, rules).kind, "ask");
});

test("stored permissions are checked, so a bad value never reads as allow", () => {
  const checked = checkedPermissions({
    defaults: { browsing: "yes", actions: "allow", downloads: 1 },
    sites: { "Example.com": { browsing: "allow" }, "bad host": { browsing: "allow" } },
  });
  assert.deepEqual(checked.defaults, { browsing: "ask", actions: "allow", downloads: "ask" });
  assert.deepEqual(Object.keys(checked.sites), ["example.com"]);
  assert.deepEqual(checked.sites["example.com"], { browsing: "allow", actions: "ask", downloads: "ask" });
});
