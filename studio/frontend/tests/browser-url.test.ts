// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import { MessageChannel } from "node:worker_threads";

// Run the shell's boot through URL installation, before the rest needs a DOM.
const source = readFileSync(
  new URL("../../backend/routes/browser.py", import.meta.url),
  "utf8",
)
  .split("const boot = (cfg) => {")[1]
  .split("const resolve = ")[0];

function setup(
  pageUrl:
    | string
    | null = "https://platform.claude.com/docs/en/api/messages/create",
) {
  const window = { URL, location: { origin: "null", href: "about:srcdoc" } };
  runInNewContext(source, {
    window,
    URL,
    location: window.location,
    cfg: { url: pageUrl },
    document: { querySelector: () => null },
    MessageChannel,
    parent: { postMessage() {} },
  });
  return window;
}

test("Claude API docs can construct a URL from the opaque window origin", () => {
  const window = setup();
  const url = new window.URL(window.location.origin);
  assert.equal(url.href, "https://platform.claude.com/");
  assert.equal(url.origin, "https://platform.claude.com");
  assert.ok(url instanceof URL);
});

test("the origin rewrite preserves non-default ports and drops the page path", () => {
  const window = setup(
    "https://docs.example.com:8443/api/messages?version=1#request",
  );
  assert.equal(
    new window.URL(window.location.origin).href,
    "https://docs.example.com:8443/",
  );
});

test("routers still resolve paths against opaque origin and srcdoc bases", () => {
  const { URL: PageURL } = setup();
  assert.equal(
    new PageURL("/docs/en/api/models", "null").href,
    "https://platform.claude.com/docs/en/api/models",
  );
  for (const base of [
    "about:srcdoc",
    "about:srcdoc?query=1",
    "about:srcdoc#section",
  ]) {
    assert.equal(
      new PageURL("next", base).href,
      "https://platform.claude.com/docs/en/api/messages/next",
    );
  }
});

test("null with an explicit valid base remains a relative path", () => {
  const { URL: PageURL } = setup();
  assert.equal(
    new PageURL("null", "https://example.com/docs/").href,
    "https://example.com/docs/null",
  );
  assert.throws(() => new PageURL("null", undefined), TypeError);
});

test("ordinary URLs and invalid inputs retain native behavior", () => {
  const { URL: PageURL } = setup();
  assert.equal(
    new PageURL("https://other.example/path?q=1#part").href,
    "https://other.example/path?q=1#part",
  );
  assert.equal(
    new PageURL("next", "https://example.com/docs/").href,
    "https://example.com/docs/next",
  );
  assert.throws(() => new PageURL("not a URL"), TypeError);
  assert.throws(() => new PageURL("/relative"), TypeError);
  assert.equal(PageURL.createObjectURL, URL.createObjectURL);
  class DocsURL extends PageURL {}
  assert.ok(new DocsURL("null") instanceof DocsURL);
});

test("pages without a remote URL keep the native constructor", () => {
  const window = setup(null);
  assert.equal(window.URL, URL);
  assert.throws(() => new window.URL(window.location.origin), TypeError);
});
