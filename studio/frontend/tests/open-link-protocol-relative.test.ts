// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A protocol-relative link ("//host", "/\host") in model output names another site. Left to
 * native navigation it would replace the Desktop main window, so openLink must hand it to the
 * browser panel / system browser like any https link, while app routes stay native.
 */

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./bundler-resolver.mjs", import.meta.url);

const opened: string[] = [];
(globalThis as { window?: unknown }).window = {
  location: { hash: "" },
  open: (url: string) => opened.push(url),
};

const { openLink, setInAppLinkHandler } = await import(
  "../src/lib/open-link.ts"
);

test("protocol-relative links open as https instead of navigating the app window", () => {
  const panel: string[] = [];
  setInAppLinkHandler((url) => {
    panel.push(url);
    return true;
  });
  assert.equal(openLink("//evil.example/login"), true);
  assert.equal(openLink("/\\evil.example/x"), true);
  assert.deepEqual(panel, [
    "https://evil.example/login",
    "https://evil.example/x",
  ]);
  setInAppLinkHandler(null);
  assert.equal(openLink("\\\\evil.example/y"), true);
  assert.deepEqual(opened, ["https://evil.example/y"]);
});

test("app routes and relative links still navigate natively", () => {
  for (const url of ["/chat", "/settings/models", "docs/page", "./x", "?q=1"]) {
    assert.equal(openLink(url), false, url);
  }
});
