// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Reveal opens a file manager on the server's host. It is offered to a browser on that machine,
// however the URL spells loopback, is named as the server's platform names it, and is hidden where
// the server says no window can open (a container, a headless Linux box).

import assert from "node:assert/strict";
import test from "node:test";

import { isLoopbackHost, revealLabelFor } from "../src/features/library/reveal-label.ts";

test("every spelling of this machine counts as local", () => {
  for (const host of [
    "localhost",
    "LOCALHOST",
    "studio.localhost",
    "127.0.0.1",
    "127.1.2.3",
    "[::1]",
    "::1",
    "0.0.0.0",
    "[::]",
    "[::ffff:127.0.0.1]",
    "[::ffff:7f00:1]",
  ]) {
    assert.equal(isLoopbackHost(host), true, host);
  }
});

test("other hosts do not", () => {
  for (const host of ["192.168.1.20", "studio.example.com", "localhost.example.com", "128.0.0.1", "[::2]"]) {
    assert.equal(isLoopbackHost(host), false, host);
  }
});

test("the label key follows the server's file manager", () => {
  assert.equal(revealLabelFor("finder", "mac"), "library.reveal.finder");
  assert.equal(revealLabelFor("explorer", "windows"), "library.reveal.explorer");
  // WSL reports linux, but reveals in the Windows host's Explorer.
  assert.equal(revealLabelFor("explorer", "linux"), "library.reveal.explorer");
  assert.equal(revealLabelFor("files", "linux"), "library.reveal.files");
  assert.equal(revealLabelFor(null, "linux"), null);
});

test("an older server that reports no file manager is named by its platform", () => {
  assert.equal(revealLabelFor(undefined, "mac"), "library.reveal.finder");
  assert.equal(revealLabelFor(undefined, "windows"), "library.reveal.explorer");
  assert.equal(revealLabelFor(undefined, "linux"), "library.reveal.files");
});
