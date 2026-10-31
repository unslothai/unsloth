// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A packaged build applies the CSP from src-tauri/tauri.conf.json, and its connect-src allows no
// data: or blob: URL. The video opener used to build a data: URL and fetch() it, which dev builds
// got away with because `tauri dev` runs without a policy at all: clicking a sent video showed
// "Could not open <name>.mp4" in an installed app only. The opener now decodes the data URI in
// place, so this guards the decode and the CSP premise that makes it necessary.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const SRC = new URL("../src/", import.meta.url);
const read = (relative: string) => readFileSync(new URL(relative, SRC), "utf8");

test("the video opener decodes its data URI instead of fetching it", () => {
  const source = read("components/assistant-ui/attachment-browser-open.tsx");

  assert.match(source, /import \{ decodeDataUri, isDataUri \} from "@\/lib\/data-uri";/);
  assert.match(source, /if \(isDataUri\(url\)\) \{/);
  assert.match(source, /decodeDataUri\(url\)/);
});

test("the packaged app's connect-src still forbids data and blob URLs", () => {
  const config = read("../../src-tauri/tauri.conf.json");
  const connect = /connect-src([^;"]+)/.exec(config)?.[1] ?? "";

  assert.ok(connect.length > 0, "connect-src not found in tauri.conf.json");
  assert.doesNotMatch(
    connect,
    /data:/,
    "connect-src now allows data: URLs; the video opener no longer needs to decode in place",
  );
  assert.doesNotMatch(
    connect,
    /blob:/,
    "connect-src now allows blob: URLs; the video opener no longer needs to decode in place",
  );
});
