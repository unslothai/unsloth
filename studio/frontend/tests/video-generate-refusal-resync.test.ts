// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// An idle auto-unload is not pushed, so a Generate 409 is the page's only signal to resync.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const VIDEO = readSrc("features/video/video-page.tsx");

const RESYNC = VIDEO.slice(
  VIDEO.indexOf("const resyncAfterGenerateRefusal = useCallback("),
  VIDEO.indexOf("// Track mount so a long generate"),
);

test("a refused generation re-reads the model status", () => {
  const generateCatch = VIDEO.slice(
    VIDEO.indexOf(
      "const refusal = err instanceof Error ? err.message : null;",
    ),
  ).slice(0, 600);
  assert.match(generateCatch, /void resyncAfterGenerateRefusal\(\);/);
});

test("the re-read is the page's own ticketed refresh", () => {
  // Not a second reader: that would race the activation read statusTicket guards.
  assert.match(RESYNC, /const next = await refreshStatus\(\);/);
  assert.match(
    VIDEO,
    /return ticket === statusTicket\.current \? next : null;/,
    "refreshStatus answers with what it wrote, and null when superseded",
  );
});

test("a status that comes back unloaded clears the resident-only state", () => {
  assert.match(RESYNC, /dropResidentState\(\);\s*setQuant\(null\);/);
});

test("a failed or superseded re-read changes nothing", () => {
  // A network blip or stale answer must not look like an unload.
  assert.match(RESYNC, /if \(!isMounted\.current \|\| next === null \|\| next\.loaded\) return;/);
});

test("a load started while the re-read was in flight is left alone", () => {
  // /video/status says loaded: false mid-load: gate on loadSeq across await or it's torn down.
  assert.match(RESYNC, /const startLoad = loadSeq\.current;\s*const next = await refreshStatus\(\);/);
  assert.match(RESYNC, /if \(startLoad !== loadSeq\.current\) return;/);
  assert.ok(
    RESYNC.indexOf("if (startLoad !== loadSeq.current) return;") <
      RESYNC.indexOf("dropResidentState();"),
  );
});

test("the images page already corrects itself on every generate exit", () => {
  const images = readSrc("features/images/images-page.tsx");
  const finallyBlock = images.slice(images.indexOf("      cancelRequested.current = false;"));
  assert.match(finallyBlock.slice(0, 1200), /if \(isMounted\.current\) await refreshStatus\(\);/);
});
