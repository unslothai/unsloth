// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// An idle auto-unload frees the video runtime on the server. Nothing tells the browser:
// the eject event is raised by whoever clicked eject, and this page re-reads its status
// when the route becomes active, not while it stays active. So status.loaded is still
// true, Generate is still enabled, the POST 409s with "No video model is loaded." and the
// catch only toasted -- every retry repeated it until the user changed tabs.
//
// The refusal is the page's only news that the model is gone, so it has to act on it.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const VIDEO = readFileSync(
  fileURLToPath(new URL("../src/features/video/video-page.tsx", import.meta.url)),
  "utf8",
);

const RESYNC = VIDEO.slice(
  VIDEO.indexOf("const resyncAfterGenerateRefusal = useCallback("),
  VIDEO.indexOf("// Track mount so a long generate"),
);

test("a refused generation re-reads the model status", () => {
  const generateCatch = VIDEO.slice(
    VIDEO.indexOf('toast.error(err instanceof Error ? err.message : "Video generation failed")'),
  ).slice(0, 400);
  assert.match(generateCatch, /void resyncAfterGenerateRefusal\(\);/);
});

test("the re-read is the page's own ticketed refresh", () => {
  // Not a second reader: a read of its own would be one more writer racing the
  // activation read, which is the thing statusTicket exists to stop.
  assert.match(RESYNC, /const next = await refreshStatus\(\);/);
  assert.match(
    VIDEO,
    /return ticket === statusTicket\.current \? next : null;/,
    "refreshStatus answers with what it wrote, and null when superseded",
  );
});

test("a status that comes back unloaded clears the resident-only state", () => {
  // Same correction the indicator eject makes: Reapply must not point at a model that is
  // no longer resident, and the quant pick belongs to the load that is gone.
  assert.match(RESYNC, /dropResidentState\(\);\s*setQuant\(null\);/);
});

test("a failed or superseded re-read changes nothing", () => {
  // Status is best-effort everywhere else on this page; a network blip must not look
  // like an unload and tear the page's state down, and neither must a stale answer.
  assert.match(RESYNC, /if \(!isMounted\.current \|\| next === null \|\| next\.loaded\) return;/);
});

test("a load started while the re-read was in flight is left alone", () => {
  // /video/status reports committed state, so it answers loaded: false for a load that has
  // just started -- a true answer that means the opposite of what this continuation reads
  // into it. Acting on it dismisses the new load's toast and stops its progress poll, and
  // while its start request is still out the cancel it counts as sends the compensating
  // unload that tears the whole multi-gigabyte load down. handleLoad already bumps the
  // counter that says so, so the fence is to snapshot it across the await.
  assert.match(RESYNC, /const startLoad = loadSeq\.current;\s*const next = await refreshStatus\(\);/);
  assert.match(RESYNC, /if \(startLoad !== loadSeq\.current\) return;/);
  // And the fence has to come before the teardown, not after it.
  assert.ok(
    RESYNC.indexOf("if (startLoad !== loadSeq.current) return;") <
      RESYNC.indexOf("dropResidentState();"),
  );
});

test("the images page already corrects itself on every generate exit", () => {
  // It has the same dead-button shape, but its finally re-reads status on success,
  // failure and cancel alike, so there is nothing to fix there.
  const images = readFileSync(
    fileURLToPath(new URL("../src/features/images/images-page.tsx", import.meta.url)),
    "utf8",
  );
  const finallyBlock = images.slice(images.indexOf("      cancelRequested.current = false;"));
  assert.match(finallyBlock.slice(0, 1200), /if \(isMounted\.current\) await refreshStatus\(\);/);
});
