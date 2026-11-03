// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Source-level, since the page builds this flow inline. The task choice must keep the staged
 * plan (disk preflight for ~66 GB), local copies must reach the dialog, and hiding the page
 * must take the cancellation path.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const source = readSrc("features/video/video-page.tsx");

test("only a non-hub pick skips the plan, so a cached hub pick still gets one", () => {
  // A curated artifact on disk is still source "hub", so it keeps the plan; H3 repos can be half downloaded.
  assert.match(source, /if \(source !== "hub"\) return handleLoadRef\.current\(repoId, opts, advanced\);/);
  assert.doesNotMatch(source, /if \(isDownloaded !== false\) return handleLoadRef\.current/);
  // The deferred choice re-enters loadOrStage so the plan is redone with the chosen partition.
  const choose = source.slice(
    source.indexOf("const chooseH3Task = useCallback("),
    source.indexOf("const cancelH3TaskChoice = useCallback("),
  );
  assert.match(choose, /loadOrStage\(/);
  assert.match(choose, /h3Task: task/);
  assert.doesNotMatch(choose, /handleLoadRef\.current\(/);
});

test("an on-device copy of the pipeline reaches the same dialog", () => {
  const predicate = source.slice(
    source.indexOf("function isH3PipelinePick("),
    source.indexOf("// What a pick optimistically replaced"),
  );
  assert.match(predicate, /split\("\/"\)\.at\(-1\)/);
  assert.match(predicate, /familyOverride\?\.trim\(\)\.toLowerCase\(\) === "minimax-h3"/);
  assert.match(predicate, /H3_BF16_REPO\.split\("\/"\)\[1\]\.toLowerCase\(\)/);
  assert.equal(
    source.split('isH3PipelinePick(id, "pipeline", nextFamilyOverride)').length - 1,
    1,
    "the local-pipeline branch must intercept an H3 pick exactly once",
  );
});

test("hiding the page cancels the pending pick rather than dropping it", () => {
  const hide = source.slice(
    source.indexOf("// A hidden page owns nothing"),
    source.indexOf("// A diffusion model picked from the chat picker"),
  );
  assert.match(hide, /abandonPick\(\)/);
  assert.match(hide, /setPendingH3Load\(\(pending\) => \{/);
  assert.match(hide, /if \(pending\) abandonPick\(\);/);
});
