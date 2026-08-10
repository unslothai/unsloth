// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Three contracts around the H3 denoiser-partition dialog, asserted against the page source the
 * way video-download-plan-payload.test.ts does, since the page builds this flow inline.
 *
 * 1. The task choice must not skip the staged plan. Choosing the other partition can need the
 *    ~66 GB of shards the cache does not hold, and only the plan gives that a disk preflight,
 *    progress and a cancel. The plan is cache-aware, so a fully cached pick still costs nothing.
 * 2. An on-device copy of the pipeline has to reach the dialog too, or its transformer_ref
 *    partition is unreachable with the weights sitting on disk.
 * 3. Hiding the page has to take the dialog's cancellation path, not just close it.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import test from "node:test";

const source = readFileSync(
  fileURLToPath(new URL("../src/features/video/video-page.tsx", import.meta.url)),
  "utf8",
);

test("only a non-hub pick skips the plan, so a cached hub pick still gets one", () => {
  // The bypass is keyed on the pick's SOURCE. A curated artifact already on disk is still
  // source "hub" (localModelMeta is the only thing that emits "local"), so it keeps the plan:
  // "downloaded" is a property of the repo, and an H3 repo can be half downloaded.
  assert.match(source, /if \(source !== "hub"\) return handleLoadRef\.current\(repoId, opts\);/);
  assert.doesNotMatch(source, /if \(isDownloaded !== false\) return handleLoadRef\.current/);
  // And the deferred choice re-enters loadOrStage rather than loading directly, so the plan
  // decision is made again with the chosen partition in hand.
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
  // Not a Hub-id equality test any more: the local directory never matches one.
  assert.match(predicate, /split\("\/"\)\.at\(-1\)/);
  assert.match(predicate, /H3_BF16_REPO\.split\("\/"\)\[1\]\.toLowerCase\(\)/);
  // And the generic local-pipeline branch consults it, not only the curated branch.
  assert.equal(
    source.split('isH3PipelinePick(id, "pipeline")').length - 1,
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
  // Guarded on there actually being one, so hiding an idle page reverts nothing.
  assert.match(hide, /setPendingH3Load\(\(pending\) => \{/);
  assert.match(hide, /if \(pending\) abandonPick\(\);/);
});
