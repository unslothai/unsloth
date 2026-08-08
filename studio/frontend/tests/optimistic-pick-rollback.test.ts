// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A rejected model pick must roll back its OWN optimistic settings and nobody else's.
 *
 * Nothing on this path is gated on `busy`, so picking a second model while the first
 * download-plan request is still in flight runs both. The older request then came back with
 * its incompatibility, restored the steps and guidance it had captured over the newer,
 * accepted pick, and cleared that newer pick's pending revert on the way out -- leaving the
 * newly loaded model generating at another model's settings.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import test from "node:test";

const source = readFileSync(
  fileURLToPath(new URL("../src/features/images/images-page.tsx", import.meta.url)),
  "utf8",
);

test("every optimistic pick arms its rollback with a selection token", () => {
  const armed = source.match(/quantRevert\.current = \{ prev: prevQuant/g) ?? [];
  assert.ok(armed.length >= 4, "expected the optimistic pick branches to be found");
  const stamped = source.match(/const token = \+\+selectionToken\.current;/g) ?? [];
  assert.equal(stamped.length, armed.length);
  assert.ok(source.includes("prevGuidance, token }"), "the revert must carry its token");
});

test("no rollback runs without checking that its selection is still current", () => {
  // Every block that restores the captured settings has to be token-guarded; an unguarded one
  // is exactly the bug. (`pendingDeploy`'s own !started check is not a settings rollback.)
  const blocks =
    source.match(/if \(!started\) \{\s*\n\s*setQuant\(prevQuant\)/g) ?? [];
  assert.equal(blocks.length, 0, "a settings rollback runs without checking the selection");
  const guarded =
    source.match(
      /if \(!started && selectionToken\.current === token\) \{\s*\n\s*setQuant\(prevQuant\)/g,
    ) ?? [];
  assert.ok(guarded.length >= 4, `expected every rollback guarded, found ${guarded.length}`);
});
