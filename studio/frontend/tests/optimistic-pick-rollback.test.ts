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

function modelSelectBody(): string {
  const start = source.indexOf("const handleModelSelect = useCallback(");
  assert.ok(start > 0, "handleModelSelect not found");
  const end = source.indexOf("[busy, handleLoad, loadOrStage, quant, steps, guidance]", start);
  assert.ok(end > start, "handleModelSelect dependency list not found");
  return source.slice(start, end);
}

test("every accepted selection advances the token, curated picks included", () => {
  // The curated non-GGUF branch applied its defaults and returned without touching the
  // token or the pending revert, so a GGUF pick that failed afterwards still matched its
  // own token and restored ITS quant, steps and guidance over the curated selection --
  // the token mechanism bypassed entirely by the one branch that did not opt in.
  const body = modelSelectBody();
  const applied = body.match(/defaultsFor\(id\)/g) ?? [];
  const stamped = body.match(/const token = \+\+selectionToken\.current;/g) ?? [];
  assert.ok(applied.length >= 5, `expected every selection branch, found ${applied.length}`);
  assert.equal(
    stamped.length,
    applied.length,
    "a branch applies a model's defaults without advancing the selection token",
  );
  // And each of them replaces the pending revert rather than leaving an older one armed.
  const armed = body.match(/quantRevert\.current = \{ prev: prevQuant/g) ?? [];
  assert.equal(armed.length, applied.length);
});

test("an edit made while a pick is pending survives that pick's rollback", () => {
  // Staging an undownloaded model deliberately leaves `busy` unset for as long as the
  // download takes, so Steps and Guidance stay live the whole time. The pending revert holds
  // the values from BEFORE the pick, and every rollback path replayed them unconditionally,
  // so a load that then failed or was cancelled silently undid the user's own edit. The
  // selection token cannot cover this: it moves on another pick, not on a slider.
  assert.ok(
    /onChange=\{handleStepsChange\}/.test(source) &&
      /onChange=\{handleGuidanceChange\}/.test(source),
    "the sliders must go through handlers that retire the pending settings rollback",
  );
  assert.ok(
    /const handleStepsChange = useCallback\(\(value: number\) => \{\s*\n\s*settingsEdited\.current = true;/.test(
      source,
    ),
    "the steps handler must mark the settings as edited",
  );
  assert.ok(
    /const handleGuidanceChange = useCallback\(\(value: number\) => \{\s*\n\s*settingsEdited\.current = true;/.test(
      source,
    ),
    "the guidance handler must mark the settings as edited",
  );

  // Every settings restore is behind that flag, and no bare one is left.
  const guarded = source.match(/if \(!settingsEdited\.current\) \{/g) ?? [];
  const restores = source.match(/setSteps\((?:prevSteps|quantRevert\.current\.prevSteps)\)/g) ?? [];
  assert.ok(restores.length >= 7, `expected every rollback site, found ${restores.length}`);
  assert.equal(guarded.length, restores.length);

  // And arming a fresh pick clears the flag, else one edit would disarm every later pick.
  const armed = source.match(/quantRevert\.current = \{ prev: prevQuant/g) ?? [];
  const cleared = source.match(/settingsEdited\.current = false;/g) ?? [];
  assert.equal(cleared.length, armed.length);
});
