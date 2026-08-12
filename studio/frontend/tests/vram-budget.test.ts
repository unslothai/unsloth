// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();

const {
  VRAM_BUDGET_PERCENT_DEFAULT,
  VRAM_BUDGET_PERCENT_MAX,
  VRAM_BUDGET_PERCENT_MIN,
  vramFractionToPercent,
  vramPercentToFraction,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

test("percent and fraction round-trip exactly across the whole range", () => {
  for (
    let percent = VRAM_BUDGET_PERCENT_MIN;
    percent <= VRAM_BUDGET_PERCENT_MAX;
    percent += 1
  ) {
    assert.equal(
      vramFractionToPercent(vramPercentToFraction(percent)),
      percent,
    );
  }
});

test("the default fraction is exactly 0.97, not a float-drifted neighbour", () => {
  // A drifted 0.9700000000000001 never equals the backend default, so the UI would
  // show the budget as changed after a drag that ended where it began.
  assert.equal(vramPercentToFraction(VRAM_BUDGET_PERCENT_DEFAULT), 0.97);
  assert.equal(vramFractionToPercent(0.97), VRAM_BUDGET_PERCENT_DEFAULT);
});

test("the bounds mirror the backend range", () => {
  // vram_budget_settings.py: VRAM_FRACTION_MIN 0.80, MAX 1.00, DEFAULT 0.97.
  assert.equal(vramPercentToFraction(VRAM_BUDGET_PERCENT_MIN), 0.8);
  assert.equal(vramPercentToFraction(VRAM_BUDGET_PERCENT_MAX), 1);
  assert.equal(VRAM_BUDGET_PERCENT_DEFAULT, 97);
});

test("fractionToPercent rounds rather than truncating", () => {
  // A value set through UNSLOTH_VRAM_FRACTION need not be a whole percent.
  assert.equal(vramFractionToPercent(0.855), 86);
  assert.equal(vramFractionToPercent(0.854), 85);
});

test("percentToFraction tolerates a non-integer slider value", () => {
  assert.equal(vramPercentToFraction(90.4), 0.9);
  assert.equal(vramPercentToFraction(90.6), 0.91);
});

// The component cannot be mounted here (no DOM, and renderToStaticMarkup never
// runs effects), so the unmount contract is asserted against the source, as the
// chat-adapter tests do. The bug it guards: clearing the timer without sending the
// pending fraction discarded a drag followed within 400ms by Run, the Advanced
// toggle or closing the panel, and the server-wide budget lives nowhere else.
const pageSource = readFileSync(
  fileURLToPath(
    new URL(
      "../src/features/model-picker/components/model-config-page.tsx",
      import.meta.url,
    ),
  ),
  "utf8",
);

function vramBudgetRowSource(): string {
  const start = pageSource.indexOf("function VramBudgetRow()");
  assert.ok(start >= 0, "VramBudgetRow is no longer defined");
  const end = pageSource.indexOf("\n}\n", start);
  assert.ok(end > start, "could not delimit VramBudgetRow");
  return pageSource.slice(start, end);
}

test("unmount flushes the pending budget save instead of dropping it", () => {
  const row = vramBudgetRowSource();
  const cleanupStart = row.indexOf("useEffect(\n    () => () => {");
  assert.ok(cleanupStart >= 0, "the unmount-only effect is gone");
  const cleanup = row.slice(
    cleanupStart,
    row.indexOf("\n    [],\n  );", cleanupStart),
  );
  // Still cancels the timer, so no callback fires against a torn-down view...
  assert.match(cleanup, /clearTimeout\(saveTimer\.current\)/);
  // ...but the value the user set is sent rather than discarded.
  assert.match(cleanup, /flushVramBudgetSave\(\)/);
  // Fire-and-forget: the component is gone, so no response may reach its state.
  assert.doesNotMatch(cleanup, /\.then\(setSettings\)/);
});

test("commit stages the fraction before arming the debounce", () => {
  const row = vramBudgetRowSource();
  const commitStart = row.indexOf("const commit = (next: number) => {");
  assert.ok(commitStart >= 0, "commit is gone");
  const commit = row.slice(commitStart);
  // Staged before the timer, so an unfinished drag still has a value to flush.
  assert.ok(
    commit.indexOf("stageVramBudgetSave(vramPercentToFraction(next))") <
      commit.indexOf("setTimeout("),
    "the fraction must be staged before the debounce is armed",
  );
  // The debounced save uses the same flush, which clears the staged value as it
  // sends, so unmount cannot re-send a save that already happened.
  const timer = commit.slice(commit.indexOf("setTimeout("));
  assert.match(timer, /flushVramBudgetSave\(\)/);
  assert.doesNotMatch(timer, /updateVramBudgetSettings\(/);
});

test("the staged fraction is held outside the component that unmounts", () => {
  // The row unmounts on Run, so a ref inside it cannot be read by the load.
  const client = readFileSync(
    fileURLToPath(
      new URL("../src/features/settings/api/vram-budget.ts", import.meta.url),
    ),
    "utf8",
  );
  assert.match(client, /export function stageVramBudgetSave/);
  assert.match(client, /export function flushVramBudgetSave/);
  // Cleared as it sends, so two flushes cannot write the same edit twice.
  const flush = client.slice(
    client.indexOf("export function flushVramBudgetSave"),
  );
  assert.ok(
    flush.indexOf("stagedVramBudgetFraction = null") <
      flush.indexOf("updateVramBudgetSettings(fraction)"),
    "the flush must clear the staged value before sending it",
  );
  // Nothing staged must stay cheap for the caller: null, not a resolved promise.
  assert.match(flush, /fraction === null \? null :/);
});

test("Run waits for a staged budget save before starting the load", () => {
  // The control promises the budget applies on the next load, but if Run stages
  // the load while the PUT is open, that load uses the old fraction.
  const handlerStart = pageSource.indexOf("const handleRun = () => {");
  assert.ok(handlerStart >= 0, "handleRun is gone");
  const handler = pageSource.slice(
    handlerStart,
    pageSource.indexOf("\n  };", handlerStart),
  );
  const flushAt = handler.indexOf("settleVramBudgetSave()");
  assert.ok(flushAt >= 0, "handleRun no longer flushes the staged budget");
  assert.ok(
    flushAt < handler.indexOf("onRun(effectiveLoadConfig"),
    "the flush must come before the load is staged",
  );
  // A failed save must not swallow the load.
  assert.match(handler, /\.finally\(\(\) => \{/);
});

test("the row adopts published settings instead of only its own read", () => {
  const row = vramBudgetRowSource();
  assert.match(row, /subscribeVramBudgetSettings\(/);
  // A queued edit outranks the publish, or a mid-drag save moves the slider.
  const subscribeAt = row.indexOf("subscribeVramBudgetSettings(");
  const guard = row.slice(subscribeAt, subscribeAt + 260);
  assert.match(guard, /if \(saveTimer\.current\) \{\s*return;/);
});

test("Run reports a rejected budget flush instead of voiding it", () => {
  const handlerStart = pageSource.indexOf("const handleRun = () => {");
  const handler = pageSource.slice(
    handlerStart,
    pageSource.indexOf("\n  };", handlerStart),
  );
  const flush = handler.slice(handler.indexOf("settleVramBudgetSave()"));
  // finally alone re-rejects into an unhandled rejection, and the load proceeds
  // on the old fraction with nothing said.
  assert.ok(
    flush.indexOf(".catch(") < flush.indexOf(".finally("),
    "the rejection must be handled before finally starts the load",
  );
  assert.match(flush, /Failed to save VRAM budget/);
});

test("budget writes are serialised and only the newest publishes", () => {
  const client = readFileSync(
    fileURLToPath(
      new URL("../src/features/settings/api/vram-budget.ts", import.meta.url),
    ),
    "utf8",
  );
  // Two debounced saves can overlap on a slow link; out-of-order responses would
  // let the older edit win both the row and the stored value.
  assert.match(client, /vramBudgetWriteChain/);
  assert.match(client, /vramBudgetWriteGeneration/);
  const update = client.slice(
    client.indexOf("export function updateVramBudgetSettings"),
  );
  assert.match(
    update,
    /generation === vramBudgetWriteGeneration\s*\?\s*publishVramBudget/,
  );
  // A failed save must not strand every later one behind it.
  assert.match(update, /vramBudgetWriteChain = write\.catch/);
});

test("Run also waits for a save the debounce already sent", () => {
  const client = readFileSync(
    fileURLToPath(
      new URL("../src/features/settings/api/vram-budget.ts", import.meta.url),
    ),
    "utf8",
  );
  // Pause past the 400 ms debounce, then click Load: nothing is staged any more,
  // but the PUT is still open and the load would use the fraction it replaces.
  const settle = client.slice(client.indexOf("export function settleVramBudgetSave"));
  assert.match(settle, /flushVramBudgetSave\(\) \?\?/);
  assert.match(settle, /vramBudgetWritesOpen > 0 \? vramBudgetWriteChain : null/);
  // The counter has to come back down however the write ends.
  assert.match(client, /\.finally\(\(\) => \{\s*vramBudgetWritesOpen -= 1;/);
});

test("a read waits behind an open write", () => {
  const client = readFileSync(
    fileURLToPath(
      new URL("../src/features/settings/api/vram-budget.ts", import.meta.url),
    ),
    "utf8",
  );
  // A remount right after a flushed drag can read before the PUT commits and
  // answer after it, repainting the row with the value the server just replaced.
  const read = client.slice(
    client.indexOf("export async function loadVramBudgetSettings"),
  );
  assert.match(read, /vramBudgetWritesOpen > 0 \? vramBudgetWriteChain/);
  assert.ok(
    read.indexOf("pendingWrites") < read.indexOf(".then(fetchVramBudgetSettings)"),
    "the fetch must be chained behind the open writes, not raced with them",
  );
});
