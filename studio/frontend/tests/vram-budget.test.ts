// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  readSrc,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();

const {
  VRAM_BUDGET_PERCENT_DEFAULT,
  VRAM_BUDGET_PERCENT_MAX,
  VRAM_BUDGET_PERCENT_MIN,
  VRAM_BUDGET_PERCENT_STEP,
  vramFractionToPercent,
  vramPercentToFraction,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

const VRAM_BUDGET = readSrc("features/settings/api/vram-budget.ts");

test("percent and fraction round-trip exactly across the whole range", () => {
  // A value that does not survive the round trip reads as changed and re-saves on every remount.
  const steps = Math.round(
    (VRAM_BUDGET_PERCENT_MAX - VRAM_BUDGET_PERCENT_MIN) /
      VRAM_BUDGET_PERCENT_STEP,
  );
  for (let i = 0; i <= steps; i += 1) {
    const percent =
      Math.round(
        (VRAM_BUDGET_PERCENT_MIN + i * VRAM_BUDGET_PERCENT_STEP) * 10,
      ) / 10;
    assert.equal(
      vramFractionToPercent(vramPercentToFraction(percent)),
      percent,
    );
  }
});

test("a tenth of a percent survives the trip to the backend and back", () => {
  assert.equal(vramPercentToFraction(97.5), 0.975);
  assert.equal(vramFractionToPercent(0.975), 97.5);
  assert.equal(VRAM_BUDGET_PERCENT_STEP, 0.1);
});

test("the default fraction is exactly 0.97, not a float-drifted neighbour", () => {
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
  assert.equal(vramFractionToPercent(0.8555), 85.6);
  assert.equal(vramFractionToPercent(0.8554), 85.5);
});

test("percentToFraction tolerates an off-grid slider value", () => {
  assert.equal(vramPercentToFraction(90.44), 0.904);
  assert.equal(vramPercentToFraction(90.46), 0.905);
});

// Source-level (no DOM). Unmount must flush the pending fraction, not just clear the timer:
// the server-wide budget lives nowhere else.
const pageSource = readSrc(
  "features/model-picker/components/model-config-page.tsx",
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
  assert.match(cleanup, /clearTimeout\(saveTimer\.current\)/);
  assert.match(cleanup, /flushVramBudgetSave\(\)/);
  // Fire-and-forget: the component is gone, so no response may reach its state.
  assert.doesNotMatch(cleanup, /\.then\(setSettings\)/);
});

test("commit stages the fraction before arming the debounce", () => {
  const row = vramBudgetRowSource();
  const commitStart = row.indexOf("const commit = (next: number) => {");
  assert.ok(commitStart >= 0, "commit is gone");
  const commit = row.slice(commitStart);
  assert.ok(
    commit.indexOf("stageVramBudgetSave(vramPercentToFraction(next))") <
      commit.indexOf("setTimeout("),
    "the fraction must be staged before the debounce is armed",
  );
  // The flush clears the staged value as it sends, so unmount cannot re-send it.
  const timer = commit.slice(commit.indexOf("setTimeout("));
  assert.match(timer, /flushVramBudgetSave\(\)/);
  assert.doesNotMatch(timer, /updateVramBudgetSettings\(/);
});

test("the staged fraction is held outside the component that unmounts", () => {
  // The row unmounts on Run, so a ref inside it cannot be read by the load.
  assert.match(VRAM_BUDGET, /export function stageVramBudgetSave/);
  assert.match(VRAM_BUDGET, /export function flushVramBudgetSave/);
  const flush = VRAM_BUDGET.slice(
    VRAM_BUDGET.indexOf("export function flushVramBudgetSave"),
  );
  assert.ok(
    flush.indexOf("stagedVramBudgetFraction = null") <
      flush.indexOf("updateVramBudgetSettings(fraction)"),
    "the flush must clear the staged value before sending it",
  );
  assert.match(flush, /fraction === null \? null :/);
});

test("Run waits for a staged budget save before starting the load", () => {
  // If Run stages the load while the PUT is open, the load uses the old fraction.
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
  // finally alone re-rejects into an unhandled rejection.
  assert.ok(
    flush.indexOf(".catch(") < flush.indexOf(".finally("),
    "the rejection must be handled before finally starts the load",
  );
  assert.match(flush, /Failed to save VRAM budget/);
});

test("budget writes are serialised and only the newest publishes", () => {
  // Overlapping saves could resolve out of order and let the older edit win.
  assert.match(VRAM_BUDGET, /vramBudgetWriteChain/);
  assert.match(VRAM_BUDGET, /vramBudgetWriteGeneration/);
  const update = VRAM_BUDGET.slice(
    VRAM_BUDGET.indexOf("export function updateVramBudgetSettings"),
  );
  assert.match(
    update,
    /generation === vramBudgetWriteGeneration\s*\?\s*publishVramBudget/,
  );
  assert.match(update, /vramBudgetWriteChain = write\.catch/);
});

test("Run also waits for a save the debounce already sent", () => {
  // After the debounce nothing is staged but the PUT may still be open.
  const settle = VRAM_BUDGET.slice(
    VRAM_BUDGET.indexOf("export function settleVramBudgetSave"),
  );
  assert.match(settle, /flushVramBudgetSave\(\) \?\?/);
  assert.match(
    settle,
    /vramBudgetWritesOpen > 0 \? vramBudgetNewestWrite : null/,
  );
  assert.match(
    VRAM_BUDGET,
    /\.finally\(\(\) => \{\s*vramBudgetWritesOpen -= 1;/,
  );
});

test("a read waits behind an open write", () => {
  // A remount read can start before the PUT commits and answer after it.
  const read = VRAM_BUDGET.slice(
    VRAM_BUDGET.indexOf("export async function loadVramBudgetSettings"),
  );
  assert.match(read, /vramBudgetWritesOpen > 0 \? vramBudgetWriteChain/);
  assert.ok(
    read.indexOf("pendingWrites") <
      read.indexOf(".then(fetchVramBudgetSettings)"),
    "the fetch must be chained behind the open writes, not raced with them",
  );
});

test("the budget reads as a percentage and steps in tenths", () => {
  const row = vramBudgetRowSource();
  assert.match(row, /displayValue=\{`\$\{percent\}%`\}/);
  assert.match(row, /step=\{VRAM_BUDGET_PERCENT_STEP\}/);
  const slider = pageSource.slice(
    pageSource.indexOf("function AdvancedGpuSlider"),
  );
  assert.match(slider.slice(0, slider.indexOf("</div>")), /step = 1,/);
});

test("a failed save is re-staged, but never over a newer edit", () => {
  // The flush cleared the staged value, so a failure restores it while it is still the newest intent.
  const update = VRAM_BUDGET.slice(
    VRAM_BUDGET.indexOf("export function updateVramBudgetSettings"),
  );
  // Whitespace-collapsed: the formatter wraps this condition across lines.
  const rejection = update
    .slice(update.indexOf("(error: unknown) =>"))
    .replace(/\s+/g, " ");
  assert.match(
    rejection,
    /generation === vramBudgetWriteGeneration && stagedVramBudgetFraction === null/,
  );
  assert.match(rejection, /stageVramBudgetSave\(fraction\);/);
  assert.match(rejection, /throw error;/);
});

test("the reload notice is refreshed once a load finishes", () => {
  const row = vramBudgetRowSource().replace(/\s+/g, " ");
  // Nothing remounts this row on a reload in the sidebar editor, so it must refetch.
  assert.match(
    row,
    /const modelLoading = useChatRuntimeStore\(\(s\) => s\.modelLoading\)/,
  );
  // Falling edge only: a read during the load describes the child being replaced.
  assert.match(
    row,
    /const finished = wasModelLoading\.current && !modelLoading; wasModelLoading\.current = modelLoading;/,
  );
  assert.match(row, /\}, \[isMac, modelLoading\]\)/);
});

test("Run waits out the budget save without starting two loads", () => {
  const run = pageSource.slice(pageSource.indexOf("const handleRun = () => {"));
  const body = run.slice(0, run.indexOf("\n  return (")).replace(/\s+/g, " ");
  // A second click during the PUT would settle the chain again and call onRun twice.
  assert.match(body, /if \(budgetSettling\) \{ return; \}/);
  assert.match(body, /setBudgetSettling\(true\);/);
  assert.match(body, /setBudgetSettling\(false\); onRun\(/);
  const disabled = pageSource
    .slice(pageSource.indexOf("onClick={handleRun}") - 400)
    .replace(/\s+/g, " ");
  assert.match(disabled, /budgetSettling \|\|/);
});

test("a save that fails during Run is dropped, not left to race the load", () => {
  const run = pageSource.slice(pageSource.indexOf("const handleRun = () => {"));
  const rejection = run
    .slice(run.indexOf("void stagedBudget"))
    .replace(/\s+/g, " ");
  // onRun's teardown flushes staged values, so a re-staged retry would race the load request.
  assert.match(rejection, /dropVramBudgetRetry\(\); toast\.error\(/);
});

test("only the retry is dropped, never a newer edit staged over it", () => {
  const flat = VRAM_BUDGET.replace(/\s+/g, " ");
  // A drag landing during that PUT stages a newer value that must not be dropped.
  assert.match(flat, /stagedVramBudgetSequence \+= 1;/);
  assert.match(
    flat,
    /export function dropVramBudgetRetry\(\) \{ if \(stagedVramBudgetSequence === retryVramBudgetSequence\)/,
  );
  assert.match(
    flat,
    /stageVramBudgetSave\(fraction\); retryVramBudgetSequence = stagedVramBudgetSequence;/,
  );
});

test("a post-load read is not answered by one taken before the load finished", () => {
  const flat = VRAM_BUDGET.replace(/\s+/g, " ");
  // An in-flight GET describes the child being replaced, so a forced read must not share it.
  assert.match(flat, /if \(options\.force\) \{[^}]*inFlightVramBudget = null;/);
  assert.match(
    flat,
    /if \(inFlightVramBudget === read\) \{ inFlightVramBudget = null;/,
  );
  const row = readSrc("features/model-picker/components/model-config-page.tsx");
  assert.match(row, /loadVramBudgetSettings\(\{ force: true \}\)/);
});

test("the budget closes while Run settles it, instead of racing the load", () => {
  const run = pageSource.slice(pageSource.indexOf("const handleRun = () => {"));
  const body = run.slice(0, run.indexOf("\n  return (")).replace(/\s+/g, " ");
  // Looping the settle only shrinks the race window; locking the control closes it.
  assert.match(
    body,
    /setVramBudgetLocked\(true\); const stagedBudget = settleVramBudgetSave\(\);/,
  );
  assert.match(
    body,
    /setVramBudgetLocked\(false\); setBudgetSettling\(false\); onRun\(/,
  );
  assert.match(body, /setVramBudgetLocked\(false\); onRun\(/);
  const row = vramBudgetRowSource().replace(/\s+/g, " ");
  assert.match(
    row,
    /useEffect\(\(\) => subscribeVramBudgetLock\(setLocked\), \[\]\)/,
  );
  assert.match(row, /disabled=\{locked\}/);
});

test("a stored budget can be cleared back to the inherited one", () => {
  const row = vramBudgetRowSource().replace(/\s+/g, " ");
  // Stored beats UNSLOTH_VRAM_FRACTION, so a reset is the only way back to the env value.
  assert.match(row, /\{settings\.isStored && \( <button/);
  assert.match(row, /updateVramBudgetSettings\(null\)/);
  // A queued drag would otherwise store back what the reset just cleared.
  assert.match(
    row,
    /stageVramBudgetSave\(null\); updateVramBudgetSettings\(null\)/,
  );
});

test("Reload is reachable when only the server-wide budget changed", () => {
  // The budget is on no per-model field, so the page baseline never changes.
  assert.match(
    pageSource.replace(/\s+/g, " "),
    /isActiveModel && atBaseline && !rememberChanged && !budgetReloadRequired/,
  );
  assert.match(
    pageSource.replace(/\s+/g, " "),
    /subscribeVramBudgetSettings\(\(next\) => \{ setBudgetReloadRequired\(next\.reloadRequired\); \}\)/,
  );
});

test("a read displaced by a forced one does not publish", () => {
  // The displaced GET still resolves for the old child, so it must not publish.
  assert.match(
    VRAM_BUDGET.replace(/\s+/g, " "),
    /if \( inFlightVramBudget !== read \|\|/,
  );
});

test("settling before a load hears about the write that failed", () => {
  const flat = VRAM_BUDGET.replace(/\s+/g, " ");
  // The chain swallows rejections, so Run must observe the save's own result.
  assert.match(
    flat,
    /vramBudgetWritesOpen > 0 \? vramBudgetNewestWrite : null/,
  );
  assert.match(
    flat,
    /vramBudgetWriteChain = write\.catch\(\(\) => undefined\); vramBudgetNewestWrite = write;/,
  );
});

test("a read does not repaint over a write issued while it was in the air", () => {
  const flat = VRAM_BUDGET.replace(/\s+/g, " ");
  // A PUT made while the GET is in the air can publish first; the generation detects it.
  assert.match(flat, /const generationAtRead = vramBudgetWriteGeneration;/);
  assert.match(
    flat,
    /generationAtRead !== vramBudgetWriteGeneration \) \{ throw new Error\("superseded"\)/,
  );
});

test("Manual with automatic layers still shows the budget", () => {
  // --fit-target carries the budget into that mode, so the row must stay visible there.
  assert.match(
    pageSource.replace(/\s+/g, " "),
    /\{!isDiffusion && \(!isManual \|\| autoLayers\) && gpuDevices\.length > 0 && \( <VramBudgetRow \/> \)\}/,
  );
});

test("a superseded read is refused, not handed back to the caller", () => {
  // The row applies the return value, so an overtaken read must return null, not stale settings.
  const flat = VRAM_BUDGET.replace(/\s+/g, " ");
  assert.match(flat, /throw new Error\("superseded"\); \}/);
  // One caller inspects the error to tell an absent route from a failed read.
  assert.match(
    flat,
    /try \{ return await inFlightVramBudget; \} catch( \(error\))? \{/,
  );
  const row = vramBudgetRowSource().replace(/\s+/g, " ");
  assert.match(row, /if \(cancelled \|\| !loaded\) \{ return; \}/);
});
