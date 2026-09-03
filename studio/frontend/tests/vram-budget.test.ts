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
  VRAM_BUDGET_PERCENT_STEP,
  vramFractionToPercent,
  vramPercentToFraction,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

test("percent and fraction round-trip exactly across the whole range", () => {
  // Every stop the slider can land on: a value that does not survive the trip
  // reads as "changed" and re-saves itself on every remount.
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
  // A value set through UNSLOTH_VRAM_FRACTION need not land on the grid.
  assert.equal(vramFractionToPercent(0.8555), 85.6);
  assert.equal(vramFractionToPercent(0.8554), 85.5);
});

test("percentToFraction tolerates an off-grid slider value", () => {
  assert.equal(vramPercentToFraction(90.44), 0.904);
  assert.equal(vramPercentToFraction(90.46), 0.905);
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
  const settle = client.slice(
    client.indexOf("export function settleVramBudgetSave"),
  );
  assert.match(settle, /flushVramBudgetSave\(\) \?\?/);
  assert.match(
    settle,
    /vramBudgetWritesOpen > 0 \? vramBudgetNewestWrite : null/,
  );
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
    read.indexOf("pendingWrites") <
      read.indexOf(".then(fetchVramBudgetSettings)"),
    "the fetch must be chained behind the open writes, not raced with them",
  );
});

test("the budget reads as a percentage and steps in tenths", () => {
  const row = vramBudgetRowSource();
  // Without the suffix, 97 sits between controls measured in layers and tokens
  // and reads as neither.
  assert.match(row, /displayValue=\{`\$\{percent\}%`\}/);
  assert.match(row, /step=\{VRAM_BUDGET_PERCENT_STEP\}/);
  // The shared slider defaults to whole steps, so the other callers are untouched.
  const slider = pageSource.slice(
    pageSource.indexOf("function AdvancedGpuSlider"),
  );
  assert.match(slider.slice(0, slider.indexOf("</div>")), /step = 1,/);
});

test("a failed save is re-staged, but never over a newer edit", () => {
  const client = readFileSync(
    fileURLToPath(
      new URL("../src/features/settings/api/vram-budget.ts", import.meta.url),
    ),
    "utf8",
  );
  // The flush clears the staged value as it sends, so without putting it back the
  // control shows a fraction the server never took. It goes back only while it is
  // still the newest intent.
  const update = client.slice(
    client.indexOf("export function updateVramBudgetSettings"),
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
  // Still rejects, so the caller can report it.
  assert.match(rejection, /throw error;/);
});

test("the reload notice is refreshed once a load finishes", () => {
  const row = vramBudgetRowSource().replace(/\s+/g, " ");
  // reloadRequired describes the running child. In the sidebar editor nothing
  // remounts this row on a reload, so without a refetch the notice kept asking for
  // a reload the user had just done.
  assert.match(
    row,
    /const modelLoading = useChatRuntimeStore\(\(s\) => s\.modelLoading\)/,
  );
  // Falling edge only: a read taken during the load answers about the child being
  // replaced, and would re-arm the very notice it is meant to clear.
  assert.match(
    row,
    /const finished = wasModelLoading\.current && !modelLoading; wasModelLoading\.current = modelLoading;/,
  );
  assert.match(row, /\}, \[isMac, modelLoading\]\)/);
});

test("Run waits out the budget save without starting two loads", () => {
  const run = pageSource.slice(pageSource.indexOf("const handleRun = () => {"));
  const body = run.slice(0, run.indexOf("\n  return (")).replace(/\s+/g, " ");
  // The click is answered by a PUT, so the button stays live for a round trip. A
  // second click would settle the same chain again and call onRun twice.
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
  // onRun tears the picker down, and that unmount flushes whatever is staged. A
  // re-staged retry would therefore PUT alongside the load request this click is
  // sending, and either fraction could size the child.
  assert.match(rejection, /dropVramBudgetRetry\(\); toast\.error\(/);
});

test("only the retry is dropped, never a newer edit staged over it", () => {
  const client = readFileSync(
    fileURLToPath(
      new URL("../src/features/settings/api/vram-budget.ts", import.meta.url),
    ),
    "utf8",
  );
  const flat = client.replace(/\s+/g, " ");
  // Run drops the failed fraction so it cannot race the load, but a drag landing
  // during that PUT stages a newer one, and dropping that would discard the edit
  // the user is looking at.
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
  const client = readFileSync(
    fileURLToPath(
      new URL("../src/features/settings/api/vram-budget.ts", import.meta.url),
    ),
    "utf8",
  );
  const flat = client.replace(/\s+/g, " ");
  // reloadRequired describes the running child, so an in-flight GET answers about
  // the child being replaced; sharing it republishes the stale notice.
  assert.match(flat, /if \(options\.force\) \{[^}]*inFlightVramBudget = null;/);
  // The displaced read must not then clear the newer handle on its way out.
  assert.match(
    flat,
    /if \(inFlightVramBudget === read\) \{ inFlightVramBudget = null;/,
  );
  const row = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/model-picker/components/model-config-page.tsx",
        import.meta.url,
      ),
    ),
    "utf8",
  );
  assert.match(row, /loadVramBudgetSettings\(\{ force: true \}\)/);
});

test("the budget closes while Run settles it, instead of racing the load", () => {
  const run = pageSource.slice(pageSource.indexOf("const handleRun = () => {"));
  const body = run.slice(0, run.indexOf("\n  return (")).replace(/\s+/g, " ");
  // Settling again in a loop only shrinks the window: an edit made during the last
  // attempt is still staged when onRun tears the picker down and flushes it
  // alongside the load request. Closing the control closes the window.
  assert.match(
    body,
    /setVramBudgetLocked\(true\); const stagedBudget = settleVramBudgetSave\(\);/,
  );
  assert.match(
    body,
    /setVramBudgetLocked\(false\); setBudgetSettling\(false\); onRun\(/,
  );
  // Nothing staged means nothing to wait for, so the control never closes.
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
  // Stored beats UNSLOTH_VRAM_FRACTION, so without this the first drag masks the
  // variable for good: dragging back to the same number stores that number.
  assert.match(row, /\{settings\.isStored && \( <button/);
  assert.match(row, /updateVramBudgetSettings\(null\)/);
  // A queued drag would otherwise store back what the reset just cleared.
  assert.match(
    row,
    /stageVramBudgetSave\(null\); updateVramBudgetSettings\(null\)/,
  );
});

test("Reload is reachable when only the server-wide budget changed", () => {
  // The budget is on no per-model field, so the page stays at its baseline and the
  // button stayed disabled while the row asked for a reload.
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
  const client = readFileSync(
    fileURLToPath(
      new URL("../src/features/settings/api/vram-budget.ts", import.meta.url),
    ),
    "utf8",
  );
  // Clearing the handle is not enough: the displaced GET still resolves, and it
  // describes the child being replaced, so publishing would restore the notice and
  // the Reload button that the forced read had just cleared.
  assert.match(
    client.replace(/\s+/g, " "),
    /if \( inFlightVramBudget !== read \|\|/,
  );
});

test("settling before a load hears about the write that failed", () => {
  const client = readFileSync(
    fileURLToPath(
      new URL("../src/features/settings/api/vram-budget.ts", import.meta.url),
    ),
    "utf8",
  );
  const flat = client.replace(/\s+/g, " ");
  // The chain swallows rejections so one failed save cannot strand the ones behind
  // it, so a Run waiting on the chain was told the save succeeded and never
  // dropped the retry, which the teardown then flushed against the load.
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
  const client = readFileSync(
    fileURLToPath(
      new URL("../src/features/settings/api/vram-budget.ts", import.meta.url),
    ),
    "utf8",
  );
  const flat = client.replace(/\s+/g, " ");
  // Waiting behind the writes open at read time says nothing about a save made
  // while the GET is in the air; that PUT can publish first and the late GET would
  // then restore the fraction the server no longer holds.
  assert.match(flat, /const generationAtRead = vramBudgetWriteGeneration;/);
  assert.match(
    flat,
    /generationAtRead !== vramBudgetWriteGeneration \) \{ throw new Error\("superseded"\)/,
  );
});

test("Manual with automatic layers still shows the budget", () => {
  // --fit-target carries the budget into that mode, and the launched fraction is
  // recorded for it, so hiding the row left a model loaded under an older budget
  // with neither the notice nor a way to reload.
  assert.match(
    pageSource.replace(/\s+/g, " "),
    /\{!isDiffusion && \(!isManual \|\| autoLayers\) && gpuDevices\.length > 0 && \( <VramBudgetRow \/> \)\}/,
  );
});

test("a superseded read is refused, not handed back to the caller", () => {
  const client = readFileSync(
    fileURLToPath(
      new URL("../src/features/settings/api/vram-budget.ts", import.meta.url),
    ),
    "utf8",
  );
  // Holding back the publish is not enough: the row applies the return value with
  // setSettings, so a read overtaken by a save would put back the isStored and
  // reloadRequired that save had just changed. Null is already this function's
  // "no usable answer", which every caller treats as keep what you have.
  const flat = client.replace(/\s+/g, " ");
  assert.match(flat, /throw new Error\("superseded"\); \}/);
  // The binding is optional: one caller inspects the error to tell an absent route from
  // a failed read, and the contract for everyone else is still null.
  assert.match(
    flat,
    /try \{ return await inFlightVramBudget; \} catch( \(error\))? \{/,
  );
  const row = vramBudgetRowSource().replace(/\s+/g, " ");
  assert.match(row, /if \(cancelled \|\| !loaded\) \{ return; \}/);
});
