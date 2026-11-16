// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { subscribeResidentStatusRefresh } from "../src/features/hub/lib/resident-status-refresh.ts";
import { fakeTargets, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { subscribeGenerationRecoveryTriggers } = await import(
  "../src/features/chat/utils/chat-generation-recovery.ts"
);

function recoveryHarness(wakeKey = () => "thread-1" as string | undefined) {
  const windowTarget = new EventTarget();
  const documentTarget = Object.assign(new EventTarget(), {
    visibilityState: "hidden",
  });
  const clock = { now: 0 };
  let recoveries = 0;
  const unsubscribe = subscribeGenerationRecoveryTriggers(
    windowTarget,
    documentTarget,
    () => {
      recoveries += 1;
    },
    { wakeKey, now: () => clock.now },
  );
  const returnToTab = (focusFirst = false) => {
    documentTarget.visibilityState = "hidden";
    documentTarget.dispatchEvent(new Event("visibilitychange"));
    if (focusFirst) windowTarget.dispatchEvent(new Event("focus"));
    documentTarget.visibilityState = "visible";
    documentTarget.dispatchEvent(new Event("visibilitychange"));
    if (!focusFirst) windowTarget.dispatchEvent(new Event("focus"));
  };
  return {
    windowTarget,
    clock,
    returnToTab,
    unsubscribe,
    get recoveries() {
      return recoveries;
    },
  };
}

test("a tab return recovers the open chat thread once", () => {
  const harness = recoveryHarness();
  harness.returnToTab();
  assert.equal(harness.recoveries, 1, "visible + focus must cost one recovery");
  harness.unsubscribe();
});

test("a focus that lands while still hidden does not use up the wake", () => {
  const harness = recoveryHarness();
  harness.returnToTab(true);
  assert.equal(harness.recoveries, 1);
  harness.unsubscribe();
});

test("a wake after the dedupe window still recovers", () => {
  const harness = recoveryHarness();
  harness.returnToTab();
  harness.clock.now += 999;
  harness.windowTarget.dispatchEvent(new Event("focus"));
  assert.equal(harness.recoveries, 1);
  harness.clock.now += 1;
  harness.windowTarget.dispatchEvent(new Event("focus"));
  assert.equal(harness.recoveries, 2);
  harness.clock.now += 5_000;
  harness.returnToTab();
  assert.equal(harness.recoveries, 3);
  harness.unsubscribe();
});

test("online and pageshow always recover, even right after a wake", () => {
  const harness = recoveryHarness();
  harness.returnToTab();
  harness.windowTarget.dispatchEvent(new Event("online"));
  harness.windowTarget.dispatchEvent(new Event("pageshow"));
  harness.windowTarget.dispatchEvent(new Event("online"));
  assert.equal(harness.recoveries, 4);
  harness.unsubscribe();
});

test("the wake dedupe is per thread", () => {
  let thread = "thread-1";
  const harness = recoveryHarness(() => thread);
  harness.returnToTab();
  thread = "thread-2";
  harness.windowTarget.dispatchEvent(new Event("focus"));
  assert.equal(harness.recoveries, 2);
  harness.windowTarget.dispatchEvent(new Event("focus"));
  assert.equal(harness.recoveries, 2);
  harness.unsubscribe();
});

test("a tab return re-reads the resident model status once", () => {
  for (const order of [
    ["document", "window"],
    ["window", "document"],
  ] as const) {
    const targets = fakeTargets();
    let clock = 0;
    let reads = 0;
    subscribeResidentStatusRefresh(
      () => {
        reads += 1;
      },
      targets,
      () => clock,
    );
    const returnToTab = () => {
      for (const target of order) {
        targets.fire(
          target,
          target === "window" ? "focus" : "visibilitychange",
        );
      }
    };
    returnToTab();
    assert.equal(reads, 1, `${order.join(" then ")} must cost one read`);
    clock += 1_000;
    returnToTab();
    assert.equal(reads, 2, "the next tab return still reads");
  }
});

test("the loaded-models card wakes once per tab return", () => {
  const source = readFileSync(
    new URL(
      "../src/features/loaded-models/use-loaded-models.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const effect = source.slice(
    source.indexOf("const onWake = () => {"),
    source.indexOf('window.addEventListener("focus", onWake);'),
  );
  assert.match(
    effect,
    /if \(!document\.hidden && isFreshWake\(\)\) refresh\(\);/,
  );
  assert.match(source, /const isFreshWake = createWakeDedupe\(\);/);
});
