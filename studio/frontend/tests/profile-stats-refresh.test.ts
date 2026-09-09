// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { subscribeResidentStatusRefresh } from "../src/features/hub/lib/resident-status-refresh.ts";

const HOOK = readFileSync(
  new URL("../src/features/profile/hooks/use-profile-stats.ts", import.meta.url),
  "utf8",
);

type Listener = () => void;

function fakeTargets(hidden = false) {
  const windowListeners = new Map<string, Set<Listener>>();
  const documentListeners = new Map<string, Set<Listener>>();
  const state = { hidden };
  const add = (map: Map<string, Set<Listener>>) => (type: string, fn: Listener) => {
    if (!map.has(type)) map.set(type, new Set());
    map.get(type)?.add(fn);
  };
  const remove = (map: Map<string, Set<Listener>>) => (type: string, fn: Listener) => {
    map.get(type)?.delete(fn);
  };
  return {
    windowListeners,
    documentListeners,
    setHidden: (value: boolean) => {
      state.hidden = value;
    },
    fire: (map: Map<string, Set<Listener>>, type: string) => {
      for (const fn of [...(map.get(type) ?? [])]) fn();
    },
    targets: {
      window: {
        addEventListener: add(windowListeners) as never,
        removeEventListener: remove(windowListeners) as never,
      },
      document: {
        addEventListener: add(documentListeners) as never,
        removeEventListener: remove(documentListeners) as never,
        get hidden() {
          return state.hidden;
        },
      },
    },
  };
}

test("the profile stats hook refreshes when the tab returns to the foreground", () => {
  // Source-asserted: one line of wiring, but dropping it silently restores the stale-stats bug.
  assert.match(HOOK, /subscribeResidentStatusRefresh/);
  assert.match(
    HOOK,
    /useEffect\(\(\)\s*=>\s*subscribeResidentStatusRefresh\(load\), \[load\]\)/,
    "the subscription must be RETURNED from the effect, so React uses its " +
      "unsubscribe as the cleanup",
  );
});

test("the hook does not watch the settings dialog store", () => {
  // DialogContent is not force-mounted, so the panel remounts and refetches; a watcher is dead code.
  assert.doesNotMatch(HOOK, /useSettingsDialogStore/);
});

test("the refresh subscription hands back a working unsubscribe", () => {
  const harness = fakeTargets();
  let calls = 0;
  const stop = subscribeResidentStatusRefresh(() => {
    calls += 1;
  }, harness.targets);
  assert.equal(typeof stop, "function");

  harness.fire(harness.windowListeners, "focus");
  assert.equal(calls, 1);

  stop();
  harness.fire(harness.windowListeners, "focus");
  harness.fire(harness.documentListeners, "visibilitychange");
  assert.equal(calls, 1, "a listener survived unmount");
});

test("a tab going hidden does not refetch; coming back does", () => {
  const harness = fakeTargets(true);
  let calls = 0;
  const stop = subscribeResidentStatusRefresh(() => {
    calls += 1;
  }, harness.targets);

  harness.fire(harness.documentListeners, "visibilitychange");
  assert.equal(calls, 0, "refetched while the tab was hidden");

  harness.setHidden(false);
  harness.fire(harness.documentListeners, "visibilitychange");
  assert.equal(calls, 1);
  stop();
});

test("mounting and unmounting repeatedly leaves no listeners behind", () => {
  const harness = fakeTargets();
  let calls = 0;
  const stops = Array.from({ length: 20 }, () =>
    subscribeResidentStatusRefresh(() => {
      calls += 1;
    }, harness.targets),
  );
  for (const stop of stops) stop();

  harness.fire(harness.windowListeners, "focus");
  assert.equal(calls, 0);
});
