// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { mock } from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const { store } = installLocalStorageFake();
const DAY = 24 * 60 * 60 * 1000;

type Effect = {
  deps?: unknown[];
  cleanup?: () => void;
  pending?: () => undefined | (() => void);
};

/** A one-component React: renders until no state or effect changes anything. */
function fakeReact() {
  const state: unknown[] = [];
  const effects: Effect[] = [];
  let index = 0;
  let effectIndex = 0;
  let dirty = false;
  const react = {
    useState<T>(initial: T) {
      const i = index++;
      if (!(i in state)) state[i] = initial;
      const set = (next: T | ((prev: T) => T)) => {
        const value =
          typeof next === "function"
            ? (next as (prev: T) => T)(state[i] as T)
            : next;
        if (!Object.is(value, state[i])) {
          state[i] = value;
          dirty = true;
        }
      };
      return [state[i] as T, set] as const;
    },
    useEffect(effect: () => undefined | (() => void), deps: unknown[]) {
      const slot = (effects[effectIndex++] ??= {});
      if (!slot.deps || deps.some((d, k) => !Object.is(d, slot.deps?.[k]))) {
        slot.deps = deps;
        slot.pending = effect;
      }
    },
    useSyncExternalStore: (_subscribe: unknown, get: () => unknown) => get(),
  };
  const render = <T>(hook: () => T): T => {
    let result!: T;
    for (let pass = 0; pass < 20; pass++) {
      dirty = false;
      index = 0;
      effectIndex = 0;
      result = hook();
      for (const slot of effects) {
        if (!slot.pending) continue;
        slot.cleanup?.();
        slot.cleanup = slot.pending() ?? undefined;
        slot.pending = undefined;
      }
      if (!dirty) return result;
    }
    throw new Error("render did not settle");
  };
  return { react, render, isDirty: () => dirty };
}

function load() {
  const fake = fakeReact();
  const mod = loadWithStubs<
    typeof import("../src/hooks/use-notification-frequency.ts")
  >(new URL("../src/hooks/use-notification-frequency.ts", import.meta.url), {
    react: fake.react,
  });
  return { ...fake, mod };
}

test("a monthly wait longer than setTimeout allows re-arms until it is due", () => {
  store.clear();
  const start = Date.UTC(2026, 0, 1);
  mock.timers.enable({ apis: ["setTimeout", "Date"], now: start });
  try {
    const { mod, render, isDirty } = load();
    mod.setNotificationFrequency("unsloth", "monthly");
    mod.markNotificationShown("unsloth", start);
    const due = () => mod.useNotificationDue("unsloth");
    assert.equal(render(due), false);
    // The first timer is capped near 24.9 days; it must hand over to a second one.
    mock.timers.tick(25 * DAY);
    assert.equal(isDirty(), true, "the capped wake-up re-renders");
    assert.equal(render(due), false, "still inside the 30 days");
    mock.timers.tick(5 * DAY + 2000);
    assert.equal(isDirty(), true, "the second timer fires at day 30");
    assert.equal(render(due), true);
  } finally {
    mock.timers.reset();
  }
});

test("turning a channel off releases an open notification's hold", () => {
  store.clear();
  mock.timers.enable({
    apis: ["setTimeout", "Date"],
    now: Date.UTC(2026, 0, 1),
  });
  try {
    const { mod, render } = load();
    mod.setNotificationFrequency("unsloth", "daily");
    const gate = () => mod.useNotificationGate("unsloth", true);
    assert.equal(render(gate), true, "first showing");
    mod.setNotificationFrequency("unsloth", "off");
    assert.equal(render(gate), false);
    // Back on a quiet period right after it was shown: it waits out the day.
    mod.setNotificationFrequency("unsloth", "daily");
    assert.equal(render(gate), false);
  } finally {
    mock.timers.reset();
  }
});

test("a hidden tab does not use up the quiet period", () => {
  store.clear();
  const doc = {
    visibilityState: "hidden",
    addEventListener() {},
    removeEventListener() {},
  };
  Object.assign(globalThis, { document: doc });
  mock.timers.enable({
    apis: ["setTimeout", "Date"],
    now: Date.UTC(2026, 0, 1),
  });
  try {
    const { mod, render } = load();
    mod.setNotificationFrequency("unsloth", "weekly");
    const gate = () => mod.useNotificationGate("unsloth", true);
    render(gate);
    assert.equal(store.has("unsloth_unsloth_notification_last_shown"), false);
    doc.visibilityState = "visible";
    assert.equal(render(gate), true);
    assert.equal(store.has("unsloth_unsloth_notification_last_shown"), true);
  } finally {
    mock.timers.reset();
    Reflect.deleteProperty(globalThis, "document");
  }
});
