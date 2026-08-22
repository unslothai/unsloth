// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test, { after } from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

const source = await readFile(
  new URL("../src/hooks/use-tauri-update.ts", import.meta.url),
  "utf8",
);

function constant(name: string): number {
  const match = new RegExp(`const ${name} = ([^;]+);`).exec(source);
  assert.ok(match, `${name} is no longer declared in use-tauri-update.ts`);
  return Function(`"use strict"; return (${match[1]});`)() as number;
}

const effectStart = source.indexOf(
  "    if (!isTauri) {",
  source.indexOf("const scheduledCheckRef"),
);
const effectEndMarker = "      replaceUpdate(null);\n    };";
const effectEnd = source.indexOf(effectEndMarker, effectStart);
assert.ok(
  effectStart > 0 && effectEnd > effectStart,
  "the desktop update schedule moved",
);
const effectBody = source.slice(
  effectStart,
  effectEnd + effectEndMarker.length,
);

const startSchedule = new Function(
  "isTauri",
  "lifecycleRef",
  "checkedRef",
  "scheduledCheckRef",
  "replaceUpdate",
  "setTimeout",
  "clearTimeout",
  "setInterval",
  "clearInterval",
  "STARTUP_UPDATE_CHECK_DELAY_MS",
  "PERIODIC_UPDATE_CHECK_INTERVAL_MS",
  effectBody,
) as (
  isTauri: boolean,
  lifecycleRef: { current: number },
  checkedRef: { current: boolean },
  scheduledCheckRef: { current: () => void },
  replaceUpdate: (update: null) => void,
  setTimeout: (fn: () => void, delay: number) => number,
  clearTimeout: (id: number) => void,
  setInterval: (fn: () => void, delay: number) => number,
  clearInterval: (id: number) => void,
  startupDelay: number,
  periodicInterval: number,
) => (() => void) | undefined;

function harness({ checked = false, tauri = true } = {}) {
  const timeouts = new Map<number, () => void>();
  const intervals = new Map<number, () => void>();
  const delays: number[] = [];
  let nextId = 1;
  let checks = 0;
  const cleanup = startSchedule(
    tauri,
    { current: 0 },
    { current: checked },
    {
      current: () => {
        checks += 1;
      },
    },
    () => {},
    (fn, delay) => {
      const id = nextId++;
      timeouts.set(id, fn);
      delays.push(delay);
      return id;
    },
    (id) => {
      timeouts.delete(id);
    },
    (fn, delay) => {
      const id = nextId++;
      intervals.set(id, fn);
      delays.push(delay);
      return id;
    },
    (id) => {
      intervals.delete(id);
    },
    constant("STARTUP_UPDATE_CHECK_DELAY_MS"),
    constant("PERIODIC_UPDATE_CHECK_INTERVAL_MS"),
  );
  return {
    cleanup,
    delays,
    checks: () => checks,
    fireStartup: () => {
      for (const fn of timeouts.values()) {
        fn();
      }
    },
    firePeriodic: () => {
      for (const fn of intervals.values()) {
        fn();
      }
    },
    activeTimers: () => timeouts.size + intervals.size,
  };
}

test("desktop update checks continue after the startup check", () => {
  const schedule = harness();
  assert.deepEqual(schedule.delays, [5000, 60 * 60 * 1000]);
  schedule.fireStartup();
  schedule.firePeriodic();
  schedule.firePeriodic();
  assert.equal(schedule.checks(), 3);
});

test("a manual check suppresses only the delayed startup check", () => {
  const schedule = harness({ checked: true });
  schedule.fireStartup();
  schedule.firePeriodic();
  assert.equal(schedule.checks(), 1);
});

test("desktop update timers stop when the update layer unmounts", () => {
  const schedule = harness();
  assert.equal(schedule.activeTimers(), 2);
  schedule.cleanup?.();
  assert.equal(schedule.activeTimers(), 0);
  schedule.fireStartup();
  schedule.firePeriodic();
  assert.equal(schedule.checks(), 0);
});

test("web sessions do not schedule desktop update checks", () => {
  const schedule = harness({ tauri: false });
  assert.equal(schedule.activeTimers(), 0);
  assert.equal(schedule.cleanup, undefined);
});

type FakeUpdate = {
  version: string;
  currentVersion: string;
  rawJson: Record<string, unknown> | undefined;
  body?: string;
  date?: string;
  close: () => Promise<void>;
};

type UpdateController = {
  checkForUpdate: () => Promise<void>;
};

function fakeUpdate(version: string, valid = true) {
  let closeCalls = 0;
  const update: FakeUpdate = {
    version,
    currentVersion: "1.0.0",
    rawJson: valid ? {} : undefined,
    close: async () => {
      closeCalls += 1;
    },
  };
  return { update, closeCalls: () => closeCalls };
}

function createHookReact() {
  const effects: Array<() => unknown> = [];
  const cleanups: Array<() => void> = [];
  return {
    react: {
      useState<T>(initial: T): [T, (next: unknown) => void] {
        return [initial, () => {}];
      },
      useRef<T>(initial: T): { current: T } {
        return { current: initial };
      },
      useEffect(effect: () => unknown): void {
        effects.push(effect);
      },
    },
    mount(): void {
      for (const effect of effects) {
        const cleanup = effect();
        if (typeof cleanup === "function") {
          cleanups.push(cleanup as () => void);
        }
      }
    },
    unmount(): void {
      for (const cleanup of cleanups) {
        cleanup();
      }
      cleanups.length = 0;
    },
  };
}

const originalTimers = {
  setTimeout: globalThis.setTimeout,
  clearTimeout: globalThis.clearTimeout,
  setInterval: globalThis.setInterval,
  clearInterval: globalThis.clearInterval,
};
let timerId = 1;
Object.assign(globalThis, {
  setTimeout: (() => timerId++) as unknown as typeof setTimeout,
  clearTimeout: (() => {}) as typeof clearTimeout,
  setInterval: (() => timerId++) as unknown as typeof setInterval,
  clearInterval: (() => {}) as typeof clearInterval,
});
after(() => Object.assign(globalThis, originalTimers));

function hookHarness() {
  const host = createHookReact();
  const checks: Array<() => Promise<FakeUpdate | null>> = [];
  let policyMode: "in_app" | "manual_linux_package" = "in_app";
  let manualUpdate: Record<string, unknown> | null = null;
  const hook = loadWithStubs<{
    useTauriUpdate: () => UpdateController;
  }>(new URL("../src/hooks/use-tauri-update.ts", import.meta.url), {
    react: host.react,
    "@/lib/api-base": { isTauri: true },
    "@/lib/tauri-diagnostics": {
      copySupportDiagnostics: async () => ({ copied: true }),
    },
    "@/lib/tauri-updater": {
      checkDesktopUpdate: () => {
        const check = checks.shift();
        assert.ok(check, "an unexpected desktop update check ran");
        return check();
      },
    },
    "@/lib/toast": { toast: { error: () => {} } },
    "@tauri-apps/api/core": {
      invoke: async (command: string) => {
        if (command === "desktop_update_policy") {
          return {
            mode: policyMode,
            releasePageBaseUrl: "https://example.com/",
            releaseTagPrefix: "v",
          };
        }
        if (command === "check_desktop_manual_update") {
          return manualUpdate;
        }
        throw new Error(`unexpected invoke: ${command}`);
      },
    },
  });
  const controller = hook.useTauriUpdate();
  host.mount();
  return {
    controller,
    host,
    enqueue: (check: () => Promise<FakeUpdate | null>) => checks.push(check),
    useManualUpdate: (update: Record<string, unknown> | null) => {
      policyMode = "manual_linux_package";
      manualUpdate = update;
    },
  };
}

test("periodic update resources close on replacement and no-update", async () => {
  const hook = hookHarness();
  const first = fakeUpdate("2.0.0");
  const second = fakeUpdate("2.0.1");
  hook.enqueue(async () => first.update);
  await hook.controller.checkForUpdate();
  hook.enqueue(async () => second.update);
  await hook.controller.checkForUpdate();
  assert.equal(first.closeCalls(), 1);
  assert.equal(second.closeCalls(), 0);

  hook.enqueue(async () => null);
  await hook.controller.checkForUpdate();
  assert.equal(second.closeCalls(), 1);
  hook.host.unmount();
});

test("a post-check error closes the unowned update resource", async (t) => {
  t.mock.method(console, "error", () => {});
  const hook = hookHarness();
  const retained = fakeUpdate("2.0.0");
  const malformed = fakeUpdate("2.0.1", false);
  hook.enqueue(async () => retained.update);
  await hook.controller.checkForUpdate();
  hook.enqueue(async () => malformed.update);
  await hook.controller.checkForUpdate();
  assert.equal(malformed.closeCalls(), 1);
  assert.equal(retained.closeCalls(), 0);
  hook.host.unmount();
  assert.equal(retained.closeCalls(), 1);
});

test("manual update discovery releases the in-app update resource", async () => {
  const hook = hookHarness();
  const retained = fakeUpdate("2.0.0");
  hook.enqueue(async () => retained.update);
  await hook.controller.checkForUpdate();
  hook.useManualUpdate({
    version: "2.0.1",
    currentVersion: "1.0.0",
  });
  await hook.controller.checkForUpdate();
  assert.equal(retained.closeCalls(), 1);
  hook.host.unmount();
});

test("unmount closes owned and late in-flight update resources", async () => {
  const hook = hookHarness();
  const retained = fakeUpdate("2.0.0");
  hook.enqueue(async () => retained.update);
  await hook.controller.checkForUpdate();

  let resolveLate: ((update: FakeUpdate) => void) | undefined;
  let markStarted: (() => void) | undefined;
  const started = new Promise<void>((resolve) => {
    markStarted = resolve;
  });
  const lateResult = new Promise<FakeUpdate>((resolve) => {
    resolveLate = resolve;
  });
  hook.enqueue(() => {
    markStarted?.();
    return lateResult;
  });
  const inFlight = hook.controller.checkForUpdate();
  await started;
  hook.host.unmount();
  assert.equal(retained.closeCalls(), 1);

  const late = fakeUpdate("2.0.1");
  resolveLate?.(late.update);
  await inFlight;
  assert.equal(late.closeCalls(), 1);
});
