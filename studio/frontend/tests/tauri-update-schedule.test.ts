// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { type TestContext } from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

const STARTUP_DELAY_MS = 5_000;
const PERIODIC_INTERVAL_MS = 60 * 60 * 1_000;

type UpdateController = {
  checkForUpdate: () => Promise<void>;
  installUpdate: () => Promise<void>;
};

type Listener = EventListenerOrEventListenerObject;

interface HookHarnessOptions {
  failCheckAt?: number;
  holdPreparation?: boolean;
  noUpdateAt?: number;
  rejectDiscard?: boolean;
  tauri?: boolean;
}

function createEventTarget() {
  const listeners = new Map<string, Set<Listener>>();
  return {
    addEventListener(type: string, listener: Listener): void {
      const registered = listeners.get(type) ?? new Set<Listener>();
      registered.add(listener);
      listeners.set(type, registered);
    },
    removeEventListener(type: string, listener: Listener): void {
      listeners.get(type)?.delete(listener);
    },
    fire(type: string): void {
      const event = new Event(type);
      for (const listener of listeners.get(type) ?? []) {
        if (typeof listener === "function") listener(event);
        else listener.handleEvent(event);
      }
    },
    listenerCount(): number {
      let count = 0;
      for (const registered of listeners.values()) count += registered.size;
      return count;
    },
  };
}

function restoreProperty(
  target: object,
  key: PropertyKey,
  descriptor: PropertyDescriptor | undefined,
): void {
  if (descriptor) Object.defineProperty(target, key, descriptor);
  else Reflect.deleteProperty(target, key);
}

function installBrowserClock() {
  const windowTarget = createEventTarget();
  const documentTarget = createEventTarget();
  const timeouts = new Map<number, { callback: () => void; delay: number }>();
  const intervals = new Map<number, { callback: () => void; delay: number }>();
  const originalDescriptors = {
    window: Object.getOwnPropertyDescriptor(globalThis, "window"),
    document: Object.getOwnPropertyDescriptor(globalThis, "document"),
    setTimeout: Object.getOwnPropertyDescriptor(globalThis, "setTimeout"),
    clearTimeout: Object.getOwnPropertyDescriptor(globalThis, "clearTimeout"),
    setInterval: Object.getOwnPropertyDescriptor(globalThis, "setInterval"),
    clearInterval: Object.getOwnPropertyDescriptor(globalThis, "clearInterval"),
    dateNow: Object.getOwnPropertyDescriptor(Date, "now"),
  };
  let hidden = false;
  let now = 1_000;
  let nextTimerId = 1;

  const windowStub = {
    addEventListener: windowTarget.addEventListener,
    removeEventListener: windowTarget.removeEventListener,
  };
  const documentStub = {
    addEventListener: documentTarget.addEventListener,
    removeEventListener: documentTarget.removeEventListener,
  };
  Object.defineProperty(documentStub, "hidden", {
    configurable: true,
    get: () => hidden,
  });

  Object.defineProperties(globalThis, {
    window: { configurable: true, writable: true, value: windowStub },
    document: { configurable: true, writable: true, value: documentStub },
    setTimeout: {
      configurable: true,
      writable: true,
      value: ((callback: () => void, delay = 0) => {
        const id = nextTimerId++;
        timeouts.set(id, { callback, delay });
        return id;
      }) as unknown as typeof setTimeout,
    },
    clearTimeout: {
      configurable: true,
      writable: true,
      value: ((id: number) =>
        timeouts.delete(id)) as unknown as typeof clearTimeout,
    },
    setInterval: {
      configurable: true,
      writable: true,
      value: ((callback: () => void, delay = 0) => {
        const id = nextTimerId++;
        intervals.set(id, { callback, delay });
        return id;
      }) as unknown as typeof setInterval,
    },
    clearInterval: {
      configurable: true,
      writable: true,
      value: ((id: number) =>
        intervals.delete(id)) as unknown as typeof clearInterval,
    },
  });
  Object.defineProperty(Date, "now", {
    configurable: true,
    writable: true,
    value: () => now,
  });

  return {
    activeTimers: () => timeouts.size + intervals.size,
    advance: (elapsed: number) => {
      now += elapsed;
    },
    delays: () => [
      ...[...timeouts.values()].map(({ delay }) => delay),
      ...[...intervals.values()].map(({ delay }) => delay),
    ],
    fireDocument: (type: string) => documentTarget.fire(type),
    fireIntervals: (delay: number) => {
      for (const timer of intervals.values()) {
        if (timer.delay === delay) timer.callback();
      }
    },
    fireTimeouts: (delay: number) => {
      for (const [id, timer] of [...timeouts]) {
        if (timer.delay !== delay) continue;
        timeouts.delete(id);
        timer.callback();
      }
    },
    fireWindow: (type: string) => windowTarget.fire(type),
    listenerCount: () =>
      windowTarget.listenerCount() + documentTarget.listenerCount(),
    setHidden: (nextHidden: boolean) => {
      hidden = nextHidden;
    },
    restore(): void {
      restoreProperty(globalThis, "window", originalDescriptors.window);
      restoreProperty(globalThis, "document", originalDescriptors.document);
      restoreProperty(globalThis, "setTimeout", originalDescriptors.setTimeout);
      restoreProperty(
        globalThis,
        "clearTimeout",
        originalDescriptors.clearTimeout,
      );
      restoreProperty(
        globalThis,
        "setInterval",
        originalDescriptors.setInterval,
      );
      restoreProperty(
        globalThis,
        "clearInterval",
        originalDescriptors.clearInterval,
      );
      restoreProperty(Date, "now", originalDescriptors.dateNow);
    },
  };
}

function createHookReact() {
  const effects: Array<() => unknown> = [];
  const cleanups: Array<() => void> = [];
  const statusUpdates: string[] = [];
  let stateIndex = 0;
  return {
    react: {
      useState<T>(initial: T): [T, (next: unknown) => void] {
        const index = stateIndex++;
        return [
          initial,
          (next: unknown) => {
            if (index === 0 && typeof next === "string")
              statusUpdates.push(next);
          },
        ];
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
        if (typeof cleanup === "function") cleanups.push(cleanup as () => void);
      }
    },
    unmount(): void {
      for (const cleanup of cleanups.splice(0)) cleanup();
    },
    statusUpdates,
  };
}

function hookHarness(
  t: TestContext,
  {
    failCheckAt,
    holdPreparation = false,
    noUpdateAt,
    rejectDiscard = false,
    tauri = true,
  }: HookHarnessOptions = {},
) {
  const browser = installBrowserClock();
  const host = createHookReact();
  t.after(() => {
    host.unmount();
    browser.restore();
  });
  let checks = 0;
  const initialPreparation = {
    shell: "pending",
    backend: "pending",
    shellProgress: 0,
  };
  const hook = loadWithStubs<{
    useTauriUpdate: () => UpdateController;
  }>(new URL("../src/hooks/use-tauri-update.ts", import.meta.url), {
    react: host.react,
    "@/features/training": {
      isTrainingStartPending: () => false,
      useTrainingRuntimeStore: { getState: () => ({}) },
    },
    "@/lib/api-base": { apiUrl: (path: string) => path, isTauri: tauri },
    "@/lib/tauri-diagnostics": {
      copySupportDiagnostics: async () => ({ copied: true }),
    },
    "@/lib/tauri-updater": {
      adoptStagedUpdate: () => Promise.resolve({}),
      cancelStagedUpdate: () => Promise.resolve(),
      checkDesktopUpdate: () => {
        checks += 1;
        if (checks === failCheckAt) throw new Error("update check failed");
        if (checks === noUpdateAt) return Promise.resolve(null);
        return Promise.resolve({
          version: "2.0.0",
          currentVersion: "1.0.0",
          rawJson: {},
        });
      },
      desktopUpdateBundleStatus: () =>
        holdPreparation
          ? new Promise(() => {})
          : Promise.resolve({ downloaded: false }),
      discardStagedUpdate: () =>
        rejectDiscard
          ? Promise.reject(new Error("discard failed"))
          : Promise.resolve(),
      downloadDesktopUpdate: () => Promise.resolve(),
      installDesktopUpdate: () => Promise.resolve(),
      stagedUpdateStatus: () => Promise.resolve({ staging: false }),
      startStagedUpdate: () => Promise.resolve(),
      waitForDesktopUpdateDownload: () => Promise.resolve(),
    },
    "@/lib/toast": { toast: { error: () => undefined } },
    "@/lib/update-preparation": {
      INITIAL_PREPARATION: initialPreparation,
      backendIdle: () => true,
      desktopDownloadDecision: () => "ready",
      preparationStatus: (preparation: typeof initialPreparation) =>
        preparation.shell === "done" && preparation.backend === "skipped"
          ? "ready"
          : "preparing",
      restartPlan: () => "classic",
      sameUpdateVersion: () => true,
      settleWithin: async () => null,
      stagingDecision: () => "skip",
      waitForBackendIdle: async () => "cancelled",
    },
    "@tauri-apps/api/core": {
      invoke: async (command: string) => {
        if (command === "desktop_update_policy") {
          return {
            mode: "in_app",
            releasePageBaseUrl: "https://example.com/",
            releaseTagPrefix: "v",
          };
        }
        if (command === "desktop_update_cleanup_armed") return true;
        throw new Error(`unexpected invoke: ${command}`);
      },
    },
    "@tauri-apps/api/event": {
      listen: async () => {
        throw new Error("backend update failed");
      },
    },
  });
  const controller = hook.useTauriUpdate();
  host.mount();
  return {
    browser,
    checks: () => checks,
    controller,
    host,
    statusUpdates: host.statusUpdates,
  };
}

function settle(): Promise<void> {
  return new Promise((resolve) => setImmediate(resolve));
}

test("the desktop hook checks at startup and every hour", async (t) => {
  const hook = hookHarness(t);
  assert.deepEqual(hook.browser.delays(), [
    STARTUP_DELAY_MS,
    PERIODIC_INTERVAL_MS,
  ]);

  hook.browser.fireWindow("focus");
  await settle();
  assert.equal(hook.checks(), 0);

  hook.browser.fireTimeouts(STARTUP_DELAY_MS);
  await settle();
  assert.equal(hook.checks(), 1);

  hook.browser.fireIntervals(PERIODIC_INTERVAL_MS);
  await settle();
  assert.equal(hook.checks(), 2);
});

test("a manual check suppresses only the startup check", async (t) => {
  const hook = hookHarness(t);
  await hook.controller.checkForUpdate();

  hook.browser.fireTimeouts(STARTUP_DELAY_MS);
  await settle();
  assert.equal(hook.checks(), 1);

  hook.browser.fireIntervals(PERIODIC_INTERVAL_MS);
  await settle();
  assert.equal(hook.checks(), 2);
});

test("a periodic recheck preserves a prepared update", async (t) => {
  const hook = hookHarness(t);
  hook.browser.fireTimeouts(STARTUP_DELAY_MS);
  await settle();
  await hook.controller.installUpdate();
  await settle();
  assert.equal(hook.statusUpdates.at(-1), "ready");

  hook.browser.fireIntervals(PERIODIC_INTERVAL_MS);
  await settle();
  assert.equal(hook.statusUpdates.at(-1), "ready");
});

test("a failed periodic recheck preserves an untouched offer", async (t) => {
  const hook = hookHarness(t, { failCheckAt: 2 });
  hook.browser.fireTimeouts(STARTUP_DELAY_MS);
  await settle();
  assert.equal(hook.statusUpdates.at(-1), "available");

  hook.browser.fireIntervals(PERIODIC_INTERVAL_MS);
  await settle();
  assert.equal(hook.checks(), 2);
  assert.equal(hook.statusUpdates.at(-1), "available");
});

test("a cleanup failure does not restore a withdrawn offer", async (t) => {
  const hook = hookHarness(t, { noUpdateAt: 2, rejectDiscard: true });
  hook.browser.fireTimeouts(STARTUP_DELAY_MS);
  await settle();
  assert.equal(hook.statusUpdates.at(-1), "available");

  hook.browser.fireIntervals(PERIODIC_INTERVAL_MS);
  await settle();
  assert.equal(hook.checks(), 2);
  assert.equal(hook.statusUpdates.at(-1), "idle");
});

test("scheduled checks wait for update preparation", async (t) => {
  const hook = hookHarness(t, { holdPreparation: true });
  hook.browser.fireTimeouts(STARTUP_DELAY_MS);
  await settle();
  await hook.controller.installUpdate();
  assert.equal(hook.statusUpdates.at(-1), "preparing");

  hook.browser.advance(PERIODIC_INTERVAL_MS + 1);
  hook.browser.fireIntervals(PERIODIC_INTERVAL_MS);
  hook.browser.fireWindow("focus");
  await settle();
  assert.equal(hook.checks(), 1);
  assert.equal(hook.statusUpdates.at(-1), "preparing");
});

test("scheduled checks preserve update recovery", async (t) => {
  const hook = hookHarness(t);
  hook.browser.fireTimeouts(STARTUP_DELAY_MS);
  await settle();
  await hook.controller.installUpdate();
  await settle();
  assert.equal(hook.statusUpdates.at(-1), "ready");

  await hook.controller.installUpdate();
  assert.equal(hook.statusUpdates.at(-1), "error");

  hook.browser.advance(PERIODIC_INTERVAL_MS + 1);
  hook.browser.fireIntervals(PERIODIC_INTERVAL_MS);
  hook.browser.fireWindow("focus");
  await settle();
  assert.equal(hook.checks(), 1);
  assert.equal(hook.statusUpdates.at(-1), "error");
});

test("restoring an overdue hidden window checks immediately", async (t) => {
  const hook = hookHarness(t);
  hook.browser.fireTimeouts(STARTUP_DELAY_MS);
  await settle();
  assert.equal(hook.checks(), 1);

  hook.browser.setHidden(true);
  hook.browser.advance(PERIODIC_INTERVAL_MS + 1);
  hook.browser.fireDocument("visibilitychange");
  await settle();
  assert.equal(hook.checks(), 1);

  hook.browser.setHidden(false);
  hook.browser.fireDocument("visibilitychange");
  hook.browser.fireWindow("focus");
  await settle();
  assert.equal(hook.checks(), 2);
});

test("unmount removes update timers and wake listeners", async (t) => {
  const hook = hookHarness(t);
  assert.equal(hook.browser.activeTimers(), 2);
  assert.equal(hook.browser.listenerCount(), 2);

  hook.host.unmount();
  assert.equal(hook.browser.activeTimers(), 0);
  assert.equal(hook.browser.listenerCount(), 0);

  hook.browser.advance(PERIODIC_INTERVAL_MS + 1);
  hook.browser.fireTimeouts(STARTUP_DELAY_MS);
  hook.browser.fireIntervals(PERIODIC_INTERVAL_MS);
  hook.browser.fireDocument("visibilitychange");
  hook.browser.fireWindow("focus");
  await settle();
  assert.equal(hook.checks(), 0);
});

test("web sessions do not schedule desktop update checks", (t) => {
  const hook = hookHarness(t, { tauri: false });
  assert.equal(hook.browser.activeTimers(), 0);
  assert.equal(hook.browser.listenerCount(), 0);
});
