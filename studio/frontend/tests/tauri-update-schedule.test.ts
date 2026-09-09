// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { type TestContext } from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

const STARTUP_DELAY_MS = 5_000;
const PERIODIC_INTERVAL_MS = 60 * 60 * 1_000;
const BUNDLE_POLL_MS = 500;
const BUNDLE_WAIT_MS = 10 * 60 * 1_000;

/** The shape `patchPreparation` writes; only the fields the tests read. */
interface PreparationState {
  shell: string;
  backend: string;
  shellProgress: number;
}

interface BundleState {
  version: string | null;
  downloaded: boolean;
  downloading: boolean;
}

type UpdateController = {
  checkForUpdate: () => Promise<void>;
  installUpdate: () => Promise<void>;
  prepareUpdate: (version: string) => Promise<void>;
};

/** Every prefetch bridge call the hook can make, in the order it made them. */
type PrefetchCalls = string[];

type Listener = EventListenerOrEventListenerObject;

interface HookHarnessOptions {
  failCheckAt?: number;
  noUpdateAt?: number;
  tauri?: boolean;
  /** Whether `start_backend_update` resolves; the shell steps only run if it does. */
  backendUpdate?: "completes" | "fails";
  /** Whether `downloadDesktopUpdate` resolves; the shell refuses a second one. */
  bundleDownload?: "completes" | "refuses";
  /** One entry per `desktopUpdateBundleStatus` poll; the last one repeats. */
  bundleStates?: BundleState[];
  /** Version `checkDesktopUpdate` starts answering with from this call on. */
  newVersionAt?: number;
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
  const progressUpdates: number[] = [];
  const preparationUpdates: PreparationState[] = [];
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
            if (typeof next === "number") progressUpdates.push(next);
            // The preparation is the only object state, and named by its shape.
            if (typeof next === "object" && next !== null && "shell" in next)
              preparationUpdates.push(next as PreparationState);
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
    preparationUpdates,
    progressUpdates,
    statusUpdates,
  };
}

function hookHarness(
  t: TestContext,
  {
    failCheckAt,
    noUpdateAt,
    newVersionAt,
    tauri = true,
    backendUpdate = "fails",
    bundleDownload = "completes",
    bundleStates = [{ version: null, downloaded: false, downloading: false }],
  }: HookHarnessOptions = {},
) {
  const browser = installBrowserClock();
  const host = createHookReact();
  t.after(() => {
    host.unmount();
    browser.restore();
  });
  let checks = 0;
  let polls = 0;
  let relaunches = 0;
  const events = new Map<string, Set<(event: { payload: unknown }) => void>>();
  const emit = (name: string, payload?: unknown) => {
    for (const callback of events.get(name) ?? []) callback({ payload });
  };
  // What the hook does with the download it is only watching, not running.
  const download: {
    attached: string[];
    released: number;
    started: number;
    report: (percent: number) => void;
  } = {
    attached: [],
    released: 0,
    started: 0,
    report: () => {
      throw new Error("no download listener is attached");
    },
  };
  const prefetchCalls: PrefetchCalls = [];
  let prefetch = {
    state: "none" as const,
    backendVersion: null,
    shellVersion: null,
    cacheDir: null,
    createdAt: null,
    running: false,
    runningShellVersion: null,
  };
  const updater = {
    checkDesktopUpdate: () => {
      checks += 1;
      if (checks === failCheckAt) throw new Error("update check failed");
      if (checks === noUpdateAt) return Promise.resolve(null);
      return Promise.resolve({
        version: newVersionAt !== undefined && checks >= newVersionAt ? "3.0.0" : "2.0.0",
        currentVersion: "1.0.0",
        rawJson: {},
      });
    },
    desktopUpdateBundleStatus: () => {
      const state = bundleStates[Math.min(polls, bundleStates.length - 1)];
      polls += 1;
      return Promise.resolve(state);
    },
    downloadDesktopUpdate: () => {
      download.started += 1;
      if (bundleDownload === "refuses")
        return Promise.reject(new Error("a download is already running"));
      return Promise.resolve();
    },
    installDesktopUpdate: () => Promise.resolve(),
    listenDesktopUpdateDownload: (
      version: string,
      onProgress: (percent: number) => void,
    ) => {
      download.attached.push(version);
      download.report = onProgress;
      return Promise.resolve(() => {
        download.released += 1;
      });
    },
    sameUpdateVersion: (left: string | null | undefined, right: string) =>
      Boolean(left) && left === right,
    prefetchStatus: () => {
      prefetchCalls.push("status");
      return Promise.resolve(prefetch);
    },
    startPrefetch: (version: string) => {
      prefetchCalls.push(`start:${version}`);
      return Promise.resolve("ready");
    },
    adoptPrefetch: () => {
      prefetchCalls.push("adopt");
      return Promise.resolve(prefetch);
    },
    cancelPrefetch: () => {
      prefetchCalls.push("cancel");
      return Promise.resolve();
    },
    discardPrefetch: () => {
      prefetchCalls.push("discard");
      return Promise.resolve();
    },
  };
  // The real decision table, so the harness cannot disagree with the shipped one.
  const preparation = loadWithStubs<Record<string, unknown>>(
    new URL("../src/lib/update-preparation.ts", import.meta.url),
    { "@/lib/tauri-updater": updater },
  );
  const hook = loadWithStubs<{
    useTauriUpdate: () => UpdateController;
  }>(new URL("../src/hooks/use-tauri-update.ts", import.meta.url), {
    react: host.react,
    "@/lib/api-base": { isTauri: tauri },
    "@/lib/tauri-diagnostics": {
      copySupportDiagnostics: async () => ({ copied: true }),
    },
    "@/lib/tauri-updater": updater,
    "@/lib/update-preparation": preparation,
    "@/lib/toast": { toast: { error: () => undefined } },
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
        if (command === "start_backend_update") {
          // The command itself decides the backend step, rather than a stub that happens to throw.
          if (backendUpdate === "fails")
            throw new Error("backend update failed");
          queueMicrotask(() => emit("update-complete"));
          return undefined;
        }
        if (command === "set_renderer_activity") return undefined;
        if (command === "mark_in_app_relaunch") return undefined;
        throw new Error(`unexpected invoke: ${command}`);
      },
    },
    "@tauri-apps/api/event": {
      listen: async (
        name: string,
        callback: (event: { payload: unknown }) => void,
      ) => {
        const registered =
          events.get(name) ?? new Set<(event: { payload: unknown }) => void>();
        registered.add(callback);
        events.set(name, registered);
        return () => {
          registered.delete(callback);
        };
      },
    },
    "@tauri-apps/plugin-process": {
      relaunch: async () => {
        relaunches += 1;
      },
    },
  });
  const controller = hook.useTauriUpdate();
  host.mount();
  return {
    browser,
    checks: () => checks,
    controller,
    download,
    host,
    polls: () => polls,
    prefetchCalls,
    preparationUpdates: host.preparationUpdates,
    progressUpdates: host.progressUpdates,
    relaunches: () => relaunches,
    setPrefetch: (next: Partial<typeof prefetch>) => {
      prefetch = { ...prefetch, ...next } as typeof prefetch;
    },
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

test("a periodic recheck keeps an offered update available", async (t) => {
  const hook = hookHarness(t);
  hook.browser.fireTimeouts(STARTUP_DELAY_MS);
  await settle();
  assert.equal(hook.statusUpdates.at(-1), "available");

  hook.browser.fireIntervals(PERIODIC_INTERVAL_MS);
  await settle();
  assert.equal(hook.checks(), 2);
  assert.equal(hook.statusUpdates.at(-1), "available");
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

test("a withdrawn offer goes back to idle", async (t) => {
  const hook = hookHarness(t, { noUpdateAt: 2 });
  hook.browser.fireTimeouts(STARTUP_DELAY_MS);
  await settle();
  assert.equal(hook.statusUpdates.at(-1), "available");

  hook.browser.fireIntervals(PERIODIC_INTERVAL_MS);
  await settle();
  assert.equal(hook.checks(), 2);
  assert.equal(hook.statusUpdates.at(-1), "idle");
});

test("scheduled checks leave a failed install in its error state", async (t) => {
  const hook = hookHarness(t);
  hook.browser.fireTimeouts(STARTUP_DELAY_MS);
  await settle();
  assert.equal(hook.statusUpdates.at(-1), "available");

  // First press prepares in the background; the offer only becomes installable
  // once both halves have settled.
  await hook.controller.installUpdate();
  await settle();
  await settle();
  assert.equal(hook.statusUpdates.at(-1), "ready");

  // Second press is the restart, and start_backend_update itself refuses, which
  // is the failure the classic path reports.
  await hook.controller.installUpdate();
  await settle();
  assert.equal(hook.statusUpdates.at(-1), "error");

  hook.browser.advance(PERIODIC_INTERVAL_MS + 1);
  hook.browser.fireIntervals(PERIODIC_INTERVAL_MS);
  hook.browser.fireWindow("focus");
  await settle();
  assert.equal(hook.checks(), 1);
  assert.equal(hook.statusUpdates.at(-1), "error");
});

test("a bundle download the update did not start reports its progress", async (t) => {
  const hook = hookHarness(t, {
    backendUpdate: "completes",
    bundleStates: [
      // The first press only prepares, and finds the bundle already retained.
      { version: "2.0.0", downloaded: true, downloading: false },
      // By Restart the retained bundle is gone and a native download this
      // renderer did not start is in flight; download_desktop_update would
      // refuse a second one, so the update watches this one instead.
      { version: "2.0.0", downloaded: false, downloading: true },
      { version: "2.0.0", downloaded: false, downloading: true },
      { version: "2.0.0", downloaded: true, downloading: false },
    ],
  });
  hook.browser.fireTimeouts(STARTUP_DELAY_MS);
  await settle();

  // First press prepares; the bundle is already there, so nothing is downloaded
  // and nothing is watched yet.
  await hook.controller.installUpdate();
  await settle();
  await settle();
  assert.equal(hook.statusUpdates.at(-1), "ready");
  assert.deepEqual(hook.download.attached, []);

  const installing = hook.controller.installUpdate();
  await settle();
  await settle();
  assert.deepEqual(hook.download.attached, ["2.0.0"]);
  hook.download.report(40);

  hook.browser.fireTimeouts(BUNDLE_POLL_MS);
  await settle();
  assert.deepEqual(hook.download.attached, ["2.0.0"]);

  hook.browser.fireTimeouts(BUNDLE_POLL_MS);
  await settle();
  await installing;

  // One poll for the preparation and three for the wait it took over.
  assert.equal(hook.polls(), 4);
  // Watched to the end, not restarted, and the listener let go either way.
  assert.equal(hook.download.started, 0);
  assert.equal(hook.download.released, 1);
  assert.ok(hook.progressUpdates.includes(40));
  assert.equal(hook.progressUpdates.at(-1), 100);
  assert.equal(hook.relaunches(), 1);
});

test("waiting out a bundle download the update did not start is bounded", async (t) => {
  const hook = hookHarness(t, {
    backendUpdate: "completes",
    bundleStates: [
      // The first press only prepares, and finds the bundle already retained.
      { version: "2.0.0", downloaded: true, downloading: false },
      // Stuck: the flag never clears, so without the bound the update waits forever.
      { version: "2.0.0", downloaded: false, downloading: true },
    ],
  });
  hook.browser.fireTimeouts(STARTUP_DELAY_MS);
  await settle();

  await hook.controller.installUpdate();
  await settle();
  await settle();
  assert.equal(hook.statusUpdates.at(-1), "ready");
  assert.equal(hook.download.started, 0);

  const installing = hook.controller.installUpdate();
  await settle();
  await settle();
  assert.deepEqual(hook.download.attached, ["2.0.0"]);

  hook.browser.advance(BUNDLE_WAIT_MS);
  hook.browser.fireTimeouts(BUNDLE_POLL_MS);
  await settle();
  await installing;

  // Handed back to the real download, which is what surfaces the failure.
  assert.equal(hook.download.started, 1);
  assert.equal(hook.download.released, 1);
});

test("a preparation that watches a download somebody else started shows it", async (t) => {
  const hook = hookHarness(t, {
    // A webview reload left a native download running, and the press that
    // prepares runs into it before the press that installs ever happens.
    bundleStates: [
      { version: "2.0.0", downloaded: false, downloading: true },
      { version: "2.0.0", downloaded: false, downloading: true },
      { version: "2.0.0", downloaded: true, downloading: false },
    ],
  });
  hook.browser.fireTimeouts(STARTUP_DELAY_MS);
  await settle();

  await hook.controller.installUpdate();
  await settle();
  await settle();
  assert.deepEqual(hook.download.attached, ["2.0.0"]);
  hook.download.report(40);

  hook.browser.fireTimeouts(BUNDLE_POLL_MS);
  await settle();
  hook.browser.fireTimeouts(BUNDLE_POLL_MS);
  await settle();

  // The adopted download is what the preparation reports, all the way to done.
  assert.ok(
    hook.preparationUpdates.some((state) => state.shellProgress === 40),
  );
  assert.equal(hook.preparationUpdates.at(-1)?.shell, "done");
  // Watched to the end, never restarted, and the listener let go once.
  assert.equal(hook.statusUpdates.at(-1), "ready");
  assert.equal(hook.download.started, 0);
  assert.equal(hook.download.released, 1);
  assert.deepEqual(hook.download.attached, ["2.0.0"]);
});

test("a preparation waiting on somebody else's download is bounded", async (t) => {
  const hook = hookHarness(t, {
    // Stuck: the flag never clears, so without the bound the offer sits at
    // "preparing" for good and the Restart button never arrives.
    bundleStates: [{ version: "2.0.0", downloaded: false, downloading: true }],
    bundleDownload: "refuses",
  });
  hook.browser.fireTimeouts(STARTUP_DELAY_MS);
  await settle();

  await hook.controller.installUpdate();
  await settle();
  await settle();
  assert.deepEqual(hook.download.attached, ["2.0.0"]);
  assert.equal(hook.statusUpdates.at(-1), "preparing");
  assert.equal(hook.download.started, 0);

  hook.browser.advance(BUNDLE_WAIT_MS);
  hook.browser.fireTimeouts(BUNDLE_POLL_MS);
  await settle();

  // Handed back to the real download, and its refusal puts the plain Update
  // button back rather than leaving the offer stuck on a bar that never moves.
  assert.equal(hook.download.started, 1);
  assert.equal(hook.download.released, 1);
  assert.equal(hook.preparationUpdates.at(-1)?.shell, "failed");
  assert.equal(hook.statusUpdates.at(-1), "available");
});

test("a recheck of the version being prepared does not restart the prefetch", async (t) => {
  const hook = hookHarness(t);
  hook.browser.fireTimeouts(STARTUP_DELAY_MS);
  await settle();

  await hook.controller.installUpdate();
  await settle();
  await settle();
  assert.equal(hook.statusUpdates.at(-1), "ready");
  assert.deepEqual(hook.prefetchCalls, ["status", "start:2.0.0"]);

  hook.browser.fireIntervals(PERIODIC_INTERVAL_MS);
  await settle();
  assert.equal(hook.checks(), 2);
  // The offer is unchanged, so the pill stays on Restart and nothing is redone.
  assert.equal(hook.statusUpdates.at(-1), "ready");
  assert.deepEqual(hook.prefetchCalls, ["status", "start:2.0.0"]);
});

test("a newer offer cancels the preparation and starts it again", async (t) => {
  const hook = hookHarness(t, { newVersionAt: 2 });
  hook.browser.fireTimeouts(STARTUP_DELAY_MS);
  await settle();

  await hook.controller.installUpdate();
  await settle();
  await settle();
  assert.deepEqual(hook.prefetchCalls, ["status", "start:2.0.0"]);

  hook.browser.fireIntervals(PERIODIC_INTERVAL_MS);
  await settle();
  await settle();
  await settle();
  assert.deepEqual(hook.prefetchCalls, [
    "status",
    "start:2.0.0",
    "cancel",
    "status",
    "start:3.0.0",
  ]);
  assert.equal(hook.statusUpdates.at(-1), "ready");
});

test("a withdrawn offer discards what was prepared for it", async (t) => {
  const hook = hookHarness(t, { noUpdateAt: 2 });
  hook.browser.fireTimeouts(STARTUP_DELAY_MS);
  await settle();

  await hook.controller.installUpdate();
  await settle();
  await settle();
  assert.deepEqual(hook.prefetchCalls, ["status", "start:2.0.0"]);

  hook.browser.fireIntervals(PERIODIC_INTERVAL_MS);
  await settle();
  assert.equal(hook.statusUpdates.at(-1), "idle");
  // Nothing is on offer any more, so the prepared copy is holding disk for nothing.
  assert.equal(hook.prefetchCalls.at(-1), "discard");
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
