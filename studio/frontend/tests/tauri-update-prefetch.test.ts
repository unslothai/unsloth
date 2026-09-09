// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { type TestContext } from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type PrefetchOutcome = "ready" | "unsupported" | "busy" | "failed";

interface HarnessOptions {
  /** What `startPrefetch` settles on. */
  outcome?: PrefetchOutcome;
  /** The bundle is already on disk before anything is prepared. */
  bundleReady?: boolean;
  /** Hold `start_backend_update` open until this resolves. */
  holdUpdate?: () => Promise<void>;
}

interface Controller {
  checkForUpdate: () => Promise<void>;
  installUpdate: () => Promise<void>;
}

/**
 * Enough of a browser for the hook's scheduling effect to mount, with timers that
 * never fire. Without it the effect throws on `window`, leaves a real one-hour
 * interval behind, and the test runner never exits.
 */
function installBrowserStubs() {
  const saved = new Map<PropertyKey, PropertyDescriptor | undefined>();
  const define = (key: PropertyKey, value: unknown) => {
    saved.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, {
      configurable: true,
      writable: true,
      value,
    });
  };
  const noListeners = { addEventListener: () => {}, removeEventListener: () => {} };
  define("window", noListeners);
  define("document", { ...noListeners, hidden: false });
  define("setTimeout", () => 0);
  define("clearTimeout", () => {});
  define("setInterval", () => 0);
  define("clearInterval", () => {});
  return () => {
    for (const [key, descriptor] of saved) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else Reflect.deleteProperty(globalThis, key);
    }
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
            if (index === 0 && typeof next === "string") statusUpdates.push(next);
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

/**
 * The hook with a native side that answers rather than throws, so both presses
 * of the update button can be driven: the first prepares, the second restarts.
 */
function harness(
  t: TestContext,
  { outcome = "ready", bundleReady = false, holdUpdate }: HarnessOptions = {},
) {
  const host = createHookReact();
  const calls: string[] = [];
  let downloaded = bundleReady;
  const listeners = new Map<string, (event: { payload: unknown }) => void>();

  const updater = {
    checkDesktopUpdate: () =>
      Promise.resolve({ version: "2.0.0", currentVersion: "1.0.0", rawJson: {} }),
    desktopUpdateBundleStatus: () =>
      Promise.resolve({
        version: "2.0.0",
        downloaded,
        downloading: false,
      }),
    downloadDesktopUpdate: (_version: string, onProgress: (p: number) => void) => {
      calls.push("download_desktop_update");
      downloaded = true;
      onProgress(100);
      return Promise.resolve();
    },
    installDesktopUpdate: () => {
      calls.push("install_desktop_update");
      return Promise.resolve();
    },
    sameUpdateVersion: (left: string | null | undefined, right: string) =>
      Boolean(left) && left === right,
    prefetchStatus: () => {
      calls.push("prefetch_status");
      return Promise.resolve({
        state: "none",
        backendVersion: null,
        shellVersion: null,
        cacheDir: null,
        createdAt: null,
        running: false,
        runningShellVersion: null,
      });
    },
    startPrefetch: () => {
      calls.push("start_prefetch_update");
      return Promise.resolve(outcome);
    },
    adoptPrefetch: () => Promise.resolve({ state: "ready" }),
    cancelPrefetch: () => {
      calls.push("cancel_prefetch_update");
      return Promise.resolve();
    },
    discardPrefetch: () => {
      calls.push("discard_prefetch");
      return Promise.resolve();
    },
  };
  const preparation = loadWithStubs<Record<string, unknown>>(
    new URL("../src/lib/update-preparation.ts", import.meta.url),
    { "@/lib/tauri-updater": updater },
  );

  const hook = loadWithStubs<{ useTauriUpdate: () => Controller }>(
    new URL("../src/hooks/use-tauri-update.ts", import.meta.url),
    {
      react: host.react,
      "@/lib/api-base": { isTauri: true },
      "@/lib/tauri-diagnostics": {
        copySupportDiagnostics: async () => ({ copied: true }),
      },
      "@/lib/tauri-updater": updater,
      "@/lib/update-preparation": preparation,
      "@/lib/toast": { toast: { error: () => undefined } },
      "@tauri-apps/api/core": {
        invoke: async (command: string) => {
          calls.push(command);
          if (command === "desktop_update_policy") {
            return {
              mode: "in_app",
              releasePageBaseUrl: "https://example.com/",
              releaseTagPrefix: "v",
            };
          }
          if (command === "desktop_update_cleanup_armed") return true;
          if (command === "start_backend_update") {
            // The hook registers its listeners inside the same executor that
            // calls this, and those registrations are promises, so answering
            // synchronously would emit into nothing and park the update forever.
            await settleUntil(() => listeners.has("update-complete"));
            if (holdUpdate) await holdUpdate();
            listeners.get("update-complete")?.({ payload: undefined });
            return undefined;
          }
          return undefined;
        },
      },
      "@tauri-apps/api/event": {
        listen: async (
          name: string,
          handler: (event: { payload: unknown }) => void,
        ) => {
          listeners.set(name, handler);
          return () => listeners.delete(name);
        },
      },
      "@tauri-apps/plugin-process": {
        relaunch: () => {
          calls.push("relaunch");
          return Promise.resolve();
        },
      },
    },
  );
  const controller = hook.useTauriUpdate();
  const restore = installBrowserStubs();
  host.mount();
  t.after(() => {
    host.unmount();
    restore();
  });
  return { calls, controller, statusUpdates: host.statusUpdates };
}

function settle(): Promise<void> {
  return new Promise((resolve) => setImmediate(resolve));
}

/** Give the microtask queue a bounded number of turns to make `ready` true. */
async function settleUntil(ready: () => boolean): Promise<void> {
  for (let turn = 0; turn < 100 && !ready(); turn += 1) await settle();
}

test("a prepared offer reaches ready without installing anything", async (t) => {
  const hook = harness(t);
  await hook.controller.checkForUpdate();
  await settle();
  assert.equal(hook.statusUpdates.at(-1), "available");

  await hook.controller.installUpdate();
  await settle();
  await settle();

  assert.equal(hook.statusUpdates.at(-1), "ready");
  assert.ok(hook.calls.includes("start_prefetch_update"));
  // Preparation downloads; it must not start the update or touch the app bundle.
  assert.ok(!hook.calls.includes("start_backend_update"));
  assert.ok(!hook.calls.includes("install_desktop_update"));
});

test("a backend without the command still gets the offer to ready", async (t) => {
  const hook = harness(t, { outcome: "unsupported" });
  await hook.controller.checkForUpdate();
  await settle();

  await hook.controller.installUpdate();
  await settle();
  await settle();

  // The release that introduces the prefetch is running against the previous
  // backend, which has no such command. That is the expected answer, not a fault.
  assert.equal(hook.statusUpdates.at(-1), "ready");
});

test("a failed prefetch still reaches ready", async (t) => {
  const hook = harness(t, { outcome: "failed" });
  await hook.controller.checkForUpdate();
  await settle();

  await hook.controller.installUpdate();
  await settle();
  await settle();

  // Nothing was warmed, so the restart downloads its own wheels. That is the
  // behaviour of every release before this one, and it is not an error.
  assert.equal(hook.statusUpdates.at(-1), "ready");
});

test("the restart never downloads the app bundle a second time", async (t) => {
  const hook = harness(t);
  await hook.controller.checkForUpdate();
  await settle();

  await hook.controller.installUpdate();
  await settle();
  await settle();
  const downloads = hook.calls.filter((c) => c === "download_desktop_update").length;
  assert.equal(downloads, 1);

  await hook.controller.installUpdate();
  await settle();

  assert.equal(
    hook.calls.filter((c) => c === "download_desktop_update").length,
    downloads,
  );
  assert.ok(hook.calls.includes("start_backend_update"));
  assert.ok(hook.calls.includes("install_desktop_update"));
  assert.ok(hook.calls.includes("relaunch"));
  // The restart owns the environment from here, so nothing may be preparing.
  const cancel = hook.calls.indexOf("cancel_prefetch_update");
  assert.ok(cancel !== -1 && cancel < hook.calls.indexOf("start_backend_update"));
});

test("an offer whose bundle is already on disk prepares without downloading", async (t) => {
  const hook = harness(t, { bundleReady: true });
  await hook.controller.checkForUpdate();
  await settle();

  await hook.controller.installUpdate();
  await settle();
  await settle();

  assert.equal(hook.statusUpdates.at(-1), "ready");
  assert.ok(!hook.calls.includes("download_desktop_update"));
});

test("a check already in flight cannot reopen the offer mid-install", async (t) => {
  // Assigned inside the executor, which runs synchronously; the explicit type keeps
  // TypeScript from narrowing the binding to `never` at the call below.
  let release: () => void = () => {};
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  const hook = harness(t, { holdUpdate: () => held });
  await hook.controller.checkForUpdate();
  await settle();

  await hook.controller.installUpdate();
  await settle();
  await settle();
  assert.equal(hook.statusUpdates.at(-1), "ready");

  const restart = hook.controller.installUpdate();
  await settleUntil(() => hook.calls.includes("start_backend_update"));
  assert.equal(hook.statusUpdates.at(-1), "updating-backend");

  // The hourly check fires while the update child is running. Before this was
  // guarded it put the status back to "ready" and started a second download.
  const downloads = hook.calls.filter((c) => c === "download_desktop_update").length;
  await hook.controller.checkForUpdate();
  await settle();
  assert.equal(hook.statusUpdates.at(-1), "updating-backend");
  assert.equal(
    hook.calls.filter((c) => c === "download_desktop_update").length,
    downloads,
  );

  release();
  await restart;
});
