// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { mock, test } from "node:test";
import * as lifecycle from "../src/lib/llama-job-lifecycle.ts";
import type * as HookModule from "../src/hooks/use-llama-update-check.ts";
import type * as SwitchModule from "../src/features/settings/hooks/use-llama-backend-switch.ts";
import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

function reactStub(options: { runEffects?: boolean; states?: unknown[] } = {}) {
  const states = [...(options.states ?? [])];
  const cleanups: (() => void)[] = [];
  return {
    cleanups,
    react: {
      useState: (initial: unknown) => [
        states.length ? states.shift() : initial,
        () => {},
      ],
      useRef: (current: unknown) => ({ current }),
      useCallback: (fn: unknown) => fn,
      useEffect(effect: () => (() => void) | undefined) {
        if (!options.runEffects) return;
        const cleanup = effect();
        if (cleanup) cleanups.push(cleanup);
      },
    },
  };
}

function loadHook(
  jobState: string,
  options: { installedTag?: () => string | null; runEffects?: boolean } = {},
) {
  const calls: string[] = [];
  const committed: (string | null)[] = [];
  const { react, cleanups } = reactStub(options);
  react.useState = (initial: unknown) => [
    initial,
    (next: unknown) => {
      const status = next as { llama?: { installed_tag: string | null } };
      if (status?.llama) committed.push(status.llama.installed_tag);
    },
  ];
  const job = (state: string) => ({
    state,
    operation: "update",
    started_at: "2026-11-25T10:00:00Z",
    finished_at: state === "running" ? null : "2026-11-25T10:01:00Z",
  });
  const authFetch = async (url: string) => {
    if (url.startsWith("/api/llama/update-status")) calls.push("status");
    return {
      ok: true,
      json: async () =>
        url === "/api/llama/update"
          ? { started: true, job: job("running") }
          : {
              update_available: false,
              installed_tag: options.installedTag?.() ?? null,
              job: job(jobState),
            },
    };
  };
  const hook = loadWithStubs<typeof HookModule>(
    new URL("../src/hooks/use-llama-update-check.ts", import.meta.url),
    {
      react,
      "@/features/auth": { authFetch, getAuthToken: () => "token" },
      "@/features/model-picker/api/llama-flags": {
        invalidateLlamaFlagCatalog: () => calls.push("flags"),
      },
      "@/hooks/use-hardware-info": {
        refreshHardwareInfo: async () => calls.push("hardware"),
      },
      "@/lib/model-lifecycle-events": { notifyModelLifecycle() {} },
      "@/lib/llama-job-events": {
        signalRunningLlamaJob() {},
        subscribeToLlamaJobStarted: () => () => {},
      },
      "@/lib/llama-job-lifecycle": lifecycle,
    },
  );
  const unmount = () => {
    for (const cleanup of cleanups) cleanup();
  };
  return { hook, calls, committed, unmount };
}

const flush = () => new Promise((resolve) => setImmediate(resolve));

function withWindow(run: () => Promise<void>) {
  return async () => {
    const previous = (globalThis as { window?: unknown }).window;
    (globalThis as { window?: unknown }).window = {
      addEventListener() {},
      removeEventListener() {},
      dispatchEvent() {},
    };
    mock.timers.enable({ apis: ["setTimeout", "setInterval"] });
    try {
      await run();
    } finally {
      mock.timers.reset();
      (globalThis as { window?: unknown }).window = previous;
    }
  };
}

test("the shared refresh drops the flag catalog and re-reads the hardware info", () => {
  const { hook, calls } = loadHook("idle");
  hook.afterLlamaInstallChanged();
  assert.deepEqual(calls, ["flags", "hardware"]);
  calls.length = 0;
  hook.afterLlamaFolderChanged();
  assert.deepEqual(calls, ["flags", "hardware"]);
});

for (const state of ["success", "error"]) {
  test(
    `an update job that ends in ${state} refreshes the caches without re-reading the status`,
    withWindow(async () => {
      const { hook, calls } = loadHook(state, { runEffects: true });
      const done = hook.useLlamaUpdateCheck().apply();
      await flush();
      mock.timers.tick(500);
      await done;
      await flush();
      assert.deepEqual(calls, ["status", "flags", "hardware"]);
    }),
  );
}

test(
  "a folder change makes a mounted update check re-read the installed tag",
  withWindow(async () => {
    let installed: string | null = "b9585";
    const { hook, calls, committed, unmount } = loadHook("idle", {
      installedTag: () => installed,
      runEffects: true,
    });
    hook.useLlamaUpdateCheck({ passive: true });
    await flush();
    assert.deepEqual(committed, ["b9585"]);

    // A custom folder reports no managed release.
    installed = null;
    hook.afterLlamaFolderChanged();
    await flush();
    assert.deepEqual(committed, ["b9585", null]);

    unmount();
    calls.length = 0;
    hook.afterLlamaFolderChanged();
    await flush();
    assert.deepEqual(calls, ["flags", "hardware"]);
  }),
);

test(
  "a backend switch runs the shared refresh when its own job ends",
  withWindow(async () => {
    const calls: string[] = [];
    const switchJob = (state: string) => ({
      state,
      operation: "switch",
      startedAt: "2026-11-25T10:00:00Z",
      message: "",
      error: null,
    });
    let current = switchJob("idle");
    const { react } = reactStub({
      runEffects: true,
      states: [{ backendRequest: "cuda", backend: "cpu", job: current }],
    });
    const { useLlamaBackendSwitch } = loadWithStubs<typeof SwitchModule>(
      new URL(
        "../src/features/settings/hooks/use-llama-backend-switch.ts",
        import.meta.url,
      ),
      {
        react,
        "@/hooks/use-llama-update-check": {
          afterLlamaInstallChanged: () => calls.push("refresh"),
        },
        "@/i18n": { useT: () => (key: string) => key },
        "@/lib/llama-job-events": {
          signalLlamaJobStarted() {},
          subscribeToLlamaJobStarted: () => () => {},
        },
        "@/lib/llama-job-lifecycle": lifecycle,
        "@/lib/toast": {
          toast: {
            success: (title: string) => calls.push(title),
            error: (title: string) => calls.push(title),
          },
        },
        "../api/llama-backend": {
          loadLlamaBackendStatus: async () => ({
            backendRequest: "cuda",
            backend: "cuda",
            job: current,
          }),
          switchLlamaBackend: async () => {
            current = switchJob("running");
            return { started: true, job: current };
          },
        },
        "../lib/llama-backend-labels": { backendDisplayName: (b: string) => b },
      },
    );
    useLlamaBackendSwitch().apply();
    await flush();
    assert.deepEqual(calls, []);

    current = switchJob("success");
    mock.timers.tick(700);
    await flush();
    assert.deepEqual(calls, [
      "refresh",
      "settings.resources.llamaBackend.switchedTo",
    ]);
  }),
);

test(
  "closing Settings during the folder-change status read starts no poll",
  withWindow(async () => {
    const { hook, calls, unmount } = loadHook("running", { runEffects: true });
    hook.useLlamaUpdateCheck();
    hook.afterLlamaFolderChanged();
    unmount();
    await flush();
    mock.timers.tick(1500);
    await flush();
    assert.deepEqual(calls, ["flags", "hardware", "status"]);
  }),
);

test("saving or resetting the llama.cpp folder runs the folder refresh", () => {
  const section = readSrc(
    "features/settings/components/llama-backend-section.tsx",
  );
  const row = section.slice(section.indexOf("<LlamaCppPathRow"));
  assert.match(
    row,
    /^<LlamaCppPathRow\s+onChanged=\{\(\) => \{\s*afterLlamaFolderChanged\(\);\s*void refresh\(true\);\s*\}\}/,
  );
});
