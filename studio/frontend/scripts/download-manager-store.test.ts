// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { registerHooks } from "node:module";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

const srcRoot = resolve(import.meta.dirname, "../src");
const fetchSymbol = Symbol.for("unsloth.downloadManagerTestFetch");
const hfTokenSymbol = Symbol.for("unsloth.downloadManagerTestHfToken");
const inventoryBumpSymbol = Symbol.for(
  "unsloth.downloadManagerTestInventoryBump",
);

function dataModule(source: string): string {
  return `data:text/javascript;charset=utf-8,${encodeURIComponent(source)}`;
}

const mockModules = new Map<string, string>([
  [
    "@/features/auth",
    dataModule(`
      const fetchSymbol = Symbol.for("unsloth.downloadManagerTestFetch");
      export function hfTokenHeader(token) {
        return token ? { "X-HF-Token": token } : {};
      }
      export async function authFetch(input, init) {
        return globalThis[fetchSymbol](input, init);
      }
    `),
  ],
  [
    "@/lib/toast",
    dataModule(`
      export const toast = {
        error() {},
        info() {},
        success() {},
        warning() {},
      };
    `),
  ],
  [
    "@/stores/hf-token-store",
    dataModule(`
      const hfTokenSymbol = Symbol.for("unsloth.downloadManagerTestHfToken");
      export function getHfToken() { return globalThis[hfTokenSymbol] || ""; }
    `),
  ],
  [
    "@/stores/inventory-events",
    dataModule(`
      const inventoryBumpSymbol = Symbol.for(
        "unsloth.downloadManagerTestInventoryBump",
      );
	      export function bumpInventoryVersion() {
	        globalThis[inventoryBumpSymbol] = (globalThis[inventoryBumpSymbol] || 0) + 1;
	      }
	      export function getInventoryVersion() {
	        return globalThis[inventoryBumpSymbol] || 0;
	      }
	      export function notifyInventoryChanged() {
	        bumpInventoryVersion();
	      }
	    `),
  ],
]);

function resolveLocalModule(base: string, specifier: string): string | null {
  const target = resolve(base, specifier);
  const candidates = [
    target,
    `${target}.ts`,
    `${target}.tsx`,
    resolve(target, "index.ts"),
    resolve(target, "index.tsx"),
  ];
  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return pathToFileURL(candidate).href;
    }
  }
  return null;
}

registerHooks({
  resolve(specifier, context, nextResolve) {
    const mock = mockModules.get(specifier);
    if (mock) {
      return { shortCircuit: true, url: mock };
    }
    if (specifier.startsWith("@/")) {
      const url = resolveLocalModule(srcRoot, specifier.slice(2));
      if (url) return { shortCircuit: true, url };
    }
    if (specifier.startsWith(".") && context.parentURL?.startsWith("file:")) {
      const url = resolveLocalModule(
        dirname(fileURLToPath(context.parentURL)),
        specifier,
      );
      if (url) return { shortCircuit: true, url };
    }
    return nextResolve(specifier, context);
  },
});

class MemoryStorage implements Storage {
  private readonly map = new Map<string, string>();

  get length(): number {
    return this.map.size;
  }

  clear(): void {
    this.map.clear();
  }

  getItem(key: string): string | null {
    return this.map.get(key) ?? null;
  }

  key(index: number): string | null {
    return Array.from(this.map.keys())[index] ?? null;
  }

  removeItem(key: string): void {
    this.map.delete(key);
  }

  setItem(key: string, value: string): void {
    this.map.set(key, value);
  }
}

type CapturedTimer = {
  id: number;
  delay: number;
  callback: () => void;
};

function installBrowserShim(): Map<number, CapturedTimer> {
  const timers = new Map<number, CapturedTimer>();
  const storage = new MemoryStorage();
  let nextTimerId = 1;
  const windowShim = {
    localStorage: storage,
    setTimeout(callback: () => void, delay = 0): number {
      const id = nextTimerId;
      nextTimerId += 1;
      timers.set(id, { id, delay, callback });
      return id;
    },
    clearTimeout(id: number): void {
      timers.delete(id);
    },
  };
  Object.defineProperty(globalThis, "window", {
    value: windowShim,
    configurable: true,
  });
  Object.defineProperty(globalThis, "localStorage", {
    value: storage,
    configurable: true,
  });
  Object.defineProperty(globalThis, "document", {
    value: { hidden: false },
    configurable: true,
  });
  return timers;
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

function flushMicrotasks(): Promise<void> {
  return new Promise((resolve) => {
    queueMicrotask(resolve);
  });
}

function flushAsyncWork(): Promise<void> {
  return new Promise((resolve) => {
    setImmediate(resolve);
  });
}

function setTestHfToken(token: string): void {
  Object.defineProperty(globalThis, hfTokenSymbol, {
    value: token,
    configurable: true,
    writable: true,
  });
}

function resetInventoryBumps(): void {
  Object.defineProperty(globalThis, inventoryBumpSymbol, {
    value: 0,
    configurable: true,
    writable: true,
  });
}

function inventoryBumpCount(): number {
  const value = Reflect.get(globalThis, inventoryBumpSymbol);
  return typeof value === "number" ? value : 0;
}

function fireNextPollTimer(timers: Map<number, CapturedTimer>): CapturedTimer {
  const timer = Array.from(timers.values())
    .filter((entry) => entry.delay >= 100 && entry.delay <= 2_000)
    .sort((a, b) => a.id - b.id)[0];
  assert.ok(timer);
  timers.delete(timer.id);
  timer.callback();
  return timer;
}

function headerValue(
  init: RequestInit | undefined,
  name: string,
): string | null {
  const headers = init?.headers;
  if (!headers) return null;
  if (headers instanceof Headers) return headers.get(name);
  if (Array.isArray(headers)) {
    const pair = headers.find(
      ([key]) => key.toLowerCase() === name.toLowerCase(),
    );
    return pair?.[1] ?? null;
  }
  return (headers as Record<string, string>)[name] ?? null;
}

function requestBody(init: RequestInit | undefined): Record<string, unknown> {
  return JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
}

test("stale cancel watchdog and response cannot finalize a replacement download", async () => {
  const timers = installBrowserShim();
  let resolveCancel: (() => void) | null = null;
  let cancelSignal: AbortSignal | undefined;
  let startGeneration = 0;
  let cancelPayload: Record<string, unknown> | null = null;
  const fetchDownloadApi: typeof fetch = (input, init) => {
    const raw =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.href
          : input.url;
    const url = new URL(raw, "http://studio.test");
    const method = init?.method ?? "GET";

    if (method === "POST" && url.pathname === "/api/models/download") {
      startGeneration += 1;
      return Promise.resolve(
        jsonResponse({
          accepted: true,
          state: "running",
          job_key: "model",
          generation: startGeneration,
        }),
      );
    }
    if (method === "POST" && url.pathname === "/api/models/download/cancel") {
      cancelSignal = init?.signal ?? undefined;
      cancelPayload = requestBody(init);
      return new Promise<Response>((resolveRequest) => {
        resolveCancel = () => {
          resolveRequest(
            jsonResponse({ job_key: "model", state: "cancelling" }),
          );
        };
      });
    }
    if (method === "GET" && url.pathname === "/api/models/download-status") {
      return Promise.resolve(jsonResponse({ state: "running" }));
    }
    if (method === "GET" && url.pathname === "/api/models/download-progress") {
      return Promise.resolve(
        jsonResponse({
          downloaded_bytes: 1,
          expected_bytes: 100,
          progress: 0.01,
          cache_path: null,
        }),
      );
    }
    return Promise.resolve(jsonResponse({}));
  };
  Object.defineProperty(globalThis, fetchSymbol, {
    value: fetchDownloadApi,
    configurable: true,
  });

  const { downloadManager, jobKeyOf, useDownloadManagerStore } = await import(
    "../src/features/download-jobs/download-manager-store.ts"
  );
  useDownloadManagerStore.setState({
    jobs: {},
    conflicts: {},
    completedHintSignature: "",
    completedInventoryHints: [],
  });

  const req = {
    kind: "model" as const,
    repoId: "Org/Model",
    variant: null,
    expectedBytes: 100,
  };
  const key = jobKeyOf(req.kind, req.repoId, req.variant);

  await downloadManager.start(req, { useXet: false });
  await flushMicrotasks();
  const cancelPromise = downloadManager.cancel(key);
  const watchdog = Array.from(timers.values()).find(
    (timer) => timer.delay === 20_000,
  );
  assert.ok(watchdog);
  assert.ok(cancelSignal);
  assert.equal(cancelPayload?.generation, 1);
  assert.equal(cancelSignal.aborted, false);

  await downloadManager.start(req, { useXet: false });
  await flushMicrotasks();
  assert.equal(cancelSignal.aborted, true);
  watchdog.callback();
  assert.equal(useDownloadManagerStore.getState().jobs[key]?.state, "running");

  assert.ok(resolveCancel);
  resolveCancel();
  await cancelPromise;
  assert.equal(useDownloadManagerStore.getState().jobs[key]?.state, "running");
  await flushAsyncWork();
  useDownloadManagerStore.setState({
    jobs: {},
    conflicts: {},
    completedHintSignature: "",
    completedInventoryHints: [],
  });
});

test("status ticks can run without progress scans and progress uses the current HF token", async () => {
  const timers = installBrowserShim();
  let now = 0;
  const originalDateNow = Date.now;
  Object.defineProperty(Date, "now", {
    value: () => now,
    configurable: true,
  });
  setTestHfToken("old-token");

  const startTokens: (string | null)[] = [];
  const progressTokens: (string | null)[] = [];
  const statusSignals: (AbortSignal | null)[] = [];
  const progressSignals: (AbortSignal | null)[] = [];
  let statusCalls = 0;
  let progressCalls = 0;
  const fetchDownloadApi: typeof fetch = (input, init) => {
    const raw =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.href
          : input.url;
    const url = new URL(raw, "http://studio.test");
    const method = init?.method ?? "GET";

    if (method === "POST" && url.pathname === "/api/models/download") {
      startTokens.push(headerValue(init, "X-HF-Token"));
      return Promise.resolve(
        jsonResponse({ accepted: true, state: "running", job_key: "model" }),
      );
    }
    if (method === "GET" && url.pathname === "/api/models/download-status") {
      statusCalls += 1;
      statusSignals.push(init?.signal ?? null);
      return Promise.resolve(jsonResponse({ state: "running" }));
    }
    if (method === "GET" && url.pathname === "/api/models/download-progress") {
      progressCalls += 1;
      progressTokens.push(headerValue(init, "X-HF-Token"));
      progressSignals.push(init?.signal ?? null);
      return Promise.resolve(
        jsonResponse({
          downloaded_bytes: progressCalls * 10,
          expected_bytes: 100,
          progress: progressCalls / 10,
          cache_path: null,
        }),
      );
    }
    return Promise.resolve(jsonResponse({}));
  };
  Object.defineProperty(globalThis, fetchSymbol, {
    value: fetchDownloadApi,
    configurable: true,
  });

  try {
    const {
      downloadManager,
      jobKeyOf,
      repoKeyOf,
      selectActiveJob,
      useDownloadManagerStore,
    } = await import("../src/features/download-jobs/download-manager-store.ts");
    useDownloadManagerStore.setState({
      jobs: {},
      conflicts: {},
      completedHintSignature: "",
      completedInventoryHints: [],
    });

    const req = {
      kind: "model" as const,
      repoId: "Org/Cadence",
      variant: null,
      expectedBytes: 100,
    };
    const key = jobKeyOf(req.kind, req.repoId, req.variant);

    await downloadManager.start(req, { useXet: false });
    await flushAsyncWork();
    assert.deepEqual(startTokens, ["old-token"]);
    assert.equal(statusCalls, 1);
    assert.equal(progressCalls, 1);
    assert.deepEqual(progressTokens, ["old-token"]);
    assert.notEqual(statusSignals[0], progressSignals[0]);
    assert.equal(
      useDownloadManagerStore.getState().activeJobKeyByRepoKey[
        repoKeyOf(req.kind, req.repoId)
      ],
      key,
    );
    assert.equal(
      selectActiveJob(
        useDownloadManagerStore.getState(),
        req.kind,
        req.repoId,
      )?.key,
      key,
    );
    const jobSnapshot = useDownloadManagerStore.getState().jobs[key] ?? {};
    assert.equal(Object.hasOwn(jobSnapshot, "hfToken"), false);
    assert.equal(Object.hasOwn(jobSnapshot, "hfTokenFingerprint"), false);

    setTestHfToken("new-token");
    now = 500;
    fireNextPollTimer(timers);
    await flushAsyncWork();
    assert.equal(statusCalls, 2);
    assert.equal(progressCalls, 1);

    now = 1_000;
    fireNextPollTimer(timers);
    await flushAsyncWork();
    assert.equal(statusCalls, 3);
    assert.equal(progressCalls, 2);
    assert.deepEqual(progressTokens, ["old-token", "new-token"]);

    useDownloadManagerStore.setState((state) => {
      const jobs = { ...state.jobs };
      delete jobs[key];
      return { ...state, jobs };
    });
  } finally {
    Object.defineProperty(Date, "now", {
      value: originalDateNow,
      configurable: true,
    });
    setTestHfToken("");
  }
});

test("poll terminal error keeps the last failed request reason", async () => {
  const timers = installBrowserShim();
  let statusCalls = 0;
  const fetchDownloadApi: typeof fetch = (input, init) => {
    const raw =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.href
          : input.url;
    const url = new URL(raw, "http://studio.test");
    const method = init?.method ?? "GET";

    if (method === "POST" && url.pathname === "/api/models/download") {
      return Promise.resolve(
        jsonResponse({ accepted: true, state: "running", job_key: "model" }),
      );
    }
    if (method === "GET" && url.pathname === "/api/models/download-status") {
      statusCalls += 1;
      return Promise.reject(new Error(`backend unavailable ${statusCalls}`));
    }
    return Promise.resolve(jsonResponse({}));
  };
  Object.defineProperty(globalThis, fetchSymbol, {
    value: fetchDownloadApi,
    configurable: true,
  });

  const {
    __resetDownloadManagerForTests,
    downloadManager,
    jobKeyOf,
    useDownloadManagerStore,
  } = await import("../src/features/download-jobs/download-manager-store.ts");
  __resetDownloadManagerForTests();

  const req = {
    kind: "model" as const,
    repoId: "Org/PollFailure",
    variant: null,
    expectedBytes: 100,
  };
  const key = jobKeyOf(req.kind, req.repoId, req.variant);

  await downloadManager.start(req, { useXet: false });
  await flushAsyncWork();
  for (let i = 0; i < 5; i += 1) {
    fireNextPollTimer(timers);
    await flushAsyncWork();
  }

  const job = useDownloadManagerStore.getState().jobs[key];
  assert.equal(statusCalls, 6);
  assert.equal(job?.state, "error");
  assert.match(job?.error ?? "", /Lost contact with the download/);
  assert.match(job?.error ?? "", /backend unavailable 6/);

  __resetDownloadManagerForTests();
});

test("download manager keeps one active variant per model repo", async () => {
  installBrowserShim();
  const startBodies: Record<string, unknown>[] = [];
  let transportStatusCalls = 0;
  const fetchDownloadApi: typeof fetch = (input, init) => {
    const raw =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.href
          : input.url;
    const url = new URL(raw, "http://studio.test");
    const method = init?.method ?? "GET";

    if (method === "POST" && url.pathname === "/api/models/download") {
      startBodies.push(requestBody(init));
      return Promise.resolve(
        jsonResponse({ accepted: true, state: "running", job_key: "model" }),
      );
    }
    if (method === "GET" && url.pathname === "/api/models/transport-status") {
      transportStatusCalls += 1;
      return Promise.resolve(
        jsonResponse({
          has_partial: false,
          last_transport: null,
          resumable: false,
        }),
      );
    }
    if (method === "GET" && url.pathname === "/api/models/download-status") {
      return Promise.resolve(jsonResponse({ state: "running" }));
    }
    if (
      method === "GET" &&
      (url.pathname === "/api/models/download-progress" ||
        url.pathname === "/api/models/gguf-download-progress")
    ) {
      return Promise.resolve(
        jsonResponse({
          downloaded_bytes: 0,
          expected_bytes: 100,
          progress: 0,
          cache_path: null,
        }),
      );
    }
    return Promise.resolve(jsonResponse({}));
  };
  Object.defineProperty(globalThis, fetchSymbol, {
    value: fetchDownloadApi,
    configurable: true,
  });

  const {
    __resetDownloadManagerForTests,
    downloadManager,
    jobKeyOf,
    useDownloadManagerStore,
  } = await import("../src/features/download-jobs/download-manager-store.ts");
  __resetDownloadManagerForTests();

  try {
    const repoId = "Org/MultiVariant";
    const first = {
      kind: "model" as const,
      repoId,
      variant: "Q4_K_M",
      expectedBytes: 100,
    };
    const second = { ...first, variant: "Q5_K_M" };
    const third = { ...first, variant: "Q8_0" };
    const firstKey = jobKeyOf(first.kind, first.repoId, first.variant);
    const secondKey = jobKeyOf(second.kind, second.repoId, second.variant);

    await downloadManager.start(first, { useXet: false });
    await flushAsyncWork();
    await downloadManager.start(second, { useXet: false });
    await flushAsyncWork();
    await downloadManager.requestStart(third);
    await flushAsyncWork();

    const jobs = useDownloadManagerStore.getState().jobs;
    assert.equal(startBodies.length, 1);
    assert.equal(startBodies[0].gguf_variant, "Q4_K_M");
    assert.equal(transportStatusCalls, 0);
    assert.equal(jobs[firstKey]?.state, "running");
    assert.equal(jobs[secondKey], undefined);
  } finally {
    __resetDownloadManagerForTests();
  }
});

test("repo-level active job selection is deterministic when state is invalid", async () => {
  installBrowserShim();
  const {
    __resetDownloadManagerForTests,
    jobKeyOf,
    repoKeyOf,
    selectActiveJob,
    useDownloadManagerStore,
  } = await import("../src/features/download-jobs/download-manager-store.ts");
  __resetDownloadManagerForTests();

  const repoId = "Org/Corrupt";
  const olderKey = jobKeyOf("model", repoId, "Q4_K_M");
  const tiedWinnerKey = jobKeyOf("model", repoId, "Q5_K_M");
  const tiedLoserKey = jobKeyOf("model", repoId, "Q8_0");
  useDownloadManagerStore.setState({
    jobs: {
      [olderKey]: {
        key: olderKey,
        kind: "model",
        repoId,
        variant: "Q4_K_M",
        state: "running",
        downloadedBytes: 0,
        expectedBytes: 100,
        fraction: 0,
        bytesPerSec: 0,
        error: null,
        startedAt: 100,
      },
      [tiedWinnerKey]: {
        key: tiedWinnerKey,
        kind: "model",
        repoId,
        variant: "Q5_K_M",
        state: "running",
        downloadedBytes: 0,
        expectedBytes: 100,
        fraction: 0,
        bytesPerSec: 0,
        error: null,
        startedAt: 200,
      },
      [tiedLoserKey]: {
        key: tiedLoserKey,
        kind: "model",
        repoId,
        variant: "Q8_0",
        state: "running",
        downloadedBytes: 0,
        expectedBytes: 100,
        fraction: 0,
        bytesPerSec: 0,
        error: null,
        startedAt: 200,
      },
    },
    activeJobKeyByRepoKey: {
      [repoKeyOf("model", repoId)]: olderKey,
    },
    conflicts: {},
    completedHintSignature: "",
    completedInventoryHints: [],
  });

  try {
    const state = useDownloadManagerStore.getState();
    assert.equal(
      selectActiveJob(state, "model", repoId, "Q4_K_M")?.key,
      olderKey,
    );
    assert.equal(selectActiveJob(state, "model", repoId)?.key, tiedWinnerKey);
  } finally {
    __resetDownloadManagerForTests();
  }
});

test("terminal completion schedules one debounced inventory invalidation", async () => {
  const timers = installBrowserShim();
  resetInventoryBumps();
  const fetchDownloadApi: typeof fetch = (input, init) => {
    const raw =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.href
          : input.url;
    const url = new URL(raw, "http://studio.test");
    const method = init?.method ?? "GET";

    if (method === "POST" && url.pathname === "/api/models/download") {
      return Promise.resolve(
        jsonResponse({ accepted: true, state: "running", job_key: "model" }),
      );
    }
    if (method === "GET" && url.pathname === "/api/models/download-status") {
      return Promise.resolve(jsonResponse({ state: "complete" }));
    }
    return Promise.resolve(jsonResponse({}));
  };
  Object.defineProperty(globalThis, fetchSymbol, {
    value: fetchDownloadApi,
    configurable: true,
  });

  const { downloadManager, useDownloadManagerStore } = await import(
    "../src/features/download-jobs/download-manager-store.ts"
  );
  useDownloadManagerStore.setState({
    jobs: {},
    conflicts: {},
    completedHintSignature: "",
    completedInventoryHints: [],
  });

  await downloadManager.start(
    {
      kind: "model",
      repoId: "Org/Complete",
      variant: null,
      expectedBytes: 100,
    },
    { useXet: false },
  );
  await flushAsyncWork();
  assert.equal(inventoryBumpCount(), 0);

  const inventoryTimer = Array.from(timers.values()).find(
    (timer) => timer.delay === 250,
  );
  assert.ok(inventoryTimer);
  timers.delete(inventoryTimer.id);
  inventoryTimer.callback();
  assert.equal(inventoryBumpCount(), 1);
});

test("job listener notifications snapshot subscribers before dispatch", async () => {
  installBrowserShim();
  const fetchDownloadApi: typeof fetch = (input, init) => {
    const raw =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.href
          : input.url;
    const url = new URL(raw, "http://studio.test");
    const method = init?.method ?? "GET";

    if (method === "POST" && url.pathname === "/api/models/download") {
      return Promise.resolve(
        jsonResponse({ accepted: true, state: "running", job_key: "model" }),
      );
    }
    if (method === "GET" && url.pathname === "/api/models/download-status") {
      return Promise.resolve(jsonResponse({ state: "complete" }));
    }
    return Promise.resolve(jsonResponse({}));
  };
  Object.defineProperty(globalThis, fetchSymbol, {
    value: fetchDownloadApi,
    configurable: true,
  });

  const { downloadManager, subscribeJobListeners, useDownloadManagerStore } =
    await import("../src/features/download-jobs/download-manager-store.ts");
  useDownloadManagerStore.setState({
    jobs: {},
    conflicts: {},
    completedHintSignature: "",
    completedInventoryHints: [],
  });

  const calls: string[] = [];
  let unsubscribeSecond = () => {};
  const unsubscribeFirst = subscribeJobListeners("model", "Org/Snapshot", {
    onComplete: () => {
      calls.push("first");
      unsubscribeSecond();
    },
  });
  unsubscribeSecond = subscribeJobListeners("model", "Org/Snapshot", {
    onComplete: () => {
      calls.push("second");
    },
  });

  await downloadManager.start(
    {
      kind: "model",
      repoId: "Org/Snapshot",
      variant: null,
      expectedBytes: 100,
    },
    { useXet: false },
  );
  await flushAsyncWork();

  assert.deepEqual(calls, ["first", "second"]);
  unsubscribeFirst();
  unsubscribeSecond();
});

test("job listener errors warn and do not stop other listeners", async () => {
  installBrowserShim();
  const fetchDownloadApi: typeof fetch = (input, init) => {
    const raw =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.href
          : input.url;
    const url = new URL(raw, "http://studio.test");
    const method = init?.method ?? "GET";

    if (method === "POST" && url.pathname === "/api/models/download") {
      return Promise.resolve(
        jsonResponse({ accepted: true, state: "running", job_key: "model" }),
      );
    }
    if (method === "GET" && url.pathname === "/api/models/download-status") {
      return Promise.resolve(jsonResponse({ state: "complete" }));
    }
    return Promise.resolve(jsonResponse({}));
  };
  Object.defineProperty(globalThis, fetchSymbol, {
    value: fetchDownloadApi,
    configurable: true,
  });

  const originalWarn = console.warn;
  const warnings: unknown[][] = [];
  console.warn = (...args: unknown[]) => {
    warnings.push(args);
  };

  try {
    const { downloadManager, subscribeJobListeners, useDownloadManagerStore } =
      await import("../src/features/download-jobs/download-manager-store.ts");
    useDownloadManagerStore.setState({
      jobs: {},
      conflicts: {},
      completedHintSignature: "",
      completedInventoryHints: [],
    });

    const calls: string[] = [];
    const unsubscribeThrowing = subscribeJobListeners("model", "Org/Warn", {
      onComplete: () => {
        calls.push("throwing");
        throw new Error("listener failed");
      },
    });
    const unsubscribeSecond = subscribeJobListeners("model", "Org/Warn", {
      onComplete: () => {
        calls.push("second");
      },
    });

    await downloadManager.start(
      {
        kind: "model",
        repoId: "Org/Warn",
        variant: null,
        expectedBytes: 100,
      },
      { useXet: false },
    );
    await flushAsyncWork();

    assert.deepEqual(calls, ["throwing", "second"]);
    assert.equal(warnings.length, 1);
    assert.equal(warnings[0][0], "Download job listener failed");
    unsubscribeThrowing();
    unsubscribeSecond();
  } finally {
    console.warn = originalWarn;
  }
});

test("dismissing an active job tears down its runtime", async () => {
  installBrowserShim();
  let statusSignal: AbortSignal | undefined;
  let resolveStatus: ((value: Response) => void) | null = null;
  const fetchDownloadApi: typeof fetch = (input, init) => {
    const raw =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.href
          : input.url;
    const url = new URL(raw, "http://studio.test");
    const method = init?.method ?? "GET";

    if (method === "POST" && url.pathname === "/api/models/download") {
      return Promise.resolve(
        jsonResponse({ accepted: true, state: "running", job_key: "model" }),
      );
    }
    if (method === "GET" && url.pathname === "/api/models/download-status") {
      statusSignal = init?.signal ?? undefined;
      return new Promise<Response>((resolve) => {
        resolveStatus = resolve;
      });
    }
    return Promise.resolve(jsonResponse({}));
  };
  Object.defineProperty(globalThis, fetchSymbol, {
    value: fetchDownloadApi,
    configurable: true,
  });

  const { downloadManager, jobKeyOf, useDownloadManagerStore } = await import(
    "../src/features/download-jobs/download-manager-store.ts"
  );
  useDownloadManagerStore.setState({
    jobs: {},
    conflicts: {},
    completedHintSignature: "",
    completedInventoryHints: [],
  });

  const req = {
    kind: "model" as const,
    repoId: "Org/Dismiss",
    variant: null,
    expectedBytes: 100,
  };
  const key = jobKeyOf(req.kind, req.repoId, req.variant);

  await downloadManager.start(req, { useXet: false });
  assert.ok(statusSignal);
  assert.equal(statusSignal.aborted, false);

  downloadManager.dismiss(key);
  assert.equal(statusSignal.aborted, true);
  assert.equal(useDownloadManagerStore.getState().jobs[key], undefined);

  assert.ok(resolveStatus);
  resolveStatus(jsonResponse({ state: "running" }));
  await flushAsyncWork();
  assert.equal(useDownloadManagerStore.getState().jobs[key], undefined);
});

test("late expected byte updates do not mutate cancelling jobs", async () => {
  installBrowserShim();
  const { downloadManager, jobKeyOf, useDownloadManagerStore } = await import(
    "../src/features/download-jobs/download-manager-store.ts"
  );
  const key = jobKeyOf("model", "Org/Cancelling", null);
  useDownloadManagerStore.setState({
    jobs: {
      [key]: {
        key,
        kind: "model",
        repoId: "Org/Cancelling",
        variant: null,
        state: "cancelling",
        downloadedBytes: 5,
        expectedBytes: 10,
        fraction: 0.5,
        bytesPerSec: 0,
        error: null,
        startedAt: Date.now(),
      },
    },
    conflicts: {},
    completedHintSignature: "",
    completedInventoryHints: [],
  });

  downloadManager.setExpected("model", "Org/Cancelling", null, 100);

  const job = useDownloadManagerStore.getState().jobs[key];
  assert.equal(job?.expectedBytes, 10);
  assert.equal(job?.fraction, 0.5);
});

test("completed hint suppression resets when the same repo starts again", async () => {
  installBrowserShim();
  const fetchDownloadApi: typeof fetch = (input, init) => {
    const raw =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.href
          : input.url;
    const url = new URL(raw, "http://studio.test");
    const method = init?.method ?? "GET";

    if (method === "POST" && url.pathname === "/api/models/download") {
      return Promise.resolve(
        jsonResponse({ accepted: true, state: "running", job_key: "model" }),
      );
    }
    if (method === "GET" && url.pathname === "/api/models/download-status") {
      return Promise.resolve(jsonResponse({ state: "complete" }));
    }
    if (method === "GET" && url.pathname === "/api/models/download-progress") {
      return Promise.resolve(
        jsonResponse({
          downloaded_bytes: 100,
          expected_bytes: 100,
          progress: 1,
          cache_path: null,
        }),
      );
    }
    return Promise.resolve(jsonResponse({}));
  };
  Object.defineProperty(globalThis, fetchSymbol, {
    value: fetchDownloadApi,
    configurable: true,
  });

  const {
    clearCompletedInventoryHint,
    downloadManager,
    getCompletedInventoryHints,
    jobKeyOf,
    useDownloadManagerStore,
  } = await import("../src/features/download-jobs/download-manager-store.ts");

  const req = {
    kind: "model" as const,
    repoId: "Org/Model",
    variant: null,
    expectedBytes: 100,
  };
  const key = jobKeyOf(req.kind, req.repoId, req.variant);
  const completedHint = {
    kind: "model" as const,
    repoId: "Org/Model",
    bytes: 100,
  };
  useDownloadManagerStore.setState({
    jobs: {
      [key]: {
        key,
        kind: req.kind,
        repoId: req.repoId,
        variant: req.variant,
        state: "complete",
        downloadedBytes: 100,
        expectedBytes: 100,
        fraction: 1,
        bytesPerSec: 0,
        error: null,
        startedAt: Date.now(),
      },
    },
    conflicts: {},
    completedHintSignature: "model:Org/Model:100",
    completedInventoryHints: [completedHint],
  });

  clearCompletedInventoryHint(completedHint);
  assert.deepEqual(getCompletedInventoryHints(), []);

  await downloadManager.start(req, { useXet: false });
  await flushAsyncWork();

  assert.deepEqual(getCompletedInventoryHints(), [completedHint]);
});

test("completed download inventory hints converge after stale cache suppression", async () => {
  installBrowserShim();
  const fetchDownloadApi: typeof fetch = (input, init) => {
    const raw =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.href
          : input.url;
    const url = new URL(raw, "http://studio.test");
    const method = init?.method ?? "GET";

    if (method === "POST" && url.pathname === "/api/models/download") {
      return Promise.resolve(
        jsonResponse({ accepted: true, state: "running", job_key: "model" }),
      );
    }
    if (method === "GET" && url.pathname === "/api/models/download-status") {
      return Promise.resolve(jsonResponse({ state: "complete" }));
    }
    return Promise.resolve(jsonResponse({}));
  };
  Object.defineProperty(globalThis, fetchSymbol, {
    value: fetchDownloadApi,
    configurable: true,
  });

  const {
    __resetDownloadManagerForTests,
    clearCompletedInventoryHint,
    downloadManager,
    getCompletedInventoryHints,
  } = await import("../src/features/download-jobs/download-manager-store.ts");
  const {
    pendingWithInventoryHints,
    useInventoryHintStore,
  } = await import("../src/features/inventory/inventory-hint-store.ts");
  const {
    createPendingInventoryHints,
    reconcileInventoryHints,
  } = await import("../src/features/inventory/inventory-hints.ts");

  __resetDownloadManagerForTests();
  useInventoryHintStore.setState({
    pending: createPendingInventoryHints(),
    observedKeys: {
      model: new Set(["org/converge"]),
      gguf: new Set(),
      dataset: new Set(),
    },
  });

  await downloadManager.start(
    {
      kind: "model",
      repoId: "Org/Converge",
      variant: null,
      expectedBytes: 42,
    },
    { useXet: false },
  );
  await flushAsyncWork();

  const completedHints = getCompletedInventoryHints();
  assert.deepEqual(completedHints, [
    { kind: "model", repoId: "Org/Converge", bytes: 42 },
  ]);

  const baseline = pendingWithInventoryHints(
    useInventoryHintStore.getState().pending,
    completedHints,
  );
  const optimistic = reconcileInventoryHints({
    pending: baseline,
    kind: "model",
    rows: [],
    previouslyObserved: new Set(),
  });
  assert.deepEqual(optimistic.rows, [
    { repo_id: "Org/Converge", size_bytes: 42, partial: false },
  ]);

  const stale = reconcileInventoryHints({
    pending: baseline,
    kind: "model",
    rows: [],
    previouslyObserved: useInventoryHintStore.getState().observedKeys.model,
  });
  const suppress = useInventoryHintStore
    .getState()
    .commitReconciliations(completedHints, baseline, [
      { kind: "model", reconciliation: stale },
    ]);
  assert.deepEqual(suppress, completedHints);
  for (const hint of suppress) {
    clearCompletedInventoryHint(hint);
  }
  assert.deepEqual(getCompletedInventoryHints(), []);
  assert.equal(
    useInventoryHintStore.getState().pending.model.has("org/converge"),
    false,
  );

  const settledBaseline = pendingWithInventoryHints(
    useInventoryHintStore.getState().pending,
    getCompletedInventoryHints(),
  );
  const settled = reconcileInventoryHints({
    pending: settledBaseline,
    kind: "model",
    rows: [],
    previouslyObserved: useInventoryHintStore.getState().observedKeys.model,
  });
  const settledSuppress = useInventoryHintStore
    .getState()
    .commitReconciliations([], settledBaseline, [
      { kind: "model", reconciliation: settled },
    ]);

  assert.deepEqual(settledSuppress, []);
  assert.equal(
    useInventoryHintStore.getState().pending.model.has("org/converge"),
    false,
  );
  assert.deepEqual(getCompletedInventoryHints(), []);
  __resetDownloadManagerForTests();
});

test("transport conflicts resume with previous mode and restart with selected mode", async () => {
  installBrowserShim();
  localStorage.setItem("unsloth.studio.transportMode", "xet");
  const startBodies: Record<string, unknown>[] = [];
  const fetchDownloadApi: typeof fetch = (input, init) => {
    const raw =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.href
          : input.url;
    const url = new URL(raw, "http://studio.test");
    const method = init?.method ?? "GET";

    if (method === "GET" && url.pathname === "/api/models/transport-status") {
      return Promise.resolve(
        jsonResponse({
          has_partial: true,
          last_transport: "http",
          resumable: true,
        }),
      );
    }
    if (
      method === "GET" &&
      url.pathname === "/api/studio/download-transport-capabilities"
    ) {
      return Promise.resolve(
        jsonResponse({
          http: { available: true, reason: null },
          xet: { available: true, reason: null },
        }),
      );
    }
    if (method === "POST" && url.pathname === "/api/models/download") {
      startBodies.push(requestBody(init));
      return Promise.resolve(
        jsonResponse({ accepted: true, state: "running", job_key: "model" }),
      );
    }
    if (method === "GET" && url.pathname === "/api/models/download-status") {
      return Promise.resolve(jsonResponse({ state: "running" }));
    }
    if (method === "GET" && url.pathname === "/api/models/download-progress") {
      return Promise.resolve(
        jsonResponse({
          downloaded_bytes: 0,
          expected_bytes: 100,
          progress: 0,
          cache_path: null,
        }),
      );
    }
    return Promise.resolve(jsonResponse({}));
  };
  Object.defineProperty(globalThis, fetchSymbol, {
    value: fetchDownloadApi,
    configurable: true,
  });

  const { downloadManager, jobKeyOf, repoKeyOf, useDownloadManagerStore } =
    await import("../src/features/download-jobs/download-manager-store.ts");
  const { __resetDownloadTransportCapabilitiesForTests } = await import(
    "../src/features/download-jobs/api.ts"
  );
  __resetDownloadTransportCapabilitiesForTests();
  useDownloadManagerStore.setState({
    jobs: {},
    conflicts: {},
    completedHintSignature: "",
    completedInventoryHints: [],
  });
  const req = {
    kind: "model" as const,
    repoId: "Org/Conflict",
    variant: null,
    expectedBytes: 100,
  };
  const repoKey = repoKeyOf(req.kind, req.repoId);

  await downloadManager.requestStart(req);
  assert.equal(startBodies.length, 0);
  assert.equal(
    useDownloadManagerStore.getState().conflicts[repoKey]?.info.previous,
    "http",
  );

  downloadManager.resumeConflict(repoKey);
  await flushAsyncWork();
  assert.equal(startBodies[0].use_xet, false);
  downloadManager.dismiss(jobKeyOf(req.kind, req.repoId, req.variant));

  await downloadManager.requestStart(req);
  downloadManager.restartConflict(repoKey);
  await flushAsyncWork();
  assert.equal(startBodies[1].use_xet, true);
});

test("download manager downgrades Xet when backend capability is unavailable", async () => {
  installBrowserShim();
  localStorage.setItem("unsloth.studio.transportMode", "xet");
  const startBodies: Record<string, unknown>[] = [];
  const fetchDownloadApi: typeof fetch = (input, init) => {
    const raw =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.href
          : input.url;
    const url = new URL(raw, "http://studio.test");
    const method = init?.method ?? "GET";

    if (
      method === "GET" &&
      url.pathname === "/api/studio/download-transport-capabilities"
    ) {
      return Promise.resolve(
        jsonResponse({
          http: { available: true, reason: null },
          xet: { available: false, reason: "hf_xet is missing" },
        }),
      );
    }
    if (method === "GET" && url.pathname === "/api/models/transport-status") {
      return Promise.resolve(
        jsonResponse({
          has_partial: false,
          last_transport: null,
          resumable: false,
        }),
      );
    }
    if (method === "POST" && url.pathname === "/api/models/download") {
      startBodies.push(requestBody(init));
      return Promise.resolve(
        jsonResponse({ accepted: true, state: "running", job_key: "model" }),
      );
    }
    if (method === "GET" && url.pathname === "/api/models/download-status") {
      return Promise.resolve(jsonResponse({ state: "running" }));
    }
    if (method === "GET" && url.pathname === "/api/models/download-progress") {
      return Promise.resolve(
        jsonResponse({
          downloaded_bytes: 0,
          expected_bytes: 100,
          progress: 0,
          cache_path: null,
        }),
      );
    }
    return Promise.resolve(jsonResponse({}));
  };
  Object.defineProperty(globalThis, fetchSymbol, {
    value: fetchDownloadApi,
    configurable: true,
  });

  const { downloadManager, useDownloadManagerStore } = await import(
    "../src/features/download-jobs/download-manager-store.ts"
  );
  const { __resetDownloadTransportCapabilitiesForTests } = await import(
    "../src/features/download-jobs/api.ts"
  );
  __resetDownloadTransportCapabilitiesForTests();
  useDownloadManagerStore.setState({
    jobs: {},
    conflicts: {},
    completedHintSignature: "",
    completedInventoryHints: [],
  });

  await downloadManager.requestStart({
    kind: "model",
    repoId: "Org/XetUnavailable",
    variant: null,
    expectedBytes: 100,
  });
  await flushAsyncWork();

  assert.equal(startBodies.length, 1);
  assert.equal(startBodies[0].use_xet, false);
});

test("idle backend records evict active jobs after the grace window", async () => {
  const timers = installBrowserShim();
  let now = 0;
  const originalDateNow = Date.now;
  Object.defineProperty(Date, "now", {
    value: () => now,
    configurable: true,
  });
  const fetchDownloadApi: typeof fetch = (input, init) => {
    const raw =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.href
          : input.url;
    const url = new URL(raw, "http://studio.test");
    const method = init?.method ?? "GET";

    if (method === "POST" && url.pathname === "/api/models/download") {
      return Promise.resolve(
        jsonResponse({ accepted: true, state: "running", job_key: "model" }),
      );
    }
    if (method === "GET" && url.pathname === "/api/models/download-status") {
      return Promise.resolve(jsonResponse({ state: "idle" }));
    }
    if (method === "GET" && url.pathname === "/api/models/download-progress") {
      return Promise.resolve(
        jsonResponse({
          downloaded_bytes: 0,
          expected_bytes: 100,
          progress: 0,
          cache_path: null,
        }),
      );
    }
    return Promise.resolve(jsonResponse({}));
  };
  Object.defineProperty(globalThis, fetchSymbol, {
    value: fetchDownloadApi,
    configurable: true,
  });

  try {
    const { downloadManager, jobKeyOf, useDownloadManagerStore } = await import(
      "../src/features/download-jobs/download-manager-store.ts"
    );
    useDownloadManagerStore.setState({
      jobs: {},
      conflicts: {},
      completedHintSignature: "",
      completedInventoryHints: [],
    });
    const req = {
      kind: "model" as const,
      repoId: "Org/Idle",
      variant: null,
      expectedBytes: 100,
    };
    const key = jobKeyOf(req.kind, req.repoId, req.variant);

    await downloadManager.start(req, { useXet: false });
    await flushAsyncWork();
    assert.equal(
      useDownloadManagerStore.getState().jobs[key]?.state,
      "running",
    );

    now = 60_001;
    fireNextPollTimer(timers);
    await flushAsyncWork();
    assert.equal(useDownloadManagerStore.getState().jobs[key], undefined);
  } finally {
    Object.defineProperty(Date, "now", {
      value: originalDateNow,
      configurable: true,
    });
  }
});

test("hydrated active jobs settle from backend status before polling", async () => {
  installBrowserShim();
  const fetchDownloadApi: typeof fetch = (input, init) => {
    const raw =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.href
          : input.url;
    const url = new URL(raw, "http://studio.test");
    const method = init?.method ?? "GET";

    if (method === "GET" && url.pathname === "/api/models/download-status") {
      if (url.searchParams.get("repo_id") === "Org/HydrateIdle") {
        return Promise.resolve(jsonResponse({ state: "idle" }));
      }
      return Promise.resolve(jsonResponse({ state: "complete" }));
    }
    if (method === "GET" && url.pathname === "/api/models/download-progress") {
      return Promise.resolve(
        jsonResponse({
          downloaded_bytes: 25,
          expected_bytes: 100,
          progress: 0.25,
          cache_path: null,
        }),
      );
    }
    return Promise.resolve(jsonResponse({}));
  };
  Object.defineProperty(globalThis, fetchSymbol, {
    value: fetchDownloadApi,
    configurable: true,
  });

  const { hydrateDownloadManager, jobKeyOf, useDownloadManagerStore } =
    await import("../src/features/download-jobs/download-manager-store.ts");
  const req = {
    kind: "model" as const,
    repoId: "Org/Hydrate",
    variant: null,
    expectedBytes: 100,
  };
  const key = jobKeyOf(req.kind, req.repoId, req.variant);
  const idleReq = {
    kind: "model" as const,
    repoId: "Org/HydrateIdle",
    variant: null,
    expectedBytes: 100,
  };
  const idleKey = jobKeyOf(idleReq.kind, idleReq.repoId, idleReq.variant);
  useDownloadManagerStore.setState({
    jobs: {
      [key]: {
        key,
        kind: req.kind,
        repoId: req.repoId,
        variant: req.variant,
        state: "running",
        downloadedBytes: 100,
        expectedBytes: 100,
        fraction: 1,
        bytesPerSec: 0,
        error: null,
        startedAt: Date.now(),
      },
      [idleKey]: {
        key: idleKey,
        kind: idleReq.kind,
        repoId: idleReq.repoId,
        variant: idleReq.variant,
        state: "running",
        downloadedBytes: 0,
        expectedBytes: 100,
        fraction: 0,
        bytesPerSec: 0,
        error: null,
        startedAt: Date.now(),
      },
    },
    conflicts: {},
    completedHintSignature: "",
    completedInventoryHints: [],
  });

  hydrateDownloadManager();
  await flushAsyncWork();

  assert.equal(useDownloadManagerStore.getState().jobs[key]?.state, "complete");
  assert.deepEqual(useDownloadManagerStore.getState().completedInventoryHints, [
    { kind: "model", repoId: "Org/Hydrate", bytes: 100 },
  ]);
  const idleJob = useDownloadManagerStore.getState().jobs[idleKey];
  assert.equal(idleJob?.state, "running");
  assert.equal(idleJob?.downloadedBytes, 25);
  assert.equal(idleJob?.expectedBytes, 100);
});
