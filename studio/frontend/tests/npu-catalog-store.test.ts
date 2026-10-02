// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import * as zustand from "zustand";
import type * as NpuCatalogStore from "../src/features/npu/npu-catalog-store.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

type Progress = (event: { event: string; percent?: number }) => void;
type Pull = {
  id: string;
  follow: boolean;
  onProgress: Progress;
  finish: (error?: Error) => void;
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

function harness() {
  const pulls: Pull[] = [];
  const listings: ReturnType<typeof deferred<{ id: string }[]>>[] = [];
  const toasts: string[] = [];
  const activity: boolean[] = [];
  // What GET /api/npu/downloads answers: the pulls the backend is still running.
  const running: { model: string; percent: number | null }[] = [];
  const backend = { reachable: true };
  const pull = (follow: boolean) => (id: string, onProgress: Progress) =>
    new Promise<void>((resolve, reject) => {
      pulls.push({
        id,
        follow,
        onProgress,
        finish: (error) => (error ? reject(error) : resolve()),
      });
    });
  const store = loadWithStubs<typeof NpuCatalogStore>(
    new URL("../src/features/npu/npu-catalog-store.ts", import.meta.url),
    {
      zustand,
      "@/lib/toast": {
        toast: { error: (title: string) => toasts.push(title) },
      },
      "@/lib/downloads-activity": {
        reportDownloadsActive: (source: string, active: boolean) => {
          if (source === "npu") activity.push(active);
        },
      },
      "./api": {
        NpuDownloadError,
        listNpuDownloads: async () => {
          if (!backend.reachable) throw new Error("Failed to fetch");
          return running;
        },
        downloadNpuModel: pull(false),
        followNpuModelDownload: pull(true),
        listNpuModels: () => {
          if (!backend.reachable) {
            return Promise.reject(new Error("Failed to fetch"));
          }
          const listing = deferred<{ id: string }[]>();
          listings.push(listing);
          return listing.promise;
        },
      },
    },
  );
  return { store, pulls, listings, toasts, activity, running, backend };
}

const tick = () => new Promise((resolve) => setImmediate(resolve));

class NpuDownloadError extends Error {}

type Clock = {
  enable: (options: { apis: "setTimeout"[] }) => void;
  tick: (ms: number) => void;
};

const settle = async () => {
  for (let i = 0; i < 20; i++) await tick();
};

/** Wait for `condition`, letting promises settle and moving a mocked clock through reconnect delays. */
async function until(condition: () => boolean, clock?: Clock): Promise<void> {
  for (let i = 0; i < 200 && !condition(); i++) {
    await tick();
    clock?.tick(1000);
  }
  assert.ok(condition());
}

/** Reconnects wait real seconds; tests run them on a mocked clock instead. */
function mockedClock(t: { mock: { timers: Clock } }): Clock {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  return t.mock.timers;
}

test("a broken stream follows the pull the backend still runs, and the quit warning holds", async (t) => {
  const clock = mockedClock(t);
  const { store, pulls, listings, toasts, activity, running } = harness();
  const job = store.followNpuDownload("qwen3.6-moe-35b-a3b-FLM");
  pulls[0].onProgress({ event: "progress", percent: 12 });
  running.push({ model: "qwen3.6-moe-35b-a3b-FLM", percent: 30 });
  pulls[0].finish(new Error("network error"));
  await until(() => pulls.length === 2, clock);
  assert.equal(pulls[1].follow, true);
  assert.deepEqual(toasts, []);
  assert.equal(activity.at(-1), true);
  assert.equal(
    store.useNpuCatalogStore.getState().progress["qwen3.6-moe-35b-a3b-FLM"],
    30,
  );
  pulls[1].finish();
  await tick();
  listings[0].resolve([
    { id: "qwen3.6-moe-35b-a3b-FLM", downloaded: true },
  ] as never);
  assert.equal(await job, true);
  assert.equal(activity.at(-1), false);
});

test("a backend that cannot be asked yet is followed again, not given up on", async (t) => {
  const clock = mockedClock(t);
  const { store, pulls, toasts, activity, backend } = harness();
  void store.followNpuDownload("qwen3-8b-FLM");
  backend.reachable = false;
  pulls[0].finish(new TypeError("Failed to fetch"));
  await until(() => pulls.length === 2, clock);
  assert.equal(pulls[1].follow, true);
  assert.deepEqual(toasts, []);
  assert.equal(activity.at(-1), true);
});

test("streams that keep breaking never end a pull the backend still runs", async (t) => {
  const clock = mockedClock(t);
  const { store, pulls, listings, toasts, activity, running } = harness();
  const id = "lfm2-1.2b-FLM";
  running.push({ model: id, percent: 25 });
  const job = store.followNpuDownload(id);
  pulls[0].finish(new Error("network error"));
  // Seven streams in a row break while the backend reports the pull moving from 25% to 50%.
  for (let i = 1; i <= 7; i++) {
    await until(() => pulls.length === i + 1, clock);
    assert.equal(
      store.useNpuCatalogStore.getState().progress[id],
      running[0].percent,
    );
    assert.equal(activity.at(-1), true);
    running[0].percent = Math.min(50, 25 + i * 5);
    pulls[i].finish(new Error("network error"));
  }
  await until(() => pulls.length === 9, clock);
  assert.deepEqual(toasts, []);
  assert.equal(store.useNpuCatalogStore.getState().progress[id], 50);
  // The pull ends on the backend; the list says it finished.
  running.length = 0;
  pulls[8].finish(new Error("network error"));
  await until(() => listings.length === 1, clock);
  listings[0].resolve([{ id, downloaded: true }] as never);
  assert.equal(await job, true);
  assert.equal(activity.at(-1), false);
});

test("a pull stalled on the backend is still followed, at most every 5 s", async (t) => {
  const clock = mockedClock(t);
  const { store, pulls, toasts, activity, running } = harness();
  const id = "gemma3-1b-FLM";
  running.push({ model: id, percent: 40 });
  void store.followNpuDownload(id);
  pulls[0].finish(new Error("network error"));
  for (let i = 1; i <= 7; i++) {
    await until(() => pulls.length === i + 1, clock);
    assert.equal(store.useNpuCatalogStore.getState().progress[id], 40);
    assert.equal(activity.at(-1), true);
    pulls[i].finish(new Error("network error"));
  }
  await settle();
  clock.tick(4999);
  await settle();
  assert.equal(pulls.length, 8);
  clock.tick(1);
  await settle();
  assert.equal(pulls.length, 9);
  assert.deepEqual(toasts, []);
});

test("an unreachable backend leaves the pull active and reconnecting until it answers", async (t) => {
  const clock = mockedClock(t);
  const { store, pulls, listings, toasts, activity, running, backend } =
    harness();
  const id = "qwen3-it-4b-FLM";
  running.push({ model: id, percent: 40 });
  const job = store.followNpuDownload(id);
  pulls[0].onProgress({ event: "progress", percent: 40 });
  // The connection drops: the stream and every check fail, many times over.
  backend.reachable = false;
  for (let i = 0; i < 8; i++) {
    await until(() => pulls.length === i + 1, clock);
    pulls[i].finish(new TypeError("Failed to fetch"));
    await settle();
    const state = store.useNpuCatalogStore.getState();
    assert.equal(state.progress[id], 40);
    assert.equal(id in state.reconnecting, true);
    assert.equal(activity.at(-1), true);
  }
  assert.deepEqual(toasts, []);
  // Connectivity returns while the backend is still downloading; the first stream breaks again
  // before any event, so the listing alone has to end the reconnecting state.
  backend.reachable = true;
  running[0].percent = 70;
  await until(() => pulls.length === 9, clock);
  pulls[8].finish(new Error("network error"));
  await settle();
  assert.equal(id in store.useNpuCatalogStore.getState().reconnecting, false);
  assert.equal(store.useNpuCatalogStore.getState().progress[id], 70);
  await until(() => pulls.length === 10, clock);
  pulls[9].onProgress({ event: "progress", percent: 72 });
  assert.equal(id in store.useNpuCatalogStore.getState().reconnecting, false);
  assert.equal(store.useNpuCatalogStore.getState().progress[id], 72);
  pulls[9].onProgress({ event: "complete", percent: 100 });
  pulls[9].finish();
  await until(() => listings.length === 1, clock);
  listings[0].resolve([]);
  assert.equal(await job, true);
  assert.equal(activity.at(-1), false);
  assert.deepEqual(store.useNpuCatalogStore.getState().reconnecting, {});
});

test("a pull lost while the backend was unreachable is reported when it answers", async (t) => {
  const clock = mockedClock(t);
  const { store, pulls, listings, toasts, activity, backend } = harness();
  const id = "deepseek-r1-8b-FLM";
  const job = store.followNpuDownload(id);
  backend.reachable = false;
  pulls[0].finish(new TypeError("Failed to fetch"));
  await until(() => pulls.length === 2, clock);
  // Studio restarted meanwhile: the follow finds no pull (a 404 resolves the stream).
  backend.reachable = true;
  pulls[1].finish();
  await until(() => listings.length === 1, clock);
  listings[0].resolve([{ id, downloaded: false }] as never);
  assert.equal(await job, false);
  assert.deepEqual(toasts, [`Could not download ${id}`]);
  const state = store.useNpuCatalogStore.getState();
  assert.deepEqual(state.progress, {});
  assert.deepEqual(state.reconnecting, {});
  assert.equal(activity.at(-1), false);
});

test("a pull that finished while its stream was broken is not reported failed", async () => {
  const { store, pulls, listings, toasts } = harness();
  const job = store.followNpuDownload("phi4-mini-it-4b-FLM");
  pulls[0].finish(new Error("network error"));
  await until(() => listings.length === 1);
  listings[0].resolve([
    { id: "phi4-mini-it-4b-FLM", downloaded: true },
  ] as never);
  assert.equal(await job, true);
  assert.deepEqual(toasts, []);
});

test("a complete event counts even if the list refresh after it goes stale", async (t) => {
  const clock = mockedClock(t);
  const { store, pulls, listings, running } = harness();
  running.push({ model: "qwen3-4b-FLM", percent: 90 });
  const job = store.followNpuDownload("qwen3-4b-FLM");
  pulls[0].finish(new Error("network error"));
  await until(() => pulls.length === 2, clock);
  pulls[1].onProgress({ event: "complete", percent: 100 });
  pulls[1].finish();
  await until(() => listings.length === 1, clock);
  listings[0].resolve([]);
  assert.equal(await job, true);
});

test("a download the backend reports failed is not followed again", async () => {
  const { store, pulls, listings, toasts, running } = harness();
  running.push({ model: "gemma3-4b-FLM", percent: 50 });
  const job = store.followNpuDownload("gemma3-4b-FLM");
  pulls[0].finish(new NpuDownloadError("disk full"));
  await until(() => listings.length === 1);
  listings[0].resolve([]);
  assert.equal(await job, false);
  assert.equal(pulls.length, 1);
  assert.deepEqual(toasts, ["Could not download gemma3-4b-FLM"]);
});

test("progress outlives the picker and one stream serves every follower", async () => {
  const { store, pulls, listings } = harness();
  const first = store.followNpuDownload("qwen3-0.6b-FLM");
  // A remounted picker asking again reuses the running stream.
  assert.equal(store.followNpuDownload("qwen3-0.6b-FLM"), first);
  assert.equal(pulls.length, 1);
  assert.equal(pulls[0].follow, false);
  pulls[0].onProgress({ event: "progress", percent: 42 });
  assert.equal(
    store.useNpuCatalogStore.getState().progress["qwen3-0.6b-FLM"],
    42,
  );
  // downloadNpuModel resolves only after the stream's complete event.
  pulls[0].onProgress({ event: "complete", percent: 100 });
  pulls[0].finish();
  await tick();
  // Still downloading until the refreshed list says otherwise, so the row never flickers back.
  assert.equal(
    "qwen3-0.6b-FLM" in store.useNpuCatalogStore.getState().progress,
    true,
  );
  listings[0].resolve([{ id: "qwen3-0.6b-FLM" }]);
  assert.equal(await first, true);
  const state = store.useNpuCatalogStore.getState();
  assert.deepEqual(state.progress, {});
  assert.deepEqual(state.models, [{ id: "qwen3-0.6b-FLM" }]);
});

test("a repeated percent leaves the store untouched", () => {
  const { store, pulls } = harness();
  void store.followNpuDownload("gemma3-4b-FLM");
  pulls[0].onProgress({ event: "progress", percent: 7 });
  let notified = 0;
  const stop = store.useNpuCatalogStore.subscribe(() => notified++);
  pulls[0].onProgress({ event: "progress", percent: 7 });
  pulls[0].onProgress({ event: "progress" });
  stop();
  assert.equal(notified, 0);
});

test("following a backend pull never starts one and opens at its last percent", () => {
  const { store, pulls } = harness();
  void store.followNpuDownload("gemma3-4b-FLM", { follow: true, percent: 61 });
  assert.equal(pulls[0].follow, true);
  assert.equal(
    store.useNpuCatalogStore.getState().progress["gemma3-4b-FLM"],
    61,
  );
});

test("a followed pull that ended before its stream opened reports how it ended", async () => {
  const { store, pulls, listings } = harness();
  const failed = store.followNpuDownload("gemma3-4b-FLM", { follow: true });
  // The backend answers 404 once the pull is over, which the API reads as the stream ending.
  pulls[0].finish();
  await tick();
  listings[0].resolve([{ id: "gemma3-4b-FLM", downloaded: false }] as never);
  assert.equal(await failed, false);
  const finished = store.followNpuDownload("gemma3-4b-FLM", { follow: true });
  pulls[1].finish();
  await tick();
  listings[1].resolve([{ id: "gemma3-4b-FLM", downloaded: true }] as never);
  assert.equal(await finished, true);
});

test("a failed pull is reported once, clears its progress, and the next try streams anew", async () => {
  const { store, pulls, listings, toasts } = harness();
  const job = store.followNpuDownload("llama3.2-1b-FLM");
  void store.followNpuDownload("llama3.2-1b-FLM");
  pulls[0].finish(new Error("connection reset"));
  // Relisted, so the row offers to resume from what the failed pull kept.
  await until(() => listings.length === 1);
  listings[0].resolve([]);
  assert.equal(await job, false);
  assert.deepEqual(toasts, ["Could not download llama3.2-1b-FLM"]);
  assert.deepEqual(store.useNpuCatalogStore.getState().progress, {});
  void store.followNpuDownload("llama3.2-1b-FLM");
  assert.equal(pulls.length, 2);
});

test("an older list answer never overwrites a newer one", async () => {
  const { store, listings } = harness();
  const older = store.refreshNpuModels();
  const newer = store.refreshNpuModels();
  listings[1].resolve([{ id: "downloaded" }]);
  await newer;
  listings[0].resolve([]);
  await older;
  assert.deepEqual(store.useNpuCatalogStore.getState().models, [
    { id: "downloaded" },
  ]);
});

test("a running pull counts as a download for the desktop quit warning", async () => {
  const { store, pulls, listings, activity } = harness();
  const job = store.followNpuDownload("qwen3-0.6b-FLM");
  assert.equal(activity.at(-1), true);
  pulls[0].finish();
  await tick();
  listings[0].resolve([]);
  await job;
  assert.equal(activity.at(-1), false);
});
