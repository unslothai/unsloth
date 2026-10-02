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

async function until(condition: () => boolean): Promise<void> {
  for (let i = 0; i < 300 && !condition(); i++) {
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  assert.ok(condition());
}

test("a broken stream follows the pull the backend still runs, and the quit warning holds", async () => {
  const { store, pulls, listings, toasts, activity, running } = harness();
  const job = store.followNpuDownload("qwen3.6-moe-35b-a3b-FLM");
  pulls[0].onProgress({ event: "progress", percent: 12 });
  running.push({ model: "qwen3.6-moe-35b-a3b-FLM", percent: 30 });
  pulls[0].finish(new Error("network error"));
  await until(() => pulls.length === 2);
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

test("a backend that cannot be asked yet is followed again, not given up on", async () => {
  const { store, pulls, toasts, activity, backend } = harness();
  void store.followNpuDownload("qwen3-8b-FLM");
  backend.reachable = false;
  pulls[0].finish(new TypeError("Failed to fetch"));
  await until(() => pulls.length === 2);
  assert.equal(pulls[1].follow, true);
  assert.deepEqual(toasts, []);
  assert.equal(activity.at(-1), true);
});

test("replayed progress does not reset the reconnect limit", async () => {
  const { store, pulls, listings, toasts, running } = harness();
  running.push({ model: "lfm2-1.2b-FLM", percent: 20 });
  const job = store.followNpuDownload("lfm2-1.2b-FLM");
  pulls[0].onProgress({ event: "progress", percent: 20 });
  pulls[0].finish(new Error("network error"));
  // Each reconnect replays the same percent and breaks again: five are allowed after the
  // last one that moved it.
  for (let i = 1; i <= 6; i++) {
    await until(() => pulls.length === i + 1);
    pulls[i].onProgress({ event: "progress", percent: 20 });
    pulls[i].finish(new Error("network error"));
  }
  await until(() => listings.length === 1);
  listings[0].resolve([]);
  assert.equal(await job, false);
  assert.equal(pulls.length, 7);
  assert.deepEqual(toasts, ["Could not download lfm2-1.2b-FLM"]);
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

test("a complete event counts even if the list refresh after it goes stale", async () => {
  const { store, pulls, listings, running } = harness();
  running.push({ model: "qwen3-4b-FLM", percent: 90 });
  const job = store.followNpuDownload("qwen3-4b-FLM");
  pulls[0].finish(new Error("network error"));
  await until(() => pulls.length === 2);
  pulls[1].onProgress({ event: "complete", percent: 100 });
  pulls[1].finish();
  await until(() => listings.length === 1);
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
