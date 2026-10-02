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
  return { store, pulls, listings, toasts, activity };
}

const tick = () => new Promise((resolve) => setImmediate(resolve));

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
  assert.equal(await job, false);
  assert.deepEqual(toasts, ["Could not download llama3.2-1b-FLM"]);
  // Relisted, so the row offers to resume from what the failed pull kept.
  assert.equal(listings.length, 1);
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
