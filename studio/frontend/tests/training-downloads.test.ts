// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  TrainingDownloadDiscovery,
  trainingDownloads,
} from "../src/features/hub/download-manager/training-downloads.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";
import type { ActiveModelDownload } from "../src/features/hub/download-manager/api.ts";
import type { ManagedDownload } from "../src/features/hub/download-manager/download-manager-types.ts";

registerBundlerResolver();
const { resolveProgressUpdate } = await import(
  "../src/features/hub/download-manager/progress-reconcile.ts"
);

const job: ActiveModelDownload = {
  repo_id: "unsloth/FLUX.1-dev",
  variant: `@training-${"a".repeat(32)}`,
  owner: "training",
  generation: 4,
  state: "running",
};

test("discovery uses real registry identity and never adopts another subsystem's jobs", () => {
  assert.deepEqual(
    trainingDownloads([
      job,
      { ...job, owner: "load" },
      { ...job, variant: "Q4_K_M" },
      { ...job, state: "complete" },
      { ...job, generation: undefined },
    ]),
    [job],
  );
});

test("global discovery adopts a live training download after navigating or reloading", async () => {
  const adopted: ActiveModelDownload[] = [];
  const discovery = new TrainingDownloadDiscovery({
    fetch: async () => [job],
    session: () => "account-a",
    adopt: (row) => adopted.push(row),
  });
  assert.equal(await discovery.poll(), true);
  assert.deepEqual(adopted, [job]);
  discovery.dispose();
});

test("session changes and teardown discard in-flight adoption responses", async () => {
  for (const action of ["switch", "reset", "dispose"]) {
    let session = "account-a";
    let resolve!: (rows: ActiveModelDownload[]) => void;
    const adopted: ActiveModelDownload[] = [];
    const discovery = new TrainingDownloadDiscovery({
      fetch: () =>
        new Promise((done) => {
          resolve = done;
        }),
      session: () => session,
      adopt: (row) => adopted.push(row),
    });
    const polling = discovery.poll();
    if (action === "switch") session = "account-b";
    else if (action === "reset") discovery.reset();
    else discovery.dispose();
    resolve([job]);
    assert.equal(await polling, false);
    assert.deepEqual(adopted, []);
    discovery.dispose();
  }
});

test("current file batches reset truthfully instead of reusing an earlier total or percentage", () => {
  const previous = {
    kind: "model",
    variant: job.variant,
    expectedBytes: 22,
    downloadedBytes: 13,
    completedBytes: 0,
    fraction: 0.6,
  } as ManagedDownload;
  const next = resolveProgressUpdate(previous, {
    progress_scope: "active_files",
    downloaded_bytes: 1,
    expected_bytes: 4,
    progress: 0.25,
    complete_on_disk: true,
    completed_bytes: 4,
  });
  assert.equal(next.downloadedBytes, 1);
  assert.equal(next.expected, 4);
  assert.equal(next.fraction, 0.25);
  assert.equal(next.completeOnDisk, false);
  assert.equal(next.completedBytes, 0);
  const gap = resolveProgressUpdate(previous, {
    progress_scope: "active_files",
    downloaded_bytes: 0,
    expected_bytes: 0,
    progress: 0,
  });
  assert.equal(gap.expected, 0);
  assert.equal(gap.downloadedBytes, 0);
  assert.equal(gap.fraction, 0);
});
