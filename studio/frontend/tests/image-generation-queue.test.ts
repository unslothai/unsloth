// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { ImageQueueJob } from "../src/features/images/api.ts";
import {
  newlyFinished,
  promptSnippet,
  queueBusy,
  queuedJobs,
  queueJobSummary,
  runningJob,
} from "../src/features/images/lib/generation-queue.ts";
import { readSrc } from "./helpers/kit.ts";

function job(id: string, status: ImageQueueJob["status"], extra: Partial<ImageQueueJob> = {}): ImageQueueJob {
  return {
    id,
    status,
    position: null,
    prompt: `prompt ${id}`,
    width: 1024,
    height: 768,
    steps: 9,
    seed: 42,
    batch_size: 1,
    workflow: null,
    loras: [],
    created_at: 0,
    started_at: null,
    finished_at: null,
    error: null,
    image_ids: [],
    seed_used: null,
    ...extra,
  };
}

test("the running job and the waiting ones are told apart; finished jobs are neither", () => {
  const jobs = [job("a", "running"), job("b", "queued"), job("c", "queued"), job("d", "done")];
  assert.equal(runningJob(jobs)?.id, "a");
  assert.deepEqual(queuedJobs(jobs).map((j) => j.id), ["b", "c"]);
  assert.equal(queueBusy(jobs), true);
  assert.equal(queueBusy([job("d", "done"), job("e", "cancelled")]), false);
  assert.equal(runningJob([job("b", "queued")]), null);
});

test("a job pending on the last poll, or queued and finished since, counts as newly finished", () => {
  const prev = [job("a", "running"), job("b", "queued"), job("old", "done")];
  const next = [job("b", "running"), job("a", "done"), job("api", "failed"), job("old", "done")];
  assert.deepEqual(newlyFinished(prev, next).map((j) => j.id), ["a", "api"]);
  // The first poll after a reload reports history, not news.
  assert.deepEqual(newlyFinished(null, next), []);
  assert.deepEqual(
    newlyFinished([job("b", "queued")], [job("b", "cancelled")]).map((j) => j.status),
    ["cancelled"],
  );
});

test("a row shows a one-line prompt snippet and the settings that tell jobs apart", () => {
  assert.equal(promptSnippet("a  red\n fox"), "a red fox");
  const long = promptSnippet("x".repeat(100), 20);
  assert.equal(long.length, 20);
  assert.ok(long.endsWith("…"));
  assert.equal(queueJobSummary(job("a", "queued")), "1024×768 · 9 steps · seed 42");
  assert.equal(
    queueJobSummary(job("a", "queued", { steps: null, seed: null, batch_size: 4 })),
    "1024×768 · model steps · random seed · ×4",
  );
});

test("while a render runs the page offers Queue beside Stop, and Generate queues behind waiting jobs", () => {
  const page = readSrc("features/images/images-page.tsx");
  assert.match(page, /\{canQueue && compareRun === null && \(\s*<Button[^>]*onClick=\{handleQueue\}/);
  assert.match(page, /onClick=\{queueWaiting && canQueue \? handleQueue : handleGenerateWithRecall\}/);
  // Queue reuses Generate's validation and request, then hands the runs to the server.
  assert.match(page, /if \(toQueue && !compare && !external\) \{[\s\S]*?enqueueDiffusionImage\(requestFor\(i\)\)/);
  assert.match(page, /res = await generateDiffusionImage\(requestFor\(i\), { id: sequenceId, more: i < runTotal - 1 }\);/);
});
