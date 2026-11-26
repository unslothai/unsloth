// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A busy engine queues a video submit server-side instead of answering 409; the page shows the
// wait, lets the user cancel it, and resumes it after a reload.

import assert from "node:assert/strict";
import test from "node:test";

import type { VideoQueueJob } from "../src/features/video/api.ts";
import {
  oldestQueuedVideo,
  progressIsForVideo,
  queuedVideoLabel,
  queuedVideoNext,
} from "../src/features/video/video-queue.ts";
import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

function job(id: string, status: VideoQueueJob["status"], extra: Partial<VideoQueueJob> = {}): VideoQueueJob {
  return {
    video_id: id,
    status,
    queue_position: 1,
    eta_seconds: null,
    prompt: `prompt ${id}`,
    created_at: 0,
    source: "page",
    error: null,
    ...extra,
  };
}

test("the queued label shows the place and the ETA rounded up to minutes", () => {
  assert.equal(queuedVideoLabel(2, 125), "Queued #2, about 3 min");
  assert.equal(queuedVideoLabel(1, 60), "Queued #1, about 1 min");
  assert.equal(queuedVideoLabel(1, 61), "Queued #1, about 2 min");
  assert.equal(queuedVideoLabel(3, 59), "Queued #3, less than a minute");
  assert.equal(queuedVideoLabel(3, 0), "Queued #3, less than a minute");
  // Unknown ETA drops the ETA part; unknown place drops the number.
  assert.equal(queuedVideoLabel(4, null), "Queued #4");
  assert.equal(queuedVideoLabel(null, null), "Queued");
  assert.equal(queuedVideoLabel(null, 300), "Queued, about 5 min");
});

test("a queue poll keeps waiting, starts, fails or clears", () => {
  const queued = queuedVideoNext([job("other", "queued"), job("v1", "queued", { queue_position: 2, eta_seconds: 90 })], "v1");
  assert.deepEqual(queued, { kind: "queued", job: { videoId: "v1", position: 2, etaSeconds: 90 } });
  // Gone from the list means it renders now: switch to generate-progress.
  assert.deepEqual(queuedVideoNext([job("other", "queued")], "v1"), { kind: "started" });
  assert.deepEqual(queuedVideoNext([], "v1"), { kind: "started" });
  assert.deepEqual(
    queuedVideoNext([job("v1", "failed", { error: "Video generation failed: OOM" })], "v1"),
    { kind: "failed", error: "Video generation failed: OOM" },
  );
  assert.deepEqual(queuedVideoNext([job("v1", "cancelled")], "v1"), { kind: "cancelled" });
});

test("a starting job keeps the queue poll until its start settles, queued again or failed", () => {
  // The start can go back to "queued" (engine busy again) or end "failed" before it renders.
  assert.deepEqual(queuedVideoNext([job("v1", "starting", { queue_position: 1, eta_seconds: 4 })], "v1"), {
    kind: "queued",
    job: { videoId: "v1", position: 1, etaSeconds: 4 },
  });
  assert.deepEqual(queuedVideoNext([job("v1", "queued", { queue_position: 1 })], "v1").kind, "queued");
  assert.deepEqual(queuedVideoNext([job("v1", "failed", { error: "No video model loaded." })], "v1"), {
    kind: "failed",
    error: "No video model loaded.",
  });
});

test("after a reload the oldest queued job is resumed, whatever its source", () => {
  const jobs = [
    job("newer", "queued", { created_at: 300, source: "page" }),
    job("failed", "failed", { created_at: 50 }),
    job("oldest", "queued", { created_at: 100, source: "api", queue_position: 3, eta_seconds: 400 }),
    job("chat", "queued", { created_at: 200, source: "chat" }),
  ];
  assert.deepEqual(oldestQueuedVideo(jobs), { videoId: "oldest", position: 3, etaSeconds: 400 });
  assert.equal(oldestQueuedVideo([job("a", "cancelled"), job("b", "failed")]), null);
  assert.equal(oldestQueuedVideo([]), null);
});

test("after a reload a job caught starting is resumed too, and its start keeps being followed", () => {
  const jobs = [
    job("queued", "queued", { created_at: 200 }),
    job("starting", "starting", { created_at: 10, queue_position: 1, eta_seconds: 0 }),
  ];
  assert.deepEqual(oldestQueuedVideo(jobs), { videoId: "starting", position: 1, etaSeconds: 0 });
  assert.equal(queuedVideoNext(jobs, "starting").kind, "queued");
});

test("Cancel while rendering names the followed clip, so a successor of the account is never stopped", () => {
  const page = readSrc("features/video/video-page.tsx");
  assert.match(page, /cancelVideoGeneration\(\s*queuedVideoId \?\? genVideoId\.current \?\? undefined,?\s*\)/);
  assert.doesNotMatch(page, /cancelVideoGeneration\(queuedVideoId \?\? undefined\)/);
});

test("a progress record tagged with another job is not ours", () => {
  const base = { active: false, phase: "completed", step: 0, total: 0 };
  assert.equal(progressIsForVideo({ ...base, video_id: "old" }, "v1"), false);
  assert.equal(progressIsForVideo({ ...base, video_id: "v1" }, "v1"), true);
  // Older backends send no id; a submit with no id follows whatever runs.
  assert.equal(progressIsForVideo(base, "v1"), true);
  assert.equal(progressIsForVideo({ ...base, video_id: "old" }, null), true);
});

type VideoApi = {
  cancelVideoGeneration: (videoId?: string) => Promise<{ cancelled: boolean }>;
  getVideoGenerateProgress: (videoId?: string) => Promise<{ video_id?: string | null }>;
  getVideoQueue: () => Promise<VideoQueueJob[]>;
};

function loadApi(calls: { path: string; init?: RequestInit }[], reply: () => unknown): VideoApi {
  return loadWithStubs<VideoApi>(
    new URL("../src/features/video/api.ts", import.meta.url),
    {
      "@/lib/model-lifecycle-events": { withBackgroundLoadNotice: (f: unknown) => f },
      "@/features/auth": {
        authFetch: async (path: string, init?: RequestInit) => {
          calls.push({ path, init });
          return new Response(JSON.stringify(reply()), { status: 200 });
        },
      },
      "@/lib/api-base": { apiUrl: (path: string) => path },
      "@/lib/format-fastapi-error": { readFastApiError: async () => "" },
      "@/lib/multi-model-refusal": { parseLoadJson: async () => ({}) },
      "@/lib/unified-memory-load-refusal": { withAllowOversizedLoad: (body: unknown) => body },
    },
    { relativePassthrough: true },
  );
}

test("cancel sends the queued job's video_id; a bare cancel stops the render", async () => {
  const calls: { path: string; init?: RequestInit }[] = [];
  const api = loadApi(calls, () => ({ cancelled: true }));
  assert.deepEqual(await api.cancelVideoGeneration("v1"), { cancelled: true });
  assert.equal(calls[0].path, "/api/inference/video/generate/cancel");
  assert.equal(calls[0].init?.method, "POST");
  assert.deepEqual(JSON.parse(String(calls[0].init?.body)), { video_id: "v1" });
  await api.cancelVideoGeneration();
  assert.equal(calls[1].init?.body, undefined);
});

test("the progress poll asks for its own clip by id, so a successor's record cannot hide its outcome", async () => {
  const calls: { path: string; init?: RequestInit }[] = [];
  const api = loadApi(calls, () => ({ active: false, phase: "completed", video_id: "a b" }));
  await api.getVideoGenerateProgress("a b");
  assert.equal(calls[0].path, "/api/inference/video/generate-progress?video_id=a%20b");
  await api.getVideoGenerateProgress();
  assert.equal(calls[1].path, "/api/inference/video/generate-progress");
  const page = readSrc("features/video/video-page.tsx");
  assert.match(page, /const p = await getVideoGenerateProgress\(genVideoId\.current \?\? undefined\);/);
});

test("the queue route returns the account's jobs", async () => {
  const calls: { path: string; init?: RequestInit }[] = [];
  const api = loadApi(calls, () => ({ jobs: [job("v1", "queued")] }));
  const jobs = await api.getVideoQueue();
  assert.equal(calls[0].path, "/api/inference/video/queue");
  assert.deepEqual(jobs.map((j) => j.video_id), ["v1"]);
});

test("the page queues on submit, cancels by id and resumes the queue on mount", () => {
  const page = readSrc("features/video/video-page.tsx");
  assert.match(page, /if \(accepted\.status === "queued" && accepted\.video_id\) \{\s*startQueuePoll\(/);
  assert.match(page, /startGenPoll\(accepted\.video_id \?\? null\);/);
  assert.match(page, /const queued = oldestQueuedVideo\(await getVideoQueue\(\)\);/);
  assert.match(page, /if \(next\.kind === "started"\) \{\s*startGenPoll\(job\.videoId\);/);
  assert.match(page, /queuedVideoLabel\(queuedVideo\.position, queuedVideo\.etaSeconds\)/);
});

test("a new Generate drops the finished clip's id, and a Cancel during the POST stops the clip it creates", () => {
  const page = readSrc("features/video/video-page.tsx");
  // Reset before the POST, so a Cancel in flight never names the previous clip.
  assert.match(page, /genVideoId\.current = null;\s*admitting\.current = true;[\s\S]*?accepted = await generateVideo\(/);
  // Cancel while admitting waits for the new id instead of sending a stale or bare one.
  assert.match(page, /if \(admitting\.current\) \{[\s\S]*?cancelOnAdmit\.current = true;\s*return;/);
  assert.match(page, /if \(cancelOnAdmit\.current\) \{[\s\S]*?await cancelVideoGeneration\(accepted\.video_id\);/);
});
