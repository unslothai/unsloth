// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { after } from "node:test";
import { register } from "node:module";

import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";

register("./helpers/download-lifecycle-resolver.mjs", import.meta.url);

const { storage } = installLocalStorageFake();
Object.assign(globalThis.window, {
  location: { protocol: "http:", pathname: "/chat" },
  setTimeout: globalThis.setTimeout.bind(globalThis),
  clearTimeout: globalThis.clearTimeout.bind(globalThis),
});
storage.setItem("unsloth.studio.transportMode", "xet");

const originalFetch = globalThis.fetch;
after(() => {
  globalThis.fetch = originalFetch;
});

const { requestStart } =
  await import("../src/features/hub/download-manager/transport-conflict.ts");
const { jobKeyOf, removeJob, useDownloadManagerStore } =
  await import("../src/features/hub/download-manager/download-manager-state.ts");
const { finalize } =
  await import("../src/features/hub/download-manager/poll-loop.ts");
const {
  publishPendingStart,
  resetPendingStarts,
  retirePendingStart,
  usePendingStarts,
  visiblePendingStarts,
} = await import("../src/features/hub/download-manager/pending-starts.ts");

function json(body: unknown): Response {
  return Response.json(body, { status: 200 });
}

const capabilities = {
  http: { available: true, reason: null },
  xet: { available: true, reason: null },
  auto_resolves_to: "xet",
  auto_reason: null,
};

type StartAnswer = "accepted" | "rejected" | "unreachable";

function stubBackend({ start = "accepted" }: { start?: StartAnswer } = {}) {
  let release!: () => void;
  let markRequested!: () => void;
  const requested = new Promise<void>((resolve) => {
    markRequested = resolve;
  });
  const held = new Promise<Response>((resolve) => {
    release = () => resolve(json({ downloads: [] }));
  });
  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.startsWith("/api/studio/download-transport-capabilities")) {
      return json(capabilities);
    }
    if (url.startsWith("/api/hub/active-downloads")) {
      markRequested();
      return held;
    }
    if (url.startsWith("/api/hub/transport-status")) {
      return json({ has_partial: false, last_transport: null, resumable: false });
    }
    if (url === "/api/system/disk") {
      return new Response(null, { status: 503 });
    }
    if (url === "/api/hub/download") {
      if (start === "unreachable") throw new Error("backend down");
      return json({
        accepted: start === "accepted",
        attached: false,
        state: start === "accepted" ? "running" : "deleting",
        generation: 1,
        transport: "xet",
        job_key: "backend-key",
      });
    }
    if (url === "/api/settings/xet-notice/reserve") {
      return json({ granted: false, shown: 3, limit: 3 });
    }
    throw new Error(`Unexpected request: ${url}`);
  }) as typeof fetch;
  return { requested, release: () => release() };
}

test("a requested start is visible while its preflight runs, until the backend accepts it", async () => {
  const backend = stubBackend();
  const request = {
    kind: "model" as const,
    repoId: "org/pending-model",
    variant: "Q4_K_M",
    expectedBytes: 4096,
    presentation: {
      label: "Pending model",
      filename: "pending-q4_k_m.gguf",
      expectedBytes: 4096,
    },
  };
  const key = jobKeyOf(request.kind, request.repoId, request.variant);
  assert.deepEqual(usePendingStarts.getState().starts, {});

  const starting = requestStart(request);
  await backend.requested;
  assert.deepEqual(usePendingStarts.getState().starts, {
    [key]: {
      key,
      kind: "model",
      repoId: request.repoId,
      variant: request.variant,
      presentation: request.presentation,
    },
  });
  assert.equal(
    visiblePendingStarts(
      usePendingStarts.getState().starts,
      useDownloadManagerStore.getState().jobs,
      [],
    ).length,
    1,
  );

  assert.equal(await requestStart(request), "busy");
  assert.deepEqual(Object.keys(usePendingStarts.getState().starts), [key]);

  backend.release();
  assert.equal(await starting, "started");
  assert.deepEqual(usePendingStarts.getState().starts, {});
  assert.equal(useDownloadManagerStore.getState().jobs[key]?.state, "running");

  finalize(key, "complete");
  removeJob(key);
});

test("a start the backend cannot be reached for leaves no pending entry behind", async () => {
  const backend = stubBackend({ start: "unreachable" });
  const request = {
    kind: "model" as const,
    repoId: "org/refused-model",
    variant: "Q8_0",
    expectedBytes: 4096,
  };
  const key = jobKeyOf(request.kind, request.repoId, request.variant);

  const starting = requestStart(request);
  backend.release();

  assert.notEqual(await starting, "started");
  assert.deepEqual(usePendingStarts.getState().starts, {});
  removeJob(key);
});

test("a start the backend rejects ends as an error row with no pending entry behind it", async () => {
  const backend = stubBackend({ start: "rejected" });
  const request = {
    kind: "model" as const,
    repoId: "org/rejected-model",
    variant: "Q5_K_M",
    expectedBytes: 4096,
  };
  const key = jobKeyOf(request.kind, request.repoId, request.variant);

  const starting = requestStart(request);
  backend.release();

  assert.notEqual(await starting, "started");
  const job = useDownloadManagerStore.getState().jobs[key];
  assert.equal(job?.state, "error");
  assert.match(job?.error ?? "", /being removed/);
  assert.deepEqual(usePendingStarts.getState().starts, {});
  removeJob(key);
});

test("retiring a start removes only the entry that was published for it", () => {
  const request = {
    kind: "model" as const,
    repoId: "org/reset-model",
    variant: "Q4_K_M",
    expectedBytes: 4096,
  };
  const key = jobKeyOf(request.kind, request.repoId, request.variant);

  const first = publishPendingStart(key, request);
  resetPendingStarts();
  const second = publishPendingStart(key, request);

  retirePendingStart(first);
  assert.equal(usePendingStarts.getState().starts[key], second);
  retirePendingStart(second);
  assert.deepEqual(usePendingStarts.getState().starts, {});
});

test("visible pending starts yield to an active job and to a queued plan for the same transfer", () => {
  const model = {
    key: "model:org/a#q4_k_m",
    kind: "model" as const,
    repoId: "org/a",
    variant: "Q4_K_M",
  };
  const dataset = {
    key: "dataset:org/a",
    kind: "dataset" as const,
    repoId: "org/a",
    variant: null,
  };
  const starts = { [model.key]: model, [dataset.key]: dataset };

  assert.deepEqual(visiblePendingStarts(starts, {}, []), [model, dataset]);
  assert.deepEqual(
    visiblePendingStarts(starts, { [model.key]: { state: "running" as const } }, []),
    [dataset],
  );
  assert.deepEqual(
    visiblePendingStarts(starts, { [model.key]: { state: "cancelled" as const } }, []),
    [model, dataset],
  );
  assert.deepEqual(
    visiblePendingStarts(starts, {}, [{ repoId: "Org/A", variant: "q4_k_m" }]),
    [dataset],
  );
  assert.deepEqual(
    visiblePendingStarts(starts, {}, [{ repoId: "org/a", variant: "@hub-assets" }]),
    [model, dataset],
  );
});

test("a restart inside a finished row's linger keeps that row and marks it starting", async () => {
  const request = {
    kind: "model" as const,
    repoId: "org/retried-model",
    variant: "Q4_K_M",
    expectedBytes: 4096,
  };
  const key = jobKeyOf(request.kind, request.repoId, request.variant);
  stubBackend().release();
  assert.equal(await requestStart(request), "started");
  finalize(key, "cancelled");
  assert.equal(useDownloadManagerStore.getState().jobs[key]?.state, "cancelled");

  const backend = stubBackend();
  const restarting = requestStart(request);
  await backend.requested;
  const { jobs } = useDownloadManagerStore.getState();
  const { starts } = usePendingStarts.getState();
  assert.equal(jobs[key]?.state, "cancelled");
  assert.deepEqual(Object.keys(starts), [key]);
  assert.equal(visiblePendingStarts(starts, jobs, []).length, 1);

  backend.release();
  assert.equal(await restarting, "started");
  assert.equal(useDownloadManagerStore.getState().jobs[key]?.state, "running");
  assert.deepEqual(usePendingStarts.getState().starts, {});

  finalize(key, "complete");
  removeJob(key);
});

test("the panel derives a start's own row and a listed row's starting state from the same visible set", () => {
  const panel = readSrc("features/hub/download-manager/download-manager-panel.tsx");
  assert.match(
    panel,
    /const unlistedStarts = pending\.filter\(\(start\) => !\(start\.key in jobs\)\)/,
  );
  assert.match(
    panel,
    /const startingKeys = new Set\(pending\.map\(\(start\) => start\.key\)\)/,
  );
  assert.match(panel, /starting=\{startingKeys\.has\(jobKey\)\}/);
});
