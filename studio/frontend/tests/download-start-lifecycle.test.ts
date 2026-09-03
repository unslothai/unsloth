// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Exercise the real preflight -> accepted start -> terminal lifecycle. Helper-only
// tests cannot prove that preflight waits for acceptance or that finalize owns the
// same toast id.

import assert from "node:assert/strict";
import test, { after } from "node:test";
import { register } from "node:module";

import { installLocalStorageFake } from "./helpers/kit.ts";

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

const { calls } = await import("./helpers/toast-stub.mjs");
const { requestStart } =
  await import("../src/features/hub/download-manager/transport-conflict.ts");
const { jobKeyOf, removeJob } =
  await import("../src/features/hub/download-manager/download-manager-state.ts");
const { finalize } =
  await import("../src/features/hub/download-manager/poll-loop.ts");
const { dismissStartToastsForModelSelection, startToastId } =
  await import("../src/features/hub/download-manager/start-toast.ts");

function json(body: unknown): Response {
  return Response.json(body, { status: 200 });
}

function visibleToastCalls() {
  return calls.filter((call) => call.kind !== "dismiss");
}

test("restart disclosure waits for acceptance and completion dismisses its one keyed toast", async () => {
  calls.length = 0;

  let acceptStart!: () => void;
  let markStartRequested!: () => void;
  const startRequested = new Promise<void>((resolve) => {
    markStartRequested = resolve;
  });

  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.startsWith("/api/hub/active-downloads")) {
      return json({ downloads: [] });
    }
    if (url.startsWith("/api/studio/download-transport-capabilities")) {
      return json({
        http: { available: true, reason: null },
        xet: { available: true, reason: null },
        auto_resolves_to: "xet",
        auto_reason: null,
      });
    }
    if (url.startsWith("/api/hub/transport-status")) {
      return json({
        has_partial: true,
        last_transport: "xet",
        resumable: false,
      });
    }
    if (url === "/api/hub/download") {
      markStartRequested();
      return new Promise<Response>((resolve) => {
        acceptStart = () =>
          resolve(
            json({
              accepted: true,
              attached: false,
              state: "running",
              generation: 7,
              transport: "xet",
              job_key: "backend-key",
            }),
          );
      });
    }
    if (url === "/api/settings/xet-notice/reserve") {
      return json({ granted: true, shown: 1, limit: 3 });
    }
    throw new Error(`Unexpected request: ${url}`);
  }) as typeof fetch;

  const request = {
    kind: "model" as const,
    repoId: "org/restart-model",
    variant: "Q4_K_M",
    expectedBytes: 4096,
  };
  const starting = requestStart(request);
  await startRequested;
  assert.deepEqual(
    visibleToastCalls(),
    [],
    "preflight claimed a restart before the backend accepted it",
  );

  acceptStart();
  assert.equal(await starting, "started");
  await new Promise<void>((resolve) => setImmediate(resolve));

  const key = jobKeyOf(request.kind, request.repoId, request.variant);
  assert.deepEqual(visibleToastCalls(), [
    {
      kind: "info",
      title: "Restarting this download",
      options: {
        id: startToastId(key),
        description:
          "The partial can't be resumed, so Xet is starting over. The bar may stay at 0% and jump to done.",
        duration: 8000,
        classNames: { description: "!text-muted-foreground" },
      },
    },
  ]);

  finalize(key, "complete");
  assert.deepEqual(calls.at(-1), {
    kind: "dismiss",
    id: startToastId(key),
  });
  removeJob(key);
});

test("a stale model selection does not spend an Xet notice reservation", async () => {
  calls.length = 0;

  let acceptStart!: () => void;
  let markStartRequested!: () => void;
  let reservations = 0;
  const startRequested = new Promise<void>((resolve) => {
    markStartRequested = resolve;
  });

  globalThis.fetch = (async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.startsWith("/api/hub/active-downloads")) {
      return json({ downloads: [] });
    }
    if (url.startsWith("/api/studio/download-transport-capabilities")) {
      return json({
        http: { available: true, reason: null },
        xet: { available: true, reason: null },
        auto_resolves_to: "xet",
        auto_reason: null,
      });
    }
    if (url.startsWith("/api/hub/transport-status")) {
      return json({
        has_partial: false,
        last_transport: null,
        resumable: false,
      });
    }
    if (url === "/api/hub/download") {
      markStartRequested();
      return new Promise<Response>((resolve) => {
        acceptStart = () =>
          resolve(
            json({
              accepted: true,
              attached: false,
              state: "running",
              generation: 8,
              transport: "xet",
              job_key: "backend-key",
            }),
          );
      });
    }
    if (url === "/api/settings/xet-notice/reserve") {
      reservations += 1;
      return json({ granted: true, shown: reservations, limit: 3 });
    }
    throw new Error(`Unexpected request: ${url}`);
  }) as typeof fetch;

  const request = {
    kind: "model" as const,
    repoId: "org/stale-model",
    variant: "Q4_K_M",
    expectedBytes: 4096,
  };
  const starting = requestStart(request);
  await startRequested;

  dismissStartToastsForModelSelection();
  acceptStart();
  assert.equal(await starting, "started");
  await new Promise<void>((resolve) => setImmediate(resolve));

  const key = jobKeyOf(request.kind, request.repoId, request.variant);
  try {
    assert.equal(reservations, 0);
    assert.deepEqual(visibleToastCalls(), []);
  } finally {
    finalize(key, "complete");
    removeJob(key);
  }
});
