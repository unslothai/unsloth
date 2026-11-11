// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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

const starts: string[] = [];
const originalFetch = globalThis.fetch;
globalThis.fetch = (async (input: RequestInfo | URL) => {
  const url = String(input);
  if (url.startsWith("/api/studio/download-transport-capabilities")) {
    return Response.json({
      http: { available: true, reason: null },
      xet: { available: true, reason: null },
      auto_resolves_to: "xet",
      auto_reason: null,
    });
  }
  if (url.startsWith("/api/hub/transport-status")) {
    return Response.json({
      has_partial: true,
      last_transport: "http",
      resumable: true,
    });
  }
  if (url === "/api/hub/download") starts.push(url);
  return Response.json({});
}) as typeof fetch;
after(() => {
  globalThis.fetch = originalFetch;
});

const { calls } = await import("./helpers/toast-stub.mjs");
const { retryDownload } =
  await import("../src/features/hub/download-manager/retry-start.ts");
const { getState, jobKeyOf, putJob, removeJob } =
  await import("../src/features/hub/download-manager/download-manager-state.ts");

test("a retry that hits a transport conflict tells the user where to resolve it", async () => {
  calls.length = 0;
  const key = jobKeyOf("model", "Org/Repo", "Q4_K_M");
  await retryDownload({
    key,
    kind: "model",
    repoId: "Org/Repo",
    variant: "Q4_K_M",
    state: "error",
    downloadedBytes: 1024,
    completedBytes: 0,
    completeOnDisk: false,
    expectedBytes: 4096,
    fraction: 0.25,
    bytesPerSec: 0,
    etaSeconds: 0,
    error: "Connection reset",
    startedAt: 1,
    downloadDir: null,
  });
  assert.equal(starts.length, 0);
  assert.ok(getState().conflicts[key]);
  assert.ok(
    calls.some(
      (call: { kind: string; title?: string }) =>
        call.kind === "info" &&
        call.title === "Resume this download from Models",
    ),
  );
});

test("a retry refused because another job owns the repo says so", async () => {
  calls.length = 0;
  starts.length = 0;
  const running = jobKeyOf("model", "Org/Busy", null);
  putJob({
    key: running,
    kind: "model",
    repoId: "Org/Busy",
    variant: null,
    state: "running",
    downloadedBytes: 0,
    completedBytes: 0,
    completeOnDisk: false,
    expectedBytes: 4096,
    fraction: 0,
    bytesPerSec: 0,
    etaSeconds: 0,
    error: null,
    startedAt: 2,
  });
  await retryDownload({
    key: jobKeyOf("model", "Org/Busy", "@hub-assets"),
    kind: "model",
    repoId: "Org/Busy",
    variant: "@hub-assets",
    scopedFiles: ["vae.safetensors"],
    state: "error",
    downloadedBytes: 0,
    completedBytes: 0,
    completeOnDisk: false,
    expectedBytes: 1024,
    fraction: 0,
    bytesPerSec: 0,
    etaSeconds: 0,
    error: "Connection reset",
    startedAt: 1,
  });
  assert.equal(starts.length, 0);
  assert.ok(
    calls.some(
      (call: { kind: string; title?: string }) =>
        call.kind === "info" && call.title === "Download already in progress",
    ),
  );
  removeJob(running);
});
