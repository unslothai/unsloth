// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Engram table is part of a download's identity: a start for the same quant with another
// table is not "already downloading", and adopting a backend job takes the table it reports.

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

const { requestStart } =
  await import("../src/features/hub/download-manager/transport-conflict.ts");
const { jobKeyOf, putJob, removeJob, useDownloadManagerStore } =
  await import("../src/features/hub/download-manager/download-manager-state.ts");
const { adoptJob, finalize, probeAndAdopt } =
  await import("../src/features/hub/download-manager/poll-loop.ts");

const REPO = "unsloth/Qwen3.8-Flash-Next-GGUF";
const QUANT = "UD-IQ1_S";
const KEY = jobKeyOf("model", REPO, QUANT);
const starts: Record<string, unknown>[] = [];
let reportedTable: string | null = "Q5_1";

function json(body: unknown): Response {
  return Response.json(body, { status: 200 });
}

globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
  const url = String(input);
  if (url.startsWith("/api/studio/download-transport-capabilities")) {
    return json({
      http: { available: true, reason: null },
      xet: { available: true, reason: null },
      auto_resolves_to: "xet",
      auto_reason: null,
    });
  }
  if (url.startsWith("/api/hub/active-downloads")) {
    return json({
      downloads: [
        { repo_id: REPO, variant: QUANT, state: "running", generation: 1, transport: "xet",
          engram_precision: reportedTable },
      ],
    });
  }
  if (url.startsWith("/api/hub/transport-status")) {
    return json({ has_partial: false, last_transport: null, resumable: false });
  }
  if (url.startsWith("/api/hub/gguf-download-progress")) {
    return json({ downloaded_bytes: 1, expected_bytes: 100, progress: 0.01, completed_bytes: 0 });
  }
  if (url.startsWith("/api/hub/download-status")) {
    return json({ state: "running", generation: 1, attempt: 1 });
  }
  if (url === "/api/system/disk") return new Response(null, { status: 503 });
  if (url === "/api/settings/xet-notice/reserve") {
    return json({ granted: false, shown: 3, limit: 3 });
  }
  if (url === "/api/hub/download") {
    starts.push(JSON.parse(String(init?.body ?? "{}")));
    return json({ accepted: true, attached: false, state: "running", generation: 1,
                  transport: "xet", job_key: KEY });
  }
  return json({});
}) as typeof fetch;

const request = (engramPrecision?: string | null) => ({
  kind: "model" as const,
  repoId: REPO,
  variant: QUANT,
  expectedBytes: 4096,
  ...(engramPrecision === undefined ? {} : { engramPrecision }),
});

test("only the same table counts as the download already running", async () => {
  assert.equal(await requestStart(request("Q5_1")), "started");
  assert.equal(starts.length, 1);
  assert.equal(starts[0]?.engram_precision, "Q5_1");
  assert.equal(useDownloadManagerStore.getState().jobs[KEY]?.engramPrecision, "Q5_1");

  assert.equal(await requestStart(request("q5_1")), "started");
  assert.equal(await requestStart(request("Q8_0")), "busy");
  assert.equal(await requestStart(request(null)), "busy");
  assert.equal(starts.length, 1, "no start reached the backend for another table");
});

test("adopting a backend job takes the table it reports, null clearing a stale one", async () => {
  reportedTable = null;
  await probeAndAdopt("model", REPO, new AbortController().signal, { includeVariants: true, fresh: true });
  assert.equal(useDownloadManagerStore.getState().jobs[KEY]?.engramPrecision, undefined);

  reportedTable = "Q5_1";
  await probeAndAdopt("model", REPO, new AbortController().signal, { includeVariants: true, fresh: true });
  assert.equal(useDownloadManagerStore.getState().jobs[KEY]?.engramPrecision, "Q5_1");

  finalize(KEY, "complete");
  removeJob(KEY);
});

test("adopting a job not yet polled replaces a stale table with the reported one", async () => {
  const key = jobKeyOf("model", REPO, "UD-Q2_K_XL");
  putJob({
    key, kind: "model", repoId: REPO, variant: "UD-Q2_K_XL", state: "running",
    downloadedBytes: 1, completedBytes: 0, completeOnDisk: false, expectedBytes: 100,
    fraction: 0.01, bytesPerSec: 0, etaSeconds: 0, error: null, startedAt: 1,
    engramPrecision: "Q5_1",
  });
  adoptJob(
    { kind: "model", repoId: REPO, variant: "UD-Q2_K_XL", expectedBytes: 0, engramPrecision: null },
    2, "running", "xet", null,
  );
  await new Promise((resolve) => setTimeout(resolve, 200));
  assert.equal(useDownloadManagerStore.getState().jobs[key]?.engramPrecision, undefined);
  finalize(key, "complete");
  removeJob(key);
});
