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

const originalFetch = globalThis.fetch;
after(() => {
  globalThis.fetch = originalFetch;
});

const { requestStart } =
  await import("../src/features/hub/download-manager/transport-conflict.ts");
const { clearDownloadTransportCapabilitiesCache } =
  await import("../src/features/hub/download-manager/api.ts");
const { removeJob, jobKeyOf } =
  await import("../src/features/hub/download-manager/download-manager-state.ts");
const { finalize } =
  await import("../src/features/hub/download-manager/poll-loop.ts");

function json(body: unknown): Response {
  return Response.json(body, { status: 200 });
}

function stubBackend(speedLimited: boolean, sibling: "http" | "xet") {
  const starts: Array<{ transport_mode?: string }> = [];
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.startsWith("/api/studio/download-transport-capabilities")) {
      return json({
        http: { available: true, reason: null },
        xet: { available: true, reason: null },
        auto_resolves_to: "xet",
        speed_limited: speedLimited,
      });
    }
    if (url.startsWith("/api/hub/active-downloads")) {
      return json({ downloads: [{ variant: "Q8_0", transport: sibling, state: "running" }] });
    }
    if (url.startsWith("/api/hub/transport-status")) {
      return json({ has_partial: false, last_transport: null, resumable: false });
    }
    if (url === "/api/system/disk") return new Response(null, { status: 503 });
    if (url === "/api/hub/download") {
      starts.push(JSON.parse(String(init?.body)));
      return json({ accepted: true, attached: false, state: "running", generation: 1, transport: sibling });
    }
    if (url === "/api/settings/xet-notice/reserve") {
      return json({ granted: false, shown: 3, limit: 3 });
    }
    throw new Error(`Unexpected request: ${url}`);
  }) as typeof fetch;
  return starts;
}

for (const sibling of ["http", "xet"] as const) {
  test(`under a limit a variant joins a sibling already on ${sibling}`, async () => {
    clearDownloadTransportCapabilitiesCache();
    const starts = stubBackend(true, sibling);
    const req = { kind: "model" as const, repoId: `org/limited-${sibling}`, variant: "Q4_K_M", expectedBytes: 4096 };
    const key = jobKeyOf(req.kind, req.repoId, req.variant);
    try {
      assert.equal(await requestStart(req), "started");
      // The preference goes through: only the backend knows whether a file fits under the HTTPS ceiling.
      assert.equal(starts[0]?.transport_mode, "xet");
    } finally {
      finalize(key, "complete");
      removeJob(key);
    }
  });
}

test("without a limit an explicit Xet choice still refuses an HTTP sibling", async () => {
  clearDownloadTransportCapabilitiesCache();
  const starts = stubBackend(false, "http");
  const req = { kind: "model" as const, repoId: "org/unlimited", variant: "Q4_K_M", expectedBytes: 4096 };
  assert.equal(await requestStart(req), "busy");
  assert.equal(starts.length, 0);
});
