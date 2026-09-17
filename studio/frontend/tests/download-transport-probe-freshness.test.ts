// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");
const { getDownloadTransportCapabilities } = await import(
  "../src/features/hub/download-manager/api.ts"
);

const CAPS_URL = "/api/studio/download-transport-capabilities";

function serve(verdicts: string[]): { probes: string[]; polls: string[] } {
  const probes: string[] = [];
  const polls: string[] = [];
  setAuthFetchHandler((input) => {
    const url = String(input);
    assert.ok(url.startsWith(CAPS_URL), `unexpected request: ${url}`);
    const probed = url.includes("probe=1");
    (probed ? probes : polls).push(url);
    const auto = verdicts[Math.min(probes.length - 1, verdicts.length - 1)];
    return new Response(
      JSON.stringify({
        http: { available: true, reason: null },
        xet: { available: true, reason: null },
        auto_resolves_to: probed ? auto : "xet",
        auto_reason: null,
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  });
  return { probes, polls };
}

test("every Auto download start re-asks the backend, so admission sees live reservations", async () => {
  // The free-RAM gate lives in the probe (get_download_transport_capabilities(probe=True)), and
  // its verdict subtracts RAM already promised to running Xet workers. Serving the previous
  // probe's answer from cache admits every download started inside the TTL on the SAME
  // pre-reservation verdict: they all submit transport_mode="xet", and the start path honours an
  // explicit "xet" without re-reading free RAM.
  const { probes } = serve(["xet", "http", "http"]);

  const first = await getDownloadTransportCapabilities({ probe: true });
  const second = await getDownloadTransportCapabilities({ probe: true });
  const third = await getDownloadTransportCapabilities({ probe: true });

  assert.equal(probes.length, 3, "a start reused a stale Auto verdict");
  assert.equal(first.auto_resolves_to, "xet");
  assert.equal(second.auto_resolves_to, "http", "start 2 skipped the gate");
  assert.equal(third.auto_resolves_to, "http", "start 3 skipped the gate");
});

test("an ordinary render poll still answers from cache", async () => {
  // The probe is per download start; the picker polls on render and must not connect per poll.
  const { probes, polls } = serve(["xet"]);

  await getDownloadTransportCapabilities({ probe: true });
  await getDownloadTransportCapabilities();
  await getDownloadTransportCapabilities();

  assert.equal(probes.length, 1);
  assert.equal(polls.length, 0, "a render poll re-fetched needlessly");
});

test("a probe failure is not cached as a verdict", async () => {
  setAuthFetchHandler(() => new Response("nope", { status: 500 }));
  const caps = await getDownloadTransportCapabilities({ probe: true });
  assert.equal(caps.auto_resolves_to, "xet", "optimistic fallback stands");

  const { probes } = serve(["http"]);
  const next = await getDownloadTransportCapabilities({ probe: true });
  assert.equal(probes.length, 1);
  assert.equal(next.auto_resolves_to, "http");
  setAuthFetchHandler(null);
});
