// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { normalizeDownloadTransportCapabilities } = await import(
  "../src/features/hub/download-manager/transport-capabilities.ts"
);

test("the backend's auto verdict survives normalization", () => {
  // Rebuilding from http/xet alone discarded the backend's verdict.
  const caps = normalizeDownloadTransportCapabilities({
    http: { available: true, reason: null },
    xet: { available: true, reason: null },
    auto_resolves_to: "http",
    auto_reason: "Xet stalled twice on this machine",
  });

  assert.equal(caps.auto_resolves_to, "http");
  assert.equal(caps.auto_reason, "Xet stalled twice on this machine");
});

test("a backend with no auto fields still resolves to xet", () => {
  // Older backends predate Auto, but their download ladder falls back to HTTP, so Xet is safe.
  const caps = normalizeDownloadTransportCapabilities({
    http: { available: true, reason: null },
    xet: { available: true, reason: null },
  });

  assert.equal(caps.auto_resolves_to, "xet");
  assert.equal(caps.auto_reason, null);
});

test("the resumability verdict survives normalization", () => {
  const caps = normalizeDownloadTransportCapabilities({
    http: { available: true, reason: null },
    xet: { available: true, reason: null },
    partials_resumable: true,
  });

  assert.equal(caps.partials_resumable, true);
});

test("an unverified backend never claims a partial is resumable", () => {
  for (const value of [{}, { partials_resumable: "yes" }]) {
    const caps = normalizeDownloadTransportCapabilities({
      http: { available: true, reason: null },
      xet: { available: true, reason: null },
      ...value,
    });
    assert.equal(caps.partials_resumable, false);
  }
});

test("a junk auto verdict is not trusted", () => {
  const caps = normalizeDownloadTransportCapabilities({
    http: { available: true, reason: null },
    xet: { available: true, reason: null },
    auto_resolves_to: "carrier-pigeon",
    auto_reason: 42,
  });

  assert.equal(caps.auto_resolves_to, "xet");
  assert.equal(caps.auto_reason, null);
});
