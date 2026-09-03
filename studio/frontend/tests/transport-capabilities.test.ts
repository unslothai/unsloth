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
  // Rebuilding the object from http/xet alone discarded the verdict, so Auto resolved to Xet on
  // every machine, including ones the backend had just demoted to HTTP.
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
  // Older backend, predating Auto: the download-time ladder still falls back to HTTP, so Xet is
  // the safe assumption.
  const caps = normalizeDownloadTransportCapabilities({
    http: { available: true, reason: null },
    xet: { available: true, reason: null },
  });

  assert.equal(caps.auto_resolves_to, "xet");
  assert.equal(caps.auto_reason, null);
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
