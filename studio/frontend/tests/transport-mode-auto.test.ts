// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

// constants.ts only, deliberately: transport-preference.ts reaches lib/toast -> a .tsx component
// barrel, which node's type stripping cannot load. It adds no decision logic of its own -- its
// readStored() is exactly `isTransportMode(raw) ? raw : DEFAULT_TRANSPORT_MODE` -- so the storage
// contract is fully pinned by the two exports tested here.
const {
  DEFAULT_TRANSPORT_MODE,
  RESOLVED_TRANSPORTS,
  TRANSPORT,
  TRANSPORT_MODES,
  isResolvedTransport,
  isTransportMode,
} = await import("../src/features/hub/download-manager/constants.ts");

test("auto is the default transport preference", () => {
  // The backend picks per machine (RAM, hf_xet build, recent Xet failures); a user who never
  // touches this control is exactly the one who most needs that.
  assert.equal(DEFAULT_TRANSPORT_MODE, TRANSPORT.AUTO);
});

test("auto is a preference, not a transport a download can run on", () => {
  // Only "xet"/"http" ever reach a .transport marker on disk: the marker records which writer
  // produced a partial, and a resume picks its strategy from it.
  assert.ok(TRANSPORT_MODES.includes(TRANSPORT.AUTO));
  assert.ok(!RESOLVED_TRANSPORTS.includes(TRANSPORT.AUTO as never));
  assert.ok(isResolvedTransport("xet"));
  assert.ok(isResolvedTransport("http"));
  assert.ok(!isResolvedTransport("auto"));
});

test("a previously stored explicit preference is still valid", () => {
  // Someone who deliberately pinned HTTP (or Xet) before this change must not be moved onto Auto:
  // isTransportMode() accepting the old values is what keeps readStored() returning them.
  assert.ok(isTransportMode("http"));
  assert.ok(isTransportMode("xet"));
});

test("an unrecognised stored value is rejected, so readStored falls back to auto", () => {
  for (const junk of ["ftp", "", "AUTO", null, undefined, 3]) {
    assert.ok(!isTransportMode(junk), `${String(junk)} should not be a transport mode`);
  }
});
