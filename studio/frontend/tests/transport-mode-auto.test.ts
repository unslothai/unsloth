// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

// constants.ts only: transport-preference.ts reaches lib/toast, a .tsx barrel Node cannot load.
const {
  DEFAULT_TRANSPORT_MODE,
  RESOLVED_TRANSPORTS,
  TRANSPORT,
  TRANSPORT_MODES,
  isResolvedTransport,
  isTransportMode,
} = await import("../src/features/hub/download-manager/constants.ts");

test("auto is the default transport preference", () => {
  // The backend picks per machine (RAM, hf_xet build, recent Xet failures).
  assert.equal(DEFAULT_TRANSPORT_MODE, TRANSPORT.AUTO);
});

test("auto is a preference, not a transport a download can run on", () => {
  // Only "xet"/"http" reach a .transport marker, which records the writer a resume must match.
  assert.ok(TRANSPORT_MODES.includes(TRANSPORT.AUTO));
  assert.ok(!RESOLVED_TRANSPORTS.includes(TRANSPORT.AUTO as never));
  assert.ok(isResolvedTransport("xet"));
  assert.ok(isResolvedTransport("http"));
  assert.ok(!isResolvedTransport("auto"));
});

test("every previously stored preference is still valid", () => {
  // A pinned transport is kept; only an unset value falls through to the install setting.
  assert.ok(isTransportMode("http"));
  assert.ok(isTransportMode("xet"));
  assert.ok(isTransportMode("auto"));
});

test("an unrecognised stored value is rejected, so readStored reports no choice", () => {
  for (const junk of ["ftp", "", "AUTO", null, undefined, 3]) {
    assert.ok(
      !isTransportMode(junk),
      `${String(junk)} should not be a transport mode`,
    );
  }
});
