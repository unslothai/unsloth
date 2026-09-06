// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  capabilityOffersNetworkAllowlist,
  effectiveToolNetworkPolicy,
  queuedToolNetworkPolicy,
} from "../src/features/chat/utils/tool-network-policy.ts";

const offers = { network_policies: ["deny", "allowlist"] as ("deny" | "allowlist")[] };
const denyOnly = { network_policies: ["deny"] as ("deny" | "allowlist")[] };

test("the allowlist is offered only when the backend lists it", () => {
  assert.equal(capabilityOffersNetworkAllowlist(offers), true);
  assert.equal(capabilityOffersNetworkAllowlist(denyOnly), false);
  assert.equal(capabilityOffersNetworkAllowlist(null), false);
});

test("allowlist goes on the wire only for Required on a host that offers it", () => {
  assert.equal(effectiveToolNetworkPolicy("allowlist", "os_isolation_required", offers), "allowlist");
  // Limited cannot fence the network and Full has no sandbox: both collapse to deny.
  assert.equal(effectiveToolNetworkPolicy("allowlist", "limited", offers), "deny");
  assert.equal(effectiveToolNetworkPolicy("allowlist", "full", offers), "deny");
  // A Windows or pre-proxy backend never advertised it, so it is never asked for one.
  assert.equal(effectiveToolNetworkPolicy("allowlist", "os_isolation_required", denyOnly), "deny");
  assert.equal(effectiveToolNetworkPolicy("allowlist", "os_isolation_required", null), "deny");
  assert.equal(effectiveToolNetworkPolicy("deny", "os_isolation_required", offers), "deny");
});

test("a queued allowlist send is clamped by the live store in both directions", () => {
  // Withdrawn while queued: the revocation wins.
  assert.equal(queuedToolNetworkPolicy("allowlist", "deny"), "deny");
  // Turned on after queueing: the snapshot keeps the send narrow.
  assert.equal(queuedToolNetworkPolicy("deny", "allowlist"), "deny");
  assert.equal(queuedToolNetworkPolicy("allowlist", "allowlist"), "allowlist");
  assert.equal(queuedToolNetworkPolicy("deny", "deny"), "deny");
});
