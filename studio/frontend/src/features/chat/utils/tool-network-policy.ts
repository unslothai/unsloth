// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  ToolExecutionMode,
  ToolIsolationCapability,
  ToolNetworkPolicy,
} from "../tool-isolation";

/** Whether this host's Required-mode backend can run the allowlist proxy at all. */
export function capabilityOffersNetworkAllowlist(
  capability: Pick<ToolIsolationCapability, "network_policies"> | null,
): boolean {
  return capability?.network_policies.includes("allowlist") === true;
}

/**
 * The policy a request may carry. "allowlist" is sent only for a Required launch on a host whose
 * capability lists it: Full has no sandbox to attach a proxy to, Limited cannot fence the network,
 * and a backend that never advertised the allowlist must not be asked for one (an older Studio
 * would reject or, worse, misread the field). Everything else collapses to "deny", which is also
 * what an omitted field means on the wire.
 */
/**
 * A queued send carries the policy chosen when it was queued, but opening the network is a
 * grant, and a grant withdrawn while the message waited must not be honoured. The snapshot
 * still guards the other direction: turning the allowlist on after queueing does not widen a
 * send that was prepared without it.
 */
export function queuedToolNetworkPolicy(
  snapshot: ToolNetworkPolicy,
  live: ToolNetworkPolicy,
): ToolNetworkPolicy {
  return snapshot === "allowlist" && live === "allowlist" ? "allowlist" : "deny";
}

export function effectiveToolNetworkPolicy(
  requested: ToolNetworkPolicy,
  mode: ToolExecutionMode,
  capability: Pick<ToolIsolationCapability, "network_policies"> | null,
): ToolNetworkPolicy {
  if (
    requested === "allowlist" &&
    mode === "os_isolation_required" &&
    capabilityOffersNetworkAllowlist(capability)
  ) {
    return "allowlist";
  }
  return "deny";
}
