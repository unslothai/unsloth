// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Type-only imports keep this module loadable by the node tests.
import type { SandboxCapability } from "./api/sandbox-capability";
import type { PermissionMode, SandboxLevel } from "./stores/chat-runtime-store";

/** A settled answer says Python or Terminal runs without OS isolation. Unknown (no answer, an old
 *  server) or still probing is not "missing". */
export function osSandboxMissing(capability: SandboxCapability | null): boolean {
  if (capability === null) return false;
  if (capability.pythonOsIsolated && capability.terminalOsIsolated) return false;
  // Same as capabilityPending: the startup probe has not answered yet.
  return capability.backend !== "unknown";
}

/** Switch on = High (OS sandboxing). Without an OS sandbox the calls run on software sandboxing,
 *  so the switch shows Low whatever was saved. Full access drops the sandbox, so the switch is
 *  disabled there and keeps showing the level. */
export function sandboxSwitchState(
  level: SandboxLevel,
  permissionMode: PermissionMode,
  capability: SandboxCapability | null = null,
): { checked: boolean; disabled: boolean } {
  return {
    checked: level === "high" && !osSandboxMissing(capability),
    disabled: permissionMode === "full",
  };
}
