// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Type-only imports keep this module loadable by the node tests.
import type { SandboxCapability } from "./api/sandbox-capability";
import type { PermissionMode, SandboxLevel } from "./stores/chat-runtime-store";

/** The permission menu's "OS sandbox is not set up" banner: only when High is asked for, Full
 *  access is not overriding it, and a settled capability answer says Python or Terminal runs
 *  without OS isolation. Unknown (no answer, an old server) or still probing shows nothing. */
export function sandboxBannerVisible(
  level: SandboxLevel,
  permissionMode: PermissionMode,
  capability: SandboxCapability | null,
): boolean {
  if (level !== "high" || permissionMode === "full" || capability === null) {
    return false;
  }
  if (capability.pythonOsIsolated && capability.terminalOsIsolated) return false;
  // Same as capabilityPending: the startup probe has not answered yet.
  return capability.backend !== "unknown";
}

/** Switch on = High. Full access drops the sandbox, so the switch is disabled there and keeps
 *  showing the saved level. */
export function sandboxSwitchState(
  level: SandboxLevel,
  permissionMode: PermissionMode,
): { checked: boolean; disabled: boolean } {
  return { checked: level === "high", disabled: permissionMode === "full" };
}
