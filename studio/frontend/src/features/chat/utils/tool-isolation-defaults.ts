// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PermissionMode } from "../stores/chat-runtime-store";
import type { ToolExecutionMode } from "../tool-isolation";

/** The store fields a "return to protected defaults" transition rewrites. */
export type ProtectedIsolationDefaults = {
  toolExecutionMode: ToolExecutionMode;
  limitedToolGrant: null;
  bypassPermissions: false;
  permissionMode: PermissionMode;
  confirmToolCalls: boolean;
  toolIsolationConsentOpen: false;
};

/**
 * One transition for every path that must leave Full or Limited access behind: the auth
 * session changed, Deep Research took over the composer, or a chat switch pinned a lower
 * level. Full and Limited are session decisions, never persisted, so the only level that may
 * survive is the persisted one handed in here. A Full level cannot be restored through this
 * helper; it clamps to "auto" so no caller can re-enter Full by accident.
 */
export function protectedIsolationDefaults(
  permissionMode: PermissionMode,
): ProtectedIsolationDefaults {
  const level: PermissionMode =
    permissionMode === "full" ? "auto" : permissionMode;
  return {
    toolExecutionMode: "os_isolation_required",
    limitedToolGrant: null,
    bypassPermissions: false,
    permissionMode: level,
    confirmToolCalls: level === "ask" || level === "auto",
    toolIsolationConsentOpen: false,
  };
}
