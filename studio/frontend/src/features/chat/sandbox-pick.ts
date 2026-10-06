// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type SandboxCapability,
  cachedSandboxCapability,
  capabilityPending,
  loadSettledSandboxCapability,
  sandboxReady,
} from "./api/sandbox-capability";
import type { PermissionMode } from "./stores/chat-runtime-store";

let sandboxedPicks = 0;

/** Re-picking "Run automatically" without a working sandbox reopens the setup dialog. */
export function samePickIsIgnored(
  value: PermissionMode,
  current: PermissionMode,
  sandboxUnavailable: boolean,
): boolean {
  return value === current && !(value === "off" && sandboxUnavailable);
}

/** Without a working OS sandbox, offers setup instead of applying. A stale read does nothing. */
export function pickSandboxedMode(
  setPermissionMode: (mode: PermissionMode) => void,
  onRequestSandboxSetup: () => void,
  currentMode: () => PermissionMode,
  load: () => Promise<SandboxCapability | null> = loadSettledSandboxCapability,
  watchModeChanges?: (onChange: () => void) => () => void,
  peek: () => SandboxCapability | null = cachedSandboxCapability,
): Promise<void> {
  const pick = ++sandboxedPicks;
  // Known ready: apply at once instead of waiting on a fresh probe. The backend still checks each call.
  const known = peek();
  if (known !== null && sandboxReady(known)) {
    setPermissionMode("off");
    return Promise.resolve();
  }
  const before = currentMode();
  let changed = false;
  const stopWatching = watchModeChanges?.(() => {
    changed = true;
  });
  return load().then((capability) => {
    stopWatching?.();
    if (changed || pick !== sandboxedPicks || currentMode() !== before) return;
    // Still unknown: apply it; the backend asks before risky calls until isolation is confirmed.
    if (
      capability !== null &&
      !capabilityPending(capability) &&
      !sandboxReady(capability)
    ) {
      onRequestSandboxSetup();
    } else {
      setPermissionMode("off");
    }
  });
}
