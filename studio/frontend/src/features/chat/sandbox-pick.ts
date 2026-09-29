// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type SandboxCapability,
  loadSandboxCapability,
  sandboxReady,
} from "./api/sandbox-capability";
import type { PermissionMode } from "./stores/chat-runtime-store";

let sandboxedPicks = 0;

/** Re-picking "Full access in sandbox" without a working sandbox reopens the setup dialog. */
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
  load: () => Promise<SandboxCapability | null> = loadSandboxCapability,
  watchModeChanges?: (onChange: () => void) => () => void,
): Promise<void> {
  const pick = ++sandboxedPicks;
  const before = currentMode();
  let changed = false;
  const stopWatching = watchModeChanges?.(() => {
    changed = true;
  });
  return load().then((capability) => {
    stopWatching?.();
    if (changed || pick !== sandboxedPicks || currentMode() !== before) return;
    if (capability !== null && !sandboxReady(capability)) {
      onRequestSandboxSetup();
    } else {
      setPermissionMode("off");
    }
  });
}
