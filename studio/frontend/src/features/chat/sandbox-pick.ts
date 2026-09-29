// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type SandboxCapability,
  loadSandboxCapability,
  sandboxReady,
} from "./api/sandbox-capability";
import type { PermissionMode } from "./stores/chat-runtime-store";

let sandboxedPicks = 0;

/** Picking the row that is already selected does nothing, except "Full access in sandbox" on a
 *  computer without a working OS sandbox (e.g. restored from a saved chat): that pick is the way
 *  back to the setup dialog. */
export function samePickIsIgnored(
  value: PermissionMode,
  current: PermissionMode,
  sandboxUnavailable: boolean,
): boolean {
  return value === current && !(value === "off" && sandboxUnavailable);
}

/** "Full access in sandbox" only holds with a working OS sandbox, so picking it without one
 *  offers the setup instead of applying it; nothing is installed until the owner asks. A read
 *  that comes back after the user moved on (another pick, or any mode change, even one that
 *  ends on the same mode) does nothing. `watchModeChanges` reports every change while the read
 *  is pending; without it only a different current mode counts. */
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
