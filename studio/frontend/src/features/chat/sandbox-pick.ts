// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type SandboxCapability,
  loadSandboxCapability,
  sandboxReady,
} from "./api/sandbox-capability";
import type { PermissionMode } from "./stores/chat-runtime-store";

let sandboxedPicks = 0;

/** "Full access in sandbox" only holds with a working OS sandbox, so picking it without one
 *  offers the setup instead of applying it; nothing is installed until the owner asks. A read
 *  that comes back after the user moved on (another pick, or another mode) does nothing. */
export function pickSandboxedMode(
  setPermissionMode: (mode: PermissionMode) => void,
  onRequestSandboxSetup: () => void,
  currentMode: () => PermissionMode,
  load: () => Promise<SandboxCapability | null> = loadSandboxCapability,
): Promise<void> {
  const pick = ++sandboxedPicks;
  const before = currentMode();
  return load().then((capability) => {
    if (pick !== sandboxedPicks || currentMode() !== before) return;
    if (capability !== null && !sandboxReady(capability)) {
      onRequestSandboxSetup();
    } else {
      setPermissionMode("off");
    }
  });
}
