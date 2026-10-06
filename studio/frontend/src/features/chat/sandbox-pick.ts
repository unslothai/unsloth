// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type SandboxCapability,
  cachedSandboxCapability,
  capabilityPending,
  loadSettledSandboxCapability,
  sandboxReady,
} from "./api/sandbox-capability";
import type { SandboxLevel } from "./stores/chat-runtime-store";

let levelPicks = 0;

/** Switching to High without a working OS sandbox opens the setup popup instead of applying;
 *  Low always applies. A stale read (a later switch) does nothing. */
export function pickSandboxLevel(
  next: SandboxLevel,
  setSandboxLevel: (level: SandboxLevel) => void,
  onOsSandboxMissing: () => void,
  load: () => Promise<SandboxCapability | null> = loadSettledSandboxCapability,
  peek: () => SandboxCapability | null = cachedSandboxCapability,
): Promise<void> {
  const pick = ++levelPicks;
  if (next === "low") {
    setSandboxLevel("low");
    return Promise.resolve();
  }
  const known = peek();
  if (known !== null && sandboxReady(known)) {
    setSandboxLevel("high");
    return Promise.resolve();
  }
  return load().then((capability) => {
    if (pick !== levelPicks) return;
    if (
      capability !== null &&
      !capabilityPending(capability) &&
      !sandboxReady(capability)
    ) {
      onOsSandboxMissing();
    } else {
      setSandboxLevel("high");
    }
  });
}
