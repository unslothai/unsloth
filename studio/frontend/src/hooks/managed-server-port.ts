// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const MANAGED_STARTUP_POLL_MS = 500;

export type ManagedStartupResult =
  | { status: "ready"; port: number }
  | { status: "aborted" };

/** A poll interval that `notify` can cut short, so a server-port event is acted on at once. */
export interface PortWakeSignal {
  notify: () => void;
  wait: (ms: number) => Promise<void>;
}

export function createPortWakeSignal(): PortWakeSignal {
  let wake: (() => void) | null = null;
  return {
    notify() {
      const current = wake;
      wake = null;
      current?.();
    },
    wait(ms: number) {
      return new Promise<void>((resolve) => {
        const done = () => {
          clearTimeout(timer);
          if (wake === done) wake = null;
          resolve();
        };
        const timer = setTimeout(done, ms);
        wake = done;
      });
    },
  };
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function waitForManagedServerPort(
  getPort: () => number | null,
  shouldContinue: () => boolean,
  wait: (ms: number) => Promise<void> = sleep,
): Promise<ManagedStartupResult> {
  while (true) {
    if (!shouldContinue()) {
      return { status: "aborted" };
    }

    const port = getPort();
    if (port === null) {
      await wait(MANAGED_STARTUP_POLL_MS);
      continue;
    }

    return { status: "ready", port };
  }
}
