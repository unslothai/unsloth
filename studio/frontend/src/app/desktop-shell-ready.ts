// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { useSyncExternalStore } from "react";

// Whether the desktop app shows the app rather than its install, startup or recovery screen.
// Always true on the web. Read by the root, which sits above the wrapper that knows.
let ready = !isTauri;
const listeners = new Set<() => void>();

export function setDesktopShellReady(value: boolean): void {
  if (ready === value) return;
  ready = value;
  listeners.forEach((listener) => listener());
}

export function useDesktopShellReady(): boolean {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => ready,
  );
}
