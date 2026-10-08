// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import { useSyncExternalStore } from "react";
import { listEngines, vllmHostSupported } from "../api/engines";

// One request shared by every catalog row, not a poller per row. The backend answers "Checking
// for a supported NVIDIA GPU." while its first driver probe runs, so that answer is re-asked.
let available = false;
let started = false;
let retries = 0;
const listeners = new Set<() => void>();

function publish(next: boolean) {
  if (next === available) return;
  available = next;
  for (const listener of listeners) listener();
}

async function refresh() {
  try {
    const engines = await listEngines();
    publish(vllmHostSupported(engines));
    const vllm = engines.find((engine) => engine.engine === "vllm");
    if (vllm?.unsupported_reason?.startsWith("Checking") && retries < 10) {
      retries += 1;
      setTimeout(() => void refresh(), 3000);
    }
  } catch {
    // An older backend without /api/engines, or a failed read: keep today's labels.
    publish(false);
  }
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  if (!started) {
    started = true;
    window.addEventListener("studio-engines-changed", () => {
      retries = 0;
      void refresh();
    });
    void refresh();
  }
  return () => {
    listeners.delete(listener);
  };
}

export function useVllmAvailable(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => available,
    () => false,
  );
}
