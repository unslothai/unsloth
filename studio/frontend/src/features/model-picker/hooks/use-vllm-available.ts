// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import { useSyncExternalStore } from "react";
import { listEngines, vllmHostSupported } from "../api/engines";

// One request shared by every catalog row, not a poller per row. The backend answers "Checking
// for a supported NVIDIA GPU." while its first driver probe runs (up to a minute on a loaded
// host), and a read can fail while it restarts, so both are re-asked for about two minutes.
let available = false;
let started = false;
let retries = 0;
let latest = 0;
const listeners = new Set<() => void>();

function publish(next: boolean) {
  if (next === available) return;
  available = next;
  for (const listener of listeners) listener();
}

function retry() {
  if (retries >= 24) return;
  retries += 1;
  setTimeout(() => void refresh(), 5000);
}

async function refresh() {
  const request = ++latest;
  try {
    const engines = await listEngines();
    // An older read finishing late must not overwrite a newer answer.
    if (request !== latest) return;
    publish(vllmHostSupported(engines));
    const vllm = engines.find((engine) => engine.engine === "vllm");
    if (vllm?.unsupported_reason?.startsWith("Checking")) retry();
  } catch {
    // An older backend without /api/engines stays false (today's labels); a passing failure
    // keeps the last answer.
    if (request === latest) retry();
  }
}

function subscribe(listener: () => void) {
  const first = listeners.size === 0;
  listeners.add(listener);
  if (!started) {
    started = true;
    window.addEventListener("studio-engines-changed", () => {
      retries = 0;
      void refresh();
    });
  }
  // Re-read each time the hub opens again, since the host or the backend may have changed.
  if (first) {
    retries = 0;
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
