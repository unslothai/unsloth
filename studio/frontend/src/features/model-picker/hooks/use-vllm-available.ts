// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import { useSyncExternalStore } from "react";
import { listEngines, vllmHostSupported } from "../api/engines";

// One read shared by every hub row; "Checking..." (GPU probe, up to a minute) and failed reads are retried.
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
    if (request !== latest) return;
    publish(vllmHostSupported(engines));
    const vllm = engines.find((engine) => engine.engine === "vllm");
    if (vllm?.unsupported_reason?.startsWith("Checking")) retry();
  } catch {
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
  // Re-read whenever the hub opens again.
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
