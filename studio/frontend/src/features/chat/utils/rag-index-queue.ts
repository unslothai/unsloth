// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "../stores/chat-runtime-store";

/** Bounds how many documents index in parallel. Each RAG upload
 *  acquires a slot before it starts and releases it once its ingestion
 *  job finishes (complete/error/already-indexed), so bulk/folder uploads
 *  drain at the user-configured `ragIndexConcurrency` rate instead of
 *  spawning a subprocess per file at once. Module-scoped singleton,
 *  shared across both composer surfaces. */

let active = 0;
const waiters: Array<() => void> = [];

function limit(): number {
  const n = useChatRuntimeStore.getState().ragIndexConcurrency;
  return Math.max(1, Number.isFinite(n) ? Math.round(n) : 1);
}

function admitWaiters(): void {
  while (waiters.length > 0 && active < limit()) {
    active += 1;
    const next = waiters.shift();
    next?.();
  }
}

/** Resolves once a slot is free (immediately if under the limit). */
export function acquireIndexSlot(): Promise<void> {
  return new Promise<void>((resolve) => {
    waiters.push(resolve);
    admitWaiters();
  });
}

/** Release a previously-acquired slot and admit the next waiter. */
export function releaseIndexSlot(): void {
  active = Math.max(0, active - 1);
  admitWaiters();
}
