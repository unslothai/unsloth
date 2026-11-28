// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Session-scoped pill positions keyed by the same thread identity as the
// collection. Dragging one thread's pill never moves another's.

import type { QuotePillPosition } from "./message-quote-position";

const positionsByThread = new Map<string, QuotePillPosition>();

export function getQuotePillPosition(
  threadId: string,
): QuotePillPosition | null {
  return positionsByThread.get(threadId) ?? null;
}

export function setQuotePillPosition(
  threadId: string,
  position: QuotePillPosition,
): void {
  positionsByThread.set(threadId, {
    left: position.left,
    top: position.top,
  });
}

export function clearQuotePillPosition(threadId: string): boolean {
  return positionsByThread.delete(threadId);
}
