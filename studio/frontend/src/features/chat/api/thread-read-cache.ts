// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ThreadRecord } from "../types";

export type InFlightThreadRead = {
  promise: Promise<ThreadRecord | null>;
  controller: AbortController;
  readers: number;
  startedAt: number;
  // Auth session it was read under: a read never answers a later session.
  authEpoch: number;
};

export const inFlightThreadReads = new Map<string, InFlightThreadRead>();

export function forgetThreadReads(threadIds: readonly string[] | null): void {
  if (threadIds === null) {
    inFlightThreadReads.clear();
    return;
  }
  for (const threadId of threadIds) inFlightThreadReads.delete(threadId);
}
