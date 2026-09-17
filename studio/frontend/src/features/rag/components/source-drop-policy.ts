// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isSupportedSourceName } from "../types/rag.ts";

/** Split a drop into what can be indexed and the names of what cannot, so the
 * caller can report the rejects instead of discarding them silently. */
export function partitionSupported<T>(
  entries: T[],
  nameOf: (entry: T) => string,
): { supported: T[]; unsupported: string[] } {
  const supported: T[] = [];
  const unsupported: string[] = [];
  for (const entry of entries) {
    const name = nameOf(entry);
    if (isSupportedSourceName(name)) supported.push(entry);
    else unsupported.push(name);
  }
  return { supported, unsupported };
}
