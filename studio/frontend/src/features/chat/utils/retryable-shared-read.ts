// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function createRetryableSharedRead<T>(
  read: () => Promise<T>,
  shouldCache: (value: T) => boolean = () => true,
): () => Promise<T> {
  let shared: Promise<T> | null = null;
  return () => {
    if (shared) {
      return shared;
    }
    const current = Promise.resolve()
      .then(read)
      .then((value) => {
        if (!shouldCache(value) && shared === current) {
          shared = null;
        }
        return value;
      })
      .catch((error) => {
        if (shared === current) {
          shared = null;
        }
        throw error;
      });
    shared = current;
    return current;
  };
}
