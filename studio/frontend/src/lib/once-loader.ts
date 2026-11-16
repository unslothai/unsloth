// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type OnceLoader<T> = {
  /** The loaded value, or null until the first load resolves. */
  current: () => T | null;
  /** Starts the load once; concurrent callers share it, and a failure lets the next call retry. */
  load: () => Promise<T>;
};

export function onceLoader<T>(load: () => Promise<T>): OnceLoader<T> {
  let value: { loaded: T } | null = null;
  let pending: Promise<T> | null = null;
  return {
    current: () => (value ? value.loaded : null),
    load: () => {
      if (value) return Promise.resolve(value.loaded);
      pending ??= load().then(
        (loaded) => {
          value = { loaded };
          return loaded;
        },
        (error: unknown) => {
          pending = null;
          throw error;
        },
      );
      return pending;
    },
  };
}
