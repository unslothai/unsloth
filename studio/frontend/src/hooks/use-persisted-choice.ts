// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useState } from "react";

/** For a GPU pick that status cannot reseed. The stored value is only a hint: callers must check it
 * against the live inventory, since a vanished card would 400. Storage failures are tolerated. */
export function usePersistedChoice(
  key: string,
  fallback: string,
): [string, (next: string) => void] {
  const [value, setValueState] = useState(() => {
    try {
      return localStorage.getItem(key) ?? fallback;
    } catch {
      return fallback;
    }
  });
  const setValue = useCallback(
    (next: string) => {
      setValueState(next);
      try {
        if (next === fallback) {
          localStorage.removeItem(key);
        } else {
          localStorage.setItem(key, next);
        }
      } catch {
        // storage unavailable
      }
    },
    [key, fallback],
  );
  return [value, setValue];
}
