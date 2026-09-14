// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useState } from "react";

/** A string choice that survives reloads, falling back to `fallback`.
 *
 * The string sibling of `usePersistedToggle`, for a control that cannot be reseeded from the
 * loaded build: the backend reports the device a pipeline is on but not the physical card, so a
 * refresh would reset the GPU pick to Auto while the model stayed put and the next Reapply moved
 * it. The stored value is only a hint, so the caller must check it against the live inventory: a
 * card that has gone (driver reset, eGPU unplugged) falls back to automatic rather than being
 * sent to a backend that would 400 it.
 *
 * Storage failures keep the control working for the session.
 */
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
