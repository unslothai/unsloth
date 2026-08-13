// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useState } from "react";

/** A string choice that survives reloads, falling back to `fallback`.
 *
 * The string sibling of `usePersistedToggle`, for a control whose value cannot be reseeded from
 * the loaded build: the backend reports the device a pipeline is on but not the physical card id,
 * so a refresh would otherwise reset the GPU pick to Auto while the model stayed where it was, and
 * the next Reapply would quietly move it. The stored value is only ever a hint, so the caller
 * still has to check it against the live inventory: a card that has since gone (a driver reset, an
 * eGPU unplugged) must fall back to automatic rather than be sent to a backend that would 400 it.
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
