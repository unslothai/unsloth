// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useState } from "react";

/** A boolean that survives reloads, off by default. Only "true" is stored, so the
 *  default stays off. Storage failures keep the toggle working for the session. */
export function usePersistedToggle(
  key: string,
): [boolean, (next: boolean) => void] {
  const [on, setOnState] = useState(() => {
    try {
      return localStorage.getItem(key) === "true";
    } catch {
      return false;
    }
  });
  const setOn = useCallback(
    (next: boolean) => {
      setOnState(next);
      try {
        if (next) {
          localStorage.setItem(key, "true");
        } else {
          localStorage.removeItem(key);
        }
      } catch {
        // storage unavailable
      }
    },
    [key],
  );
  return [on, setOn];
}
