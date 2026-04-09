// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useState } from "react";

const PINNED_KEY = "sidebar_pinned";

export function useSidebarPin() {
  const [pinned, setPinnedState] = useState(() => {
    try {
      return localStorage.getItem(PINNED_KEY) === "true";
    } catch {
      return false;
    }
  });
  const [hovered, setHovered] = useState(false);

  const setPinned = useCallback((value: boolean) => {
    setPinnedState(value);
    try {
      localStorage.setItem(PINNED_KEY, String(value));
    } catch {}
  }, []);

  const togglePinned = useCallback(() => {
    setPinnedState((prev) => {
      const next = !prev;
      try {
        localStorage.setItem(PINNED_KEY, String(next));
      } catch {}
      return next;
    });
  }, []);

  return { pinned, setPinned, togglePinned, hovered, setHovered };
}
