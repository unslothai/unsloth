// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useUiSpaceScale } from "@/hooks/use-ui-space-scale";
import { useSyncExternalStore } from "react";

/**
 * Stack the tab rail over the pane when the dialog is narrower than it is at
 * sm (608px, 640px less its 2rem margin) scaled by the UI. At 100% that is
 * max-sm exactly; at 200% the 960px cap is always too narrow.
 */
export function useStackedLayout(): boolean {
  const width = 608 * useUiSpaceScale();
  const query = `(width < ${width + 32}px)`;
  const narrow = useSyncExternalStore(
    (onChange) => {
      const list = window.matchMedia(query);
      list.addEventListener("change", onChange);
      return () => list.removeEventListener("change", onChange);
    },
    () => window.matchMedia(query).matches,
    () => false,
  );
  return width > 960 || narrow;
}
