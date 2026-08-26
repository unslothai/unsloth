// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatPreferencesStore } from "@/features/chat/stores/chat-preferences-store";
import { useEffect, useState } from "react";

/** Shared automatic visibility for tool cards that manage their own open state. */
export function useToolActivityOpen(isRunning: boolean, hasText: boolean) {
  const collapseByDefault = useChatPreferencesStore(
    (state) => state.collapseToolActivityByDefault,
  );
  const [open, setOpen] = useState(isRunning && !collapseByDefault);

  useEffect(() => {
    if (collapseByDefault) {
      setOpen(false);
    } else if (isRunning) {
      setOpen(true);
    } else if (hasText) {
      setOpen(false);
    }
  }, [isRunning, hasText, collapseByDefault]);

  return [open, setOpen] as const;
}
