// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports -- the feature barrel imports consumers of this hook
import { useChatPreferencesStore } from "@/features/chat/stores/chat-preferences-store";
import { useEffect, useRef, useState } from "react";
import { resolveToolActivityOpen } from "./tool-activity-open-state";

export function useToolActivityOpen(isRunning: boolean, hasText: boolean) {
  const collapseByDefault = useChatPreferencesStore(
    (state) => state.collapseToolActivityByDefault,
  );
  const [open, setOpen] = useState(isRunning && !collapseByDefault);
  const previousCollapseByDefault = useRef(collapseByDefault);

  useEffect(() => {
    const previousPreference = previousCollapseByDefault.current;
    previousCollapseByDefault.current = collapseByDefault;
    setOpen((currentOpen) =>
      resolveToolActivityOpen({
        currentOpen,
        collapseByDefault,
        previousCollapseByDefault: previousPreference,
        isRunning,
        hasText,
      }),
    );
  }, [isRunning, hasText, collapseByDefault]);

  return [open, setOpen] as const;
}
