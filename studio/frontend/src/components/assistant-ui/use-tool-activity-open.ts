// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports -- the feature barrel imports consumers of this hook
import { useChatPreferencesStore } from "@/features/chat/stores/chat-preferences-store";
// eslint-disable-next-line no-restricted-imports -- this file is in the startup cycle; the chat barrel closes it.
import { defaultOpenFor } from "@/features/chat/utils/display-visibility";
import { useEffect, useRef, useState } from "react";
import { resolveToolActivityOpen } from "./tool-activity-open-state";

export function useToolActivityOpen(isRunning: boolean, hasText: boolean) {
  const visibility = useChatPreferencesStore((state) => state.toolVisibility);
  const [open, setOpen] = useState(() =>
    defaultOpenFor(visibility, isRunning && !hasText),
  );
  const previousVisibility = useRef(visibility);

  useEffect(() => {
    const previous = previousVisibility.current;
    previousVisibility.current = visibility;
    setOpen((currentOpen) =>
      resolveToolActivityOpen({
        currentOpen,
        visibility,
        previousVisibility: previous,
        isRunning,
        hasText,
      }),
    );
  }, [isRunning, hasText, visibility]);

  return [open, setOpen] as const;
}
