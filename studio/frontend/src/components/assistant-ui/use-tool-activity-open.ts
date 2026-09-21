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
  const [state, setState] = useState(() => ({
    open: defaultOpenFor(visibility, isRunning),
    // null until the user clicks the trigger, so a controlled card can keep a manual open
    // the same way the uncontrolled cards and groups do.
    override: null as boolean | null,
  }));
  const previousVisibility = useRef(visibility);

  useEffect(() => {
    const previous = previousVisibility.current;
    previousVisibility.current = visibility;
    setState((current) => {
      // A setting change hands the card back to the setting, matching the uncontrolled cards.
      const override = previous === visibility ? current.override : null;
      return {
        override,
        open: resolveToolActivityOpen({
          currentOpen: current.open,
          visibility,
          previousVisibility: previous,
          isRunning,
          hasText,
          override,
        }),
      };
    });
  }, [isRunning, hasText, visibility]);

  return [
    state.open,
    (open: boolean) => setState({ open, override: open }),
  ] as const;
}
