// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// eslint-disable-next-line no-restricted-imports -- the feature barrel imports consumers of this hook
import { useChatPreferencesStore } from "@/features/chat/stores/chat-preferences-store";
// eslint-disable-next-line no-restricted-imports -- this file is in the startup cycle; the chat barrel closes it.
import { defaultOpenFor } from "@/features/chat/utils/display-visibility";
import { useEffect, useRef, useState } from "react";
import {
  resolveToolActivityOpen,
  startsNewToolRound,
} from "./tool-activity-open-state";

export function useToolActivityOpen(isRunning: boolean, hasText: boolean) {
  const visibility = useChatPreferencesStore((state) => state.toolVisibility);
  const [state, setState] = useState(() => ({
    open: defaultOpenFor(visibility, isRunning),
    override: null as boolean | null,
  }));
  const previousVisibility = useRef(visibility);
  const previousRunning = useRef(isRunning);

  useEffect(() => {
    const previous = previousVisibility.current;
    previousVisibility.current = visibility;
    // Regenerate reuses this card, so a re-run of the same call would otherwise inherit the
    // previous round's hand-set state. Same rule the reasoning block applies to its round.
    const startedNewRound = startsNewToolRound(isRunning, previousRunning.current);
    previousRunning.current = isRunning;
    setState((current) => {
      const override =
        previous === visibility && !startedNewRound ? current.override : null;
      return {
        override,
        open: resolveToolActivityOpen({
          currentOpen: current.open,
          visibility,
          previousVisibility: previous,
          isRunning,
          hasText,
          override,
          startedNewRound,
        }),
      };
    });
  }, [isRunning, hasText, visibility]);

  return [
    state.open,
    (open: boolean) => setState({ open, override: open }),
  ] as const;
}
