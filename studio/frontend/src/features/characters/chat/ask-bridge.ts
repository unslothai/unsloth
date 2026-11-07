// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ThreadMessage } from "@assistant-ui/react";
import { create } from "zustand";

type Ask = (characterId: string, name: string) => void;
type Messages = () => readonly ThreadMessage[];

interface Bridge {
  threadId: string | null;
  ask: Ask | null;
  messages: Messages | null;
}

/** Lets UI outside the chat runtime (the side panel) reach the chat on screen: ask a character to
 *  reply, or read the branch being shown. */
export const useAskBridge = create<Bridge>(() => ({ threadId: null, ask: null, messages: null }));

export function registerAsk(threadId: string, ask: Ask, messages: Messages): () => void {
  useAskBridge.setState({ threadId, ask, messages });
  return () => {
    if (useAskBridge.getState().ask === ask) useAskBridge.setState({ threadId: null, ask: null, messages: null });
  };
}
