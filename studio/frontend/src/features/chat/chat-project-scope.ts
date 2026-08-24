// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createContext, useContext } from "react";

/**
 * The project the runtime provider on this subtree is showing.
 *
 * The composer needs it at the instant Send is pressed, to stamp the send with the project
 * it was made from (see `chat-thread-project-claim.ts`). Read through context rather than
 * the store's `activeProjectId`: this is the same value the adapter files threads under,
 * and one source for both is what makes the claim and the fallback agree.
 */
export const ChatProjectScopeContext = createContext<string | null>(null);

export function useChatProjectScope(): string | null {
  return useContext(ChatProjectScopeContext);
}
