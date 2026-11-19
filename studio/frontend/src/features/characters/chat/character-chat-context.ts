// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createContext, useContext } from "react";
import type { ThreadScene } from "./use-thread-scene";

export interface CharacterChat extends ThreadScene {
  threadId: string;
}

/** The character chat a thread renders, worked out once by CharacterStage; null for an ordinary chat. */
export const CharacterChatContext = createContext<CharacterChat | null>(null);

export function useCharacterChat(): CharacterChat | null {
  return useContext(CharacterChatContext);
}
