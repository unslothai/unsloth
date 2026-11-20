// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createContext, useContext } from "react";
import type { ThreadScene } from "./use-thread-scene";

/** The character chat a thread renders, worked out once by CharacterStage; null for an ordinary chat. */
export const CharacterChatContext = createContext<ThreadScene | null>(null);

export function useCharacterChat(): ThreadScene | null {
  return useContext(CharacterChatContext);
}

export function useInCharacterChat(): boolean {
  return useContext(CharacterChatContext) !== null;
}
