// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { clearStoredChats, countStoredChats } from "./chat-history-storage";
import { requestPromptQueueStop } from "./prompt-queue-boundary";

export const countAllChats = countStoredChats;

export async function clearAllChats() {
  requestPromptQueueStop();
  return clearStoredChats();
}
