// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { clearStoredChats, countStoredChats } from "./chat-history-storage";

export const countAllChats = countStoredChats;

export const clearAllChats = clearStoredChats;
