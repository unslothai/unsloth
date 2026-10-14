// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

/**
 * The open Rewind dialog, if any. `threadKey` is the assistant-ui thread list item id, so only
 * the Thread that asked opens it when compare panes mount several. A null `messageId` is the
 * `/rewind` picker; a message id skips straight to the choices.
 */
export interface RewindDialogRequest {
  threadKey: string;
  messageId: string | null;
}

export const useRewindDialogStore = create<{
  request: RewindDialogRequest | null;
  openRewind: (request: RewindDialogRequest) => void;
  closeRewind: () => void;
}>((set) => ({
  request: null,
  openRewind: (request) => set({ request }),
  closeRewind: () => set({ request: null }),
}));
