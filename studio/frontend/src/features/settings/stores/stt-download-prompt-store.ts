// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { SttModel } from "./voice-settings-store";

export interface SttDownloadRequest {
  model: SttModel;
  /** pinned quant for a package folder model. */
  ggufVariant?: string | null;
  /** switches dictation to local only after confirmation when browser speech is unavailable. */
  selectLocalEngine?: boolean;
}

/** pending download confirmation shared so the mic can open it without Voice settings mounted. */
interface SttDownloadPromptState {
  /** request awaiting confirmation, or null when idle. */
  pending: SttDownloadRequest | null;
  requestDownload: (request: SttDownloadRequest) => void;
  dismiss: () => void;
}

export const useSttDownloadPromptStore = create<SttDownloadPromptState>(
  (set) => ({
    pending: null,
    requestDownload: (pending) => set({ pending }),
    dismiss: () => set({ pending: null }),
  }),
);

/** Ask the user to download `model`. Safe to call from non-React code. */
export function requestSttDownload(
  model: SttModel,
  options?: Omit<SttDownloadRequest, "model">,
): void {
  useSttDownloadPromptStore.getState().requestDownload({ model, ...options });
}
