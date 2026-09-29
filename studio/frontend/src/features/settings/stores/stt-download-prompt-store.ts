// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { SttModel } from "./voice-settings-store";

export interface SttDownloadRequest {
  model: SttModel;
  /** Also switch dictation to local on confirm, for a browser whose speech
   * service cannot work at all. Left alone if the user cancels. */
  selectLocalEngine?: boolean;
}

/**
 * The one pending "download this dictation model?" confirmation.
 *
 * A store, not local state, so the mic can raise the dialog without Voice
 * settings being open.
 */
interface SttDownloadPromptState {
  /** Request awaiting a yes/no, or null when nothing is asked. */
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
