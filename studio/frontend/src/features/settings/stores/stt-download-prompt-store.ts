// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { SttModel } from "./voice-settings-store";

/**
 * The one pending "download this dictation model?" confirmation.
 *
 * A store, not local state, so the mic can raise the dialog without Voice
 * settings being open.
 */
interface SttDownloadPromptState {
  /** Model awaiting a yes/no, or null when nothing is asked. */
  pendingModel: SttModel | null;
  requestDownload: (model: SttModel) => void;
  dismiss: () => void;
}

export const useSttDownloadPromptStore = create<SttDownloadPromptState>(
  (set) => ({
    pendingModel: null,
    requestDownload: (pendingModel) => set({ pendingModel }),
    dismiss: () => set({ pendingModel: null }),
  }),
);

/** Ask the user to download `model`. Safe to call from non-React code. */
export function requestSttDownload(model: SttModel): void {
  useSttDownloadPromptStore.getState().requestDownload(model);
}
