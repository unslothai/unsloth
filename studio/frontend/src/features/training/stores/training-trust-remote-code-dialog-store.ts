// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

type ConfirmationResolver = (confirmed: boolean) => void;

type TrainingTrustRemoteCodeDialogStore = {
  open: boolean;
  modelName: string | null;
  requestConfirmation: (modelName?: string | null) => Promise<boolean>;
  resolve: (confirmed: boolean) => void;
};

let pendingResolver: ConfirmationResolver | null = null;

export const useTrainingTrustRemoteCodeDialogStore =
  create<TrainingTrustRemoteCodeDialogStore>()((set) => ({
    open: false,
    modelName: null,
    requestConfirmation: (modelName = null) =>
      new Promise<boolean>((resolve) => {
        pendingResolver?.(false);
        pendingResolver = resolve;
        set({ open: true, modelName });
      }),
    resolve: (confirmed) => {
      const resolver = pendingResolver;
      pendingResolver = null;
      set({ open: false, modelName: null });
      resolver?.(confirmed);
    },
  }));
