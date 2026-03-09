// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { create } from "zustand";
import type { CheckFormatResponse } from "../types/datasets";

export type DatasetPreviewDialogMode = "preview" | "mapping";

type DatasetPreviewDialogState = {
  open: boolean;
  mode: DatasetPreviewDialogMode;
  initialData: CheckFormatResponse | null;
};

type DatasetPreviewDialogActions = {
  openPreview: () => void;
  openMapping: (data: CheckFormatResponse) => void;
  close: () => void;
};

const initialState: DatasetPreviewDialogState = {
  open: false,
  mode: "preview",
  initialData: null,
};

export const useDatasetPreviewDialogStore = create<
  DatasetPreviewDialogState & DatasetPreviewDialogActions
>()((set) => ({
  ...initialState,

  openPreview: () => set({ open: true, mode: "preview", initialData: null }),
  openMapping: (data) => set({ open: true, mode: "mapping", initialData: data }),
  close: () => set({ open: false, initialData: null, mode: "preview" }),
}));

