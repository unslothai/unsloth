// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { CheckFormatResponse } from "../types/datasets";
import type { DatasetSource } from "@/types/training";

export type DatasetPreviewDialogMode = "preview" | "mapping";

export type DatasetPreviewTarget = {
  source: DatasetSource;
  path: string;
  subset: string | null;
  split: string | null;
};

type DatasetPreviewDialogState = {
  open: boolean;
  mode: DatasetPreviewDialogMode;
  initialData: CheckFormatResponse | null;
  previewTarget: DatasetPreviewTarget | null;
};

type DatasetPreviewDialogActions = {
  openPreview: (target?: DatasetPreviewTarget) => void;
  openMapping: (data: CheckFormatResponse) => void;
  close: () => void;
};

const initialState: DatasetPreviewDialogState = {
  open: false,
  mode: "preview",
  initialData: null,
  previewTarget: null,
};

export const useDatasetPreviewDialogStore = create<
  DatasetPreviewDialogState & DatasetPreviewDialogActions
>()((set) => ({
  ...initialState,

  openPreview: (previewTarget) =>
    set({
      open: true,
      mode: "preview",
      initialData: null,
      previewTarget: previewTarget ?? null,
    }),
  openMapping: (data) =>
    set({
      open: true,
      mode: "mapping",
      initialData: data,
      previewTarget: null,
    }),
  close: () =>
    set({
      open: false,
      initialData: null,
      mode: "preview",
      previewTarget: null,
    }),
}));
