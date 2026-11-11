// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  ExternalModelOption,
  LoraModelOption,
  ModelOption,
  ModelSelectorChangeMeta,
} from "@/features/model-picker";
import { createContext, useContext } from "react";

export interface ComposerModelPickerSource {
  models: readonly ModelOption[];
  loraModels: readonly LoraModelOption[];
  externalModels: readonly ExternalModelOption[];
  onPick: (id: string, meta: ModelSelectorChangeMeta) => void;
  openFullPicker: () => void;
  eject: (modelId?: string) => void;
  openRunSettings: (
    id: string,
    displayName: string,
    meta: ModelSelectorChangeMeta,
  ) => void;
}

export const ComposerModelPickerContext =
  createContext<ComposerModelPickerSource | null>(null);

export function useComposerModelPickerSource(): ComposerModelPickerSource | null {
  return useContext(ComposerModelPickerContext);
}
