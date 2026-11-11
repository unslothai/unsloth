// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  DeletedModelRef,
  ExternalModelOption,
  LoraModelOption,
  ModelConfigHandoffRequest,
  ModelOption,
  ModelSelectorChangeMeta,
  PerModelConfig,
} from "@/features/model-picker";
import { createContext, useContext } from "react";

export interface ComposerModelPickerSource {
  models: readonly ModelOption[];
  loraModels: readonly LoraModelOption[];
  externalModels: readonly ExternalModelOption[];
  onPick: (id: string, meta: ModelSelectorChangeMeta) => void;
  eject: (modelId?: string) => void;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Run settings asked for from elsewhere (the Hub's Run, the API page). */
  configRequest: ModelConfigHandoffRequest | null;
  onConfigRequestAdopted: (requestId: string) => void;
  onFoldersChange: () => void;
  onModelsChange: (deleted?: DeletedModelRef) => void;
  deleteDisabled: boolean;
  activeModelConfig: PerModelConfig | null;
  loadedContextLength: number | null;
  loaded: boolean;
}

export const ComposerModelPickerContext =
  createContext<ComposerModelPickerSource | null>(null);

export function useComposerModelPickerSource(): ComposerModelPickerSource | null {
  return useContext(ComposerModelPickerContext);
}
