// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  DeletedModelRef,
  ExternalConnectionRef,
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
  ejectAll: () => void;
  /** So a model whose connection is gone keeps its name. */
  externalConnections: readonly ExternalConnectionRef[];
  open: boolean;
  onOpenChange: (open: boolean) => void;
  configRequest: ModelConfigHandoffRequest | null;
  onConfigRequestAdopted: (requestId: string) => void;
  onFoldersChange: () => void;
  onModelsChange: (deleted?: DeletedModelRef) => void;
  deleteDisabled: boolean;
  activeModelConfig: PerModelConfig | null;
  loadedContextLength: number | null;
  loaded: boolean;
  /** False where only local models run (the API page): no Connected or provider tabs. */
  connections?: boolean;
  /** A choice owned outside Chat (Compare, API page): no reasoning controls, Eject only while `loaded`. */
  selection?: {
    id: string;
    ggufVariant: string | null;
    loaded?: boolean;
    loading?: boolean;
    placeholder?: string;
    /** The settings this selection runs with, seeding its Run settings. */
    config?: PerModelConfig | null;
  };
}

export const ComposerModelPickerContext =
  createContext<ComposerModelPickerSource | null>(null);

export function useComposerModelPickerSource(): ComposerModelPickerSource | null {
  return useContext(ComposerModelPickerContext);
}
