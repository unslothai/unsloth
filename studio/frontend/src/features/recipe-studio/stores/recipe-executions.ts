// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { RecipeExecutionKind } from "../execution-types";
import type { RecipeExecutionRecord } from "../execution-types";
import { sortExecutions, withExecutionDefaults } from "../executions/execution-helpers";

export type RecipeRunSettings = {
  batchSize: number;
  batchEnabled: boolean;
  mergeBatches: boolean;
  llmParallelRequests: number | null;
  nonInferenceWorkers: number;
  maxConversationRestarts: number;
  maxConversationCorrectionSteps: number;
  disableEarlyShutdown: boolean;
  shutdownErrorRate: number;
  shutdownErrorWindow: number;
};

const DEFAULT_RUN_SETTINGS: RecipeRunSettings = {
  batchSize: 1000,
  batchEnabled: false,
  mergeBatches: false,
  llmParallelRequests: null,
  nonInferenceWorkers: 4,
  maxConversationRestarts: 5,
  maxConversationCorrectionSteps: 0,
  disableEarlyShutdown: true,
  shutdownErrorRate: 0.5,
  shutdownErrorWindow: 10,
};

type RecipeExecutionsState = {
  runDialogOpen: boolean;
  runDialogKind: RecipeExecutionKind;
  previewRows: number;
  fullRows: number;
  fullRunName: string;
  runErrors: string[];
  runSettings: RecipeRunSettings;
  previewLoading: boolean;
  fullLoading: boolean;
  executions: RecipeExecutionRecord[];
  selectedExecutionId: string | null;
  setRunDialogOpen: (open: boolean) => void;
  setRunDialogKind: (kind: RecipeExecutionKind) => void;
  setPreviewRows: (rows: number) => void;
  setFullRows: (rows: number) => void;
  setFullRunName: (name: string) => void;
  setRunErrors: (errors: string[]) => void;
  setRunSettings: (patch: Partial<RecipeRunSettings>) => void;
  setPreviewLoading: (loading: boolean) => void;
  setFullLoading: (loading: boolean) => void;
  setExecutions: (records: RecipeExecutionRecord[]) => void;
  upsertExecution: (record: RecipeExecutionRecord) => void;
  selectExecution: (id: string | null) => void;
  resetForRecipe: () => void;
};

const INITIAL_STATE = {
  runDialogOpen: false,
  runDialogKind: "preview",
  previewRows: 5,
  fullRows: 100,
  fullRunName: "",
  runErrors: [],
  runSettings: DEFAULT_RUN_SETTINGS,
  previewLoading: false,
  fullLoading: false,
  executions: [],
  selectedExecutionId: null,
} satisfies Pick<
  RecipeExecutionsState,
  | "runDialogOpen"
  | "runDialogKind"
  | "previewRows"
  | "fullRows"
  | "fullRunName"
  | "runErrors"
  | "runSettings"
  | "previewLoading"
  | "fullLoading"
  | "executions"
  | "selectedExecutionId"
>;

export const useRecipeExecutionsStore = create<RecipeExecutionsState>((set) => ({
  ...INITIAL_STATE,
  setRunDialogOpen: (open) => set({ runDialogOpen: open }),
  setRunDialogKind: (kind) =>
    set((state) => {
      if (state.runDialogKind === "preview" && kind === "full") {
        return {
          runDialogKind: kind,
          fullRows: 100,
          runSettings: {
            ...state.runSettings,
            batchEnabled: false,
          },
        };
      }
      return { runDialogKind: kind };
    }),
  setPreviewRows: (rows) =>
    set({ previewRows: Number.isFinite(rows) && rows > 0 ? Math.floor(rows) : 1 }),
  setFullRows: (rows) =>
    set({ fullRows: Number.isFinite(rows) && rows > 0 ? Math.floor(rows) : 1 }),
  setFullRunName: (name) => set({ fullRunName: name }),
  setRunErrors: (errors) => set({ runErrors: errors }),
  setRunSettings: (patch) =>
    set((state) => ({
      runSettings: {
        ...state.runSettings,
        ...patch,
      },
    })),
  setPreviewLoading: (loading) => set({ previewLoading: loading }),
  setFullLoading: (loading) => set({ fullLoading: loading }),
  setExecutions: (records) =>
    set(() => {
      const normalized = sortExecutions(records.map(withExecutionDefaults));
      return {
        executions: normalized,
        selectedExecutionId: normalized[0]?.id ?? null,
      };
    }),
  upsertExecution: (record) =>
    set((state) => {
      const normalized = withExecutionDefaults(record);
      const withoutCurrent = state.executions.filter((item) => item.id !== normalized.id);
      return {
        executions: sortExecutions([normalized, ...withoutCurrent]),
        selectedExecutionId: normalized.id,
      };
    }),
  selectExecution: (id) => set({ selectedExecutionId: id }),
  resetForRecipe: () => set(INITIAL_STATE),
}));
