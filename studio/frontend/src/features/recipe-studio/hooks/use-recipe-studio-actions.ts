// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useRecipeExecutions } from "./use-recipe-executions";
import { useRecipePersistence } from "./use-recipe-persistence";
import type {
  RecipeExecutionKind,
  RecipeExecutionRecord,
} from "../execution-types";
import type { RecipeRunSettings } from "../stores/recipe-executions";
import type { RecipeSnapshot } from "../utils/import";
import type { RecipePayload, RecipePayloadResult } from "../utils/payload/types";

type SaveTone = "success" | "error";

type PersistRecipeFn = (input: {
  id: string | null;
  name: string;
  payload: RecipePayload;
}) => Promise<{
  id: string;
  updatedAt: number;
}>;

type UseRecipeStudioActionsParams = {
  recipeId: string;
  initialRecipeName: string;
  initialPayload: RecipePayload;
  initialSavedAt: number;
  payloadResult: RecipePayloadResult;
  onPersistRecipe: PersistRecipeFn;
  resetRecipe: () => void;
  loadRecipe: (snapshot: RecipeSnapshot) => void;
  getCurrentPayloadFromStore: () => RecipePayload;
  onExecutionStart?: () => void;
  onPreviewSuccess?: () => void;
};

type UseRecipeStudioActionsResult = {
  initialRecipeReady: boolean;
  workflowName: string;
  setWorkflowName: (value: string) => void;
  saveLoading: boolean;
  saveTone: SaveTone;
  savedAtLabel: string;
  copied: boolean;
  importOpen: boolean;
  setImportOpen: (open: boolean) => void;
  runDialogOpen: boolean;
  runDialogKind: RecipeExecutionKind;
  setRunDialogKind: (kind: RecipeExecutionKind) => void;
  setRunDialogOpen: (open: boolean) => void;
  previewRows: number;
  fullRows: number;
  fullRunName: string;
  setPreviewRows: (rows: number) => void;
  setFullRows: (rows: number) => void;
  setFullRunName: (name: string) => void;
  runErrors: string[];
  runSettings: RecipeRunSettings;
  setRunSettings: (patch: Partial<RecipeRunSettings>) => void;
  previewLoading: boolean;
  fullLoading: boolean;
  currentSignature: string;
  executions: RecipeExecutionRecord[];
  selectedExecutionId: string | null;
  setSelectedExecutionId: (id: string) => void;
  persistRecipe: () => Promise<void>;
  openRunDialog: (kind: RecipeExecutionKind) => void;
  runFromDialog: () => Promise<boolean>;
  validateFromDialog: () => Promise<boolean>;
  validateLoading: boolean;
  validateResult: {
    valid: boolean;
    errors: string[];
    rawDetail: string | null;
  } | null;
  runPreview: () => Promise<boolean>;
  runFull: () => Promise<boolean>;
  cancelExecution: (id: string) => Promise<void>;
  loadExecutionDatasetPage: (id: string, page: number) => Promise<void>;
  copyRecipe: () => Promise<void>;
  importRecipe: (value: string) => string | null;
};

export function useRecipeStudioActions({
  recipeId,
  initialRecipeName,
  initialPayload,
  initialSavedAt,
  payloadResult,
  onPersistRecipe,
  resetRecipe,
  loadRecipe,
  getCurrentPayloadFromStore,
  onExecutionStart,
  onPreviewSuccess,
}: UseRecipeStudioActionsParams): UseRecipeStudioActionsResult {
  const persistence = useRecipePersistence({
    recipeId,
    initialRecipeName,
    initialPayload,
    initialSavedAt,
    payloadResult,
    onPersistRecipe,
    resetRecipe,
    loadRecipe,
    getCurrentPayloadFromStore,
  });

  const executions = useRecipeExecutions({
    recipeId,
    currentSignature: persistence.currentSignature,
    payloadResult,
    initialRunRows:
      typeof initialPayload?.run?.rows === "number"
        ? initialPayload.run.rows
        : null,
    onExecutionStart,
    onPreviewSuccess,
  });

  return {
    initialRecipeReady: persistence.initialRecipeReady,
    workflowName: persistence.workflowName,
    setWorkflowName: persistence.setWorkflowName,
    saveLoading: persistence.saveLoading,
    saveTone: persistence.saveTone,
    savedAtLabel: persistence.savedAtLabel,
    copied: persistence.copied,
    importOpen: persistence.importOpen,
    setImportOpen: persistence.setImportOpen,
    runDialogOpen: executions.runDialogOpen,
    runDialogKind: executions.runDialogKind,
    setRunDialogKind: executions.setRunDialogKind,
    setRunDialogOpen: executions.setRunDialogOpen,
    previewRows: executions.previewRows,
    fullRows: executions.fullRows,
    fullRunName: executions.fullRunName,
    setPreviewRows: executions.setPreviewRows,
    setFullRows: executions.setFullRows,
    setFullRunName: executions.setFullRunName,
    runErrors: executions.runErrors,
    runSettings: executions.runSettings,
    setRunSettings: executions.setRunSettings,
    previewLoading: executions.previewLoading,
    fullLoading: executions.fullLoading,
    currentSignature: persistence.currentSignature,
    executions: executions.executions,
    selectedExecutionId: executions.selectedExecutionId,
    setSelectedExecutionId: executions.setSelectedExecutionId,
    persistRecipe: persistence.persistRecipe,
    openRunDialog: executions.openRunDialog,
    runFromDialog: executions.runFromDialog,
    validateFromDialog: executions.validateFromDialog,
    validateLoading: executions.validateLoading,
    validateResult: executions.validateResult,
    runPreview: executions.runPreview,
    runFull: executions.runFull,
    cancelExecution: executions.cancelExecution,
    loadExecutionDatasetPage: executions.loadExecutionDatasetPage,
    copyRecipe: persistence.copyRecipe,
    importRecipe: persistence.importRecipe,
  };
}
