import { useRecipeExecutions } from "./use-recipe-executions";
import { useRecipePersistence } from "./use-recipe-persistence";
import type { RecipeExecutionRecord } from "../execution-types";
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
  workflowName: string;
  setWorkflowName: (value: string) => void;
  saveLoading: boolean;
  saveTone: SaveTone;
  savedAtLabel: string;
  copied: boolean;
  importOpen: boolean;
  setImportOpen: (open: boolean) => void;
  previewDialogOpen: boolean;
  setPreviewDialogOpen: (open: boolean) => void;
  previewRows: number;
  setPreviewRows: (rows: number) => void;
  previewErrors: string[];
  previewLoading: boolean;
  fullLoading: boolean;
  currentSignature: string;
  executions: RecipeExecutionRecord[];
  selectedExecutionId: string | null;
  setSelectedExecutionId: (id: string) => void;
  persistRecipe: () => Promise<void>;
  openPreviewDialog: () => void;
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
    onExecutionStart,
    onPreviewSuccess,
  });

  return {
    workflowName: persistence.workflowName,
    setWorkflowName: persistence.setWorkflowName,
    saveLoading: persistence.saveLoading,
    saveTone: persistence.saveTone,
    savedAtLabel: persistence.savedAtLabel,
    copied: persistence.copied,
    importOpen: persistence.importOpen,
    setImportOpen: persistence.setImportOpen,
    previewDialogOpen: executions.previewDialogOpen,
    setPreviewDialogOpen: executions.setPreviewDialogOpen,
    previewRows: executions.previewRows,
    setPreviewRows: executions.setPreviewRows,
    previewErrors: executions.previewErrors,
    previewLoading: executions.previewLoading,
    fullLoading: executions.fullLoading,
    currentSignature: persistence.currentSignature,
    executions: executions.executions,
    selectedExecutionId: executions.selectedExecutionId,
    setSelectedExecutionId: executions.setSelectedExecutionId,
    persistRecipe: persistence.persistRecipe,
    openPreviewDialog: executions.openPreviewDialog,
    runPreview: executions.runPreview,
    runFull: executions.runFull,
    cancelExecution: executions.cancelExecution,
    loadExecutionDatasetPage: executions.loadExecutionDatasetPage,
    copyRecipe: persistence.copyRecipe,
    importRecipe: persistence.importRecipe,
  };
}
