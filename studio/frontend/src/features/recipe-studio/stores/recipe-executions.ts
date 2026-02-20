import { create } from "zustand";
import type { RecipeExecutionRecord } from "../execution-types";
import { sortExecutions, withExecutionDefaults } from "../executions/execution-helpers";

type RecipeExecutionsState = {
  previewDialogOpen: boolean;
  previewRows: number;
  previewErrors: string[];
  previewLoading: boolean;
  fullLoading: boolean;
  executions: RecipeExecutionRecord[];
  selectedExecutionId: string | null;
  setPreviewDialogOpen: (open: boolean) => void;
  setPreviewRows: (rows: number) => void;
  setPreviewErrors: (errors: string[]) => void;
  setPreviewLoading: (loading: boolean) => void;
  setFullLoading: (loading: boolean) => void;
  setExecutions: (records: RecipeExecutionRecord[]) => void;
  upsertExecution: (record: RecipeExecutionRecord) => void;
  selectExecution: (id: string | null) => void;
  resetForRecipe: () => void;
};

const INITIAL_STATE = {
  previewDialogOpen: false,
  previewRows: 5,
  previewErrors: [],
  previewLoading: false,
  fullLoading: false,
  executions: [],
  selectedExecutionId: null,
} satisfies Pick<
  RecipeExecutionsState,
  | "previewDialogOpen"
  | "previewRows"
  | "previewErrors"
  | "previewLoading"
  | "fullLoading"
  | "executions"
  | "selectedExecutionId"
>;

export const useRecipeExecutionsStore = create<RecipeExecutionsState>((set) => ({
  ...INITIAL_STATE,
  setPreviewDialogOpen: (open) => set({ previewDialogOpen: open }),
  setPreviewRows: (rows) =>
    set({ previewRows: Number.isFinite(rows) && rows > 0 ? Math.floor(rows) : 1 }),
  setPreviewErrors: (errors) => set({ previewErrors: errors }),
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
