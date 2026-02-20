import { useCallback, useEffect, useMemo, useState } from "react";
import { toastError, toastSuccess } from "@/shared/toast";
import { normalizeNonEmptyName } from "@/utils";
import {
  buildSignature,
  copyTextToClipboard,
  formatSavedLabel,
} from "../executions/execution-helpers";
import { importRecipePayload, type RecipeSnapshot } from "../utils/import";
import type { RecipePayloadResult } from "../utils/payload/types";

type SaveTone = "success" | "error";

type PersistRecipeFn = (input: {
  id: string | null;
  name: string;
  payload: RecipePayloadResult["payload"];
}) => Promise<{
  id: string;
  updatedAt: number;
}>;

type UseRecipePersistenceParams = {
  recipeId: string;
  initialRecipeName: string;
  initialPayload: RecipePayloadResult["payload"];
  initialSavedAt: number;
  payloadResult: RecipePayloadResult;
  onPersistRecipe: PersistRecipeFn;
  resetRecipe: () => void;
  loadRecipe: (snapshot: RecipeSnapshot) => void;
  getCurrentPayloadFromStore: () => RecipePayloadResult["payload"];
};

type UseRecipePersistenceResult = {
  workflowName: string;
  setWorkflowName: (value: string) => void;
  saveLoading: boolean;
  saveTone: SaveTone;
  savedAtLabel: string;
  copied: boolean;
  importOpen: boolean;
  setImportOpen: (open: boolean) => void;
  currentSignature: string;
  persistRecipe: () => Promise<void>;
  copyRecipe: () => Promise<void>;
  importRecipe: (value: string) => string | null;
};

export function useRecipePersistence({
  recipeId,
  initialRecipeName,
  initialPayload,
  initialSavedAt,
  payloadResult,
  onPersistRecipe,
  resetRecipe,
  loadRecipe,
  getCurrentPayloadFromStore,
}: UseRecipePersistenceParams): UseRecipePersistenceResult {
  const [workflowName, setWorkflowName] = useState("Unnamed");
  const [lastSavedAt, setLastSavedAt] = useState<number | null>(null);
  const [savedSignature, setSavedSignature] = useState("");
  const [saveLoading, setSaveLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [importOpen, setImportOpen] = useState(false);

  const normalizedWorkflowName = useMemo(
    () => normalizeNonEmptyName(workflowName, "Unnamed"),
    [workflowName],
  );
  const currentPayload = payloadResult.payload;
  const currentSignature = useMemo(
    () => buildSignature(normalizedWorkflowName, currentPayload),
    [currentPayload, normalizedWorkflowName],
  );
  const isDirty = savedSignature.length > 0 && currentSignature !== savedSignature;
  const saveTone: SaveTone = !isDirty && Boolean(lastSavedAt) ? "success" : "error";
  const savedAtLabel = formatSavedLabel(lastSavedAt);
  const payloadErrorMessage = payloadResult.errors[0] ?? "Invalid payload.";

  useEffect(() => {
    const nextName = normalizeNonEmptyName(initialRecipeName, "Unnamed");
    resetRecipe();
    setWorkflowName(nextName);
    setLastSavedAt(initialSavedAt);
    setCopied(false);

    const parsed = importRecipePayload(JSON.stringify(initialPayload));
    if (parsed.snapshot) {
      loadRecipe(parsed.snapshot);
    } else {
      console.error("Failed to load recipe payload.", parsed.errors);
    }

    const payload = getCurrentPayloadFromStore();
    setSavedSignature(buildSignature(nextName, payload));
  }, [
    getCurrentPayloadFromStore,
    initialPayload,
    initialRecipeName,
    initialSavedAt,
    loadRecipe,
    recipeId,
    resetRecipe,
  ]);

  const persistRecipe = useCallback(async (): Promise<void> => {
    if (saveLoading) {
      return;
    }
    const nextName = normalizeNonEmptyName(workflowName, "Unnamed");
    if (nextName !== workflowName) {
      setWorkflowName(nextName);
    }

    setSaveLoading(true);
    try {
      const result = await onPersistRecipe({
        id: recipeId,
        name: nextName,
        payload: currentPayload,
      });
      setLastSavedAt(result.updatedAt);
      setSavedSignature(buildSignature(nextName, currentPayload));
    } catch (error) {
      console.error("Save recipe failed:", error);
      toastError("Save failed", "Could not save recipe.");
    } finally {
      setSaveLoading(false);
    }
  }, [currentPayload, onPersistRecipe, recipeId, saveLoading, workflowName]);

  useEffect(() => {
    if (!isDirty || saveLoading) {
      return;
    }
    const timeoutId = window.setTimeout(() => {
      void persistRecipe();
    }, 800);
    return () => window.clearTimeout(timeoutId);
  }, [isDirty, persistRecipe, saveLoading]);

  const copyRecipe = useCallback(async (): Promise<void> => {
    setCopied(false);
    if (payloadResult.errors.length > 0) {
      toastError("Copy failed", payloadErrorMessage);
      return;
    }
    try {
      const ok = await copyTextToClipboard(JSON.stringify(payloadResult.payload, null, 2));
      if (!ok) {
        throw new Error("Clipboard not available.");
      }
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
      toastSuccess("Payload copied");
    } catch (error) {
      console.error("Copy failed:", error);
      toastError("Copy failed", "Could not copy payload.");
    }
  }, [payloadErrorMessage, payloadResult.errors.length, payloadResult.payload]);

  const importRecipe = useCallback(
    (value: string): string | null => {
      const result = importRecipePayload(value);
      if (result.errors.length > 0 || !result.snapshot) {
        return result.errors[0] ?? "Invalid payload.";
      }
      loadRecipe(result.snapshot);
      toastSuccess("Recipe imported");
      return null;
    },
    [loadRecipe],
  );

  return {
    workflowName,
    setWorkflowName,
    saveLoading,
    saveTone,
    savedAtLabel,
    copied,
    importOpen,
    setImportOpen,
    currentSignature,
    persistRecipe,
    copyRecipe,
    importRecipe,
  };
}
