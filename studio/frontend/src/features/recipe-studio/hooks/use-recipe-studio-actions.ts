import { useCallback, useEffect, useMemo, useState } from "react";
import { toastError, toastSuccess } from "@/shared/toast";
import { normalizeNonEmptyName } from "@/utils";
import { previewRecipe, validateRecipe } from "../api";
import { importRecipePayload, type RecipeSnapshot } from "../utils/import";
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
  persistRecipe: () => Promise<void>;
  openPreviewDialog: () => void;
  runPreview: () => Promise<void>;
  copyRecipe: () => Promise<void>;
  importRecipe: (value: string) => string | null;
};

function buildSignature(name: string, payload: RecipePayload): string {
  return JSON.stringify({ name, payload });
}

function formatSavedLabel(savedAt: number | null): string {
  if (!savedAt) {
    return "Not saved yet";
  }
  const time = new Date(savedAt).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
  return `Saved ${time}`;
}

function toErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error) {
    return error.message;
  }
  return fallback;
}

async function copyTextToClipboard(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // fallthrough to legacy path
  }

  try {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.top = "0";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(textarea);
    return ok;
  } catch {
    return false;
  }
}

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
}: UseRecipeStudioActionsParams): UseRecipeStudioActionsResult {
  const [workflowName, setWorkflowName] = useState("Unnamed");
  const [lastSavedAt, setLastSavedAt] = useState<number | null>(null);
  const [savedSignature, setSavedSignature] = useState<string>("");
  const [saveLoading, setSaveLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  const [previewDialogOpen, setPreviewDialogOpen] = useState(false);
  const [previewRows, setPreviewRows] = useState(5);
  const [previewErrors, setPreviewErrors] = useState<string[]>([]);
  const [previewLoading, setPreviewLoading] = useState(false);

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
    setPreviewErrors([]);
    setPreviewDialogOpen(false);

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

  const readPayload = useCallback((): RecipePayload | null => {
    if (payloadResult.errors.length === 0) {
      return payloadResult.payload;
    }
    return null;
  }, [payloadResult.errors.length, payloadResult.payload]);

  function openPreviewDialog(): void {
    setPreviewErrors([]);
    setPreviewDialogOpen(true);
  }

  const runPreview = useCallback(async (): Promise<void> => {
    setPreviewLoading(true);
    const payload = readPayload();
    if (!payload) {
      setPreviewErrors(payloadResult.errors);
      toastError("Invalid recipe payload", payloadErrorMessage);
      setPreviewLoading(false);
      return;
    }

    const previewPayload = {
      ...payload,
      run: {
        ...payload.run,
        rows: previewRows,
      },
    };

    try {
      const validation = await validateRecipe(previewPayload);
      if (!validation.valid) {
        const errors = validation.errors.map((item) => item.message);
        const fallback = validation.raw_detail ?? "Validation failed.";
        const nextErrors = errors.length > 0 ? errors : [fallback];
        setPreviewErrors(nextErrors);
        toastError("Validation failed", nextErrors[0]);
        return;
      }

      await previewRecipe(previewPayload);
      setPreviewDialogOpen(false);
      setPreviewErrors([]);
      toastSuccess(`Preview generated (${previewRows} rows).`);
    } catch (error) {
      console.error("Preview failed:", error);
      const message = toErrorMessage(error, "Preview request failed.");
      setPreviewErrors([message]);
      toastError("Preview failed", message);
    } finally {
      setPreviewLoading(false);
    }
  }, [payloadErrorMessage, payloadResult.errors, previewRows, readPayload]);

  const copyRecipe = useCallback(async (): Promise<void> => {
    setCopied(false);
    const payload = readPayload();
    if (!payload) {
      toastError("Copy failed", payloadErrorMessage);
      return;
    }
    try {
      const ok = await copyTextToClipboard(JSON.stringify(payload, null, 2));
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
  }, [payloadErrorMessage, readPayload]);

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
    previewDialogOpen,
    setPreviewDialogOpen,
    previewRows,
    setPreviewRows,
    previewErrors,
    previewLoading,
    persistRecipe,
    openPreviewDialog,
    runPreview,
    copyRecipe,
    importRecipe,
  };
}
