// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
  initialRecipeReady: boolean;
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

function stripApiKeys(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(stripApiKeys);
  }
  if (!value || typeof value !== "object") {
    return value;
  }
  const output: Record<string, unknown> = {};
  for (const [key, entry] of Object.entries(value)) {
    if (key === "api_key") {
      continue;
    }
    output[key] = stripApiKeys(entry);
  }
  if (
    output.provider_type === "stdio" &&
    output.env &&
    typeof output.env === "object" &&
    !Array.isArray(output.env)
  ) {
    output.env = Object.fromEntries(
      Object.keys(output.env as Record<string, unknown>).map((envKey) => [envKey, ""]),
    );
  }
  return output;
}

function inferHfRepoIdFromPath(pathValue: unknown): string {
  if (typeof pathValue !== "string") {
    return "";
  }
  const parts = pathValue
    .trim()
    .split("/")
    .filter(Boolean);
  if (parts.length >= 3 && parts[0] === "datasets") {
    return `${parts[1]}/${parts[2]}`;
  }
  if (parts.length >= 2) {
    return `${parts[0]}/${parts[1]}`;
  }
  return "";
}

function sanitizeSeedForShare(payload: unknown): unknown {
  if (!payload || typeof payload !== "object") {
    return payload;
  }
  const root = payload as Record<string, unknown>;
  const recipe =
    root.recipe && typeof root.recipe === "object"
      ? (root.recipe as Record<string, unknown>)
      : null;
  const ui =
    root.ui && typeof root.ui === "object"
      ? (root.ui as Record<string, unknown>)
      : null;

  const seedConfig =
    recipe?.seed_config && typeof recipe.seed_config === "object"
      ? (recipe.seed_config as Record<string, unknown>)
      : null;
  const source =
    seedConfig?.source && typeof seedConfig.source === "object"
      ? (seedConfig.source as Record<string, unknown>)
      : null;

  if (source && "token" in source) {
    delete source.token;
  }

  const uiSourceType =
    typeof ui?.seed_source_type === "string" ? ui.seed_source_type : null;
  const sourceType =
    typeof source?.seed_type === "string" ? source.seed_type : null;
  const shouldResetHfState =
    sourceType === "hf" || uiSourceType === "hf";
  const shouldResetLocalState =
    sourceType === "local" ||
    sourceType === "unstructured" ||
    uiSourceType === "local" ||
    uiSourceType === "unstructured";

  if (shouldResetHfState) {
    const repoId = inferHfRepoIdFromPath(source?.path);
    if (source && "path" in source) {
      source.path = repoId;
    }
    if (ui) {
      ui.seed_columns = [];
      ui.seed_drop_columns = [];
      ui.seed_preview_rows = [];
      ui.local_file_name = "";
      ui.unstructured_file_ids = [];
      ui.unstructured_file_names = [];
      ui.unstructured_file_sizes = [];
    }
  }

  if (shouldResetLocalState) {
    if (source && "path" in source) {
      source.path = "";
    }
    if (source && "paths" in source) {
      source.paths = [];
    }
    if (seedConfig) {
      seedConfig.resolved_paths = [];
    }
    if (ui) {
      ui.seed_columns = [];
      ui.seed_drop_columns = [];
      ui.seed_preview_rows = [];
      ui.local_file_name = "";
      ui.unstructured_file_ids = [];
      ui.unstructured_file_names = [];
      ui.unstructured_file_sizes = [];
    }
  }

  return root;
}

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
  const [initialRecipeReady, setInitialRecipeReady] = useState(false);
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

  useEffect(() => {
    setInitialRecipeReady(false);
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
    setInitialRecipeReady(true);
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
    try {
      const safePayload = sanitizeSeedForShare(stripApiKeys(payloadResult.payload));
      const ok = await copyTextToClipboard(JSON.stringify(safePayload, null, 2));
      if (!ok) {
        throw new Error("Clipboard not available.");
      }
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
      toastSuccess("👨‍🍳 Recipe copied");
    } catch (error) {
      console.error("Copy failed:", error);
      toastError("Copy failed", "Could not copy payload.");
    }
  }, [payloadResult.payload]);

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
    initialRecipeReady,
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
