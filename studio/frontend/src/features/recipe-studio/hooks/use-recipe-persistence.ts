// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { UserAssetApiError } from "@/features/user-assets";
import { toastError, toastSuccess } from "@/shared/toast";
import { normalizeNonEmptyName } from "@/utils";
import { useCallback, useEffect, useMemo, useState } from "react";
import { removeUnstructuredBlock } from "../api";
import {
  buildSignature,
  copyTextToClipboard,
  formatSavedLabel,
} from "../executions/execution-helpers";
import { useRecipeStudioStore } from "../stores/recipe-studio";
import { type RecipeSnapshot, importRecipePayload } from "../utils/import";
import type { RecipePayloadResult } from "../utils/payload/types";

type SaveTone = "success" | "error";

type PersistRecipeFn = (input: {
  id: string | null;
  name: string;
  payload: RecipePayloadResult["payload"];
  revision?: number;
}) => Promise<{
  id: string;
  updatedAt: number;
  revision: number;
  payload: RecipePayloadResult["payload"];
  removedCredentialPaths: string[];
}>;

type UseRecipePersistenceParams = {
  recipeId: string;
  initialRecipeName: string;
  initialPayload: RecipePayloadResult["payload"];
  initialSavedAt: number;
  initialRevision: number;
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
      Object.keys(output.env as Record<string, unknown>).map((envKey) => [
        envKey,
        "",
      ]),
    );
  }
  return output;
}

function inferHfRepoIdFromPath(pathValue: unknown): string {
  if (typeof pathValue !== "string") {
    return "";
  }
  const parts = pathValue.trim().split("/").filter(Boolean);
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
  const shouldResetHfState = sourceType === "hf" || uiSourceType === "hf";
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
      ui.unstructured_upload_uid = "";
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
      ui.unstructured_upload_uid = "";
      ui.unstructured_file_ids = [];
      ui.unstructured_file_names = [];
      ui.unstructured_file_sizes = [];
    }
  }

  return root;
}

// Delete queued upload directories once a save stops referencing them, so a
// reload before autosave can never leave the saved recipe pointing at
// already-deleted files. Skips any uid the just-saved payload still uses.
function drainQueuedUploadCleanups(
  savedPayload: RecipePayloadResult["payload"],
): void {
  const pending = useRecipeStudioStore.getState().pendingUploadCleanups;
  if (pending.length === 0) {
    return;
  }
  const ui =
    savedPayload && typeof savedPayload === "object"
      ? (savedPayload as { ui?: Record<string, unknown> }).ui
      : undefined;
  const savedUid =
    ui && typeof ui.unstructured_upload_uid === "string"
      ? ui.unstructured_upload_uid
      : "";
  const ready = pending.filter((uid) => uid !== savedUid);
  if (ready.length === 0) {
    return;
  }
  for (const uid of ready) {
    void removeUnstructuredBlock(uid)
      .then(() => {
        useRecipeStudioStore.setState((state) => ({
          pendingUploadCleanups: state.pendingUploadCleanups.filter(
            (pendingUid) => pendingUid !== uid,
          ),
        }));
      })
      .catch((error) => {
        console.warn("Failed to clean up uploaded documents:", error);
      });
  }
}

export function useRecipePersistence({
  recipeId,
  initialRecipeName,
  initialPayload,
  initialSavedAt,
  initialRevision,
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
  const [currentRevision, setCurrentRevision] = useState(initialRevision);
  const [conflict, setConflict] = useState<"changed" | "unavailable" | null>(
    null,
  );

  const normalizedWorkflowName = useMemo(
    () => normalizeNonEmptyName(workflowName, "Unnamed"),
    [workflowName],
  );
  const currentPayload = payloadResult.payload;
  const currentSignature = useMemo(
    () => buildSignature(normalizedWorkflowName, currentPayload),
    [currentPayload, normalizedWorkflowName],
  );
  const isDirty =
    savedSignature.length > 0 && currentSignature !== savedSignature;
  const saveTone: SaveTone =
    !isDirty && Boolean(lastSavedAt) ? "success" : "error";
  const savedAtLabel =
    conflict === "changed"
      ? "Changed elsewhere. Save again to overwrite."
      : conflict === "unavailable"
        ? "Recipe is no longer available."
        : formatSavedLabel(lastSavedAt);

  useEffect(() => {
    setInitialRecipeReady(false);
    const nextName = normalizeNonEmptyName(initialRecipeName, "Unnamed");
    resetRecipe();
    setWorkflowName(nextName);
    setLastSavedAt(initialSavedAt);
    setCurrentRevision(initialRevision);
    setConflict(null);
    setCopied(false);

    const parsed = importRecipePayload(JSON.stringify(initialPayload), {
      preserveUnstructuredUploads: true,
    });
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
    initialRevision,
    initialSavedAt,
    loadRecipe,
    recipeId,
    resetRecipe,
  ]);

  const persistRecipe = useCallback(async (): Promise<void> => {
    if (saveLoading) {
      return;
    }
    if (conflict === "unavailable") {
      toastError(
        "Recipe is unavailable",
        "It was deleted in another session and cannot be saved.",
      );
      return;
    }
    if (conflict === "changed") {
      setConflict(null);
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
        revision: currentRevision,
      });
      setLastSavedAt(result.updatedAt);
      setCurrentRevision(result.revision);
      setConflict(null);
      setSavedSignature(
        buildSignature(
          nextName,
          result.removedCredentialPaths.length > 0
            ? currentPayload
            : result.payload,
        ),
      );
      drainQueuedUploadCleanups(result.payload);
    } catch (error) {
      console.error("Save recipe failed:", error);
      if (error instanceof UserAssetApiError) {
        if (error.status === 409) {
          if (
            typeof error.detail.currentRevision === "number" &&
            Number.isInteger(error.detail.currentRevision) &&
            error.detail.currentRevision > 0
          ) {
            setCurrentRevision(error.detail.currentRevision);
          }
          setConflict("changed");
          toastError(
            "Recipe changed elsewhere",
            "Your edits are still here. Review them, then save again to overwrite the newer server version.",
          );
          return;
        }
        if (error.status === 404 || error.status === 410) {
          setConflict("unavailable");
          toastError(
            "Recipe is unavailable",
            "It was deleted in another session and cannot be saved.",
          );
          return;
        }
      }
      toastError("Save failed", "Could not save recipe.");
    } finally {
      setSaveLoading(false);
    }
  }, [
    currentPayload,
    currentRevision,
    conflict,
    onPersistRecipe,
    recipeId,
    saveLoading,
    workflowName,
  ]);

  useEffect(() => {
    if (!isDirty || saveLoading || conflict) {
      return;
    }
    const timeoutId = window.setTimeout(() => {
      void persistRecipe();
    }, 800);
    return () => window.clearTimeout(timeoutId);
  }, [
    conflict,
    isDirty,
    persistRecipe,
    saveLoading,
  ]);

  // Drain queued cleanups even when autosave is skipped: a net-zero edit (add
  // then remove an unstructured seed before the 800ms debounce) keeps isDirty
  // false, so the autosave effect never drains and the queued uid leaks its
  // upload dir. Not-dirty means currentPayload equals the saved recipe, and
  // drain skips the uid it still references, so only dirs no saved recipe
  // points at are deleted (keeps the save-first invariant).
  useEffect(() => {
    if (!initialRecipeReady || isDirty || saveLoading) {
      return;
    }
    drainQueuedUploadCleanups(currentPayload);
  }, [currentPayload, initialRecipeReady, isDirty, saveLoading]);

  const copyRecipe = useCallback(async (): Promise<void> => {
    setCopied(false);
    try {
      const safePayload = sanitizeSeedForShare(
        stripApiKeys(payloadResult.payload),
      );
      const ok = await copyTextToClipboard(
        JSON.stringify(safePayload, null, 2),
      );
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
