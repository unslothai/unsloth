// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  BROWSER_ACCOUNT_KEY,
  accountTransitionPending,
  sameBrowserAccount,
} from "@/lib/account-transition";
import { useTimeFormat } from "@/lib/time-format";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toastError, toastSuccess } from "@/shared/toast";
import { normalizeNonEmptyName } from "@/utils";
import { removeUnstructuredBlock } from "../api";
import {
  buildSignature,
  formatSavedLabel,
} from "../executions/execution-helpers";
import { useRecipeStudioStore } from "../stores/recipe-studio";
import { importRecipePayload, type RecipeSnapshot } from "../utils/import";
import type { RecipePayloadResult } from "../utils/payload/types";

type SaveTone = "success" | "error";

type PersistRecipeFn = (input: {
  id: string | null;
  name: string;
  payload: RecipePayloadResult["payload"];
  keepalive?: boolean;
  // Exit saves get no later chance, so they may also try a write whose answer was lost.
  recover?: boolean;
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
      delete ui.seed_total_rows;
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
      delete ui.seed_total_rows;
      ui.local_file_name = "";
      ui.unstructured_upload_uid = "";
      ui.unstructured_file_ids = [];
      ui.unstructured_file_names = [];
      ui.unstructured_file_sizes = [];
    }
  }

  return root;
}

// Delete queued upload dirs only once a save no longer references them, so a reload before
// autosave never leaves the saved recipe pointing at deleted files.
function drainQueuedUploadCleanups(
  savedPayload: RecipePayloadResult["payload"],
  // A save that settles after its editor closed may only drain what was queued before it closed.
  pending: readonly string[] = useRecipeStudioStore.getState().pendingUploadCleanups,
): void {
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
  // Do not retry the exact content that just failed (e.g. a 409 from another window).
  const [failedSignature, setFailedSignature] = useState<string | null>(null);
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
  const isDirty =
    savedSignature.length > 0 && currentSignature !== savedSignature;
  const saveTone: SaveTone =
    !isDirty && Boolean(lastSavedAt) ? "success" : "error";
  const savedAtLabel = formatSavedLabel(lastSavedAt);
  useTimeFormat();

  useEffect(() => {
    setInitialRecipeReady(false);
    const nextName = normalizeNonEmptyName(initialRecipeName, "Unnamed");
    resetRecipe();
    setWorkflowName(nextName);
    setLastSavedAt(initialSavedAt);
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
    initialSavedAt,
    loadRecipe,
    recipeId,
    resetRecipe,
  ]);

  const latest = useRef({
    ready: false,
    recipeId,
    onPersistRecipe,
    name: normalizedWorkflowName,
    payload: currentPayload,
    signature: currentSignature,
  });
  useEffect(() => {
    latest.current = {
      ready: initialRecipeReady,
      recipeId,
      onPersistRecipe,
      name: normalizedWorkflowName,
      payload: currentPayload,
      signature: currentSignature,
    };
  });
  const savedRef = useRef({
    saved: "",
    failed: null as string | null,
    seq: 0,
    applied: 0,
    outstanding: 0,
    uncertain: false,
  });
  const inFlight = useRef<{ signature: string; done: Promise<void> } | null>(
    null,
  );
  // Null while open; once closed, the upload queue as it stood then.
  const closedQueue = useRef<readonly string[] | null>(null);

  useEffect(() => {
    savedRef.current.saved = savedSignature;
  }, [savedSignature]);

  const runSave = useCallback(
    (
      name: string,
      payload: RecipePayloadResult["payload"],
      signature: string,
      keepalive: boolean,
      recover = keepalive,
    ): Promise<void> => {
      const { recipeId: id, onPersistRecipe: persist } = latest.current;
      const seq = ++savedRef.current.seq;
      savedRef.current.outstanding += 1;
      setSaveLoading(true);
      const done: Promise<void> = persist({ id, name, payload, keepalive, recover })
        .then(
          (result) => {
            if (seq < savedRef.current.applied) {
              return;
            }
            savedRef.current.applied = seq;
            savedRef.current.uncertain = false;
            setLastSavedAt(result.updatedAt);
            savedRef.current.failed = null;
            setFailedSignature(null);
            savedRef.current.saved = signature;
            setSavedSignature(signature);
            // A newer save still unsettled may reference uploads this older payload dropped.
            if (seq === savedRef.current.seq) {
              drainQueuedUploadCleanups(payload, closedQueue.current ?? undefined);
            }
          },
          (error: unknown) => {
            // A newer save already landed, so this content is superseded, not rejected.
            if (seq < savedRef.current.applied) {
              return;
            }
            console.error("Save recipe failed:", error);
            // An HTTP refusal carries a status; anything else may have landed unseen.
            if (typeof (error as { status?: unknown } | null)?.status !== "number") {
              savedRef.current.uncertain = true;
            }
            savedRef.current.failed = signature;
            setFailedSignature(signature);
            toastError(
              "Save failed",
              error instanceof Error ? error.message : "Could not save recipe.",
            );
          },
        )
        .finally(() => {
          if (inFlight.current?.done === done) inFlight.current = null;
          // Idle (and so free to clean uploads) only once every overlapping save has settled.
          if (--savedRef.current.outstanding === 0) setSaveLoading(false);
        });
      inFlight.current = { signature, done };
      return done;
    },
    [],
  );

  const persistRecipe = useCallback(async (): Promise<void> => {
    if (inFlight.current) {
      return;
    }
    const nextName = normalizeNonEmptyName(workflowName, "Unnamed");
    if (nextName !== workflowName) {
      setWorkflowName(nextName);
    }
    await runSave(
      nextName,
      currentPayload,
      buildSignature(nextName, currentPayload),
      false,
    );
  }, [currentPayload, runSave, workflowName]);

  useEffect(() => {
    if (!isDirty || saveLoading || failedSignature === currentSignature) {
      return;
    }
    const timeoutId = window.setTimeout(() => {
      void persistRecipe();
    }, 800);
    return () => window.clearTimeout(timeoutId);
  }, [currentSignature, failedSignature, isDirty, persistRecipe, saveLoading]);

  useEffect(() => {
    // Another tab may switch accounts first; its tokens must never carry this account's recipe.
    const account = window.localStorage.getItem(BROWSER_ACCOUNT_KEY);
    closedQueue.current = null;
    const flush = (keepalive: boolean, unloading = keepalive): void => {
      const { ready, name, payload, signature } = latest.current;
      const { saved, failed, uncertain } = savedRef.current;
      const running = inFlight.current;
      // Compare with what the save in flight will leave stored: undoing it still needs a save.
      const settled = running ? running.signature : saved;
      // After a lost answer the server may hold other content, so even this one must be sent.
      const current = !uncertain && (signature === settled || signature === failed);
      if (!ready || !saved || current || running?.signature === signature) {
        return;
      }
      if (accountTransitionPending() || !sameBrowserAccount(account)) {
        return;
      }
      // Only an unloading page sends on top of the save in flight; a hidden one may come back.
      if (running && !unloading) {
        void running.done.then(() => flush(keepalive, false));
        return;
      }
      void runSave(name, payload, signature, keepalive, true);
    };
    // A persisted pagehide enters the back/forward cache and may come back, like a hidden tab.
    const onPageHide = (event: PageTransitionEvent) => flush(true, !event.persisted);
    const onVisibilityChange = () => {
      if (document.visibilityState === "hidden") flush(true, false);
    };
    window.addEventListener("pagehide", onPageHide);
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      window.removeEventListener("pagehide", onPageHide);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      closedQueue.current = [...useRecipeStudioStore.getState().pendingUploadCleanups];
      flush(false);
    };
  }, [runSave]);

  // Drain even when not dirty: a net-zero edit inside the debounce would otherwise leak the upload
  // dir. Drain skips uids the saved recipe still references.
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
      const ok = await copyToClipboard(JSON.stringify(safePayload, null, 2));
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
