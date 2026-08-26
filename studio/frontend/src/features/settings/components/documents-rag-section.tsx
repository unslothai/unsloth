// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogMedia,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { useChatRuntimeStore } from "@/features/chat";
import { formatBytes, listCachedGguf, listCachedModels } from "@/features/hub";
import {
  DOWNLOAD_KIND,
  downloadManager,
  scopedVariant,
} from "@/features/hub/download-manager";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { FileDatabaseIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useCallback, useEffect, useState } from "react";
import {
  EmbeddingModelBlockedError,
  type EmbeddingModelResolution,
  EmbeddingModelVerificationError,
  resetEmbeddingModelSettings,
  resolveEmbeddingModel,
  updateEmbeddingModelSettings,
} from "../api/embedding-model";
import { useEmbeddingModelStore } from "../stores/embedding-model-store";
import { EmbeddingModelPicker } from "./embedding-model-picker";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

/** One slot per repo for the embedder's GGUF, so re-picking adopts the running
 * transfer and a full-repo Hub download keeps its own. */
const EMBEDDING_DOWNLOAD_SCOPE = "rag-embedding";

/**
 * Which model indexes uploaded documents. Rendered in both General and Data,
 * off one shared store, so a save on one is what the other reads.
 *
 * Picking applies immediately, as dictation does; a model that is not on disk
 * is then offered as a download instead of being fetched at the first index.
 */
export function DocumentsRagSection(): ReactElement {
  const t = useT();
  const hfToken = useChatRuntimeStore((s) => s.hfToken);
  const embeddingModel = useEmbeddingModelStore((s) => s.settings);
  const loadError = useEmbeddingModelStore((s) => s.loadError);
  const save = useEmbeddingModelStore((s) => s.save);
  const [saveError, setSaveError] = useState<string | null>(null);
  // The store carries the backend's reason; an unreadable failure has none.
  const loadFailure =
    loadError === null
      ? null
      : loadError || t("settings.general.rag.loadError");
  const embeddingModelError = saveError ?? loadFailure;
  // Set after a 409 (unverifiable model); "Save anyway" applies to that pick only.
  const [forceCandidate, setForceCandidate] = useState<string | null>(null);
  const [isSavingEmbeddingModel, setIsSavingEmbeddingModel] = useState(false);
  // Pending "download this?" confirmation, raised after a save lands on a
  // model this machine does not hold.
  const [pendingDownload, setPendingDownload] =
    useState<EmbeddingModelResolution | null>(null);
  // Repo ids with a complete cache, for the picker's on-device dot.
  const [cachedRepos, setCachedRepos] = useState<ReadonlySet<string>>(
    () => new Set(),
  );

  useEffect(() => {
    void useEmbeddingModelStore.getState().load();
  }, []);

  const refreshCachedRepos = useCallback(async () => {
    try {
      const [models, gguf] = await Promise.all([
        listCachedModels(hfToken || undefined),
        listCachedGguf(hfToken || undefined),
      ]);
      setCachedRepos(
        new Set(
          [...models, ...gguf]
            .filter((repo) => !repo.partial)
            .map((repo) => repo.repo_id),
        ),
      );
    } catch {
      // Advisory: without it rows just lose their dot.
    }
  }, [hfToken]);

  useEffect(() => {
    void refreshCachedRepos();
  }, [refreshCachedRepos]);

  /** Persist the pick, recording the GGUF repo /resolve named so the loader
   * opens what was downloaded rather than re-deriving a name. */
  const persist = async (
    model: string,
    ggufRepo: string | null,
    force: boolean,
  ): Promise<boolean> => {
    try {
      const stood = await save(() =>
        updateEmbeddingModelSettings(model, {
          hfToken: hfToken || undefined,
          ggufRepo,
          force,
        }),
      );
      // A later save owns the setting now, so this answer says nothing current.
      if (!stood) return false;
      setForceCandidate(null);
      toast.success(t("settings.general.rag.saved"), {
        description: t("settings.general.rag.reindexWarning"),
      });
      return true;
    } catch (error) {
      // A hard security block cannot be forced; keep "Save anyway" hidden.
      if (error instanceof EmbeddingModelBlockedError) {
        setForceCandidate(null);
      } else if (error instanceof EmbeddingModelVerificationError) {
        setForceCandidate(model);
      }
      setSaveError(
        error instanceof Error
          ? error.message
          : t("settings.general.rag.saveError"),
      );
      return false;
    }
  };

  /** Resolve first, so a model that needs fetching is offered as a download
   * rather than saved and quietly fetched at the first index. */
  const applyEmbeddingModel = async (model: string, force: boolean) => {
    const trimmed = model.trim();
    if (!trimmed) {
      setSaveError(t("settings.general.rag.emptyError"));
      return;
    }
    setIsSavingEmbeddingModel(true);
    setSaveError(null);
    try {
      if (force) {
        await persist(trimmed, null, true);
        return;
      }
      let resolution: EmbeddingModelResolution | null = null;
      try {
        resolution = await resolveEmbeddingModel(trimmed, {
          hfToken: hfToken || undefined,
        });
      } catch {
        // Offline or the probe failed: let the save decide, as it did before.
        await persist(trimmed, null, false);
        return;
      }
      if (resolution.error) {
        // Forceable: the user may know something the probe cannot see.
        setForceCandidate(trimmed);
        setSaveError(resolution.error);
        return;
      }
      if (resolution.cached || !resolution.downloadRepo) {
        await persist(trimmed, resolution.downloadRepo, false);
        return;
      }
      setPendingDownload(resolution);
    } finally {
      setIsSavingEmbeddingModel(false);
    }
  };

  const confirmDownload = async (resolution: EmbeddingModelResolution) => {
    const saved = await persist(
      resolution.embeddingModel,
      resolution.downloadRepo,
      false,
    );
    if (saved) await startDownload(resolution);
  };

  const startDownload = async (resolution: EmbeddingModelResolution) => {
    const repoId = resolution.downloadRepo;
    if (!repoId) return;
    // Scoped when the backend named a file: the companion repo carries every
    // quant, and the embedder opens one.
    const scoped = resolution.files !== null && resolution.files.length > 0;
    try {
      const outcome = await downloadManager.requestStart({
        kind: DOWNLOAD_KIND.MODEL,
        repoId,
        variant: scoped ? scopedVariant(EMBEDDING_DOWNLOAD_SCOPE) : null,
        scopeId: scoped ? EMBEDDING_DOWNLOAD_SCOPE : null,
        files: scoped ? (resolution.files ?? undefined) : undefined,
        expectedBytes: resolution.sizeBytes ?? 0,
      });
      if (outcome === "started") {
        toast.success(
          t("settings.general.rag.downloading", {
            model: resolution.embeddingModel,
          }),
          { description: t("settings.general.rag.downloadingDescription") },
        );
      }
      // "conflict" and "busy" already raise their own notice from the manager.
    } catch (error) {
      toast.error(t("settings.general.rag.downloadFailed"), {
        description: error instanceof Error ? error.message : undefined,
      });
    }
  };

  const resetEmbeddingModel = async () => {
    setIsSavingEmbeddingModel(true);
    setSaveError(null);
    setForceCandidate(null);
    try {
      await save(resetEmbeddingModelSettings);
    } catch (error) {
      setSaveError(
        error instanceof Error
          ? error.message
          : t("settings.general.rag.saveError"),
      );
    } finally {
      setIsSavingEmbeddingModel(false);
    }
  };

  const pendingSize = pendingDownload?.sizeBytes
    ? formatBytes(pendingDownload.sizeBytes)
    : null;

  return (
    <SettingsSection title={t("settings.general.rag.sectionTitle")}>
      <SettingsRow
        label={t("settings.general.rag.embeddingModel")}
        description={t("settings.general.rag.embeddingModelDescription", {
          defaultModel: embeddingModel?.defaultEmbeddingModel ?? "",
        })}
        className="max-[360px]:flex-col max-[360px]:items-stretch max-[360px]:gap-3"
      >
        <div className="flex flex-col items-end gap-1 max-[360px]:w-full">
          <EmbeddingModelPicker
            value={embeddingModel?.embeddingModel ?? ""}
            onSelect={(model) => void applyEmbeddingModel(model, false)}
            defaultModel={embeddingModel?.defaultEmbeddingModel}
            cachedModels={cachedRepos}
            accessToken={hfToken || undefined}
            disabled={!embeddingModel}
            busy={isSavingEmbeddingModel}
            className="w-[260px] max-[360px]:w-full"
          />
          {embeddingModelError ? (
            <span className="max-w-[300px] text-right text-xs text-destructive">
              {embeddingModelError}
            </span>
          ) : null}
          <div className="flex items-center gap-2">
            {forceCandidate ? (
              <Button
                variant="outline"
                size="sm"
                disabled={isSavingEmbeddingModel}
                onClick={() => void applyEmbeddingModel(forceCandidate, true)}
              >
                {t("settings.general.rag.saveAnyway")}
              </Button>
            ) : null}
            {embeddingModel?.isCustom ? (
              <Button
                variant="ghost"
                size="sm"
                disabled={isSavingEmbeddingModel}
                onClick={() => void resetEmbeddingModel()}
              >
                {t("settings.general.rag.resetAction")}
              </Button>
            ) : null}
          </div>
          <span className="max-w-[300px] text-right text-xs text-muted-foreground">
            {t("settings.general.rag.reindexWarning")}
          </span>
        </div>
      </SettingsRow>

      <AlertDialog
        open={pendingDownload !== null}
        onOpenChange={(open) => {
          if (!open) setPendingDownload(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogMedia className="size-12">
              <HugeiconsIcon
                icon={FileDatabaseIcon}
                strokeWidth={1.75}
                className="text-muted-foreground size-5"
              />
            </AlertDialogMedia>
            <AlertDialogTitle>
              {t("settings.general.rag.downloadConfirmTitle", {
                model: pendingDownload?.embeddingModel ?? "",
              })}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {pendingSize
                ? t("settings.general.rag.downloadConfirmBody", {
                    model: pendingDownload?.embeddingModel ?? "",
                    size: pendingSize,
                  })
                : t("settings.general.rag.downloadConfirmBodyUnsized", {
                    model: pendingDownload?.embeddingModel ?? "",
                  })}
            </AlertDialogDescription>
            {pendingDownload?.downloadRepo ? (
              <span className="text-muted-foreground font-mono text-ui-11">
                {t("settings.general.rag.downloadSource", {
                  repo: pendingDownload.downloadRepo,
                })}
              </span>
            ) : null}
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction
              onClick={(event) => {
                event.preventDefault();
                const request = pendingDownload;
                setPendingDownload(null);
                if (request) void confirmDownload(request);
              }}
            >
              {t("settings.general.rag.download")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </SettingsSection>
  );
}
