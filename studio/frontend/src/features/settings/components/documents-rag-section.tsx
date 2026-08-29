// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useChatRuntimeStore } from "@/features/chat";
import { formatBytes, listCachedGguf, listCachedModels } from "@/features/hub";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import {
  DOWNLOAD_KIND,
  downloadManager,
  jobKeyOf,
  scopedVariant,
  useDownloadManagerStore,
} from "@/features/hub/download-manager";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { type ReactElement, useCallback, useEffect, useState } from "react";
import {
  EmbeddingModelBlockedError,
  type EmbeddingModelResolution,
  EmbeddingModelVerificationError,
  resolveEmbeddingModel,
  unloadEmbeddingModel,
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
 * is marked pending by the server and offered as a download. Its loader remains
 * cache-only, so closing or cancelling cannot turn into a first-index transfer.
 */
// Residency is not a function of anything this component does, so it is polled
// rather than read only on a settings mutation.
const RESIDENCY_POLL_MS = 5000;

export function DocumentsRagSection(): ReactElement {
  const t = useT();
  const hfToken = useChatRuntimeStore((s) => s.hfToken);
  const embeddingModel = useEmbeddingModelStore((s) => s.settings);
  const loadError = useEmbeddingModelStore((s) => s.loadError);
  const beginSave = useEmbeddingModelStore((s) => s.beginSave);
  const isSaveCurrent = useEmbeddingModelStore((s) => s.isSaveCurrent);
  const save = useEmbeddingModelStore((s) => s.save);
  // Unloading releases residency and leaves the selection alone, so it must not
  // take a place in save order and retire an in-flight selection's reservation.
  const applyResidency = useEmbeddingModelStore((s) => s.applyResidency);
  const [saveError, setSaveError] = useState<string | null>(null);
  // The store carries the backend's reason; an unreadable failure has none.
  const loadFailure =
    loadError === null
      ? null
      : loadError || t("settings.general.rag.loadError");
  const embeddingModelError = saveError ?? loadFailure;
  // Set after a 409 (unverifiable model); "Save anyway" applies to that pick only.
  const [forceCandidate, setForceCandidate] = useState<string | null>(null);
  /** Bumped when a save leaves the model string unchanged, which the effect
   * below keys on and would otherwise not re-run for. */
  const [resolveNonce, setResolveNonce] = useState(0);
  const [isSavingEmbeddingModel, setIsSavingEmbeddingModel] = useState(false);
  // What /resolve last said about the saved model: drives the status line and
  // whether the action row offers Download.
  const [resolution, setResolution] = useState<EmbeddingModelResolution | null>(
    null,
  );
  // Repo ids with a complete cache, for the picker's on-device dot.
  const [cachedRepos, setCachedRepos] = useState<ReadonlySet<string>>(
    () => new Set(),
  );

  useEffect(() => {
    void useEmbeddingModelStore.getState().load();
  }, []);

  // Residency changes with no settings mutation: a running job reaching its
  // first encode makes a backend resident, and the store loads only on mount.
  // No lifecycle event to subscribe to, so re-read while visible; a hidden tab
  // does not poll and catches up when it returns.
  useEffect(() => {
    const refresh = () => {
      if (document.hidden) return;
      void useEmbeddingModelStore.getState().load();
    };
    const timer = window.setInterval(refresh, RESIDENCY_POLL_MS);
    document.addEventListener("visibilitychange", refresh);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", refresh);
    };
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

  // Resolve the model already saved, so the row reports its state on open
  // rather than only after a pick.
  const savedModel = embeddingModel?.embeddingModel;
  useEffect(() => {
    // Another mounted settings surface may have saved a different model. Its
    // model-scoped actions and errors say nothing about this new selection.
    setResolution(null);
    setForceCandidate(null);
    setSaveError(null);
    if (!savedModel) return;
    let live = true;
    void resolveEmbeddingModel(savedModel, { hfToken: hfToken || undefined })
      .then((next) => {
        if (live && next.embeddingModel === savedModel) setResolution(next);
      })
      .catch(() => {
        // Offline: the row simply says nothing about disk state.
      });
    return () => {
      live = false;
    };
  }, [savedModel, hfToken, resolveNonce]);

  /** Persist the pick, recording the GGUF repo /resolve named so the loader
   * opens what was downloaded rather than re-deriving a name. */
  const persist = async (
    model: string,
    plan: EmbeddingModelResolution | null,
    force: boolean,
    reservation: number,
  ): Promise<boolean> => {
    try {
      const stood = await save(
        () =>
          updateEmbeddingModelSettings(model, {
            hfToken: hfToken || undefined,
            ggufRepo:
              plan?.backend === "llama" ? (plan.downloadRepo ?? null) : null,
            backend: plan?.backend ?? null,
            force,
          }),
        reservation,
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
    // A force affordance belongs only to the request that produced its 409.
    setForceCandidate(null);
    const trimmed = model.trim();
    if (!trimmed) {
      setSaveError(t("settings.general.rag.emptyError"));
      return;
    }
    // Claim cross-surface ordering before the resolver await. Otherwise a
    // slower older selection can call save last and overwrite a newer pick.
    const reservation = beginSave();
    setIsSavingEmbeddingModel(true);
    setSaveError(null);
    setResolution(null);
    try {
      if (force) {
        // A force save can leave the model string unchanged, so the savedModel
        // effect does not re-run and nothing restores the plan this call cleared,
        // leaving no Download offered while the loader refuses to index.
        if (await persist(trimmed, null, true, reservation)) {
          setResolveNonce((n) => n + 1);
        }
        return;
      }
      let resolution: EmbeddingModelResolution;
      try {
        resolution = await resolveEmbeddingModel(trimmed, {
          hfToken: hfToken || undefined,
        });
      } catch {
        if (!isSaveCurrent(reservation)) return;
        // Offline or the probe failed: let the save decide, as it did before.
        await persist(trimmed, null, false, reservation);
        return;
      }
      if (!isSaveCurrent(reservation)) return;
      if (resolution.error) {
        // Forceable: the user may know something the probe cannot see.
        setResolution(resolution);
        setForceCandidate(trimmed);
        setSaveError(resolution.error);
        return;
      }
      // Retain the plan only after the server accepted the matching setting.
      // A rejected save must not expose a Download action for an unsaved repo.
      if (await persist(trimmed, resolution, false, reservation)) {
        setResolution(resolution);
      }
    } finally {
      setIsSavingEmbeddingModel(false);
    }
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
        inventoryKind: scoped ? "gguf" : undefined,
        expectedBytes: resolution.sizeBytes ?? 0,
      });
      if (outcome === "started") {
        toast.success(
          t("settings.general.rag.downloading", {
            model: resolution.embeddingModel,
          }),
          { description: t("settings.general.rag.downloadingDescription") },
        );
      } else if (outcome === "conflict") {
        // Not a failure: an earlier partial used a different transport and the
        // Hub's own card is where it resumes. "Couldn't start" would send the
        // user looking for a problem instead of for that row.
        toast.info(t("settings.general.rag.downloadConflict"));
      } else if (outcome === "busy") {
        // The repo is occupied by a sibling transfer, which the downloads panel
        // is already showing. Reselect once it lands.
        toast.info(t("settings.general.rag.downloadBusy"));
      } else {
        // requestStart turns refused starts into outcomes rather than throws, so
        // every remaining non-start still needs feedback from this caller.
        toast.error(t("settings.general.rag.downloadFailed"));
      }
    } catch (error) {
      toast.error(t("settings.general.rag.downloadFailed"), {
        description: error instanceof Error ? error.message : undefined,
      });
    }
  };

  // The live job for the repo this model needs, so the button reflects a
  // transfer started here or from anywhere else.
  const downloadJobKey =
    resolution?.downloadRepo && !resolution.cached
      ? jobKeyOf(
          DOWNLOAD_KIND.MODEL,
          resolution.downloadRepo,
          resolution.files?.length
            ? scopedVariant(EMBEDDING_DOWNLOAD_SCOPE)
            : null,
        )
      : null;
  const fullSnapshotJobKey =
    resolution?.downloadRepo && !resolution.cached
      ? jobKeyOf(DOWNLOAD_KIND.MODEL, resolution.downloadRepo, null)
      : null;
  const downloadState = useDownloadManagerStore((state) =>
    downloadJobKey ? (state.jobs[downloadJobKey]?.state ?? null) : null,
  );
  const fullSnapshotDownloadState = useDownloadManagerStore((state) =>
    fullSnapshotJobKey ? (state.jobs[fullSnapshotJobKey]?.state ?? null) : null,
  );
  const downloading =
    downloadState === "running" ||
    downloadState === "cancelling" ||
    fullSnapshotDownloadState === "running" ||
    fullSnapshotDownloadState === "cancelling";

  // The original resolve correctly said uncached. Once the shared manager
  // completes, ask again so the status/button and picker dots reflect disk.
  useEffect(() => {
    if (
      (downloadState !== "complete" &&
        fullSnapshotDownloadState !== "complete") ||
      !savedModel
    )
      return;
    let live = true;
    void Promise.all([
      resolveEmbeddingModel(savedModel, { hfToken: hfToken || undefined }),
      refreshCachedRepos(),
    ])
      .then(([next]) => {
        if (live && next.embeddingModel === savedModel) setResolution(next);
      })
      .catch(() => {
        // The completed transfer remains in the shared panel; a later open
        // retries the advisory resolve/cache inventory.
      });
    return () => {
      live = false;
    };
  }, [
    downloadState,
    fullSnapshotDownloadState,
    savedModel,
    hfToken,
    refreshCachedRepos,
  ]);
  const canDownload = Boolean(
    resolution && !resolution.cached && resolution.downloadRepo,
  );
  const onDevice = Boolean(resolution?.cached);
  const statusTone: "pending" | "ready" | "error" | null = !embeddingModel
    ? "pending"
    : embeddingModelError
      ? "error"
      : downloading || isSavingEmbeddingModel
        ? "pending"
        : onDevice
          ? "ready"
          : null;
  const statusText = !embeddingModel
    ? t("settings.general.rag.checking")
    : downloading
      ? t("settings.general.rag.downloadingStatus")
      : canDownload
        ? resolution?.sizeBytes
          ? t("settings.general.rag.notDownloadedSized", {
              size: formatBytes(resolution.sizeBytes),
            })
          : t("settings.general.rag.notDownloaded")
        : onDevice
          ? embeddingModel.loaded
            ? t("settings.general.rag.loaded")
            : t("settings.general.rag.onDevice")
          : "";

  const unload = async () => {
    setIsSavingEmbeddingModel(true);
    try {
      await applyResidency(unloadEmbeddingModel);
    } catch (error) {
      toast.error(t("settings.general.rag.unloadFailed"), {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setIsSavingEmbeddingModel(false);
    }
  };

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
          <div className="flex min-h-7 w-full items-center justify-between gap-3">
            <span className="flex min-w-0 items-center gap-2 text-xs text-muted-foreground">
              {statusTone ? (
                <span
                  className={cn(
                    "size-1.5 shrink-0 rounded-full",
                    statusTone === "pending"
                      ? "animate-pulse bg-current"
                      : statusTone === "ready"
                        ? "bg-emerald-500"
                        : "bg-destructive",
                  )}
                />
              ) : null}
              <span className="truncate">{statusText}</span>
            </span>
            {forceCandidate ? (
              <Button
                variant="outline"
                size="sm"
                className="h-7 shrink-0 px-2.5 text-xs"
                disabled={isSavingEmbeddingModel}
                onClick={() => void applyEmbeddingModel(forceCandidate, true)}
              >
                {t("settings.general.rag.saveAnyway")}
              </Button>
            ) : canDownload ? (
              <Button
                variant="outline"
                size="sm"
                className="h-7 shrink-0 px-2.5 text-xs"
                disabled={downloading || isSavingEmbeddingModel}
                onClick={() => resolution && void startDownload(resolution)}
              >
                {downloading ? <Spinner className="mr-1.5" /> : null}
                {t("settings.general.rag.download")}
              </Button>
            ) : null}
            {/* Outside the chain above: saving a new model does not release the
                old one, so while Download shows for an uncached pick the previous
                model can still be resident. backendLoaded asks about any. */}
            {embeddingModel?.backendLoaded ? (
              <Button
                variant="outline"
                size="sm"
                className="h-7 shrink-0 px-2.5 text-xs"
                disabled={isSavingEmbeddingModel}
                onClick={() => void unload()}
              >
                {t("settings.general.rag.unload")}
              </Button>
            ) : null}
          </div>
          <span className="max-w-[300px] text-right text-xs text-muted-foreground">
            {t("settings.general.rag.reindexWarning")}
          </span>
        </div>
      </SettingsRow>
    </SettingsSection>
  );
}
