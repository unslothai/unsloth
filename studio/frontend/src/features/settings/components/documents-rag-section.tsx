// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useChatRuntimeStore } from "@/features/chat";
import { formatBytes, listCachedGguf, listCachedModels } from "@/features/hub";
import { Spinner } from "@/components/ui/spinner";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  DOWNLOAD_KIND,
  downloadManager,
  jobKeyOf,
  scopedVariant,
  useDownloadManagerStore,
} from "@/features/hub/download-manager";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { type ReactElement, useCallback, useEffect, useRef, useState } from "react";
import {
  EmbeddingModelBlockedError,
  type EmbeddingModelResolution,
  EmbeddingModelVerificationError,
  resolveEmbeddingModel,
  unloadEmbeddingModel,
  updateEmbeddingModelSettings,
} from "../api/embedding-model";
import { useEmbeddingModelStore } from "../stores/embedding-model-store";
import { useEmbeddingPinsStore } from "../stores/embedding-pins-store";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";
import { EmbeddingModelPicker } from "./embedding-model-picker";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

/** One slot per repo for the embedder's GGUF, so re-picking adopts the running transfer and a full-repo Hub download keeps its own. */
const EMBEDDING_DOWNLOAD_SCOPE = "rag-embedding";

/**
 * Which model indexes uploaded documents, rendered in both General and Data off one shared store.
 * A model not on disk is marked pending and offered as a download, and its loader stays cache-only,
 * so closing or cancelling cannot turn into a first-index transfer.
 */
const RESIDENCY_POLL_MS = 5000;

export function DocumentsRagSection(): ReactElement {
  const t = useT();
  const hfToken = useChatRuntimeStore((s) => s.hfToken);
  const embeddingModel = useEmbeddingModelStore((s) => s.settings);
  const loadError = useEmbeddingModelStore((s) => s.loadError);
  const beginSave = useEmbeddingModelStore((s) => s.beginSave);
  const isSaveCurrent = useEmbeddingModelStore((s) => s.isSaveCurrent);
  const save = useEmbeddingModelStore((s) => s.save);
  // Unloading leaves the selection alone, so it must not take a place in save order and retire an in-flight selection's reservation.
  const applyResidency = useEmbeddingModelStore((s) => s.applyResidency);
  const [saveError, setSaveError] = useState<string | null>(null);
  const loadFailure =
    loadError === null
      ? null
      : loadError || t("settings.general.rag.loadError");
  const embeddingModelError = saveError ?? loadFailure;
  const [forceCandidate, setForceCandidate] = useState<string | null>(null);
  /** Bumped when a save leaves the model string unchanged, which the effect below keys on. */
  const [resolveNonce, setResolveNonce] = useState(0);
  const [isSavingEmbeddingModel, setIsSavingEmbeddingModel] = useState(false);
  const [resolution, setResolution] = useState<EmbeddingModelResolution | null>(
    null,
  );
  const [cachedRepos, setCachedRepos] = useState<ReadonlySet<string>>(
    () => new Set(),
  );
  const pinnedModels = useEmbeddingPinsStore((s) => s.pinned);
  const togglePin = useEmbeddingPinsStore((s) => s.togglePin);
  const sectionRef = useRef<HTMLElement | null>(null);
  const scrollTarget = useSettingsDialogStore((s) => s.scrollTarget);
  const consumeScrollTarget = useSettingsDialogStore((s) => s.consumeScrollTarget);

  // "Change model" in the RAG menu lands here.
  useEffect(() => {
    if (scrollTarget !== "general-rag-embedding") return;
    const frame = window.requestAnimationFrame(() => {
      sectionRef.current?.scrollIntoView({ block: "start", behavior: "smooth" });
      consumeScrollTarget("general-rag-embedding");
    });
    return () => window.cancelAnimationFrame(frame);
  }, [consumeScrollTarget, scrollTarget]);

  useEffect(() => {
    void useEmbeddingModelStore.getState().load();
  }, []);

  // Residency changes with no settings mutation and there is no lifecycle event to subscribe to, so re-read while visible; a hidden tab catches up when it returns.
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
    }
  }, [hfToken]);

  useEffect(() => {
    void refreshCachedRepos();
  }, [refreshCachedRepos]);

  const savedModel = embeddingModel?.embeddingModel;
  useEffect(() => {
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
      });
    return () => {
      live = false;
    };
  }, [savedModel, hfToken, resolveNonce]);

  /** Persist the pick, recording the GGUF repo /resolve named so the loader opens what was downloaded rather than re-deriving a name. */
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

  /** Resolve first, so a model that needs fetching is offered as a download rather than saved and quietly fetched at the first index. */
  const applyEmbeddingModel = async (model: string, force: boolean) => {
    setForceCandidate(null);
    const trimmed = model.trim();
    if (!trimmed) {
      setSaveError(t("settings.general.rag.emptyError"));
      return;
    }
    // Claim cross-surface ordering before the resolver await, else a slower older selection saves last and overwrites a newer pick.
    const reservation = beginSave();
    setIsSavingEmbeddingModel(true);
    setSaveError(null);
    setResolution(null);
    try {
      if (force) {
        // A force save can leave the model string unchanged, so the savedModel effect does not re-run and nothing restores the plan this call cleared.
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
        await persist(trimmed, null, false, reservation);
        return;
      }
      if (!isSaveCurrent(reservation)) return;
      if (resolution.error) {
        setResolution(resolution);
        setForceCandidate(trimmed);
        setSaveError(resolution.error);
        return;
      }
      // Retain the plan only after the server accepted the matching setting: a rejected save must not expose Download for an unsaved repo.
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
    // Scoped when the backend named a file: the companion repo carries every quant, and the embedder opens one.
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
        // Not a failure: an earlier partial used a different transport and the Hub's own card is where it resumes.
        toast.info(t("settings.general.rag.downloadConflict"));
      } else if (outcome === "busy") {
        toast.info(t("settings.general.rag.downloadBusy"));
      } else {
        // requestStart turns refused starts into outcomes rather than throws, so every remaining non-start needs feedback here.
        toast.error(t("settings.general.rag.downloadFailed"));
      }
    } catch (error) {
      toast.error(t("settings.general.rag.downloadFailed"), {
        description: error instanceof Error ? error.message : undefined,
      });
    }
  };

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

  // On disk but not in memory: said under the picker, since Eject only shows once it loads.
  const notLoaded = onDevice && !embeddingModel?.loaded && !downloading;
  const statusHint =
    onDevice && !downloading && !canDownload
      ? t(
          embeddingModel?.loaded
            ? "settings.general.rag.loadedHint"
            : "settings.general.rag.onDeviceHint",
        )
      : "";

  const unload = async () => {
    setIsSavingEmbeddingModel(true);
    try {
      await applyResidency(unloadEmbeddingModel);
      toast.success(t("settings.general.rag.ejected"));
    } catch (error) {
      toast.error(t("settings.general.rag.unloadFailed"), {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setIsSavingEmbeddingModel(false);
    }
  };

  return (
    <SettingsSection ref={sectionRef} title={t("settings.general.rag.sectionTitle")}>
      <SettingsRow
        label={t("settings.general.rag.embeddingModel")}
        // Long explanation behind the info icon; the row shows the model's status.
        hint={`${t("settings.general.rag.embeddingModelDescription", {
          defaultModel: embeddingModel?.defaultEmbeddingModel ?? "",
        })} ${t("settings.general.rag.reindexWarning")}`}
        description={
          <span className="flex flex-col gap-1">
            <span>{t("settings.general.rag.embeddingModelShort")}</span>
            {statusText ? (
              <span className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1">
                <span className="flex min-w-0 items-center gap-1.5">
                  {statusTone ? (
                    <span
                      className={cn(
                        "size-[calc(6px*var(--ui-space-scale,1))] shrink-0 rounded-full",
                        statusTone === "pending"
                          ? "animate-pulse bg-muted-foreground"
                          : statusTone === "ready"
                            ? "bg-status-success"
                            : "bg-destructive",
                      )}
                    />
                  ) : null}
                  <span className="truncate">{statusText}</span>
                  {statusHint ? <StatusHint text={statusHint} /> : null}
                </span>
              </span>
            ) : null}
          </span>
        }
        className="max-[360px]:flex-col max-[360px]:items-stretch max-[360px]:gap-3"
        below={
          embeddingModelError ? (
            <span className="max-w-[calc(300px*var(--ui-space-scale,1))] text-right text-xs text-destructive">
              {embeddingModelError}
            </span>
          ) : undefined
        }
      >
        <div className="flex items-start gap-2 max-[360px]:w-full">
          {forceCandidate ? (
            <Button
              variant="outline"
              size="sm"
              className="shrink-0"
              disabled={isSavingEmbeddingModel}
              onClick={() => void applyEmbeddingModel(forceCandidate, true)}
            >
              {t("settings.general.rag.saveAnyway")}
            </Button>
          ) : canDownload ? (
            <Button
              variant="outline"
              size="sm"
              className="shrink-0"
              disabled={downloading || isSavingEmbeddingModel}
              onClick={() => resolution && void startDownload(resolution)}
            >
              {downloading ? <Spinner /> : null}
              {t("settings.general.rag.download")}
            </Button>
          ) : null}
          {/* Status sits in the picker's column, so it lines up under the model name. */}
          <div className="flex flex-col gap-1 max-[360px]:w-full">
            <EmbeddingModelPicker
              value={embeddingModel?.embeddingModel ?? ""}
              onSelect={(model) => void applyEmbeddingModel(model, false)}
              defaultModel={embeddingModel?.defaultEmbeddingModel}
              cachedModels={cachedRepos}
              pinnedModels={pinnedModels}
              onTogglePin={togglePin}
              accessToken={hfToken || undefined}
              disabled={!embeddingModel}
              busy={isSavingEmbeddingModel}
              loaded={embeddingModel?.loaded}
              // Any resident embedder, not just this one: switching does not release the old one.
              onEject={embeddingModel?.backendLoaded ? () => void unload() : undefined}
              className="w-[calc(260px*var(--ui-space-scale,1))] max-[360px]:w-full"
            />
            {notLoaded ? (
              <span className="flex items-center gap-1.5 px-3.5 text-xs text-muted-foreground">
                <span className="size-[calc(6px*var(--ui-space-scale,1))] shrink-0 rounded-full bg-muted-foreground" />
                {t("settings.general.rag.notLoaded")}
                <StatusHint text={t("settings.general.rag.notLoadedHint")} />
              </span>
            ) : null}
          </div>
        </div>
      </SettingsRow>
    </SettingsSection>
  );
}

/** Small info icon after a status, explaining what it means. */
function StatusHint({ text }: { text: string }): ReactElement {
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>
        <button
          type="button"
          aria-label={text}
          className="flex shrink-0 items-center rounded text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        >
          <HugeiconsIcon icon={InformationCircleIcon} className="size-[calc(10px*var(--ui-space-scale,1))]" />
        </button>
      </TooltipTrigger>
      <TooltipContent className="max-w-[calc(260px*var(--ui-space-scale,1))] text-ui-11 leading-snug">
        {text}
      </TooltipContent>
    </Tooltip>
  );
}
