// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useRef, useState } from "react";
import {
  fetchSttStatus,
  loadSttModel,
  startSttDownload,
  sttEngineStatusFor,
  SttModelNotDownloadedError,
  unloadSttModel,
} from "@/features/chat/adapters/studio-model-dictation-adapter";
import { getHfToken, hfApiToken } from "@/features/hub/stores/hf-token-store";
import {
  isTrackingSttDownload,
  trackSttDownload,
} from "@/features/settings/lib/stt-download-mirror";
import { usePersistedChoice } from "@/hooks/use-persisted-choice";
import { toast } from "@/lib/toast";
import type { AudioCppRuntimeStatus } from "../audio-cpp-catalog";
import {
  reconcileSttSelection,
  resolveSttResidency,
  type SttDownloadedArtifact,
  sttDownloadedArtifacts,
  sttSelectionReady,
} from "../audio-page-policy";
import { audioModelLabel } from "../audio-workspace-utils";
import {
  type AudioSttEngine,
  sttEngineForRepoId,
  sttRepoIdForSidecarKey,
  sttSidecarKeyFor,
} from "../catalog";
import { useAudioWorkspaceStore } from "../stores/audio-workspace-store";
import type { AudioHostState } from "./audio-host-state";

/** Transcribe's model: the STT sidecar pick, its residency and the load that prepares it. */
export function useSttSidecar({
  setBusy,
  activeRef,
}: Pick<AudioHostState, "setBusy" | "activeRef">) {
  const [lastSttRepo, setLastSttRepoChoice] = usePersistedChoice("unsloth:audio:last-stt-model", "");
  // The quant a package pick (Moonshine tiny or small) asked for, so a restart reloads that one, not the default.
  const [lastSttVariant, setLastSttVariant] = usePersistedChoice("unsloth:audio:last-stt-variant", "");
  const [selectedSttRepo, setSelectedSttRepo] = useState<string | null>(null);
  const [sttLoadedModel, setSttLoadedModel] = useState<string | null>(null);
  const [sttLoadedEngine, setSttLoadedEngine] = useState<
    AudioSttEngine | null
  >(null);
  const [downloadedSttArtifacts, setDownloadedSttArtifacts] = useState<
    SttDownloadedArtifact[]
  >([]);
  const selectedSttRepoRef = useRef<string | null>(selectedSttRepo);
  selectedSttRepoRef.current = selectedSttRepo;
  const sttStatusRefreshGeneration = useRef(0);
  // What the installed GGUF audio runtime can run, from the STT status poll. Null until it answers.
  const audioCppRuntime = useRef<AudioCppRuntimeStatus | null>(null);
  const sttLoadGeneration = useRef(0);
  /** The GGUF quant each picked STT repo asked for, by lowercased repo id. */
  const sttGgufVariants = useRef(
    new Map<string, string>(
      lastSttRepo && lastSttVariant ? [[lastSttRepo.toLowerCase(), lastSttVariant]] : [],
    ),
  );
  const setLastSttRepo = useCallback(
    (repo: string) => {
      setLastSttRepoChoice(repo);
      setLastSttVariant(sttGgufVariants.current.get(repo.toLowerCase()) ?? "");
    },
    [setLastSttRepoChoice, setLastSttVariant],
  );
  const sttLoadingGeneration = useRef<number | null>(null);
  // Residency is not ownership: the activation resync adopts whatever a sidecar already holds, including a model
  // chat dictation loaded. The identity, not a boolean, since another surface can replace the sidecar's model
  // while Audio is inactive and a bare flag would claim that too. Keyed on the model alone, since a "gguf" pick
  // without whisper-server comes back resident under the Transformers fallback.
  const sttLoadedByThisPage = useRef<string | null>(null);
  const sttLoadAbort = useRef<AbortController | null>(null);
  const deferredSttLoad = useRef<{
    repoId: string;
    sidecarKey: string;
    engine: AudioSttEngine;
  } | null>(null);

  const refreshSttStatus = useCallback(async () => {
    const generation = ++sttStatusRefreshGeneration.current;
    try {
      const selectedRepo = selectedSttRepoRef.current;
      const selectedKey = selectedRepo ? sttSidecarKeyFor(selectedRepo) : null;
      const selectedEngine = selectedRepo
        ? sttEngineForRepoId(selectedRepo)
        : null;
      const stt = await fetchSttStatus(
        undefined,
        selectedEngine === "transformers"
          ? (selectedKey ?? undefined)
          : undefined,
      );
      if (generation !== sttStatusRefreshGeneration.current) return;
      audioCppRuntime.current = stt.audio_cpp_runtime ?? null;
      const nextDownloadedArtifacts = sttDownloadedArtifacts(
        stt,
        sttRepoIdForSidecarKey,
      );
      setDownloadedSttArtifacts((current) =>
        current.length === nextDownloadedArtifacts.length &&
        current.every(
          (artifact, index) =>
            artifact.repoId === nextDownloadedArtifacts[index].repoId &&
            artifact.sidecarKey === nextDownloadedArtifacts[index].sidecarKey &&
            artifact.engine === nextDownloadedArtifacts[index].engine,
        )
          ? current
          : nextDownloadedArtifacts,
      );
      const selectedBlock = selectedKey
        ? (sttEngineStatusFor(stt, selectedKey, selectedEngine ?? undefined) ??
          (selectedEngine === "transformers" ? stt : undefined))
        : undefined;
      const preservePending = Boolean(
        selectedBlock?.loading ||
          sttLoadingGeneration.current !== null ||
          deferredSttLoad.current !== null,
      );
      const residency = resolveSttResidency(
        stt,
        selectedEngine,
        preservePending,
      );
      const loadedModel = residency?.model ?? null;
      setSttLoadedModel(loadedModel);
      setSttLoadedEngine(residency?.engine ?? null);
      const reconciled = reconcileSttSelection({
        selectedRepo,
        loadedModel,
        loadedEngine: residency?.engine,
        preservePending,
        sidecarKeyFor: sttSidecarKeyFor,
        repoIdForSidecarKey: sttRepoIdForSidecarKey,
        engineForRepo: sttEngineForRepoId,
      });
      selectedSttRepoRef.current = reconciled;
      setSelectedSttRepo(reconciled);
    } catch {
      if (generation !== sttStatusRefreshGeneration.current) return;
      setSttLoadedModel(null);
      setSttLoadedEngine(null);
    }
  }, []);

  const sttSelected = selectedSttRepo !== null;
  const sttReady = sttSelectionReady(
    selectedSttRepo,
    sttLoadedModel,
    sttSidecarKeyFor,
    selectedSttRepo ? sttEngineForRepoId(selectedSttRepo) : null,
    sttLoadedEngine,
  );

  /** Forget the Transcribe pick, releasing its sidecar when this page owns it. */
  const releaseTranscribeSelection = useCallback(async () => {
    const selected = selectedSttRepoRef.current;
    const claim = sttLoadedByThisPage.current;
    const owned =
      sttReady &&
      selected !== null &&
      claim !== null &&
      claim === sttLoadedModel;
    const forget = () => {
      deferredSttLoad.current = null;
      selectedSttRepoRef.current = null;
      sttLoadGeneration.current += 1;
      sttLoadedByThisPage.current = null;
      setSelectedSttRepo(null);
    };
    if (!owned) {
      forget();
      return;
    }
    // Forget only once the sidecar is actually released: clearing first left a failed unload with the model in
    // VRAM and no Eject to retry with. Scoped to the model this page claimed, since another surface can switch the
    // same engine before the request lands.
    await unloadSttModel(sttEngineForRepoId(selected), claim);
    forget();
    await refreshSttStatus();
  }, [refreshSttStatus, sttReady, sttLoadedModel]);

  const ensureSttLoaded = useCallback(
    async (
      repoId: string,
      sidecarKey: string,
      engine: AudioSttEngine,
    ) => {
      const generation = ++sttLoadGeneration.current;
      const controller = new AbortController();
      sttLoadingGeneration.current = generation;
      sttLoadAbort.current = controller;
      const isCurrent = () =>
        generation === sttLoadGeneration.current &&
        activeRef.current &&
        selectedSttRepoRef.current === repoId;

      setBusy("loading");
      // Ownership is claimed only once the requested model is actually resident: claiming it up front meant a
      // cancelled download left the flag set while the backend kept the previous model, so leaving Transcribe
      // unloaded another surface's model.
      const toastId = toast.loading(`Preparing ${audioModelLabel(sidecarKey)}…`);
      // The quant picked for this repo (Moonshine tiny or small), sent to the audio runtime's sidecar.
      const ggufVariant =
        engine === "audiocpp"
          ? (sttGgufVariants.current.get(repoId.toLowerCase()) ?? null)
          : null;
      try {
        try {
          await loadSttModel(
            sidecarKey,
            engine,
            controller.signal,
            undefined,
            ggufVariant,
          );
          sttLoadedByThisPage.current = sidecarKey;
        } catch (error) {
          if (!(error instanceof SttModelNotDownloadedError)) throw error;
          if (!isCurrent()) return;
          await startSttDownload(
            sidecarKey,
            hfApiToken(getHfToken()),
            engine,
            ggufVariant,
          );
          // STT owns its specialized transfer, but the existing mirror gives it the same global Downloads
          // row, progress and Cancel as every other model download. Do not reset an adopted row.
          if (!isTrackingSttDownload(sidecarKey, engine)) {
            trackSttDownload(sidecarKey, {
              // Audio owns the final load through its active/selection generation guards, so the
              // Voice-settings mirror must not warm a stale sidecar behind those guards.
              warmSelectedVoiceModelOnComplete: false,
              engine,
              repoId,
            });
          }
          // The shared Downloads panel now owns transfer progress; keep this toast for the short model-load
          // phase after the bytes land.
          toast.dismiss(toastId);
          // A completed download may outlive this page or selection, so re-check ownership around every
          // await or an old pick could replace a newer sidecar.
          for (;;) {
            await new Promise((resolve) => setTimeout(resolve, 1000));
            if (!isCurrent()) return;
            const stt = await fetchSttStatus(
              undefined,
              engine === "transformers" ? sidecarKey : undefined,
            );
            if (!isCurrent()) return;
            const block = sttEngineStatusFor(stt, sidecarKey, engine);
            const download = block?.download;
            // Cancel comes from the shared Downloads row. It is terminal for this preparation attempt, not
            // permission to load a partial checkpoint.
            if (download?.cancelled) return;
            if (download?.error) throw new Error(download.error);
            if (!download?.downloading) break;
          }
          if (!isCurrent()) return;
          toast.loading(`Loading ${audioModelLabel(sidecarKey)}…`, {
            id: toastId,
          });
          await loadSttModel(
            sidecarKey,
            engine,
            controller.signal,
            undefined,
            ggufVariant,
          );
          sttLoadedByThisPage.current = sidecarKey;
        }
        if (isCurrent()) {
          setLastSttRepo(repoId);
          useAudioWorkspaceStore.getState().rememberModel("transcribe", repoId);
          toast.success("Transcription model ready", { id: toastId });
          return true;
        }
        return false;
      } catch (error) {
        if (isCurrent()) {
          toast.error(
            error instanceof Error
              ? error.message
              : "Transcription model failed.",
            { id: toastId },
          );
        }
      } finally {
        if (!isCurrent()) toast.dismiss(toastId);
        if (sttLoadingGeneration.current === generation) {
          sttLoadingGeneration.current = null;
          await refreshSttStatus();
          if (
            generation === sttLoadGeneration.current &&
            sttLoadingGeneration.current === null
          ) {
            setBusy(null);
          }
        } else {
          toast.dismiss(toastId);
        }
        if (sttLoadAbort.current === controller) sttLoadAbort.current = null;
      }
    },
    [refreshSttStatus, setLastSttRepo],
  );

  return {
    lastSttRepo,
    selectedSttRepo,
    setSelectedSttRepo,
    sttLoadedModel,
    sttLoadedEngine,
    downloadedSttArtifacts,
    selectedSttRepoRef,
    sttStatusRefreshGeneration,
    audioCppRuntime,
    sttLoadGeneration,
    sttGgufVariants,
    setLastSttRepo,
    sttLoadingGeneration,
    sttLoadedByThisPage,
    sttLoadAbort,
    deferredSttLoad,
    refreshSttStatus,
    sttSelected,
    sttReady,
    releaseTranscribeSelection,
    ensureSttLoaded,
  };
}

export type SttSidecar = ReturnType<typeof useSttSidecar>;
