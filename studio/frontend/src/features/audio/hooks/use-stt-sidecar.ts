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
  sttEjectSucceeded,
  sttSelectionReady,
} from "../audio-page-policy";
import { audioModelLabel } from "../audio-workspace-utils";
import {
  type AudioSttEngine,
  sttEngineForRepoId,
  sttRepoIdForSidecarKey,
  sttSidecarKeyFor,
} from "../catalog";
import { isAudioSttEngine } from "../stt-artifacts";
import type { AudioHostState } from "./audio-host-state";

export function useSttSidecar({
  setBusy,
  activeRef,
}: Pick<AudioHostState, "setBusy" | "activeRef">) {
  const [lastSttRepo, setLastSttRepoChoice] = usePersistedChoice("unsloth:audio:last-stt-model", "");
  const [lastSttVariant, setLastSttVariant] = usePersistedChoice("unsloth:audio:last-stt-variant", "");
  const [lastSttEngine, setLastSttEngine] = usePersistedChoice("unsloth:audio:last-stt-engine", "");
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
  const audioCppRuntime = useRef<AudioCppRuntimeStatus | null>(null);
  const sttLoadGeneration = useRef(0);
  const sttGgufVariants = useRef(
    new Map<string, string>(
      lastSttRepo && lastSttVariant ? [[lastSttRepo.toLowerCase(), lastSttVariant]] : [],
    ),
  );
  // The engine resolved at pick time (a GGUF repo without a -GGUF suffix runs on audio.cpp);
  // re-deriving it from the id alone sends later status reads and loads to Transformers.
  const sttEngines = useRef(
    new Map<string, AudioSttEngine>(
      lastSttRepo && isAudioSttEngine(lastSttEngine)
        ? [[lastSttRepo.toLowerCase(), lastSttEngine]]
        : [],
    ),
  );
  const sttEngineFor = useCallback(
    (repo: string): AudioSttEngine =>
      sttEngines.current.get(repo.toLowerCase()) ?? sttEngineForRepoId(repo),
    [],
  );
  const setLastSttRepo = useCallback(
    (repo: string) => {
      setLastSttRepoChoice(repo);
      setLastSttVariant(sttGgufVariants.current.get(repo.toLowerCase()) ?? "");
      setLastSttEngine(sttEngines.current.get(repo.toLowerCase()) ?? "");
    },
    [setLastSttRepoChoice, setLastSttVariant, setLastSttEngine],
  );
  const sttLoadingGeneration = useRef<number | null>(null);
  // Residency is not ownership: track the owned model identity, since other surfaces can swap the sidecar.
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
      const selectedEngine = selectedRepo ? sttEngineFor(selectedRepo) : null;
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
        engineForRepo: sttEngineFor,
      });
      // A pick made while this read was in flight owns the selection now.
      if (selectedSttRepoRef.current !== selectedRepo) return;
      // The resident engine wins even for the same repo id: another tab can swap a
      // suffix-less repo onto audio.cpp under the same name.
      if (reconciled && residency) {
        sttEngines.current.set(reconciled.toLowerCase(), residency.engine);
      }
      selectedSttRepoRef.current = reconciled;
      setSelectedSttRepo(reconciled);
    } catch {
      if (generation !== sttStatusRefreshGeneration.current) return;
      setSttLoadedModel(null);
      setSttLoadedEngine(null);
    }
  }, [sttEngineFor]);

  const sttSelected = selectedSttRepo !== null;
  const sttReady = sttSelectionReady(
    selectedSttRepo,
    sttLoadedModel,
    sttSidecarKeyFor,
    selectedSttRepo ? sttEngineFor(selectedSttRepo) : null,
    sttLoadedEngine,
    sttRepoIdForSidecarKey,
  );

  const forgetSttSelection = useCallback(() => {
    deferredSttLoad.current = null;
    selectedSttRepoRef.current = null;
    sttLoadGeneration.current += 1;
    sttLoadedByThisPage.current = null;
    setSelectedSttRepo(null);
  }, []);

  const releaseTranscribeSelection = useCallback(async () => {
    const selected = selectedSttRepoRef.current;
    const claim = sttLoadedByThisPage.current;
    const owned =
      sttReady &&
      selected !== null &&
      claim !== null &&
      claim === sttLoadedModel;
    if (!owned) {
      forgetSttSelection();
      return;
    }
    // Forget only once released, else a failed unload leaves the model in VRAM with no Eject.
    await unloadSttModel(sttEngineFor(selected), claim);
    forgetSttSelection();
    await refreshSttStatus();
  }, [forgetSttSelection, refreshSttStatus, sttEngineFor, sttReady, sttLoadedModel]);

  /** An explicit Eject releases whatever is resident, whoever loaded it (a reload, Settings > Voice
   *  or dictation leave no page claim); the backend's expected-model guard keeps a newer model. */
  const ejectTranscribeModel = useCallback(async (): Promise<
    "unloaded" | "kept" | "none"
  > => {
    if (!sttLoadedModel) {
      forgetSttSelection();
      await refreshSttStatus();
      return "none";
    }
    const { released } = await unloadSttModel(
      sttLoadedEngine ?? undefined,
      sttLoadedModel,
    );
    forgetSttSelection();
    await refreshSttStatus();
    return sttEjectSucceeded(released) ? "unloaded" : "kept";
  }, [forgetSttSelection, refreshSttStatus, sttLoadedEngine, sttLoadedModel]);

  /** Read residency fresh: the sidecar unloads itself after its idle keep-alive. */
  const readSttResidency = useCallback(async (engine: AudioSttEngine) => {
    const stt = await fetchSttStatus();
    return resolveSttResidency(stt, engine, false);
  }, []);

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
      // Claim ownership only once resident: a cancelled download would otherwise unload another surface's model.
      const toastId = toast.loading(`Preparing ${audioModelLabel(sidecarKey)}…`);
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
          const download = await startSttDownload(
            sidecarKey,
            hfApiToken(getHfToken()),
            engine,
            ggufVariant,
          );
          trackSttDownload(sidecarKey, {
            warmSelectedVoiceModelOnComplete: false,
            engine,
            repoId,
            ggufVariant,
            downloadId: download.download_id,
          });
          toast.dismiss(toastId);
          // Re-check ownership around every await: an old pick could replace a newer sidecar.
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
            // Cancel is terminal: never load a partial checkpoint.
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
    sttEngines,
    sttEngineFor,
    setLastSttRepo,
    sttLoadingGeneration,
    sttLoadedByThisPage,
    sttLoadAbort,
    deferredSttLoad,
    refreshSttStatus,
    sttSelected,
    sttReady,
    releaseTranscribeSelection,
    ejectTranscribeModel,
    readSttResidency,
    ensureSttLoaded,
  };
}

export type SttSidecar = ReturnType<typeof useSttSidecar>;
