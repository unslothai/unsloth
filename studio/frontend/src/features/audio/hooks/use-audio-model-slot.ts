// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import {
  cancelPreStreamRunReservations,
  confirmStopRunningChatsIfNeeded,
  listGgufVariants,
  listLoras,
  loadModel,
  offloadCountsFrom,
  offloadWarning,
  requestLocalPromptQueueStop,
  unloadModel,
  useChatRuntimeStore,
} from "@/features/chat";
import {
  fetchSttStatus,
  sttEngineStatusFor,
} from "@/features/chat/adapters/studio-model-dictation-adapter";
import { useStagedDownload } from "@/features/hub/download-manager";
import { getHfToken, hfApiToken } from "@/features/hub/stores/hf-token-store";
import type {
  ModelOption,
  ModelSelectorChangeMeta,
} from "@/features/model-picker/components/model-selector/types";
import { confirmRemoteCodeIfNeeded } from "@/features/security";
import { fetchSystemInfo } from "@/hooks/use-system";
import { toast } from "@/lib/toast";
import { getAudioDownloadPlan } from "../api";
import {
  audioCppRuntimeProblem,
  canTransitionAudioMode,
  exactGgufLoadSelector,
  expectedGgufDownloadBytes,
  isGgufTtsTarget,
  isTtsAudioType,
  macTtsPickAction,
  resolveAudioPickTask,
  selectAutoGgufVariant,
  stagedTtsLoadIsOwned,
  trainedTtsCheckpointIsLoadable,
  trainedTtsCheckpointIsRunnableOnMac,
} from "../audio-page-policy";
import {
  TTS_MAX_TOKENS,
  TTS_PROMPT_CONTEXT_RESERVE,
} from "../audio-workspace-constants";
import {
  audioModelLabel,
  type CreateMode,
  loadedAudioKind,
  type RemoteCodeApproval,
} from "../audio-workspace-utils";
import {
  audioModelRequiresRemoteCode,
  audioTaskFor,
  ggufSiblingFor,
  isMusicGenerationModel,
  musicGenerationRequiresCuda,
  sttEngineForRepoId,
  sttSidecarKeyFor,
  usesNativeAudioRuntime,
} from "../catalog";
import { useAudioWorkspaceStore } from "../stores/audio-workspace-store";
import {
  type AudioWorkflowId,
  slotForWorkflow,
  workflowForLoadedModel,
} from "../workflows";
import type { AudioHostState } from "./audio-host-state";
import type { SttSidecar } from "./use-stt-sidecar";
import type { Transcription } from "./use-transcription";

/** The main inference slot Speak and Music share: picks, loads, staged downloads, mode switches and eject. */
export function useAudioModelSlot({
  active,
  activeRef,
  busy,
  busyRef,
  setBusy,
  mode,
  setMode,
  modeRef,
  generationPhaseRef,
  handleStopGeneration,
  refreshStatus,
  audioDevice,
  isMac,
  status,
  isRecording,
  stopAndDiscardRecording,
  releaseTranscribeSelection,
  ensureSttLoaded,
  sttGgufVariants,
  selectedSttRepoRef,
  setSelectedSttRepo,
  deferredSttLoad,
  sttLoadingGeneration,
  sttLoadGeneration,
  sttLoadAbort,
  audioCppRuntime,
  selectedSttRepo,
  sttReady,
  sttStatusRefreshGeneration,
}: Pick<
  AudioHostState,
  | "active"
  | "activeRef"
  | "busy"
  | "busyRef"
  | "setBusy"
  | "mode"
  | "setMode"
  | "modeRef"
  | "generationPhaseRef"
  | "handleStopGeneration"
  | "refreshStatus"
  | "audioDevice"
  | "isMac"
  | "status"
> &
  Pick<Transcription, "isRecording" | "stopAndDiscardRecording"> &
  Pick<
    SttSidecar,
    | "releaseTranscribeSelection"
    | "ensureSttLoaded"
    | "sttGgufVariants"
    | "selectedSttRepoRef"
    | "setSelectedSttRepo"
    | "deferredSttLoad"
    | "sttLoadingGeneration"
    | "sttLoadGeneration"
    | "sttLoadAbort"
    | "audioCppRuntime"
    | "selectedSttRepo"
    | "sttReady"
    | "sttStatusRefreshGeneration"
  >) {
  const ttsLoadInFlight = useRef(false);
  // A pick that lost the race with a load still settling. Replayed once it does.
  const pendingRoutedTtsPick = useRef<{
    repoId: string;
    ggufFilename?: string | null;
    loadId?: string | null;
    audioType?: string | null;
    remoteCodeApproval?: RemoteCodeApproval;
    isGguf?: boolean | null;
  } | null>(null);
  const ttsLoadGeneration = useRef(0);
  const pendingTtsLoad = useRef<{
    generation: number;
    repoId: string;
    /** What the load request actually sent, which is what a cancel has to name. */
    loadTarget: string;
    loadRequestId: string;
    controller: AbortController;
    requestStarted: boolean;
  } | null>(null);

  const ttsPickGeneration = useRef(0);
  const ttsInspectionGeneration = useRef<number | null>(null);
  const stagedTtsGeneration = useRef(0);
  const pendingStagedTtsLoad = useRef<{
    repoId: string;
    ggufFilename: string | null;
    loadId?: string | null;
    audioType?: string | null;
    remoteCodeApproval?: RemoteCodeApproval;
    isGguf?: boolean | null;
    generation: number;
  } | null>(null);
  const stagedTtsLoadDeferred = useRef(false);

  const loadTtsModel = useCallback(
    async (
      repoId: string,
      ggufFilename?: string | null,
      // Where the weights actually are: a row cached in a NON-ACTIVE HF cache is loadable only by its snapshot path,
      // which the picker supplies as meta.loadId, and sending the display repo id instead failed offline or
      // re-downloaded into the active cache. Chat threads the same field (chat-page.tsx).
      loadId?: string | null,
      audioType?: string | null,
      remoteCodeApproval?: RemoteCodeApproval,
      // The catalog's answer: the ids alone miss a GGUF repo that does not spell it.
      isGguf?: boolean | null,
    ) => {
      // A routed pick arriving while a previous load is still tearing down would otherwise be dropped,
      // and the route effect has already cleared ?model=, so replay it from the finally below.
      if (ttsLoadInFlight.current || busyRef.current === "generating") {
        pendingRoutedTtsPick.current = {
          repoId,
          ggufFilename,
          loadId,
          audioType,
          remoteCodeApproval,
          isGguf,
        };
        return;
      }
      // A load stops every chat on the shared llama-server, so ask the way Chat does instead of dead-ending on the
      // backend's 409. Claimed before the await: a routed pick arriving while the dialog is open must queue.
      ttsLoadInFlight.current = true;
      // Chat's gate, held across the question and the load. Without it a queue can materialize while
      // the dialog is open, outside the snapshot the answer was given for.
      const lifecycleLease = useChatRuntimeStore.getState().beginModelLoading();
      if (lifecycleLease === null) {
        ttsLoadInFlight.current = false;
        pendingRoutedTtsPick.current = null;
        toast.info("Wait for the current model to finish loading.");
        return;
      }
      const releaseLifecycle = () =>
        useChatRuntimeStore.getState().endModelLoading(lifecycleLease);
      const stopDecision = await confirmStopRunningChatsIfNeeded();
      if (!stopDecision.proceed) {
        releaseLifecycle();
        ttsLoadInFlight.current = false;
        // Declining refuses the swap, so a queued pick must not reopen the dialog.
        pendingRoutedTtsPick.current = null;
        return;
      }
      // The page can go away while the dialog is open, and pendingTtsLoad is still null then, so the
      // deactivation effect has nothing to abort. Queue it for the activation replay.
      if (!activeRef.current) {
        releaseLifecycle();
        ttsLoadInFlight.current = false;
        pendingRoutedTtsPick.current = {
          repoId,
          ggufFilename,
          loadId,
          audioType,
          remoteCodeApproval,
          isGguf,
        };
        return;
      }
      const generation = ++ttsLoadGeneration.current;
      const controller = new AbortController();
      const loadRequestId = crypto.randomUUID();
      const pending = {
        generation,
        repoId,
        // What the request actually sent. Cancelling under the display id works only when the load target is a
        // standard HF cache snapshot; a pinned directory elsewhere does not match and _cancel_scoped_load_attempt then
        // refuses.
        loadTarget: loadId || repoId,
        loadRequestId,
        controller,
        requestStarted: false,
      };
      pendingTtsLoad.current = pending;
      const isCurrent = () =>
        generation === ttsLoadGeneration.current && activeRef.current;
      ttsLoadInFlight.current = true;
      busyRef.current = "loading";
      setBusy("loading");
      const toastId = toast.loading(`Loading ${audioModelLabel(repoId)}…`);
      try {
        const hfToken = hfApiToken(getHfToken()) ?? null;
        let trustRemoteCode = remoteCodeApproval?.trustRemoteCode ?? false;
        let approvedRemoteCodeFingerprint =
          remoteCodeApproval?.approvedRemoteCodeFingerprint ?? null;
        if (
          audioModelRequiresRemoteCode(repoId, audioType) &&
          !remoteCodeApproval
        ) {
          const approved = await confirmRemoteCodeIfNeeded({
            modelName: loadId || repoId,
            hfToken,
            requiresTrustRemoteCode: true,
            onApprove: (fingerprint) => {
              trustRemoteCode = true;
              approvedRemoteCodeFingerprint = fingerprint;
            },
          });
          if (!approved)
            throw new Error(
              "Custom code approval is required to load this model.",
            );
          if (controller.signal.aborted || !isCurrent()) return;
        }
        const wantsCpu = audioDevice === "cpu";
        const isGgufLoad = isGgufTtsTarget({ repoId, ggufFilename, loadId, isGguf });
        const res = await loadModel(
          {
            model_path: loadId || repoId,
            load_request_id: loadRequestId,
            force_cancel_active: stopDecision.forceCancelActive,
            hf_token: hfToken,
            max_seq_length: TTS_MAX_TOKENS + TTS_PROMPT_CONTEXT_RESERVE,
            load_in_4bit: false,
            is_lora: false,
            gguf_variant: ggufFilename ?? null,
            trust_remote_code: trustRemoteCode,
            approved_remote_code_fingerprint: approvedRemoteCodeFingerprint,
            audio_device: wantsCpu ? "cpu" : "auto",
            // GGUF ignores audio_device: llama.cpp offloads unless told not to. An absent speculative_type resolves to
            // "auto", which may attach a GPU drafter, and the backend then evicts image/video for a CPU load.
            ...(wantsCpu && isGgufLoad
              ? // biome-ignore lint/style/useNamingConvention: API schema
                {
                  gpu_memory_mode: "manual" as const,
                  gpu_layers: 0,
                  speculative_type: "off" as const,
                }
              : {}),
          },
          {
            signal: controller.signal,
            runtime: "tts",
            onRequestStart: () => {
              pending.requestStarted = true;
              // Queued prompts would otherwise start on the model this load replaces. Only once /load is actually going
              // out: loadModel returns without sending when a stored token is invalid, and cancelling earlier threw away
              // accepted sends for a swap that never happened.
              cancelPreStreamRunReservations(stopDecision.preStreamRunTokens);
              requestLocalPromptQueueStop(stopDecision.promptQueueThreadIds);
            },
          },
        );
        if (!isCurrent()) return;
        if (res.is_audio && isTtsAudioType(res.audio_type)) {
          const offloadNotice = offloadWarning(offloadCountsFrom(res));
          const showToast = offloadNotice ? toast.warning : toast.success;
          showToast(
            `Model loaded (${loadedAudioKind(res.audio_type)})${offloadNotice?.titleSuffix ?? ""}`,
            {
              id: toastId,
              description: offloadNotice?.description,
              duration: offloadNotice ? 8000 : undefined,
            },
          );
          // The page follows what it just loaded: a music model opens Music, a clone-only model Clone,
          // and a model that can run the open page keeps it.
          const loadedWorkflow = workflowForLoadedModel({
            current: useAudioWorkspaceStore.getState().workflow,
            audioWorkflows: res.audio_workflows,
            music: isMusicGenerationModel(repoId, res.audio_type),
          });
          const workspace = useAudioWorkspaceStore.getState();
          workspace.rememberModel(loadedWorkflow, repoId);
          if (modeRef.current === "speak") workspace.commitWorkflow(loadedWorkflow);
          // Only the native runtime and GGUF can be held in RAM.
          if (
            wantsCpu &&
            !isGgufLoad &&
            !usesNativeAudioRuntime(repoId, res.audio_type)
          ) {
            toast.info(
              "This model does not support CPU RAM yet, so it loaded on the GPU.",
              { duration: 6000 },
            );
          }
        } else {
          toast.error(`${repoId} loaded but is not a supported TTS model.`, {
            id: toastId,
          });
        }
      } catch (error) {
        if (isCurrent()) {
          toast.error(
            error instanceof Error ? error.message : "Model load failed.",
            { id: toastId },
          );
        } else {
          toast.dismiss(toastId);
        }
      } finally {
        if (pendingTtsLoad.current?.generation === generation)
          pendingTtsLoad.current = null;
        if (activeRef.current) await refreshStatus();
        ttsLoadInFlight.current = false;
        // Before the replay below, which needs the gate for its own attempt.
        releaseLifecycle();
        busyRef.current = null;
        setBusy(null);
        // Only while Audio is visible: replaying unconditionally started a load with activeRef already false, which
        // the deactivation effect never saw to cancel, so a hidden page could replace the model Chat had loaded.
        if (activeRef.current) replayQueuedTtsPick();
      }
    },
    [refreshStatus, audioDevice],
  );

  // Stage uncached Hub GGUFs through the shared manager so Audio gets the same progress,
  // cancellation, resume and disk preflight as Chat/Images/Video.
  const loadTtsModelRef = useRef(loadTtsModel);
  loadTtsModelRef.current = loadTtsModel;
  /** Start a pick that lost the race with a load still settling. Visible pages only: a load started
   *  while hidden outlives the deactivation effect that would cancel it. */
  const replayQueuedTtsPick = useCallback(() => {
    const queued = pendingRoutedTtsPick.current;
    if (!queued) return;
    pendingRoutedTtsPick.current = null;
    void loadTtsModelRef.current(
      queued.repoId,
      queued.ggufFilename,
      queued.loadId,
      queued.audioType,
      queued.remoteCodeApproval,
      queued.isGguf,
    );
  }, []);
  const invalidatePendingStagedTts = useCallback(() => {
    stagedTtsGeneration.current += 1;
    pendingStagedTtsLoad.current = null;
    stagedTtsLoadDeferred.current = false;
  }, []);
  const invalidatePendingTtsSelection = useCallback(() => {
    ttsPickGeneration.current += 1;
    pendingRoutedTtsPick.current = null;
    invalidatePendingStagedTts();
  }, [invalidatePendingStagedTts]);
  const transitionMode = useCallback(
    (nextMode: CreateMode) => {
      if (nextMode === mode) {
        if (nextMode === "transcribe") invalidatePendingTtsSelection();
        return true;
      }
      if (
        !canTransitionAudioMode(busyRef.current, generationPhaseRef.current)
      ) {
        toast.info(
          "Wait for the active audio task to finish before switching modes.",
        );
        return false;
      }

      if (nextMode === "transcribe") invalidatePendingTtsSelection();
      if (busyRef.current === "generating") handleStopGeneration();
      stopAndDiscardRecording();
      setMode(nextMode);
      // Held through Generate, the sidecar keeps a dictation model in VRAM beside the speech one.
      if (mode === "transcribe") {
        // Resolves to whether the sidecar is actually gone: swallowing the rejection made a failed unload
        // look like a release, so the speech load went ahead with the dictation model and OOMed.
        const release = releaseTranscribeSelection().then(
          () => true,
          (error) => {
            toast.error(
              error instanceof Error
                ? error.message
                : "Failed to release the transcription model.",
            );
            return false;
          },
        );
        // Recorded so a TTS load can wait for the teardown: allocating while the sidecar still holds its
        // model is what OOMs a device that fits either one alone.
        pendingTranscribeRelease.current = release;
        void release.then((released) => {
          if (pendingTranscribeRelease.current !== release) return;
          pendingTranscribeRelease.current = null;
          // The sidecar is still holding its model, so the page must not sit in Speak claiming otherwise:
          // back to Transcribe, where Eject can retry. Only if nothing has moved on since.
          if (!released && modeRef.current === "speak") setMode("transcribe");
        });
      }
      return true;
    },
    [
      invalidatePendingTtsSelection,
      handleStopGeneration,
      mode,
      releaseTranscribeSelection,
      stopAndDiscardRecording,
    ],
  );
  /** Switch pages. Speak and Music share the main slot, so moving between them loads and releases
   *  nothing; a move to or from Transcribe is the mode switch, with its gate and sidecar release. */
  const transitionWorkflow = useCallback(
    (next: AudioWorkflowId) => {
      if (!transitionMode(slotForWorkflow(next))) return false;
      useAudioWorkspaceStore.getState().commitWorkflow(next);
      return true;
    },
    [transitionMode],
  );
  const { stage: stageTtsDownload } = useStagedDownload({
    scopeId: "audio",
    onReady: () => {
      const pending = pendingStagedTtsLoad.current;
      if (
        !pending ||
        !stagedTtsLoadIsOwned(
          pending.generation,
          stagedTtsGeneration.current,
          modeRef.current,
        )
      ) {
        pendingStagedTtsLoad.current = null;
        stagedTtsLoadDeferred.current = false;
        return;
      }
      // Audio stays mounted across tabs, so loading from a hidden page would evict the visible page's
      // model; likewise, wait for an active generation to end.
      if (!active || busyRef.current !== null) {
        stagedTtsLoadDeferred.current = true;
        return;
      }
      pendingStagedTtsLoad.current = null;
      void loadTtsModelRef.current(
        pending.repoId,
        pending.ggufFilename,
        pending.loadId,
        pending.audioType,
        pending.remoteCodeApproval,
        pending.isGguf,
      );
    },
  });

  useEffect(() => {
    if (!active || busy !== null || !stagedTtsLoadDeferred.current) return;
    stagedTtsLoadDeferred.current = false;
    const pending = pendingStagedTtsLoad.current;
    if (
      !pending ||
      !stagedTtsLoadIsOwned(
        pending.generation,
        stagedTtsGeneration.current,
        modeRef.current,
      )
    ) {
      pendingStagedTtsLoad.current = null;
      return;
    }
    pendingStagedTtsLoad.current = null;
    void loadTtsModelRef.current(
      pending.repoId,
      pending.ggufFilename,
      pending.loadId,
      pending.audioType,
      pending.remoteCodeApproval,
      pending.isGguf,
    );
  }, [active, busy]);

  const loadOrStageTtsModel = useCallback(
    async (
      repoId: string,
      ggufFilename: string | null,
      meta: ModelSelectorChangeMeta,
    ) => {
      const generation = ++stagedTtsGeneration.current;
      pendingStagedTtsLoad.current = null;
      stagedTtsLoadDeferred.current = false;
      stageTtsDownload([]);

      let remoteCodeApproval: RemoteCodeApproval | undefined;
      const hfToken = hfApiToken(getHfToken()) ?? null;
      if (
        meta.source === "hub" &&
        !ggufFilename &&
        audioModelRequiresRemoteCode(repoId, meta.audioType)
      ) {
        try {
          const approved = await confirmRemoteCodeIfNeeded({
            modelName: meta.loadId || repoId,
            hfToken,
            requiresTrustRemoteCode: true,
            onApprove: (fingerprint) => {
              remoteCodeApproval = {
                trustRemoteCode: true,
                approvedRemoteCodeFingerprint: fingerprint,
              };
            },
          });
          if (generation !== stagedTtsGeneration.current) return;
          if (!approved) {
            toast.error("Custom code approval is required to load this model.");
            return;
          }
        } catch (error) {
          if (generation !== stagedTtsGeneration.current) return;
          toast.error(
            error instanceof Error
              ? error.message
              : `Could not verify the code for ${repoId}.`,
          );
          return;
        }
      }

      // Hub TTS picks use the same managed path as Chat. Native models can also depend on a second codec
      // repository, so the selected repo's downloaded badge is not enough: the cache-aware backend plan owns every
      // missing file.
      if (meta.source === "hub" && !ggufFilename) {
        let plan;
        try {
          plan = await getAudioDownloadPlan(
            meta.loadId || repoId,
            hfToken ?? undefined,
          );
        } catch (error) {
          if (generation !== stagedTtsGeneration.current) return;
          if (meta.isDownloaded === true) {
            void loadTtsModelRef.current(
              repoId,
              ggufFilename,
              meta.loadId,
              meta.audioType,
              remoteCodeApproval,
              meta.isGguf,
            );
            return;
          }
          toast.error(
            error instanceof Error
              ? error.message
              : `Could not prepare the download for ${repoId}.`,
          );
          return;
        }
        if (generation !== stagedTtsGeneration.current) return;
        const plannedEntries = plan.entries;
        if (plannedEntries.length > 0) {
          pendingStagedTtsLoad.current = {
            repoId,
            ggufFilename,
            loadId: meta.loadId,
            audioType: meta.audioType,
            remoteCodeApproval,
            isGguf: meta.isGguf,
            generation,
          };
          stageTtsDownload(
            plannedEntries.map((entry) => ({
              repoId: entry.repo_id,
              files: entry.files,
              bytes: entry.bytes,
              ggufFilename: entry.gguf_filename,
              checkpoint: entry.checkpoint,
            })),
          );
          return;
        }
      }

      if (
        meta.source === "hub" &&
        meta.isDownloaded === false &&
        ggufFilename
      ) {
        pendingStagedTtsLoad.current = {
          repoId,
          ggufFilename,
          loadId: meta.loadId,
          audioType: meta.audioType,
          remoteCodeApproval,
          isGguf: meta.isGguf,
          generation,
        };
        // A named quant is fetched as the standard variant download, as Chat does, so the backend's plan
        // brings the files it needs beside the weights and the Downloads row reads "<repo> · <quant>".
        stageTtsDownload([
          meta.ggufVariant
            ? {
                repoId,
                files: [],
                bytes: meta.expectedBytes ?? 0,
                ggufFilename,
                ggufVariant: meta.ggufVariant,
              }
            : {
                repoId,
                files: [ggufFilename],
                bytes: meta.expectedBytes ?? 0,
                ggufFilename,
              },
        ]);
        return;
      }

      // A cached/local/direct pick supersedes any staged auto-load: the manager may keep downloading
      // globally, but its old completion cannot load here.
      void loadTtsModelRef.current(
        repoId,
        ggufFilename,
        meta.loadId,
        meta.audioType,
        remoteCodeApproval,
        meta.isGguf,
      );
    },
    [stageTtsDownload],
  );

  // A hidden page may let the shared download continue, but it must not load the sidecar. Returning
  // to the same still-selected repo resumes preparation once; a different selection clears it.
  useEffect(() => {
    if (!active) {
      ttsPickGeneration.current += 1;
      const pending = pendingTtsLoad.current;
      if (pending) {
        ttsLoadGeneration.current += 1;
        pendingTtsLoad.current = null;
        pending.controller.abort();
        if (pending.requestStarted)
          void unloadModel({
            model_path: pending.loadTarget,
            cancel_load_request_id: pending.loadRequestId,
          }).catch(() => {});
      }
      if (ttsInspectionGeneration.current !== null) {
        ttsInspectionGeneration.current = null;
        busyRef.current = null;
        setBusy(null);
      }

      if (sttLoadingGeneration.current !== null) {
        const repoId = selectedSttRepoRef.current;
        deferredSttLoad.current = repoId
          ? {
              repoId,
              sidecarKey: sttSidecarKeyFor(repoId),
              engine: sttEngineForRepoId(repoId),
            }
          : null;
        sttLoadGeneration.current += 1;
        sttLoadingGeneration.current = null;
        sttLoadAbort.current?.abort();
        sttLoadAbort.current = null;
        busyRef.current = null;
        setBusy(null);
      }
      return;
    }

    // A pick queued behind a settling load, held back because the page went away before the load
    // finished. Now that Audio is visible again the attempt is cancellable.
    replayQueuedTtsPick();

    const deferred = deferredSttLoad.current;
    deferredSttLoad.current = null;
    if (deferred && selectedSttRepoRef.current === deferred.repoId) {
      void (async () => {
        try {
          const status = await fetchSttStatus(
            undefined,
            deferred.engine === "transformers"
              ? deferred.sidecarKey
              : undefined,
          );
          const download = sttEngineStatusFor(
            status,
            deferred.sidecarKey,
            deferred.engine,
          )?.download;
          // `model` is null once the download thread has stopped, so a cancellation made while this page
          // was hidden matched nothing here and the deferred load restarted the whole download.
          if (
            download?.cancelled &&
            (download.model ?? download.cancelled_model) === deferred.sidecarKey
          )
            return;
        } catch {
          // Status is advisory here; the normal preparation path reports errors.
        }
        if (activeRef.current && selectedSttRepoRef.current === deferred.repoId)
          void ensureSttLoaded(
            deferred.repoId,
            deferred.sidecarKey,
            deferred.engine,
          );
      })();
    }
  }, [active, ensureSttLoaded]);

  /** An in-flight Transcribe teardown a following TTS load has to wait behind. */
  const pendingTranscribeRelease = useRef<Promise<boolean> | null>(null);

  const handleModelSelect = useCallback(
    async (id: string, meta: ModelSelectorChangeMeta) => {
      if (busyRef.current !== null) return;
      // Catalog first; an uncurated Hub pick falls back to its pipeline tag, or every community ASR
      // repo would load into the TTS slot.
      const task = resolveAudioPickTask(audioTaskFor(id), meta.pipelineTag);
      // A recommended GGUF speech or music row the installed audio runtime cannot run: say why
      // instead of loading into a 501.
      const runtimeProblem =
        task === "stt" ? null : audioCppRuntimeProblem(id, audioCppRuntime.current);
      if (runtimeProblem) {
        toast.error(runtimeProblem, { duration: 7000 });
        return;
      }
      const cudaMusicPick =
        task !== "stt" && musicGenerationRequiresCuda(id, meta.audioType);
      const selectionGeneration = ++ttsPickGeneration.current;
      if (cudaMusicPick) {
        const system = await fetchSystemInfo();
        if (selectionGeneration !== ttsPickGeneration.current) return;
        if (system?.device_backend !== "cuda") {
          toast.error(
            `${id} requires a verified NVIDIA CUDA GPU for local generation.`,
            { duration: 7000 },
          );
          return;
        }
      }
      // Selecting a different artifact while recording is a lifecycle change even in Transcribe mode;
      // never let the old capture submit against a sidecar this pick is replacing.
      stopAndDiscardRecording();
      deferredSttLoad.current = null;
      if (task === "stt") {
        // An STT pick owns Transcribe: it runs on the sidecar, not the main slot.
        if (!transitionMode("transcribe")) return;
        const sidecarKey = sttSidecarKeyFor(id);
        const engine = sttEngineForRepoId(id, meta.isGguf);
        // Remembered per repo, so a deferred or on-demand load of this pick asks for the same quant.
        if (meta.ggufVariant) {
          sttGgufVariants.current.set(id.toLowerCase(), meta.ggufVariant);
        } else {
          sttGgufVariants.current.delete(id.toLowerCase());
        }
        deferredSttLoad.current = null;
        selectedSttRepoRef.current = id;
        setSelectedSttRepo(id);
        void ensureSttLoaded(id, sidecarKey, engine);
        return;
      }
      // TTS (or an uncurated repo the user pasted, which /load will validate).
      if (!transitionMode("speak")) return;
      // Serialize against a Transcribe release started by that transition.
      const releaseInFlight = pendingTranscribeRelease.current;
      // A release that failed leaves the sidecar resident, so do not stack a speech model on top of it: back to
      // Transcribe, where Eject can retry. Claimed before the await below, not after: the button only disables on
      // `busy`, so a slow release let several clicks through, each resuming into its own generateAudio while
      // generateAbort tracked only the last.
      if (releaseInFlight && !(await releaseInFlight)) {
        setMode("transcribe");
        return;
      }
      if (ttsPickGeneration.current !== selectionGeneration) return;
      const exactGguf = exactGgufLoadSelector(meta);
      const isGguf = isGgufTtsTarget({
        repoId: id,
        ggufFilename: exactGguf,
        isGguf: meta.isGguf,
      });
      const ggufSibling = isGguf ? null : ggufSiblingFor(id);
      const nativeRuntime =
        usesNativeAudioRuntime(id, meta.audioType) && !cudaMusicPick;
      const macAction = macTtsPickAction({
        isMac,
        isGguf,
        ggufSibling,
        nativeRuntime,
      });
      if (macAction === "reject") {
        toast.error(
          cudaMusicPick
            ? `${id} currently requires an NVIDIA CUDA GPU and cannot run locally on this Mac.`
            : `${id} has no runnable GGUF TTS build. MLX cannot generate text-to-speech from its safetensors checkpoint on this Mac.`,
          { duration: 7000 },
        );
        return;
      }
      if (macAction === "use-gguf-sibling" && ggufSibling) {
        toast.info(
          `Loading the GGUF build of ${id}. MLX has no text-to-speech decoder, so the safetensors build cannot generate on this Mac.`,
          { duration: 7000 },
        );
        // Resolving the sibling is part of the model load lifecycle. Reserve the slot so Generate cannot
        // run the old resident model and then be evicted by this inspection's completion.
        ttsInspectionGeneration.current = selectionGeneration;
        busyRef.current = "loading";
        setBusy("loading");
        try {
          const listing = await listGgufVariants(
            ggufSibling,
            hfApiToken(getHfToken()),
          );
          if (selectionGeneration !== ttsPickGeneration.current) return;
          const variant = selectAutoGgufVariant(
            listing.variants,
            listing.default_variant,
          );
          if (!variant) {
            toast.error(
              `${ggufSibling} does not publish a runnable GGUF file.`,
            );
            return;
          }
          if (ttsInspectionGeneration.current === selectionGeneration) {
            ttsInspectionGeneration.current = null;
            busyRef.current = null;
            setBusy(null);
          }
          await loadOrStageTtsModel(ggufSibling, variant.filename, {
            ...meta,
            source: "hub",
            isGguf: true,
            ggufFilename: variant.filename,
            ggufVariant: variant.quant,
            isDownloaded: variant.downloaded === true && !variant.partial,
            expectedBytes: expectedGgufDownloadBytes(variant),
          });
        } catch (error) {
          if (selectionGeneration !== ttsPickGeneration.current) return;
          toast.error(
            error instanceof Error
              ? error.message
              : `Could not inspect ${ggufSibling}.`,
          );
        } finally {
          if (ttsInspectionGeneration.current === selectionGeneration) {
            ttsInspectionGeneration.current = null;
            busyRef.current = null;
            setBusy(null);
          }
        }
        return;
      }
      await loadOrStageTtsModel(id, exactGguf, meta);
    },
    [
      ensureSttLoaded,
      isMac,
      loadOrStageTtsModel,
      stopAndDiscardRecording,
      transitionMode,
    ],
  );

  const handleEject = useCallback(() => {
    if (busy !== null || isRecording) {
      toast.info("Stop the active audio task before ejecting its model.");
      return;
    }

    // Eject also owns unresolved permission requests. Invalidating here makes their eventual streams
    // self-discard instead of recording for an old STT pick.
    stopAndDiscardRecording();

    if (mode === "transcribe") {
      if (!selectedSttRepo) return;
      // A selection can exist before its sidecar is resident, so an unowned pick is only forgotten.
      if (!sttReady) {
        sttStatusRefreshGeneration.current += 1;
        void releaseTranscribeSelection();
        return;
      }

      setBusy("unloading");
      const toastId = toast.loading("Unloading transcription model…");
      void (async () => {
        try {
          await releaseTranscribeSelection();
          toast.success("Transcription model unloaded", {
            id: toastId,
            duration: 1200,
          });
        } catch (error) {
          toast.error(
            error instanceof Error
              ? error.message
              : "Failed to unload transcription model.",
            { id: toastId },
          );
        } finally {
          setBusy(null);
        }
      })();
      return;
    }

    const activeModel = status?.active_model;
    if (!activeModel) return;

    // Chat's gate, taken before the question so a queue cannot materialize while the dialog is open
    // and then be stopped by the blanket queue stop.
    const lifecycleLease = useChatRuntimeStore.getState().beginModelLoading();
    if (lifecycleLease === null) {
      toast.info("Wait for the current model to finish loading.");
      return;
    }

    // Busy before the dialog, so a second eject cannot start behind the first.
    setBusy("unloading");
    void (async () => {
      try {
        // Ejecting stops every chat on the shared llama-server, and unforced the backend refused with a 409 the
        // user could only read. Nothing is torn down until the answer is in, so declining leaves the page as it was.
        const stopDecision = await confirmStopRunningChatsIfNeeded(
          "Unloading the model",
          "unload",
        );
        if (!stopDecision.proceed) {
          setBusy(null);
          return;
        }

        // An old managed completion must not immediately replace the model the user just ejected. The
        // global download may continue for later use.
        invalidatePendingStagedTts();
        stageTtsDownload([]);

        const toastId = toast.loading("Unloading model…");
        try {
          cancelPreStreamRunReservations(stopDecision.preStreamRunTokens);
          requestLocalPromptQueueStop(stopDecision.promptQueueThreadIds);
          await unloadModel({
            model_path: activeModel,
            force_cancel_active: stopDecision.forceCancelActive,
          });
          requestLocalPromptQueueStop();
          await refreshStatus();
          toast.success("Model unloaded", { id: toastId, duration: 1200 });
        } catch (error) {
          toast.error(
            error instanceof Error ? error.message : "Failed to unload model.",
            { id: toastId },
          );
        } finally {
          setBusy(null);
        }
      } finally {
        useChatRuntimeStore.getState().endModelLoading(lifecycleLease);
      }
    })();
  }, [
    busy,
    isRecording,
    mode,
    refreshStatus,
    releaseTranscribeSelection,
    selectedSttRepo,
    invalidatePendingStagedTts,
    stageTtsDownload,
    status?.active_model,
    sttReady,
    stopAndDiscardRecording,
  ]);

  // Trained TTS checkpoints. A scan row carries no modality until the backend tags it, so without
  // this a checkpoint you just fine-tuned here is unreachable.
  const [trainedTtsModels, setTrainedTtsModels] = useState<ModelOption[]>([]);
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    listLoras()
      .then((res) => {
        if (cancelled) return;
        setTrainedTtsModels(
          res.loras
            // Merged native speech checkpoints bypass MLX through the portable audio worker. Other
            // safetensors exports still need a GGUF build on Mac.
            .filter(
              (lora) =>
                !isMac ||
                trainedTtsCheckpointIsRunnableOnMac(
                  lora.audio_type,
                  lora.export_type,
                ),
            )
            // The GGUF flag matters: GGUF_TTS_AUDIO_TYPES leaves csm out because llama.cpp has no CSM
            // decoder, so a csm LoRA exported to GGUF fails at load.
            .filter((lora) =>
              isTtsAudioType(lora.audio_type, lora.export_type === "gguf"),
            )
            .filter((lora) =>
              trainedTtsCheckpointIsLoadable(lora.audio_type, lora.export_type),
            )
            .map((lora) => ({
              id: lora.adapter_path,
              name: audioModelLabel(lora.adapter_path),
              description:
                lora.export_type === "merged"
                  ? `Fine-tuned - ${lora.base_model || "unknown base"}`
                  : `LoRA - ${lora.base_model || "unknown base"}`,
              audioType: lora.audio_type ?? null,
            })),
        );
      })
      .catch(() => {
        // Listing trained models is additive; the catalog rows still work without it.
        if (!cancelled) setTrainedTtsModels([]);
      });
    return () => {
      cancelled = true;
    };
  }, [active, isMac]);

  // A one-click pick from outside the picker (Clone's recommended models): resolve the quant the
  // picker would, then take the same path, so an undownloaded model downloads first.
  const pickRecommendedModel = useCallback(
    async (id: string) => {
      if (busyRef.current !== null) return;
      try {
        const listing = await listGgufVariants(id, hfApiToken(getHfToken()));
        const variant = selectAutoGgufVariant(
          listing.variants,
          listing.default_variant,
        );
        if (!variant) {
          toast.error(`${id} does not publish a runnable GGUF file.`);
          return;
        }
        await handleModelSelect(id, {
          source: "hub",
          isLora: false,
          isGguf: true,
          ggufFilename: variant.filename,
          ggufVariant: variant.quant,
          isDownloaded: variant.downloaded === true && !variant.partial,
          expectedBytes: expectedGgufDownloadBytes(variant),
        });
      } catch (error) {
        toast.error(
          error instanceof Error ? error.message : `Could not inspect ${id}.`,
        );
      }
    },
    [busyRef, handleModelSelect],
  );

  return {
    replayQueuedTtsPick,
    transitionMode,
    transitionWorkflow,
    pendingTranscribeRelease,
    handleModelSelect,
    pickRecommendedModel,
    handleEject,
    trainedTtsModels,
  };
}

export type AudioModelSlot = ReturnType<typeof useAudioModelSlot>;
