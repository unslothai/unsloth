// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AUTH_SESSION_CLEARED_EVENT,
  AUTH_SESSION_STORED_EVENT,
  getAuthSessionEpoch,
} from "@/features/auth";
import {
  applyDictationDictionary,
  recordRecentDictation,
  requestSttDownload,
  useSettingsDialogStore,
  useVoiceSettingsStore,
} from "@/features/settings";
import { useT } from "@/i18n";
import { accountTransitionPending } from "@/lib/account-transition";
import { toast } from "@/lib/toast";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import {
  type SttEngine,
  SttModelNotDownloadedError,
  fetchSttStatus,
  sttEngineFor,
  sttEngineStatusFor,
  transcribeAudioBlob,
} from "../adapters/studio-model-dictation-adapter";
import { resolveDictationChatId } from "../adapters/studio-web-speech-dictation-adapter";
import {
  type ChatAudioUploadFence,
  MAX_AUDIO_SIZE_LABEL,
  appendChatAudioTranscript,
  chatAudioUploadFenceMatches,
  chatAudioUploadFileError,
  completeChatAudioUpload,
} from "../utils/chat-audio-upload";

export interface UseChatAudioUploadOptions {
  owner: string;
  chatId?: string | null;
  disabled?: boolean;
  readDraft: () => string;
  writeDraft: (value: string) => void;
  focusDraft?: () => void;
}

export type ChatAudioUploadReadiness =
  | { state: "idle" | "checking" | "error" | "unavailable"; model: string }
  | { state: "ready" | "missing"; model: string }
  | { state: "downloading"; model: string; progress: number | null };

interface ChatAudioUploadSnapshot extends ChatAudioUploadFence {
  model: string;
  engine: SttEngine;
  language: string;
  device: "auto" | "cpu";
  chatId: string | undefined;
}

interface RetainedRecording {
  file: File;
  snapshot: ChatAudioUploadSnapshot;
  error: string;
}

function readinessProgress(
  done: number | null,
  total: number | null,
): number | null {
  if (done === null || total === null || total <= 0) return null;
  return Math.max(0, Math.min(100, Math.round((done / total) * 100)));
}

export function useChatAudioUpload({
  owner,
  chatId,
  disabled = false,
  readDraft,
  writeDraft,
  focusDraft,
}: UseChatAudioUploadOptions) {
  const t = useT();
  const model = useVoiceSettingsStore((state) => state.sttModel);
  const language = useVoiceSettingsStore((state) => state.dictationLanguage);
  const device = useVoiceSettingsStore((state) => state.sttDevice);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [retained, setRetained] = useState<RetainedRecording | null>(null);
  const [readiness, setReadiness] = useState<ChatAudioUploadReadiness>({
    state: "idle",
    model,
  });
  const generationRef = useRef(0);
  const readinessGenerationRef = useRef(0);
  const ownerRef = useRef(owner);
  const controllerRef = useRef<AbortController | null>(null);
  const activeFenceRef = useRef<ChatAudioUploadFence | null>(null);
  const pickerSnapshotRef = useRef<ChatAudioUploadSnapshot | null>(null);
  const retainedRef = useRef<RetainedRecording | null>(null);
  const readDraftRef = useRef(readDraft);
  const writeDraftRef = useRef(writeDraft);
  const focusDraftRef = useRef(focusDraft);

  const invalidateRefs = useCallback(() => {
    generationRef.current += 1;
    readinessGenerationRef.current += 1;
    pickerSnapshotRef.current = null;
    activeFenceRef.current = null;
    controllerRef.current?.abort();
    controllerRef.current = null;
  }, []);

  const clearOperation = useCallback(() => {
    invalidateRefs();
    retainedRef.current = null;
    setRetained(null);
    setDialogOpen(false);
    setBusy(false);
  }, [invalidateRefs]);

  useLayoutEffect(() => {
    const ownerChanged = ownerRef.current !== owner;
    ownerRef.current = owner;
    readDraftRef.current = readDraft;
    writeDraftRef.current = writeDraft;
    focusDraftRef.current = focusDraft;
    if (ownerChanged) clearOperation();
  }, [clearOperation, focusDraft, owner, readDraft, writeDraft]);

  useLayoutEffect(() => {
    if (!disabled) return;
    invalidateRefs();
    const invalidatedGeneration = generationRef.current;
    queueMicrotask(() => {
      if (generationRef.current !== invalidatedGeneration) return;
      retainedRef.current = null;
      setRetained(null);
      setDialogOpen(false);
      setBusy(false);
    });
  }, [disabled, invalidateRefs]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.addEventListener(AUTH_SESSION_CLEARED_EVENT, clearOperation);
    window.addEventListener(AUTH_SESSION_STORED_EVENT, clearOperation);
    return () => {
      window.removeEventListener(AUTH_SESSION_CLEARED_EVENT, clearOperation);
      window.removeEventListener(AUTH_SESSION_STORED_EVENT, clearOperation);
    };
  }, [clearOperation]);

  useEffect(() => clearOperation, [clearOperation]);

  const targetSettings = useCallback(() => {
    const retry = retainedRef.current;
    return retry
      ? {
          model: retry.snapshot.model,
          language: retry.snapshot.language,
          device: retry.snapshot.device,
          engine: retry.snapshot.engine,
        }
      : {
          model: model.trim(),
          language,
          device,
          engine: sttEngineFor(model.trim()),
        };
  }, [device, language, model]);

  const refreshReadiness = useCallback(
    async (silent = false) => {
      const target = targetSettings();
      const targetModel = target.model;
      const attempt = readinessGenerationRef.current + 1;
      readinessGenerationRef.current = attempt;
      const ownerAtStart = ownerRef.current;
      const authAtStart = getAuthSessionEpoch();
      if (!targetModel) {
        setReadiness({ state: "error", model: targetModel });
        return;
      }
      if (!silent) setReadiness({ state: "checking", model: targetModel });
      try {
        const status = await fetchSttStatus(undefined, targetModel);
        if (
          readinessGenerationRef.current !== attempt ||
          ownerRef.current !== ownerAtStart ||
          getAuthSessionEpoch() !== authAtStart ||
          accountTransitionPending()
        ) {
          return;
        }
        const engineStatus = sttEngineStatusFor(
          status,
          targetModel,
          target.engine,
        );
        if (!engineStatus?.available) {
          setReadiness({ state: "unavailable", model: targetModel });
          return;
        }
        if (
          engineStatus.download.downloading &&
          engineStatus.download.model === targetModel
        ) {
          setReadiness({
            state: "downloading",
            model: targetModel,
            progress: readinessProgress(
              engineStatus.download.bytes_done,
              engineStatus.download.bytes_total,
            ),
          });
          return;
        }
        setReadiness({
          state: engineStatus.downloaded_models.includes(targetModel)
            ? "ready"
            : "missing",
          model: targetModel,
        });
      } catch {
        if (
          readinessGenerationRef.current === attempt &&
          ownerRef.current === ownerAtStart &&
          getAuthSessionEpoch() === authAtStart
        ) {
          setReadiness({ state: "error", model: targetModel });
        }
      }
    },
    [targetSettings],
  );

  useEffect(() => {
    if (!dialogOpen || disabled) return;
    queueMicrotask(() => void refreshReadiness());
  }, [dialogOpen, disabled, refreshReadiness]);

  useEffect(() => {
    if (
      !dialogOpen ||
      (readiness.state !== "missing" && readiness.state !== "downloading")
    ) {
      return;
    }
    const timer = window.setInterval(() => void refreshReadiness(true), 1500);
    return () => window.clearInterval(timer);
  }, [dialogOpen, readiness.state, refreshReadiness]);

  const openDialog = useCallback(() => {
    if (disabled || busy) return;
    setDialogOpen(true);
  }, [busy, disabled]);

  const closeDialog = useCallback(() => clearOperation(), [clearOperation]);

  const snapshotForPicker = useCallback((): ChatAudioUploadSnapshot | null => {
    if (disabled || busy || accountTransitionPending()) return null;
    const target = targetSettings();
    if (!target.model) {
      toast.error(t("settings.voice.dictation.audioUploadChooseModel"), {
        action: {
          label: t("settings.voice.dictation.sttOpenVoiceSettings"),
          onClick: () => useSettingsDialogStore.getState().openDialog("voice"),
        },
      });
      return null;
    }
    if (readiness.model !== target.model || readiness.state !== "ready") {
      if (readiness.model === target.model && readiness.state === "missing") {
        requestSttDownload(target.model);
      } else if (readiness.state === "error") {
        void refreshReadiness();
      }
      return null;
    }
    const generation = generationRef.current + 1;
    generationRef.current = generation;
    const snapshot: ChatAudioUploadSnapshot = {
      generation,
      owner: ownerRef.current,
      authSessionEpoch: getAuthSessionEpoch(),
      model: target.model,
      engine: target.engine,
      language: target.language,
      device: target.device,
      chatId: resolveDictationChatId(chatId),
    };
    pickerSnapshotRef.current = snapshot;
    return snapshot;
  }, [busy, chatId, disabled, readiness, refreshReadiness, t, targetSettings]);

  const pickerCancelled = useCallback(() => {
    pickerSnapshotRef.current = null;
  }, []);

  const runTranscription = useCallback(
    async (file: File, source: ChatAudioUploadSnapshot) => {
      if (disabled || controllerRef.current || accountTransitionPending())
        return;
      if (
        !chatAudioUploadFenceMatches(source, {
          generation: generationRef.current,
          owner: ownerRef.current,
          authSessionEpoch: getAuthSessionEpoch(),
        })
      ) {
        clearOperation();
        return;
      }
      const controller = new AbortController();
      controllerRef.current = controller;
      activeFenceRef.current = source;
      setDialogOpen(false);
      setBusy(true);
      let failure: string | null = null;
      try {
        const completion = await completeChatAudioUpload({
          started: source,
          current: () => ({
            generation: generationRef.current,
            owner: ownerRef.current,
            authSessionEpoch: getAuthSessionEpoch(),
          }),
          signal: controller.signal,
          blocked: accountTransitionPending,
          transcribe: () =>
            transcribeAudioBlob(file, {
              model: source.model,
              engine: source.engine,
              language: source.language,
              device: source.device,
              signal: controller.signal,
            }),
          commit: (transcript) => {
            const corrected = applyDictationDictionary(transcript).trim();
            if (!corrected) return;
            writeDraftRef.current(
              appendChatAudioTranscript(readDraftRef.current(), corrected),
            );
            recordRecentDictation(corrected, source.chatId);
          },
        });
        if (completion === "committed") {
          retainedRef.current = null;
          setRetained(null);
          requestAnimationFrame(() => focusDraftRef.current?.());
        } else if (completion === "empty") {
          failure = t("settings.voice.dictation.audioUploadNoSpeech");
        } else {
          clearOperation();
        }
      } catch (error) {
        if (controller.signal.aborted) return;
        if (error instanceof SttModelNotDownloadedError) {
          setReadiness({ state: "missing", model: source.model });
          requestSttDownload(source.model);
        }
        failure =
          error instanceof Error && error.message
            ? error.message
            : t("settings.voice.dictation.audioUploadFailed");
      } finally {
        if (activeFenceRef.current === source) {
          activeFenceRef.current = null;
          controllerRef.current = null;
          setBusy(false);
          if (failure) {
            const next = { file, snapshot: source, error: failure };
            retainedRef.current = next;
            setRetained(next);
            setDialogOpen(true);
          }
        }
      }
    },
    [clearOperation, disabled, t],
  );

  const selectFile = useCallback(
    (file: File) => {
      const snapshot = pickerSnapshotRef.current;
      pickerSnapshotRef.current = null;
      if (!snapshot) return;
      const validationError = chatAudioUploadFileError(file);
      if (validationError) {
        toast.error(
          validationError === "empty"
            ? t("settings.voice.dictation.audioUploadEmpty")
            : t("settings.voice.dictation.audioUploadTooLarge", {
                size: MAX_AUDIO_SIZE_LABEL,
              }),
        );
        return;
      }
      retainedRef.current = null;
      setRetained(null);
      void runTranscription(file, snapshot);
    },
    [runTranscription, t],
  );

  const retry = useCallback(() => {
    const failed = retainedRef.current;
    if (!failed || disabled || busy || accountTransitionPending()) return;
    if (
      failed.snapshot.owner !== ownerRef.current ||
      failed.snapshot.authSessionEpoch !== getAuthSessionEpoch()
    ) {
      clearOperation();
      return;
    }
    if (
      readiness.model !== failed.snapshot.model ||
      readiness.state !== "ready"
    ) {
      if (
        readiness.model === failed.snapshot.model &&
        readiness.state === "missing"
      ) {
        requestSttDownload(failed.snapshot.model);
      } else {
        void refreshReadiness();
      }
      return;
    }
    const generation = generationRef.current + 1;
    generationRef.current = generation;
    const snapshot = { ...failed.snapshot, generation };
    retainedRef.current = null;
    setRetained(null);
    void runTranscription(failed.file, snapshot);
  }, [
    busy,
    clearOperation,
    disabled,
    readiness,
    refreshReadiness,
    runTranscription,
  ]);

  const display = retained?.snapshot ?? {
    model: model.trim(),
    language,
    device,
  };

  return {
    model: display.model,
    language: display.language,
    device: display.device,
    busy,
    dialogOpen,
    readiness,
    failedFileName: retained?.file.name ?? null,
    failureMessage: retained?.error ?? null,
    openDialog,
    closeDialog,
    snapshotForPicker,
    pickerCancelled,
    selectFile,
    retry,
    refreshReadiness,
    cancel: clearOperation,
  };
}
