// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AUTH_SESSION_CLEARED_EVENT,
  AUTH_SESSION_STORED_EVENT,
  getAuthSessionEpoch,
} from "@/features/auth";
import {
  SttModelNotDownloadedError,
  sttEngineFor,
  transcribeAudioBlob,
} from "../adapters/studio-model-dictation-adapter";
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
  applyDictationDictionary,
  recordRecentDictation,
  requestSttDownload,
  useSettingsDialogStore,
  useVoiceSettingsStore,
} from "@/features/settings";
import {
  appendChatAudioTranscript,
  chatAudioUploadFileError,
  completeChatAudioUpload,
  type ChatAudioUploadFence,
} from "../utils/chat-audio-upload";

export interface UseChatAudioUploadOptions {
  owner: string;
  chatId?: string | null;
  disabled?: boolean;
  readDraft: () => string;
  writeDraft: (value: string) => void;
}

export function useChatAudioUpload({
  owner,
  chatId,
  disabled = false,
  readDraft,
  writeDraft,
}: UseChatAudioUploadOptions) {
  const model = useVoiceSettingsStore((state) => state.sttModel);
  const language = useVoiceSettingsStore((state) => state.dictationLanguage);
  const device = useVoiceSettingsStore((state) => state.sttDevice);
  const [busy, setBusy] = useState(false);
  const generationRef = useRef(0);
  const ownerRef = useRef(owner);
  const controllerRef = useRef<AbortController | null>(null);
  const activeFenceRef = useRef<ChatAudioUploadFence | null>(null);
  const readDraftRef = useRef(readDraft);
  const writeDraftRef = useRef(writeDraft);

  useLayoutEffect(() => {
    ownerRef.current = owner;
    readDraftRef.current = readDraft;
    writeDraftRef.current = writeDraft;
  }, [owner, readDraft, writeDraft]);

  const invalidate = useCallback(() => {
    generationRef.current += 1;
    activeFenceRef.current = null;
    controllerRef.current?.abort();
    controllerRef.current = null;
  }, []);

  const cancel = useCallback(() => {
    invalidate();
    setBusy(false);
  }, [invalidate]);

  useEffect(() => {
    const fence = activeFenceRef.current;
    if (fence && fence.owner !== owner) cancel();
  }, [cancel, owner]);

  useEffect(() => {
    if (disabled) {
      invalidate();
      const invalidatedGeneration = generationRef.current;
      queueMicrotask(() => {
        if (generationRef.current === invalidatedGeneration) {
          setBusy(false);
        }
      });
    }
  }, [disabled, invalidate]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.addEventListener(AUTH_SESSION_CLEARED_EVENT, cancel);
    window.addEventListener(AUTH_SESSION_STORED_EVENT, cancel);
    return () => {
      window.removeEventListener(AUTH_SESSION_CLEARED_EVENT, cancel);
      window.removeEventListener(AUTH_SESSION_STORED_EVENT, cancel);
    };
  }, [cancel]);

  useEffect(() => cancel, [cancel]);

  const selectFile = useCallback(
    async (file: File) => {
      if (disabled || controllerRef.current) return;
      const validationError = chatAudioUploadFileError(file);
      if (validationError) {
        toast.error(validationError);
        return;
      }

      const sessionModel = model.trim();
      if (!sessionModel) {
        toast.error("Choose a local transcription model in Voice settings.", {
          action: {
            label: "Open Voice settings",
            onClick: () => useSettingsDialogStore.getState().openDialog("voice"),
          },
        });
        return;
      }

      const generation = generationRef.current + 1;
      generationRef.current = generation;
      const fence: ChatAudioUploadFence = {
        generation,
        owner,
        authSessionEpoch: getAuthSessionEpoch(),
      };
      activeFenceRef.current = fence;
      const controller = new AbortController();
      controllerRef.current = controller;
      const sessionEngine = sttEngineFor(sessionModel);
      const sessionLanguage = language;
      const sessionDevice = device;
      const sessionChatId = chatId || undefined;
      setBusy(true);

      try {
        const completion = await completeChatAudioUpload({
          started: fence,
          current: () => ({
            generation: generationRef.current,
            owner: ownerRef.current,
            authSessionEpoch: getAuthSessionEpoch(),
          }),
          signal: controller.signal,
          blocked: accountTransitionPending,
          transcribe: () =>
            transcribeAudioBlob(file, {
              model: sessionModel,
              engine: sessionEngine,
              language: sessionLanguage,
              device: sessionDevice,
              signal: controller.signal,
            }),
          commit: (transcript) => {
            const corrected = applyDictationDictionary(transcript).trim();
            if (!corrected) return;
            writeDraftRef.current(
              appendChatAudioTranscript(readDraftRef.current(), corrected),
            );
            recordRecentDictation(corrected, sessionChatId);
          },
        });
        if (completion === "empty") {
          toast.info("The model heard no speech in that audio.");
        }
      } catch (error) {
        if (controller.signal.aborted) return;
        if (error instanceof SttModelNotDownloadedError) {
          requestSttDownload(sessionModel);
          return;
        }
        const message =
          error instanceof Error && error.message
            ? error.message
            : "The audio file could not be transcribed.";
        toast.error(message, {
          action: {
            label: "Open Voice settings",
            onClick: () => useSettingsDialogStore.getState().openDialog("voice"),
          },
        });
      } finally {
        if (activeFenceRef.current === fence) {
          activeFenceRef.current = null;
          controllerRef.current = null;
          setBusy(false);
        }
      }
    }, [chatId, device, disabled, language, model, owner],
  );

  return { model, language, busy, selectFile, cancel };
}
