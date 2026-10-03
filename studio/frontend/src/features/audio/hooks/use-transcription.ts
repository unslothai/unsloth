// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AUTH_SESSION_ENDING_EVENT } from "@/features/auth";
import {
  createAudioRecorder,
  PcmRecorder,
  type SegmentRecorder,
} from "@/features/chat/adapters/pcm-recorder";
import {
  StudioModelDictationAdapter,
} from "@/features/chat/adapters/studio-model-dictation-adapter";
import { useVoiceSettingsStore } from "@/features/settings";
import { isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { transcribeWithProgress } from "../api";
import { micStreamRequestIsCurrent } from "../audio-page-policy";
import {
  RECORDING_CHUNK_MS,
  RECORDING_MAX_BYTES,
  RECORDING_MAX_SECONDS,
} from "../audio-workspace-constants";
import { sttEngineForRepoId, sttSidecarKeyFor } from "../catalog";
import { downloadTranscript } from "../transcript-download";
import {
  readTranscriptDraft,
  transcriptDraftKey,
  writeTranscriptDraft,
} from "../transcript-draft";
import type {
  TranscriptProgress,
  TranscriptRecord,
} from "../transcript-stream";
import type { AudioHostState } from "./audio-host-state";
import type { SttSidecar } from "./use-stt-sidecar";

export function useTranscription({
  active,
  activeRef,
  busyRef,
  setBusy,
  lastSttRepo,
  selectedSttRepoRef,
  setSelectedSttRepo,
  sttLoadedModel,
  sttLoadedEngine,
  ensureSttLoaded,
  setLastSttRepo,
  refreshSttStatus,
}: Pick<AudioHostState, "active" | "activeRef" | "busyRef" | "setBusy"> &
  Pick<
    SttSidecar,
    | "lastSttRepo"
    | "selectedSttRepoRef"
    | "setSelectedSttRepo"
    | "sttLoadedModel"
    | "sttLoadedEngine"
    | "ensureSttLoaded"
    | "setLastSttRepo"
    | "refreshSttStatus"
  >) {
  const [draftKey] = useState(transcriptDraftKey);
  const [recoveredTranscript] = useState(() => readTranscriptDraft(draftKey));
  const [transcript, setTranscript] = useState(recoveredTranscript?.text ?? "");
  const [transcribedName, setTranscribedName] = useState<string | null>(recoveredTranscript?.title ?? null);
  const [transcriptModel, setTranscriptModel] = useState(recoveredTranscript?.model ?? "");
  const [transcriptRecord, setTranscriptRecord] = useState<TranscriptRecord | null>(null);
  const [transcriptExported, setTranscriptExported] = useState(false);
  const transcriptVersion = useRef(0);
  const [transcriptionStartedAt, setTranscriptionStartedAt] = useState<number | null>(null);
  const [transcriptionFinishedAt, setTranscriptionFinishedAt] = useState<number | null>(null);
  const [transcriptionStopping, setTranscriptionStopping] = useState(false);
  const [transcriptionProgress, setTranscriptionProgress] = useState<TranscriptProgress | null>(null);
  const [transcriptError, setTranscriptError] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [micRequestPending, setMicRequestPending] = useState(false);
  /** WebKit lacks MediaRecorder, and http LAN origins have no navigator.mediaDevices. */
  const recordingSupported = useMemo(
    () => StudioModelDictationAdapter.isSupported(),
    [],
  );
  const recorderRef = useRef<SegmentRecorder | null>(null);
  const recordStreamRef = useRef<MediaStream | null>(null);
  const discardRecordingRef = useRef(false);
  const micRequestGeneration = useRef(0);
  const micPendingGeneration = useRef<number | null>(null);
  const transcriptionAbort = useRef<AbortController | null>(null);

  const stopRecordStream = useCallback(() => {
    for (const track of recordStreamRef.current?.getTracks() ?? [])
      track.stop();
    recordStreamRef.current = null;
  }, []);
  const stopAndDiscardRecording = useCallback(() => {
    micRequestGeneration.current += 1;
    micPendingGeneration.current = null;
    setMicRequestPending(false);
    const recorder = recorderRef.current;
    if (recorder) {
      discardRecordingRef.current = true;
      if (recorder.state !== "inactive") recorder.stop();
      recorderRef.current = null;
      setIsRecording(false);
    }
    stopRecordStream();
  }, [stopRecordStream]);

  const clearTranscript = useCallback(() => {
    transcriptVersion.current += 1;
    setTranscript("");
    setTranscribedName(null);
    setTranscriptError(null);
    setTranscriptModel("");
    setTranscriptRecord(null);
    setTranscriptExported(false);
  }, []);

  const confirmTranscriptReplacement = useCallback(() => {
    if (busyRef.current !== null) return false;
    return (
      !transcript ||
      transcriptRecord !== null ||
      transcriptExported ||
      window.confirm(
        "This transcript could not be saved. Download it before continuing, or continue and discard it?",
      )
    );
  }, [transcript, transcriptRecord, transcriptExported]);

  useEffect(() => {
    const unsaved = Boolean(
      transcript && transcriptRecord === null && !transcriptExported,
    );
    if (!writeTranscriptDraft(
      draftKey,
      unsaved ? {
        text: transcript,
        title: transcribedName ?? "Transcript",
        model: transcriptModel,
      } : null,
    )) {
      toast.error("Could not update transcript recovery. Download unsaved text before leaving.");
    }
    if (isTauri) {
      void import("@tauri-apps/api/core")
        .then(({ invoke }) =>
          invoke("set_renderer_activity", {
            kind: "unsaved_transcript",
            active: unsaved,
          }),
        )
        .catch(() =>
          toast.error("Could not update transcript close protection."),
        );
    }
    if (!unsaved) return;
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    const confirmLogout = (event: Event) => {
      if (
        !window.confirm(
          "This transcript could not be saved. Download it before logging out, or log out and discard it?",
        )
      ) {
        event.preventDefault();
      } else if (!writeTranscriptDraft(draftKey, null)) {
        event.preventDefault();
        toast.error("Could not discard the transcript recovery copy. Try again.");
      }
    };
    window.addEventListener("beforeunload", warn);
    window.addEventListener(AUTH_SESSION_ENDING_EVENT, confirmLogout);
    return () => {
      window.removeEventListener("beforeunload", warn);
      window.removeEventListener(AUTH_SESSION_ENDING_EVENT, confirmLogout);
    };
  }, [draftKey, transcript, transcribedName, transcriptModel, transcriptRecord, transcriptExported]);

  const prepareTranscriptionModel = useCallback(async () => {
    const repo = selectedSttRepoRef.current ?? lastSttRepo;
    if (!repo) {
      toast.info("Pick a speech-to-text model first.");
      return null;
    }
    selectedSttRepoRef.current = repo;
    setSelectedSttRepo(repo);
    const model = sttSidecarKeyFor(repo);
    const engine = sttEngineForRepoId(repo);
    if (sttLoadedModel !== model || sttLoadedEngine !== engine) {
      if (!(await ensureSttLoaded(repo, model, engine))) return null;
    }
    setLastSttRepo(repo);
    return { model, engine };
  }, [
    lastSttRepo,
    sttLoadedModel,
    sttLoadedEngine,
    ensureSttLoaded,
    setLastSttRepo,
  ]);

  const runTranscription = useCallback(
    async (blob: Blob, name: string, confirmedVersion?: number) => {
      if (
        transcriptionAbort.current ||
        busyRef.current !== null ||
        (confirmedVersion !== transcriptVersion.current &&
          !confirmTranscriptReplacement())
      ) return;
      let started = false;
      const controller = new AbortController();
      transcriptionAbort.current = controller;
      try {
        const target = await prepareTranscriptionModel();
        if (!target || controller.signal.aborted || !activeRef.current) return;
        setBusy("transcribing");
        clearTranscript();
        setTranscribedName(name);
        setTranscriptModel(target.model);
        started = true;
        setTranscriptionStartedAt(Date.now());
        setTranscriptionFinishedAt(null);
        setTranscriptionStopping(false);
        setTranscriptionProgress(null);
        const result = await transcribeWithProgress(
          blob,
          name,
          {
            ...target,
            device: useVoiceSettingsStore.getState().sttDevice,
            signal: controller.signal,
          },
          (progress) => {
            if (!controller.signal.aborted) setTranscriptionProgress(progress);
          },
        );
        if (controller.signal.aborted) return;
        setTranscript(result.text);
        setTranscriptModel(result.model);
        setTranscriptRecord(result.record);
        if (!result.text)
          toast.info("The model heard no speech in that audio.");
        else if (!result.record)
          toast.error(
            "Transcript could not be saved. Download a copy to keep it.",
          );
      } catch (error) {
        if (controller.signal.aborted) {
          setTranscriptError("Transcription cancelled.");
          return;
        }
        const message =
          error instanceof Error ? error.message : "Transcription failed.";
        setTranscriptError(message);
        toast.error(message);
      } finally {
        if (transcriptionAbort.current === controller) {
          transcriptionAbort.current = null;
          if (started) setTranscriptionFinishedAt(Date.now());
          setBusy(null);
          if (activeRef.current) void refreshSttStatus();
        }
      }
    },
    [
      clearTranscript,
      confirmTranscriptReplacement,
      prepareTranscriptionModel,
      refreshSttStatus,
    ],
  );

  const handleRecordToggle = useCallback(async () => {
    if (isRecording) {
      const recorder = recorderRef.current;
      if (recorder && recorder.state !== "inactive") recorder.stop();
      return;
    }
    if (micPendingGeneration.current !== null || busyRef.current !== null) return;
    if (!confirmTranscriptReplacement()) return;
    const confirmedVersion = transcriptVersion.current;
    const requestGeneration = ++micRequestGeneration.current;
    micPendingGeneration.current = requestGeneration;
    setMicRequestPending(true);
    try {
      if (
        !(await prepareTranscriptionModel()) ||
        !activeRef.current ||
        micRequestGeneration.current !== requestGeneration
      ) return;
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      });
      if (
        !micStreamRequestIsCurrent(
          requestGeneration,
          micRequestGeneration.current,
          activeRef.current,
        )
      ) {
        for (const track of stream.getTracks()) track.stop();
        return;
      }
      recordStreamRef.current = stream;
      const recorder = createAudioRecorder(stream);
      const maxSeconds =
        recorder instanceof PcmRecorder
          ? Math.min(
              RECORDING_MAX_SECONDS,
              recorder.secondsWithin(RECORDING_MAX_BYTES),
            )
          : RECORDING_MAX_SECONDS;
      const chunks: Blob[] = [];
      let recordedBytes = 0;
      let limitHit: "duration" | "size" | null = null;
      const stopAtLimit = (reason: "duration" | "size") => {
        if (limitHit) return;
        limitHit = reason;
        toast.warning(
          reason === "duration"
            ? `Recording stopped at the ${Math.floor(maxSeconds / 60)} minute limit.`
            : "Recording stopped: it reached the maximum upload size.",
        );
        try {
          recorder.stop();
        } catch {
        }
      };
      const durationTimer = window.setTimeout(
        () => stopAtLimit("duration"),
        maxSeconds * 1000,
      );
      recorder.addEventListener("dataavailable", (event) => {
        if (event.data.size > 0) {
          if (recordedBytes + event.data.size > RECORDING_MAX_BYTES) {
            stopAtLimit("size");
            return;
          }
          chunks.push(event.data);
          recordedBytes += event.data.size;
        }
      });
      recorder.addEventListener("stop", () => {
        window.clearTimeout(durationTimer);
        const discard = discardRecordingRef.current;
        discardRecordingRef.current = false;
        setIsRecording(false);
        stopRecordStream();
        recorderRef.current = null;
        const blob = new Blob(chunks, {
          type: recorder.mimeType || "audio/webm",
        });
        if (!discard && blob.size > 0)
          void runTranscription(blob, "Recording", confirmedVersion);
      });
      recorderRef.current = recorder;
      // A timeslice makes the byte cap observable; without one some browsers emit only on stop.
      recorder.start(RECORDING_CHUNK_MS);
      setIsRecording(true);
    } catch {
      // Release the stream if MediaRecorder construction failed, or the mic stays live.
      recorderRef.current = null;
      setIsRecording(false);
      stopRecordStream();
      if (
        micStreamRequestIsCurrent(
          requestGeneration,
          micRequestGeneration.current,
          activeRef.current,
        )
      )
        toast.error("Could not access the microphone.");
    } finally {
      if (micPendingGeneration.current === requestGeneration) {
        micPendingGeneration.current = null;
        setMicRequestPending(false);
      }
    }
  }, [
    isRecording,
    runTranscription,
    stopRecordStream,
    prepareTranscriptionModel,
    confirmTranscriptReplacement,
  ]);

  // Release the microphone on unmount AND whenever the page goes inactive: the page stays mounted
  // across tab switches, so unmount alone left a hidden recorder capturing.
  useEffect(() => {
    if (!active) {
      stopAndDiscardRecording();
    }
    return stopAndDiscardRecording;
  }, [active, stopAndDiscardRecording]);

  useEffect(() => () => transcriptionAbort.current?.abort(), []);

  const handleTranscribeFile = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      void runTranscription(file, file.name);
    },
    [runTranscription],
  );

  const handleCopyTranscript = useCallback(() => {
    void copyToClipboard(transcript).then((ok) =>
      ok
        ? toast.success("Transcript copied")
        : toast.error("Could not copy the transcript."),
    );
  }, [transcript]);

  const handleDownloadTranscript = useCallback(async () => {
    const version = transcriptVersion.current;
    if (
      (await downloadTranscript(transcript, transcribedName ?? "transcript")) &&
      transcriptVersion.current === version
    )
      setTranscriptExported(true);
  }, [transcript, transcribedName]);

  return {
    transcript,
    setTranscript,
    transcribedName,
    setTranscribedName,
    transcriptModel,
    setTranscriptModel,
    transcriptRecord,
    setTranscriptRecord,
    transcriptExported,
    setTranscriptExported,
    transcriptVersion,
    transcriptionStartedAt,
    setTranscriptionStartedAt,
    transcriptionFinishedAt,
    transcriptionStopping,
    setTranscriptionStopping,
    transcriptionProgress,
    transcriptError,
    setTranscriptError,
    isRecording,
    micRequestPending,
    recordingSupported,
    transcriptionAbort,
    stopAndDiscardRecording,
    clearTranscript,
    confirmTranscriptReplacement,
    handleRecordToggle,
    handleTranscribeFile,
    handleCopyTranscript,
    handleDownloadTranscript,
  };
}

export type Transcription = ReturnType<typeof useTranscription>;
