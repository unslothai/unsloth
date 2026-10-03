// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type SegmentRecorder,
  createAudioRecorder,
} from "@/features/chat/adapters/pcm-recorder";
import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { AudioApiError, fetchAudioBlob, uploadAudioInput } from "../api";
import {
  type AudioSourceSelection,
  selectionExpired,
  sourceFileUrl,
} from "../audio-run-request";
import { computePeaks } from "../components/waveform-peaks";
import {
  INITIAL_AUDIO_SOURCE_STATE,
  audioFileProblem,
  audioSourceReducer,
} from "./audio-source-state";

// Decoding holds every sample in memory; past this only the duration is read.
const DECODE_MAX_BYTES = 60 * 1024 * 1024;
// A reference only needs seconds; the server keeps the first 30 s.
const DEFAULT_RECORD_MAX_SECONDS = 5 * 60;
const RECORDING_CHUNK_MS = 1000;

export async function decodePeaks(
  blob: Blob,
): Promise<{ peaks: number[] | null; durationS: number | null }> {
  if (blob.size > DECODE_MAX_BYTES) {
    return { peaks: null, durationS: await mediaDuration(blob) };
  }
  const Offline =
    window.OfflineAudioContext ||
    (
      window as unknown as {
        webkitOfflineAudioContext?: typeof OfflineAudioContext;
      }
    ).webkitOfflineAudioContext;
  if (!Offline) return { peaks: null, durationS: await mediaDuration(blob) };
  try {
    const context = new Offline(1, 1, 22050);
    const buffer = await context.decodeAudioData(await blob.arrayBuffer());
    const channels = Array.from(
      { length: buffer.numberOfChannels },
      (_, index) => buffer.getChannelData(index),
    );
    return { peaks: computePeaks(channels), durationS: buffer.duration };
  } catch {
    // Some containers decode in a media element but not in Web Audio.
    return { peaks: null, durationS: await mediaDuration(blob) };
  }
}

function mediaDuration(blob: Blob): Promise<number | null> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(blob);
    const audio = new Audio();
    const done = (value: number | null) => {
      URL.revokeObjectURL(url);
      resolve(value);
    };
    audio.preload = "metadata";
    audio.onloadedmetadata = () =>
      done(Number.isFinite(audio.duration) ? audio.duration : null);
    audio.onerror = () => done(null);
    audio.src = url;
  });
}

function errorText(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

export function recordingSupported(): boolean {
  return (
    typeof navigator !== "undefined" &&
    typeof navigator.mediaDevices?.getUserMedia === "function"
  );
}

export function useAudioSource({
  value,
  onChange,
  maxRecordSeconds = DEFAULT_RECORD_MAX_SECONDS,
  active = true,
}: {
  value: AudioSourceSelection | null;
  onChange: (next: AudioSourceSelection | null) => void;
  maxRecordSeconds?: number;
  active?: boolean;
}) {
  const [state, dispatch] = useReducer(
    audioSourceReducer,
    INITIAL_AUDIO_SOURCE_STATE,
  );
  const [elapsed, setElapsed] = useState(0);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;
  const uploadAbort = useRef<AbortController | null>(null);
  const loadAbort = useRef<AbortController | null>(null);
  const recorderRef = useRef<SegmentRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const discardRecording = useRef(false);
  const activeRef = useRef(active);
  const acquiring = useRef(false);
  const acquisition = useRef(0);
  const objectUrl = useRef<string | null>(null);
  const setObjectUrl = useCallback((blob: Blob | null) => {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = blob ? URL.createObjectURL(blob) : null;
    return objectUrl.current;
  }, []);

  const drawBlob = useCallback(
    async (key: string, blob: Blob) => {
      const url = setObjectUrl(blob);
      dispatch({ type: "preview", key, peaks: null, durationS: null, url });
      const { peaks, durationS } = await decodePeaks(blob);
      dispatch({ type: "preview", key, peaks, durationS, url });
    },
    [setObjectUrl],
  );

  const valueKey = value ? `${value.kind}:${value.id}` : null;
  const previewKey = state.preview.key;
  const phase = state.status.phase;
  // A new pick replaces an error about an earlier one (a failed upload, then a history clip).
  const seenKey = useRef(valueKey);
  useEffect(() => {
    if (seenKey.current === valueKey) return;
    seenKey.current = valueKey;
    if (valueKey && phase === "error") dispatch({ type: "reset" });
  }, [valueKey, phase]);
  useEffect(() => {
    if (!(value && valueKey)) return;
    if (
      previewKey === valueKey ||
      phase === "uploading" ||
      phase === "recording" ||
      phase === "expired" ||
      phase === "error"
    )
      return;
    if (selectionExpired(value, Date.now())) {
      dispatch({ type: "expire" });
      return;
    }
    loadAbort.current?.abort();
    const controller = new AbortController();
    loadAbort.current = controller;
    // Not aborted by this effect's cleanup: load-start changes its deps. A new selection or
    // unmount aborts it instead.
    dispatch({ type: "load-start", key: valueKey });
    fetchAudioBlob(sourceFileUrl(value), controller.signal)
      .then(async (blob) => {
        if (controller.signal.aborted) return;
        dispatch({ type: "load-done" });
        await drawBlob(valueKey, blob);
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        if (error instanceof AudioApiError && error.status === 404) {
          if (value.kind === "input") {
            dispatch({ type: "expire" });
          } else {
            dispatch({
              type: "fail",
              message:
                value.kind === "voice"
                  ? "This saved voice was deleted. Pick another one."
                  : "This clip was deleted. Pick another one.",
            });
          }
          return;
        }
        // The audio could not be drawn, but the server still has it; the run can go ahead.
        dispatch({ type: "load-done" });
      });
  }, [value, valueKey, previewKey, phase, drawBlob]);
  useEffect(() => () => loadAbort.current?.abort(), [valueKey]);

  useEffect(() => {
    if (
      value ||
      phase === "recording" ||
      phase === "error" ||
      phase === "uploading"
    )
      return;
    if (phase !== "idle" || previewKey !== null) {
      setObjectUrl(null);
      dispatch({ type: "reset" });
    }
  }, [value, phase, previewKey, setObjectUrl]);

  const pickFile = useCallback(
    async (file: File | Blob, name?: string) => {
      const fileName =
        name ?? ("name" in file && file.name ? file.name : "Recording");
      const problem = audioFileProblem({
        size: file.size,
        type: file.type,
        name: fileName,
      });
      if (problem) {
        dispatch({ type: "fail", message: problem });
        return;
      }
      uploadAbort.current?.abort();
      loadAbort.current?.abort();
      const controller = new AbortController();
      uploadAbort.current = controller;
      dispatch({ type: "upload-start", name: fileName });
      void drawBlob("local", file);
      try {
        const record = await uploadAudioInput(file, fileName, {
          signal: controller.signal,
          onProgress: (progress) =>
            dispatch({ type: "upload-progress", progress }),
        });
        if (controller.signal.aborted) return;
        dispatch({ type: "upload-done", key: `input:${record.id}` });
        onChangeRef.current({
          kind: "input",
          id: record.id,
          name: record.name || fileName,
          durationS: record.duration_s,
          expiresAt: record.expires_at,
          transcript: null,
          language: null,
        });
      } catch (error) {
        if (controller.signal.aborted) return;
        dispatch({
          type: "fail",
          message: errorText(error, "The upload failed. Try another file."),
        });
      } finally {
        if (uploadAbort.current === controller) uploadAbort.current = null;
      }
    },
    [drawBlob],
  );

  const stopStream = useCallback(() => {
    for (const track of streamRef.current?.getTracks() ?? []) track.stop();
    streamRef.current = null;
  }, []);

  const startRecording = useCallback(async () => {
    if (recorderRef.current || acquiring.current) return;
    acquiring.current = true;
    const ticket = ++acquisition.current;
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      });
    } catch {
      acquiring.current = false;
      if (ticket !== acquisition.current) return;
      dispatch({
        type: "fail",
        message:
          "Could not use the microphone. Allow access, or upload a file instead.",
      });
      return;
    }
    acquiring.current = false;
    // Cleared, unmounted or hidden while the permission prompt was open: release the mic at once.
    if (ticket !== acquisition.current || !activeRef.current) {
      for (const track of stream.getTracks()) track.stop();
      return;
    }
    streamRef.current = stream;
    let recorder: SegmentRecorder;
    try {
      recorder = createAudioRecorder(stream);
    } catch {
      stopStream();
      dispatch({
        type: "fail",
        message:
          "This browser cannot record audio here. Upload a file instead.",
      });
      return;
    }
    const chunks: Blob[] = [];
    const limit = window.setTimeout(() => {
      if (recorder.state !== "inactive") recorder.stop();
    }, maxRecordSeconds * 1000);
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    });
    recorder.addEventListener("stop", () => {
      window.clearTimeout(limit);
      recorderRef.current = null;
      stopStream();
      dispatch({ type: "record-stop" });
      if (discardRecording.current) {
        discardRecording.current = false;
        return;
      }
      const type = recorder.mimeType || "audio/webm";
      const blob = new Blob(chunks, { type });
      const extension = type.includes("wav")
        ? "wav"
        : type.includes("mp4")
          ? "m4a"
          : type.includes("ogg")
            ? "ogg"
            : "webm";
      if (blob.size > 0) void pickFile(blob, `Recording.${extension}`);
    });
    recorderRef.current = recorder;
    discardRecording.current = false;
    recorder.start(RECORDING_CHUNK_MS);
    dispatch({ type: "record-start", now: Date.now() });
  }, [pickFile, stopStream, maxRecordSeconds]);

  const stopRecording = useCallback(() => {
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }, []);

  // Leaving the Audio page ends a recording; what was captured is kept.
  useEffect(() => {
    activeRef.current = active;
    if (!active) stopRecording();
  }, [active, stopRecording]);

  const recordingStartedAt =
    state.status.phase === "recording" ? state.status.startedAt : null;
  useEffect(() => {
    if (recordingStartedAt === null) {
      setElapsed(0);
      return;
    }
    const timer = window.setInterval(
      () => setElapsed(Math.floor((Date.now() - recordingStartedAt) / 1000)),
      250,
    );
    return () => window.clearInterval(timer);
  }, [recordingStartedAt]);

  const abortAll = useCallback(() => {
    acquisition.current += 1;
    uploadAbort.current?.abort();
    loadAbort.current?.abort();
    if (recorderRef.current) {
      discardRecording.current = true;
      recorderRef.current.stop();
    }
  }, []);

  const dismissError = useCallback(() => {
    setObjectUrl(null);
    dispatch({ type: "reset" });
  }, [setObjectUrl]);

  const clear = useCallback(() => {
    abortAll();
    dismissError();
    onChangeRef.current(null);
  }, [abortAll, dismissError]);

  useEffect(
    () => () => {
      abortAll();
      for (const track of streamRef.current?.getTracks() ?? []) track.stop();
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    },
    [abortAll],
  );

  const fail = useCallback(
    (message: string) => dispatch({ type: "fail", message }),
    [],
  );

  const expire = useCallback(() => dispatch({ type: "expire" }), []);

  return {
    status: state.status,
    preview: state.preview,
    elapsed,
    expire,
    pickFile,
    startRecording,
    stopRecording,
    clear,
    dismissError,
    fail,
  };
}
