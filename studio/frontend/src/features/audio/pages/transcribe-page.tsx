// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Download01Icon,
  Mic01Icon,
  StopIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Field } from "../components/field";
import { TranscriptGallery } from "../transcript-gallery";
import { TranscriptionProgress } from "../transcription-progress";
import type { AudioHostState } from "../hooks/audio-host-state";
import type { SttSidecar } from "../hooks/use-stt-sidecar";
import type { Transcription } from "../hooks/use-transcription";

/** Transcribe's rail: record from the microphone or pick a file. */
export function TranscribeRail({
  recordingSupported,
  isRecording,
  sttSelected,
  lastSttRepo,
  busy,
  micRequestPending,
  handleRecordToggle,
  handleTranscribeFile,
}: Pick<
  Transcription,
  | "recordingSupported"
  | "isRecording"
  | "micRequestPending"
  | "handleRecordToggle"
  | "handleTranscribeFile"
> &
  Pick<SttSidecar, "sttSelected" | "lastSttRepo"> &
  Pick<AudioHostState, "busy">) {
  return (
    <>
      <Field
        label="Microphone"
        htmlFor="audio-record"
        hint={
          recordingSupported
            ? "Record a clip and it is transcribed when you stop."
            : "This browser cannot record. Open Unsloth over https or on localhost, or upload a file below."
        }
      >
        <Button
          data-tour="audio-record"
          id="audio-record"
          variant={isRecording ? "destructive" : "secondary"}
          disabled={
            !recordingSupported ||
            (!isRecording && (!(sttSelected || lastSttRepo) || busy !== null)) ||
            micRequestPending
          }
          onClick={handleRecordToggle}
        >
          <HugeiconsIcon
            icon={isRecording ? StopIcon : Mic01Icon}
            className="mr-2 size-4"
          />
          {isRecording
            ? "Stop recording"
            : micRequestPending
              ? busy === "loading" ? "Loading model…" : "Waiting for microphone…"
              : "Record"}
        </Button>
      </Field>
      <Field
        label="Audio file"
        htmlFor="audio-file"
        hint="Or transcribe an existing recording (wav, mp3, m4a, webm…)."
      >
        <input
          id="audio-file"
          type="file"
          accept="audio/*"
          disabled={
            !(sttSelected || lastSttRepo) ||
            busy !== null ||
            isRecording ||
            micRequestPending
          }
          onChange={(event) => {
            handleTranscribeFile(event.target.files?.[0]);
            event.target.value = "";
          }}
          className="text-ui-13 file:mr-3 file:rounded-md file:border-0 file:bg-muted file:px-3 file:py-1.5 file:text-ui-13 file:font-medium"
        />
      </Field>
      {sttSelected || lastSttRepo ? null : (
        <p className="text-ui-11p5 leading-snug text-muted-foreground">
          Pick a speech-to-text model (Whisper or Qwen3-ASR) from the
          selector above to transcribe.
        </p>
      )}
    </>
  );
}

/** Transcribe's output: progress, the transcript and its actions, then saved transcripts. */
export function TranscribeOutput({
  transcriptionStartedAt,
  transcriptionFinishedAt,
  transcriptionStopping,
  transcriptionProgress,
  setTranscriptionStopping,
  transcriptionAbort,
  transcript,
  handleCopyTranscript,
  handleDownloadTranscript,
  transcribedName,
  transcriptModel,
  transcriptRecord,
  transcriptExported,
  transcriptError,
  busy,
  active,
  mode,
  confirmTranscriptReplacement,
  transcriptVersion,
  setTranscript,
  setTranscribedName,
  setTranscriptModel,
  setTranscriptRecord,
  setTranscriptError,
  setTranscriptExported,
  setTranscriptionStartedAt,
  clearTranscript,
}: Pick<
  Transcription,
  | "transcriptionStartedAt"
  | "transcriptionFinishedAt"
  | "transcriptionStopping"
  | "transcriptionProgress"
  | "setTranscriptionStopping"
  | "transcriptionAbort"
  | "transcript"
  | "handleCopyTranscript"
  | "handleDownloadTranscript"
  | "transcribedName"
  | "transcriptModel"
  | "transcriptRecord"
  | "transcriptExported"
  | "transcriptError"
  | "confirmTranscriptReplacement"
  | "transcriptVersion"
  | "setTranscript"
  | "setTranscribedName"
  | "setTranscriptModel"
  | "setTranscriptRecord"
  | "setTranscriptError"
  | "setTranscriptExported"
  | "setTranscriptionStartedAt"
  | "clearTranscript"
> &
  Pick<AudioHostState, "busy" | "active" | "mode">) {
  return (
    <>
      <div className="hover-scrollbar flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto">
        {transcriptionStartedAt !== null && (
          <TranscriptionProgress
            startedAt={transcriptionStartedAt}
            finishedAt={transcriptionFinishedAt}
            stopping={transcriptionStopping}
            progress={transcriptionProgress}
            onCancel={() => {
              setTranscriptionStopping(true);
              transcriptionAbort.current?.abort();
            }}
          />
        )}
        {transcript ? (
          <>
            <div className="flex items-center gap-2">
              <Button
                variant="secondary"
                size="sm"
                onClick={handleCopyTranscript}
              >
                Copy
              </Button>
              <Button
                variant="secondary"
                size="sm"
                onClick={handleDownloadTranscript}
              >
                <HugeiconsIcon icon={Download01Icon} className="mr-2 size-3.5" />
                Download .txt
              </Button>
            </div>
            <div className="flex items-center gap-2 text-ui-11p5 text-muted-foreground">
              <span className="truncate">{transcribedName}</span>
              <span>·</span>
              <span className="truncate">{transcriptModel}</span>
              {!transcriptRecord && !transcriptExported && (
                <span>· Not saved</span>
              )}
            </div>
            <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground">
              {transcript}
            </p>
          </>
        ) : transcriptError ? (
          <div className="flex flex-col gap-1" role="alert">
            <p className="text-ui-13 font-medium text-destructive">
              Could not transcribe {transcribedName ?? "that audio"}.
            </p>
            <p className="text-ui-13 text-muted-foreground">{transcriptError}</p>
          </div>
        ) : busy !== "transcribing" ? (
          <p className="text-ui-13 text-muted-foreground">
            Record or upload audio to transcribe. Completed transcripts are saved
            to history.
          </p>
        ) : null}
      </div>
      <div className="shrink-0 pt-4">
        <TranscriptGallery
          autoSelect={!transcript && !transcribedName && busy === null}
          active={active && mode === "transcribe"}
          currentId={transcriptRecord?.id ?? null}
          latest={transcriptRecord}
          canSelect={confirmTranscriptReplacement}
          onSelect={(record) => {
            transcriptVersion.current += 1;
            setTranscript(record.text);
            setTranscribedName(record.title);
            setTranscriptModel(record.model);
            setTranscriptRecord(record);
            setTranscriptError(null);
            setTranscriptExported(false);
            setTranscriptionStartedAt(null);
          }}
          onDelete={(ids) => {
            if (
              transcriptRecord &&
              (ids === null
                ? !transcriptRecord.archived
                : ids.includes(transcriptRecord.id))
            )
              clearTranscript();
          }}
        />
      </div>
    </>
  );
}
