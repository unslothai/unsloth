// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";
import { Copy01Icon, Download01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useRef, useState } from "react";
import { TranscriptPlayer } from "../components/transcript-player";
import { TranscriptView } from "../components/transcript-view";
import type { WaveformControl } from "../components/waveform";
import type { AudioHostState } from "../hooks/audio-host-state";
import type { Transcription } from "../hooks/use-transcription";
import { useAudioTranscribeStore } from "../stores/audio-transcribe-store";
import { downloadTranscriptFile } from "../transcript-download";
import {
  TRANSCRIPT_EXPORT_FORMATS,
  type TranscriptExportFormat,
  exportTranscript,
  formatNeedsTimestamps,
} from "../transcript-export";
import { TranscriptGallery } from "../transcript-gallery";
import {
  type TranscriptDetails,
  formatTimestamp,
  hasTimestamps,
} from "../transcript-model";
import type { TranscriptRecord } from "../transcript-stream";
import { TranscriptionProgress } from "../transcription-progress";

interface TranscribeOutputProps
  extends Pick<
      Transcription,
      | "transcriptionStartedAt"
      | "transcriptionFinishedAt"
      | "transcriptionStopping"
      | "transcriptionProgress"
      | "setTranscriptionStopping"
      | "transcriptionAbort"
      | "transcript"
      | "handleCopyTranscript"
      | "transcribedName"
      | "transcriptModel"
      | "transcriptRecord"
      | "transcriptExported"
      | "transcriptError"
      | "confirmTranscriptReplacement"
      | "clearTranscript"
    >,
    Pick<AudioHostState, "busy" | "active" | "mode"> {
  transcriptVersion: { current: number };
  markExported: (version: number) => void;
  transcriptDetails: TranscriptDetails;
  speakerNames: Record<string, string>;
  renameSpeaker: (id: string, name: string) => void;
  selectRecord: (record: TranscriptRecord) => void;
}

const FORMAT_LABELS: Record<TranscriptExportFormat, string> = {
  txt: "Text (.txt)",
  srt: "Subtitles (.srt)",
  vtt: "Web subtitles (.vtt)",
  json: "JSON with timings (.json)",
};

export function TranscribeOutput({
  transcriptionStartedAt,
  transcriptionFinishedAt,
  transcriptionStopping,
  transcriptionProgress,
  setTranscriptionStopping,
  transcriptionAbort,
  transcript,
  handleCopyTranscript,
  transcribedName,
  transcriptModel,
  transcriptRecord,
  transcriptExported,
  transcriptError,
  busy,
  active,
  mode,
  confirmTranscriptReplacement,
  clearTranscript,
  transcriptVersion,
  markExported,
  transcriptDetails,
  speakerNames,
  renameSpeaker,
  selectRecord,
}: TranscribeOutputProps) {
  const view = useAudioTranscribeStore((state) => state.view);
  const setView = useAudioTranscribeStore((state) => state.setView);
  const player = useRef<WaveformControl | null>(null);
  const [position, setPosition] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [audioAvailable, setAudioAvailable] = useState(false);
  const handlePosition = useCallback((seconds: number, isPlaying: boolean) => {
    setPosition(seconds);
    setPlaying(isPlaying);
  }, []);
  const timed = hasTimestamps(transcriptDetails);
  const language = transcriptDetails.language ?? transcriptRecord?.language;
  const duration = transcriptDetails.duration ?? transcriptRecord?.duration;

  const handleDownload = async (format: TranscriptExportFormat) => {
    const version = transcriptVersion.current;
    const title = transcribedName ?? "transcript";
    const file = exportTranscript(format, {
      title,
      text: transcript,
      model: transcriptModel,
      details: transcriptDetails,
      names: speakerNames,
    });
    if (await downloadTranscriptFile(file.content, title, file.ext, file.mime))
      markExported(version);
  };

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
          // Focused and announced by the host after a run.
          <section
            id="transcribe-result"
            tabIndex={-1}
            aria-label="Transcript"
            aria-busy={busy === "loading" ? true : undefined}
            className={cn(
              "flex flex-col gap-3 transition-opacity duration-150 focus-visible:outline-none motion-reduce:transition-none",
              busy === "loading" && "opacity-50",
            )}
          >
            <TranscriptPlayer
              source={transcriptDetails.source}
              durationS={duration ?? null}
              controlRef={player}
              onPositionChange={handlePosition}
              onAvailableChange={setAudioAvailable}
            />
            <div className="flex flex-wrap items-center gap-2">
              <Button
                variant="secondary"
                size="sm"
                onClick={handleCopyTranscript}
              >
                <HugeiconsIcon icon={Copy01Icon} className="mr-2 size-3.5" />
                Copy
              </Button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild={true}>
                  <Button variant="secondary" size="sm">
                    <HugeiconsIcon
                      icon={Download01Icon}
                      className="mr-2 size-3.5"
                    />
                    Download
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start">
                  {TRANSCRIPT_EXPORT_FORMATS.map((format) => {
                    const blocked = formatNeedsTimestamps(format) && !timed;
                    return (
                      <DropdownMenuItem
                        key={format}
                        disabled={blocked}
                        onClick={() => void handleDownload(format)}
                      >
                        <span>{FORMAT_LABELS[format]}</span>
                        {blocked ? (
                          <span className="ml-auto pl-3 text-ui-11p5 text-muted-foreground">
                            Needs timestamps
                          </span>
                        ) : null}
                      </DropdownMenuItem>
                    );
                  })}
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
            <div className="flex min-w-0 flex-wrap items-center gap-x-2 text-ui-11p5 text-muted-foreground">
              <span className="min-w-0 truncate">{transcribedName}</span>
              <span>·</span>
              <span className="min-w-0 truncate" title={transcriptModel}>
                {transcriptModel.split("/").pop() || transcriptModel}
              </span>
              {language ? <span>· {language}</span> : null}
              {duration ? (
                <span>
                  ·{" "}
                  <span className="font-mono tabular-nums">
                    {formatTimestamp(duration)}
                  </span>
                </span>
              ) : null}
              {!transcriptRecord && !transcriptExported && (
                <span>· Not saved</span>
              )}
            </div>
            <TranscriptView
              text={transcript}
              details={transcriptDetails}
              names={speakerNames}
              onRename={renameSpeaker}
              view={view}
              onViewChange={setView}
              position={position}
              playing={playing}
              player={audioAvailable ? player : null}
            />
          </section>
        ) : transcriptError ? (
          <div className="flex flex-col gap-1" role="alert">
            <p className="text-ui-13 font-medium text-destructive">
              Could not transcribe {transcribedName ?? "that audio"}.
            </p>
            <p className="text-ui-13 text-muted-foreground">
              {transcriptError}
            </p>
          </div>
        ) : busy !== "transcribing" ? (
          <p className="text-ui-13 text-muted-foreground">
            Record or upload audio to transcribe. Completed transcripts are
            saved to history.
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
          onSelect={selectRecord}
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
