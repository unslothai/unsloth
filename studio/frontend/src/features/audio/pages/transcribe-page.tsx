// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import {
  Copy01Icon,
  Download01Icon,
  StopIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type Ref, useEffect, useState } from "react";
import type { AudioGalleryClip } from "../api";
import { RECORDING_MAX_SECONDS } from "../audio-workspace-constants";
import { formatClipDuration } from "../audio-workspace-utils";
import {
  AudioHistoryProvider,
  AudioSourceInput,
  type AudioSourceInputHandle,
} from "../components/audio-source-input";
import { Field } from "../components/field";
import { TranscriptView } from "../components/transcript-view";
import { TranscriptExportItems, TranscriptGallery } from "../transcript-gallery";
import { TranscriptionProgress } from "../transcription-progress";
import type { AudioHostState } from "../hooks/audio-host-state";
import type { AudioSourceStatus } from "../hooks/audio-source-state";
import type { Transcription } from "../hooks/use-transcription";
import { useAudioTranscribeStore } from "../stores/audio-transcribe-store";
import {
  SPEAKERS_MODEL_NAME,
  type TranscribeSwitch,
  type TranscribeSwitches,
} from "../transcribe-capabilities";
import { transcribeLanguageFor } from "../transcribe-languages";
import type { TranscriptExportFormat } from "../transcript-export";
import { formatTimestamp } from "../transcript-model";
import type { TranscriptProgress } from "../transcript-stream";
import { GenerateActions, type GenerateBlocker } from "./tts-workspace";

// Radix Select cannot hold an empty value, so "detect" travels as this.
const AUTO = "__auto__";

function SwitchRow({
  name,
  label,
  value,
  disabled,
  onUseSpeakersModel,
}: {
  name: "timestamps" | "speakers";
  label: string;
  value: TranscribeSwitch;
  disabled: boolean;
  onUseSpeakersModel?: () => void;
}) {
  const id = `transcribe-${name}`;
  return (
    <div className="grid gap-1.5">
      <label
        htmlFor={id}
        className="flex items-center justify-between gap-3 text-ui-13 font-medium text-foreground"
      >
        {label}
        {value.always ? (
          <span
            id={id}
            className="text-ui-11p5 font-normal text-muted-foreground"
          >
            Always on
          </span>
        ) : (
          <Switch
            id={id}
            checked={value.checked}
            disabled={disabled || value.disabled}
            onCheckedChange={(checked) =>
              useAudioTranscribeStore.setState({ [name]: checked })
            }
            aria-describedby={`${id}-hint`}
          />
        )}
      </label>
      <p
        id={`${id}-hint`}
        className="text-ui-11p5 leading-snug text-muted-foreground"
      >
        {value.hint}
        {value.suggestSpeakersModel && onUseSpeakersModel ? (
          <GenerateActions
            actions={[
              {
                label: `Use ${SPEAKERS_MODEL_NAME}`,
                onClick: onUseSpeakersModel,
              },
            ]}
          />
        ) : null}
      </p>
    </div>
  );
}

export function TranscribeRail({
  historyClips,
  disabled,
  sourceHandle,
  onSourceStatusChange,
  switches,
  languages,
  onUseSpeakersModel,
}: {
  historyClips: readonly AudioGalleryClip[];
  disabled: boolean;
  sourceHandle: Ref<AudioSourceInputHandle>;
  onSourceStatusChange: (status: AudioSourceStatus) => void;
  switches: TranscribeSwitches;
  languages: readonly { code: string; name: string }[];
  onUseSpeakersModel?: () => void;
}) {
  const source = useAudioTranscribeStore((state) => state.source);
  const language = useAudioTranscribeStore((state) => state.language);
  // A saved language the model no longer lists falls back to Auto rather than vanishing.
  const languageValue = transcribeLanguageFor(language, languages) || AUTO;
  return (
    <AudioHistoryProvider value={historyClips}>
      <div data-tour="audio-record">
        <AudioSourceInput
          id="transcribe-source"
          label="Audio"
          hint="Up to 30 minutes. Drop a file, record, or reuse a clip."
          value={source}
          onChange={(next) => useAudioTranscribeStore.setState({ source: next })}
          disabled={disabled}
          handleRef={sourceHandle}
          onStatusChange={onSourceStatusChange}
          maxRecordSeconds={RECORDING_MAX_SECONDS}
          expiredMessage="This upload expired. Add it again."
          usesFirstSeconds={null}
          recordHint="Record up to 30 minutes, then transcribe it."
        />
      </div>

      {languages.length > 1 ? (
        <Field
          label="Language"
          htmlFor="transcribe-language"
          hint="Picking the spoken language helps with short or noisy clips."
        >
          <Select
            value={languageValue}
            onValueChange={(next) =>
              useAudioTranscribeStore.setState({
                language: next === AUTO ? "" : next,
              })
            }
            disabled={disabled}
          >
            <SelectTrigger
              id="transcribe-language"
              size="sm"
              className="w-full"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {languages.map((entry) => (
                <SelectItem key={entry.code || AUTO} value={entry.code || AUTO}>
                  {entry.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Field>
      ) : null}

      {(["timestamps", "speakers"] as const).map((name) => (
        <SwitchRow
          key={name}
          name={name}
          label={name === "timestamps" ? "Timestamps" : "Speakers"}
          value={switches[name]}
          disabled={disabled}
          // One "Use" link when both switches would offer the same model.
          onUseSpeakersModel={
            name === "speakers" && switches.timestamps.suggestSpeakersModel
              ? undefined
              : onUseSpeakersModel
          }
        />
      ))}
    </AudioHistoryProvider>
  );
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
  clearTranscript,
  transcriptDetails,
  speakerNames,
  renameSpeaker,
  selectRecord,
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
  | "clearTranscript"
  | "transcriptDetails"
  | "speakerNames"
  | "renameSpeaker"
  | "selectRecord"
> &
  Pick<AudioHostState, "busy" | "active" | "mode">) {
  const language = transcriptDetails.language ?? transcriptRecord?.language;
  const duration = transcriptDetails.duration ?? transcriptRecord?.duration;
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
            <TranscriptView
              text={transcript}
              details={transcriptDetails}
              names={speakerNames}
              onRename={renameSpeaker}
              durationS={duration ?? null}
            >
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
                    <TranscriptExportItems
                      timed={transcriptDetails.segments.length > 0}
                      label={(format) => <span>{FORMAT_LABELS[format]}</span>}
                      onExport={(format) =>
                        void handleDownloadTranscript(format)
                      }
                    />
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
            </TranscriptView>
          </section>
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

export function TranscribeFooter({
  busy,
  blocker,
  notice,
  shortcutLabel,
  stopping,
  onTranscribe,
  onStop,
  modelName,
  progress,
}: Pick<AudioHostState, "busy"> & {
  blocker: GenerateBlocker | null;
  notice: string | null;
  shortcutLabel: string;
  stopping: boolean;
  onTranscribe: () => void;
  onStop: () => void;
  modelName: string;
  progress: TranscriptProgress | null;
}) {
  const running = busy === "transcribing";
  const working = running || busy === "loading";
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!working) {
      setStartedAt(null);
      return;
    }
    setStartedAt((value) => value ?? Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [working]);
  const elapsed =
    startedAt === null ? 0 : Math.max(0, Math.floor((now - startedAt) / 1000));
  const status =
    busy === "loading" || progress?.phase === "loading"
      ? `Loading ${modelName || "the model"}…`
      : progress?.phase === "downloading_aligner"
        ? "Downloading the timing aligner…"
        : stopping
          ? "Stopping…"
          : "Transcribing…";
  return (
    <div className="flex w-full max-w-sm flex-col gap-2">
      {working ? (
        <>
          <output
            aria-live="polite"
            aria-atomic="true"
            className="text-center text-ui-12 text-muted-foreground"
          >
            {status}
            <span className="ml-1.5 font-mono tabular-nums">
              {formatClipDuration(elapsed)}
            </span>
          </output>
          <Progress
            indeterminate
            aria-label="Transcription in progress"
            className="h-1.5"
          />
        </>
      ) : null}
      <div className="flex flex-wrap items-center justify-center gap-2">
        <Button
          className="relative z-10 mx-auto h-11 px-8 disabled:bg-muted disabled:text-muted-foreground disabled:opacity-100"
          onClick={running ? onStop : onTranscribe}
          disabled={running ? stopping : busy !== null || blocker !== null}
          variant={running ? "destructive" : "default"}
          aria-describedby={
            blocker
              ? "transcribe-blocker"
              : notice
                ? "transcribe-notice"
                : undefined
          }
          aria-keyshortcuts="Control+Enter Meta+Enter"
          title={running ? undefined : `Transcribe (${shortcutLabel})`}
        >
          {running ? (
            <>
              <HugeiconsIcon icon={StopIcon} className="mr-2 size-4" />
              {stopping ? "Stopping…" : "Stop"}
            </>
          ) : busy === "loading" ? (
            "Loading model…"
          ) : (
            "Transcribe"
          )}
        </Button>
      </div>
      {(blocker || notice) && !running ? (
        <p
          id={blocker ? "transcribe-blocker" : "transcribe-notice"}
          className="text-center text-ui-11p5 leading-snug text-muted-foreground"
        >
          {blocker ? blocker.reason : notice}
          <GenerateActions actions={blocker?.actions} />
        </p>
      ) : null}
    </div>
  );
}
