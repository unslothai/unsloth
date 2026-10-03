// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { StopIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Progress } from "@/components/ui/progress";
import { type Ref, useEffect, useState } from "react";
import type { AudioGalleryClip } from "../api";
import { RECORDING_MAX_SECONDS } from "../audio-workspace-constants";
import { formatClipDuration } from "../audio-workspace-utils";
import {
  AudioHistoryProvider,
  AudioSourceInput,
  type AudioSourceInputHandle,
} from "../components/audio-source-input";
import type { AudioHostState } from "../hooks/audio-host-state";
import type { AudioSourceStatus } from "../hooks/audio-source-state";
import { useAudioTranscribeStore } from "../stores/audio-transcribe-store";
import {
  SPEAKERS_MODEL_NAME,
  type TranscribeSwitch,
  type TranscribeSwitches,
} from "../transcribe-capabilities";
import { transcribeLanguageFor } from "../transcribe-languages";
import type { TranscriptProgress } from "../transcript-stream";
import { GenerateActions, type GenerateBlocker } from "./tts-workspace";

// Radix Select cannot hold an empty value, so "detect" travels as this.
const AUTO = "__auto__";

function SwitchRow({
  id,
  label,
  value,
  onChange,
  disabled,
  onUseSpeakersModel,
}: {
  id: string;
  label: string;
  value: TranscribeSwitch;
  onChange: (checked: boolean) => void;
  disabled: boolean;
  onUseSpeakersModel?: () => void;
}) {
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
            onCheckedChange={onChange}
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
  const setSource = useAudioTranscribeStore((state) => state.setSource);
  const language = useAudioTranscribeStore((state) => state.language);
  const setLanguage = useAudioTranscribeStore((state) => state.setLanguage);
  const setTimestamps = useAudioTranscribeStore((state) => state.setTimestamps);
  const setSpeakers = useAudioTranscribeStore((state) => state.setSpeakers);
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
          onChange={setSource}
          disabled={disabled}
          handleRef={sourceHandle}
          onStatusChange={onSourceStatusChange}
          maxRecordSeconds={RECORDING_MAX_SECONDS}
          expiredMessage="This upload expired. Add it again."
          usesFirstSeconds={null}
        />
      </div>

      {languages.length > 1 ? (
        <div className="grid gap-1.5">
          <label
            className="text-ui-13 font-medium text-foreground"
            htmlFor="transcribe-language"
          >
            Language
          </label>
          <Select
            value={languageValue}
            onValueChange={(next) => setLanguage(next === AUTO ? "" : next)}
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
          <p className="text-ui-11p5 leading-snug text-muted-foreground">
            Picking the spoken language helps with short or noisy clips.
          </p>
        </div>
      ) : null}

      <SwitchRow
        id="transcribe-timestamps"
        label="Timestamps"
        value={switches.timestamps}
        onChange={setTimestamps}
        disabled={disabled}
        onUseSpeakersModel={onUseSpeakersModel}
      />
      <SwitchRow
        id="transcribe-speakers"
        label="Speakers"
        value={switches.speakers}
        onChange={setSpeakers}
        disabled={disabled}
        onUseSpeakersModel={
          switches.timestamps.suggestSpeakersModel
            ? undefined
            : onUseSpeakersModel
        }
      />
    </AudioHistoryProvider>
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
      {blocker && !running ? (
        <p
          id="transcribe-blocker"
          className="text-center text-ui-11p5 leading-snug text-muted-foreground"
        >
          {blocker.reason}
          <GenerateActions actions={blocker.actions} />
        </p>
      ) : notice && !running ? (
        <p
          id="transcribe-notice"
          className="text-center text-ui-11p5 leading-snug text-muted-foreground"
        >
          {notice}
        </p>
      ) : null}
    </div>
  );
}
