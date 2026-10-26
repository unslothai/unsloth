// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { InfoHint } from "@/components/ui/info-hint";
import {
  nativeAttachmentIntentToFile,
  registerNativeAttachmentPath,
  useNativeDropTarget,
} from "@/features/native-intents";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  Cancel01Icon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import { StopIcon } from "@/lib/media-control-icons";
import { AiSpeechIcon } from "@/lib/hugeicons-derived";
import { MicIcon } from "@/lib/mic-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ReactNode,
  type Ref,
  createContext,
  useContext,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import type { AudioGalleryClip } from "../api";
import {
  type AudioSourceSelection,
  REFERENCE_MAX_SECONDS,
  clipReference,
} from "../audio-run-request";
import { referenceLengthHint } from "../clone-policy";
import {
  type AudioSourceStatus,
  REFERENCE_EXPIRED_MESSAGE,
} from "../hooks/audio-source-state";
import { recordingSupported, useAudioSource } from "../hooks/use-audio-source";
import { clipWorkflow } from "../workflows";
import { VoicePicker } from "./voice-picker";
import { Waveform } from "./waveform";
import { formatSeconds } from "./waveform-peaks";

const AudioHistoryContext = createContext<readonly AudioGalleryClip[]>([]);
export const AudioHistoryProvider = AudioHistoryContext.Provider;
// False while the persistently mounted Audio page is hidden, so no card keeps the mic.
const AudioActiveContext = createContext(true);
export const AudioActiveProvider = AudioActiveContext.Provider;

type SourceTab = "upload" | "record" | "history" | "voice";

const AUDIO_EXTS = "wav mp3 flac ogg oga opus m4a aac webm mp4".split(" ");
const AUDIO_ACCEPT = `audio/*,${AUDIO_EXTS.map((ext) => `.${ext}`).join(",")}`;

const SOURCE_LABEL: Record<AudioSourceSelection["kind"], string> = {
  input: "Upload",
  clip: "From history",
  voice: "Saved voice",
};

export interface AudioSourcePreviewView {
  peaks: number[] | null;
  durationS: number | null;
  src: string | null;
  label: string;
}

export interface AudioSourceInputHandle {
  focus: () => void;
  browse: () => void;
  markExpired: () => void;
}

/** The Library header's tabs, flat: no track, a grey pill for the active one. */
function SourceTabs({
  tabs,
  value,
  onValueChange,
  ariaLabel,
  disabled,
}: {
  tabs: { value: string; label: string }[];
  value: string;
  onValueChange: (value: SourceTab) => void;
  ariaLabel: string;
  disabled: boolean;
}) {
  return (
    <div role="tablist" aria-label={ariaLabel} className="flex flex-wrap gap-1">
      {tabs.map((tab, index) => (
        <button
          key={tab.value}
          type="button"
          role="tab"
          aria-selected={value === tab.value}
          tabIndex={value === tab.value ? 0 : -1}
          disabled={disabled}
          onKeyDown={(event) => {
            if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
            event.preventDefault();
            const step = event.key === "ArrowRight" ? 1 : -1;
            const next = (index + step + tabs.length) % tabs.length;
            onValueChange(tabs[next].value as SourceTab);
            event.currentTarget.parentElement
              ?.querySelectorAll<HTMLElement>('button[role="tab"]')
              .item(next)
              ?.focus();
          }}
          onClick={() => onValueChange(tab.value as SourceTab)}
          className={cn(
            "flex h-7 shrink-0 items-center rounded-full px-2.5 text-ui-12p5 text-foreground/80 transition-colors hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50",
            value === tab.value &&
              "bg-[color-mix(in_srgb,var(--foreground)_7%,var(--background))] font-medium text-foreground dark:bg-[color-mix(in_srgb,var(--card),white_8%)]",
          )}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

function AlertLine({ children }: { children: ReactNode }) {
  return (
    <p
      role="alert"
      className="flex items-start gap-1.5 text-ui-12 leading-snug text-foreground"
    >
      <HugeiconsIcon
        icon={Alert02Icon}
        className="mt-0.5 size-3.5 shrink-0 text-destructive"
      />
      {children}
    </p>
  );
}

export function AudioSourceInput({
  id,
  label,
  hint,
  value,
  onChange,
  disabled = false,
  allowHistory = true,
  allowSavedVoice = true,
  handleRef,
  onStatusChange,
  renderWaveform,
  maxRecordSeconds,
  expiredMessage = REFERENCE_EXPIRED_MESSAGE,
  usesFirstSeconds = REFERENCE_MAX_SECONDS,
  recordHint = "Read a sentence or two in a quiet room.",
  idealSeconds,
}: {
  id: string;
  label: string;
  hint?: string;
  value: AudioSourceSelection | null;
  onChange: (next: AudioSourceSelection | null) => void;
  disabled?: boolean;
  allowHistory?: boolean;
  allowSavedVoice?: boolean;
  handleRef?: Ref<AudioSourceInputHandle>;
  onStatusChange?: (status: AudioSourceStatus) => void;
  renderWaveform?: (preview: AudioSourcePreviewView) => ReactNode;
  maxRecordSeconds?: number;
  expiredMessage?: string;
  usesFirstSeconds?: number | null;
  recordHint?: string;
  /** Seconds that work best, e.g. a clone reference's 5 to 15 s; adds a length hint. */
  idealSeconds?: readonly [number, number];
}) {
  const active = useContext(AudioActiveContext);
  const source = useAudioSource({ value, onChange, maxRecordSeconds, active });
  const history = useContext(AudioHistoryContext);
  const [tab, setTab] = useState<SourceTab>("upload");
  const [dragging, setDragging] = useState(false);
  const cardRef = useRef<HTMLDivElement | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);
  const { status, preview } = source;
  const recording = status.phase === "recording";
  const uploading = status.phase === "uploading";
  const showSelected = value !== null || uploading;

  const onStatusChangeRef = useRef(onStatusChange);
  onStatusChangeRef.current = onStatusChange;
  useEffect(() => {
    onStatusChangeRef.current?.(status);
  }, [status]);

  useImperativeHandle(handleRef, () => ({
    focus: () => cardRef.current?.focus(),
    browse: () => fileRef.current?.click(),
    markExpired: source.expire,
  }));

  // Tauri hands drops over as paths.
  const nativeDropRef = useNativeDropTarget({
    enabled: !disabled,
    onDragOver: setDragging,
    onDrop: (paths) => {
      setDragging(false);
      const path = paths[0];
      if (!path) return;
      const extension = path.split(".").pop()?.toLowerCase() ?? "";
      if (!AUDIO_EXTS.includes(extension)) {
        source.fail(
          "This is not an audio file. Pick a WAV, MP3, FLAC, OGG or M4A file.",
        );
        return;
      }
      void registerNativeAttachmentPath(path)
        .then(nativeAttachmentIntentToFile)
        .then((file) => source.pickFile(file))
        .catch(() => source.fail("Could not read the dropped file."));
    },
  });

  const tabs = [
    { value: "upload", label: "Upload" },
    ...(recordingSupported() ? [{ value: "record", label: "Record" }] : []),
    ...(allowHistory ? [{ value: "history", label: "From history" }] : []),
    ...(allowSavedVoice ? [{ value: "voice", label: "Saved voice" }] : []),
  ];

  const durationS = preview.durationS ?? value?.durationS ?? null;
  const lengthHint = idealSeconds
    ? referenceLengthHint(durationS, idealSeconds)
    : null;
  const name = status.phase === "uploading" ? status.name : (value?.name ?? "");

  return (
    // biome-ignore lint/a11y/noNoninteractiveTabindex: the card takes focus so "Add reference audio" can bring the user to it.
    <section
      ref={(element) => {
        cardRef.current = element as HTMLDivElement | null;
        nativeDropRef(element);
      }}
      tabIndex={-1}
      aria-labelledby={`${id}-label`}
      aria-busy={uploading || status.phase === "loading"}
      onDragOver={(event) => {
        if (disabled || !event.dataTransfer.types.includes("Files")) return;
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null))
          setDragging(false);
      }}
      onDrop={(event) => {
        if (disabled) return;
        event.preventDefault();
        setDragging(false);
        const file = event.dataTransfer.files?.[0];
        if (file) void source.pickFile(file);
      }}
      className="grid gap-2.5 rounded-md outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      <div className="flex items-center justify-between gap-2">
        <span className="flex items-center gap-1">
          <span
            id={`${id}-label`}
            className="text-ui-13 font-medium text-foreground"
          >
            {label}
          </span>
          {hint ? <InfoHint>{hint}</InfoHint> : null}
        </span>
        {showSelected ? (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-auto px-2 py-1 text-ui-11p5"
            disabled={disabled}
            onClick={source.clear}
          >
            <HugeiconsIcon icon={Cancel01Icon} className="size-3" />
            Remove
          </Button>
        ) : null}
      </div>

      <input
        ref={fileRef}
        id={`${id}-file`}
        type="file"
        accept={AUDIO_ACCEPT}
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          if (file) void source.pickFile(file);
        }}
      />

      {status.phase === "expired" ? (
        <div className="grid gap-2">
          <AlertLine>{expiredMessage}</AlertLine>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="self-start"
            onClick={() => {
              source.clear();
              fileRef.current?.click();
            }}
          >
            <HugeiconsIcon icon={PlusSignIcon} data-icon="inline-start" />
            Add it again
          </Button>
        </div>
      ) : showSelected && status.phase !== "error" ? (
        <div className="grid gap-2">
          <div className="flex min-w-0 items-center gap-2 text-ui-12">
            <HugeiconsIcon
              icon={AiSpeechIcon}
              className="size-3.5 shrink-0 text-muted-foreground"
            />
            <span
              className="min-w-0 flex-1 truncate text-foreground"
              title={name}
            >
              {name}
            </span>
            {value ? (
              <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-ui-11 text-muted-foreground">
                {SOURCE_LABEL[value.kind]}
              </span>
            ) : null}
          </div>
          {renderWaveform ? (
            renderWaveform({
              peaks: preview.peaks,
              durationS,
              src: preview.url,
              label: name || label,
            })
          ) : (
            <Waveform
              peaks={preview.peaks}
              durationS={durationS}
              src={preview.url}
              label={name || label}
            />
          )}
          {status.phase === "uploading" ? (
            <div className="grid gap-1">
              <Progress
                value={
                  status.progress === null ? undefined : status.progress * 100
                }
                indeterminate={status.progress === null}
                aria-label="Upload progress"
                className="h-1"
              />
              <output
                aria-live="polite"
                className="text-ui-11p5 text-muted-foreground"
              >
                {status.progress === null
                  ? "Uploading…"
                  : `Uploading ${Math.round(status.progress * 100)} %`}
              </output>
            </div>
          ) : status.phase === "loading" ? (
            <output className="text-ui-11p5 text-muted-foreground">
              Loading the clip…
            </output>
          ) : usesFirstSeconds !== null &&
            durationS !== null &&
            durationS > usesFirstSeconds ? (
            <p className="text-ui-11p5 leading-snug text-muted-foreground">
              {usesFirstSeconds % 60 === 0 && usesFirstSeconds >= 60
                ? `Uses the first ${usesFirstSeconds / 60} min.`
                : `Uses the first ${usesFirstSeconds} s.`}
            </p>
          ) : lengthHint ? (
            <p className="text-ui-11p5 leading-snug text-muted-foreground">
              {lengthHint}
            </p>
          ) : null}
        </div>
      ) : (
        <div className="grid gap-3">
          {status.phase === "error" ? (
            <div className="grid gap-2">
              <AlertLine>{status.message}</AlertLine>
              <div className="flex flex-wrap gap-2">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    source.dismissError();
                    fileRef.current?.click();
                  }}
                >
                  {tab === "record" ? "Upload a file" : "Try another file"}
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={source.dismissError}
                >
                  Dismiss
                </Button>
              </div>
            </div>
          ) : null}
          {tabs.length > 1 ? (
            <SourceTabs
              ariaLabel={`${label} source`}
              value={recording ? "record" : tab}
              onValueChange={setTab}
              disabled={disabled || recording}
              tabs={tabs}
            />
          ) : null}
          {tab === "upload" && !recording ? (
            <button
              type="button"
              disabled={disabled}
              onClick={() => fileRef.current?.click()}
              className={cn(
                "flex min-h-[calc(72px*var(--ui-space-scale,1))] flex-col items-center justify-center gap-1 rounded-xl border border-dashed border-border px-3 text-ui-12 text-muted-foreground transition-colors hover:border-[color-mix(in_oklab,var(--foreground)_calc(30%*var(--contrast-edge-gain,1)),transparent)] hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                dragging &&
                  "border-[color-mix(in_oklab,var(--foreground)_calc(40%*var(--contrast-edge-gain,1)),transparent)] bg-accent text-foreground",
              )}
            >
              <HugeiconsIcon icon={PlusSignIcon} className="size-5" />
              <span>Drop audio or choose a file</span>
            </button>
          ) : null}
          {tab === "record" || recording ? (
            <div className="flex items-center gap-3">
              <Button
                type="button"
                variant={recording ? "destructive" : "outline"}
                disabled={disabled}
                onClick={() =>
                  recording
                    ? source.stopRecording()
                    : void source.startRecording()
                }
              >
                {recording ? (
                  <HugeiconsIcon icon={StopIcon} className="size-4" />
                ) : (
                  <MicIcon className="size-4" />
                )}
                {recording ? "Stop" : "Record"}
              </Button>
              <output
                aria-live="polite"
                className="text-ui-12 text-muted-foreground"
              >
                {recording ? (
                  <>
                    Recording{" "}
                    <span className="font-mono tabular-nums">
                      {formatSeconds(source.elapsed)}
                    </span>
                  </>
                ) : (
                  recordHint
                )}
              </output>
            </div>
          ) : null}
          {tab === "history" && !recording ? (
            history.length === 0 ? (
              <p className="text-ui-12 leading-snug text-muted-foreground">
                Clips you make in Audio show up here.
              </p>
            ) : (
              <ul className="hover-scrollbar grid min-w-0 max-h-[calc(196px*var(--ui-space-scale,1))] grid-cols-[minmax(0,1fr)] gap-0.5 overflow-y-auto">
                {history.map((clip) => (
                  <li key={clip.id} className="min-w-0">
                    <button
                      type="button"
                      disabled={disabled}
                      onClick={() =>
                        onChange(clipReference({ ...clip, workflow: clipWorkflow(clip) }))
                      }
                      className="flex w-full min-w-0 items-center gap-2 rounded-full px-3 py-1.5 text-left text-ui-13 transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    >
                      <span className="min-w-0 flex-1 truncate">
                        {clip.prompt}
                      </span>
                      <span className="shrink-0 font-mono text-ui-11p5 tabular-nums text-muted-foreground">
                        {formatSeconds(clip.duration_s)}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            )
          ) : null}
          {tab === "voice" && !recording ? (
            <VoicePicker
              disabled={disabled}
              onSelect={(voice) =>
                onChange({
                  kind: "voice",
                  id: voice.id,
                  name: voice.name,
                  durationS: voice.duration_s,
                  transcript: voice.transcript,
                  language: voice.language,
                })
              }
            />
          ) : null}
        </div>
      )}
    </section>
  );
}
