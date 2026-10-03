// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { PillTabs } from "@/features/model-picker/components/model-selector/pill-tabs";
import {
  nativeAttachmentIntentToFile,
  registerNativeAttachmentPath,
  useNativeDropTarget,
} from "@/features/native-intents";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  AudioWave01Icon,
  Cancel01Icon,
  Mic01Icon,
  StopIcon,
  Upload04Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
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
} from "../audio-run-request";
import {
  type AudioSourceStatus,
  REFERENCE_EXPIRED_MESSAGE,
} from "../hooks/audio-source-state";
import { recordingSupported, useAudioSource } from "../hooks/use-audio-source";
import { clipWorkflow } from "../workflows";
import { VoicePicker } from "./voice-picker";
import { Waveform } from "./waveform";
import { formatSeconds } from "./waveform-peaks";

const SPOKEN_PROMPT_WORKFLOWS: ReadonlySet<string> = new Set([
  "speak",
  "clone",
]);

/** The gallery clips an input card offers under From history. Provided by the page. */
const AudioHistoryContext = createContext<readonly AudioGalleryClip[]>([]);
export const AudioHistoryProvider = AudioHistoryContext.Provider;

type SourceTab = "upload" | "record" | "history" | "voice";

const AUDIO_ACCEPT = "audio/*,.wav,.mp3,.flac,.ogg,.oga,.opus,.m4a,.aac,.webm";
const NATIVE_AUDIO_EXTS = [
  "wav",
  "mp3",
  "flac",
  "ogg",
  "oga",
  "opus",
  "m4a",
  "aac",
  "webm",
];

const SOURCE_LABEL: Record<AudioSourceSelection["kind"], string> = {
  input: "Upload",
  clip: "From history",
  voice: "Saved voice",
};

export interface AudioSourceInputHandle {
  /** Moves focus to the card, for "Add reference audio" actions. */
  focus: () => void;
  /** Opens the file picker. */
  browse: () => void;
  /** Shows the card's expired state, for an upload the server no longer has. */
  markExpired: () => void;
}

/** One audio input as the shared card: drop a file anywhere on it, pick one, record, reuse a
 *  history clip or a saved voice. A picked file shows its length and waveform as soon as it
 *  decodes, while it uploads. */
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
  /** Hears what the card is doing (uploading, failed, expired), for the page's Generate blocker. */
  onStatusChange?: (status: AudioSourceStatus) => void;
}) {
  const source = useAudioSource({ value, onChange });
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

  // Tauri hands drops over as paths; read them through the native side as the image picker does.
  const nativeDropRef = useNativeDropTarget({
    enabled: !disabled,
    onDragOver: setDragging,
    onDrop: (paths) => {
      setDragging(false);
      const path = paths[0];
      if (!path) return;
      const extension = path.split(".").pop()?.toLowerCase() ?? "";
      if (!NATIVE_AUDIO_EXTS.includes(extension)) {
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
  const name =
    uploading && status.phase === "uploading"
      ? status.name
      : (value?.name ?? "");

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
      className={cn(
        "corner-squircle grid gap-3 rounded-4xl bg-card p-4 ring-1 outline-none transition-colors duration-150 focus-visible:ring-2 focus-visible:ring-ring",
        dragging
          ? "bg-accent ring-[color-mix(in_oklab,var(--foreground)_calc(40%*var(--contrast-edge-gain,1)),transparent)]"
          : "ring-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-edge-gain,1)),transparent)]",
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span
          id={`${id}-label`}
          className="text-ui-13 font-medium text-foreground"
        >
          {label}
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
          <p
            role="alert"
            className="flex items-start gap-1.5 text-ui-12 leading-snug text-foreground"
          >
            <HugeiconsIcon
              icon={Alert02Icon}
              className="mt-0.5 size-3.5 shrink-0 text-destructive"
            />
            {REFERENCE_EXPIRED_MESSAGE}
          </p>
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
            Add it again
          </Button>
        </div>
      ) : showSelected && status.phase !== "error" ? (
        <div className="grid gap-2">
          <div className="flex min-w-0 items-center gap-2 text-ui-12">
            <HugeiconsIcon
              icon={AudioWave01Icon}
              className="size-3.5 shrink-0 text-muted-foreground"
            />
            <span
              className="min-w-0 flex-1 truncate text-foreground"
              title={name}
            >
              {name}
            </span>
            {value ? (
              <span className="shrink-0 rounded-4xl bg-muted px-2 py-0.5 text-ui-11 text-muted-foreground">
                {SOURCE_LABEL[value.kind]}
              </span>
            ) : null}
          </div>
          <Waveform
            peaks={preview.peaks}
            durationS={durationS}
            src={preview.url}
            label={name || label}
          />
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
          ) : durationS !== null && durationS > REFERENCE_MAX_SECONDS ? (
            <p className="text-ui-11p5 leading-snug text-muted-foreground">
              Uses the first {REFERENCE_MAX_SECONDS} s.
            </p>
          ) : null}
        </div>
      ) : (
        <div className="grid gap-3">
          {status.phase === "error" ? (
            <div className="grid gap-2">
              <p
                role="alert"
                className="flex items-start gap-1.5 text-ui-12 leading-snug text-foreground"
              >
                <HugeiconsIcon
                  icon={Alert02Icon}
                  className="mt-0.5 size-3.5 shrink-0 text-destructive"
                />
                {status.message}
              </p>
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
                  Try another file
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
            <PillTabs
              ariaLabel={`${label} source`}
              value={recording ? "record" : tab}
              onValueChange={(next) => setTab(next as SourceTab)}
              disabled={disabled || recording}
              fit={true}
              compact={true}
              className="[&>button]:px-3"
              tabs={tabs}
            />
          ) : null}
          {tab === "upload" && !recording ? (
            <button
              type="button"
              disabled={disabled}
              onClick={() => fileRef.current?.click()}
              className="flex min-h-[calc(88px*var(--ui-space-scale,1))] flex-col items-center justify-center gap-1 rounded-xl border border-dashed border-border px-3 text-ui-12 text-muted-foreground transition-colors hover:border-[color-mix(in_oklab,var(--foreground)_calc(30%*var(--contrast-edge-gain,1)),transparent)] hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <HugeiconsIcon icon={Upload04Icon} className="size-5" />
              <span>Drop an audio file here, or choose one</span>
              {hint ? <span className="text-ui-11p5">{hint}</span> : null}
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
                <HugeiconsIcon
                  icon={recording ? StopIcon : Mic01Icon}
                  className="size-4"
                />
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
                  "Read a sentence or two in a quiet room."
                )}
              </output>
            </div>
          ) : null}
          {tab === "history" && !recording ? (
            history.length === 0 ? (
              <p className="text-ui-12 leading-snug text-muted-foreground">
                Clips you generate on Speak, Clone or Music show up here.
              </p>
            ) : (
              <ul className="hover-scrollbar grid max-h-[calc(196px*var(--ui-space-scale,1))] gap-0.5 overflow-y-auto">
                {history.map((clip) => (
                  <li key={clip.id}>
                    <button
                      type="button"
                      disabled={disabled}
                      onClick={() =>
                        onChange({
                          kind: "clip",
                          id: clip.id,
                          name: clip.prompt || "Generated clip",
                          durationS: clip.duration_s,
                          // Only Speak and Clone clips say their prompt; Convert and Music prompts are labels.
                          transcript: SPOKEN_PROMPT_WORKFLOWS.has(
                            clipWorkflow(clip),
                          )
                            ? clip.prompt || null
                            : null,
                          language: null,
                        })
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
