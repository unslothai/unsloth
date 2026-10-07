// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { PillTabs } from "@/features/model-picker/components/model-selector/pill-tabs";
import { prefersReducedMotion } from "@/features/settings";
import { cn } from "@/lib/utils";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";
import { fetchAudioBlob } from "../api";
import { sourceFileUrl } from "../audio-run-request";
import {
  type TranscriptView as TranscriptViewMode,
  useAudioTranscribeStore,
} from "../stores/audio-transcribe-store";
import {
  SPEAKER_NAME_MAX_LENGTH,
  type TranscriptDetails,
  type TranscriptSource,
  activeSegmentIndex,
  formatTimestamp,
  hasSpeakers,
  paragraphs,
  sanitizeSpeakerName,
  speakerLabel,
} from "../transcript-model";
import { Waveform, type WaveformControl } from "./waveform";
import { decodePeaks } from "./waveform-decode";

const VIEW_TABS = [
  { value: "text", label: "Text" },
  { value: "segments", label: "Segments" },
];

const SPEAKER_COLORS = 5;

type SourceAudio =
  | { status: "idle" | "loading" | "missing" }
  | {
      status: "ready";
      url: string;
      peaks: number[] | null;
      durationS: number | null;
    };

function useSourceAudio(source: TranscriptSource | null): SourceAudio {
  const [state, setState] = useState<SourceAudio>({ status: "idle" });
  const kind = source?.kind ?? null;
  const id = source?.id ?? null;
  useEffect(() => {
    if (!(kind && id)) {
      setState({ status: "idle" });
      return;
    }
    const controller = new AbortController();
    let url: string | null = null;
    setState({ status: "loading" });
    fetchAudioBlob(
      sourceFileUrl({ kind, id, name: "", durationS: null }),
      controller.signal,
    )
      .then(async (blob) => {
        const decoded = await decodePeaks(blob);
        if (controller.signal.aborted) return;
        url = URL.createObjectURL(blob);
        setState({ status: "ready", url, ...decoded });
      })
      .catch(() => {
        if (!controller.signal.aborted) setState({ status: "missing" });
      });
    return () => {
      controller.abort();
      if (url) URL.revokeObjectURL(url);
    };
  }, [kind, id]);
  return state;
}

function SpeakerChip({
  id,
  speakers,
  names,
  onRename,
}: Pick<TranscriptDetails, "speakers"> & {
  id: string;
  names: Readonly<Record<string, string>>;
  onRename: (id: string, name: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState("");
  const inputId = useId();
  const label = speakerLabel(id, speakers, names);
  const index = speakers.findIndex((speaker) => speaker.id === id);
  const color = (Math.max(0, index) % SPEAKER_COLORS) + 1;
  const save = () => {
    onRename(id, sanitizeSpeakerName(draft));
    setOpen(false);
  };
  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        if (next) setDraft(names[id] ?? "");
        setOpen(next);
      }}
    >
      <PopoverTrigger asChild={true}>
        <button
          type="button"
          aria-label={`Rename ${label}`}
          className="inline-flex max-w-full shrink-0 items-center gap-1.5 rounded-full bg-muted px-2 py-0.5 text-ui-11p5 font-medium text-foreground transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <span
            aria-hidden="true"
            className="size-[calc(8px*var(--ui-space-scale,1))] shrink-0 rounded-full"
            style={{ backgroundColor: `var(--chart-${color})` }}
          />
          <span className="min-w-0 truncate">{label}</span>
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-64 gap-2 p-3">
        <label htmlFor={inputId} className="text-ui-13 font-medium">
          Name
        </label>
        <Input
          id={inputId}
          value={draft}
          placeholder={speakers[index]?.label ?? id}
          maxLength={SPEAKER_NAME_MAX_LENGTH}
          autoFocus={true}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              save();
            }
          }}
        />
        <p className="text-ui-11p5 leading-snug text-muted-foreground">
          Renames every line by this speaker.
        </p>
        <div className="flex justify-end gap-2">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => setOpen(false)}
          >
            Cancel
          </Button>
          <Button type="button" variant="secondary" size="sm" onClick={save}>
            Save
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
}

export function TranscriptView({
  text,
  details,
  names,
  onRename,
  durationS,
  children,
}: {
  text: string;
  details: TranscriptDetails;
  names: Readonly<Record<string, string>>;
  onRename: (id: string, name: string) => void;
  durationS: number | null;
  children: ReactNode;
}) {
  const { segments, speakers, source } = details;
  const view = useAudioTranscribeStore((state) => state.view);
  const audio = useSourceAudio(source);
  const ready = audio.status === "ready" ? audio : null;
  const control = useRef<WaveformControl | null>(null);
  const player = ready ? control : null;
  const [position, setPosition] = useState(0);
  const [playing, setPlaying] = useState(false);
  const handlePosition = useCallback((seconds: number, isPlaying: boolean) => {
    setPosition(seconds);
    setPlaying(isPlaying);
  }, []);
  const hasSegments = segments.length > 0;
  const speakersOn = hasSpeakers(details);
  const shown = hasSegments ? view : "text";
  const active =
    playing || position > 0 ? activeSegmentIndex(position, segments) : -1;
  // A fixed width in ch keeps rows aligned at any UI size; h:mm:ss needs 7.
  const stampWidth =
    hasSegments && segments[segments.length - 1].start >= 3600
      ? "min-w-[7ch]"
      : "min-w-[5ch]";
  const stampClass = cn(
    "shrink-0 px-1 font-mono text-ui-11p5 tabular-nums text-muted-foreground",
    stampWidth,
  );
  const tabsHint = !hasSegments
    ? "No timestamps in this transcript. Turn on Timestamps with a model that supports them and transcribe again."
    : shown === "segments" && player
      ? "Select a time to play from there."
      : null;
  const activeRow = useRef<HTMLLIElement | null>(null);

  // Only while playing, so reading ahead while paused is never yanked back.
  useEffect(() => {
    if (!playing || active < 0 || shown !== "segments") return;
    activeRow.current?.scrollIntoView({
      block: "nearest",
      behavior: prefersReducedMotion() ? "auto" : "smooth",
    });
  }, [active, playing, shown]);

  const chip = (id: string) => (
    <SpeakerChip
      id={id}
      speakers={speakers}
      names={names}
      onRename={onRename}
    />
  );

  return (
    <>
      {!source ? null : audio.status === "missing" ? (
        <p className="text-ui-11p5 leading-snug text-muted-foreground">
          The audio for this transcript is no longer available. Timestamps still
          export.
        </p>
      ) : (
        <Waveform
          peaks={ready?.peaks ?? null}
          durationS={ready?.durationS ?? durationS}
          src={ready?.url ?? null}
          label={source.name || "Transcript audio"}
          controlRef={control}
          onPositionChange={handlePosition}
        />
      )}
      {children}
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-1.5">
          <PillTabs
            ariaLabel="Transcript view"
            tabs={VIEW_TABS}
            value={shown}
            onValueChange={(next) =>
              useAudioTranscribeStore.setState({
                view: next as TranscriptViewMode,
              })
            }
            disabled={!hasSegments}
            compact={true}
            fit={true}
            className="[&>button]:px-3"
          />
          {tabsHint ? (
            <p className="text-ui-11p5 leading-snug text-muted-foreground">
              {tabsHint}
            </p>
          ) : null}
        </div>
        {shown === "segments" ? (
          <ol className="flex flex-col gap-0.5">
            {segments.map((segment, index) => {
              const current = index === active;
              const stamp = formatTimestamp(segment.start);
              return (
                <li
                  // biome-ignore lint/suspicious/noArrayIndexKey: segments are positional and never reorder.
                  key={index}
                  ref={current ? activeRow : undefined}
                  aria-current={current ? "true" : undefined}
                  className={cn(
                    "flex items-start gap-2 rounded-md border-l-2 py-1 pr-2 pl-1.5 transition-colors duration-150 motion-reduce:transition-none",
                    current
                      ? "border-foreground bg-muted"
                      : "border-transparent",
                  )}
                >
                  {player ? (
                    <button
                      type="button"
                      aria-label={`Play from ${stamp}`}
                      onClick={() => player.current?.seek(segment.start, true)}
                      className={cn(
                        stampClass,
                        "rounded-sm text-left underline-offset-2 hover:text-foreground hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset",
                      )}
                    >
                      {stamp}
                    </button>
                  ) : (
                    <span className={stampClass}>{stamp}</span>
                  )}
                  <div className="flex min-w-0 flex-1 flex-wrap items-baseline gap-x-2 gap-y-1">
                    {speakersOn && segment.speaker
                      ? chip(segment.speaker)
                      : null}
                    <p className="min-w-0 flex-1 basis-[calc(160px*var(--ui-space-scale,1))] text-sm leading-relaxed text-foreground">
                      {segment.text}
                    </p>
                  </div>
                </li>
              );
            })}
          </ol>
        ) : speakersOn ? (
          <div className="flex flex-col gap-3">
            {paragraphs(segments).map((paragraph, index) => (
              <div
                // biome-ignore lint/suspicious/noArrayIndexKey: paragraphs are positional and never reorder.
                key={index}
                className="flex flex-col items-start gap-1"
              >
                {paragraph.speaker ? chip(paragraph.speaker) : null}
                <p className="text-sm leading-relaxed text-foreground">
                  {paragraph.text}
                </p>
              </div>
            ))}
          </div>
        ) : (
          <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground">
            {text}
          </p>
        )}
      </div>
    </>
  );
}
