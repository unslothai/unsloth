// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { PillTabs } from "@/features/model-picker/components/model-selector/pill-tabs";
import { prefersReducedMotion } from "@/features/settings";
import { cn } from "@/lib/utils";
import { type RefObject, useEffect, useRef } from "react";
import type { TranscriptView as TranscriptViewMode } from "../stores/audio-transcribe-store";
import {
  type TranscriptDetails,
  activeSegmentIndex,
  formatTimestamp,
  paragraphs,
} from "../transcript-model";
import { SpeakerChip } from "./speaker-chip";
import type { WaveformControl } from "./waveform";

const VIEW_TABS = [
  { value: "text", label: "Text" },
  { value: "segments", label: "Segments" },
];

export function TranscriptView({
  text,
  details,
  names,
  onRename,
  view,
  onViewChange,
  position,
  playing,
  player,
}: {
  text: string;
  details: TranscriptDetails;
  names: Readonly<Record<string, string>>;
  onRename: (id: string, name: string) => void;
  view: TranscriptViewMode;
  onViewChange: (view: TranscriptViewMode) => void;
  position: number;
  playing: boolean;
  player: RefObject<WaveformControl | null> | null;
}) {
  const { segments, speakers } = details;
  const hasSegments = segments.length > 0;
  const hasSpeakers =
    speakers.length > 0 && segments.some((segment) => Boolean(segment.speaker));
  const shown = hasSegments ? view : "text";
  const active =
    playing || position > 0 ? activeSegmentIndex(position, segments) : -1;
  // A fixed width in ch keeps rows aligned at any UI size; h:mm:ss needs 7.
  const stampWidth =
    hasSegments && segments[segments.length - 1].start >= 3600
      ? "min-w-[7ch]"
      : "min-w-[5ch]";
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
    <div className="flex flex-col gap-3">
      <div className="flex flex-col gap-1.5">
        <PillTabs
          ariaLabel="Transcript view"
          tabs={VIEW_TABS}
          value={shown}
          onValueChange={(next) => onViewChange(next as TranscriptViewMode)}
          disabled={!hasSegments}
          compact={true}
          fit={true}
          className="[&>button]:px-3"
        />
        {hasSegments ? (
          shown === "segments" && player ? (
            <p className="text-ui-11p5 leading-snug text-muted-foreground">
              Select a time to play from there.
            </p>
          ) : null
        ) : (
          <p className="text-ui-11p5 leading-snug text-muted-foreground">
            No timestamps in this transcript. Turn on Timestamps with a model
            that supports them and transcribe again.
          </p>
        )}
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
                  current ? "border-foreground bg-muted" : "border-transparent",
                )}
              >
                {player ? (
                  <button
                    type="button"
                    aria-label={`Play from ${stamp}`}
                    onClick={() => player.current?.seek(segment.start, true)}
                    className={cn(
                      "shrink-0 rounded-sm px-1 text-left font-mono text-ui-11p5 tabular-nums text-muted-foreground underline-offset-2 hover:text-foreground hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset",
                      stampWidth,
                    )}
                  >
                    {stamp}
                  </button>
                ) : (
                  <span
                    className={cn(
                      "shrink-0 px-1 font-mono text-ui-11p5 tabular-nums text-muted-foreground",
                      stampWidth,
                    )}
                  >
                    {stamp}
                  </span>
                )}
                <div className="flex min-w-0 flex-1 flex-wrap items-baseline gap-x-2 gap-y-1">
                  {hasSpeakers && segment.speaker
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
      ) : hasSpeakers ? (
        <div className="flex flex-col gap-3">
          {paragraphs(segments, true).map((paragraph, index) => (
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
  );
}
