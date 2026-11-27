// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { PillTabs } from "@/features/model-picker/components/model-selector/pill-tabs";
import { PauseIcon } from "@/lib/media-control-icons";
import { cn } from "@/lib/utils";
import { HugeiconsIcon } from "@hugeicons/react";
import { PlayIcon } from "lucide-react";
import { type Ref, useCallback, useEffect, useRef, useState } from "react";
import { useClipPeaks } from "./ab-compare";
import { CLIP_ICON_COLUMN, CLIP_PLAY_BUTTON, CLIPS_TITLE_INSET } from "./clips-card";
import { type ABPlayback, nextPlayback } from "./convert-ab-compare-state";
import { PlaybackTime, WaveformBars } from "./waveform";

const SEEK_STEP_SECONDS = 5;

export interface ABSide {
  label: string;
  src?: string;
  /** Server URL to draw bars from when `src` is an object URL, which cannot be fetched again. */
  fileUrl?: string;
  unavailable?: string;
}

export type ABSideId = "a" | "b";

export function ABCompare({
  a,
  b,
  side,
  onSideChange,
  ariaLabel,
  playerRef,
}: {
  a: ABSide;
  b: ABSide;
  side: ABSideId;
  onSideChange: (side: ABSideId) => void;
  ariaLabel: string;
  playerRef?: Ref<HTMLButtonElement>;
}) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [position, setPosition] = useState(0);
  const [duration, setDuration] = useState(0);
  const pendingRef = useRef<ABPlayback | null>(null);

  const shownSide: ABSideId =
    side === "a" && a.unavailable && !b.unavailable
      ? "b"
      : side === "b" && b.unavailable && !a.unavailable
        ? "a"
        : side;
  const current = shownSide === "a" ? a : b;
  const src = current.unavailable ? undefined : current.src;
  const barsFor = (s: ABSide) => (s.unavailable || !s.src ? null : (s.fileUrl ?? s.src));
  // Both sides at once, so switching reuses each decode instead of fetching it again.
  const peaks = useClipPeaks([barsFor(a), barsFor(b)]);
  const barsSrc = src ? barsFor(current) : null;
  const bars = barsSrc ? (peaks[barsSrc] ?? null) : null;

  // A new side starts from the top unless a switch carries the moment over (handleLoadedMetadata).
  // biome-ignore lint/correctness/useExhaustiveDependencies: the source is the trigger, not an input.
  useEffect(() => {
    setPlaying(false);
    setPosition(0);
    setDuration(0);
  }, [src]);

  const toggle = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) audio.play().catch(() => setPlaying(false));
    else audio.pause();
  }, []);

  const seekTo = useCallback(
    (seconds: number) => {
      const audio = audioRef.current;
      if (!audio || duration <= 0) return;
      const next = Math.min(duration, Math.max(0, seconds));
      audio.currentTime = next;
      setPosition(next);
    },
    [duration],
  );

  const switchTo = useCallback(
    (next: string) => {
      if (next !== "a" && next !== "b") return;
      if (next === shownSide) return;
      if ((next === "a" ? a : b).unavailable) return;
      const audio = audioRef.current;
      if (audio) {
        pendingRef.current = {
          time: audio.currentTime,
          playing: !audio.paused && !audio.ended,
          duration: audio.duration,
        };
        audio.pause();
      }
      onSideChange(next);
    },
    [a, b, onSideChange, shownSide],
  );

  const handleLoadedMetadata = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    setDuration(Number.isFinite(audio.duration) ? audio.duration : 0);
    const pending = pendingRef.current;
    if (!pending) return;
    pendingRef.current = null;
    const { seek, resume } = nextPlayback(pending, audio.duration);
    if (seek > 0) {
      audio.currentTime = seek;
      setPosition(seek);
    }
    if (resume) {
      // Autoplay rules may refuse.
      audio.play().catch(() => undefined);
    }
  }, []);

  const reason = a.unavailable ?? b.unavailable;
  return (
    <div className="grid w-full gap-2">
      {/* Margin, not padding: padding would stretch the switch's own track back to the card edge. */}
      <div className={CLIPS_TITLE_INSET}>
      <PillTabs
        ariaLabel={ariaLabel}
        value={shownSide}
        onValueChange={switchTo}
        fit={true}
        compact={true}
        className="[&>button]:px-3"
        tabs={[
          { value: "a", label: a.label, disabled: Boolean(a.unavailable) },
          { value: "b", label: b.label, disabled: Boolean(b.unavailable) },
        ]}
      />
      </div>
      {src ? (
        <div className="flex items-center gap-3 pe-1">
          <Button
            ref={playerRef}
            type="button"
            variant="ghost"
            size="icon"
            className={cn(CLIP_PLAY_BUTTON, CLIP_ICON_COLUMN)}
            title={playing ? "Pause" : "Play"}
            aria-label={playing ? `Pause ${current.label}` : `Play ${current.label}`}
            onClick={toggle}
          >
            {playing ? (
              <HugeiconsIcon icon={PauseIcon} className="size-4" />
            ) : (
              <PlayIcon className="size-4" />
            )}
          </Button>
          <WaveformBars
            peaks={bars}
            fraction={duration > 0 ? Math.min(1, position / duration) : 0}
            label={current.label}
            className="h-[calc(56px*var(--ui-space-scale,1))]"
            barWidth={3}
            valueNow={position}
            valueMax={duration}
            onSeek={(next) => seekTo(next * duration)}
            onKeyDown={(event) => {
              if (event.key === " ") {
                event.preventDefault();
                toggle();
              } else if (event.key === "ArrowRight") {
                event.preventDefault();
                seekTo(position + SEEK_STEP_SECONDS);
              } else if (event.key === "ArrowLeft") {
                event.preventDefault();
                seekTo(position - SEEK_STEP_SECONDS);
              } else if (event.key === "Home") {
                event.preventDefault();
                seekTo(0);
              }
            }}
          />
          <PlaybackTime position={position} duration={duration} align="end" />
          {/* biome-ignore lint/a11y/useMediaCaption: generated speech has no caption track; the clip's text is shown above it. */}
          <audio
            ref={audioRef}
            src={src}
            preload="metadata"
            className="hidden"
            onLoadedMetadata={handleLoadedMetadata}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onEnded={() => {
              setPlaying(false);
              setPosition(0);
            }}
            onTimeUpdate={(event) => setPosition(event.currentTarget.currentTime)}
          />
        </div>
      ) : (
        <output className="flex h-12 w-full items-center justify-center rounded-full border border-border text-ui-12 text-muted-foreground">
          {current.unavailable ?? "Loading audio…"}
        </output>
      )}
      {reason && !current.unavailable ? (
        <p className={cn(CLIPS_TITLE_INSET, "text-ui-12/normal text-muted-foreground")}>
          {reason}
        </p>
      ) : null}
    </div>
  );
}
