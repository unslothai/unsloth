// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { PauseIcon } from "@/lib/media-control-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type CSSProperties,
  type KeyboardEvent,
  type ReactNode,
  type Ref,
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { formatSeconds, resamplePeaks } from "./waveform-peaks";
import { PlayIcon } from "lucide-react";

const SEEK_STEP_SECONDS = 5;
const BAR_GAP_PX = 2;
// Fraction of the track height above the baseline.
const BASELINE = 0.7;

/** Bar levels for drawing: lightly smoothed, then spread between the clip's quiet floor and its peak,
 *  so loud, compressed audio (music) still shows its shape instead of a flat comb. */
function shapeLevels(levels: readonly number[]): number[] {
  if (levels.length === 0) return [];
  const smoothed = levels.map((value, index) => {
    const before = levels[index - 1] ?? value;
    const after = levels[index + 1] ?? value;
    return 0.25 * before + 0.5 * value + 0.25 * after;
  });
  const sorted = [...smoothed].sort((a, b) => a - b);
  const floor = sorted[Math.floor(sorted.length * 0.05)] ?? 0;
  const top = sorted[sorted.length - 1] ?? 1;
  const span = top - floor;
  // A near-flat clip keeps its absolute levels rather than amplifying noise.
  if (span < 0.08) return smoothed;
  return smoothed.map((value) => Math.min(1, Math.max(0, (value - floor) / span)));
}

export interface WaveformControl {
  seek: (seconds: number, play?: boolean) => void;
  toggle: () => void;
}

/** The position is also spoken, so colour never carries it alone. */
export function Waveform({
  peaks,
  durationS,
  src,
  label,
  className,
  tailS = 0,
  overlay,
  controlRef,
  onPositionChange,
  onError,
  size = "default",
  hidePlay = false,
}: {
  /** 0..1; null draws a flat placeholder while decoding. */
  peaks: readonly number[] | null;
  durationS: number | null;
  src: string | null;
  label: string;
  className?: string;
  tailS?: number;
  overlay?: ReactNode;
  controlRef?: Ref<WaveformControl>;
  onPositionChange?: (seconds: number, playing: boolean) => void;
  onError?: () => void;
  /** `lg` is a taller track with a larger play button. */
  size?: "default" | "lg";
  /** The host draws its own play button through `controlRef` (the clip card's header). */
  hidePlay?: boolean;
}) {
  const large = size === "lg";
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [position, setPosition] = useState(0);
  const [mediaDuration, setMediaDuration] = useState<number | null>(null);
  const duration =
    durationS && durationS > 0 ? durationS : (mediaDuration ?? 0);
  const fraction = duration > 0 ? Math.min(1, position / duration) : 0;

  useEffect(() => {
    setPlaying(false);
    setPosition(0);
    setMediaDuration(null);
    if (src === null) audioRef.current?.pause();
  }, [src]);

  const toggle = useCallback(() => {
    const audio = audioRef.current;
    if (!(audio && src)) return;
    if (audio.paused) {
      audio.play().catch(() => setPlaying(false));
    } else {
      audio.pause();
    }
  }, [src]);

  const seekTo = useCallback(
    (seconds: number) => {
      const audio = audioRef.current;
      if (!(audio && src) || duration <= 0) return;
      const next = Math.min(duration, Math.max(0, seconds));
      audio.currentTime = next;
      setPosition(next);
    },
    [src, duration],
  );

  useImperativeHandle(
    controlRef,
    () => ({
      seek: (seconds: number, play = false) => {
        const audio = audioRef.current;
        if (!(audio && src)) return;
        const limit = duration > 0 ? duration : Number.POSITIVE_INFINITY;
        const next = Math.min(limit, Math.max(0, seconds));
        audio.currentTime = next;
        setPosition(next);
        if (play && audio.paused) audio.play().catch(() => setPlaying(false));
      },
      toggle,
    }),
    [src, duration, toggle],
  );

  useEffect(() => {
    onPositionChange?.(position, playing);
  }, [onPositionChange, position, playing]);

  const slider = (
    <WaveformBars
      peaks={peaks}
      fraction={fraction}
      label={label}
      className={large ? "h-[calc(56px*var(--ui-space-scale,1))]" : undefined}
      barWidth={large ? 3 : 2}
      valueNow={position}
      valueMax={duration}
      disabled={!src}
      style={tailS > 0 ? { flexGrow: duration > 0 ? duration : 1 } : undefined}
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
  );

  return (
    <div className={cn("flex items-center", large ? "gap-3" : "gap-2", className)}>
      {hidePlay ? null : (
        <Button
          type="button"
          variant="muted"
          size="icon"
          className={cn(
            "shrink-0",
            large
              ? "size-[calc(44px*var(--ui-space-scale,1))]"
              : "size-[calc(32px*var(--ui-space-scale,1))]",
          )}
          title={playing ? "Pause" : "Play"}
          aria-label={playing ? `Pause ${label}` : `Play ${label}`}
          disabled={!src}
          onClick={toggle}
        >
          {playing ? (
            <HugeiconsIcon icon={PauseIcon} className={large ? "size-4.5" : "size-3.5"} />
          ) : (
            <PlayIcon className={large ? "size-4.5" : "size-3.5"} />
          )}
        </Button>
      )}
      {overlay !== undefined || tailS > 0 ? (
        <div
          className={cn(
            "relative flex min-w-0 flex-1",
            large
              ? "h-[calc(56px*var(--ui-space-scale,1))]"
              : "h-[calc(32px*var(--ui-space-scale,1))]",
          )}
        >
          {slider}
          {tailS > 0 ? (
            <div
              aria-hidden="true"
              className="h-full min-w-0 basis-0 rounded-md border border-dashed border-border bg-muted/40"
              style={{ flexGrow: tailS }}
            />
          ) : null}
          {overlay}
        </div>
      ) : (
        slider
      )}
      <PlaybackTime position={position} duration={duration} />
      {src ? (
        // biome-ignore lint/a11y/useMediaCaption: voice references have no caption track; the clip's text is shown beside it.
        <audio
          ref={audioRef}
          src={src}
          preload="metadata"
          className="hidden"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onError={onError}
          onEnded={() => {
            setPlaying(false);
            setPosition(0);
          }}
          onTimeUpdate={(event) => setPosition(event.currentTarget.currentTime)}
          onLoadedMetadata={(event) => {
            const value = event.currentTarget.duration;
            if (Number.isFinite(value)) setMediaDuration(value);
          }}
        />
      ) : null}
    </div>
  );
}

export function WaveformBars({
  peaks,
  fraction,
  label,
  valueNow,
  valueMax,
  disabled = false,
  onSeek,
  onKeyDown,
  className,
  style,
  barWidth = 2,
}: {
  /** 3px on the larger players. */
  barWidth?: number;
  peaks: readonly number[] | null;
  fraction: number;
  label: string;
  valueNow: number;
  valueMax: number;
  disabled?: boolean;
  onSeek?: (fraction: number) => void;
  onKeyDown?: (event: KeyboardEvent<HTMLDivElement>) => void;
  className?: string;
  style?: CSSProperties;
}) {
  const bars = peaks && peaks.length > 0 ? peaks : null;
  // Where the pointer is, as a 0..1 fraction: bars up to it light softly to preview the seek.
  const [hover, setHover] = useState<number | null>(null);
  return (
    <div
      role="slider"
      tabIndex={disabled ? -1 : 0}
      aria-label={`${label} playback position`}
      aria-valuemin={0}
      aria-valuemax={Math.round(valueMax)}
      aria-valuenow={Math.round(valueNow)}
      aria-valuetext={`${formatSeconds(valueNow)} of ${formatSeconds(valueMax)}`}
      aria-disabled={disabled}
      style={style}
      className={cn(
        "relative h-[calc(32px*var(--ui-space-scale,1))] min-w-0 flex-1 cursor-pointer rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        className,
      )}
      onKeyDown={onKeyDown}
      onPointerMove={(event) => {
        if (disabled || !onSeek || event.pointerType === "touch") return;
        const box = event.currentTarget.getBoundingClientRect();
        if (box.width > 0)
          setHover(Math.min(1, Math.max(0, (event.clientX - box.left) / box.width)));
      }}
      onPointerLeave={() => setHover(null)}
      onClick={(event) => {
        if (disabled || !onSeek) return;
        const box = event.currentTarget.getBoundingClientRect();
        if (box.width <= 0) return;
        onSeek(
          Math.min(1, Math.max(0, (event.clientX - box.left) / box.width)),
        );
      }}
    >
      <WaveformBarGlyphs
        peaks={bars}
        barWidth={barWidth}
        isPlayed={(at) => fraction > 0 && at <= fraction}
        isHovered={(at) => hover !== null && at <= hover}
      />
    </div>
  );
}

/** Holds the width of "duration / duration", so starting playback never narrows the bars beside it; start-aligned so the time hugs the bars. */
export function PlaybackTime({
  position,
  duration,
}: {
  position: number;
  duration: number;
}) {
  return (
    <span className="grid shrink-0 justify-items-start font-mono text-ui-11 tabular-nums text-muted-foreground">
      <span aria-hidden="true" className="invisible col-start-1 row-start-1">
        {formatSeconds(duration)}
        <span className="px-0.5">/</span>
        {formatSeconds(duration)}
      </span>
      <span className="col-start-1 row-start-1">
        {position > 0 ? (
          <>
            {formatSeconds(position)}
            <span className="px-0.5">/</span>
          </>
        ) : null}
        {formatSeconds(duration)}
      </span>
    </span>
  );
}

/** HTML pills, not SVG: preserveAspectRatio="none" squashes rx; whole-pixel bars so none renders thinner. */
export function WaveformBarGlyphs({
  peaks,
  isPlayed,
  isHovered,
  barWidth = 2,
}: {
  peaks: readonly number[] | null;
  /** `at` is the bar's centre as a 0..1 fraction of the track. */
  isPlayed: (at: number) => boolean;
  /** Bars a click would play up to; drawn between played and unplayed. */
  isHovered?: (at: number) => boolean;
  barWidth?: number;
}) {
  const boxRef = useRef<HTMLDivElement | null>(null);
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    const box = boxRef.current;
    if (!box) return;
    setWidth(box.clientWidth);
    const observer = new ResizeObserver(() => setWidth(box.clientWidth));
    observer.observe(box);
    return () => observer.disconnect();
  }, []);
  const count =
    width > 0
      ? Math.max(1, Math.floor((width + BAR_GAP_PX) / (barWidth + BAR_GAP_PX)))
      : 0;
  const levels = peaks ? shapeLevels(resamplePeaks(peaks, count)) : null;
  return (
    <div
      ref={boxRef}
      aria-hidden="true"
      className="flex size-full justify-between"
    >
      {Array.from({ length: count }, (_, index) => {
        const peak = levels ? (levels[index] ?? 0) : 0;
        // Quiet passages keep a visible stub; loud ones reach the top.
        const height = 12 + peak ** 0.9 * 88;
        const at = (index + 0.5) / count;
        return (
          <span
            // biome-ignore lint/suspicious/noArrayIndexKey: bars are positional and never reorder.
            key={index}
            style={{ width: barWidth }}
            className={cn(
              "relative h-full shrink-0",
              isPlayed(at)
                ? "text-foreground"
                : isHovered?.(at)
                  ? "text-foreground opacity-70"
                  : cn(
                      "text-muted-foreground",
                      peaks ? "opacity-50" : "opacity-25",
                    ),
            )}
          >
            <span
              className="absolute inset-x-0 rounded-full bg-current"
              style={{
                top: `${BASELINE * (100 - height)}%`,
                height: `max(2px, ${height}%)`,
              }}
            />
          </span>
        );
      })}
    </div>
  );
}
