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
const BAR_PITCH_PX = 2 + BAR_GAP_PX;
// Fraction of the track height above the baseline.
const BASELINE = 0.7;

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
}) {
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
    <div className={cn("flex items-center gap-2", className)}>
      <Button
        type="button"
        variant="muted"
        size="icon"
        className="size-[calc(32px*var(--ui-space-scale,1))] shrink-0"
        aria-label={playing ? `Pause ${label}` : `Play ${label}`}
        disabled={!src}
        onClick={toggle}
      >
        {playing ? (
          <HugeiconsIcon icon={PauseIcon} className="size-3.5" />
        ) : (
          <PlayIcon className="size-3.5" />
        )}
      </Button>
      {overlay !== undefined || tailS > 0 ? (
        <div className="relative flex h-[calc(32px*var(--ui-space-scale,1))] min-w-0 flex-1">
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
}: {
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
        isPlayed={(at) => fraction > 0 && at <= fraction}
      />
    </div>
  );
}

/** Holds the width of "duration / duration", so starting playback never narrows the bars beside it. */
export function PlaybackTime({
  position,
  duration,
}: {
  position: number;
  duration: number;
}) {
  return (
    <span className="grid shrink-0 justify-items-end font-mono text-ui-11 tabular-nums text-muted-foreground">
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
}: {
  peaks: readonly number[] | null;
  /** `at` is the bar's centre as a 0..1 fraction of the track. */
  isPlayed: (at: number) => boolean;
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
      ? Math.max(1, Math.floor((width + BAR_GAP_PX) / BAR_PITCH_PX))
      : 0;
  const levels = peaks ? resamplePeaks(peaks, count) : null;
  return (
    <div
      ref={boxRef}
      aria-hidden="true"
      className="flex size-full justify-center gap-[2px]"
    >
      {Array.from({ length: count }, (_, index) => {
        const peak = levels ? (levels[index] ?? 0) : 0;
        const height = 10 + peak ** 0.8 * 90;
        return (
          <span
            // biome-ignore lint/suspicious/noArrayIndexKey: bars are positional and never reorder.
            key={index}
            className={cn(
              "relative h-full w-[2px] shrink-0",
              isPlayed((index + 0.5) / count)
                ? "text-foreground"
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
