// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { PauseIcon, PlayIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type Ref,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import { WAVEFORM_BARS, formatSeconds } from "./waveform-peaks";

const SEEK_STEP_SECONDS = 5;

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
  controlRef,
  onPositionChange,
}: {
  /** 0..1; null draws a flat placeholder while decoding. */
  peaks: readonly number[] | null;
  durationS: number | null;
  src: string | null;
  label: string;
  className?: string;
  controlRef?: Ref<WaveformControl>;
  onPositionChange?: (seconds: number, playing: boolean) => void;
}) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [position, setPosition] = useState(0);
  const [mediaDuration, setMediaDuration] = useState<number | null>(null);
  const duration =
    durationS && durationS > 0 ? durationS : (mediaDuration ?? 0);
  const fraction = duration > 0 ? Math.min(1, position / duration) : 0;
  const bars = peaks && peaks.length > 0 ? peaks : null;
  const count = bars?.length ?? WAVEFORM_BARS;

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
        <HugeiconsIcon
          icon={playing ? PauseIcon : PlayIcon}
          className="size-3.5"
        />
      </Button>
      <div
        role="slider"
        tabIndex={src ? 0 : -1}
        aria-label={`${label} playback position`}
        aria-valuemin={0}
        aria-valuemax={Math.round(duration)}
        aria-valuenow={Math.round(position)}
        aria-valuetext={`${formatSeconds(position)} of ${formatSeconds(duration)}`}
        aria-disabled={!src}
        className="relative h-[calc(32px*var(--ui-space-scale,1))] min-w-0 flex-1 cursor-pointer rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
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
        onClick={(event) => {
          const box = event.currentTarget.getBoundingClientRect();
          if (box.width <= 0) return;
          seekTo(((event.clientX - box.left) / box.width) * duration);
        }}
      >
        <svg
          aria-hidden="true"
          className="size-full"
          viewBox={`0 0 ${count * 3} 32`}
          preserveAspectRatio="none"
        >
          {Array.from({ length: count }, (_, index) => {
            const peak = bars ? (bars[index] ?? 0) : 0;
            const height = Math.max(2, peak * 30);
            const played = (index + 0.5) / count <= fraction;
            return (
              <rect
                // biome-ignore lint/suspicious/noArrayIndexKey: bars are positional and never reorder.
                key={index}
                x={index * 3 + 0.5}
                y={(32 - height) / 2}
                width={2}
                height={height}
                rx={1}
                className={cn(
                  "fill-current",
                  played && position > 0
                    ? "text-foreground"
                    : "text-muted-foreground",
                  bars ? "opacity-100" : "opacity-40",
                )}
              />
            );
          })}
        </svg>
      </div>
      <span className="shrink-0 font-mono text-ui-11p5 tabular-nums text-muted-foreground">
        {position > 0 ? `${formatSeconds(position)} / ` : ""}
        {formatSeconds(duration)}
      </span>
      {src ? (
        // biome-ignore lint/a11y/useMediaCaption: voice references have no caption track; the clip's text is shown beside it.
        <audio
          ref={audioRef}
          src={src}
          preload="metadata"
          className="hidden"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
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
