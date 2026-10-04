// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { PillTabs } from "@/features/model-picker/components/model-selector/pill-tabs";
import { cn } from "@/lib/utils";
import { PauseIcon, PlayIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type Ref, useCallback, useEffect, useRef, useState } from "react";
import { fetchAudioBlob } from "../api";
import {
  type ABSide,
  type ABState,
  INITIAL_AB_STATE,
  clampPosition,
  resumeAfterLoad,
  switchSide,
} from "./ab-compare-state";
import { WAVEFORM_BARS, computePeaks, formatSeconds } from "./waveform-peaks";

const SEEK_STEP_SECONDS = 5;
// Decoding holds every sample in memory; past this the bars stay flat.
const DECODE_MAX_BYTES = 60 * 1024 * 1024;

export interface ABClip {
  src: string | null;
  /** Server URL to draw bars from: the page CSP blocks fetching an object URL. */
  fileUrl?: string | null;
  label: string;
  durationS: number | null;
}

async function decodeClipPeaks(
  src: string,
  signal: AbortSignal,
): Promise<number[] | null> {
  if (src.startsWith("blob:")) return null;
  const blob = src.startsWith("data:")
    ? await (await fetch(src, { signal })).blob()
    : await fetchAudioBlob(src, signal);
  if (blob.size > DECODE_MAX_BYTES) return null;
  const Offline =
    window.OfflineAudioContext ||
    (
      window as unknown as {
        webkitOfflineAudioContext?: typeof OfflineAudioContext;
      }
    ).webkitOfflineAudioContext;
  if (!Offline) return null;
  const context = new Offline(1, 1, 22050);
  const buffer = await context.decodeAudioData(await blob.arrayBuffer());
  return computePeaks(
    Array.from({ length: buffer.numberOfChannels }, (_, index) =>
      buffer.getChannelData(index),
    ),
  );
}

function useClipPeaks(srcs: readonly (string | null)[]) {
  const [peaks, setPeaks] = useState<Record<string, number[] | null>>({});
  const key = srcs.join("\n");
  // biome-ignore lint/correctness/useExhaustiveDependencies: `key` is the srcs, joined.
  useEffect(() => {
    const controller = new AbortController();
    for (const src of srcs) {
      if (!src) continue;
      decodeClipPeaks(src, controller.signal)
        .then((bars) => {
          if (!controller.signal.aborted)
            setPeaks((prev) => ({ ...prev, [src]: bars }));
        })
        .catch(() => {
          if (!controller.signal.aborted)
            setPeaks((prev) => ({ ...prev, [src]: null }));
        });
    }
    return () => controller.abort();
  }, [key]);
  return peaks;
}

/** Original | Edited on one player; switching sides keeps the moment and the playing state. */
export function ABCompare({
  original,
  edited,
  autoFocusRef,
  className,
}: {
  original: ABClip;
  edited: ABClip;
  autoFocusRef?: Ref<HTMLButtonElement>;
  className?: string;
}) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [ab, setAb] = useState<ABState>(INITIAL_AB_STATE);
  const [mediaDuration, setMediaDuration] = useState<
    Partial<Record<ABSide, number>>
  >({});
  // Set while the element loads the other side, so its pause/time events are not the user's.
  const pendingRef = useRef<ABState | null>(null);
  const originalBarsSrc = original.fileUrl ?? original.src;
  const editedBarsSrc = edited.fileUrl ?? edited.src;
  const peaks = useClipPeaks([originalBarsSrc, editedBarsSrc]);

  const clips: Record<ABSide, ABClip> = { original, edited };
  const current = clips[ab.side];
  const durationOf = (side: ABSide): number | null => {
    const fromProps = clips[side].durationS;
    return fromProps && fromProps > 0
      ? fromProps
      : (mediaDuration[side] ?? null);
  };
  const duration = durationOf(ab.side) ?? 0;
  const fraction = duration > 0 ? Math.min(1, ab.position / duration) : 0;
  const barsSrc = ab.side === "original" ? originalBarsSrc : editedBarsSrc;
  const bars = barsSrc ? (peaks[barsSrc] ?? null) : null;
  const count = bars?.length ?? WAVEFORM_BARS;
  const src = current.src;

  // A new pair resets the player; the Original's object URL arriving late does not.
  const originalKey = original.fileUrl ?? original.src;
  const editedKey = edited.fileUrl ?? edited.src;
  // biome-ignore lint/correctness/useExhaustiveDependencies: the clip keys are the trigger, not inputs.
  useEffect(() => {
    pendingRef.current = null;
    setAb(INITIAL_AB_STATE);
    setMediaDuration({});
  }, [originalKey, editedKey]);

  const toggle = useCallback(() => {
    const audio = audioRef.current;
    if (!(audio && src)) return;
    if (audio.paused) {
      audio.play().catch(() => setAb((s) => ({ ...s, playing: false })));
    } else {
      audio.pause();
    }
  }, [src]);

  const seekTo = useCallback(
    (seconds: number) => {
      const audio = audioRef.current;
      if (!(audio && src) || duration <= 0) return;
      const next = clampPosition(seconds, duration);
      audio.currentTime = next;
      setAb((s) => ({ ...s, position: next }));
    },
    [src, duration],
  );

  const choose = (side: ABSide) => {
    if (side === ab.side) return;
    const audio = audioRef.current;
    const now: ABState = {
      ...ab,
      position: audio ? audio.currentTime : ab.position,
      playing: audio ? !audio.paused : ab.playing,
    };
    const next = switchSide(now, side, durationOf(side));
    pendingRef.current = clips[side].src ? next : null;
    // A side still loading has nothing playing yet.
    setAb(clips[side].src ? next : { ...next, playing: false });
  };

  return (
    <div className={cn("grid gap-2", className)}>
      <PillTabs
        ariaLabel="Compare"
        value={ab.side}
        onValueChange={(side) => choose(side as ABSide)}
        fit={true}
        compact={true}
        className="[&>button]:px-3"
        tabs={[
          { value: "original", label: "Original" },
          { value: "edited", label: "Edited" },
        ]}
      />
      <div className="flex items-center gap-2">
        <Button
          ref={autoFocusRef}
          type="button"
          variant="muted"
          size="icon"
          className="size-[calc(32px*var(--ui-space-scale,1))] shrink-0"
          aria-label={
            ab.playing ? `Pause ${current.label}` : `Play ${current.label}`
          }
          disabled={!src}
          onClick={toggle}
        >
          <HugeiconsIcon
            icon={ab.playing ? PauseIcon : PlayIcon}
            className="size-3.5"
          />
        </Button>
        <div
          role="slider"
          tabIndex={src ? 0 : -1}
          aria-label={`${current.label} playback position`}
          aria-valuemin={0}
          aria-valuemax={Math.round(duration)}
          aria-valuenow={Math.round(ab.position)}
          aria-valuetext={`${formatSeconds(ab.position)} of ${formatSeconds(duration)}`}
          aria-disabled={!src}
          className="relative h-[calc(32px*var(--ui-space-scale,1))] min-w-0 flex-1 cursor-pointer rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          onKeyDown={(event) => {
            if (event.key === " ") {
              event.preventDefault();
              toggle();
            } else if (event.key === "ArrowRight") {
              event.preventDefault();
              seekTo(ab.position + SEEK_STEP_SECONDS);
            } else if (event.key === "ArrowLeft") {
              event.preventDefault();
              seekTo(ab.position - SEEK_STEP_SECONDS);
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
              // A floor keeps silence visible as a line rather than a gap.
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
                    played && ab.position > 0
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
          {ab.position > 0 ? `${formatSeconds(ab.position)} / ` : ""}
          {formatSeconds(duration)}
        </span>
      </div>
      {src ? (
        // biome-ignore lint/a11y/useMediaCaption: speech clips have no caption track; the clip's text is shown beside it.
        <audio
          ref={audioRef}
          src={src}
          preload="metadata"
          className="hidden"
          onPlay={() => {
            if (!pendingRef.current) setAb((s) => ({ ...s, playing: true }));
          }}
          onPause={() => {
            if (!pendingRef.current) setAb((s) => ({ ...s, playing: false }));
          }}
          onEnded={() => setAb((s) => ({ ...s, playing: false, position: 0 }))}
          onTimeUpdate={(event) => {
            if (pendingRef.current) return;
            const time = event.currentTarget.currentTime;
            setAb((s) => ({ ...s, position: time }));
          }}
          onLoadedMetadata={(event) => {
            const audio = event.currentTarget;
            const length = Number.isFinite(audio.duration)
              ? audio.duration
              : null;
            const side = ab.side;
            if (length !== null)
              setMediaDuration((prev) => ({ ...prev, [side]: length }));
            const pending = pendingRef.current;
            if (!pending) return;
            pendingRef.current = null;
            const { time, play } = resumeAfterLoad(pending, length);
            audio.currentTime = time;
            setAb((s) => ({ ...s, position: time, playing: play }));
            if (play)
              audio
                .play()
                .catch(() => setAb((s) => ({ ...s, playing: false })));
          }}
        />
      ) : null}
    </div>
  );
}
