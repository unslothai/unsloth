// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { PillTabs } from "@/features/model-picker/components/model-selector/pill-tabs";
import { type Ref, useCallback, useRef } from "react";
import { type ABPlayback, nextPlayback } from "./ab-compare-state";

export interface ABSide {
  label: string;
  src?: string;
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
  playerRef?: Ref<HTMLAudioElement>;
}) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const setAudio = useCallback(
    (element: HTMLAudioElement | null) => {
      audioRef.current = element;
      if (typeof playerRef === "function") playerRef(element);
      else if (playerRef) playerRef.current = element;
    },
    [playerRef],
  );
  const pendingRef = useRef<ABPlayback | null>(null);

  const shownSide: ABSideId =
    side === "a" && a.unavailable && !b.unavailable
      ? "b"
      : side === "b" && b.unavailable && !a.unavailable
        ? "a"
        : side;
  const current = shownSide === "a" ? a : b;
  const src = current.unavailable ? undefined : current.src;

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
    const pending = pendingRef.current;
    if (!audio || !pending) return;
    pendingRef.current = null;
    const { seek, resume } = nextPlayback(pending, audio.duration);
    if (seek > 0) audio.currentTime = seek;
    if (resume) {
      // Autoplay rules may refuse.
      audio.play().catch(() => undefined);
    }
  }, []);

  const reason = a.unavailable ?? b.unavailable;
  return (
    <div className="grid w-full gap-2">
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
      {src ? (
        // biome-ignore lint/a11y/useMediaCaption: generated speech has no caption track; the clip's text is shown above it.
        <audio
          ref={setAudio}
          controls={true}
          src={src}
          preload="metadata"
          onLoadedMetadata={handleLoadedMetadata}
          className="w-full rounded-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        />
      ) : (
        <output className="flex h-12 w-full items-center justify-center rounded-md border border-border text-ui-12 text-muted-foreground">
          {current.unavailable ?? "Loading audio…"}
        </output>
      )}
      {reason && !current.unavailable ? (
        <p className="text-ui-11p5 leading-snug text-muted-foreground">
          {reason}
        </p>
      ) : null}
    </div>
  );
}
