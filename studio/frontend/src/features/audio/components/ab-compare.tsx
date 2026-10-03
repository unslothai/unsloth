// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One player for two versions of a clip (before and after a conversion or an edit). A toggle swaps
// what it plays and keeps the moment and whether it was playing, so the listener hears the change.

import { PillTabs } from "@/features/model-picker/components/model-selector/pill-tabs";
import { type Ref, useCallback, useEffect, useRef } from "react";
import { type ABPlayback, nextPlayback } from "./ab-compare-state";

export interface ABSide {
  label: string;
  /** The side's playable URL; absent while it loads or when it is unavailable. */
  src?: string;
  /** Why this side cannot play ("Source no longer available"); its tab is then disabled. */
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
  autoFocus = false,
}: {
  a: ABSide;
  b: ABSide;
  side: ABSideId;
  onSideChange: (side: ABSideId) => void;
  /** Names the toggle, e.g. "Compare source and converted". */
  ariaLabel: string;
  playerRef?: Ref<HTMLAudioElement>;
  /** Focus the player once it mounts, for a fresh result. */
  autoFocus?: boolean;
}) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  // Hands the one player to the page too (to focus a fresh result), also when it remounts.
  const setAudio = useCallback(
    (element: HTMLAudioElement | null) => {
      audioRef.current = element;
      if (typeof playerRef === "function") playerRef(element);
      else if (playerRef) playerRef.current = element;
    },
    [playerRef],
  );
  // The moment to carry over, captured when the side changes and applied on the new metadata.
  const pendingRef = useRef<ABPlayback | null>(null);

  // A side that cannot play is never the one shown when the other can.
  const shownSide: ABSideId =
    side === "a" && a.unavailable && !b.unavailable
      ? "b"
      : side === "b" && b.unavailable && !a.unavailable
        ? "a"
        : side;
  const current = shownSide === "a" ? a : b;
  const src = current.unavailable ? undefined : current.src;

  useEffect(() => {
    if (autoFocus) audioRef.current?.focus();
  }, [autoFocus]);

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
      // Autoplay rules may refuse; the listener can still press play.
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
          // The focus ring follows the player's pill instead of boxing it.
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
