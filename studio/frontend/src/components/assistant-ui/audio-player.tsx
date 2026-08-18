// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useShieldedFromDismissingPress } from "@/lib/menu-dismiss";
import { downloadUrl, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { Download01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { PauseIcon, PlayIcon } from "lucide-react";
import { type FC, useRef, useState } from "react";

interface AudioPlayerProps {
  src: string;
}

export const AudioPlayer: FC<AudioPlayerProps> = ({ src }) => {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);

  // A native <input type="range"> COMMITS ON POINTERDOWN: pressing the track moves the thumb and
  // fires `input` there and then, so the press that dismisses a non-modal menu has already
  // seeked the audio by the time the click swallower in lib/menu-dismiss.ts runs. Same shape as
  // Radix Slider, and the same answer: out of the hit test for exactly as long as such a menu is
  // open. Measured on chromium with the composer "+" menu open, one press on the visible
  // scrubber: currentTime 0 -> 4.08 s, read BEFORE the release.
  const shielded = useShieldedFromDismissingPress();

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleTimeUpdate = () => {
    const audio = audioRef.current;
    if (!audio) return;
    setProgress(audio.currentTime);
  };

  const handleLoadedMetadata = () => {
    const audio = audioRef.current;
    if (!audio) return;
    setDuration(audio.duration);
  };

  const handleEnded = () => {
    setIsPlaying(false);
    setProgress(0);
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    if (!audio) return;
    const time = parseFloat(e.target.value);
    audio.currentTime = time;
    setProgress(time);
  };

  const handleDownload = () => {
    void downloadUrl(src, "generated-audio.wav").catch((error) => {
      if (!isDownloadCancelled(error)) {
        toast.error("Could not save audio.");
      }
    });
  };

  const formatTime = (t: number) => {
    const mins = Math.floor(t / 60);
    const secs = Math.floor(t % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="my-2 flex max-w-md items-center gap-3 rounded-xl border bg-muted/50 px-4 py-3">
      <audio
        ref={audioRef}
        src={src}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={handleEnded}
        preload="metadata"
      />
      <Button
        variant="ghost"
        size="icon"
        className="size-8 shrink-0 rounded-full"
        onClick={togglePlay}
      >
        {isPlaying ? (
          <PauseIcon className="size-4" />
        ) : (
          <PlayIcon className="size-4" />
        )}
      </Button>
      <div className="flex flex-1 flex-col gap-1">
        <input
          type="range"
          min={0}
          max={duration || 0}
          step={0.01}
          value={progress}
          onChange={handleSeek}
          style={shielded ? { pointerEvents: "none" } : undefined}
          className={cn(
            // Marks the control for the static popper exception in index.css, so a scrubber
            // that ever lives INSIDE an open menu stays the user's to press.
            "pointerdown-commits",
            "h-1.5 w-full cursor-pointer accent-primary",
          )}
        />
        <div className="flex justify-between text-ui-10 text-muted-foreground">
          <span>{formatTime(progress)}</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>
      <Button
        variant="ghost"
        size="icon"
        className="size-7 shrink-0 text-muted-foreground"
        onClick={handleDownload}
        title="Download audio"
      >
        <HugeiconsIcon icon={Download01Icon} className="size-3.5" />
      </Button>
    </div>
  );
};
