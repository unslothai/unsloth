// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "@/lib/toast";
import { fetchGalleryAudioSignedUrl } from "../api";

/** One shared player for history rows, so a past clip plays without replacing the one in the preview. */
export function useRowPreview(
  srcById: Readonly<Record<string, string>>,
  active: boolean,
) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playingId, setPlayingId] = useState<string | null>(null);

  useEffect(() => {
    const audio = new Audio();
    audioRef.current = audio;
    const onEnded = () => setPlayingId(null);
    // The preview player (or any other on the page) starting stops the row.
    const onOtherPlay = (event: Event) => {
      if (event.target === audio || audio.paused) return;
      audio.pause();
      setPlayingId(null);
    };
    audio.addEventListener("ended", onEnded);
    document.addEventListener("play", onOtherPlay, true);
    return () => {
      audio.pause();
      audio.removeEventListener("ended", onEnded);
      document.removeEventListener("play", onOtherPlay, true);
      audioRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (active) return;
    audioRef.current?.pause();
    setPlayingId(null);
  }, [active]);

  const toggle = useCallback(
    async (id: string) => {
      const audio = audioRef.current;
      if (!audio) return;
      if (playingId === id) {
        audio.pause();
        setPlayingId(null);
        return;
      }
      // Two clips never play over each other.
      for (const element of document.querySelectorAll("audio")) element.pause();
      setPlayingId(id);
      try {
        audio.src = srcById[id] ?? (await fetchGalleryAudioSignedUrl(id));
        await audio.play();
      } catch {
        setPlayingId(null);
        toast.error("Could not play this clip.");
      }
    },
    [playingId, srcById],
  );

  return { playingId, toggle };
}
