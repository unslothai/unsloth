// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "@/lib/toast";
import { fetchGalleryAudioSignedUrl } from "../api";

/** One shared player for history rows, so a past clip plays without replacing the one in the preview. */
export function useRowPreview(
  srcById: Readonly<Record<string, string>>,
  active: boolean,
  shownIds: readonly string[],
) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const requestRef = useRef(0);

  useEffect(() => {
    const audio = new Audio();
    audioRef.current = audio;
    const onEnded = () => setPlayingId(null);
    // The preview player (or any other on the page) starting stops the row.
    const onOtherPlay = (event: Event) => {
      if (event.target === audio) return;
      requestRef.current += 1;
      audio.pause();
      setPlayingId(null);
    };
    audio.addEventListener("ended", onEnded);
    document.addEventListener("play", onOtherPlay, true);
    return () => {
      requestRef.current += 1;
      audio.pause();
      audio.removeEventListener("ended", onEnded);
      document.removeEventListener("play", onOtherPlay, true);
      audioRef.current = null;
    };
  }, []);

  // Off-page, or its row left the list (selected, archived, deleted): nothing would show its Pause.
  const shown = active && (playingId === null || shownIds.includes(playingId));
  useEffect(() => {
    if (shown) return;
    requestRef.current += 1;
    audioRef.current?.pause();
    setPlayingId(null);
  }, [shown]);

  const toggle = useCallback(
    async (id: string) => {
      const audio = audioRef.current;
      if (!audio) return;
      const request = ++requestRef.current;
      if (playingId === id) {
        audio.pause();
        setPlayingId(null);
        return;
      }
      for (const element of document.querySelectorAll("audio")) element.pause();
      audio.pause();
      setPlayingId(id);
      try {
        // A later click, another player or leaving the page during a fetch wins.
        const cached = srcById[id];
        if (cached) {
          audio.src = cached;
          try {
            await audio.play();
            return;
          } catch {
            // A cached signed URL can expire; remint once below.
            if (request !== requestRef.current) return;
          }
        }
        const src = await fetchGalleryAudioSignedUrl(id);
        if (request !== requestRef.current) return;
        audio.src = src;
        await audio.play();
      } catch {
        if (request !== requestRef.current) return;
        setPlayingId(null);
        toast.error("Could not play this clip.");
      }
    },
    [playingId, srcById],
  );

  return { playingId, toggle };
}
