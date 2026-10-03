// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Delete02Icon,
  MoreHorizontalIcon,
  PauseIcon,
  PencilEdit02Icon,
  PlayIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { type AudioVoice, fetchAudioBlob } from "../api";
import { useAudioVoicesStore } from "../stores/audio-voices-store";
import { SaveVoiceDialog } from "./save-voice-dialog";
import { formatSeconds } from "./waveform-peaks";

// Voice clips fetched for playback, by voice id, for the session.
const voiceUrls = new Map<string, string>();

/** The account's saved voices: play one, pick one, rename or delete it. Space on a focused row
 *  plays it; Enter picks it. */
export function VoicePicker({
  selectedId,
  onSelect,
  disabled,
  className,
}: {
  selectedId?: string | null;
  /** Without it the list only manages voices. */
  onSelect?: (voice: AudioVoice) => void;
  disabled?: boolean;
  className?: string;
}) {
  const voices = useAudioVoicesStore((state) => state.voices);
  const loaded = useAudioVoicesStore((state) => state.loaded);
  const loading = useAudioVoicesStore((state) => state.loading);
  const error = useAudioVoicesStore((state) => state.error);
  const refresh = useAudioVoicesStore((state) => state.refresh);
  const rename = useAudioVoicesStore((state) => state.rename);
  const remove = useAudioVoicesStore((state) => state.remove);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [editing, setEditing] = useState<AudioVoice | null>(null);

  useEffect(() => {
    if (!(loaded || loading)) void refresh();
  }, [loaded, loading, refresh]);

  useEffect(
    () => () => {
      audioRef.current?.pause();
    },
    [],
  );

  const togglePlay = useCallback(
    async (voice: AudioVoice) => {
      const audio = audioRef.current;
      if (!audio) return;
      if (playingId === voice.id) {
        audio.pause();
        setPlayingId(null);
        return;
      }
      try {
        let url = voiceUrls.get(voice.id);
        if (!url) {
          url = URL.createObjectURL(await fetchAudioBlob(voice.url));
          voiceUrls.set(voice.id, url);
        }
        audio.src = url;
        await audio.play();
        setPlayingId(voice.id);
      } catch {
        setPlayingId(null);
        toast.error(`Could not play ${voice.name}.`);
      }
    },
    [playingId],
  );

  const handleDelete = useCallback(
    async (voice: AudioVoice) => {
      if (playingId === voice.id) {
        audioRef.current?.pause();
        setPlayingId(null);
      }
      try {
        await remove(voice.id);
        toast.success(`Deleted ${voice.name}.`);
      } catch (reason) {
        toast.error(
          reason instanceof Error
            ? reason.message
            : "Could not delete the voice.",
        );
      }
    },
    [playingId, remove],
  );

  return (
    <div className={cn("grid gap-1", className)}>
      {/* biome-ignore lint/a11y/useMediaCaption: short voice samples with their names listed beside them. */}
      <audio
        ref={audioRef}
        className="hidden"
        onEnded={() => setPlayingId(null)}
        onPause={() => setPlayingId(null)}
      />
      {voices.length === 0 ? (
        <p className="px-1 py-2 text-ui-12 leading-snug text-muted-foreground">
          {error ? (
            <>
              {error}{" "}
              <Button
                type="button"
                variant="link"
                className="h-auto p-0 text-ui-12 font-medium text-foreground"
                onClick={() => void refresh()}
              >
                Try again
              </Button>
            </>
          ) : loading || !loaded ? (
            "Loading saved voices…"
          ) : (
            "No saved voices yet. Add a reference on Clone, then press Save voice."
          )}
        </p>
      ) : (
        <ul className="hover-scrollbar grid max-h-[calc(196px*var(--ui-space-scale,1))] gap-0.5 overflow-y-auto">
          {voices.map((voice) => {
            const selected = voice.id === selectedId;
            const playing = voice.id === playingId;
            return (
              <li
                key={voice.id}
                className={cn(
                  "group flex items-center gap-1 rounded-full pr-1 transition-colors hover:bg-accent",
                  selected && "bg-muted",
                )}
              >
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="size-[calc(28px*var(--ui-space-scale,1))] shrink-0"
                  aria-label={
                    playing ? `Pause ${voice.name}` : `Play ${voice.name}`
                  }
                  onClick={() => void togglePlay(voice)}
                >
                  <HugeiconsIcon
                    icon={playing ? PauseIcon : PlayIcon}
                    className="size-3.5"
                  />
                </Button>
                <button
                  type="button"
                  disabled={disabled || !onSelect}
                  aria-pressed={onSelect ? selected : undefined}
                  className="flex min-w-0 flex-1 items-center gap-2 rounded-full py-1.5 text-left text-ui-13 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-default"
                  onClick={() => onSelect?.(voice)}
                  onKeyDown={(event) => {
                    // Space plays, as on every clip; Enter picks.
                    if (event.key === " ") {
                      event.preventDefault();
                      void togglePlay(voice);
                    }
                  }}
                  onKeyUp={(event) => {
                    if (event.key === " ") event.preventDefault();
                  }}
                >
                  <span className="min-w-0 flex-1 truncate">{voice.name}</span>
                  {voice.language ? (
                    <span className="shrink-0 text-ui-11p5 text-muted-foreground">
                      {voice.language}
                    </span>
                  ) : null}
                  <span className="shrink-0 font-mono text-ui-11p5 tabular-nums text-muted-foreground">
                    {formatSeconds(voice.duration_s)}
                  </span>
                  {selected ? (
                    <HugeiconsIcon
                      icon={Tick02Icon}
                      className="size-3.5 shrink-0 text-foreground"
                      aria-label="Selected"
                    />
                  ) : null}
                </button>
                <DropdownMenu>
                  <DropdownMenuTrigger asChild={true}>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="size-[calc(28px*var(--ui-space-scale,1))] shrink-0"
                      aria-label={`More for ${voice.name}`}
                    >
                      <HugeiconsIcon
                        icon={MoreHorizontalIcon}
                        className="size-3.5"
                      />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onClick={() => setEditing(voice)}>
                      <HugeiconsIcon
                        icon={PencilEdit02Icon}
                        strokeWidth={1.75}
                        className="size-icon"
                      />
                      Rename…
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      variant="destructive"
                      onClick={() => void handleDelete(voice)}
                    >
                      <HugeiconsIcon
                        icon={Delete02Icon}
                        strokeWidth={1.75}
                        className="size-icon"
                      />
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </li>
            );
          })}
        </ul>
      )}
      <SaveVoiceDialog
        open={editing !== null}
        onOpenChange={(open) => {
          if (!open) setEditing(null);
        }}
        mode="edit"
        initial={{
          name: editing?.name ?? "",
          transcript: editing?.transcript ?? "",
          language: editing?.language ?? "",
        }}
        onSubmit={async (details) => {
          if (!editing) return;
          await rename(editing.id, {
            name: details.name,
            transcript: details.transcript || null,
            language: details.language || null,
          });
        }}
      />
    </div>
  );
}
