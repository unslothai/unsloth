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
import { BlobUrlCache } from "@/lib/blob-url-cache";
import { useEffect, useRef, useState } from "react";
import { type AudioVoice, fetchAudioBlob } from "../api";
import { useAudioVoicesStore } from "../stores/audio-voices-store";
import { SaveVoiceDialog } from "./save-voice-dialog";
import { formatSeconds } from "./waveform-peaks";

// About 20 saved voices (30 s of 24 kHz mono each); older previews re-fetch.
const voiceUrls = new BlobUrlCache(32 * 1024 * 1024);

export function VoicePicker({
  selectedId,
  onSelect,
  onDeselect,
  disabled,
  className,
}: {
  selectedId?: string | null;
  onSelect?: (voice: AudioVoice) => void;
  onDeselect?: () => void;
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
  const previewRequest = useRef(0);
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

  const togglePlay = async (voice: AudioVoice) => {
    const audio = audioRef.current;
    if (!audio) return;
    if (playingId === voice.id) {
      audio.pause();
      setPlayingId(null);
      return;
    }
    const request = ++previewRequest.current;
    try {
      let url = voiceUrls.get(voice.id);
      if (url) {
        voiceUrls.touch(voice.id);
      } else {
        const blob = await fetchAudioBlob(voice.url);
        url = URL.createObjectURL(blob);
        voiceUrls.set(voice.id, url, blob.size);
        voiceUrls.prune([voice.id]);
      }
      // A later preview click wins over this one's slower fetch.
      if (request !== previewRequest.current) return;
      audio.src = url;
      await audio.play();
      setPlayingId(voice.id);
    } catch {
      if (request !== previewRequest.current) return;
      setPlayingId(null);
      toast.error(`Could not play ${voice.name}.`);
    }
  };

  const handleDelete = async (voice: AudioVoice) => {
    if (playingId === voice.id) {
      audioRef.current?.pause();
      setPlayingId(null);
    }
    try {
      await remove(voice.id);
      voiceUrls.delete(voice.id);
      if (voice.id === selectedId) onDeselect?.();
      toast.success(`Deleted ${voice.name}.`);
    } catch (reason) {
      toast.error(
        reason instanceof Error
          ? reason.message
          : "Could not delete the voice.",
      );
    }
  };

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
        <ul className="hover-scrollbar grid min-w-0 max-h-[calc(196px*var(--ui-space-scale,1))] grid-cols-[minmax(0,1fr)] gap-0.5 overflow-y-auto">
          {voices.map((voice) => {
            const selected = voice.id === selectedId;
            const playing = voice.id === playingId;
            return (
              <li
                key={voice.id}
                className={cn(
                  "group flex min-w-0 items-center gap-1 rounded-full pr-1 transition-colors hover:bg-accent",
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
                      disabled={disabled}
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
        voiceId={editing?.id}
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
