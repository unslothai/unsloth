// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Spinner } from "@/components/ui/spinner";
import { audioCppDisplayName } from "@/features/audio/audio-cpp-catalog";
import { useAudioVoicesStore } from "@/features/audio/stores/audio-voices-store";
import { useChatRuntimeStore } from "@/features/chat";
import { VOICE_TTS_MODELS, voiceTtsCanClone } from "@/features/chat/voice/voice-models";
import { DotTag } from "@/features/hub/catalog/dot-tag";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Speech } from "lucide-react";
import { useEffect, useState, type FC } from "react";

interface VoiceModelSelectorProps {
  value: string;
  onValueChange: (id: string) => void;
  loading?: boolean;
  /** The voice is loaded on the backend, not only picked. */
  loaded?: boolean;
  /** No chat model yet: a voice would have nothing to speak for. */
  disabled?: boolean;
  className?: string;
}

const ROW =
  "flex w-full items-center gap-2 rounded-full px-2 py-1.5 text-left text-sm transition-colors hover:bg-[#ececec] dark:hover:bg-[var(--sidebar-accent)]";
const ROW_ACTIVE = "bg-[#ececec] dark:bg-[var(--sidebar-accent)]";
const HEADING =
  "px-2 pb-1 pt-0.5 text-ui-11 font-semibold uppercase tracking-wide text-muted-foreground";

/** Voice mode's "Speak with" picker: an audio.cpp voice, plus a saved Audio voice when it can clone. */
export const VoiceModelSelector: FC<VoiceModelSelectorProps> = ({
  value,
  onValueChange,
  loading = false,
  loaded = false,
  disabled = false,
  className,
}) => {
  const [open, setOpen] = useState(false);
  const savedVoiceId = useChatRuntimeStore((s) => s.selectedVoiceId);
  const setSavedVoiceId = useChatRuntimeStore((s) => s.setSelectedVoiceId);
  const savedVoices = useAudioVoicesStore((s) => s.voices);
  const canClone = voiceTtsCanClone(value);

  useEffect(() => {
    if (open && canClone) void useAudioVoicesStore.getState().refresh();
  }, [open, canClone]);

  const savedName = savedVoices.find((v) => v.id === savedVoiceId)?.name;
  const metaText = canClone && savedName ? savedName : null;

  return (
    <Popover open={disabled ? false : open} onOpenChange={(next) => !disabled && setOpen(next)}>
      <PopoverTrigger asChild>
        <button
          type="button"
          disabled={disabled}
          className={cn(
            "flex min-w-0 items-center gap-2 rounded-[10px] transition-colors",
            disabled ? "cursor-not-allowed opacity-50" : "hover:bg-[#ececec] dark:hover:bg-[#2d2e32]",
            "h-9 px-3.5 text-sm",
            className,
          )}
          aria-label={disabled ? "Select a chat model first" : "Select voice model"}
        >
          {loading ? (
            <Spinner className="size-3.5 shrink-0 text-amber-500" />
          ) : (
            <Speech
              className={cn(
                "size-3.5 shrink-0",
                loaded ? "text-emerald-500" : "text-muted-foreground",
              )}
            />
          )}
          <span className="flex min-w-0 items-baseline">
            <span className="min-w-0 truncate font-heading text-ui-16 font-medium leading-tight text-black dark:text-white">
              {disabled ? "Select model first" : audioCppDisplayName(value)}
            </span>
            {metaText && (
              <span className="ml-2 shrink-0 text-xs leading-none text-muted-foreground">
                {metaText}
              </span>
            )}
          </span>
          {!disabled && (
            <span className="flex size-4 shrink-0 items-center justify-center">
              <HugeiconsIcon
                icon={ArrowDown01Icon}
                className="size-3.5 text-muted-foreground"
                strokeWidth={2}
              />
            </span>
          )}
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        sideOffset={6}
        className="unsloth-model-selector-menu menu-soft-surface max-h-[70vh] w-85 gap-0 overflow-y-auto rounded-lg border-0 p-1.5 ring-0"
      >
        <div className={HEADING}>Speak with</div>
        {VOICE_TTS_MODELS.map((id) => (
          <button
            key={id}
            type="button"
            onClick={() => {
              setOpen(false);
              onValueChange(id);
            }}
            className={cn(ROW, value === id && ROW_ACTIVE)}
          >
            <span className="min-w-0 flex-1 truncate">{audioCppDisplayName(id)}</span>
            {value === id && (
              <DotTag
                tone="success"
                label="Active"
                className="ml-auto h-4.5 gap-1 rounded-md px-1.5"
                dotClassName="size-1.25"
              />
            )}
          </button>
        ))}
        {canClone && (
          <>
            <div className="my-1.5 h-px bg-[rgb(0_0_0_/_calc(0.08*var(--contrast-wash-gain,1)))] dark:bg-[rgb(255_255_255_/_calc(0.08*var(--contrast-wash-gain,1)))]" />
            <div className={HEADING}>Voice</div>
            {[{ id: "", name: "Model's own voice" }, ...savedVoices].map((voice) => (
              <button
                key={voice.id || "model"}
                type="button"
                onClick={() => {
                  setOpen(false);
                  setSavedVoiceId(voice.id);
                }}
                className={cn(ROW, savedVoiceId === voice.id && ROW_ACTIVE)}
              >
                <span className="min-w-0 flex-1 truncate">{voice.name}</span>
              </button>
            ))}
          </>
        )}
      </PopoverContent>
    </Popover>
  );
};
