// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Spinner } from "@/components/ui/spinner";
import { usePlatformStore } from "@/config/env";
import { audioModelsForTask } from "@/features/audio/catalog";
import { VoicePicker } from "@/features/audio/components/voice-picker";
import { speakPageModels } from "@/features/audio/pages/speak-page";
import { audioRowMatchesWorkflow } from "@/features/audio/picker-filter";
import { useAudioVoicesStore } from "@/features/audio/stores/audio-voices-store";
import { useChatRuntimeStore } from "@/features/chat";
import {
  VOICE_TTS_MODELS,
  voiceTtsCanClone,
} from "@/features/chat/voice/voice-models";
import { ModelSelector } from "@/features/model-picker";
import { AUDIO_CATALOG } from "@/features/model-picker/components/model-selector/model-catalog";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type FC, useEffect, useMemo, useState } from "react";

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

const VOICE_TTS_RANK = new Map(
  VOICE_TTS_MODELS.map((id, i) => [id.toLowerCase(), i]),
);
const TTS_TASKS = ["text-to-speech", "text-to-audio"] as const;
const speakRow = (row: Parameters<typeof audioRowMatchesWorkflow>[0]) =>
  audioRowMatchesWorkflow(row, "speak");

/**
 * Voice mode's "Speak with" picker: the Audio tab's model picker, narrowed to the audio.cpp
 * voices the loop can load, plus the saved Audio voice when the pick can clone.
 */
export const VoiceModelSelector: FC<VoiceModelSelectorProps> = ({
  value,
  onValueChange,
  loading = false,
  loaded = false,
  disabled = false,
  className,
}) => {
  const [open, setOpen] = useState(false);
  const [voiceOpen, setVoiceOpen] = useState(false);
  const isMac = usePlatformStore((s) => s.deviceType) === "mac";
  const savedVoiceId = useChatRuntimeStore((s) => s.selectedVoiceId);
  const setSavedVoiceId = useChatRuntimeStore((s) => s.setSelectedVoiceId);
  const savedVoices = useAudioVoicesStore((s) => s.voices);
  const canClone = voiceTtsCanClone(value);

  const models = useMemo(
    () =>
      speakPageModels(audioModelsForTask("tts"), isMac)
        .filter((m) => VOICE_TTS_RANK.has(m.id.toLowerCase()))
        .sort(
          (a, b) =>
            (VOICE_TTS_RANK.get(a.id.toLowerCase()) ?? 0) -
            (VOICE_TTS_RANK.get(b.id.toLowerCase()) ?? 0),
        ),
    [isMac],
  );

  useEffect(() => {
    if (voiceOpen) void useAudioVoicesStore.getState().refresh();
  }, [voiceOpen]);

  const savedName = savedVoices.find((v) => v.id === savedVoiceId)?.name;

  return (
    <div
      className={cn(
        "flex min-w-0 items-center gap-1",
        disabled && "pointer-events-none opacity-50",
      )}
      aria-disabled={disabled || undefined}
      title={disabled ? "Select a chat model first" : undefined}
    >
      {loading && <Spinner className="size-3.5 shrink-0 text-amber-500" />}
      <ModelSelector
        models={models}
        rowFilter={speakRow}
        value={value}
        loaded={loaded}
        loadedModelIdOverride={loaded ? value : undefined}
        onValueChange={(id) => {
          if (!VOICE_TTS_RANK.has(id.toLowerCase())) {
            toast.info("Voice mode speaks through the GGUF audio voices only.");
            return;
          }
          onValueChange(id);
        }}
        variant="ghost"
        className={cn("!h-8.5", className)}
        task={[...TTS_TASKS]}
        catalog={AUDIO_CATALOG}
        placeholder="Select voice"
        open={!disabled && open}
        onOpenChange={(next) => setOpen(!disabled && next)}
      />
      {canClone && (
        <Popover
          open={!disabled && voiceOpen}
          onOpenChange={(next) => setVoiceOpen(!disabled && next)}
        >
          <PopoverTrigger asChild={true}>
            <button
              type="button"
              className={cn(
                "flex h-8.5 min-w-0 items-center gap-1.5 rounded-[10px] px-3 text-sm transition-colors hover:bg-nav-surface-hover",
                className,
              )}
              aria-label="Choose a saved voice"
            >
              <span className="min-w-0 truncate text-ui-14 text-muted-foreground">
                {savedName ?? "Model's own voice"}
              </span>
              <HugeiconsIcon
                icon={ArrowDown01Icon}
                className="size-3.5 shrink-0 text-muted-foreground"
                strokeWidth={2}
              />
            </button>
          </PopoverTrigger>
          <PopoverContent
            align="start"
            sideOffset={6}
            className="unsloth-model-selector-menu menu-soft-surface w-85 gap-0 rounded-lg border-0 p-1.5 ring-0"
          >
            <div className="px-2 pb-1 pt-0.5 text-ui-11 font-semibold uppercase tracking-wide text-muted-foreground">
              Voice
            </div>
            <button
              type="button"
              onClick={() => {
                setSavedVoiceId("");
                setVoiceOpen(false);
              }}
              className={cn(
                "mb-0.5 flex w-full items-center rounded-full px-2 py-1.5 text-left text-ui-13 transition-colors hover:bg-accent",
                !savedVoiceId && "bg-muted",
              )}
            >
              Model's own voice
            </button>
            <VoicePicker
              selectedId={savedVoiceId || null}
              onSelect={(voice) => {
                setSavedVoiceId(voice.id);
                setVoiceOpen(false);
              }}
              onDeselect={() => setSavedVoiceId("")}
            />
          </PopoverContent>
        </Popover>
      )}
    </div>
  );
};
