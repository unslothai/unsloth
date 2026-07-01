// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { MicVocalIcon } from "lucide-react";
import { useState, type FC } from "react";
import { useChatRuntimeStore } from "@/features/chat";
import type { LoraModelOption } from "./model-selector/types";

interface VoiceModelSelectorProps {
  models: LoraModelOption[];
  value: string | null;
  onValueChange: (id: string | null) => void;
  loading?: boolean;
  /** When true (no main chat model loaded), grey out and block the selector:
   *  picking a voice would load a TTS model with no brain to generate replies. */
  disabled?: boolean;
  /** When true, the loaded chat model is a speech-LLM (Orpheus etc.) that speaks
   *  with its own voice, so the TTS voice section is disabled. STT stays active. */
  voiceOwnedByModel?: boolean;
  className?: string;
}

const BROWSER_VOICE_ID = null;
const BROWSER_VOICE_LABEL = "Browser voice";

export const VoiceModelSelector: FC<VoiceModelSelectorProps> = ({
  models,
  value,
  onValueChange,
  loading = false,
  disabled = false,
  voiceOwnedByModel = false,
  className,
}) => {
  const [open, setOpen] = useState(false);
  const sttEngine = useChatRuntimeStore((s) => s.sttEngine);
  const setSttEngine = useChatRuntimeStore((s) => s.setSttEngine);

  const selectedModel = value ? models.find((m) => m.id === value) : null;
  const displayName = selectedModel?.name ?? BROWSER_VOICE_LABEL;

  const handleSelect = (id: string | null) => {
    setOpen(false);
    onValueChange(id);
  };

  return (
    <Popover
      open={disabled ? false : open}
      onOpenChange={(next) => !disabled && setOpen(next)}
    >
      <PopoverTrigger asChild>
        <button
          type="button"
          disabled={disabled}
          className={cn(
            "flex min-w-0 items-center gap-2 rounded-[10px] transition-colors",
            disabled
              ? "cursor-not-allowed opacity-50"
              : "hover:bg-[#ececec] dark:hover:bg-[#2d2e32]",
            "h-9 px-3.5 text-sm",
            className,
          )}
          aria-label={disabled ? "Select a chat model first" : "Select voice model"}
        >
          {loading ? (
            <Spinner className="size-3.5 shrink-0 text-muted-foreground" />
          ) : (
            <MicVocalIcon className="size-3.5 shrink-0 text-muted-foreground" />
          )}
          <span className="min-w-0 truncate font-heading text-[16px] font-medium leading-tight text-black dark:text-white">
            {disabled ? "Select model first" : displayName}
          </span>
          <span className="flex size-4 shrink-0 items-center justify-center">
            <HugeiconsIcon
              icon={ArrowDown01Icon}
              className="size-3.5 text-muted-foreground"
              strokeWidth={2}
            />
          </span>
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        sideOffset={6}
        className="menu-soft-surface w-[220px] rounded-lg border-0 p-1.5 ring-0"
      >
        {/* Speech-to-text engine (listening): browser Web Speech vs backend Whisper */}
        <div className="px-2 pb-1 pt-0.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
          Listen with
        </div>
        <div className="mb-1 flex gap-1 px-0.5">
          {(["browser", "whisper"] as const).map((engine) => (
            <button
              key={engine}
              type="button"
              onClick={() => setSttEngine(engine)}
              className={cn(
                "flex-1 rounded-md px-2 py-1.5 text-[12px] font-medium transition-colors hover:bg-accent",
                sttEngine === engine && "bg-accent",
              )}
            >
              {engine === "browser" ? "Browser" : "Whisper"}
            </button>
          ))}
        </div>
        <div className="my-1 h-px bg-black/[0.08] dark:bg-white/[0.08]" />
        <div className="px-2 pb-1 pt-0.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
          Speak with
        </div>

        {voiceOwnedByModel ? (
          <p className="px-3 py-2 text-[12px] leading-[1.5] text-muted-foreground">
            The loaded model is a speech-LLM and speaks with its own voice, so a
            separate TTS voice isn&apos;t needed.
          </p>
        ) : (
          <>
            {/* Browser voice fallback */}
            <button
              type="button"
              onClick={() => handleSelect(BROWSER_VOICE_ID)}
              className={cn(
                "flex w-full items-center gap-2.5 rounded-md px-3 py-2 text-left text-[13px] font-medium leading-[1.4] tracking-nav transition-colors hover:bg-accent",
                value === null && "bg-accent",
              )}
            >
              <MicVocalIcon className="size-3.5 shrink-0 text-muted-foreground" />
              <span className="min-w-0 truncate">{BROWSER_VOICE_LABEL}</span>
              {value === null && (
                <span className="ml-auto text-[11px] text-muted-foreground">
                  active
                </span>
              )}
            </button>

            {models.length > 0 && (
              <div className="my-1.5 h-px bg-black/[0.08] dark:bg-white/[0.08]" />
            )}

            {models.map((model) => (
              <button
                key={model.id}
                type="button"
                onClick={() => handleSelect(model.id)}
                className={cn(
                  "flex w-full items-center gap-2.5 rounded-md px-3 py-2 text-left text-[13px] font-medium leading-[1.4] tracking-nav transition-colors hover:bg-accent",
                  value === model.id && "bg-accent",
                )}
              >
                <span className="min-w-0 flex-1 truncate">{model.name}</span>
                {model.isGguf && !model.source && (
                  <span className="shrink-0 text-[11px] text-muted-foreground">
                    GGUF
                  </span>
                )}
                {value === model.id && (
                  <span className="ml-auto text-[11px] text-muted-foreground">
                    active
                  </span>
                )}
              </button>
            ))}

            {models.length === 0 && (
              <p className="px-3 py-2 text-[12px] text-muted-foreground">
                No TTS models found. Train or export a voice model first.
              </p>
            )}
          </>
        )}
      </PopoverContent>
    </Popover>
  );
};
