// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Spinner } from "@/components/ui/spinner";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon, CloudIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { MicVocalIcon, PhoneIcon } from "lucide-react";
import { useState, type FC } from "react";
import { useChatRuntimeStore } from "@/features/chat";
import { DotTag } from "@/features/hub/catalog/dot-tag";
import { splitRepoLabel } from "./model-selector/row-meta";
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
  // "configuring": dropdown is open (or openable) but the voice ball hasn't
  // been entered yet — picking Listen/Speak options here only configures
  // settings. Entering the ball is a separate, explicit action below.
  const voiceMode = useChatRuntimeStore((s) => s.voiceMode);
  const setVoiceMode = useChatRuntimeStore((s) => s.setVoiceMode);

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
        className="menu-soft-surface w-[300px] gap-0 rounded-lg border-0 p-1.5 ring-0"
      >
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
            {/* Browser's built-in speech synthesis -- not a downloadable file,
                so it's badged like a cloud/API entry rather than an on-device one. */}
            <button
              type="button"
              onClick={() => handleSelect(BROWSER_VOICE_ID)}
              className={cn(
                "flex w-full items-center gap-2 rounded-full px-2 py-1.5 text-left text-sm transition-colors hover:bg-[#ececec] dark:hover:bg-[var(--sidebar-accent)]",
                value === null && "bg-[#ececec] dark:bg-[var(--sidebar-accent)]",
              )}
            >
              <MicVocalIcon className="size-3.5 shrink-0 text-muted-foreground" />
              <span className="min-w-0 flex-1 truncate">{BROWSER_VOICE_LABEL}</span>
              <span className="ml-auto flex shrink-0 items-center gap-1.5">
                <HugeiconsIcon
                  icon={CloudIcon}
                  strokeWidth={1.75}
                  className="size-3.5 text-muted-foreground"
                />
                {value === null && (
                  <DotTag
                    tone="success"
                    label="Active"
                    className="h-[18px] gap-1 rounded-md px-1.5"
                    dotClassName="size-[5px]"
                  />
                )}
              </span>
            </button>

            {models.length > 0 && (
              <div className="my-1.5 h-px bg-black/[0.08] dark:bg-white/[0.08]" />
            )}

            {models.map((model) => {
              const { name } = splitRepoLabel(model.id);
              return (
                <button
                  key={model.id}
                  type="button"
                  onClick={() => handleSelect(model.id)}
                  className={cn(
                    "flex w-full items-center gap-2 rounded-full px-2 py-1.5 text-left text-sm transition-colors hover:bg-[#ececec] dark:hover:bg-[var(--sidebar-accent)]",
                    value === model.id && "bg-[#ececec] dark:bg-[var(--sidebar-accent)]",
                  )}
                >
                  <span className="min-w-0 flex-1 truncate">{name}</span>
                  <span className="ml-auto flex shrink-0 items-center gap-1.5">
                    {!model.source && model.isGguf && (
                      <DotTag
                        tone="gguf"
                        label="GGUF"
                        className="h-[18px] gap-1 rounded-md px-1.5"
                        dotClassName="size-[5px]"
                      />
                    )}
                    {value === model.id && (
                      <DotTag
                        tone="success"
                        label="Active"
                        className="h-[18px] gap-1 rounded-md px-1.5"
                        dotClassName="size-[5px]"
                      />
                    )}
                  </span>
                </button>
              );
            })}

            {models.length === 0 && (
              <p className="px-3 py-2 text-[12px] text-muted-foreground">
                No TTS models found. Train or export a voice model first.
              </p>
            )}
          </>
        )}

        {/* Entering the ball is explicit: picking a voice above only
            configures it, so you can leave this open and keep reading/using
            the text chat until you're ready to talk. */}
        {voiceMode === "configuring" && (
          <>
            <div className="my-1 h-px bg-black/[0.08] dark:bg-white/[0.08]" />
            <button
              type="button"
              disabled={loading}
              onClick={() => {
                setOpen(false);
                setVoiceMode("active");
              }}
              className="flex w-full items-center justify-center gap-2 rounded-md px-3 py-2 text-[13px] font-medium text-primary transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
            >
              <PhoneIcon className="size-3.5" />
              Start voice mode
            </button>
          </>
        )}
      </PopoverContent>
    </Popover>
  );
};
