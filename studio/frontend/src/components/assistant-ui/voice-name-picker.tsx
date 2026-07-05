// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { cn } from "@/lib/utils";
import { CheckIcon, ChevronDownIcon } from "lucide-react";

// orpheus-3b-0.1-ft speakers. Orpheus randomizes the voice unless one is pinned,
// so this picker lets the user choose which built-in speaker the TTS uses.
const ORPHEUS_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"];
const cap = (v: string) => v.charAt(0).toUpperCase() + v.slice(1);

/** Speaker picker for the Orpheus voice slot. Renders only when an Orpheus voice
 *  is selected -- other TTS models (Qwen3-TTS, Spark) pick their voice differently
 *  (premade speaker id or a reference clip), so a named-speaker list doesn't apply.
 *  Uses the themed dropdown so it tracks light/dark like the other top-bar menus. */
export function VoiceNamePicker({ className }: { className?: string }) {
  const voiceModelId = useChatRuntimeStore((s) => s.selectedVoiceModelId);
  const voiceName = useChatRuntimeStore((s) => s.selectedVoiceName);
  const setVoiceName = useChatRuntimeStore((s) => s.setSelectedVoiceName);

  if (!voiceModelId || !voiceModelId.toLowerCase().includes("orpheus")) {
    return null;
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild={true}>
        <button
          type="button"
          aria-label="Voice speaker"
          className={cn(
            "flex h-[34px] shrink-0 items-center gap-1 rounded-lg pl-2.5 pr-1.5",
            "text-[13.5px] text-foreground transition-colors hover:bg-accent/60",
            "focus-visible:outline-none",
            className,
          )}
        >
          <span>{cap(voiceName)}</span>
          <ChevronDownIcon className="size-3.5 text-muted-foreground" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="min-w-[150px]">
        <DropdownMenuLabel className="text-[11px] font-normal text-muted-foreground">
          Orpheus voice
        </DropdownMenuLabel>
        {ORPHEUS_VOICES.map((v) => (
          <DropdownMenuItem
            key={v}
            onSelect={() => setVoiceName(v)}
            className="flex items-center justify-between gap-4"
          >
            <span>{cap(v)}</span>
            {v === voiceName && <CheckIcon className="size-4 text-primary" />}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
