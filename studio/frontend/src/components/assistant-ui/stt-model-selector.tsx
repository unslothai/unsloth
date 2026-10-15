// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { MicVocalIcon } from "lucide-react";
import { useState, type FC } from "react";
import { sttModelName } from "@/features/settings/stores/voice-settings-store";
import { DotTag } from "@/features/hub/catalog/dot-tag";
import { formatBytes } from "@/features/hub/lib/format";
import type { LoraModelOption } from "@/features/model-picker/components/model-selector/types";

interface SttModelSelectorProps {
  /** On-device transcription models available to pick. */
  models: LoraModelOption[];
  /** The transcription model the loop listens with. */
  value: string;
  onValueChange: (id: string) => void;
  /** When true (no main chat model loaded), grey out and block the selector:
   *  listening is pointless with no brain to generate replies. */
  disabled?: boolean;
  /** True when the STT model is loaded and ready (after warmup), so the listen
   *  icon goes green instead of grey. */
  ready?: boolean;
  /** True while the STT model warms up, so the listen icon shows amber (loading)
   *  rather than grey (idle). Takes precedence over `ready`. */
  loading?: boolean;
  className?: string;
}

// Voice mode's "Listen with" picker. Its own pick, not Settings > Voice's: that one
// belongs to the Dictate button. A model not on disk downloads on first use.
export const SttModelSelector: FC<SttModelSelectorProps> = ({
  models,
  value,
  onValueChange,
  disabled = false,
  ready = false,
  loading = false,
  className,
}) => {
  const [open, setOpen] = useState(false);
  const displayName = models.find((m) => m.id === value)?.name ?? sttModelName(value);

  const handleSelect = (id: string) => {
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
          aria-label={disabled ? "Select a chat model first" : "Select listening engine"}
        >
          <MicVocalIcon
            className={cn(
              "size-3.5 shrink-0",
              loading
                ? "text-amber-500"
                : ready && !disabled
                  ? "text-emerald-500"
                  : "text-muted-foreground",
            )}
          />
          <span className="min-w-0 truncate font-heading text-ui-16 font-medium leading-tight text-black dark:text-white">
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
        className="unsloth-model-selector-menu menu-soft-surface w-85 gap-0 rounded-lg border-0 p-1.5 ring-0"
      >
        <div className="px-2 pb-1 pt-0.5 text-ui-11 font-semibold uppercase tracking-wide text-muted-foreground">
          Listen with
        </div>

        {models.map((model) => {
          const name = model.name;
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
                {model.deviceSizeBytes != null && (
                  <span className="shrink-0 text-ui-11 text-muted-foreground">
                    {formatBytes(model.deviceSizeBytes)}
                  </span>
                )}
                {!model.source && (
                  <DotTag
                    tone={model.isGguf ? "gguf" : "checkpoint"}
                    label={model.isGguf ? "GGUF" : "Safetensors"}
                    className="h-4.5 gap-1 rounded-md px-1.5"
                    dotClassName="size-1.25"
                  />
                )}
                {value === model.id && (
                  <DotTag
                    tone="success"
                    label="Active"
                    className="h-4.5 gap-1 rounded-md px-1.5"
                    dotClassName="size-1.25"
                  />
                )}
              </span>
            </button>
          );
        })}

        {models.length === 0 && (
          <p className="px-3 py-2 text-ui-12 text-muted-foreground">
            No transcription models on device. Download one from the model
            dropdown.
          </p>
        )}
      </PopoverContent>
    </Popover>
  );
};
