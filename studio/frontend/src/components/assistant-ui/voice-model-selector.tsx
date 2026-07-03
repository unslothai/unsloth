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
import { Speech } from "lucide-react";
import { useState, type FC } from "react";
import { useChatRuntimeStore } from "@/features/chat";
import { DotTag } from "@/features/hub/catalog/dot-tag";
import { formatBytes } from "@/features/hub/lib/format";
import { GgufVariantExpander } from "./model-selector/pickers";
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
  /** True only when the backend voice slot is actually loaded with this voice
   *  (from /voice/status), so the speak icon goes green. A mere selection is not
   *  enough -- it persists but doesn't load the slot until voice mode starts. */
  loaded?: boolean;
  className?: string;
}

const BROWSER_VOICE_ID = null;
const BROWSER_VOICE_LABEL = "Browser voice";

// How many sentence chunks a GGUF voice slot synthesizes at once (llama-server
// --parallel N). If a GGUF voice is loaded, changing it hot-reloads that slot so
// backend and UI stay in sync. In-process voices (Qwen3-TTS) ignore it.
const ParallelVoicesPicker: FC<{
  reloadVoiceId: string | null;
  onReload: (id: string) => void;
}> = ({ reloadVoiceId, onReload }) => {
  const value = useChatRuntimeStore((s) => s.voiceParallelN);
  const setValue = useChatRuntimeStore((s) => s.setVoiceParallelN);
  return (
    <>
      <div className="my-1.5 h-px bg-black/[0.08] dark:bg-white/[0.08]" />
      <div className="flex items-center justify-between gap-2 px-2 py-1">
        <span className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
          Parallel synthesis
        </span>
        <div className="flex items-center gap-1">
          {[1, 2, 3, 4].map((n) => (
            <button
              key={n}
              type="button"
              onClick={() => {
                if (n === value) return;
                setValue(n);
                // Hot-reload the active GGUF voice so its --parallel matches now.
                if (reloadVoiceId) onReload(reloadVoiceId);
              }}
              className={cn(
                "flex size-6 items-center justify-center rounded-md text-[13px] transition-colors",
                value === n
                  ? "bg-[#ececec] text-foreground dark:bg-[var(--sidebar-accent)]"
                  : "text-muted-foreground hover:bg-[#ececec] dark:hover:bg-[var(--sidebar-accent)]",
              )}
              aria-label={`${n} parallel voice${n > 1 ? "s" : ""}`}
            >
              {n}
            </button>
          ))}
        </div>
      </div>
      <p className="px-2 pb-1 text-[10px] leading-tight text-muted-foreground/70">
        GGUF voices only.
      </p>
    </>
  );
};

export const VoiceModelSelector: FC<VoiceModelSelectorProps> = ({
  models,
  value,
  onValueChange,
  loading = false,
  disabled = false,
  voiceOwnedByModel = false,
  loaded = false,
  className,
}) => {
  const [open, setOpen] = useState(false);
  // Which GGUF voice row is expanded to show its quant list.
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const setSelectedVoiceVariant = useChatRuntimeStore(
    (s) => s.setSelectedVoiceVariant,
  );

  const selectedVoiceVariant = useChatRuntimeStore((s) => s.selectedVoiceVariant);
  const selectedModel = value ? models.find((m) => m.id === value) : null;

  // The picker is non-interactive when there's no chat model to answer (disabled)
  // OR the chat model is a speech-LLM that speaks with its own voice
  // (voiceOwnedByModel) -- in both cases it greys out and can't be opened.
  const inactive = disabled || voiceOwnedByModel;
  // Green only when a voice is truly loaded: the backend slot is up with this
  // voice, or the chat model speaks with its own voice. A persisted selection
  // alone (slot not yet loaded, e.g. right after a reload) stays grey.
  const ready = voiceOwnedByModel || loaded;
  // Match the chat-model trigger: bold name + a muted "GGUF · <quant>" suffix,
  // rather than appending the quant in the same weight as the name.
  const nameText = disabled
    ? "Select model first"
    : voiceOwnedByModel
      ? "Model's own voice"
      : selectedModel
        ? selectedModel.name
        : BROWSER_VOICE_LABEL;
  const metaText =
    !inactive && selectedModel?.isGguf && selectedVoiceVariant
      ? `GGUF · ${selectedVoiceVariant}`
      : null;

  const handleSelect = (id: string | null, variant: string | null = null) => {
    setOpen(false);
    setSelectedVoiceVariant(variant);
    onValueChange(id);
  };

  return (
    <Popover
      open={inactive ? false : open}
      onOpenChange={(next) => !inactive && setOpen(next)}
    >
      <PopoverTrigger asChild>
        <button
          type="button"
          disabled={inactive}
          title={
            voiceOwnedByModel
              ? "The loaded model is a speech-LLM and speaks with its own voice."
              : undefined
          }
          className={cn(
            "flex min-w-0 items-center gap-2 rounded-[10px] transition-colors",
            inactive
              ? "cursor-not-allowed opacity-50"
              : "hover:bg-[#ececec] dark:hover:bg-[#2d2e32]",
            "h-9 px-3.5 text-sm",
            className,
          )}
          aria-label={
            disabled
              ? "Select a chat model first"
              : voiceOwnedByModel
                ? "The chat model provides its own voice"
                : "Select voice model"
          }
        >
          {loading && !voiceOwnedByModel ? (
            <Spinner className="size-3.5 shrink-0 text-amber-500" />
          ) : (
            <Speech
              className={cn(
                "size-3.5 shrink-0",
                ready ? "text-emerald-500" : "text-muted-foreground",
              )}
            />
          )}
          <span className="flex min-w-0 items-baseline">
            <span className="min-w-0 truncate font-heading text-[16px] font-medium leading-tight text-black dark:text-white">
              {nameText}
            </span>
            {metaText && (
              <span className="ml-2 shrink-0 text-xs leading-none text-muted-foreground">
                {metaText}
              </span>
            )}
          </span>
          {/* Chevron only when the picker can actually open. */}
          {!inactive && (
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
        className="unsloth-model-selector-menu menu-soft-surface w-[340px] gap-0 rounded-lg border-0 p-1.5 ring-0"
      >
        <div className="px-2 pb-1 pt-0.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
          Speak with
        </div>

        {/* Browser's built-in speech synthesis -- not a downloadable file, so
            it's badged like a cloud/API entry rather than an on-device one. The
            picker only opens for a regular chat model; a speech-LLM greys the
            trigger out (voiceOwnedByModel), so no "own voice" message is needed
            here. Entering the ball is a separate phone button in the composer. */}
        <button
          type="button"
          onClick={() => handleSelect(BROWSER_VOICE_ID)}
          className={cn(
            "flex w-full items-center gap-2 rounded-full px-2 py-1.5 text-left text-sm transition-colors hover:bg-[#ececec] dark:hover:bg-[var(--sidebar-accent)]",
            value === null && "bg-[#ececec] dark:bg-[var(--sidebar-accent)]",
          )}
        >
          <Speech className="size-3.5 shrink-0 text-muted-foreground" />
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
          const isExpanded = expandedId === model.id;
          // GGUF voices expand to the same quant list as the model picker; clicking
          // the row toggles the quants instead of loading the default. Non-GGUF
          // voices (Qwen3-TTS safetensors) load directly on click.
          return (
            <div key={model.id}>
              <button
                type="button"
                onClick={() =>
                  model.isGguf
                    ? setExpandedId(isExpanded ? null : model.id)
                    : handleSelect(model.id)
                }
                className={cn(
                  "flex w-full items-center gap-2 rounded-full px-2 py-1.5 text-left text-sm transition-colors hover:bg-[#ececec] dark:hover:bg-[var(--sidebar-accent)]",
                  value === model.id && "bg-[#ececec] dark:bg-[var(--sidebar-accent)]",
                )}
              >
                <span className="min-w-0 flex-1 truncate">{name}</span>
                <span className="ml-auto flex shrink-0 items-center gap-1.5">
                  {/* GGUF sizes live in the per-quant expander below; show a size
                      tag only for single-file (safetensors) voices here. */}
                  {model.sizeBytes != null && !model.isGguf && (
                    <span className="shrink-0 text-[11px] text-muted-foreground">
                      {formatBytes(model.sizeBytes)}
                    </span>
                  )}
                  {!model.source && (
                    <DotTag
                      tone={model.isGguf ? "gguf" : "checkpoint"}
                      label={model.isGguf ? "GGUF" : "Safetensors"}
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
                  {model.isGguf && (
                    <HugeiconsIcon
                      icon={ArrowDown01Icon}
                      strokeWidth={2}
                      className={cn(
                        "size-3.5 text-muted-foreground transition-transform",
                        isExpanded && "rotate-180",
                      )}
                    />
                  )}
                </span>
              </button>
              {model.isGguf && isExpanded && (
                <div className="max-h-[240px] overflow-y-auto pl-2">
                  <GgufVariantExpander
                    repoId={model.id}
                    onDevice
                    onSelect={(id, meta) =>
                      handleSelect(id, meta.ggufVariant ?? null)
                    }
                  />
                </div>
              )}
            </div>
          );
        })}

        {models.length === 0 && (
          <p className="px-3 py-2 text-[12px] text-muted-foreground">
            No TTS models found. Train or export a voice model first.
          </p>
        )}

        <ParallelVoicesPicker
          reloadVoiceId={value && selectedModel?.isGguf ? value : null}
          onReload={(id) => onValueChange(id)}
        />
      </PopoverContent>
    </Popover>
  );
};
