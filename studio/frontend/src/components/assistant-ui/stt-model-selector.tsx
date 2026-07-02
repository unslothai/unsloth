// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon, CloudIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { MicIcon } from "lucide-react";
import { useEffect, useState, type FC } from "react";
import { useChatRuntimeStore } from "@/features/chat";
import { DotTag } from "@/features/hub/catalog/dot-tag";
import { splitRepoLabel } from "./model-selector/row-meta";
import type { LoraModelOption } from "./model-selector/types";

interface SttModelSelectorProps {
  /** On-device Whisper models available to pick, in addition to Browser. */
  models: LoraModelOption[];
  /** Selected Whisper model id, or null for the browser's built-in engine. */
  value: string | null;
  onValueChange: (id: string | null) => void;
  /** When true (no main chat model loaded), grey out and block the selector:
   *  listening is pointless with no brain to generate replies. */
  disabled?: boolean;
  className?: string;
}

const BROWSER_ENGINE_ID = null;
const BROWSER_ENGINE_LABEL = "Browser";

// Input-device picker: pins the voice-loop mic to a specific device so it
// captures the user's headset mic and not a loopback / "Stereo Mix" / default-
// communications device that mixes in system/app audio (e.g. a Discord call).
// Applies to the Whisper capture path; the browser Web Speech API can't target a
// device, so it's a no-op there. Device labels are only populated once mic
// permission has been granted; before that they read as generic names.
const MicDevicePicker: FC = () => {
  const selectedMicDeviceId = useChatRuntimeStore((s) => s.selectedMicDeviceId);
  const setSelectedMicDeviceId = useChatRuntimeStore((s) => s.setSelectedMicDeviceId);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);

  useEffect(() => {
    let cancelled = false;
    const enumerate = () => {
      navigator.mediaDevices
        ?.enumerateDevices()
        .then((all) => {
          if (cancelled) return;
          setDevices(all.filter((d) => d.kind === "audioinput"));
        })
        .catch(() => {});
    };
    enumerate();
    navigator.mediaDevices?.addEventListener?.("devicechange", enumerate);
    return () => {
      cancelled = true;
      navigator.mediaDevices?.removeEventListener?.("devicechange", enumerate);
    };
  }, []);

  if (devices.length === 0) return null;

  return (
    <>
      <div className="my-1.5 h-px bg-black/[0.08] dark:bg-white/[0.08]" />
      <div className="px-2 pb-1 pt-0.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
        Microphone
      </div>
      <select
        value={selectedMicDeviceId ?? ""}
        onChange={(e) => setSelectedMicDeviceId(e.target.value || null)}
        className="mx-1 mb-1 w-[calc(100%-0.5rem)] rounded-md bg-transparent px-2 py-1.5 text-sm outline-none ring-1 ring-black/[0.12] focus:ring-black/25 dark:ring-white/[0.12] dark:focus:ring-white/25"
      >
        <option value="">Default input</option>
        {devices.map((d, i) => (
          <option key={d.deviceId || i} value={d.deviceId}>
            {d.label || `Microphone ${i + 1}`}
          </option>
        ))}
      </select>
    </>
  );
};

// Speech-to-text picker, split out from the TTS ("Speak with") picker so each
// can grey out independently -- listening stays available even when the
// loaded chat model is a speech-LLM that owns its own output voice. Lists
// on-device Whisper models the same way the TTS picker lists voices;
// downloading new ones happens in the main model dropdown's Hub search, not here.
export const SttModelSelector: FC<SttModelSelectorProps> = ({
  models,
  value,
  onValueChange,
  disabled = false,
  className,
}) => {
  const [open, setOpen] = useState(false);
  const setSttEngine = useChatRuntimeStore((s) => s.setSttEngine);

  const selectedModel = value ? models.find((m) => m.id === value) : null;
  const displayName = selectedModel?.name ?? BROWSER_ENGINE_LABEL;

  const handleSelect = (id: string | null) => {
    setOpen(false);
    setSttEngine(id === BROWSER_ENGINE_ID ? "browser" : "whisper");
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
          <MicIcon className="size-3.5 shrink-0 text-muted-foreground" />
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
          Listen with
        </div>

        {/* Browser's built-in speech recognition -- not a downloadable file,
            so it's badged like a cloud/API entry rather than an on-device one. */}
        <button
          type="button"
          onClick={() => handleSelect(BROWSER_ENGINE_ID)}
          className={cn(
            "flex w-full items-center gap-2 rounded-full px-2 py-1.5 text-left text-sm transition-colors hover:bg-[#ececec] dark:hover:bg-[var(--sidebar-accent)]",
            value === null && "bg-[#ececec] dark:bg-[var(--sidebar-accent)]",
          )}
        >
          <MicIcon className="size-3.5 shrink-0 text-muted-foreground" />
          <span className="min-w-0 flex-1 truncate">{BROWSER_ENGINE_LABEL}</span>
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
            No Whisper models on device. Download one from the model dropdown.
          </p>
        )}

        <MicDevicePicker />
      </PopoverContent>
    </Popover>
  );
};
