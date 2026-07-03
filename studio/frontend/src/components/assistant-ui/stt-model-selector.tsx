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
import { MicIcon, MicVocalIcon } from "lucide-react";
import { useEffect, useState, type FC } from "react";
import { useChatRuntimeStore } from "@/features/chat";
import { DotTag } from "@/features/hub/catalog/dot-tag";
import { formatBytes } from "@/features/hub/lib/format";
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
  /** True when the STT engine is loaded and ready (browser always; Whisper after
   *  warmup), so the listen icon goes green instead of grey. */
  ready?: boolean;
  /** True while Whisper is warming up, so the listen icon shows amber (loading)
   *  rather than grey (idle). Takes precedence over `ready`. */
  loading?: boolean;
  className?: string;
}

const BROWSER_ENGINE_ID = null;
const BROWSER_ENGINE_LABEL = "Browser";

// Input-device picker: pins the voice-loop mic to a specific device so it
// captures the user's headset mic and not a loopback / "Stereo Mix" / default-
// communications device that mixes in system/app audio (e.g. a Discord call).
// Applies to both the Whisper and browser capture paths. Rendered as themed
// button rows (not a native <select>, whose option list renders unstyled
// white-on-white). Device labels are only populated once mic permission has been
// granted; before that they read as generic names.
const MicDevicePicker: FC = () => {
  const selectedMicDeviceId = useChatRuntimeStore((s) => s.selectedMicDeviceId);
  const setSelectedMicDeviceId = useChatRuntimeStore((s) => s.setSelectedMicDeviceId);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [open, setOpen] = useState(false);

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

  const rows: { id: string | null; label: string }[] = [
    { id: null, label: "Default input" },
    ...devices.map((d, i) => ({
      id: d.deviceId,
      label: d.label || `Microphone ${i + 1}`,
    })),
  ];
  const current =
    rows.find((r) => r.id === (selectedMicDeviceId ?? null)) ?? rows[0];

  return (
    <>
      <div className="my-1.5 h-px bg-black/[0.08] dark:bg-white/[0.08]" />
      <div className="px-2 pb-1 pt-0.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
        Microphone
      </div>
      {/* Collapsed selector: shows the current mic; expands the device list on
          click so many inputs don't blow out the popover height. */}
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 rounded-full px-2 py-1.5 text-left text-sm transition-colors hover:bg-[#ececec] dark:hover:bg-[var(--sidebar-accent)]"
      >
        <MicIcon className="size-3.5 shrink-0 text-muted-foreground" />
        <span className="min-w-0 flex-1 truncate">{current.label}</span>
        <HugeiconsIcon
          icon={ArrowDown01Icon}
          strokeWidth={2}
          className={cn(
            "size-3.5 shrink-0 text-muted-foreground transition-transform",
            open && "rotate-180",
          )}
        />
      </button>
      {open && (
        <div className="max-h-[8.5rem] overflow-y-auto">
          {rows.map((row, i) => {
            const active = (selectedMicDeviceId ?? null) === row.id;
            return (
              <button
                key={row.id ?? `default-${i}`}
                type="button"
                onClick={() => {
                  setSelectedMicDeviceId(row.id);
                  setOpen(false);
                }}
                className={cn(
                  "flex w-full items-center gap-2 rounded-full px-2 py-1.5 pl-7 text-left text-sm transition-colors hover:bg-[#ececec] dark:hover:bg-[var(--sidebar-accent)]",
                  active && "bg-[#ececec] dark:bg-[var(--sidebar-accent)]",
                )}
              >
                <span className="min-w-0 flex-1 truncate">{row.label}</span>
                {active && (
                  <DotTag
                    tone="success"
                    label="Active"
                    className="ml-auto h-[18px] shrink-0 gap-1 rounded-md px-1.5"
                    dotClassName="size-[5px]"
                  />
                )}
              </button>
            );
          })}
        </div>
      )}
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
  ready = false,
  loading = false,
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
        className="unsloth-model-selector-menu menu-soft-surface w-[340px] gap-0 rounded-lg border-0 p-1.5 ring-0"
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
          <MicVocalIcon className="size-3.5 shrink-0 text-muted-foreground" />
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
                {model.sizeBytes != null && (
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
