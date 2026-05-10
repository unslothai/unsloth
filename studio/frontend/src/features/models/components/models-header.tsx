// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { useSettingsDialogStore } from "@/features/settings";
import { cn } from "@/lib/utils";
import {
  AiSecurity03Icon,
  ChipIcon,
  Database02Icon,
  Logout01Icon,
  PackageIcon,
  RamMemoryIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";

function HfTokenIndicator() {
  const hfToken = useChatRuntimeStore((s) => s.hfToken);
  const openDialog = useSettingsDialogStore((s) => s.openDialog);
  const hasToken = Boolean(hfToken && hfToken.trim());

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={() => openDialog("general")}
          aria-label={
            hasToken
              ? "Hugging Face token configured"
              : "Set Hugging Face token"
          }
          className={cn(
            "inline-flex items-center justify-center px-2.5 py-1 text-[11.5px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
            hasToken
              ? "tag-soft text-muted-foreground hover:text-foreground/80"
              : "rounded-[11px] bg-[#b42323] text-white hover:bg-[#9e1e1e] dark:bg-[#5e1a1a] dark:hover:bg-[#4d1414]",
          )}
        >
          <HugeiconsIcon
            icon={AiSecurity03Icon}
            strokeWidth={1.75}
            className="size-4"
          />
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom" sideOffset={6} className="tooltip-compact">
        {hasToken
          ? "Hugging Face token set"
          : "No Hugging Face token, click to set"}
      </TooltipContent>
    </Tooltip>
  );
}

function StatPill({
  icon,
  label,
  value,
  tone = "default",
}: {
  icon: IconSvgElement;
  label: string;
  value: string;
  tone?: "default" | "active";
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div
          className={cn(
            "inline-flex cursor-default items-center gap-1.5 px-2.5 py-1 text-[11.5px] transition-colors duration-150",
            tone === "active"
              ? "rounded-[11px] bg-emerald-500/10 text-emerald-700 ring-1 ring-inset ring-emerald-500/20 dark:text-emerald-300"
              : "tag-soft text-muted-foreground hover:text-foreground/80",
          )}
        >
          <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-3.5" />
          <span className="font-medium tabular-nums">{value}</span>
        </div>
      </TooltipTrigger>
      <TooltipContent className="tooltip-compact">{label}</TooltipContent>
    </Tooltip>
  );
}

export function ModelsHeader({
  cachedCount,
  localCount,
  gpuLabel,
  ramLabel,
  activeCheckpoint,
  activeGgufVariant,
  onEject,
}: {
  cachedCount: number;
  localCount: number;
  gpuLabel: string;
  ramLabel: string;
  activeCheckpoint: string | null;
  activeGgufVariant: string | null;
  onEject: () => void;
}) {
  return (
    <header className="flex flex-wrap items-center justify-between gap-3">
      <div>
        <h1 className="text-[26px] font-semibold leading-[1.1] tracking-[-0.025em] text-foreground">
          Hub
        </h1>
        <p className="text-[12.5px] leading-[18px] text-muted-foreground">
          Discover, download, and run inference models locally.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <HfTokenIndicator />
        <StatPill
          icon={PackageIcon}
          label="Cache"
          value={String(cachedCount)}
        />
        <StatPill
          icon={Database02Icon}
          label="Local"
          value={String(localCount)}
        />
        <StatPill icon={ChipIcon} label="VRAM" value={gpuLabel} />
        <StatPill icon={RamMemoryIcon} label="CPU RAM" value={ramLabel} />

        {activeCheckpoint && (
          <div className="tag-soft ml-1 inline-flex items-center gap-1.5 px-2 py-1 text-[11.5px]">
            <span
              className="size-1.5 rounded-full bg-emerald-500"
              aria-hidden="true"
            />
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="max-w-[120px] cursor-default truncate font-medium text-primary">
                  {activeCheckpoint}
                  {activeGgufVariant ? ` · ${activeGgufVariant}` : ""}
                </span>
              </TooltipTrigger>
              <TooltipContent
                side="bottom"
                sideOffset={6}
                className="tooltip-compact"
              >
                {activeCheckpoint}
                {activeGgufVariant ? ` · ${activeGgufVariant}` : ""}
              </TooltipContent>
            </Tooltip>
            <button
              type="button"
              onClick={onEject}
              className="-mr-0.5 ml-0.5 inline-flex cursor-pointer items-center gap-1 rounded-md px-1.5 text-[11px] text-muted-foreground transition-colors hover:text-foreground"
            >
              <HugeiconsIcon
                icon={Logout01Icon}
                strokeWidth={1.75}
                className="size-3"
              />
              Eject
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
