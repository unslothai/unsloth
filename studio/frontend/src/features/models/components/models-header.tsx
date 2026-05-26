// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HfTokenIndicator } from "@/components/hf-token-indicator";
import { PageHeading } from "@/components/layout";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  ChipIcon,
  Database02Icon,
  Logout01Icon,
  PackageIcon,
  RamMemoryIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";

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
  isDataset,
  gpuLabel,
  ramLabel,
  activeCheckpoint,
  activeGgufVariant,
  onEject,
}: {
  cachedCount: number;
  localCount: number;
  isDataset: boolean;
  gpuLabel: string;
  ramLabel: string;
  activeCheckpoint: string | null;
  activeGgufVariant: string | null;
  onEject: () => void;
}) {
  return (
    <header className="font-heading flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center sm:justify-between">
      <PageHeading
        title="Hub"
        subtitle={
          isDataset
            ? "Discover, download, and train on datasets locally."
            : "Discover, download, and run inference models locally."
        }
      />

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
