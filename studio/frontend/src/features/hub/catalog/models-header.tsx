// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HfTokenIndicator } from "@/features/hub/components/hf-token-indicator";
import { PageHeading } from "@/features/hub/components/page-heading";
import { TransportToggle } from "./transport-toggle";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  ChipIcon,
  Database02Icon,
  PackageIcon,
  RamMemoryIcon,
  RemoveCircleIcon,
  CpuIcon
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";

function StatPill({
  icon,
  label,
  value,
}: {
  icon: IconSvgElement;
  label: string;
  value: string;
}) {
  return (
    <span className="hub-stat-pill">
      <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-3.5" />
      <span className="hub-stat-pill-value">{value}</span>
      <span>{label}</span>
    </span>
  );
}

export function ModelsHeader({
  cachedCount,
  localCount,
  isDataset,
  gpuLabel,
  ramLabel,
  coreLabel,
  activeCheckpoint,
  activeGgufVariant,
  onTitleClick,
  onEject,
}: {
  cachedCount: number;
  localCount: number;
  isDataset: boolean;
  gpuLabel: string;
  ramLabel: string;
  coreLabel: string;
  activeCheckpoint: string | null;
  activeGgufVariant: string | null;
  onTitleClick: () => void;
  onEject: () => void;
}) {
  return (
    <header className="font-heading flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center sm:justify-between">
      <PageHeading
        title={isDataset ? "Datasets" : "Model hub"}
        onTitleClick={onTitleClick}
        subtitle={
          isDataset
            ? "Discover, download, and train on datasets locally."
            : "Discover, download, and run inference models locally."
        }
      />

      <div className="flex min-w-0 flex-wrap items-center justify-end gap-1.5 sm:flex-1">
        <HfTokenIndicator />
        <TransportToggle />
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
        <StatPill icon={RamMemoryIcon} label="RAM" value={ramLabel} />
        <StatPill icon={CpuIcon} label="CPU" value={coreLabel} />

        {activeCheckpoint && (
          <div className="hub-tag-soft ml-1 inline-flex items-center gap-1.5 px-2 py-1 text-ui-11p5">
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
              className="-mr-0.5 ml-0.5 inline-flex cursor-pointer items-center gap-1 rounded-md px-1.5 text-ui-11 text-muted-foreground transition-colors hover:text-foreground"
            >
              <HugeiconsIcon
                icon={RemoveCircleIcon}
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
