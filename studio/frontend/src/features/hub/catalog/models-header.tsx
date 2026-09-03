// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSettingsDialogStore } from "@/features/settings";
import {
  ChipIcon,
  CpuIcon,
  Database02Icon,
  PackageIcon,
  RamMemoryIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";
import { HfTokenIndicator } from "../components/hf-token-indicator";
import { PageHeading } from "../components/page-heading";
import { TransportToggle } from "./transport-toggle";

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
  gpuSharedLabel,
  ramLabel,
  coreLabel,
  onTitleClick,
}: {
  cachedCount: number;
  localCount: number;
  isDataset: boolean;
  gpuLabel: string;
  gpuSharedLabel: string | null;
  ramLabel: string;
  coreLabel: string;
  onTitleClick: () => void;
}) {
  const openSettings = useSettingsDialogStore((s) => s.openDialog);
  const gpuMemoryValue = gpuSharedLabel
    ? `${gpuLabel} VRAM + ${gpuSharedLabel}`
    : gpuLabel;
  return (
    <header className="font-heading flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center sm:justify-between">
      <PageHeading
        title={isDataset ? "Datasets" : "Model hub"}
        onTitleClick={onTitleClick}
        subtitle={
          isDataset
            ? "Discover and download datasets locally."
            : "Discover and download models locally."
        }
      />

      <div className="flex min-w-0 flex-wrap items-center justify-end gap-1.5 sm:flex-1">
        <HfTokenIndicator onOpenSettings={() => openSettings("general")} />
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
        <StatPill
          icon={ChipIcon}
          label={gpuSharedLabel ? "shared" : "VRAM"}
          value={gpuMemoryValue}
        />
        <StatPill icon={RamMemoryIcon} label="RAM" value={ramLabel} />
        <StatPill icon={CpuIcon} label="CPU" value={coreLabel} />
      </div>
    </header>
  );
}
