// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSettingsDialogStore } from "@/features/settings";
import { useIsMobile } from "@/hooks/use-mobile";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import { useId, useState } from "react";
import { ChipIcon } from "@/lib/hugeicons-derived";
import {
  CpuIcon,
  Database02Icon,
  PackageIcon,
  RamMemoryIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";
import { HfTokenIndicator } from "../components/hf-token-indicator";
import { PageHeading } from "../components/page-heading";
import { DownloadLocationPicker } from "./download-location-picker";
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
  const mobile = useIsMobile();
  const [deviceOpen, setDeviceOpen] = useState(false);
  const deviceId = useId();
  const subtitle = isDataset
    ? "Discover and download datasets locally."
    : "Discover and download models locally.";
  const gpuMemoryValue = gpuSharedLabel
    ? `${gpuLabel} VRAM + ${gpuSharedLabel}`
    : gpuLabel;
  return (
    <header className="font-heading flex flex-wrap items-center justify-between gap-3">
      <PageHeading
        title={isDataset ? "Datasets" : "Model hub"}
        onTitleClick={onTitleClick}
        subtitle={mobile ? undefined : subtitle}
      />

      {mobile && (
        <button
          type="button"
          data-tour="hub-device"
          aria-expanded={deviceOpen}
          aria-controls={deviceId}
          onClick={() => setDeviceOpen((open) => !open)}
          className="inline-flex min-h-11 items-center gap-1.5 rounded-md px-1.5 text-ui-13 font-medium text-muted-foreground hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
        >
          <HugeiconsIcon icon={CpuIcon} className="size-4" />
          Device
          <HugeiconsIcon
            icon={ChevronDownStandardIcon}
            className={cn("size-3.5", deviceOpen && "rotate-180")}
          />
        </button>
      )}
      <div
        id={deviceId}
        data-tour={mobile ? undefined : "hub-device"}
        className={cn(
          "flex min-w-0 flex-wrap items-center gap-1.5 md:flex-1 md:justify-end",
          mobile && "w-full rounded-xl border border-border bg-muted/30 p-3",
          mobile && !deviceOpen && "hidden",
        )}
      >
        <HfTokenIndicator onOpenSettings={() => openSettings("general")} />
        <TransportToggle />
        {isDataset ? null : <DownloadLocationPicker />}
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
