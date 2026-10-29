// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Memory readouts shared by the floating Live monitor and Settings > System. */

import { Progress } from "@/components/ui/progress";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import type { ReactNode } from "react";
import {
  STORAGE_COLORS,
  StorageBar,
  type StorageSegment,
} from "@/features/settings/components/storage-overview";
import type { MemorySegment } from "./monitor-share";

export function usageIndicatorClass(percent: number): string {
  if (percent >= 90) return "bg-destructive";
  if (percent >= 70) return "bg-amber-500";
  return "bg-control-accent";
}

export function usageTextClass(percent: number): string {
  if (percent >= 90) return "text-destructive";
  if (percent >= 70) return "text-amber-600 dark:text-amber-400";
  return "text-primary";
}

export function formatGiB(value: number): string {
  // Backend reports binary units (bytes / 1024**3), so label GiB.
  return `${value.toFixed(value >= 10 ? 1 : 2)} GiB`;
}

/** Label and readout over a bar, with an optional line or legend under it. */
export function MeterRow({
  label,
  readout,
  readoutClassName,
  children,
  footer,
  testId,
}: {
  label: ReactNode;
  readout: string;
  readoutClassName?: string;
  children: ReactNode;
  footer?: ReactNode;
  testId?: string;
}) {
  return (
    <div className="space-y-1.5" data-testid={testId}>
      <div className="flex items-baseline justify-between gap-2 text-xs">
        <span className="min-w-0 truncate font-medium text-foreground">{label}</span>
        <span
          className={cn(
            "shrink-0 font-mono tabular-nums text-muted-foreground",
            readoutClassName,
          )}
        >
          {readout}
        </span>
      </div>
      {children}
      {footer && (
        <div className="space-y-0.5 text-ui-11 text-muted-foreground">{footer}</div>
      )}
    </div>
  );
}

export function UsageBar({ percent }: { percent: number | null }) {
  return (
    <Progress
      value={percent ?? 0}
      className="h-1.5 rounded-full bg-muted"
      indicatorClassName={usageIndicatorClass(percent ?? 0)}
    />
  );
}

// Settings' storage palette, read inside a function: a module-scope read can hit the chat import cycle.
function segmentColor(kind: MemorySegment["kind"], modelIndex: number): string {
  switch (kind) {
    case "model":
      return [
        STORAGE_COLORS.yellow,
        STORAGE_COLORS.orange,
        STORAGE_COLORS.green,
        STORAGE_COLORS.blue,
      ][Math.min(modelIndex, 3)];
    // Unsloth reads as "Models" in Settings > Storage, so it keeps that yellow.
    case "unsloth":
      return STORAGE_COLORS.yellow;
    case "app":
      return STORAGE_COLORS.purple;
    case "free":
      return STORAGE_COLORS.free;
    default:
      return STORAGE_COLORS.other;
  }
}

/** One pool split by owner, drawn with Settings' storage bar and its inline legend. */
export function MemoryShareBar({
  segments,
  noModels = false,
}: {
  segments: MemorySegment[];
  /** Model split is on with nothing loaded: say so, or the toggle looks dead. */
  noModels?: boolean;
}) {
  const t = useT();
  const label = (segment: MemorySegment) => {
    switch (segment.kind) {
      case "model":
        return segment.name ?? "";
      case "app":
        return t("settings.resources.liveMonitor.unslothApp");
      case "unsloth":
        return t("settings.resources.liveMonitor.unsloth");
      case "other":
        return t("settings.resources.liveMonitor.otherApps");
      case "unattributed":
        return t("settings.resources.liveMonitor.otherInUse");
      case "free":
        return t("settings.resources.liveMonitor.freeLabel");
    }
  };
  let model = 0;
  const storageSegments: StorageSegment[] = segments.map((segment) => ({
    key: segment.id,
    label: label(segment),
    bytes: segment.gb,
    value: formatGiB(segment.gb),
    color: segmentColor(segment.kind, segment.kind === "model" ? model++ : 0),
  }));
  return (
    <div className="space-y-1.5" data-testid="monitor-share-bar">
      <StorageBar
        segments={storageSegments}
        legend="inline"
        barClassName="h-1.5 rounded-full"
      />
      {noModels && (
        <div className="text-ui-11 text-muted-foreground" data-testid="monitor-no-models">
          {t("settings.resources.liveMonitor.noModelLoaded")}
        </div>
      )}
    </div>
  );
}
