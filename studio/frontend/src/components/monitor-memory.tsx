// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Memory readouts shared by the floating Live monitor and Settings > System. */

import { Progress } from "@/components/ui/progress";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import type { ReactNode } from "react";
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

const KIND_COLORS = {
  unsloth: "bg-control-accent",
  app: "bg-sky-500",
  other: "bg-muted-foreground/45",
  unattributed: "bg-muted-foreground/45",
  free: "bg-muted ring-1 ring-inset ring-border",
} as const;
// Models are Unsloth's, so they stay in its colour, stepping lighter down the list.
const MODEL_COLORS = [
  "bg-control-accent",
  "bg-control-accent/70",
  "bg-control-accent/50",
  "bg-control-accent/35",
];

function segmentColor(segment: MemorySegment, index: number): string {
  return segment.kind === "model"
    ? MODEL_COLORS[Math.min(index, MODEL_COLORS.length - 1)]
    : KIND_COLORS[segment.kind];
}

function colored(segments: MemorySegment[]) {
  let model = 0;
  return segments.map((segment) => ({
    segment,
    color: segmentColor(segment, segment.kind === "model" ? model++ : 0),
  }));
}

export function MemoryBar({ segments }: { segments: MemorySegment[] }) {
  const total = segments.reduce((sum, segment) => sum + segment.gb, 0);
  return (
    <div
      className="flex h-1.5 overflow-hidden rounded-full bg-muted"
      data-testid="monitor-share-bar"
    >
      {colored(segments)
        .filter(({ segment }) => segment.kind !== "free")
        .map(({ segment, color }) => (
          <div
            key={segment.id}
            className={color}
            style={{ width: `${total > 0 ? (segment.gb / total) * 100 : 0}%` }}
          />
        ))}
    </div>
  );
}

export function MemoryLegend({
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
  return (
    <div className="space-y-0.5" data-testid="monitor-share-legend">
      {colored(segments).map(({ segment, color }) => (
        <div key={segment.id} className="flex items-center gap-2">
          <span className={cn("size-2 shrink-0 rounded-full", color)} />
          <span
            className="min-w-0 flex-1 truncate"
            title={
              segment.kind === "unattributed"
                ? t("settings.resources.liveMonitor.otherInUseHint")
                : label(segment)
            }
          >
            {label(segment)}
          </span>
          <span className="shrink-0 font-mono tabular-nums text-foreground">
            {formatGiB(segment.gb)}
          </span>
        </div>
      )).flatMap((row, index) =>
        noModels && index === 0
          ? [
              row,
              <div
                key="no-models"
                className="ps-4 italic first-letter:uppercase"
                data-testid="monitor-no-models"
              >
                {t("settings.resources.liveMonitor.noModelLoaded")}
              </div>,
            ]
          : [row],
      )}
    </div>
  );
}
