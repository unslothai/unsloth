// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Memory readouts shared by the floating Live monitor and Settings > System. */

import { Progress } from "@/components/ui/progress";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import type { ReactNode } from "react";
import type { MemoryShare, ModelBreakdown } from "./monitor-share";

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

const SHARE_COLORS = {
  unsloth: "bg-control-accent",
  other: "bg-muted-foreground/45",
  free: "bg-muted ring-1 ring-inset ring-border",
} as const;

export function ShareBar({ share }: { share: MemoryShare }) {
  const total = share.unsloth + share.other + share.free;
  const width = (value: number) => `${total > 0 ? (value / total) * 100 : 0}%`;
  return (
    <div
      className="flex h-1.5 overflow-hidden rounded-full bg-muted"
      data-testid="monitor-share-bar"
    >
      <div className={SHARE_COLORS.unsloth} style={{ width: width(share.unsloth) }} />
      <div className={SHARE_COLORS.other} style={{ width: width(share.other) }} />
    </div>
  );
}

function LegendRow({
  swatch,
  label,
  value,
  title,
}: {
  swatch?: string;
  label: string;
  value: string;
  title?: string;
}) {
  return (
    <div className="flex items-center gap-2">
      {swatch && <span className={cn("size-2 shrink-0 rounded-full", swatch)} />}
      <span className="min-w-0 flex-1 truncate" title={title ?? label}>
        {label}
      </span>
      <span className="shrink-0 font-mono tabular-nums text-foreground">{value}</span>
    </div>
  );
}

export function ShareLegend({
  share,
  unslothLabel,
}: {
  share: MemoryShare;
  unslothLabel: string;
}) {
  const t = useT();
  return (
    <div className="space-y-0.5" data-testid="monitor-share-legend">
      <LegendRow swatch={SHARE_COLORS.unsloth} label={unslothLabel} value={formatGiB(share.unsloth)} />
      <LegendRow
        swatch={SHARE_COLORS.other}
        label={t("settings.resources.liveMonitor.otherApps")}
        value={formatGiB(share.other)}
      />
      <LegendRow
        swatch={SHARE_COLORS.free}
        label={t("settings.resources.liveMonitor.freeLabel")}
        value={formatGiB(share.free)}
      />
    </div>
  );
}

export function ModelBreakdownList({
  breakdown,
  className,
}: {
  breakdown: ModelBreakdown;
  className?: string;
}) {
  const t = useT();
  return (
    <div
      className={cn("space-y-0.5 text-ui-11 text-muted-foreground", className)}
      data-testid="monitor-model-breakdown"
    >
      <div className="text-xs font-medium text-foreground">
        {t("settings.resources.liveMonitor.byModel")}
      </div>
      {breakdown.models.length === 0 ? (
        <div>{t("settings.resources.liveMonitor.noModelLoaded")}</div>
      ) : (
        breakdown.models.map((model) => (
          <LegendRow key={model.id} label={model.name} value={formatGiB(model.gb)} />
        ))
      )}
      {breakdown.app !== null && (
        <LegendRow
          label={t("settings.resources.liveMonitor.unslothApp")}
          value={formatGiB(breakdown.app)}
        />
      )}
    </div>
  );
}
