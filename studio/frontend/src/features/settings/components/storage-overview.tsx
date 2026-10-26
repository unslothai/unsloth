// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactNode } from "react";

export const STORAGE_COLORS = {
  yellow: "bg-amber-400 dark:bg-[#f7d35f]",
  purple: "bg-violet-500 dark:bg-[#a971f2]",
  orange: "bg-orange-600 dark:bg-[#eb6e3a]",
  green: "bg-green-600 dark:bg-[#6fc77a]",
  blue: "bg-sky-500 dark:bg-sky-400",
  other: "bg-[#767676] dark:bg-[#8e8e8e]",
  free: "bg-muted-foreground/15",
} as const;

export interface StorageSegment {
  key: string;
  label: string;
  bytes: number;
  value: string;
  /** Background utility for the bar slice and the legend dot. */
  color: string;
}

export function StorageOverview({
  label,
  heading,
  action,
  segments,
  totalBytes,
  footer,
}: {
  label?: string;
  heading: ReactNode;
  action?: ReactNode;
  segments: StorageSegment[];
  /** Width of the whole bar; defaults to the sum, so the slices fill it. */
  totalBytes?: number;
  footer?: ReactNode;
}) {
  const sum = segments.reduce(
    (acc, segment) => acc + Math.max(0, segment.bytes),
    0,
  );
  const scale = Math.max(totalBytes ?? sum, sum);
  return (
    <div data-settings-label={label} className="flex flex-col gap-4 py-3">
      <div className="flex min-h-8 items-center justify-between gap-3">
        <p className="text-sm font-semibold text-foreground">{heading}</p>
        {action}
      </div>
      {/* A rounded rectangle, not a pill, by design. */}
      <div className="flex h-6 overflow-hidden rounded-sm bg-muted">
        {scale > 0
          ? segments.map((segment) =>
              segment.bytes > 0 ? (
                <div
                  key={segment.key}
                  className={cn("min-w-1", segment.color)}
                  style={{ width: `${(segment.bytes / scale) * 100}%` }}
                />
              ) : null,
            )
          : null}
      </div>
      <div className="flex flex-wrap gap-x-8 gap-y-3">
        {segments.map((segment) => (
          <div key={segment.key} className="flex flex-col gap-1">
            <span className="flex items-center gap-2 text-sm text-foreground/80">
              <span
                className={cn("size-2.5 shrink-0 rounded-full", segment.color)}
              />
              {segment.label}
            </span>
            <span className="text-sm tabular-nums text-muted-foreground">
              {segment.value}
            </span>
          </div>
        ))}
      </div>
      {footer ? (
        <div className="text-xs text-muted-foreground">{footer}</div>
      ) : null}
    </div>
  );
}

export function SettingsCard({
  children,
  className,
}: { children: ReactNode; className?: string }) {
  return (
    <div
      className={cn(
        "my-2 flex flex-col rounded-xl border border-border/60 px-3 sm:px-4",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function SettingsCardRow({
  label,
  description,
  hint,
  children,
}: {
  label: string;
  description?: ReactNode;
  hint?: string;
  children?: ReactNode;
}) {
  return (
    <div
      data-settings-label={label}
      className="flex flex-col border-border/60 py-3.5 not-first:border-t"
    >
      <div
        className={cn(
          "flex flex-wrap justify-between gap-x-6 gap-y-2",
          description ? "items-start" : "items-center",
        )}
      >
        <div className="flex min-w-0 flex-1 basis-60 flex-col gap-1">
          <span className="text-sm font-medium text-foreground">
            {label}
            {hint ? (
              <Tooltip>
                <TooltipTrigger asChild={true}>
                  <button
                    type="button"
                    aria-label={hint}
                    className="ml-1.5 inline-flex align-baseline rounded-full text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  >
                    <HugeiconsIcon
                      icon={InformationCircleIcon}
                      className="size-[var(--ui-icon-size-hint)]"
                    />
                  </button>
                </TooltipTrigger>
                <TooltipContent className="max-w-[calc(300px*var(--ui-space-scale,1))] text-ui-11 leading-snug">
                  {hint}
                </TooltipContent>
              </Tooltip>
            ) : null}
          </span>
          {description ? (
            <div className="min-w-0 text-xs text-muted-foreground">
              {description}
            </div>
          ) : null}
        </div>
        {children ? (
          <div
            data-title-line-control={description ? true : undefined}
            className={cn(
              "flex shrink-0 gap-2",
              !description && "items-center",
            )}
          >
            {children}
          </div>
        ) : null}
      </div>
    </div>
  );
}
