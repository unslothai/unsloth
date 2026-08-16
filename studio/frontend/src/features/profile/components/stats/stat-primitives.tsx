// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

/** Bordered surface every stats block sits on, matching the profile card. */
export function StatsCard({
  title,
  description,
  action,
  children,
  className,
}: {
  title?: string;
  description?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      {...(title ? { "data-settings-label": title } : {})}
      className={cn(
        "flex w-full flex-col gap-4 rounded-2xl border border-border bg-background dark:border-transparent dark:bg-white/[0.06] px-5 py-5",
        className,
      )}
    >
      {title ? (
        <header className="flex items-start justify-between gap-3">
          <div className="flex min-w-0 flex-col gap-0.5">
            <h3 className="text-sm font-semibold font-heading text-foreground">
              {title}
            </h3>
            {description ? (
              <p className="text-xs text-muted-foreground">{description}</p>
            ) : null}
          </div>
          {action ? <div className="shrink-0">{action}</div> : null}
        </header>
      ) : null}
      {children}
    </section>
  );
}

/** Big number + caption, used across the highlight and training rows. */
export function StatTile({
  value,
  label,
  hint,
  className,
}: {
  value: string;
  label: string;
  hint?: string;
  className?: string;
}) {
  return (
    <div
      data-settings-label={label}
      className={cn(
        "flex min-w-0 flex-col items-center gap-0.5 px-2 py-1 text-center",
        className,
      )}
      {...(hint ? { title: hint } : {})}
    >
      <span className="text-xl font-semibold font-heading tabular-nums text-foreground">
        {value}
      </span>
      <span className="text-xs text-muted-foreground">{label}</span>
    </div>
  );
}

/** Label left, value right: the "Activity insights" rows. */
export function StatRow({
  label,
  value,
  emphasis,
}: {
  label: string;
  value: string;
  emphasis?: boolean;
}) {
  return (
    <div
      data-settings-label={label}
      className="flex items-center justify-between gap-4 py-1.5"
    >
      <span className="min-w-0 truncate text-sm text-muted-foreground">
        {label}
      </span>
      <span
        className={cn(
          "shrink-0 text-sm tabular-nums",
          emphasis ? "font-semibold text-foreground" : "text-foreground",
        )}
      >
        {value}
      </span>
    </div>
  );
}

/** Thin progress track (level bar, achievement progress, model share). */
export function StatMeter({
  progress,
  className,
  tone = "primary",
}: {
  progress: number;
  className?: string;
  tone?: "primary" | "muted";
}) {
  const clamped = Math.min(
    1,
    Math.max(0, Number.isFinite(progress) ? progress : 0),
  );
  return (
    <div
      className={cn(
        "h-1.5 w-full overflow-hidden rounded-full bg-muted-foreground/15",
        className,
      )}
    >
      <div
        className={cn(
          "h-full rounded-full transition-[width] duration-500",
          tone === "primary" ? "bg-primary" : "bg-muted-foreground/50",
        )}
        style={{ width: `${clamped * 100}%` }}
      />
    </div>
  );
}
