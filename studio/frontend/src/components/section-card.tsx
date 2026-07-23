// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

interface SectionCardProps {
  icon: ReactNode;
  title: string;
  description: string;
  accent?: "emerald" | "indigo" | "orange" | "blue";
  featured?: boolean;
  className?: string;
  badge?: string;
  headerAction?: ReactNode;
  children: ReactNode;
}

const accentStyles = {
  // Brand accent variant; follows the active palette via --control-accent.
  emerald: {
    border: "ring-control-accent/20",
    iconBox: "ring-control-accent/25 bg-control-accent/10 text-control-accent",
  },
  indigo: {
    border: "ring-indigo-500/20",
    iconBox:
      "ring-indigo-200 bg-indigo-50 text-indigo-600 dark:ring-indigo-800 dark:bg-indigo-950 dark:text-indigo-400",
  },
  orange: {
    border: "ring-orange-500/20",
    iconBox:
      "ring-orange-200 bg-orange-50 text-orange-600 dark:ring-orange-800 dark:bg-orange-950 dark:text-orange-400",
  },
  blue: {
    border: "ring-blue-500/20",
    iconBox:
      "ring-blue-200 bg-blue-50 text-blue-600 dark:ring-blue-800 dark:bg-blue-950 dark:text-blue-400",
  },
};

export function SectionCard({
  icon,
  title,
  description,
  accent = "emerald",
  featured,
  className,
  badge,
  headerAction,
  children,
}: SectionCardProps) {
  const styles = accentStyles[accent];

  return (
    <div
      className={cn(
        "bg-card corner-squircle rounded-3xl ring-1 ring-foreground/10 flex flex-col gap-5 p-5 relative overflow-clip transition-all duration-300 ease-in-out",
        featured && styles.border,
        className,
      )}
    >
      {featured && (
        <div className="pointer-events-none absolute inset-x-0 top-0 h-24 bg-gradient-to-b from-control-accent/[0.04] to-transparent" />
      )}
      {/* Header */}
      <div className="flex items-center gap-3">
        <div
          className={cn(
            "rounded-xl corner-squircle p-2 ring-1 shrink-0",
            styles.iconBox,
          )}
        >
          {icon}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 pb-1">
            <h3 className="text-sm font-semibold">{title}</h3>
            {badge && (
              <span className="rounded-full bg-control-accent/15 px-2 py-0.5 text-[0.625rem] font-semibold text-control-accent">
                {badge}
              </span>
            )}
          </div>
          <p className="text-xs text-muted-foreground">{description}</p>
        </div>
        {headerAction && <div className="shrink-0">{headerAction}</div>}
      </div>
      {/* Content */}
      {children}
    </div>
  );
}
