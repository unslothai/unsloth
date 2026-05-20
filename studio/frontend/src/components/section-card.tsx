// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { createContext, useContext, type ReactNode } from "react";

interface SectionCardProps {
  icon: ReactNode;
  title: string;
  description: string;
  accent?: "emerald" | "indigo" | "orange" | "blue";
  featured?: boolean;
  className?: string;
  badge?: string;
  headerAction?: ReactNode;
  flat?: boolean;
  children: ReactNode;
}

// Set this to true via Provider to render every nested SectionCard in a
// flat, chrome-less style that blends into the surrounding surface (used by
// the Train page wizard to avoid box-in-box visual noise).
export const SectionCardFlatContext = createContext<boolean>(false);

// Set to true to render only the children — title/icon/description are hidden
// because the wrapping component (e.g. a WizardStep) provides its own header.
export const SectionCardHeadlessContext = createContext<boolean>(false);

const accentStyles = {
  emerald: {
    border: "ring-emerald-500/20",
    iconBox:
      "ring-emerald-200 bg-emerald-50 text-emerald-600 dark:ring-emerald-800 dark:bg-emerald-950 dark:text-emerald-400",
    flatIconBox:
      "bg-emerald-50 text-emerald-600 dark:bg-emerald-950/50 dark:text-emerald-400",
  },
  indigo: {
    border: "ring-indigo-500/20",
    iconBox:
      "ring-indigo-200 bg-indigo-50 text-indigo-600 dark:ring-indigo-800 dark:bg-indigo-950 dark:text-indigo-400",
    flatIconBox:
      "bg-indigo-50 text-indigo-600 dark:bg-indigo-950/50 dark:text-indigo-400",
  },
  orange: {
    border: "ring-orange-500/20",
    iconBox:
      "ring-orange-200 bg-orange-50 text-orange-600 dark:ring-orange-800 dark:bg-orange-950 dark:text-orange-400",
    flatIconBox:
      "bg-orange-50 text-orange-600 dark:bg-orange-950/50 dark:text-orange-400",
  },
  blue: {
    border: "ring-blue-500/20",
    iconBox:
      "ring-blue-200 bg-blue-50 text-blue-600 dark:ring-blue-800 dark:bg-blue-950 dark:text-blue-400",
    flatIconBox:
      "bg-blue-50 text-blue-600 dark:bg-blue-950/50 dark:text-blue-400",
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
  flat: flatProp,
  children,
}: SectionCardProps) {
  const styles = accentStyles[accent];
  const ctxFlat = useContext(SectionCardFlatContext);
  const headless = useContext(SectionCardHeadlessContext);
  const flat = flatProp ?? ctxFlat;

  // Headless mode: the wrapper supplies the header, so we render the
  // section's content as a bare flex column. Class overrides from the
  // consumer (e.g. min-h-studio-config-column) are intentionally dropped
  // because they'd impose a fixed card height that's now meaningless.
  if (headless) {
    return (
      <div className="flex flex-col gap-3 min-w-0">{children}</div>
    );
  }

  return (
    <div
      data-section-flat={flat ? "true" : undefined}
      className={cn(
        "flex flex-col relative overflow-clip transition-all duration-300 ease-in-out",
        flat
          ? "gap-3 p-0 bg-transparent ring-0 rounded-none"
          : "bg-card corner-squircle rounded-3xl ring-1 ring-foreground/10 gap-5 p-5",
        featured && !flat && styles.border,
        className,
      )}
    >
      {featured && !flat && (
        <div className="pointer-events-none absolute inset-x-0 top-0 h-24 bg-gradient-to-b from-emerald-500/[0.04] to-transparent" />
      )}
      {/* Header */}
      <div className="flex items-center gap-2.5">
        <div
          className={cn(
            "shrink-0",
            flat
              ? cn("rounded-lg p-1.5", styles.flatIconBox)
              : cn("rounded-xl corner-squircle p-2 ring-1", styles.iconBox),
          )}
        >
          {icon}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <h3
              className={cn(
                "font-semibold",
                flat ? "text-[13px]" : "text-sm pb-1",
              )}
            >
              {title}
            </h3>
            {badge && (
              <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-semibold text-emerald-700 dark:bg-emerald-900 dark:text-emerald-300">
                {badge}
              </span>
            )}
          </div>
          <p
            className={cn(
              "text-muted-foreground",
              flat ? "text-[11.5px] leading-tight" : "text-xs",
            )}
          >
            {description}
          </p>
        </div>
        {headerAction && <div className="shrink-0">{headerAction}</div>}
      </div>
      {/* Content */}
      {children}
    </div>
  );
}
