// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

export function SettingsRow({
  label,
  description,
  icon,
  children,
  destructive,
  className,
}: {
  label: string;
  description?: string;
  icon?: ReactNode;
  children?: ReactNode;
  destructive?: boolean;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex items-center justify-between gap-6 py-3",
        destructive && "border-t border-border/60 mt-2 pt-4",
        className,
      )}
    >
      <div className="flex min-w-0 items-center gap-2.5">
        {icon ? (
          <span className="flex shrink-0 items-center text-foreground">
            {icon}
          </span>
        ) : null}
        <div className="flex min-w-0 flex-col gap-0.5">
          <span className="text-sm font-medium text-foreground">{label}</span>
          {description ? (
            <span className="text-xs text-muted-foreground leading-snug">
              {description}
            </span>
          ) : null}
        </div>
      </div>
      {children ? <div className="flex shrink-0 items-center">{children}</div> : null}
    </div>
  );
}
