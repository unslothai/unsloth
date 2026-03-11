// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { cn } from "@/lib/utils";
import type { ReactElement, ReactNode } from "react";

type InlineFieldProps = {
  label: string;
  className?: string;
  children: ReactNode;
};

export function InlineField({
  label,
  className,
  children,
}: InlineFieldProps): ReactElement {
  return (
    <div className={cn("grid gap-1.5", className)}>
      <p className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
        {label}
      </p>
      {children}
    </div>
  );
}
