// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ComponentProps, ReactElement } from "react";
import { Handle, type HandleProps } from "@xyflow/react";

import { cn } from "@/lib/utils";

export type BaseHandleProps = HandleProps;

export function BaseHandle({
  className,
  children,
  ...props
}: ComponentProps<typeof Handle>): ReactElement {
  return (
    <Handle
      {...props}
      className={cn(
        "h-[12px] w-[12px] rounded-full border border-border/80 bg-muted shadow-[0_0_0_1px_hsl(var(--background))] transition-all hover:scale-110 hover:border-primary/70 hover:bg-primary/20",
        className,
      )}
    >
      {children}
    </Handle>
  );
}
