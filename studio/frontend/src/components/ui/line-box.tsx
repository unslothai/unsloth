// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type * as React from "react";

import { useSingleLine } from "@/hooks/use-single-line";
import { cn } from "@/lib/utils";

/** A command, URL or code box: a pill while it fits on one line, 16px corners once it wraps. */
export function LineBox({
  as: Tag = "div",
  className,
  ...props
}: React.HTMLAttributes<HTMLElement> & { as?: "code" | "div" | "p" }) {
  const [ref, single] = useSingleLine<HTMLElement>();
  return (
    <Tag
      ref={ref}
      className={cn(className, single ? "rounded-full" : "rounded-lg")}
      {...props}
    />
  );
}
