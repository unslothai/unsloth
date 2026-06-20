// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type * as React from "react";

import { cn } from "@/lib/utils";

type TextareaProps = React.ComponentProps<"textarea"> & {
  fieldSizing?: "content" | "fixed";
};

function Textarea({
  className,
  fieldSizing = "content",
  ...props
}: TextareaProps) {
  return (
    <textarea
      data-slot="textarea"
      className={cn(
        "border-input bg-input/30 focus-visible:border-ring focus-visible:ring-ring/50 aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive dark:aria-invalid:border-destructive/50 resize-none rounded-xl border px-3 py-3 text-base transition-colors focus-visible:ring-[3px] aria-invalid:ring-[3px] md:text-sm placeholder:text-muted-foreground flex min-h-16 min-w-0 max-w-full w-full whitespace-pre-wrap break-words [overflow-wrap:anywhere] outline-none disabled:cursor-not-allowed disabled:opacity-50",
        fieldSizing === "content" ? "field-sizing-content" : "[field-sizing:fixed]",
        className,
      )}
      {...props}
    />
  );
}

export { Textarea };
