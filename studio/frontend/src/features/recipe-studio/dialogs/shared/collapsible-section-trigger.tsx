// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ButtonHTMLAttributes,
  type ReactElement,
  forwardRef,
} from "react";

type CollapsibleSectionTriggerProps = {
  label: string;
  open: boolean;
  summary?: string;
} & ButtonHTMLAttributes<HTMLButtonElement>;

export const CollapsibleSectionTriggerButton = forwardRef<
  HTMLButtonElement,
  CollapsibleSectionTriggerProps
>(function CollapsibleSectionTriggerButton(
  {
    label,
    open,
    summary,
    className,
    type = "button",
    ...props
  }: CollapsibleSectionTriggerProps,
  ref,
): ReactElement {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(
        "flex w-full items-center justify-between gap-3 text-left text-xs text-muted-foreground transition hover:text-foreground",
        className,
      )}
      {...props}
    >
      <span className="flex min-w-0 items-center gap-2">
        <HugeiconsIcon
          icon={ChevronDownStandardIcon}
          className={cn(
            "size-3.5 shrink-0 transition-transform",
            open && "rotate-180",
          )}
        />
        <span className="font-semibold uppercase">{label}</span>
      </span>
      <span className="shrink-0">{summary ?? (open ? "Hide" : "Show")}</span>
    </button>
  );
});
