// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";

type DotTagTone = "success" | "warning" | "danger" | "gguf" | "checkpoint";

const TONE_CLASS: Record<DotTagTone, string> = {
  success: "bg-status-success",
  warning: "bg-status-warning",
  danger: "bg-status-danger",
  gguf: "bg-format-gguf",
  checkpoint: "bg-format-checkpoint",
};

export function DotTag({
  tone,
  label,
  className,
}: {
  tone: DotTagTone;
  label: string;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex h-5 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-[7px] border border-border/60 bg-transparent px-1.5 text-[11px] font-medium leading-none text-muted-foreground",
        className,
      )}
    >
      <span
        aria-hidden="true"
        className={cn("inline-block size-1.5 shrink-0 rounded-full", TONE_CLASS[tone])}
      />
      {label}
    </span>
  );
}
