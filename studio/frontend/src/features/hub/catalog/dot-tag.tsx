// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";

export const DOWNLOADING_DOT_CLASS =
  "bg-sky-500 dark:bg-sky-400 motion-safe:animate-pulse";

type DotTagTone =
  | "success"
  | "downloading"
  | "warning"
  | "danger"
  | "gguf"
  | "mlx"
  | "checkpoint"
  | "adapter";

const TONE_CLASS: Record<DotTagTone, string> = {
  success: "bg-status-success",
  downloading: DOWNLOADING_DOT_CLASS,
  warning: "bg-status-warning",
  danger: "bg-status-danger",
  gguf: "bg-format-gguf",
  mlx: "bg-format-mlx",
  checkpoint: "bg-format-checkpoint",
  adapter: "bg-format-adapter",
};

export function DotTag({
  tone,
  label,
  className,
  dotClassName,
  labelClassName,
}: {
  tone: DotTagTone;
  label: string;
  className?: string;
  dotClassName?: string;
  /** Wraps the label, e.g. to sr-only it and keep just the dot. */
  labelClassName?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex h-5 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-full border border-border/60 bg-transparent px-2 text-ui-11 font-medium leading-none text-muted-foreground",
        className,
      )}
    >
      <span
        aria-hidden="true"
        className={cn(
          "inline-block size-1.5 shrink-0 rounded-full",
          TONE_CLASS[tone],
          dotClassName,
        )}
      />
      {labelClassName ? <span className={labelClassName}>{label}</span> : label}
    </span>
  );
}
