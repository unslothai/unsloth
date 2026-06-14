// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { cn } from "@/lib/utils";
import { XIcon } from "lucide-react";
import type {
  ButtonHTMLAttributes,
  HTMLAttributes,
  ReactElement,
} from "react";

export const attachmentChipTokens = {
  root: "relative flex min-h-14 max-w-full items-start gap-2 rounded-lg border bg-muted/20 px-2.5 py-2 text-sm backdrop-blur-sm",
  rootInteractive:
    "cursor-pointer text-left transition-all duration-200 hover:bg-accent/40 hover:border-accent-foreground/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring",
  tile: "relative size-14 shrink-0 overflow-hidden rounded-lg border border-border/60 bg-muted/50",
  body: "flex min-w-0 flex-1 flex-col gap-1",
  title: "min-w-0 flex-1 truncate text-xs font-medium tracking-tight",
  remove:
    "flex size-8 shrink-0 items-center justify-center rounded-md text-muted-foreground/60 hover:bg-destructive/10 hover:text-destructive focus:outline-none focus-visible:ring-2 focus-visible:ring-ring",
  removeFloating:
    "absolute top-1.5 right-1.5 size-5 rounded-full bg-foreground/5 text-foreground/50 transition-all hover:bg-destructive hover:text-destructive-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring",
  progressTrack: "mt-0.5 h-1 overflow-hidden rounded-full bg-foreground/5",
  progressFill:
    "block h-full rounded-full bg-primary/60 transition-all motion-reduce:transition-none",
  progressIndeterminate:
    "block h-full w-1/3 rounded-full bg-primary/40 animate-pulse motion-reduce:animate-none",
} as const;

export function AttachmentChipRoot({
  className,
  children,
  ...props
}: HTMLAttributes<HTMLDivElement>): ReactElement {
  return (
    <div
      className={cn(attachmentChipTokens.root, "border-border/70", className)}
      {...props}
    >
      {children}
    </div>
  );
}

export function AttachmentChipButton({
  className,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement>): ReactElement {
  return (
    <button
      type="button"
      className={cn(
        attachmentChipTokens.root,
        attachmentChipTokens.rootInteractive,
        "border-border/70",
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}

export function AttachmentChipBody({
  className,
  children,
  ...props
}: HTMLAttributes<HTMLSpanElement>): ReactElement {
  return (
    <span className={cn(attachmentChipTokens.body, className)} {...props}>
      {children}
    </span>
  );
}

export function AttachmentChipTitle({
  className,
  children,
  ...props
}: HTMLAttributes<HTMLSpanElement>): ReactElement {
  return (
    <span className={cn(attachmentChipTokens.title, className)} {...props}>
      {children}
    </span>
  );
}

export function AttachmentChipProgress({
  value,
  label,
  className,
}: {
  value: number | null;
  label: string;
  className?: string;
}): ReactElement {
  if (value === null) {
    return (
      <div
        aria-busy="true"
        aria-live="polite"
        aria-label={label}
        className={cn(attachmentChipTokens.progressTrack, className)}
      >
        <span
          aria-hidden="true"
          className={attachmentChipTokens.progressIndeterminate}
        />
      </div>
    );
  }

  const pct = Math.max(0, Math.min(100, value));
  return (
    <div
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(pct)}
      aria-valuetext={label}
      className={cn(attachmentChipTokens.progressTrack, className)}
    >
      <span
        aria-hidden="true"
        className={attachmentChipTokens.progressFill}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export function AttachmentChipRemoveButton({
  className,
  tooltip = "Remove file",
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  tooltip?: string;
}): ReactElement {
  return (
    <TooltipIconButton
      tooltip={tooltip}
      className={cn(attachmentChipTokens.removeFloating, className)}
      side="top"
      {...props}
    >
      {children ?? (
        <XIcon className="size-3 dark:stroke-[2.5px]" aria-hidden="true" />
      )}
    </TooltipIconButton>
  );
}
