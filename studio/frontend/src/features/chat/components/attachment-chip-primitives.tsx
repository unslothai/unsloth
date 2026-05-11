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
  ReactNode,
} from "react";

export const attachmentChipTokens = {
  root: "relative flex min-h-14 max-w-full items-start gap-2 rounded-lg border bg-muted/20 px-2.5 py-2 text-sm backdrop-blur-sm",
  rootInteractive:
    "cursor-pointer text-left transition-all duration-200 hover:bg-accent/40 hover:border-accent-foreground/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring",
  rootReady: "border-border/60",
  rootVisual: "border-primary/20 bg-primary/5",
  rootWarning: "border-amber-500/30 bg-amber-500/5 dark:bg-amber-500/10",
  rootDanger: "border-destructive/30 bg-destructive/5",
  tile: "relative size-14 shrink-0 overflow-hidden rounded-lg border border-border/60 bg-muted/50",
  iconBox:
    "mt-0.5 flex size-9 shrink-0 items-center justify-center rounded-md border bg-background/50 backdrop-blur-sm",
  body: "flex min-w-0 flex-1 flex-col gap-1",
  titleRow: "flex min-w-0 items-center gap-1.5",
  title: "min-w-0 flex-1 truncate text-xs font-medium tracking-tight",
  meta: "flex min-w-0 flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] leading-snug text-muted-foreground/80",
  detail: "line-clamp-2 text-[11px] leading-snug",
  badge:
    "inline-flex h-5 shrink-0 items-center rounded-md border px-1.5 text-[10px] font-medium tracking-wide uppercase",
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

type Tone = "neutral" | "ready" | "visual" | "warning" | "danger";

function toneClass(tone: Tone | undefined): string {
  switch (tone) {
    case "visual":
      return attachmentChipTokens.rootVisual;
    case "warning":
      return attachmentChipTokens.rootWarning;
    case "danger":
      return attachmentChipTokens.rootDanger;
    case "ready":
      return attachmentChipTokens.rootReady;
    default:
      return "border-border/70";
  }
}

export function AttachmentChipRoot({
  className,
  tone = "neutral",
  children,
  ...props
}: HTMLAttributes<HTMLDivElement> & { tone?: Tone }): ReactElement {
  return (
    <div
      className={cn(attachmentChipTokens.root, toneClass(tone), className)}
      {...props}
    >
      {children}
    </div>
  );
}

export function AttachmentChipButton({
  className,
  tone = "neutral",
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { tone?: Tone }): ReactElement {
  return (
    <button
      type="button"
      className={cn(
        attachmentChipTokens.root,
        attachmentChipTokens.rootInteractive,
        toneClass(tone),
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}

export function AttachmentChipIcon({
  className,
  children,
  ...props
}: HTMLAttributes<HTMLSpanElement>): ReactElement {
  return (
    <span className={cn(attachmentChipTokens.iconBox, className)} {...props}>
      {children}
    </span>
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

export function AttachmentChipMeta({
  className,
  children,
  ...props
}: HTMLAttributes<HTMLSpanElement>): ReactElement {
  return (
    <span className={cn(attachmentChipTokens.meta, className)} {...props}>
      {children}
    </span>
  );
}

export function AttachmentChipStatusBadge({
  className,
  tone = "neutral",
  children,
}: {
  className?: string;
  tone?: Tone;
  children: ReactNode;
}): ReactElement {
  return (
    <span
      className={cn(
        attachmentChipTokens.badge,
        tone === "danger" &&
          "border-destructive/30 bg-destructive/10 text-destructive",
        tone === "warning" &&
          "border-amber-400/50 bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-200",
        tone === "ready" &&
          "border-emerald-500/25 bg-emerald-500/10 text-emerald-600 dark:text-emerald-300",
        tone === "visual" && "border-primary/30 bg-primary/10 text-primary",
        tone === "neutral" &&
          "border-border bg-background/70 text-muted-foreground",
        className,
      )}
    >
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
