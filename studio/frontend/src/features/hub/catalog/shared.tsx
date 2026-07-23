// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  BrainIcon,
  Chat01Icon,
  CodeIcon,
  GlobeIcon,
  HeadphonesIcon,
  ImageIcon,
  LockIcon,
  LockKeyIcon,
  SparklesIcon,
  ViewIcon,
  Wrench01Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";
import type { Capability, CapabilityKey } from "../lib/model-capabilities";

const CAPABILITY_ICON: Record<CapabilityKey, IconSvgElement> = {
  vision: ViewIcon,
  audio: HeadphonesIcon,
  tools: Wrench01Icon,
  reasoning: BrainIcon,
  code: CodeIcon,
  embedding: SparklesIcon,
  diffusion: ImageIcon,
  multilingual: GlobeIcon,
  conversational: Chat01Icon,
};

const CAPABILITY_TONE: Record<CapabilityKey, string> = {
  vision:
    "bg-indigo-500/10 text-indigo-700 dark:bg-indigo-400/20 dark:text-indigo-300",
  audio: "bg-rose-500/10 text-rose-700 dark:bg-rose-400/20 dark:text-rose-300",
  tools:
    "bg-amber-500/10 text-amber-800 dark:bg-amber-400/20 dark:text-amber-300",
  reasoning:
    "bg-violet-500/10 text-violet-700 dark:bg-violet-400/20 dark:text-violet-300",
  code: "bg-cyan-500/10 text-cyan-800 dark:bg-cyan-400/20 dark:text-cyan-300",
  embedding:
    "bg-emerald-500/10 text-emerald-700 dark:bg-emerald-400/20 dark:text-emerald-300",
  diffusion:
    "bg-pink-500/10 text-pink-700 dark:bg-pink-400/20 dark:text-pink-300",
  multilingual:
    "bg-sky-500/10 text-sky-700 dark:bg-sky-400/20 dark:text-sky-300",
  conversational:
    "bg-fuchsia-500/10 text-fuchsia-700 dark:bg-fuchsia-400/20 dark:text-fuchsia-300",
};

export function AccessChip({ label }: { label: string }) {
  return (
    <span className="inline-flex h-6 shrink-0 items-center rounded-full border border-amber-500/30 bg-amber-500/8 px-2 text-[0.6875rem] font-medium leading-none text-amber-700 dark:text-amber-300">
      {label}
    </span>
  );
}

function isGatedAccess(gated: false | "auto" | "manual" | undefined): boolean {
  return gated !== false && gated !== undefined;
}

function AccessGlyph({
  icon,
  label,
  tooltip,
  className,
}: {
  icon: IconSvgElement;
  label: string;
  tooltip: boolean;
  className: string;
}) {
  const glyph = (
    <span
      role="img"
      aria-label={label}
      className={cn("inline-flex shrink-0", className)}
    >
      <HugeiconsIcon icon={icon} strokeWidth={1.75} className="size-3" />
    </span>
  );
  if (!tooltip) return glyph;
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>{glyph}</TooltipTrigger>
      <TooltipContent side="top" className="tooltip-compact">
        {label}
      </TooltipContent>
    </Tooltip>
  );
}

export function AccessGlyphs({
  gated,
  isPrivate,
  tooltip = true,
}: {
  gated?: false | "auto" | "manual";
  isPrivate?: boolean;
  tooltip?: boolean;
}) {
  const gatedAccess = isGatedAccess(gated);
  if (!gatedAccess && !isPrivate) return null;
  return (
    <>
      {isPrivate && (
        <AccessGlyph
          icon={LockIcon}
          label="Private"
          tooltip={tooltip}
          className="text-muted-foreground/70"
        />
      )}
      {gatedAccess && (
        <AccessGlyph
          icon={LockKeyIcon}
          label="Gated"
          tooltip={tooltip}
          className="text-amber-600 dark:text-amber-400"
        />
      )}
    </>
  );
}

export function CapabilityPill({
  capability,
  iconOnly = false,
  tooltip = true,
}: {
  capability: Capability;
  iconOnly?: boolean;
  tooltip?: boolean;
}) {
  const pill = (
    <span
      aria-label={iconOnly ? capability.label : undefined}
      className={cn(
        "inline-flex h-6 shrink-0 items-center rounded-full text-[0.71875rem] font-medium",
        iconOnly ? "w-6 justify-center px-0" : "gap-1.5 px-2.5",
        CAPABILITY_TONE[capability.key],
      )}
    >
      <HugeiconsIcon
        icon={CAPABILITY_ICON[capability.key]}
        strokeWidth={1.75}
        className="size-3"
      />
      {!iconOnly && capability.label}
    </span>
  );
  if (!iconOnly || !tooltip) return pill;
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>{pill}</TooltipTrigger>
      <TooltipContent side="top" className="tooltip-compact">
        {capability.label}
      </TooltipContent>
    </Tooltip>
  );
}
