// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  BrainIcon,
  CodeIcon,
  GlobeIcon,
  HeadphonesIcon,
  SparklesIcon,
  ViewIcon,
  Wrench01Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { HugeiconsIcon } from "@hugeicons/react";
import type { Capability, CapabilityKey } from "../lib/capabilities";

const CAPABILITY_ICON: Record<CapabilityKey, IconSvgElement> = {
  vision: ViewIcon,
  audio: HeadphonesIcon,
  tools: Wrench01Icon,
  reasoning: BrainIcon,
  code: CodeIcon,
  embedding: SparklesIcon,
  multilingual: GlobeIcon,
};

const CAPABILITY_TONE: Record<CapabilityKey, string> = {
  vision:
    "bg-indigo-500/10 text-indigo-700 dark:bg-indigo-400/10 dark:text-indigo-300",
  audio:
    "bg-rose-500/10 text-rose-700 dark:bg-rose-400/10 dark:text-rose-300",
  tools:
    "bg-amber-500/10 text-amber-800 dark:bg-amber-400/10 dark:text-amber-300",
  reasoning:
    "bg-violet-500/10 text-violet-700 dark:bg-violet-400/10 dark:text-violet-300",
  code: "bg-cyan-500/10 text-cyan-800 dark:bg-cyan-400/10 dark:text-cyan-300",
  embedding:
    "bg-emerald-500/10 text-emerald-700 dark:bg-emerald-400/10 dark:text-emerald-300",
  multilingual:
    "bg-sky-500/10 text-sky-700 dark:bg-sky-400/10 dark:text-sky-300",
};

export function CapabilityPill({ capability }: { capability: Capability }) {
  return (
    <span
      className={`inline-flex h-6 items-center gap-1.5 rounded-full px-2.5 text-[11.5px] font-medium ${CAPABILITY_TONE[capability.key]}`}
    >
      <HugeiconsIcon
        icon={CAPABILITY_ICON[capability.key]}
        strokeWidth={1.75}
        className="size-3"
      />
      {capability.label}
    </span>
  );
}
