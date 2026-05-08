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
    "border-indigo-500/40 text-indigo-600 dark:text-indigo-400",
  audio: "border-rose-500/40 text-rose-600 dark:text-rose-400",
  tools: "border-amber-600/40 text-amber-700 dark:text-amber-400",
  reasoning:
    "border-violet-500/40 text-violet-600 dark:text-violet-400",
  code: "border-cyan-600/40 text-cyan-700 dark:text-cyan-400",
  embedding:
    "border-emerald-600/40 text-emerald-700 dark:text-emerald-400",
  multilingual: "border-sky-500/40 text-sky-600 dark:text-sky-400",
};

export function CapabilityPill({ capability }: { capability: Capability }) {
  return (
    <span
      className={`inline-flex h-5 items-center gap-1 rounded-full border px-2 text-[9.5px] font-semibold uppercase tracking-wider ${CAPABILITY_TONE[capability.key]}`}
    >
      <HugeiconsIcon
        icon={CAPABILITY_ICON[capability.key]}
        strokeWidth={2}
        className="size-2.5"
      />
      {capability.label}
    </span>
  );
}
