// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReasoningEffortLevel } from "../model-catalog.ts";

export interface ThinkingCapabilities {
  supportsReasoning: boolean;
  reasoningStyle: string;
  reasoningAlwaysOn: boolean;
  supportsReasoningOff: boolean;
  reasoningEffortLevels: readonly ReasoningEffortLevel[];
  /** False means the provider has not published capabilities, not that thinking is fixed. */
  reasoningKnown?: boolean;
}

export function effortLabel(level: string): string {
  if (level === "none") return "Off";
  if (level === "xhigh") return "Extra High";
  return level.charAt(0).toUpperCase() + level.slice(1);
}

export function thinkingPresentation(caps: ThinkingCapabilities) {
  const takesEffort =
    caps.reasoningStyle === "reasoning_effort" ||
    caps.reasoningStyle === "enable_thinking_effort";
  const levels: ReasoningEffortLevel[] =
    caps.supportsReasoning && takesEffort
      ? [
          ...new Set(
            caps.reasoningEffortLevels.filter((level) => level !== "none"),
          ),
        ]
      : [];
  const canDisable =
    caps.supportsReasoning &&
    !caps.reasoningAlwaysOn &&
    (caps.supportsReasoningOff ||
      (takesEffort && caps.reasoningEffortLevels.includes("none")));
  const kind =
    caps.reasoningKnown === false
      ? "unknown"
      : !caps.supportsReasoning
        ? "unsupported"
        : levels.length > 1
          ? "adjustable"
          : levels.length === 1
            ? "fixed"
            : canDisable
              ? "toggle"
              : "always-on";
  const description =
    kind === "fixed"
      ? `This model supports ${effortLabel(levels[0])} only.`
      : kind === "toggle"
        ? "Effort controlled by model."
        : kind === "always-on"
          ? "This model manages its thinking effort."
          : kind === "unknown"
            ? "Available effort levels aren’t published."
            : kind === "adjustable"
              ? "Choose how much effort the model spends thinking."
              : "This model does not support thinking.";
  return { kind, levels, canDisable, description } as const;
}

export function stepThinkingEffort(
  levels: readonly ReasoningEffortLevel[],
  current: ReasoningEffortLevel,
  delta: number,
  wrap: boolean,
) {
  if (levels.length < 2) return null;
  const index = levels.indexOf(current);
  if (index < 0) return levels[0];
  const next = wrap
    ? (index + delta + levels.length) % levels.length
    : Math.max(0, Math.min(levels.length - 1, index + delta));
  return levels[next];
}
