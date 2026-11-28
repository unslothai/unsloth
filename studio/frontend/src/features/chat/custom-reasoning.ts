// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// ".ts" extension: plain-node tests load this module.
import { REASONING_EFFORT_SCALE, type ReasoningEffortLevel } from "./model-catalog.ts";

/** Explicit connection contract, never inferred from a URL or a model name. */
export const CUSTOM_REASONING_STYLES = [
  "reasoning_effort",
  "reasoning",
  "thinking",
  "chat_template_kwargs.enable_thinking",
] as const;

export type CustomReasoningStyle = (typeof CUSTOM_REASONING_STYLES)[number];
export type CustomReasoningEffortLevel = Exclude<ReasoningEffortLevel, "none">;

/** Read on call: model-catalog sits in the chat barrel's import cycle. */
export function customReasoningEffortScale(): readonly CustomReasoningEffortLevel[] {
  return REASONING_EFFORT_SCALE.filter(
    (level): level is CustomReasoningEffortLevel => level !== "none",
  );
}
export const DEFAULT_CUSTOM_REASONING_EFFORT_LEVELS: readonly CustomReasoningEffortLevel[] =
  ["low", "medium", "high"];
export type CustomReasoningConfig = {
  enabled: boolean;
  style: CustomReasoningStyle;
  effort_levels?: CustomReasoningEffortLevel[];
};

export function customReasoningStyleTakesEffort(
  style: CustomReasoningStyle,
): boolean {
  return style === "reasoning_effort" || style === "reasoning";
}

export function customReasoningEffortLevels(
  config: CustomReasoningConfig | undefined,
): readonly CustomReasoningEffortLevel[] {
  return config?.effort_levels ?? DEFAULT_CUSTOM_REASONING_EFFORT_LEVELS;
}

export function customReasoningConfigForSave(
  config: CustomReasoningConfig | null,
): CustomReasoningConfig | null {
  if (!config) return null;
  const { effort_levels: levels, ...rest } = config;
  if (
    !levels ||
    levels.length === 0 ||
    !customReasoningStyleTakesEffort(config.style) ||
    (levels.length === DEFAULT_CUSTOM_REASONING_EFFORT_LEVELS.length &&
      DEFAULT_CUSTOM_REASONING_EFFORT_LEVELS.every((level) => levels.includes(level)))
  ) {
    return rest;
  }
  return { ...rest, effort_levels: canonicalEffortLevels(levels) };
}

function canonicalEffortLevels(
  levels: readonly unknown[],
): CustomReasoningEffortLevel[] {
  return customReasoningEffortScale().filter((level) => levels.includes(level));
}

/** Rungs every configured model reports (each must report); with none configured, every reporting model. */
export function detectedCustomEffortLevels(
  capabilities: readonly {
    id: string;
    reasoning?: { supported_efforts?: readonly string[] | null } | null;
  }[],
  selectedModelIds: readonly string[],
): CustomReasoningEffortLevel[] | undefined {
  const reported = capabilities.filter(
    (entry) => (entry.reasoning?.supported_efforts?.length ?? 0) > 0,
  );
  // The ladder applies to every configured model, so one that did not report leaves it unchanged.
  const pool =
    selectedModelIds.length > 0
      ? selectedModelIds.map((id) => reported.find((entry) => entry.id === id))
      : reported;
  if (pool.some((entry) => entry === undefined)) return undefined;
  if (pool.length === 0) return undefined;
  const shared = customReasoningEffortScale().filter((level) =>
    pool.every((entry) => entry?.reasoning?.supported_efforts?.includes(level)),
  );
  return shared.length > 0 ? shared : undefined;
}

/** A background result never replaces levels the user set while it was in flight. */
export function withDetectedEffortLevels(
  current: CustomReasoningConfig | null,
  levels: CustomReasoningEffortLevel[],
  silent: boolean,
): CustomReasoningConfig | null {
  if (!current || !customReasoningStyleTakesEffort(current.style)) return current;
  if (silent && current.effort_levels !== undefined) return current;
  return { ...current, effort_levels: levels };
}

/** Cached or legacy malformed contracts fail closed. */
export function normalizeCustomReasoningConfig(
  value: unknown,
): CustomReasoningConfig | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value))
    return undefined;
  const config = value as Record<string, unknown>;
  if (
    Object.keys(config).some(
      (key) => key !== "enabled" && key !== "style" && key !== "effort_levels",
    ) ||
    typeof config.enabled !== "boolean" ||
    !CUSTOM_REASONING_STYLES.includes(config.style as CustomReasoningStyle)
  ) {
    return undefined;
  }
  const style = config.style as CustomReasoningStyle;
  const rawLevels = config.effort_levels;
  if (rawLevels == null) return { enabled: config.enabled, style };
  if (
    !Array.isArray(rawLevels) ||
    rawLevels.length === 0 ||
    !customReasoningStyleTakesEffort(style) ||
    rawLevels.some(
      (level) =>
        !customReasoningEffortScale().includes(level as CustomReasoningEffortLevel),
    )
  ) {
    return undefined;
  }
  return {
    enabled: config.enabled,
    style,
    effort_levels: canonicalEffortLevels(rawLevels),
  };
}

/** Studio's generic controls; the backend alone translates them to the selected wire style. */
export function customReasoningRequestFields(
  value: unknown,
  enabled: boolean,
  effort: string,
): {
  reasoning_effort?: "none" | CustomReasoningEffortLevel;
  thinking?: { type: "enabled" | "disabled" };
} {
  const config = normalizeCustomReasoningConfig(value);
  if (!config?.enabled) return {};
  if (customReasoningStyleTakesEffort(config.style)) {
    if (!enabled || effort === "none") return { reasoning_effort: "none" };
    const levels = customReasoningEffortLevels(config);
    const level = levels.includes(effort as CustomReasoningEffortLevel)
      ? (effort as CustomReasoningEffortLevel)
      : levels.includes("medium")
        ? "medium"
        : undefined;
    return level ? { reasoning_effort: level } : {};
  }
  return { thinking: { type: enabled ? "enabled" : "disabled" } };
}
