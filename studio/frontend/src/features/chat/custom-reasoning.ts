// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Explicit connection contract, never inferred from a URL or a model name. */
export const CUSTOM_REASONING_STYLES = [
  "reasoning_effort",
  "reasoning",
  "thinking",
  "chat_template_kwargs.enable_thinking",
] as const;

export type CustomReasoningStyle = (typeof CUSTOM_REASONING_STYLES)[number];
export type CustomReasoningConfig = {
  enabled: boolean;
  style: CustomReasoningStyle;
};

/** Cached or legacy malformed contracts fail closed. */
export function normalizeCustomReasoningConfig(
  value: unknown,
): CustomReasoningConfig | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value))
    return undefined;
  const config = value as Record<string, unknown>;
  if (
    Object.keys(config).some((key) => key !== "enabled" && key !== "style") ||
    typeof config.enabled !== "boolean" ||
    !CUSTOM_REASONING_STYLES.includes(config.style as CustomReasoningStyle)
  ) {
    return undefined;
  }
  return {
    enabled: config.enabled,
    style: config.style as CustomReasoningStyle,
  };
}

/** Studio's generic controls; the backend alone translates them to the selected wire style. */
export function customReasoningRequestFields(
  value: unknown,
  enabled: boolean,
  effort: string,
): {
  reasoning_effort?: "none" | "low" | "medium" | "high";
  thinking?: { type: "enabled" | "disabled" };
} {
  const config = normalizeCustomReasoningConfig(value);
  if (!config?.enabled) return {};
  if (config.style === "reasoning_effort" || config.style === "reasoning") {
    const level = effort === "low" || effort === "high" ? effort : "medium";
    return { reasoning_effort: !enabled || effort === "none" ? "none" : level };
  }
  return { thinking: { type: enabled ? "enabled" : "disabled" } };
}
