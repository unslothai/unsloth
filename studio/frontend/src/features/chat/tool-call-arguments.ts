// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The `function.arguments` string to replay for a stored tool call.
 *
 * `argsText` is the text the provider streamed and is preferred so replay is
 * byte-exact. Text that does not parse would be replayed on every later request
 * in the thread, and strict chat templates reject the whole request rather than
 * one call, so it falls back to the structured args the part already carries.
 */
export function toolCallReplayArguments(
  argsText: string | undefined,
  args: unknown,
): string {
  if (typeof argsText === "string" && argsText.length > 0) {
    try {
      JSON.parse(argsText);
      return argsText;
    } catch {
      // unparsable, so the structured args below stand in for it
    }
  }
  const structured =
    args !== null && typeof args === "object" && !Array.isArray(args)
      ? (args as Record<string, unknown>)
      : null;
  if (
    typeof argsText === "string" &&
    structured !== null &&
    Object.keys(structured).length === 1 &&
    structured._raw === argsText
  ) {
    return "{}";
  }
  return JSON.stringify(args ?? {}) ?? "{}";
}

export function streamedToolCallArguments(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (value === null || typeof value !== "object") {
    return "";
  }
  try {
    return JSON.stringify(value) ?? "";
  } catch {
    return "";
  }
}
