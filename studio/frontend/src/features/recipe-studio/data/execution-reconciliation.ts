// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function mergeJsonValue(
  current: unknown,
  incoming: unknown,
  preferIncomingScalars: boolean,
): unknown {
  if (incoming === null || incoming === undefined) {
    return current;
  }
  if (current === null || current === undefined) {
    return incoming;
  }
  if (Array.isArray(current) && Array.isArray(incoming)) {
    return incoming.length > current.length ? incoming : current;
  }
  if (
    typeof current === "object" &&
    !Array.isArray(current) &&
    typeof incoming === "object" &&
    !Array.isArray(incoming)
  ) {
    const merged = { ...(current as Record<string, unknown>) };
    for (const [key, value] of Object.entries(
      incoming as Record<string, unknown>,
    )) {
      merged[key] = mergeJsonValue(merged[key], value, preferIncomingScalars);
    }
    return merged;
  }
  return preferIncomingScalars ? incoming : current;
}

export function mergeOptionalObject<T extends object>(
  current: T | null,
  incoming: T | null,
  preferIncomingScalars: boolean,
): T | null {
  return mergeJsonValue(current, incoming, preferIncomingScalars) as T | null;
}

export function shouldPreferIncomingTerminalScalars(
  incomingEvent: number,
  currentEvent: number,
): boolean {
  return incomingEvent > currentEvent;
}
