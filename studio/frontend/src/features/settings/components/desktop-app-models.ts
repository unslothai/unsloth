// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The slot → model logic of the Desktop apps cards, after Ollama's Claude Desktop model settings
// (sharedMappingModel / updateAllMappings in ClaudeDesktopModelsSettings.tsx).

/** A slot on this follows whatever Studio has loaded (utils/claude_desktop.FOLLOW_LOADED). */
export const FOLLOW_LOADED = ":loaded";

export type ClaudeRoute = { id: string; name: string };

/** The same five slots, in Claude's order, for a backend that didn't send them. */
export const CLAUDE_ROUTES: ClaudeRoute[] = [
  { id: "claude-fable-5", name: "Fable 5" },
  { id: "claude-opus-5", name: "Opus 5" },
  { id: "claude-sonnet-5", name: "Sonnet 5" },
  { id: "claude-haiku-4-5-20251001", name: "Haiku 4.5" },
  { id: "claude-sonnet-4-6", name: "Sonnet 4.6" },
];

export type Mappings = Record<string, string>;

/** Only mapped slots, in route order: an empty slot is left out, as the backend stores it. */
export function cleanMappings(
  mappings: Mappings,
  routes: ClaudeRoute[] = CLAUDE_ROUTES,
): Mappings {
  const out: Mappings = {};
  for (const route of routes) {
    const value = mappings[route.id]?.trim();
    if (value) {
      out[route.id] = value;
    }
  }
  return out;
}

/** The model every slot shares, or null when they differ or any is empty ("Mixed"). */
export function sharedMappingModel(
  mappings: Mappings,
  routes: ClaudeRoute[] = CLAUDE_ROUTES,
): string | null {
  const values = routes.map((route) => mappings[route.id] ?? "");
  const first = values[0];
  if (!first || values.some((value) => value !== first)) {
    return null;
  }
  return first;
}

/** The "All" picker: one model on every slot. */
export function updateAllMappings(
  model: string,
  routes: ClaudeRoute[] = CLAUDE_ROUTES,
): Mappings {
  return Object.fromEntries(routes.map((route) => [route.id, model]));
}

export function updateMapping(
  mappings: Mappings,
  routeId: string,
  model: string,
): Mappings {
  return { ...mappings, [routeId]: model };
}

export function mappingsEqual(
  a: Mappings,
  b: Mappings,
  routes: ClaudeRoute[] = CLAUDE_ROUTES,
): boolean {
  const left = cleanMappings(a, routes);
  const right = cleanMappings(b, routes);
  return routes.every((route) => left[route.id] === right[route.id]);
}

/** Distinct Studio models the slots pin (following slots don't count): more than one means
 *  Studio switches models between slots unless it keeps several loaded. */
export function pinnedModels(mappings: Mappings): string[] {
  return [
    ...new Set(
      Object.values(mappings).filter(
        (value) => value && value !== FOLLOW_LOADED,
      ),
    ),
  ];
}
