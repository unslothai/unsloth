// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Every page that ships a tour, and the route prefix it answers to. The user menu reads this to
 * decide whether to offer "Guided Tour"; each page passes the same id to useGuidedTourController.
 * Longest prefix first: the recipe editor sits under the Data Recipes route.
 */
export const TOUR_ROUTES: ReadonlyArray<{ prefix: string; id: string }> = [
  { prefix: "/data-recipes/", id: "recipe-editor" },
  { prefix: "/data-recipes", id: "data-recipes" },
  { prefix: "/api-monitor", id: "api-monitor" },
  { prefix: "/projects", id: "projects" },
  { prefix: "/studio", id: "studio" },
  { prefix: "/export", id: "export" },
  { prefix: "/images", id: "images" },
  { prefix: "/video", id: "video" },
  { prefix: "/audio", id: "audio" },
  { prefix: "/chat", id: "chat" },
  { prefix: "/hub", id: "hub" },
];

/** The tour id for a route, or null where that page has no tour. */
export function getTourId(pathname: string): string | null {
  return (
    TOUR_ROUTES.find((route) => pathname.startsWith(route.prefix))?.id ?? null
  );
}
