// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute, lazyRouteComponent } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const CharactersPage = lazyRouteComponent(
  () => import("@/features/characters/pages/characters-page"),
  "CharactersPage",
);

export type CharactersSearch = { view?: "personas" };

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/characters",
  staticData: { titleKey: "shell.navigation.characters" },
  validateSearch: (search: Record<string, unknown>): CharactersSearch =>
    search.view === "personas" ? { view: "personas" } : {},
  beforeLoad: () => requireAuth(),
  component: CharactersPage,
});
