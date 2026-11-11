// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute } from "@tanstack/react-router";
import type { ReactElement } from "react";
import { lazy } from "react";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const CharacterProfilePage = lazy(() =>
  import("@/features/characters/pages/character-profile-page").then((m) => ({
    default: m.CharacterProfilePage,
  })),
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/characters/$characterId",
  staticData: { titleKey: "shell.navigation.characters" },
  beforeLoad: () => requireAuth(),
  component: CharacterProfileRoute,
});

function CharacterProfileRoute(): ReactElement {
  const { characterId } = Route.useParams();
  return <CharacterProfilePage characterId={characterId} />;
}
