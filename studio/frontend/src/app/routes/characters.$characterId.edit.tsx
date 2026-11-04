// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute } from "@tanstack/react-router";
import type { ReactElement } from "react";
import { lazy } from "react";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const CharacterEditorPage = lazy(() =>
  import("@/features/characters/pages/character-editor-page").then((m) => ({
    default: m.CharacterEditorPage,
  })),
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  // `new` creates a character; any other id edits that one.
  path: "/characters/$characterId/edit",
  staticData: { titleKey: "shell.navigation.characters" },
  beforeLoad: () => requireAuth(),
  component: CharacterEditorRoute,
});

function CharacterEditorRoute(): ReactElement {
  const { characterId } = Route.useParams();
  return <CharacterEditorPage key={characterId} characterId={characterId === "new" ? null : characterId} />;
}
