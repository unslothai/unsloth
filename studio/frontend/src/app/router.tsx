// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Link, createRouter, useRouterState } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { Route as rootRoute } from "./routes/__root";
import { Route as dataRecipesRoute } from "./routes/data-recipes";
import { Route as dataRecipeRoute } from "./routes/data-recipes.$recipeId";
import { Route as chatRoute } from "./routes/chat";
import { Route as documentScoreRoute } from "./routes/document-score";
import { Route as exportRoute } from "./routes/export";
import { Route as gridTestRoute } from "./routes/grid-test";
import { Route as indexRoute } from "./routes/index";
import { Route as loginRoute } from "./routes/login";
import { Route as onboardingRoute } from "./routes/onboarding";
import { Route as changePasswordRoute } from "./routes/change-password";
import { Route as settingsRoute } from "./routes/settings";
import { Route as studioRoute } from "./routes/studio";

const routeTree = rootRoute.addChildren([
  indexRoute,
  onboardingRoute,
  loginRoute,
  changePasswordRoute,
  gridTestRoute,
  settingsRoute,
  studioRoute,
  chatRoute,
  exportRoute,
  documentScoreRoute,
  dataRecipesRoute,
  dataRecipeRoute,
]);

function DefaultNotFound() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-4 p-8 text-center">
      <img
        src="/Sloth%20emojis/sloth%20shy%20large.png"
        alt="Sloth mascot"
        className="size-24"
      />
      <div className="flex flex-col items-center gap-1">
        <h1 className="font-heading font-semibold text-2xl tracking-tight">
          Page not found
        </h1>
        <p className="text-muted-foreground text-sm break-all">
          {pathname} does not exist.
        </p>
      </div>
      <Button asChild>
        <Link to="/chat">Back to chat</Link>
      </Button>
    </div>
  );
}

export const router = createRouter({
  routeTree,
  defaultNotFoundComponent: DefaultNotFound,
});

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
