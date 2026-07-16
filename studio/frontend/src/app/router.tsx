// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Link, createRouter, useRouterState } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { MascotImg } from "@/components/mascot-img";
import { useT } from "@/i18n";
import { Route as rootRoute } from "./routes/__root";
import { Route as dataRecipesRoute } from "./routes/data-recipes";
import { Route as dataRecipeRoute } from "./routes/data-recipes.$recipeId";
import { Route as chatRoute } from "./routes/chat";
import { Route as exportRoute } from "./routes/export";
import { Route as indexRoute } from "./routes/index";
import { Route as loginRoute } from "./routes/login";
import { Route as hubRoute } from "./routes/hub";
import { Route as onboardingRoute } from "./routes/onboarding";
import { Route as projectsRoute } from "./routes/projects";
import { Route as changePasswordRoute } from "./routes/change-password";
import { Route as settingsRoute } from "./routes/settings";
import { Route as studioRoute } from "./routes/studio";

const routeTree = rootRoute.addChildren([
  indexRoute,
  onboardingRoute,
  loginRoute,
  changePasswordRoute,
  hubRoute,
  settingsRoute,
  studioRoute,
  chatRoute,
  projectsRoute,
  exportRoute,
  dataRecipesRoute,
  dataRecipeRoute,
]);

function DefaultNotFound() {
  const t = useT();
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-4 p-8 text-center">
      <MascotImg src="Sloth emojis/sloth shy large.png" className="size-24" />
      <div className="flex flex-col items-center gap-1">
        <h1 className="font-heading font-semibold text-2xl tracking-tight">
          {t("shell.notFound.title")}
        </h1>
        <p className="text-muted-foreground text-sm break-all">
          {t("shell.notFound.description", { path: pathname })}
        </p>
      </div>
      <Button asChild>
        <Link to="/chat">{t("shell.notFound.backToChat")}</Link>
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
