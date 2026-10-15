// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute, lazyRouteComponent } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const RoutinesPage = lazyRouteComponent(
  () => import("@/features/routines/routines-page"),
  "RoutinesPage",
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/routines",
  staticData: { title: "Routines" },
  beforeLoad: () => requireAuth(),
  component: RoutinesPage,
});
