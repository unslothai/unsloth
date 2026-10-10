// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute } from "@tanstack/react-router";
import { lazy } from "react";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const NotebooksPage = lazy(() =>
  import("@/features/notebook/notebooks-page").then((m) => ({
    default: m.NotebooksPage,
  })),
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/notebooks",
  staticData: { titleKey: "notebooks.routeTitle" },
  beforeLoad: () => requireAuth(),
  component: NotebooksPage,
});
