// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute } from "@tanstack/react-router";
import { lazy } from "react";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const ProjectsPage = lazy(() =>
  import("@/features/chat/projects-page").then((m) => ({
    default: m.ProjectsPage,
  })),
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/projects",
  staticData: { title: "Projects" },
  beforeLoad: () => requireAuth(),
  component: ProjectsPage,
});
