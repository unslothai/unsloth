// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute, lazyRouteComponent } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const TasksPage = lazyRouteComponent(
  () => import("@/features/scheduled-tasks/tasks-page"),
  "TasksPage",
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/tasks",
  staticData: { titleKey: "shell.navigation.tasks" },
  // ?edit=<task id> opens that task's dialog, e.g. from "Sent by scheduled task" in its chat;
  // ?create=true opens a new one, from the sidebar's "+".
  validateSearch: (search: Record<string, unknown>): { edit?: string; create?: boolean } => ({
    ...(typeof search.edit === "string" && search.edit !== "" ? { edit: search.edit } : {}),
    ...(search.create === true || search.create === "true" ? { create: true } : {}),
  }),
  beforeLoad: () => requireAuth(),
  component: TasksPage,
});
