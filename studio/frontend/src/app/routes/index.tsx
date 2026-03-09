// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { createRoute, redirect } from "@tanstack/react-router";
import { getPostAuthRoute } from "@/features/auth";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  beforeLoad: async () => {
    await requireAuth();
    throw redirect({ to: getPostAuthRoute() });
  },
  component: () => null,
});
