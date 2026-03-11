// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { createRoute } from "@tanstack/react-router";
import { lazy } from "react";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const StudioPage = lazy(() =>
  import("@/features/studio/studio-page").then((m) => ({
    default: m.StudioPage,
  })),
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/studio",
  beforeLoad: () => requireAuth(),
  component: StudioPage,
});
