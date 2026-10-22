// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute, lazyRouteComponent } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

// Gallery and editor share one lazy entry: two entries sharing modules would let the bundler hoist
// the shared part (icons, dialogs) into the startup chunks.
const FlowsGalleryPage = lazyRouteComponent(
  () => import("@/features/flows"),
  "FlowsGalleryPage",
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/flows",
  staticData: { title: "Flows" },
  beforeLoad: () => requireAuth(),
  component: FlowsGalleryPage,
});
