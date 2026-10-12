// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute, lazyRouteComponent } from "@tanstack/react-router";
import { validateLibrarySearch } from "@/features/library";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const LibraryPage = lazyRouteComponent(
  () => import("@/features/library/library-page"),
  "LibraryPage",
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/library",
  staticData: { title: "Library" },
  validateSearch: validateLibrarySearch,
  beforeLoad: () => requireAuth(),
  component: LibraryPage,
});
