// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute, lazyRouteComponent } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const ConvertPage = lazyRouteComponent(
  () => import("@/features/convert/convert-page"),
  "ConvertPage",
);

export type ConvertSearch = {
  // Preselect a training run on the Convert page (its output-dir basename, which
  // equals the checkpoint scan's model name). Set when arriving from a run view.
  run?: string;
};

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/convert",
  staticData: { title: "Convert" },
  beforeLoad: () => requireAuth(),
  validateSearch: (search: Record<string, unknown>): ConvertSearch => ({
    run: typeof search.run === "string" ? search.run : undefined,
  }),
  component: ConvertPage,
});
