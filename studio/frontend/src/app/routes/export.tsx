// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute } from "@tanstack/react-router";
import { lazy } from "react";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const ExportPage = lazy(() =>
  import("@/features/export/export-page").then((m) => ({
    default: m.ExportPage,
  })),
);

export type ExportSearch = {
  // Preselect a training run on the Export page (its output-dir basename, which
  // equals the checkpoint scan's model name). Set when arriving from a run view.
  run?: string;
};

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/export",
  staticData: { title: "Export" },
  beforeLoad: () => requireAuth(),
  validateSearch: (search: Record<string, unknown>): ExportSearch => ({
    run: typeof search.run === "string" ? search.run : undefined,
  }),
  component: ExportPage,
});
