// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute } from "@tanstack/react-router";
import { lazy } from "react";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const EvalPage = lazy(() =>
  import("@/features/eval").then((m) => ({ default: m.EvalPage })),
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/eval",
  staticData: { title: "Eval" },
  beforeLoad: () => requireAuth(),
  component: EvalPage,
});
