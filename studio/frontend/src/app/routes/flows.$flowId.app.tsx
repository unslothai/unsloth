// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute } from "@tanstack/react-router";
import { type ReactElement, lazy } from "react";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const FlowAppPage = lazy(() =>
  import("@/features/flows/app/flow-app-page").then((m) => ({
    default: m.FlowAppPage,
  })),
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/flows/$flowId/app",
  staticData: { title: "Flows" },
  beforeLoad: () => requireAuth(),
  component: FlowAppRoute,
});

function FlowAppRoute(): ReactElement {
  const { flowId } = Route.useParams();
  return <FlowAppPage flowId={flowId} />;
}
