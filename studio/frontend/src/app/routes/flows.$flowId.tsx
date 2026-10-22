// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute } from "@tanstack/react-router";
import { type ReactElement, lazy } from "react";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const FlowsEditorPage = lazy(() =>
  import("@/features/flows").then((m) => ({ default: m.FlowsEditorPage })),
);

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/flows/$flowId",
  staticData: { title: "Flows" },
  beforeLoad: () => requireAuth(),
  component: FlowEditorRoute,
});

function FlowEditorRoute(): ReactElement {
  const { flowId } = Route.useParams();
  return <FlowsEditorPage flowId={flowId} />;
}
