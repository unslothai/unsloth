// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoute } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

// RootLayout keeps the player mounted (like Images/Video) so a game keeps its state off-tab; this route owns the URL + auth gate.
export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/flows/$flowId/play",
  staticData: { title: "Flows" },
  beforeLoad: () => requireAuth(),
  component: () => null,
});
